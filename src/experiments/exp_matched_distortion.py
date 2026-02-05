"""
Matched-Distortion Learnability Test

核心思想：
把 paper IR-drop 和一个"温和但仍输入相关"的 IR-drop 替代模型（crossbar模式），
调到同等平均相对输出扰动 δ，然后看训练是否仍然崩溃。

如果同等 δ 下只有 paper IR-drop 崩，那就不是"太激进"，是结构不可学。
"""

import argparse
import yaml
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from typing import Dict, Any, List, Tuple, Optional
import logging
from tqdm import tqdm

try:
    from .run_experiment import run_experiment
    from ..models.model_zoo import get_model, wrap_model_with_memristor
    from ..memristor.device_model import MemristorDeviceModel
    from ..data.dataset import get_dataloaders
    from ..utils.checkpoint import load_checkpoint
    from ..utils.metrics import AverageMeter, accuracy
    from ..utils.seeds import set_seed
    from ..utils.logger import setup_logger
except ImportError:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.experiments.run_experiment import run_experiment
    from src.models.model_zoo import get_model, wrap_model_with_memristor
    from src.memristor.device_model import MemristorDeviceModel
    from src.data.dataset import get_dataloaders
    from src.utils.checkpoint import load_checkpoint
    from src.utils.metrics import AverageMeter, accuracy
    from src.utils.seeds import set_seed
    from src.utils.logger import setup_logger

logger = logging.getLogger(__name__)


def load_checkpoint_smart(
    checkpoint_path: str,
    base_model: nn.Module,
    device: torch.device,
    device_model: Optional[MemristorDeviceModel] = None,
) -> Dict[str, Any]:
    """
    智能加载checkpoint，根据checkpoint的key格式自动处理。
    
    如果checkpoint保存的是memristor包装后的模型（key有base_model.前缀），
    需要先加载到包装模型，然后提取base_model的权重。
    如果checkpoint保存的是原始模型（key没有base_model.前缀），
    可以直接加载到base_model。
    
    Args:
        checkpoint_path: checkpoint文件路径
        base_model: 基础模型
        device: 设备
        device_model: 设备模型（如果需要加载memristor包装后的checkpoint）
    
    Returns:
        checkpoint字典
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 获取state_dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # 检查key格式
    first_key = next(iter(state_dict.keys())) if state_dict else None
    is_memristor_checkpoint = first_key and first_key.startswith('base_model.')
    
    if is_memristor_checkpoint:
        # checkpoint保存的是memristor包装后的模型
        # 需要先创建包装模型，加载checkpoint，然后提取base_model的权重
        if device_model is None:
            # 如果没有提供device_model，尝试直接提取base_model的权重
            # 移除base_model.前缀
            base_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('base_model.'):
                    new_key = key[len('base_model.'):]
                    base_state_dict[new_key] = value
                elif not key.startswith('device_model'):  # 跳过device_model相关的key
                    base_state_dict[key] = value
            
            if base_state_dict:
                base_model.load_state_dict(base_state_dict, strict=False)
                logger.info(f"Extracted base_model weights from memristor checkpoint (no device_model provided)")
            else:
                raise ValueError("device_model is required when loading memristor-wrapped checkpoint")
        else:
            # 创建包装模型
            wrapped_model = wrap_model_with_memristor(base_model, device_model)
            wrapped_model = wrapped_model.to(device)
            
            # 加载checkpoint到包装模型
            wrapped_model.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded memristor-wrapped checkpoint from {checkpoint_path}")
            
            # 提取base_model的权重
            if hasattr(wrapped_model, 'base_model'):
                base_model.load_state_dict(wrapped_model.base_model.state_dict(), strict=False)
                logger.info("Extracted base_model weights from wrapped model")
            else:
                logger.warning("Wrapped model does not have base_model attribute, using original base_model")
    else:
        # checkpoint保存的是原始模型，直接加载
        base_model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded base model checkpoint from {checkpoint_path}")
    
    return checkpoint


def get_num_classes_from_checkpoint(checkpoint_path: str, default_num_classes: int = 10, device: Optional[torch.device] = None) -> int:
    """
    从checkpoint中获取num_classes。
    
    Args:
        checkpoint_path: checkpoint文件路径
        default_num_classes: 默认的num_classes值
        device: 设备（用于加载checkpoint）
        
    Returns:
        num_classes值
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 首先尝试从config中获取
    if 'config' in checkpoint and 'num_classes' in checkpoint['config']:
        num_classes = checkpoint['config']['num_classes']
        logger.info(f"Using num_classes={num_classes} from checkpoint config")
        return num_classes
    
    # 如果config中没有，从state_dict中推断
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        # 查找linear层的权重（通常是最后一层）
        for key in reversed(list(state_dict.keys())):
            if 'linear.weight' in key or 'fc.weight' in key:
                num_classes = state_dict[key].shape[0]
                logger.info(f"Inferred num_classes={num_classes} from checkpoint state_dict")
                return num_classes
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        for key in reversed(list(state_dict.keys())):
            if 'linear.weight' in key or 'fc.weight' in key:
                num_classes = state_dict[key].shape[0]
                logger.info(f"Inferred num_classes={num_classes} from checkpoint state_dict")
                return num_classes
    
    # 如果都找不到，返回默认值
    logger.warning(f"Could not infer num_classes from checkpoint, using default={default_num_classes}")
    return default_num_classes


def compute_delta(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    device_model: MemristorDeviceModel,
    epsilon: float = 1e-8,
) -> Dict[str, Any]:
    """
    计算相对输出扰动 δ。
    
    计算全局δ：
    δ_global = E_batch[|y_tilde - y|_2 / (|y|_2 + ε)]
    
    Args:
        model: 模型
        data_loader: 数据加载器
        device: 设备
        device_model: 忆阻器设备模型
        epsilon: 数值稳定性参数
    
    Returns:
        包含 δ_global 和相关统计的字典
    """
    model.eval()
    
    # 第一次：无IR-drop（baseline）
    original_ir_drop_mode = device_model.ir_drop_mode
    # 保存当前的强度参数（校准过程中可能已经修改）
    current_ir_drop_scaling = device_model.ir_drop_scaling
    current_ir_drop_cap = device_model.ir_drop_cap
    
    # 临时禁用IR-drop
    device_model.ir_drop_mode = 'none'
    
    outputs_clean = []
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            # 固定seed以确保可重复
            torch.manual_seed(42)
            np.random.seed(42)
            if hasattr(model, 'forward') and 't' in model.forward.__code__.co_varnames:
                output = model(data, t=0, seed=42)
            else:
                output = model(data)
            outputs_clean.append(output.cpu())
    
    # 第二次：有IR-drop（使用当前设置的强度参数，不要恢复原始值）
    device_model.ir_drop_mode = original_ir_drop_mode
    # 保持当前设置的强度参数（校准过程中设置的mid值）
    device_model.ir_drop_scaling = current_ir_drop_scaling
    device_model.ir_drop_cap = current_ir_drop_cap
    
    outputs_ir = []
    num_skipped = 0  # 记录跳过的次数（NaN/Inf）
    total_batches = 0
    
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            total_batches += 1
            # 使用相同的seed（但IR-drop会引入额外的扰动）
            torch.manual_seed(42)
            np.random.seed(42)
            try:
                if hasattr(model, 'forward') and 't' in model.forward.__code__.co_varnames:
                    output = model(data, t=0, seed=42)
                else:
                    output = model(data)
                
                # 检查是否有NaN/Inf
                # 注意：对于校准实验，我们不应该fallback到clean输出
                # 因为NaN/Inf本身就是paper IR-drop的特征，应该保留以观察其影响
                if torch.isnan(output).any() or torch.isinf(output).any():
                    num_skipped += 1
                    # 对于校准，我们仍然记录NaN/Inf输出，但标记为skipped
                    # 这样可以看到scaling变化时是否产生NaN/Inf
                    logger.debug(f"NaN/Inf detected in output at batch {len(outputs_ir)}, scaling={device_model.ir_drop_scaling if device_model.ir_drop_mode == 'paper' else 'N/A'}")
                    # 使用clean输出作为fallback（仅用于计算δ，但会记录applied_ratio）
                    output = outputs_clean[len(outputs_ir)]
                
                outputs_ir.append(output.cpu())
            except Exception as e:
                num_skipped += 1
                logger.debug(f"Exception during forward pass: {e}")
                # 使用clean输出作为fallback
                if len(outputs_ir) < len(outputs_clean):
                    outputs_ir.append(outputs_clean[len(outputs_ir)])
                else:
                    # 如果已经处理完所有clean输出，创建一个零输出
                    outputs_ir.append(torch.zeros_like(outputs_clean[0]))
    
    # 计算δ
    all_deltas = []
    all_norms_clean = []
    
    for out_clean, out_ir in zip(outputs_clean, outputs_ir):
        # 计算 |y_tilde - y|_2
        diff = out_ir - out_clean
        diff_norm = torch.norm(diff, p=2, dim=1)  # [batch]
        
        # 计算 |y|_2
        clean_norm = torch.norm(out_clean, p=2, dim=1)  # [batch]
        
        # 计算相对扰动
        delta_batch = diff_norm / (clean_norm + epsilon)  # [batch]
        all_deltas.append(delta_batch)
        all_norms_clean.append(clean_norm)
    
    # 全局δ
    if all_deltas:
        all_deltas_tensor = torch.cat(all_deltas)  # [total_samples]
        all_norms_clean_tensor = torch.cat(all_norms_clean)  # [total_samples]
        
        delta_global = all_deltas_tensor.mean().item()
        delta_std = all_deltas_tensor.std().item()
    else:
        delta_global = 0.0
        delta_std = 0.0
    
    # 计算applied_ratio
    applied_ratio = 1.0 - (num_skipped / max(total_batches, 1))
    
    return {
        'delta_global': delta_global,
        'delta_mean': delta_global,
        'delta_std': delta_std,
        'applied_ratio': applied_ratio,
        'num_skipped': num_skipped,
        'total_batches': total_batches,
    }


def calibrate_strength(
    checkpoint_path: str,
    config: Dict[str, Any],
    calibration_samples: int = 512,
    target_deltas: List[float] = [0.05, 0.10, 0.20],
    ir_drop_mode: str = 'paper',
    strength_param_name: str = 'ir_drop_scaling',
    strength_range: Tuple[float, float] = (0.0, 2.0),
    num_trials: int = 20,
) -> Dict[float, float]:
    """
    校准强度参数，使得δ达到目标值。
    
    Args:
        checkpoint_path: checkpoint路径
        config: 配置字典
        calibration_samples: 用于校准的样本数
        target_deltas: 目标δ值列表 [low, mid, high]
        ir_drop_mode: IR-drop模式 ('paper' 或 'crossbar')
        strength_param_name: 强度参数名 ('ir_drop_scaling' 或 'ir_drop_cap')
        strength_range: 强度参数搜索范围
        num_trials: 二分搜索尝试次数
    
    Returns:
        字典 {target_delta: strength_value}
    """
    device = torch.device(config.get('device', 'cuda'))
    set_seed(config.get('seed', 42))
    
    # 从checkpoint中获取正确的num_classes
    num_classes = get_num_classes_from_checkpoint(
        checkpoint_path, 
        default_num_classes=config.get('num_classes', 10),
        device=device
    )
    
    # 创建模型（使用从checkpoint中获取的num_classes）
    base_model = get_model(
        name=config['model_name'],
        num_classes=num_classes,
    )
    base_model = base_model.to(device)
    
    # 加载checkpoint到base_model（使用智能加载，处理baseline和memristor两种格式）
    checkpoint = load_checkpoint_smart(
        checkpoint_path,
        base_model,
        device,
        device_model=None,  # 校准阶段，先不创建device_model
    )
    
    # 创建device_model（临时配置，用于校准）
    memristor_config = config['memristor']
    device_model = MemristorDeviceModel(
        G_min=float(memristor_config['G_min']),
        G_max=float(memristor_config['G_max']),
        weight_clip=memristor_config['weight_clip'],
        variability_sigma=float(memristor_config.get('variability_sigma', 0.0)),
        read_noise_sigma=float(memristor_config.get('read_noise_sigma', 0.0)),
        drift_alpha=float(memristor_config.get('drift_alpha', 0.0)),
        stuck_ratio=float(memristor_config.get('stuck_ratio', 0.0)),
        stuck_low_prob=float(memristor_config.get('stuck_low_prob', 0.5)),
        ir_drop_beta=float(memristor_config.get('ir_drop_beta', 0.01)),
        mapping=memristor_config.get('mapping', 'linear'),
        ir_drop_mode=ir_drop_mode,
        ir_drop_gamma=float(memristor_config.get('ir_drop_gamma', 0.35)),
        ir_drop_scaling=1.0,  # 临时值
        ir_drop_eta=float(memristor_config.get('ir_drop_eta', 1.0)),
        ir_drop_cap=0.10,  # 临时值
        ir_drop_norm=memristor_config.get('ir_drop_norm', 'mean'),
        ir_drop_train_enabled=True,  # 确保训练时也应用
    )
    
    # 包装模型（此时base_model已经加载了权重）
    model = wrap_model_with_memristor(base_model, device_model, use_learned_mapping=False)
    model = model.to(device)
    
    # 准备校准数据
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset_name=config['dataset'],
        data_root=config['data_root'],
        batch_size=config['batch_size'],
        num_workers=config.get('num_workers', 4),
        val_split=config.get('val_split', 0.1),
    )
    
    # 取前N个样本
    calibration_indices = list(range(min(calibration_samples, len(train_loader.dataset))))
    calibration_dataset = Subset(train_loader.dataset, calibration_indices)
    calibration_loader = DataLoader(
        calibration_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 4),
    )
    
    # 对于每个目标δ，找到对应的强度参数
    result = {}
    
    for target_delta in target_deltas:
        logger.info(f"\n{'='*60}")
        logger.info(f"Calibrating {ir_drop_mode} mode for target δ = {target_delta:.3f}")
        logger.info(f"{'='*60}\n")
        print(f"\n{'='*60}")
        print(f"Calibrating {ir_drop_mode} mode for target δ = {target_delta:.3f}")
        print(f"{'='*60}\n")
        
        # 二分搜索强度参数
        low, high = strength_range
        best_strength = None
        best_delta_diff = float('inf')
        
        for trial in range(num_trials):
            mid = (low + high) / 2.0
            
            # 设置强度参数
            if strength_param_name == 'ir_drop_scaling':
                device_model.ir_drop_scaling = mid
                logger.debug(f"Set ir_drop_scaling = {mid:.6f}")
            elif strength_param_name == 'ir_drop_cap':
                device_model.ir_drop_cap = mid
                # 对于crossbar模式，还需要设置ir_drop_beta
                device_model.ir_drop_beta = 1.0  # 使用最大beta，用cap控制强度
                logger.debug(f"Set ir_drop_cap = {mid:.6f}, ir_drop_beta = 1.0")
            else:
                raise ValueError(f"Unknown strength_param_name: {strength_param_name}")
            
            # 验证设置是否生效
            if strength_param_name == 'ir_drop_scaling':
                actual_scaling = device_model.ir_drop_scaling
                if abs(actual_scaling - mid) > 1e-6:
                    logger.warning(f"Warning: Set scaling to {mid:.6f} but device_model has {actual_scaling:.6f}")
            
            # 计算当前δ
            delta_result = compute_delta(
                model, calibration_loader, device, device_model
            )
            current_delta = delta_result['delta_global']
            
            logger.info(f"Trial {trial+1}/{num_trials}: strength={mid:.4f}, δ={current_delta:.4f}, target={target_delta:.4f}, applied_ratio={delta_result.get('applied_ratio', 1.0):.3f}")
            print(
                f"Trial {trial + 1}/{num_trials}: strength={mid:.16f}, δ={current_delta:.4f}, target={target_delta:.4f}, applied_ratio={delta_result.get('applied_ratio', 1.0):.3f}")

            # 更新搜索范围
            delta_diff = abs(current_delta - target_delta)
            if delta_diff < best_delta_diff:
                best_delta_diff = delta_diff
                best_strength = mid
            
            if current_delta < target_delta:
                low = mid
            else:
                high = mid
            
            # 如果足够接近，提前退出
            if delta_diff < 0.005:  # 0.5%误差
                break
        
        result[target_delta] = best_strength
        logger.info(f"Found strength={best_strength:.4f} for target δ={target_delta:.3f} (actual δ≈{current_delta:.4f})")
    
    return result


def train_with_matched_delta(
    checkpoint_path: str,
    config: Dict[str, Any],
    ir_drop_mode: str,
    strength_value: float,
    strength_param_name: str,
    output_dir: str,
    target_delta: float,
    calibration_samples: int = 512,
) -> Dict[str, Any]:
    """
    使用校准后的强度参数进行训练。
    
    Args:
        checkpoint_path: 初始checkpoint路径
        config: 配置字典
        ir_drop_mode: IR-drop模式
        strength_value: 强度参数值
        strength_param_name: 强度参数名
        output_dir: 输出目录
        target_delta: 目标δ值（用于记录）
        calibration_samples: 用于计算δ的样本数
    
    Returns:
        训练结果字典
    """
    device = torch.device(config.get('device', 'cuda'))
    set_seed(config.get('seed', 42))
    
    # 修改配置
    config = config.copy()
    config['memristor'] = config['memristor'].copy()
    
    # 确保experiment配置存在
    if 'experiment' not in config:
        config['experiment'] = {}
    
    # 设置实验模式为memristor_with_comp（HAT训练）
    config['experiment']['mode'] = 'memristor_with_comp'
    config['experiment']['compensation_method'] = 'hat'  # 使用HAT补偿
    
    # 设置IR-drop模式
    config['memristor']['ir_drop_mode'] = ir_drop_mode
    
    if strength_param_name == 'ir_drop_scaling':
        config['memristor']['ir_drop_scaling'] = strength_value
        # 确保训练时也应用
        config['memristor']['enable_ir_drop_paper_during_training'] = True
    elif strength_param_name == 'ir_drop_cap':
        config['memristor']['ir_drop_cap'] = strength_value
        config['memristor']['ir_drop_beta'] = 1.0  # 使用最大beta
        config['memristor']['ir_drop_train_enabled'] = True
    
    # 禁用其他噪声，避免混淆
    config['memristor']['variability_sigma'] = 0.0
    config['memristor']['read_noise_sigma'] = 0.0
    config['memristor']['drift_alpha'] = 0.0
    config['memristor']['stuck_ratio'] = 0.0
    
    # 设置实验名称
    mode_str = 'paper' if ir_drop_mode == 'paper' else 'crossbar'
    config['experiment_name'] = f"matched_distortion_{mode_str}_delta{target_delta:.2f}"
    
    # 运行训练
    logger.info(f"\n{'='*60}")
    logger.info(f"Training with {ir_drop_mode} IR-drop, strength={strength_value:.4f}, target δ={target_delta:.3f}")
    logger.info(f"{'='*60}\n")
    
    try:
        results = run_experiment(config, output_dir)
        
        # 训练后计算δ（使用训练好的模型）
        logger.info("Computing δ after training...")
        
        # 确定要使用的checkpoint路径
        final_checkpoint_path = os.path.join(output_dir, 'model_final.pth')
        checkpoint_to_use = final_checkpoint_path if os.path.exists(final_checkpoint_path) else checkpoint_path
        if checkpoint_to_use == checkpoint_path:
            logger.warning(f"Final checkpoint not found at {final_checkpoint_path}, using initial checkpoint")
        
        # 从checkpoint中获取正确的num_classes
        num_classes = get_num_classes_from_checkpoint(
            checkpoint_to_use,
            default_num_classes=config.get('num_classes', 10),
            device=device
        )
        
        # 创建模型（使用从checkpoint中获取的num_classes）
        base_model = get_model(
            name=config['model_name'],
            num_classes=num_classes,
        )
        base_model = base_model.to(device)
        
        # 加载checkpoint到base_model
        load_checkpoint(checkpoint_to_use, model=base_model, device=device)
        logger.info(f"Loaded checkpoint from {checkpoint_to_use}")
        
        # 创建device_model
        memristor_config = config['memristor']
        device_model = MemristorDeviceModel(
            G_min=float(memristor_config['G_min']),
            G_max=float(memristor_config['G_max']),
            weight_clip=memristor_config['weight_clip'],
            variability_sigma=0.0,  # 已禁用
            read_noise_sigma=0.0,  # 已禁用
            drift_alpha=0.0,  # 已禁用
            stuck_ratio=0.0,  # 已禁用
            ir_drop_beta=float(memristor_config.get('ir_drop_beta', 0.01)),
            mapping=memristor_config.get('mapping', 'linear'),
            ir_drop_mode=ir_drop_mode,
            ir_drop_gamma=float(memristor_config.get('ir_drop_gamma', 0.35)),
            ir_drop_scaling=float(memristor_config.get('ir_drop_scaling', 1.0)),
            ir_drop_eta=float(memristor_config.get('ir_drop_eta', 1.0)),
            ir_drop_cap=float(memristor_config.get('ir_drop_cap', 0.10)),
            ir_drop_norm=memristor_config.get('ir_drop_norm', 'mean'),
            ir_drop_train_enabled=True,
        )
        
        # 使用智能加载checkpoint（处理memristor包装后的checkpoint）
        checkpoint = load_checkpoint_smart(
            checkpoint_to_use,
            base_model,
            device,
            device_model=device_model,
        )
        
        # 包装模型（此时base_model已经加载了权重）
        model = wrap_model_with_memristor(base_model, device_model, use_learned_mapping=False)
        model = model.to(device)
        
        # 准备数据
        train_loader, val_loader, test_loader = get_dataloaders(
            dataset_name=config['dataset'],
            data_root=config['data_root'],
            batch_size=config['batch_size'],
            num_workers=config.get('num_workers', 4),
            val_split=config.get('val_split', 0.1),
        )
        
        # 取前N个样本用于计算δ
        calibration_indices = list(range(min(calibration_samples, len(train_loader.dataset))))
        calibration_dataset = Subset(train_loader.dataset, calibration_indices)
        calibration_loader = DataLoader(
            calibration_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config.get('num_workers', 4),
        )
        
        # 计算δ
        delta_result = compute_delta(model, calibration_loader, device, device_model)
        
        logger.info(f"Final δ: {delta_result['delta_global']:.4f} (target: {target_delta:.4f})")
        logger.info(f"Applied ratio: {delta_result['applied_ratio']:.4f}")
        print(f"Final δ: {delta_result['delta_global']:.4f} (target: {target_delta:.4f})")
        print(f"Applied ratio: {delta_result['applied_ratio']:.4f}")
        
        # 将δ结果添加到results中
        if 'metrics_history' in results:
            # 在每个epoch的metrics中添加δ（如果有的话）
            for metrics in results['metrics_history']:
                if 'delta' not in metrics:
                    metrics['delta'] = None  # 训练过程中未计算
        
        # 添加最终δ到results
        results['final_delta'] = delta_result['delta_global']
        results['delta_applied_ratio'] = delta_result['applied_ratio']
        
        return {
            'success': True,
            'results': results,
            'target_delta': target_delta,
            'actual_delta': delta_result['delta_global'],
            'strength_value': strength_value,
            'ir_drop_mode': ir_drop_mode,
            'delta_applied_ratio': delta_result['applied_ratio'],
        }
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e),
            'target_delta': target_delta,
            'strength_value': strength_value,
            'ir_drop_mode': ir_drop_mode,
        }


def main():
    parser = argparse.ArgumentParser(description='Matched-Distortion Learnability Test')
    parser.add_argument('--config', type=str, required=True, help='Base config file path')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to baseline checkpoint')
    parser.add_argument('--output_dir', type=str, default='./outputs/matched_distortion', help='Output directory')
    parser.add_argument('--calibration_samples', type=int, default=512, help='Number of samples for calibration')
    parser.add_argument('--target_deltas', type=float, nargs='+', default=[0.05, 0.10, 0.20], help='Target delta values')
    parser.add_argument('--skip_calibration', action='store_true', help='Skip calibration and use provided strengths')
    parser.add_argument('--paper_strengths', type=float, nargs='+', help='Paper mode strengths (if skipping calibration)')
    parser.add_argument('--crossbar_strengths', type=float, nargs='+', help='Crossbar mode strengths (if skipping calibration)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset name (overrides config file, e.g., cifar10, cifar100, mnist)')
    
    args = parser.parse_args()
    
    # 设置日志（使用绝对路径避免相对路径问题）
    output_dir_abs = os.path.abspath(args.output_dir)
    os.makedirs(output_dir_abs, exist_ok=True)
    log_file = os.path.join(output_dir_abs, 'matched_distortion.log')
    setup_logger(log_file)
    # 更新args.output_dir为绝对路径，后续使用
    args.output_dir = output_dir_abs
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 如果命令行指定了dataset，覆盖配置文件中的值
    if args.dataset is not None:
        config['dataset'] = args.dataset
        logger.info(f"Using dataset from command line: {args.dataset}")
    else:
        # 使用配置文件中的dataset，如果没有则使用默认值
        if 'dataset' not in config:
            config['dataset'] = 'cifar10'  # 默认值
            logger.info(f"Dataset not specified in config, using default: {config['dataset']}")
        else:
            logger.info(f"Using dataset from config file: {config['dataset']}")
    
    config['seed'] = args.seed
    set_seed(args.seed)
    
    # 步骤1：校准强度
    if not args.skip_calibration:
        logger.info("="*60)
        logger.info("Step 1: Calibrating strength parameters")
        logger.info("="*60)
        
        # 校准paper模式
        logger.info("\nCalibrating paper IR-drop mode...")
        paper_strengths = calibrate_strength(
            checkpoint_path=args.checkpoint,
            config=config,
            calibration_samples=args.calibration_samples,
            target_deltas=args.target_deltas,
            ir_drop_mode='paper',
            strength_param_name='ir_drop_scaling',
            strength_range=(0.0, 2.0),
        )
        
        # 校准crossbar模式
        logger.info("\nCalibrating crossbar IR-drop mode...")
        crossbar_strengths = calibrate_strength(
            checkpoint_path=args.checkpoint,
            config=config,
            calibration_samples=args.calibration_samples,
            target_deltas=args.target_deltas,
            ir_drop_mode='crossbar',
            strength_param_name='ir_drop_cap',
            strength_range=(0.0, 0.5),
        )
        
        # 保存校准结果
        calibration_results = {
            'paper_strengths': paper_strengths,
            'crossbar_strengths': crossbar_strengths,
            'target_deltas': args.target_deltas,
        }
        # 确保输出目录存在（使用绝对路径避免相对路径问题）
        output_dir_abs = os.path.abspath(args.output_dir)
        os.makedirs(output_dir_abs, exist_ok=True)
        calibration_file = os.path.join(output_dir_abs, 'calibration_results.json')
        with open(calibration_file, 'w') as f:
            json.dump(calibration_results, f, indent=2)
        logger.info(f"Saved calibration results to {calibration_file}")
    else:
        # 使用提供的强度值
        if args.paper_strengths is None or args.crossbar_strengths is None:
            raise ValueError("Must provide --paper_strengths and --crossbar_strengths when --skip_calibration is set")
        
        paper_strengths = dict(zip(args.target_deltas, args.paper_strengths))
        crossbar_strengths = dict(zip(args.target_deltas, args.crossbar_strengths))
    
    # 步骤2：正式训练对照
    logger.info("\n" + "="*60)
    logger.info("Step 2: Training with matched delta")
    logger.info("="*60)
    
    all_results = []
    
    for target_delta in args.target_deltas:
        paper_strength = paper_strengths[target_delta]
        crossbar_strength = crossbar_strengths[target_delta]
        
        # A组：paper IR-drop
        paper_output_dir = os.path.join(args.output_dir, f'paper_delta{target_delta:.2f}')
        os.makedirs(paper_output_dir, exist_ok=True)
        
        paper_result = train_with_matched_delta(
            checkpoint_path=args.checkpoint,
            config=config,
            ir_drop_mode='paper',
            strength_value=paper_strength,
            strength_param_name='ir_drop_scaling',
            output_dir=paper_output_dir,
            target_delta=target_delta,
        )
        all_results.append(paper_result)
        
        # B组：crossbar IR-drop
        crossbar_output_dir = os.path.join(args.output_dir, f'crossbar_delta{target_delta:.2f}')
        os.makedirs(crossbar_output_dir, exist_ok=True)
        
        crossbar_result = train_with_matched_delta(
            checkpoint_path=args.checkpoint,
            config=config,
            ir_drop_mode='crossbar',
            strength_value=crossbar_strength,
            strength_param_name='ir_drop_cap',
            output_dir=crossbar_output_dir,
            target_delta=target_delta,
        )
        all_results.append(crossbar_result)
    
    # 保存所有结果（确保目录存在）
    os.makedirs(args.output_dir, exist_ok=True)
    results_file = os.path.join(args.output_dir, 'all_results.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"Saved all results to {results_file}")
    
    # 打印总结
    logger.info("\n" + "="*60)
    logger.info("Summary")
    logger.info("="*60)
    for result in all_results:
        mode = result['ir_drop_mode']
        delta = result['target_delta']
        strength = result['strength_value']
        success = result['success']
        logger.info(f"{mode} (δ={delta:.3f}, strength={strength:.4f}): {'SUCCESS' if success else 'FAILED'}")


if __name__ == '__main__':
    main()

