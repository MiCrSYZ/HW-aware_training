"""
扰动强度标定脚本

用于统一不同噪声类型的扰动程度。给定目标扰动强度δ*，找到对应的噪声参数θ*使得δ(θ*) = δ*。

扰动强度定义：
- δ_logit: logits级别的RMS相对偏差
- δ_block: block级别的RMS相对偏差（用于诊断）
"""

import argparse
import json
import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from scipy.optimize import brentq
import logging

try:
    from ..models.model_zoo import get_model, wrap_model_with_memristor
    from ..memristor.device_model import MemristorDeviceModel
    from ..data.dataset import get_dataloaders
    from ..utils.seeds import set_seed
    from ..utils.checkpoint import load_checkpoint
except ImportError:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    from src.models.model_zoo import get_model, wrap_model_with_memristor
    from src.memristor.device_model import MemristorDeviceModel
    from src.data.dataset import get_dataloaders
    from src.utils.seeds import set_seed
    from src.utils.checkpoint import load_checkpoint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_delta_logit(
    clean_outputs: torch.Tensor,
    noisy_outputs: torch.Tensor,
) -> float:
    """
    计算logits级别的RMS相对偏差。
    
    δ_logit = sqrt(E[||z_noisy - z_clean||^2] / E[||z_clean||^2])
    
    Args:
        clean_outputs: [batch, num_classes] 干净模型的logits输出
        noisy_outputs: [batch, num_classes] 噪声模型的logits输出
        
    Returns:
        δ_logit值
    """
    diff = noisy_outputs - clean_outputs
    diff_norm_sq = torch.sum(diff ** 2, dim=-1)  # [batch]
    clean_norm_sq = torch.sum(clean_outputs ** 2, dim=-1)  # [batch]
    
    # 避免除零
    clean_norm_sq = torch.clamp(clean_norm_sq, min=1e-12)
    
    # 计算每个样本的相对偏差平方
    rel_diff_sq = diff_norm_sq / clean_norm_sq  # [batch]
    
    # 取期望（平均）
    mean_rel_diff_sq = torch.mean(rel_diff_sq)
    
    # 开方得到RMS相对偏差
    delta = torch.sqrt(mean_rel_diff_sq).item()
    
    return delta


def compute_delta_block(
    clean_features: List[torch.Tensor],
    noisy_features: List[torch.Tensor],
) -> Dict[int, float]:
    """
    计算block级别的RMS相对偏差。
    
    δ_block(k) = sqrt(E[|h^(k)_noisy - h^(k)_clean|^2] / E[|h^(k)_clean|^2])
    
    Args:
        clean_features: List of [batch, ...] tensors，每个block的干净特征
        noisy_features: List of [batch, ...] tensors，每个block的噪声特征
        
    Returns:
        Dict mapping block index to δ_block值
    """
    delta_blocks = {}
    
    for k, (clean_feat, noisy_feat) in enumerate(zip(clean_features, noisy_features)):
        # Flatten features
        clean_flat = clean_feat.reshape(clean_feat.shape[0], -1)  # [batch, features]
        noisy_flat = noisy_feat.reshape(noisy_feat.shape[0], -1)  # [batch, features]
        
        diff = noisy_flat - clean_flat
        diff_norm_sq = torch.sum(diff ** 2, dim=-1)  # [batch]
        clean_norm_sq = torch.sum(clean_flat ** 2, dim=-1)  # [batch]
        
        # 避免除零
        clean_norm_sq = torch.clamp(clean_norm_sq, min=1e-12)
        
        # 计算每个样本的相对偏差平方
        rel_diff_sq = diff_norm_sq / clean_norm_sq  # [batch]
        
        # 取期望（平均）
        mean_rel_diff_sq = torch.mean(rel_diff_sq)
        
        # 开方得到RMS相对偏差
        delta = torch.sqrt(mean_rel_diff_sq).item()
        delta_blocks[k] = delta
    
    return delta_blocks


def extract_block_features(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    model_name: str,
    is_agnews: bool = False,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    提取模型的block级别特征和logits输出。
    
    Args:
        model: 模型（可能是wrapped的）
        dataloader: 数据加载器
        device: 设备
        model_name: 模型名称 ('resnet20', 'vit_tiny', or 'gru_agnews')
        is_agnews: 是否为AG News数据集（需要处理lengths）
        
    Returns:
        (block_features_list, logits_outputs)
        block_features_list: List of [batch, ...] tensors
        logits_outputs: [total_batch, num_classes]
    """
    model.eval()
    
    # 获取base_model（如果是wrapped的）
    base_model = model
    if hasattr(model, 'base_model'):
        base_model = model.base_model
    
    block_features_list = []
    logits_list = []
    
    # 注册hook来提取block特征
    activations = {}
    activation_order = []  # 保持顺序
    
    def get_activation(name):
        def hook(module, input, output):
            if name not in activations:
                activation_order.append(name)
            # 对于GRU，output是(output, hidden)元组
            if isinstance(output, tuple):
                activations[name] = output[1].detach()  # 使用hidden state
            else:
                activations[name] = output.detach()
        return hook
    
    # 根据模型类型注册hook
    hooks = []
    if model_name == 'resnet20':
        # ResNet20: 提取每个layer的输出（layer1, layer2, layer3）
        for name, module in base_model.named_modules():
            if name in ['layer1', 'layer2', 'layer3']:
                hooks.append(module.register_forward_hook(get_activation(name)))
    elif model_name == 'vit_tiny':
        # ViT: 提取每个TransformerBlock的输出（blocks.0, blocks.1, ...）
        for name, module in base_model.named_modules():
            if name.startswith('blocks.') and name.endswith('.norm2'):
                # 提取每个block的最后一个norm2的输出
                hooks.append(module.register_forward_hook(get_activation(name)))
    elif model_name == 'gru_agnews':
        # GRU: 提取两层GRU的输出（gru层）
        # 对于GRU，我们需要提取每层的hidden state
        # 注册hook到GRU层，提取每层的hidden state
        for name, module in base_model.named_modules():
            if name == 'gru':
                hooks.append(module.register_forward_hook(get_activation(name)))
                break  # 只注册一次
    
    with torch.no_grad():
        for batch in dataloader:
            # 处理数据格式
            if is_agnews:
                labels, texts, lengths = batch
                data = texts.to(device)
                lengths = lengths.to(device)
            else:
                data, _ = batch
                data = data.to(device)
                lengths = None
            
            # Forward pass
            try:
                if hasattr(model, 'forward'):
                    forward_code = model.forward.__code__
                    forward_varnames = forward_code.co_varnames
                    # 检查是否支持lengths参数
                    if 'lengths' in forward_varnames and lengths is not None:
                        if 't' in forward_varnames:
                            output = model(data, lengths=lengths, t=0, seed=42)
                        else:
                            output = model(data, lengths=lengths)
                    elif 't' in forward_varnames:
                        output = model(data, t=0, seed=42)
                    else:
                        output = model(data)
                else:
                    output = model(data)
            except TypeError:
                # Fallback
                if lengths is not None:
                    try:
                        output = model(data, lengths=lengths)
                    except TypeError:
                        output = model(data)
                else:
                    output = model(data)
            
            logits_list.append(output.cpu())
            
            # 收集block特征（按注册顺序）
            if activations:
                # 按activation_order顺序收集
                if not block_features_list:
                    # 第一次，初始化列表
                    for key in activation_order:
                        feat = activations[key].cpu()
                        # 对于GRU，hidden state是[num_layers, batch, hidden_dim]
                        # 我们需要提取每层的hidden state作为独立的block
                        if model_name == 'gru_agnews' and feat.dim() == 3:
                            # 分离两层GRU的hidden state
                            num_layers = feat.shape[0]
                            for layer_idx in range(num_layers):
                                # 提取第layer_idx层的hidden state [batch, hidden_dim]
                                layer_hidden = feat[layer_idx]
                                block_features_list.append([layer_hidden])
                        else:
                            block_features_list.append([feat])
                else:
                    # 后续，追加到对应位置
                    for idx, key in enumerate(activation_order):
                        if key in activations:
                            feat = activations[key].cpu()
                            if model_name == 'gru_agnews' and feat.dim() == 3:
                                # 分离两层GRU的hidden state
                                num_layers = feat.shape[0]
                                for layer_idx in range(num_layers):
                                    # 计算对应的block索引
                                    block_idx = idx * num_layers + layer_idx
                                    if block_idx < len(block_features_list):
                                        layer_hidden = feat[layer_idx]
                                        block_features_list[block_idx].append(layer_hidden)
                            else:
                                if idx < len(block_features_list):
                                    block_features_list[idx].append(feat)
                activations.clear()  # 清空以便下次使用
                activation_order.clear()
    
    # 合并所有batch的特征
    logits_outputs = torch.cat(logits_list, dim=0)
    block_features = [torch.cat(feat_list, dim=0) for feat_list in block_features_list]
    
    # 移除hooks
    for hook in hooks:
        hook.remove()
    
    return block_features, logits_outputs


def compute_delta_for_params(
    config: Dict[str, Any],
    base_model: nn.Module,
    calibration_loader: DataLoader,
    device: torch.device,
    noise_type: str,
    theta: float,
    model_name: str,
    vocab: Optional[Any] = None,
) -> Tuple[float, Dict[int, float]]:
    """
    计算给定噪声参数θ下的扰动强度δ。
    
    Args:
        config: 配置字典
        base_model: 基线模型（干净模型）
        calibration_loader: 标定数据加载器
        device: 设备
        noise_type: 噪声类型 ('variability_sigma', 'cond1_alpha', 'cond2_alpha', 'adc_bits')
        theta: 噪声参数值
        model_name: 模型名称
        
    Returns:
        (delta_logit, delta_blocks_dict)
    """
    # 准备模型参数
    model_kwargs = {}
    if model_name == 'vit_tiny':
        model_kwargs = {k: v for k, v in config.items() if k in ['patch_size', 'embed_dim', 'depth', 'num_heads', 'mlp_ratio', 'qkv_bias']}
    elif model_name == 'resnet20':
        model_kwargs = {k: v for k, v in config.items() if k in ['in_channels']}
    elif model_name == 'gru_agnews':
        if vocab is None:
            raise ValueError("vocab is required for GRU model")
        model_kwargs = {
            'vocab_size': len(vocab),
            'embed_dim': config.get('embed_dim', 128),
            'hidden_dim': config.get('hidden_dim', 256),
            'num_layers': config.get('num_layers', 2),
        }
    
    # 创建干净模型（无噪声）
    clean_model = get_model(
        config['model_name'],
        num_classes=config['num_classes'],
        **model_kwargs
    ).to(device)
    clean_model.load_state_dict(base_model.state_dict())
    clean_model.eval()
    
    # 创建噪声模型
    memristor_config = config['memristor'].copy()
    
    # 根据noise_type设置参数
    if noise_type == 'variability_sigma':
        memristor_config['variability_sigma'] = theta
        memristor_config['synthetic_noise_type'] = 'full_variability'
        memristor_config['read_noise_sigma'] = 0.0
        memristor_config['drift_alpha'] = 0.0
        memristor_config['stuck_ratio'] = 0.0
        memristor_config['ir_drop_mode'] = 'none'
        memristor_config['enable_adc'] = False
    elif noise_type == 'cond1_alpha':
        memristor_config['variability_sigma'] = 0.0
        memristor_config['synthetic_noise_type'] = 'cond1_variance_bounded'
        memristor_config['cond1_alpha'] = theta
        memristor_config['read_noise_sigma'] = 0.0
        memristor_config['drift_alpha'] = 0.0
        memristor_config['stuck_ratio'] = 0.0
        memristor_config['ir_drop_mode'] = 'none'
        memristor_config['enable_adc'] = False
    elif noise_type == 'cond2_alpha':
        memristor_config['variability_sigma'] = 0.0
        memristor_config['synthetic_noise_type'] = 'cond2_gradient_unbiased'
        memristor_config['cond2_alpha'] = theta
        memristor_config['read_noise_sigma'] = 0.0
        memristor_config['drift_alpha'] = 0.0
        memristor_config['stuck_ratio'] = 0.0
        memristor_config['ir_drop_mode'] = 'none'
        memristor_config['enable_adc'] = False
    elif noise_type == 'adc_bits':
        memristor_config['variability_sigma'] = 0.0
        memristor_config['synthetic_noise_type'] = 'cond3_adc_direct'
        memristor_config['adc_bits'] = max(2, int(round(theta)))  # 确保是整数且>=2
        memristor_config['enable_adc'] = True
        memristor_config['enable_adc_during_training'] = True
        memristor_config['adc_training_mode'] = 'direct'
        memristor_config['read_noise_sigma'] = 0.0
        memristor_config['drift_alpha'] = 0.0
        memristor_config['stuck_ratio'] = 0.0
        memristor_config['ir_drop_mode'] = 'none'
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")
    
    # 创建device model
    # 过滤掉不应该传递给MemristorDeviceModel的参数，并确保数值参数被正确转换
    excluded_keys = {'noise_injection', 'write'}  # write是字典，不是MemristorDeviceModel的参数
    
    # 需要转换为float的参数
    float_params = {
        'G_min', 'G_max', 'variability_sigma', 'read_noise_sigma', 'drift_alpha',
        'stuck_ratio', 'stuck_low_prob', 'ir_drop_beta', 'ir_drop_gamma',
        'ir_drop_scaling', 'ir_drop_eta', 'ir_drop_cap', 'cond1_alpha', 'cond1_nu',
        'cond2_alpha'
    }
    
    # 需要转换为int的参数
    int_params = {'array_size', 'adc_bits', 'drift_time_fixed'}
    
    # 需要转换为tuple的参数
    tuple_params = {'weight_clip'}
    
    # 需要转换为bool的参数
    bool_params = {
        'enable_update_model', 'enable_adc', 'adc_add_noise', 'enable_energy',
        'ir_drop_train_enabled', 'enable_adc_during_training',
        'enable_ir_drop_paper_during_training'
    }
    
    # 构建参数字典，进行类型转换
    device_model_kwargs = {'seed': config.get('seed', 42)}
    for k, v in memristor_config.items():
        if k in excluded_keys:
            continue
        elif k in float_params and v is not None:
            device_model_kwargs[k] = float(v)
        elif k in int_params and v is not None:
            device_model_kwargs[k] = int(v)
        elif k in tuple_params and v is not None:
            device_model_kwargs[k] = tuple(float(x) for x in v) if isinstance(v, (list, tuple)) else v
        elif k in bool_params and v is not None:
            device_model_kwargs[k] = bool(v)
        else:
            # 其他参数（字符串等）直接传递
            device_model_kwargs[k] = v
    
    device_model = MemristorDeviceModel(**device_model_kwargs)
    
    # 创建噪声模型
    noisy_model = get_model(
        config['model_name'],
        num_classes=config['num_classes'],
        **model_kwargs
    ).to(device)
    noisy_model.load_state_dict(base_model.state_dict())
    noisy_model = wrap_model_with_memristor(
        noisy_model,
        device_model,
        noise_config=memristor_config.get('noise_injection')
    )
    noisy_model.eval()
    
    # 判断是否为AG News数据集
    is_agnews = (config.get('dataset', '').lower() == 'agnews')
    
    # 提取特征
    clean_features, clean_logits = extract_block_features(
        clean_model, calibration_loader, device, model_name, is_agnews=is_agnews
    )
    noisy_features, noisy_logits = extract_block_features(
        noisy_model, calibration_loader, device, model_name, is_agnews=is_agnews
    )
    
    # 计算δ
    delta_logit = compute_delta_logit(clean_logits, noisy_logits)
    delta_blocks = compute_delta_block(clean_features, noisy_features)
    
    return delta_logit, delta_blocks


def find_theta_for_delta(
    config: Dict[str, Any],
    base_model: nn.Module,
    calibration_loader: DataLoader,
    device: torch.device,
    noise_type: str,
    target_delta: float,
    model_name: str,
    theta_bounds: Tuple[float, float],
    vocab: Optional[Any] = None,
    tol: float = 1e-3,
    max_iter: int = 50,
) -> Tuple[float, Dict[str, Any]]:
    """
    使用二分搜索找到θ*使得δ(θ*) = δ*。
    
    Args:
        config: 配置字典
        base_model: 基线模型
        calibration_loader: 标定数据加载器
        device: 设备
        noise_type: 噪声类型
        target_delta: 目标扰动强度δ*
        model_name: 模型名称
        theta_bounds: (theta_min, theta_max) 参数搜索范围
        tol: 容差
        max_iter: 最大迭代次数
        
    Returns:
        (theta_star, result_dict)
    """
    theta_min, theta_max = theta_bounds
    
    def delta_func(theta):
        """计算给定θ下的δ值"""
        try:
            delta, _ = compute_delta_for_params(
                config, base_model, calibration_loader, device,
                noise_type, theta, model_name, vocab=vocab
            )
            return delta - target_delta  # 返回差值，用于二分搜索
        except Exception as e:
            logger.warning(f"Error computing delta for theta={theta}: {e}")
            return float('inf')
    
    # 使用scipy的brentq进行二分搜索
    try:
        theta_star = brentq(delta_func, theta_min, theta_max, xtol=tol, maxiter=max_iter)
    except ValueError as e:
        logger.error(f"Brentq failed: {e}. Trying manual binary search...")
        # 手动二分搜索
        theta_star = manual_binary_search(
            delta_func, theta_min, theta_max, tol, max_iter
        )
    
    # 计算最终结果
    delta_final, delta_blocks_final = compute_delta_for_params(
        config, base_model, calibration_loader, device,
        noise_type, theta_star, model_name, vocab=vocab
    )
    
    result = {
        'theta_star': theta_star,
        'delta_logit': delta_final,
        'delta_blocks': delta_blocks_final,
        'target_delta': target_delta,
        'error': abs(delta_final - target_delta),
    }
    
    return theta_star, result


def manual_binary_search(
    func,
    x_min: float,
    x_max: float,
    tol: float,
    max_iter: int,
) -> float:
    """手动二分搜索"""
    for i in range(max_iter):
        x_mid = (x_min + x_max) / 2
        f_mid = func(x_mid)
        
        if abs(f_mid) < tol:
            return x_mid
        
        f_min = func(x_min)
        if f_min * f_mid < 0:
            x_max = x_mid
        else:
            x_min = x_mid
    
    return (x_min + x_max) / 2


def main():
    parser = argparse.ArgumentParser(description='Calibrate noise strength')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--checkpoint', type=str, help='Model checkpoint path (optional)')
    parser.add_argument('--target_delta', type=float, required=True, help='Target perturbation strength δ*')
    parser.add_argument('--noise_type', type=str, required=True,
                        choices=['variability_sigma', 'cond1_alpha', 'cond2_alpha', 'adc_bits'],
                        help='Noise type to calibrate')
    parser.add_argument('--theta_min', type=float, help='Minimum theta value for search')
    parser.add_argument('--theta_max', type=float, help='Maximum theta value for search')
    parser.add_argument('--calibration_size', type=int, default=512, help='Calibration dataset size')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 设置设备
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    
    # 加载数据
    dataset_name = config['dataset'].lower()
    is_agnews = (dataset_name == 'agnews')
    
    if is_agnews:
        # AG News返回vocab
        train_loader, _, test_loader, vocab = get_dataloaders(
            dataset_name=config['dataset'],
            data_root=config['data_root'],
            batch_size=128,
            num_workers=4,
            val_split=0.0,  # 不使用验证集
        )
        logger.info(f"AG News vocabulary size: {len(vocab)}")
    else:
        _, _, test_loader = get_dataloaders(
            dataset_name=config['dataset'],
            data_root=config['data_root'],
            batch_size=128,
            num_workers=4,
            val_split=0.0,  # 不使用验证集
        )
        vocab = None
    
    # 创建标定数据集（固定512个样本）
    calibration_indices = list(range(min(args.calibration_size, len(test_loader.dataset))))
    calibration_dataset = Subset(test_loader.dataset, calibration_indices)
    # AG News 需使用与 test_loader 相同的 collate_fn，否则 batch 只有 (labels, texts) 没有 lengths
    calibration_loader = DataLoader(
        calibration_dataset,
        batch_size=128,
        shuffle=False,  # 固定顺序
        num_workers=4,
        collate_fn=test_loader.collate_fn if is_agnews else None,
    )
    
    # 加载基线模型
    model_name = config['model_name']
    
    # 准备模型参数
    model_kwargs = {}
    if model_name == 'vit_tiny':
        model_kwargs = {k: v for k, v in config.items() if k in ['patch_size', 'embed_dim', 'depth', 'num_heads', 'mlp_ratio', 'qkv_bias']}
    elif model_name == 'resnet20':
        model_kwargs = {k: v for k, v in config.items() if k in ['in_channels']}
    elif model_name == 'gru_agnews':
        if vocab is None:
            raise ValueError("vocab is required for GRU model")
        model_kwargs = {
            'vocab_size': len(vocab),
            'embed_dim': config.get('embed_dim', 128),
            'hidden_dim': config.get('hidden_dim', 256),
            'num_layers': config.get('num_layers', 2),
        }
    
    base_model = get_model(
        model_name,
        num_classes=config['num_classes'],
        **model_kwargs
    ).to(device)
    
    if args.checkpoint:
        checkpoint = load_checkpoint(args.checkpoint, model=base_model, device=device)
        logger.info(f"Loaded checkpoint from {args.checkpoint}")
    else:
        logger.info("Using randomly initialized model")
    
    base_model.eval()
    
    # 设置参数搜索范围
    if args.theta_min is None or args.theta_max is None:
        # 默认范围
        if args.noise_type == 'variability_sigma':
            theta_min, theta_max = 0.001, 0.5
        elif args.noise_type == 'cond1_alpha':
            theta_min, theta_max = 0.001, 1.0
        elif args.noise_type == 'cond2_alpha':
            theta_min, theta_max = 0.001, 1.0
        elif args.noise_type == 'adc_bits':
            theta_min, theta_max = 2.0, 16.0
        else:
            raise ValueError(f"Unknown noise_type: {args.noise_type}")
    else:
        theta_min, theta_max = args.theta_min, args.theta_max
    
    logger.info(f"Calibrating {args.noise_type} for target δ* = {args.target_delta}")
    logger.info(f"Search range: [{theta_min}, {theta_max}]")
    
    # 执行标定
    theta_star, result = find_theta_for_delta(
        config,
        base_model,
        calibration_loader,
        device,
        args.noise_type,
        args.target_delta,
        model_name,
        (theta_min, theta_max),
        vocab=vocab,
    )
    
    # 保存结果
    output_data = {
        'config_path': args.config,
        'checkpoint_path': args.checkpoint,
        'noise_type': args.noise_type,
        'target_delta': args.target_delta,
        'theta_star': theta_star,
        'delta_logit': result['delta_logit'],
        'delta_blocks': result['delta_blocks'],
        'error': result['error'],
        'calibration_size': args.calibration_size,
        'seed': args.seed,
        'theta_bounds': [theta_min, theta_max],
    }
    
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"Calibration complete!")
    logger.info(f"  θ* = {theta_star:.6f}")
    logger.info(f"  δ_logit = {result['delta_logit']:.6f}")
    logger.info(f"  Error = {result['error']:.6f}")
    logger.info(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()

