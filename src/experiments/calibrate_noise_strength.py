"""
扰动强度标定脚本

用于统一不同噪声类型的扰动程度。给定目标扰动强度δ*，找到对应的噪声参数θ*使得δ(θ*) = δ*。

扰动强度定义：
- δ_logit: logits级别的RMS相对偏差（forward 扰动）
- δ_grad:  梯度级别的RMS相对偏差（backward 扰动，与 δ_logit 同一定义便于比较）
- δ_grad_induced: 由 forward 扰动诱发的梯度扰动（同 δ_grad 定义），便于与 backward-only 的 δ_grad 在同一尺度比较；标定 forward 噪声时会一并计算并写入输出。
- δ_block: block级别的RMS相对偏差（用于诊断）

大扰动时 θ↔δ 可能非一一对应：
- 扰动较小时，θ 与 δ 通常单调、一一对应，正向标定（求 θ* 使 δ(θ*)=δ*）与逆向验证（给定 θ 算 δ）一致。
- 扰动较大时（如 clip_c 很小、裁剪很狠），激活饱和、δ(θ) 可能非单调或同一条 δ 对应多个 θ，导致「同一 δ 标定出一个 θ、用另一个 θ 去算也得到同一 δ」。此时只能认为「小到中等」δ 范围内标定可靠，大 δ 下不宜用单一 θ 反推或预测。

兼容两种配置：
- 若 config 含 'synth_noise'：使用新合成噪声路径（不走权重-电导映射），与 run_experiment_synth 一致。
- 若 config 含 'memristor'：使用原 memristor 路径（权重-电导映射）。
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
    from ..models.model_zoo import get_model, wrap_model_with_memristor, wrap_model_with_synth_noise
    from ..memristor.device_model import MemristorDeviceModel
    from ..memristor.synth_noise_wrappers import SynthNoiseConfig, apply_logits_backward_corruption
    from ..data.dataset import get_dataloaders
    from ..utils.seeds import set_seed
    from ..utils.checkpoint import load_checkpoint
except ImportError:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    from src.models.model_zoo import get_model, wrap_model_with_memristor, wrap_model_with_synth_noise
    from src.memristor.device_model import MemristorDeviceModel
    from src.memristor.synth_noise_wrappers import SynthNoiseConfig, apply_logits_backward_corruption
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


def _collect_grad_flattened(model: nn.Module, device: torch.device) -> torch.Tensor:
    """将模型中所有 requires_grad 参数的 .grad 拼成一行向量；无 grad 则视为 0。"""
    grads = []
    for p in model.parameters():
        if p.requires_grad:
            g = p.grad if p.grad is not None else torch.zeros_like(p, device=device)
        else:
            g = torch.zeros_like(p, device=device)
        grads.append(g.reshape(-1))
    return torch.cat(grads, dim=0) if grads else torch.tensor([], device=device)


def _collect_grad_flattened_from_base(model: nn.Module, device: torch.device) -> torch.Tensor:
    """从 base 网络收集梯度（便于比较无包装的 clean 与有包装的 noisy 时同一组参数）。"""
    target = getattr(model, 'base_model', model)
    return _collect_grad_flattened(target, device)


def compute_delta_grad(
    config: Dict[str, Any],
    base_model: nn.Module,
    calibration_loader: DataLoader,
    device: torch.device,
    noise_type: str,
    theta: float,
    model_name: str,
    vocab: Optional[Any] = None,
    is_agnews: bool = False,
    seed: Optional[int] = None,
) -> float:
    """
    计算 backward-only 扰动在梯度上的 RMS 相对偏差，与 δ_logit 定义一致、便于与 forward 扰动比较。
    
    δ_grad = sqrt(E[||g_noisy - g_clean||^2] / E[||g_clean||^2])
    
    其中 g 为 loss 对全部可训练参数的梯度（拼成向量）。对每个 batch 做一次 clean backward 和一次
    noisy backward（同权重、同输入），再在多个 batch 上按样本量加权平均。
    
    仅支持 synth_noise 下的 adv_direction_beta、sign_corrupt_p。
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed % (2**31))
    base_seed = config.get('seed', 42)
    model_kwargs = {}
    if model_name == 'vit_tiny':
        model_kwargs = {k: v for k, v in config.items() if k in ['patch_size', 'embed_dim', 'depth', 'num_heads', 'mlp_ratio', 'qkv_bias']}
    elif model_name == 'resnet20':
        model_kwargs = {k: v for k, v in config.items() if k in ['in_channels']}
    elif model_name == 'gru_agnews':
        if vocab is None:
            raise ValueError("vocab required for GRU")
        model_kwargs = {
            'vocab_size': len(vocab),
            'embed_dim': config.get('embed_dim', 128),
            'hidden_dim': config.get('hidden_dim', 256),
            'num_layers': config.get('num_layers', 2),
        }
    synth_config = config['synth_noise'].copy()
    noise_injection = synth_config.get('noise_injection')

    def _f(key, default):
        v = synth_config.get(key)
        return float(v) if v is not None else default

    def _b(key, default):
        v = synth_config.get(key)
        return bool(v) if v is not None else default

    at_logits = synth_config.get('backward_corruption_at') == 'logits'

    def make_config(noise_type_key: str, beta_val: float, p_val: float) -> SynthNoiseConfig:
        return SynthNoiseConfig(
            noise_type=noise_type_key,
            variability_sigma=_f('variability_sigma', 0.05),
            heavy_tail_alpha=_f('heavy_tail_alpha', 0.1),
            heavy_tail_nu=_f('heavy_tail_nu', 2.0),
            input_dependent_alpha=_f('input_dependent_alpha', 0.1),
            decoupled_consistent_sigma=_f('decoupled_consistent_sigma', 0.05),
            decoupled_inconsistent_sigma=_f('decoupled_inconsistent_sigma', 0.05),
            coupled_consistent_alpha=_f('coupled_consistent_alpha', 0.1),
            coupled_inconsistent_alpha=_f('coupled_inconsistent_alpha', 0.1),
            adc_bits=_f('adc_bits', 8.0),
            enable_adc=False,
            adc_backward_mode=synth_config.get('adc_backward_mode'),
            adv_direction_beta=beta_val,
            adv_direction_frozen=_b('adv_direction_frozen', True),
            adv_direction_random_sign=_b('adv_direction_random_sign', False),
            sign_corrupt_p=p_val,
            sign_corrupt_mode=synth_config.get('sign_corrupt_mode', 'flip'),
            sign_corrupt_noise_sigma=_f('sign_corrupt_noise_sigma', 1.0),
            saturation_gamma=_f('saturation_gamma', 5.0),
            saturation_alpha=_f('saturation_alpha', 1.0),
            drift_beta=_f('drift_beta', 0.3),
            drift_use_norm=_b('drift_use_norm', False),
            drift_frozen=_b('drift_frozen', True),
            sign_scale_alpha=_f('sign_scale_alpha', 0.5),
            rank_k=int(synth_config.get('rank_k', 4)),
            rank_fill_sigma=_f('rank_fill_sigma', 0.0),
            rank_resample=_b('rank_resample', False),
            clip_c=_f('clip_c', 1.0),
            clip_dither=_b('clip_dither', False),
            backward_corruption_at=synth_config.get('backward_corruption_at'),
            seed=base_seed,
        )

    if noise_type == 'adv_direction_beta':
        cfg_clean = make_config('adversarial_direction_bias', 0.0, 0.5)
        cfg_noisy = make_config('adversarial_direction_bias', theta, 0.5)
    elif noise_type == 'sign_corrupt_p':
        cfg_clean = make_config('sign_gradient_corruption', 0.0, 0.0)
        cfg_noisy = make_config('sign_gradient_corruption', 0.0, theta)
    else:
        raise ValueError(f"compute_delta_grad only supports adv_direction_beta and sign_corrupt_p, got {noise_type}")

    clean_model = get_model(config['model_name'], num_classes=config['num_classes'], **model_kwargs).to(device)
    clean_model.load_state_dict(base_model.state_dict())
    clean_model = wrap_model_with_synth_noise(clean_model, cfg_clean, noise_config=noise_injection)

    noisy_model = get_model(config['model_name'], num_classes=config['num_classes'], **model_kwargs).to(device)
    noisy_model.load_state_dict(base_model.state_dict())
    noisy_model = wrap_model_with_synth_noise(noisy_model, cfg_noisy, noise_config=noise_injection)

    total_diff_sq = 0.0
    total_clean_sq = 0.0
    n_samples = 0

    for batch in calibration_loader:
        if is_agnews:
            labels, texts, lengths = batch
            data = texts.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
        else:
            data, labels = batch
            data, labels = data.to(device), labels.to(device)
            lengths = None
        B = data.shape[0]
        n_samples += B

        def forward_loss_backward(model, d, lab, len_, logits_corrupt_cfg=None):
            """logits_corrupt_cfg: 若为 backward_corruption_at=logits，noisy 时传入 cfg 以在 logits 上施加扰动。"""
            model.train()
            model.zero_grad()
            if len_ is not None:
                out = model(d, lengths=len_, seed=seed)
            else:
                out = model(d, seed=seed)
            if logits_corrupt_cfg is not None:
                out = apply_logits_backward_corruption(out, logits_corrupt_cfg, seed=seed)
            loss = F.cross_entropy(out, lab)
            loss.backward()
            return _collect_grad_flattened(model, device)

        clean_model.base_model.load_state_dict(base_model.state_dict())
        g_clean = forward_loss_backward(clean_model, data, labels, lengths, logits_corrupt_cfg=None)

        noisy_model.base_model.load_state_dict(base_model.state_dict())
        # 当配置为 backward_corruption_at: logits 时，扰动只在 logits 处施加，标定也需在此处施加
        noisy_logits_cfg = cfg_noisy if at_logits else None
        g_noisy = forward_loss_backward(noisy_model, data, labels, lengths, logits_corrupt_cfg=noisy_logits_cfg)

        diff = g_noisy - g_clean
        total_diff_sq += diff.pow(2).sum().item()
        total_clean_sq += g_clean.pow(2).sum().item()

    if total_clean_sq < 1e-20:
        return 0.0
    return (total_diff_sq / total_clean_sq) ** 0.5


def compute_delta_grad_induced_by_forward(
    clean_model: nn.Module,
    noisy_model: nn.Module,
    base_model: nn.Module,
    calibration_loader: DataLoader,
    device: torch.device,
    is_agnews: bool = False,
    seed: Optional[int] = None,
) -> float:
    """
    计算由 forward 扰动（δ_logit）诱发的梯度扰动 δ_grad。
    
    即：仅在 forward 注入噪声时，clean forward+backward 与 noisy forward+backward 的梯度
    RMS 相对偏差，与 compute_delta_grad 同一定义，便于与 backward-only 扰动比较。
    
    δ_grad_induced = sqrt(E[||g_noisy - g_clean||^2] / E[||g_clean||^2])
    
    要求：clean_model 为无包装的 base，noisy_model 为同结构且 wrap 了 synth noise 的模型。
    """
    total_diff_sq = 0.0
    total_clean_sq = 0.0

    def _forward_loss_backward(model: nn.Module, d, lab, len_, from_base: bool, pass_seed: bool) -> torch.Tensor:
        model.train()
        model.zero_grad()
        if pass_seed and seed is not None:
            if len_ is not None:
                out = model(d, lengths=len_, seed=seed)
            else:
                out = model(d, seed=seed)
        else:
            if len_ is not None:
                out = model(d, lengths=len_)
            else:
                out = model(d)
        loss = F.cross_entropy(out, lab)
        loss.backward()
        if from_base:
            return _collect_grad_flattened_from_base(model, device)
        return _collect_grad_flattened(model, device)

    for batch in calibration_loader:
        if is_agnews:
            labels, texts, lengths = batch
            data = texts.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
        else:
            data, labels = batch
            data, labels = data.to(device), labels.to(device)
            lengths = None

        clean_model.load_state_dict(base_model.state_dict())
        g_clean = _forward_loss_backward(clean_model, data, labels, lengths, from_base=False, pass_seed=False)

        noisy_model.base_model.load_state_dict(base_model.state_dict())
        g_noisy = _forward_loss_backward(noisy_model, data, labels, lengths, from_base=True, pass_seed=True)

        diff = g_noisy - g_clean
        total_diff_sq += diff.pow(2).sum().item()
        total_clean_sq += g_clean.pow(2).sum().item()

    if total_clean_sq < 1e-20:
        return 0.0
    return (total_diff_sq / total_clean_sq) ** 0.5


def extract_block_features(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    model_name: str,
    is_agnews: bool = False,
    forward_seed_base: Optional[int] = None,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    提取模型的block级别特征和logits输出。
    
    Args:
        model: 模型（可能是wrapped的）
        dataloader: 数据加载器
        device: 设备
        model_name: 模型名称 ('resnet20', 'vit_tiny', or 'gru_agnews')
        is_agnews: 是否为AG News数据集（需要处理lengths）
        forward_seed_base: 若给出，每 batch 使用 seed=forward_seed_base+batch_idx，便于标定时对多组噪声取平均
        
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
        for batch_idx, batch in enumerate(dataloader):
            # 处理数据格式
            if is_agnews:
                labels, texts, lengths = batch
                data = texts.to(device)
                lengths = lengths.to(device)
            else:
                data, _ = batch
                data = data.to(device)
                lengths = None
            seed_arg = (forward_seed_base + batch_idx) if forward_seed_base is not None else 42

            # Forward pass (固定 seed 保证可复现；标定时可用 forward_seed_base 使每 batch 不同以平滑 δ)
            try:
                if hasattr(model, 'forward'):
                    forward_code = model.forward.__code__
                    forward_varnames = forward_code.co_varnames
                    if 'lengths' in forward_varnames and lengths is not None:
                        if 't' in forward_varnames:
                            output = model(data, lengths=lengths, t=0, seed=seed_arg)
                        elif 'seed' in forward_varnames:
                            output = model(data, lengths=lengths, seed=seed_arg)
                        else:
                            output = model(data, lengths=lengths)
                    elif 't' in forward_varnames:
                        output = model(data, t=0, seed=seed_arg)
                    elif 'seed' in forward_varnames:
                        output = model(data, seed=seed_arg)
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
) -> Tuple[float, Dict[int, float], Optional[float]]:
    """
    计算给定噪声参数θ下的扰动强度δ。
    
    Args:
        config: 配置字典
        base_model: 基线模型（干净模型）
        calibration_loader: 标定数据加载器
        device: 设备
        noise_type: 噪声类型。通用: variability_sigma, cond1_alpha, cond2_alpha, adc_bits。仅 synth_noise: heavy_tail_output_alpha, decoupled_*_sigma, coupled_*_alpha, saturation_*, drift_beta, sign_scale_alpha, rank_fill_sigma, rank_k(整数离散), clip_c。
        theta: 噪声参数值
        model_name: 模型名称
        
    Returns:
        (delta_logit, delta_blocks_dict, delta_grad_induced)。backward-only 或 memristor 路径下 delta_grad_induced 为 None；
        synth forward 路径下为「由 δ_logit 诱发的 δ_grad」。
    """
    # 标定阶段固定 RNG：同一 theta 得到同一 δ，避免 synth 路径因全局 RNG 导致 δ(θ) 波动
    base_seed = config.get('seed', 42)
    theta_seed = base_seed + (int(abs(theta) * 1e6) % (2**31 - 1000))
    torch.manual_seed(theta_seed)
    np.random.seed(theta_seed % (2**31))

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
    
    if 'synth_noise' not in config and 'memristor' not in config:
        raise ValueError("Config must contain either 'synth_noise' or 'memristor' for calibration.")
    use_synth_noise = 'synth_noise' in config
    is_agnews_calib = (config.get('dataset', '').lower() == 'agnews')
    if use_synth_noise and noise_type in ('adv_direction_beta', 'sign_corrupt_p'):
        delta_grad = compute_delta_grad(
            config, base_model, calibration_loader, device,
            noise_type, theta, model_name, vocab=vocab, is_agnews=is_agnews_calib,
            seed=theta_seed,
        )
        return (delta_grad, {}, None)

    if use_synth_noise:
        # ---------- 新合成噪声路径（不走权重-电导映射）----------
        logger.debug("Using synth_noise path (no weight-conductance mapping)")
        synth_config = config['synth_noise'].copy()
        # 从配置读取默认值，再按标定参数覆盖
        if noise_type == 'variability_sigma':
            synth_config['noise_type'] = 'iid_multiplicative'
            synth_config['variability_sigma'] = theta
            synth_config['enable_adc'] = False
        elif noise_type == 'cond1_alpha':
            synth_config['noise_type'] = 'heavy_tail'
            synth_config['heavy_tail_alpha'] = theta
            synth_config['heavy_tail_nu'] = float(synth_config.get('heavy_tail_nu', synth_config.get('cond1_nu', 2.0)))
            synth_config['enable_adc'] = False
        elif noise_type == 'cond2_alpha':
            synth_config['noise_type'] = 'input_dependent'
            synth_config['input_dependent_alpha'] = theta
            synth_config['enable_adc'] = False
        elif noise_type == 'adc_bits':
            synth_config['noise_type'] = 'gradient_degenerate'
            synth_config['adc_bits'] = max(2.0, float(int(round(theta))))
            synth_config['enable_adc'] = True
        elif noise_type == 'heavy_tail_output_alpha':
            synth_config['noise_type'] = 'heavy_tail_output'
            synth_config['heavy_tail_alpha'] = theta
            synth_config['heavy_tail_nu'] = float(synth_config.get('heavy_tail_nu', synth_config.get('cond1_nu', 2.0)))
            synth_config['enable_adc'] = False
        elif noise_type == 'decoupled_consistent_sigma':
            synth_config['noise_type'] = 'decoupled_consistent'
            synth_config['decoupled_consistent_sigma'] = theta
            synth_config['enable_adc'] = False
        elif noise_type == 'decoupled_inconsistent_sigma':
            synth_config['noise_type'] = 'decoupled_inconsistent'
            synth_config['decoupled_inconsistent_sigma'] = theta
            synth_config['enable_adc'] = False
        elif noise_type == 'coupled_consistent_alpha':
            synth_config['noise_type'] = 'coupled_consistent'
            synth_config['coupled_consistent_alpha'] = theta
            synth_config['enable_adc'] = False
        elif noise_type == 'coupled_inconsistent_alpha':
            synth_config['noise_type'] = 'coupled_inconsistent'
            synth_config['coupled_inconsistent_alpha'] = theta
            synth_config['enable_adc'] = False
        elif noise_type == 'saturation_gamma':
            synth_config['noise_type'] = 'saturation_collapse'
            synth_config['saturation_gamma'] = theta
            synth_config['saturation_alpha'] = float(synth_config.get('saturation_alpha', 1.0))
            synth_config['enable_adc'] = False
        elif noise_type == 'saturation_alpha':
            synth_config['noise_type'] = 'saturation_collapse'
            synth_config['saturation_alpha'] = theta
            synth_config['saturation_gamma'] = float(synth_config.get('saturation_gamma', 5.0))
            synth_config['enable_adc'] = False
        elif noise_type == 'drift_beta':
            synth_config['noise_type'] = 'frozen_additive_drift'
            synth_config['drift_beta'] = theta
            synth_config['drift_use_norm'] = bool(synth_config.get('drift_use_norm', False))
            synth_config['drift_frozen'] = True
            synth_config['enable_adc'] = False
        elif noise_type == 'sign_scale_alpha':
            synth_config['noise_type'] = 'sign_coupled_scaling'
            synth_config['sign_scale_alpha'] = theta
            synth_config['enable_adc'] = False
        elif noise_type == 'rank_fill_sigma':
            synth_config['noise_type'] = 'rank_collapse'
            synth_config['rank_fill_sigma'] = theta
            synth_config['rank_k'] = int(synth_config.get('rank_k', 4))
            synth_config['enable_adc'] = False
        elif noise_type == 'rank_k':
            synth_config['noise_type'] = 'rank_collapse'
            synth_config['rank_k'] = max(1, int(round(theta)))
            synth_config['rank_fill_sigma'] = float(synth_config.get('rank_fill_sigma', 0.0))
            synth_config['enable_adc'] = False
        elif noise_type == 'clip_c':
            synth_config['noise_type'] = 'deterministic_clip'
            synth_config['clip_c'] = theta
            synth_config['clip_dither'] = bool(synth_config.get('clip_dither', False))
            synth_config['enable_adc'] = False
        else:
            raise ValueError(f"Unknown noise_type: {noise_type}")
        
        def _f(key, default):
            v = synth_config.get(key)
            return float(v) if v is not None else default

        def _b(key, default):
            v = synth_config.get(key)
            return bool(v) if v is not None else default

        var_sigma = _f('variability_sigma', 0.05)
        in_dep_alpha = _f('input_dependent_alpha', _f('cond2_alpha', 0.1))
        synth_noise_config = SynthNoiseConfig(
            noise_type=synth_config['noise_type'],
            variability_sigma=var_sigma,
            heavy_tail_alpha=_f('heavy_tail_alpha', _f('cond1_alpha', 0.1)),
            heavy_tail_nu=_f('heavy_tail_nu', _f('cond1_nu', 2.0)),
            input_dependent_alpha=in_dep_alpha,
            decoupled_consistent_sigma=_f('decoupled_consistent_sigma', var_sigma),
            decoupled_inconsistent_sigma=_f('decoupled_inconsistent_sigma', var_sigma),
            coupled_consistent_alpha=_f('coupled_consistent_alpha', in_dep_alpha),
            coupled_inconsistent_alpha=_f('coupled_inconsistent_alpha', in_dep_alpha),
            adc_bits=_f('adc_bits', 8.0),
            enable_adc=_b('enable_adc', False),
            adc_backward_mode=synth_config.get('adc_backward_mode'),
            adv_direction_beta=_f('adv_direction_beta', 1.0),
            adv_direction_frozen=_b('adv_direction_frozen', True),
            adv_direction_random_sign=_b('adv_direction_random_sign', False),
            sign_corrupt_p=_f('sign_corrupt_p', 0.5),
            sign_corrupt_mode=synth_config.get('sign_corrupt_mode', 'flip'),
            sign_corrupt_noise_sigma=_f('sign_corrupt_noise_sigma', 1.0),
            saturation_gamma=_f('saturation_gamma', 5.0),
            saturation_alpha=_f('saturation_alpha', 1.0),
            drift_beta=_f('drift_beta', 0.3),
            drift_use_norm=_b('drift_use_norm', False),
            drift_frozen=_b('drift_frozen', True),
            sign_scale_alpha=_f('sign_scale_alpha', 0.5),
            rank_k=int(synth_config.get('rank_k', 4)),
            rank_fill_sigma=_f('rank_fill_sigma', 0.0),
            rank_resample=_b('rank_resample', False),
            clip_c=_f('clip_c', 1.0),
            clip_dither=_b('clip_dither', False),
            seed=config.get('seed', 42),
        )
        noise_injection = synth_config.get('noise_injection')
        
        noisy_model = get_model(
            config['model_name'],
            num_classes=config['num_classes'],
            **model_kwargs
        ).to(device)
        noisy_model.load_state_dict(base_model.state_dict())
        noisy_model = wrap_model_with_synth_noise(
            noisy_model,
            synth_noise_config,
            noise_config=noise_injection,
        )
        noisy_model.eval()
        # 由 forward 扰动诱发的 δ_grad（与 backward-only 的 δ_grad 同一定义，便于比较）
        delta_grad_induced = compute_delta_grad_induced_by_forward(
            clean_model, noisy_model, base_model, calibration_loader, device,
            is_agnews=is_agnews_calib, seed=theta_seed,
        )
        # 恢复 RNG，使后续 extract_block_features 的噪声与「未算 δ_grad_induced 时」一致，保证 δ_logit 标定不变
        torch.manual_seed(theta_seed)
        np.random.seed(theta_seed % (2**31))
    else:
        delta_grad_induced = None
        # ---------- 原 memristor 路径（权重-电导映射）----------
        memristor_config = config['memristor'].copy()
        
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
            memristor_config['adc_bits'] = max(2, int(round(theta)))
            memristor_config['enable_adc'] = True
            memristor_config['enable_adc_during_training'] = True
            memristor_config['adc_training_mode'] = 'direct'
            memristor_config['read_noise_sigma'] = 0.0
            memristor_config['drift_alpha'] = 0.0
            memristor_config['stuck_ratio'] = 0.0
            memristor_config['ir_drop_mode'] = 'none'
        elif noise_type in (
            'heavy_tail_output_alpha', 'decoupled_consistent_sigma', 'decoupled_inconsistent_sigma',
            'coupled_consistent_alpha', 'coupled_inconsistent_alpha',
            'saturation_gamma', 'saturation_alpha', 'drift_beta', 'sign_scale_alpha', 'rank_fill_sigma', 'rank_k', 'clip_c',
            'adv_direction_beta', 'sign_corrupt_p',
        ):
            raise ValueError(
                f"noise_type '{noise_type}' is only supported for configs with 'synth_noise'. "
                "Use a synth config (e.g. *synth_comp.yaml) for calibration."
            )
        else:
            raise ValueError(f"Unknown noise_type: {noise_type}")
        
        excluded_keys = {'noise_injection', 'write'}
        float_params = {
            'G_min', 'G_max', 'variability_sigma', 'read_noise_sigma', 'drift_alpha',
            'stuck_ratio', 'stuck_low_prob', 'ir_drop_beta', 'ir_drop_gamma',
            'ir_drop_scaling', 'ir_drop_eta', 'ir_drop_cap', 'cond1_alpha', 'cond1_nu',
            'cond2_alpha'
        }
        int_params = {'array_size', 'adc_bits', 'drift_time_fixed'}
        tuple_params = {'weight_clip'}
        bool_params = {
            'enable_update_model', 'enable_adc', 'adc_add_noise', 'enable_energy',
            'ir_drop_train_enabled', 'enable_adc_during_training',
            'enable_ir_drop_paper_during_training'
        }
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
                device_model_kwargs[k] = v
        
        device_model = MemristorDeviceModel(**device_model_kwargs)
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
    # 标定时对 noisy 模型每 batch 用不同 seed，对多组噪声取平均，δ(θ) 更平滑、标定更稳（尤其 synth）
    forward_seed_base = base_seed if ('synth_noise' in config or 'memristor' in config) else None

    # 提取特征
    clean_features, clean_logits = extract_block_features(
        clean_model, calibration_loader, device, model_name, is_agnews=is_agnews
    )
    noisy_features, noisy_logits = extract_block_features(
        noisy_model, calibration_loader, device, model_name, is_agnews=is_agnews,
        forward_seed_base=forward_seed_base,
    )
    
    # 计算δ
    delta_logit = compute_delta_logit(clean_logits, noisy_logits)
    delta_blocks = compute_delta_block(clean_features, noisy_features)
    
    return delta_logit, delta_blocks, delta_grad_induced


def run_diagnose(
    config: Dict[str, Any],
    base_model: nn.Module,
    calibration_loader: DataLoader,
    device: torch.device,
    noise_type: str,
    model_name: str,
    vocab: Optional[Any] = None,
    thetas: Optional[List[float]] = None,
) -> None:
    """
    诊断模式：对若干 theta 采样计算 δ(θ)，并检查 wrapped 模型中实际参与噪声的层数。
    用于排查「expand 到很大 theta 仍标定不出」是否因为噪声未生效或 δ 不随 θ 变化。
    """
    if thetas is None:
        if noise_type == 'rank_k':
            thetas = [1, 2, 4, 8, 16, 32]
        elif noise_type == 'clip_c':
            # clip_c: 小 c 裁剪强 δ 大，大 c 裁剪弱 δ 小；ViT 需更大 c 才能达到相同 δ
            thetas = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0] if model_name == 'vit_tiny' else [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        elif noise_type == 'adv_direction_beta':
            thetas = [0.0, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0]
        elif noise_type == 'sign_corrupt_p':
            thetas = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]  # p 可到 1；仅部分层加噪时 δ_grad < 2*sqrt(p)
        elif noise_type in ('coupled_consistent_alpha', 'coupled_inconsistent_alpha'):
            # ViT 可标定 α 上界更大，诊断时多采几个大 α 以观察 δ(θ) 上界
            if model_name == 'vit_tiny':
                thetas = [0.01, 0.1, 0.3, 0.5, 0.64, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0]
            else:
                thetas = [0.01, 0.1, 0.3, 0.5, 0.64, 1.0, 2.0, 4.0]  # 含 0.64 便于核对此前标定
        else:
            thetas = [0.0, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0]
    logger.info("=== Diagnostic: δ(θ) for sample thetas ===")
    for theta in thetas:
        try:
            delta, _, _ = compute_delta_for_params(
                config, base_model, calibration_loader, device,
                noise_type, theta, model_name, vocab=vocab
            )
            logger.info(f"  θ={theta:.4f}  ->  δ_logit={delta:.6f}")
        except Exception as e:
            logger.warning(f"  θ={theta:.4f}  ->  Error: {e}")
    # 统计 wrapped 模型里 SynthNoise 层及 enable_noise 数量（用 theta=0.1 建一次）
    if 'synth_noise' not in config:
        return
    synth_config = config['synth_noise'].copy()
    if noise_type == 'heavy_tail_output_alpha':
        synth_config['noise_type'] = 'heavy_tail_output'
        synth_config['heavy_tail_alpha'] = 0.1
    elif noise_type == 'variability_sigma':
        synth_config['noise_type'] = 'iid_multiplicative'
        synth_config['variability_sigma'] = 0.1
    elif noise_type == 'clip_c':
        synth_config['noise_type'] = 'deterministic_clip'
        synth_config['clip_c'] = 0.5  # 用于统计层数
    elif noise_type == 'adv_direction_beta':
        synth_config['noise_type'] = 'adversarial_direction_bias'
        synth_config['adv_direction_beta'] = 0.5
    elif noise_type == 'sign_corrupt_p':
        synth_config['noise_type'] = 'sign_gradient_corruption'
        synth_config['sign_corrupt_p'] = 0.3
    elif noise_type == 'coupled_inconsistent_alpha':
        synth_config['noise_type'] = 'coupled_inconsistent'
        synth_config['coupled_inconsistent_alpha'] = 0.5
    elif noise_type == 'coupled_consistent_alpha':
        synth_config['noise_type'] = 'coupled_consistent'
        synth_config['coupled_consistent_alpha'] = 0.5
    else:
        return
    model_kwargs = {}
    if model_name == 'vit_tiny':
        model_kwargs = {k: v for k, v in config.items() if k in ['patch_size', 'embed_dim', 'depth', 'num_heads', 'mlp_ratio', 'qkv_bias']}
    elif model_name == 'resnet20':
        model_kwargs = {k: v for k, v in config.items() if k in ['in_channels']}
    elif model_name == 'gru_agnews' and vocab is not None:
        model_kwargs = {'vocab_size': len(vocab), 'embed_dim': config.get('embed_dim', 128), 'hidden_dim': config.get('hidden_dim', 256), 'num_layers': config.get('num_layers', 2)}
    try:
        from ..memristor.synth_noise_wrappers import SynthNoiseLinear, SynthNoiseConv2d
    except ImportError:
        from src.memristor.synth_noise_wrappers import SynthNoiseLinear, SynthNoiseConv2d
    def _f(key, default):
        v = synth_config.get(key)
        return float(v) if v is not None else default
    def _b(key, default):
        v = synth_config.get(key)
        return bool(v) if v is not None else default
    cfg = SynthNoiseConfig(
        noise_type=synth_config.get('noise_type', 'heavy_tail_output'),
        variability_sigma=_f('variability_sigma', 0.05),
        heavy_tail_alpha=_f('heavy_tail_alpha', 0.1),
        heavy_tail_nu=_f('heavy_tail_nu', 2.0),
        input_dependent_alpha=_f('input_dependent_alpha', 0.1),
        decoupled_consistent_sigma=_f('decoupled_consistent_sigma', 0.05),
        decoupled_inconsistent_sigma=_f('decoupled_inconsistent_sigma', 0.05),
        coupled_consistent_alpha=_f('coupled_consistent_alpha', 0.1),
        coupled_inconsistent_alpha=_f('coupled_inconsistent_alpha', 0.1),
        adc_bits=_f('adc_bits', 8.0),
        enable_adc=_b('enable_adc', False),
        adv_direction_beta=_f('adv_direction_beta', 1.0),
        sign_corrupt_p=_f('sign_corrupt_p', 0.5),
        saturation_gamma=_f('saturation_gamma', 5.0),
        saturation_alpha=_f('saturation_alpha', 1.0),
        drift_beta=_f('drift_beta', 0.3),
        drift_use_norm=_b('drift_use_norm', False),
        drift_frozen=_b('drift_frozen', True),
        sign_scale_alpha=_f('sign_scale_alpha', 0.5),
        rank_k=int(synth_config.get('rank_k', 4)),
        rank_fill_sigma=_f('rank_fill_sigma', 0.0),
        rank_resample=_b('rank_resample', False),
        clip_c=_f('clip_c', 1.0),
        clip_dither=_b('clip_dither', False),
        seed=config.get('seed', 42),
    )
    dummy = get_model(config['model_name'], num_classes=config['num_classes'], **model_kwargs).to(device)
    dummy.load_state_dict(base_model.state_dict())
    wrapped = wrap_model_with_synth_noise(dummy, cfg, noise_config=synth_config.get('noise_injection'))
    n_total = 0
    n_enabled = 0
    for m in wrapped.base_model.modules():
        if isinstance(m, SynthNoiseLinear) or isinstance(m, SynthNoiseConv2d):
            n_total += 1
            if getattr(m, 'enable_noise', True):
                n_enabled += 1
    logger.info(f"=== Wrapped model: {n_enabled}/{n_total} SynthNoise layers have enable_noise=True (noise_injection from config) ===")
    if n_enabled == 0:
        logger.warning("No layers have noise enabled! Check config 'synth_noise.noise_injection' keys (ResNet: stem, layer1, layer2, layer3, head).")


def _check_monotonicity_and_log(
    config: Dict[str, Any],
    base_model: nn.Module,
    calibration_loader: DataLoader,
    device: torch.device,
    noise_type: str,
    theta_star: float,
    target_delta: float,
    delta_final: float,
    model_name: str,
    vocab: Optional[Any],
    theta_min: float,
    theta_max: float,
    target_delta_type: str = 'logit',
) -> None:
    """
    标定后检查 δ_logit(θ) 单调性并打日志。对 clip_c 等「θ 越小 δ 越大」的类型，在 θ* 两侧各算一点 δ_logit，
    若顺序反了则告警。并提示用 --eval_theta 复验时需同一 config/checkpoint；复验时期望的是 target_delta_type 对应的 δ ≈ target。
    """
    # clip_c: 小 c 裁剪更狠 → δ 大；variability_sigma 等: 大 σ → δ 大。这里只对 clip_c 做「小 θ → 大 δ」检查
    if noise_type not in ('clip_c', 'saturation_gamma', 'saturation_alpha', 'drift_beta'):
        return
    if noise_type in ('adv_direction_beta', 'sign_corrupt_p'):
        return
    theta_lo = max(theta_min, min(theta_star * 0.5, theta_star - 1e-6))
    theta_hi = min(theta_max, max(theta_star * 1.5, theta_star + 1e-6))
    if theta_lo >= theta_star or theta_hi <= theta_star:
        return
    try:
        d_lo, _, _ = compute_delta_for_params(
            config, base_model, calibration_loader, device,
            noise_type, theta_lo, model_name, vocab=vocab,
        )
        d_hi, _, _ = compute_delta_for_params(
            config, base_model, calibration_loader, device,
            noise_type, theta_hi, model_name, vocab=vocab,
        )
    except Exception:
        return
    # clip_c / saturation / drift: 小 θ → 大 δ，故应 d_lo > delta_final > d_hi
    if d_lo <= delta_final or d_hi >= delta_final:
        logger.warning(
            f"Monotonicity check: δ(θ*={theta_star:.4f})={delta_final:.4f}, but δ(θ_lo={theta_lo:.4f})={d_lo:.4f}, δ(θ_hi={theta_hi:.4f})={d_hi:.4f}. "
            f"Expected δ(θ_lo) > δ(θ*) > δ(θ_hi) for {noise_type}. "
            "If you later run --eval_theta with the same θ* and get a different δ, use the same --config and --checkpoint."
        )
    expect_which = f"δ_{target_delta_type}"  # logit | grad | grad_induced
    logger.info(
        f"To verify: --eval_theta {theta_star:.6f} with same --config and --checkpoint (expected {expect_which} ≈ {target_delta})."
    )


def _objective_delta(delta_logit: float, delta_grad_induced: Optional[float], target_delta_type: str) -> float:
    """根据 target_delta_type 从 (delta_logit, delta_grad_induced) 取出用于标定目标的 δ。"""
    if target_delta_type == 'logit':
        return delta_logit
    # grad / grad_induced: forward 用 δ_grad_induced，backward 只有 δ_grad(=delta_logit)
    if delta_grad_induced is not None:
        return delta_grad_induced
    return delta_logit


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
    target_delta_type: str = 'logit',
) -> Tuple[float, Dict[str, Any]]:
    """
    使用二分搜索找到θ*使得 指定类型的δ(θ*) = δ*。
    
    Args:
        target_delta_type: 'logit' 则匹配 δ_logit；'grad'/'grad_induced' 则匹配 δ_grad（backward）或 δ_grad_induced（forward）。
        其余同前。
    """
    theta_min, theta_max = theta_bounds

    def delta_func(theta):
        """计算给定θ下的目标 δ 与 target 的差值"""
        try:
            d_logit, _, d_grad_i = compute_delta_for_params(
                config, base_model, calibration_loader, device,
                noise_type, theta, model_name, vocab=vocab
            )
            obj = _objective_delta(d_logit, d_grad_i, target_delta_type)
            return obj - target_delta
        except Exception as e:
            logger.warning(f"Error computing delta for theta={theta}: {e}")
            return float('inf')

    # rank_k 为整数，用离散搜索：在 [theta_min, theta_max] 内遍历 k，选 |δ(k)−target| 最小的 k
    if noise_type == 'rank_k':
        k_lo = max(1, int(theta_min))
        k_hi = max(k_lo, int(theta_max))
        best_k = k_lo
        best_err = float('inf')
        best_delta = 0.0
        best_blocks = {}
        best_delta_grad_induced = None
        for k in range(k_lo, k_hi + 1):
            try:
                delta_val, delta_blocks_val, dgi = compute_delta_for_params(
                    config, base_model, calibration_loader, device,
                    noise_type, float(k), model_name, vocab=vocab
                )
                obj = _objective_delta(delta_val, dgi, target_delta_type)
                err = abs(obj - target_delta)
                if err < best_err:
                    best_err = err
                    best_k = k
                    best_delta = delta_val
                    best_blocks = delta_blocks_val
                    best_delta_grad_induced = dgi
            except Exception as e:
                logger.warning(f"rank_k={k}: {e}")
        theta_star = float(best_k)
        result = {
            'theta_star': theta_star,
            'delta_logit': best_delta,
            'delta_blocks': best_blocks,
            'target_delta': target_delta,
            'error': best_err,
        }
        if best_delta_grad_induced is not None:
            result['delta_grad_induced'] = best_delta_grad_induced
        logger.info(f"rank_k discrete search: k in [{k_lo}, {k_hi}], best rank_k={best_k}, δ_logit={best_delta:.6f}, |δ−target|={best_err:.6f}")
        # rank_collapse 下 δ(k) 随 k 单调减：k=1 时 δ 最大。若最佳 k 落在边界且 δ 仍远小于 target，说明目标 δ 不可达
        if best_k == k_lo and best_delta < target_delta and best_err > 0.05:
            logger.warning(
                f"rank_k 标定: 最佳 k={best_k} 落在下边界，但达到的 δ_logit={best_delta:.4f} < target={target_delta}. "
                "rank_collapse 下最大可达 δ 即为 k=1 时的 δ，本模型/数据上无法达到 target，请降低 target_delta 或检查标定集。"
            )
        return theta_star, result

    # 检查区间是否括住根（f 在两端异号）；synth 路径下 θ–δ 尺度常与 memristor 不同，未括住时尝试自动扩大范围
    use_synth_noise = 'synth_noise' in config
    f_lo, f_hi = delta_func(theta_min), delta_func(theta_max)
    expand_count = 0
    max_expand = 2 if use_synth_noise else 0
    while f_lo * f_hi > 0 and expand_count < max_expand:
        if use_synth_noise:
            if f_lo > 0 and f_hi > 0:
                # δ 在区间内都 > target，根在更小 θ：向左扩
                theta_max, theta_min = theta_min, max(1e-8, theta_min / 4.0)
                logger.warning(
                    f"Synth: δ > target in current bracket. Expanding left: theta_min={theta_min:.6f}, theta_max={theta_max:.6f}"
                )
            else:
                # δ 在区间内都 < target，根在更大 θ：向右扩
                new_max = theta_max * 4.0
                if noise_type == 'sign_corrupt_p':
                    new_max = min(new_max, 1.0)  # p 为概率，不得超过 1
                    if new_max <= theta_max:
                        logger.warning(
                            "sign_corrupt_p: 已到上界 p=1，δ_grad 仍 < target。"
                            "若 noise_injection 仅部分层加噪，δ 会被稀释，请降低 --target_delta（如 0.3）或对全层/logits 加噪。"
                        )
                        break
                if noise_type in ('coupled_consistent_alpha', 'coupled_inconsistent_alpha'):
                    logger.warning(
                        "coupled/input_dependent 类扰动的 δ 依赖激活尺度：未加载 checkpoint 或与上次标定用的模型/数据不同时，"
                        "需更大 α 才能达到同一 target。建议用与实验相同的 --checkpoint 再标定，或手动加大 --theta_max。"
                    )
                theta_min, theta_max = theta_max, new_max
                logger.warning(
                    f"Synth: δ < target in current bracket. Expanding right: theta_min={theta_min:.6f}, theta_max={theta_max:.6f}"
                )
            f_lo, f_hi = delta_func(theta_min), delta_func(theta_max)
            expand_count += 1
        else:
            break
    if f_lo * f_hi > 0:
        logger.warning(
            f"f(θ_min) and f(θ_max) have the same sign (f(θ_min)={f_lo:.6f}, f(θ_max)={f_hi:.6f}). "
            "Root may be outside [theta_min, theta_max]. Consider wider --theta_min / --theta_max."
        )

    # 使用 scipy 的 brentq 先快速逼近（xtol 是根 θ 的容差，不是残差 |δ−target| 的容差）
    try:
        theta_star = brentq(delta_func, theta_min, theta_max, xtol=tol, maxiter=max_iter)
    except ValueError as e:
        logger.error(f"Brentq failed: {e}. Trying manual binary search...")
        theta_star = manual_binary_search(
            delta_func, theta_min, theta_max, tol, max_iter
        )

    # 计算当前残差；brentq 按 θ 的精度停止，残差可能仍很大（尤其 δ(θ) 较平时），故按残差再收紧
    delta_final, delta_blocks_final, delta_grad_induced_final = compute_delta_for_params(
        config, base_model, calibration_loader, device,
        noise_type, theta_star, model_name, vocab=vocab
    )
    objective_final = _objective_delta(delta_final, delta_grad_induced_final, target_delta_type)
    error = abs(objective_final - target_delta)
    if error >= tol:
        logger.warning(
            f"Brentq stopped with residual error={error:.6f} >= tol={tol}. "
            "Refining with residual-based binary search (|δ−target|<tol)."
        )
        # refinement 用更多迭代，避免 δ(θ) 较平时未收敛就停
        max_iter_refine = max(200, max_iter * 3)
        theta_star = manual_binary_search(
            delta_func, theta_min, theta_max, tol, max_iter_refine
        )
        delta_final, delta_blocks_final, delta_grad_induced_final = compute_delta_for_params(
            config, base_model, calibration_loader, device,
            noise_type, theta_star, model_name, vocab=vocab
        )
        objective_final = _objective_delta(delta_final, delta_grad_induced_final, target_delta_type)
        error = abs(objective_final - target_delta)
        if error >= tol:
            logger.warning(
                f"After refinement (max_iter={max_iter_refine}), residual still {error:.6f} >= tol={tol}. "
                "Possible causes: (1) root not in [theta_min, theta_max] — try wider bounds; "
                "(2) need more iterations — use --max_iter; (3) delta is noisy (small calibration set)."
            )

    # 自洽性检查：对 clip_c 等 δ_logit(θ) 单调的噪声，在 θ* 两侧各取一点验证单调性
    _check_monotonicity_and_log(
        config, base_model, calibration_loader, device,
        noise_type, theta_star, target_delta, delta_final,
        model_name, vocab, theta_min, theta_max,
        target_delta_type=target_delta_type,
    )

    result = {
        'theta_star': theta_star,
        'delta_logit': delta_final,
        'delta_blocks': delta_blocks_final,
        'target_delta': target_delta,
        'error': error,
    }
    if delta_grad_induced_final is not None:
        result['delta_grad_induced'] = delta_grad_induced_final
    return theta_star, result


def manual_binary_search(
    func,
    x_min: float,
    x_max: float,
    tol: float,
    max_iter: int,
) -> float:
    """手动二分搜索，按残差 |f(x)| < tol 停止；若用尽迭代则返回已见残差最小的 x。"""
    best_x, best_abs_f = (x_min + x_max) / 2, float('inf')
    for i in range(max_iter):
        x_mid = (x_min + x_max) / 2
        f_mid = func(x_mid)
        abs_f = abs(f_mid)
        if abs_f < best_abs_f:
            best_x, best_abs_f = x_mid, abs_f
        if abs_f < tol:
            return x_mid
        f_min = func(x_min)
        if f_min * f_mid < 0:
            x_max = x_mid
        else:
            x_min = x_mid
    return best_x


def main():
    parser = argparse.ArgumentParser(description='Calibrate noise strength')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--checkpoint', type=str, help='Model checkpoint path (optional)')
    parser.add_argument('--target_delta', type=float, default=None, help='Target perturbation strength δ* (not needed with --eval_theta)')
    parser.add_argument('--target_delta_type', type=str, default='logit', choices=['logit', 'grad', 'grad_induced'],
                        help='Which δ to match: logit (δ_logit, default) | grad (δ_grad, backward 或 forward 的 δ_grad_induced) | grad_induced (forward 的 δ_grad_induced)')
    parser.add_argument('--eval_theta', type=float, default=None, help='Only evaluate δ at this θ (no search). Use to see actual δ for a given strength (e.g. when calibration hits ceiling).')
    parser.add_argument('--noise_type', type=str, required=True,
                        choices=[
                            'variability_sigma', 'cond1_alpha', 'cond2_alpha', 'adc_bits',
                            'heavy_tail_output_alpha', 'decoupled_consistent_sigma', 'decoupled_inconsistent_sigma',
                            'coupled_consistent_alpha', 'coupled_inconsistent_alpha',
                            'saturation_gamma', 'saturation_alpha', 'drift_beta', 'sign_scale_alpha',
                            'rank_fill_sigma', 'rank_k', 'clip_c',
                            'adv_direction_beta', 'sign_corrupt_p',
                        ],
                        help='Noise type (forward: clip_c, drift_beta, ...; backward-only: adv_direction_beta, sign_corrupt_p → δ_grad)')
    parser.add_argument('--theta_min', type=float, help='Minimum theta value for search')
    parser.add_argument('--theta_max', type=float, help='Maximum theta value for search')
    parser.add_argument('--calibration_size', type=int, default=512, help='Calibration dataset size')
    parser.add_argument('--max_iter', type=int, default=50, help='Max iterations for root search (refinement uses max(200, 3*max_iter))')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file path (optional when using --diagnose)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--diagnose', action='store_true', help='Only run diagnostic: sample δ(θ) for several thetas and count layers with noise enabled, then exit')
    
    args = parser.parse_args()
    if not args.diagnose and args.eval_theta is None and args.output is None:
        parser.error("--output is required when not using --diagnose or --eval_theta")
    if args.eval_theta is not None and args.target_delta is not None:
        logger.warning("--eval_theta is set; ignoring --target_delta (no calibration search).")
    if not args.diagnose and args.eval_theta is None and args.target_delta is None:
        parser.error("--target_delta is required when not using --diagnose or --eval_theta")
    
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
    
    if args.diagnose:
        run_diagnose(
            config, base_model, calibration_loader, device,
            args.noise_type, model_name, vocab=vocab,
        )
        logger.info("Diagnostic done. Exiting.")
        return

    # 只评估：给定 θ 算一次 δ（不搜索），用于查看「偏大」强度对应的实际输出/梯度扰动
    if args.eval_theta is not None:
        if not args.checkpoint:
            logger.warning(
                "Running --eval_theta without --checkpoint (random init). "
                "δ depends on the model: with trained weights, activations spread out and clip_c has real effect; "
                "with random init, activations are often small so clipping barely changes output → δ can be ~0.01. "
                "To match calibration (e.g. verify θ that gave δ=1), pass the same --checkpoint as when you calibrated."
            )
        theta_val = args.eval_theta
        if args.noise_type == 'rank_k':
            theta_val = float(max(1, int(round(args.eval_theta))))
        delta_logit, delta_blocks, delta_grad_induced = compute_delta_for_params(
            config, base_model, calibration_loader, device,
            args.noise_type, theta_val, model_name, vocab=vocab,
        )
        is_backward = args.noise_type in ('adv_direction_beta', 'sign_corrupt_p')
        delta_name = "δ_grad" if is_backward else "δ_logit"
        logger.info(f"Eval θ = {theta_val} ({args.noise_type})")
        logger.info(f"  {delta_name} = {delta_logit:.6f}")
        if delta_grad_induced is not None:
            logger.info(f"  δ_grad_induced (由 δ_logit 诱发) = {delta_grad_induced:.6f}")
        if delta_blocks:
            for k, v in sorted(delta_blocks.items()):
                logger.info(f"  δ_block[{k}] = {v:.6f}")
        out_data = {
            'config_path': args.config,
            'noise_type': args.noise_type,
            'theta': theta_val,
            'delta_logit': delta_logit,
            'delta_blocks': delta_blocks,
            'calibration_size': args.calibration_size,
            'seed': args.seed,
        }
        if delta_grad_induced is not None:
            out_data['delta_grad_induced'] = delta_grad_induced
        if is_backward:
            out_data['delta_grad'] = delta_logit
        if args.output:
            os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
            with open(args.output, 'w') as f:
                json.dump(out_data, f, indent=2)
            logger.info(f"Results saved to {args.output}")
        return

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
        elif args.noise_type == 'heavy_tail_output_alpha':
            theta_min, theta_max = 0.001, 1.0
        elif args.noise_type in ('decoupled_consistent_sigma', 'decoupled_inconsistent_sigma'):
            theta_min, theta_max = 0.001, 0.5
        elif args.noise_type in ('coupled_consistent_alpha', 'coupled_inconsistent_alpha'):
            # δ 依赖激活尺度 v^T z。ViT 因 LayerNorm 等使中间激活尺度偏小，同一 α 下 δ 更小，
            # 可标定的 δ 上界低于 ResNet；对 vit_tiny 放宽 α 上界以扩大可标定 δ 范围
            if model_name == 'vit_tiny':
                theta_min, theta_max = 0.001, 15.0
            else:
                theta_min, theta_max = 0.001, 4.0
        elif args.noise_type == 'saturation_gamma':
            theta_min, theta_max = 0.5, 20.0
        elif args.noise_type == 'saturation_alpha':
            theta_min, theta_max = 0.01, 2.0
        elif args.noise_type == 'drift_beta':
            # ViT: 同强度 β 下 δ 更小（logit 范数更大 + 多层累积易饱和），可标定 δ 上界约 ~1；放宽 θ 上界便于尝试高 target_delta
            if model_name == 'vit_tiny':
                theta_min, theta_max = 0.01, 8.0
            else:
                theta_min, theta_max = 0.01, 1.0
        elif args.noise_type == 'sign_scale_alpha':
            if model_name == 'vit_tiny':
                theta_min, theta_max = 0.01, 6.0
            else:
                theta_min, theta_max = 0.01, 1.0
        elif args.noise_type == 'rank_fill_sigma':
            if model_name == 'vit_tiny':
                theta_min, theta_max = 0.01, 4.0
            else:
                theta_min, theta_max = 0.01, 1.0
        elif args.noise_type == 'rank_k':
            # 离散搜索：k 为整数，默认 [1, 64]，用 --theta_min/--theta_max 指定范围
            theta_min, theta_max = 1.0, 64.0
        elif args.noise_type == 'clip_c':
            # ViT 因 LayerNorm + 更多线性层，同一 clip_c 下累积裁剪更强，要达相同 δ 需更大 c；
            # 默认 [0.1, 10] 对 ResNet 够用，ViT 常需 c>10，故对 vit_tiny 放宽上界
            if model_name == 'vit_tiny':
                theta_min, theta_max = 0.1, 80.0
            else:
                theta_min, theta_max = 0.1, 10.0
        elif args.noise_type == 'adv_direction_beta':
            theta_min, theta_max = 0.01, 3.0   # backward: g += β‖g‖d
        elif args.noise_type == 'sign_corrupt_p':
            # p∈[0,1]。若 noise_injection 只对部分层加噪，δ_grad 会被稀释，需更大 p 才能达到同一 target，故默认上界 1.0
            theta_min, theta_max = 0.01, 1.0
        else:
            raise ValueError(f"Unknown noise_type: {args.noise_type}")
    else:
        theta_min, theta_max = args.theta_min, args.theta_max
    
    logger.info(f"Calibrating {args.noise_type} for target δ* = {args.target_delta} (match: {args.target_delta_type})")
    logger.info(f"Search range: [{theta_min}, {theta_max}]")
    #if args.noise_type in ('coupled_consistent_alpha', 'coupled_inconsistent_alpha'):
    #    logger.info(
    #        "coupled 类 δ 依赖 config 中 seed（决定 v）与 noise_injection（加噪层数）。"
    #        "若与以往标定结果不一致，请用 --diagnose 看 δ(θ) 曲线并核对上述配置是否与当时一致。"
    #    )
    
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
        max_iter=args.max_iter,
        target_delta_type=args.target_delta_type,
    )
    
    # 保存结果
    is_backward_type = args.noise_type in ('adv_direction_beta', 'sign_corrupt_p')
    output_data = {
        'config_path': args.config,
        'checkpoint_path': args.checkpoint,
        'noise_type': args.noise_type,
        'target_delta': args.target_delta,
        'target_delta_type': args.target_delta_type,
        'theta_star': theta_star,
        'delta_logit': result['delta_logit'],
        'delta_blocks': result['delta_blocks'],
        'error': result['error'],
        'calibration_size': args.calibration_size,
        'seed': args.seed,
        'theta_bounds': [theta_min, theta_max],
    }
    if is_backward_type:
        output_data['delta_type'] = 'grad'
        output_data['delta_grad'] = result['delta_logit']
    if result.get('delta_grad_induced') is not None:
        output_data['delta_grad_induced'] = result['delta_grad_induced']

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    # 本次标定匹配的是哪类 δ
    if args.target_delta_type == 'logit':
        matched_name = "δ_grad" if is_backward_type else "δ_logit"
        matched_val = result['delta_logit']
    else:
        matched_name = "δ_grad_induced" if result.get('delta_grad_induced') is not None else "δ_grad"
        matched_val = result.get('delta_grad_induced') or result['delta_logit']
    logger.info(f"Calibration complete! (target: {args.target_delta_type} = {args.target_delta})")
    logger.info(f"  θ* = {theta_star:.6f}")
    logger.info(f"  {matched_name} (matched) = {matched_val:.6f}")
    logger.info(f"  δ_logit = {result['delta_logit']:.6f}")
    if result.get('delta_grad_induced') is not None:
        logger.info(f"  δ_grad_induced = {result['delta_grad_induced']:.6f}")
    logger.info(f"  Error = {result['error']:.6f}")
    if args.target_delta >= 0.8 and args.noise_type not in ('adv_direction_beta', 'sign_corrupt_p'):
        logger.info(
            "  Note: at large δ the θ–δ relation may be non-injective; forward (calibrate) and reverse (--eval_theta) can disagree. Use small-to-moderate δ for reliable θ ↔ δ."
        )
    logger.info(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()

