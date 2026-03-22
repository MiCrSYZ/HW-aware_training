import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

try:
    from .device_model import MemristorDeviceModel
except ImportError:
    from src.memristor.device_model import MemristorDeviceModel


def hardware_linear_forward_adaptive(
    x: torch.Tensor,
    W: torch.Tensor,
    device_model: MemristorDeviceModel,
    t: int = 0,
    seed: Optional[int] = None,
    training: bool = True,
    enable_noise: bool = True,  # Control whether to inject noise
) -> tuple:

    # Map weights to differential conductances adaptively
    Gp, Gn, max_abs = device_model.map_weights_to_conductance_diff_adaptive(W)

    # Apply non-idealities separately (seed can be different or same depending on model)
    # 对于合成噪声类型，需要根据类型决定是否应用非理想性
    synthetic_noise_type = getattr(device_model, 'synthetic_noise_type', 'none')
    # full_variability应该应用variability_sigma，即使enable_noise=False（因为它是通过synthetic_noise_type控制的）
    # 但是，如果enable_noise=False且synthetic_noise_type='none'，则不应用任何噪声
    if synthetic_noise_type == 'full_variability':
        # full_variability总是应用variability_sigma（通过apply_nonidealities）
        apply_nonidealities = True
    elif synthetic_noise_type == 'none':
        # none模式：只有当enable_noise=True时才应用非理想性（通过noise_injection配置控制）
        apply_nonidealities = enable_noise
    else:
        # cond1/cond2/cond3模式：不应用传统的非理想性（variability_sigma等），只应用合成噪声
        apply_nonidealities = False
    
    if apply_nonidealities:
        Gp_noisy = device_model.apply_nonidealities(Gp, t=t, seed=seed)
        Gn_noisy = device_model.apply_nonidealities(Gn, t=t, seed=(None if seed is None else seed+1))
    else:
        # No noise: use original conductances
        Gp_noisy = Gp
        Gn_noisy = Gn

    # Effective conductance difference (this is in conductance units, ~1e-6 to 1e-4)
    W_eff_conductance = Gp_noisy - Gn_noisy

    # Recover to approximate original weight magnitude:
    # scale = max_abs / (G_max - G_min)
    # This converts from conductance difference to weight magnitude
    G_range = device_model.G_max - device_model.G_min
    scale = max_abs / (G_range + 1e-12)
    # Clip scale to avoid numerical issues
    scale = torch.clamp(scale, min=1e-3, max=1e6)
    # Convert effective conductance to effective weight
    W_eff = W_eff_conductance * scale

    # Apply cond.1 方差有界噪声 (在权重上)
    # W_eff = W ⊙ (1 + αε), where ε ~ t_ν, ν ≤ 2
    # 注意：合成噪声独立于enable_noise，因为它不是通过noise_injection配置控制的
    if hasattr(device_model, 'synthetic_noise_type') and device_model.synthetic_noise_type == 'cond1_variance_bounded':
        nu = device_model.cond1_nu
        # 使用PyTorch的StudentT分布生成t分布噪声
        t_dist = torch.distributions.StudentT(df=nu)
        if seed is not None:
            # 设置随机种子（注意：StudentT.sample不支持generator参数，所以使用全局seed）
            # 为了更好的可重复性，我们使用手动实现
            generator = torch.Generator(device=W_eff.device)
            generator.manual_seed(seed)
            # t分布 = Z / sqrt(χ²/ν)，其中 Z ~ N(0,1), χ² ~ Gamma(ν/2, 2)
            Z = torch.randn(W_eff.shape, generator=generator, device=W_eff.device, dtype=W_eff.dtype, requires_grad=False)
            # 使用Gamma分布生成卡方分布: χ² ~ Gamma(ν/2, 2)
            gamma_dist = torch.distributions.Gamma(concentration=nu / 2.0, rate=0.5)
            chi2 = gamma_dist.sample(W_eff.shape).to(W_eff.device)
            # 避免除零
            chi2 = torch.clamp(chi2, min=1e-8)
            t_noise = Z / torch.sqrt(chi2 / nu)
        else:
            t_noise = t_dist.sample(W_eff.shape).to(W_eff.device)
        
        alpha = device_model.cond1_alpha
        W_eff = W_eff * (1.0 + alpha * t_noise)

    # Perform linear operation with effective weight
    # NOTE: All operations above maintain gradients - no detach() calls
    # Always use tiling if array_size > 0 (for both comp and no_comp)
    use_tiling = (hasattr(device_model, 'array_size') and device_model.array_size > 0)

    if use_tiling:
        # Compute analog tile outputs (no quant here)
        analog_out = device_model.matmul_with_tiling(
            x, W_eff,
            adc_bits=None,
            per_tile_quant=False,  # always false: quant happens after tile-sum
            training=training
        )
        out = analog_out
    else:
        out = F.linear(x, W_eff)

    # Apply cond.2 梯度无偏噪声 (在输出上)
    # Forward: z = Wx, s(z) = 1 + α*tanh(v^T z), z̃ = s(z) ⊙ z
    # Backward: ∂z̃/∂W = detach(s(z)) · ∂z/∂W
    # 注意：合成噪声独立于enable_noise，因为它不是通过noise_injection配置控制的
    if hasattr(device_model, 'synthetic_noise_type') and device_model.synthetic_noise_type == 'cond2_gradient_unbiased':
        # 获取或生成固定的随机向量 v
        # v 的形状应该是 [out_features]，用于与输出 z 做内积
        out_features = out.shape[-1]
        v_key = (out_features,)
        
        if v_key not in device_model._cond2_v_vectors:
            # 生成固定的随机向量 v（基于seed，不训练）
            if device_model.seed is not None:
                generator = torch.Generator(device=out.device)
                # 使用不同的seed偏移来为不同层生成不同的v
                generator.manual_seed(device_model.seed + hash(v_key) % 1000000)
                v = torch.randn(out_features, generator=generator, device=out.device, dtype=out.dtype, requires_grad=False)
            else:
                v = torch.randn(out_features, device=out.device, dtype=out.dtype, requires_grad=False)
            # 归一化 v
            v = v / (torch.norm(v) + 1e-8)
            device_model._cond2_v_vectors[v_key] = v
        else:
            v = device_model._cond2_v_vectors[v_key].to(out.device)
        
        # 计算 s(z) = 1 + α * tanh(v^T z)
        # out shape: [batch, out_features] 或 [batch, num_patches, out_features]
        # v shape: [out_features]
        # 需要计算 v^T z，即对最后一个维度做内积
        if out.dim() == 2:
            # [batch, out_features]
            vTz = torch.sum(out * v.unsqueeze(0), dim=-1, keepdim=True)  # [batch, 1]
        else:
            # [batch, num_patches, out_features] 或其他形状
            vTz = torch.sum(out * v.view(1, -1), dim=-1, keepdim=True)  # [batch, num_patches, 1] 或类似
        
        alpha = device_model.cond2_alpha
        s_z = 1.0 + alpha * torch.tanh(vTz)  # [batch, 1] 或 [batch, num_patches, 1]
        
        # Forward: z̃ = s(z) ⊙ z
        # Backward: 使用 detach(s(z)) 来阻断梯度流
        s_z_detached = s_z.detach()
        out = out * s_z_detached + (out * s_z - out * s_z_detached).detach()

    # Final ADC quantization AFTER tile-sum
    # Apply ADC quantization if enabled and (inference OR training with enable_adc_during_training)
    # Only apply ADC if noise is enabled (ADC is part of noise injection)
    
    # 对于cond3_adc_direct，需要强制启用ADC
    # IMPORTANT: For cond1/cond2, do NOT apply ADC (only cond3 uses ADC)
    # For 'none'/'full_variability', apply ADC based on enable_noise and enable_adc config
    # 初始化 enable_adc_during_training，避免在日志中引用未赋值变量
    enable_adc_during_training = False
    if synthetic_noise_type == 'cond3_adc_direct':
        should_apply_adc = True  # cond3总是应用ADC
    elif synthetic_noise_type in ['cond1_variance_bounded', 'cond2_gradient_unbiased']:
        should_apply_adc = False  # cond1和cond2不应用ADC
    elif enable_noise and hasattr(device_model, "enable_adc") and device_model.enable_adc:
        # Only apply ADC for 'none' or 'full_variability' modes
        enable_adc_during_training = getattr(device_model, "enable_adc_during_training", False)
        should_apply_adc = not training or enable_adc_during_training
    else:
        should_apply_adc = False
    
    if should_apply_adc and hasattr(device_model, "enable_adc") and device_model.enable_adc:
        
        if should_apply_adc:
            if training:
                # Training mode: choose between STE and direct quantization
                adc_mode = getattr(device_model, 'adc_training_mode', 'direct')  # Default to 'direct' for backward compatibility
                
                # 对于cond3_adc_direct，强制使用direct模式
                if synthetic_noise_type == 'cond3_adc_direct':
                    adc_mode = 'direct'
                
                if adc_mode == 'ste':
                    # Use straight-through estimator (STE): forward uses quantized value, backward uses identity
                    # This allows gradients to flow through, but forward uses quantized values
                    out_quantized = device_model.adc_quant(out, bits=device_model.adc_bits)
                    out = out + (out_quantized - out).detach()
                elif adc_mode == 'direct':
                    # Direct quantization: completely block gradients by detaching the quantized output
                    # This is used to observe the effect of gradient vanishing on training dynamics
                    # Note: torch.round() actually has gradient=1 (straight-through), so we need to detach
                    # to truly block gradients and observe training collapse
                    out_quantized = device_model.adc_quant(out, bits=device_model.adc_bits)
                    # Completely detach to block all gradients (unlike STE which uses original gradients)
                    out = out_quantized.detach()
                else:
                    raise ValueError(f"Unknown adc_training_mode: {adc_mode}. Must be 'ste' or 'direct'.")
            else:
                # Inference mode: always use direct quantization
                out = device_model.adc_quant(out, bits=device_model.adc_bits)

    # Apply new IR-drop correction based on paper equations (16)-(18) if enabled
    # Apply IR-drop if enabled and (inference OR training with enable_ir_drop_paper_during_training)
    # Only apply IR-drop if noise is enabled (IR-drop is part of noise injection)
    # IMPORTANT: Do NOT apply IR-drop when synthetic_noise_type is set (cond1/cond2/cond3/full_variability)
    # IR-drop should only be applied when synthetic_noise_type is 'none' (default mode)
    should_apply_ir_drop = (
        enable_noise and 
        device_model.ir_drop_mode == "paper" and
        synthetic_noise_type == 'none'  # Only apply IR-drop when no synthetic noise is used
    )
    if should_apply_ir_drop:
        should_apply_ir = not training or (hasattr(device_model, "enable_ir_drop_paper_during_training") and device_model.enable_ir_drop_paper_during_training)
        if should_apply_ir:
            # Compute normalization factors with numerical stability protection
            W_abs_max = torch.abs(W_eff).max().clamp(min=1e-8)  # Avoid division by zero
            x_abs_max = torch.abs(x).max().clamp(min=1e-8)
            
            # Normalize weights: W_tilde = W_eff / W_abs_max (keeps values in [-1, 1] range)
            W_tilde_normalized = W_eff / W_abs_max  # [out_features, in_features]
            
            # Normalize inputs: x_tilde = x / x_abs_max (keeps values in reasonable range)
            x_tilde_normalized = x / x_abs_max  # [batch, in_features]
            
            # Normalize output: y_tilde = out / (W_abs_max * x_abs_max) to maintain scale consistency
            # This ensures y_tilde has similar magnitude to the normalized computation
            output_scale = W_abs_max * x_abs_max
            y_tilde = out / (output_scale + 1e-12)  # [batch, out_features]
            
            # Apply IR-drop correction
            # Note: We do NOT skip NaN/Inf in training - this is intentional to observe
            # whether paper IR-drop causes numerical instability (proving its unlearnability)
            try:
                y_tilde_with_ir = device_model.apply_ir_drop_paper(
                    y_tilde,
                    W_tilde_normalized,
                    x_tilde_normalized
                )
                
                # Convert back to physical scale
                out_ir = y_tilde_with_ir * output_scale
                
                # Use IR-drop output directly (including NaN/Inf if present)
                # This allows us to observe numerical instability in training
                out = out_ir
                
            except Exception as e:
                # If IR-drop computation fails, raise the error instead of silently skipping
                # This is important for the matched-distortion experiment to detect failures
                if training:
                    # In training, raise the error to observe failures
                    raise RuntimeError(f"IR-drop computation failed during training: {e}") from e
                else:
                    # In inference, log the error but continue
                    import warnings
                    warnings.warn(f"IR-drop computation failed: {e}. Using original output.")
                    pass  # out remains unchanged

    return out, (Gp_noisy, Gn_noisy, W_eff_conductance, scale)


class MemristorLinear(nn.Module):
    """
    Linear layer with memristor device non-idealities.

    This layer wraps a standard nn.Linear and applies memristor device
    modeling during the forward pass. The weights are mapped to conductance
    and non-idealities are applied before computation.
    """

    def __init__(self, linear: nn.Linear, device_model: MemristorDeviceModel, enable_noise: bool = True):
        """
        Initialize memristor-aware linear layer.

        Args:
            linear: Standard nn.Linear layer to wrap
            device_model: MemristorDeviceModel instance
            enable_noise: Whether to inject noise in this layer (default: True for backward compatibility)
        """
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.device_model = device_model
        self.enable_noise = enable_noise  # Control flag for noise injection

        # Copy weights and bias
        self.weight = nn.Parameter(linear.weight.data.clone())
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.data.clone())
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor, t: int = 0, seed: Optional[int] = None) -> torch.Tensor:
        """
        Forward pass with memristor non-idealities using adaptive differential mapping.

        Args:
            x: Input tensor [batch, in_features]
            t: Time/cycle index for drift
            seed: Random seed for reproducibility

        Returns:
            Output tensor [batch, out_features]
        """
        out, _ = hardware_linear_forward_adaptive(
            x, self.weight, self.device_model, 
            t=t, seed=seed, training=self.training,
            enable_noise=self.enable_noise
        )
        if self.bias is not None:
            out = out + self.bias
        return out


class MemristorConv2d(nn.Module):
    """
    Convolutional layer with memristor device non-idealities.

    This layer wraps a standard nn.Conv2d and applies memristor device
    modeling during the forward pass. The convolution is implemented by
    unfolding the input, applying linear operations with memristor-mapped
    weights, and folding back.
    """

    def __init__(self, conv: nn.Conv2d, device_model: MemristorDeviceModel, enable_noise: bool = True):
        """
        Initialize memristor-aware convolutional layer.

        Args:
            conv: Standard nn.Conv2d layer to wrap
            device_model: MemristorDeviceModel instance
            enable_noise: Whether to inject noise in this layer (default: True for backward compatibility)
        """
        super().__init__()
        self.device_model = device_model
        self.enable_noise = enable_noise  # Control flag for noise injection

        # Copy conv parameters
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size if isinstance(conv.kernel_size, tuple) else (conv.kernel_size,
                                                                                         conv.kernel_size)
        self.stride = conv.stride if isinstance(conv.stride, tuple) else (conv.stride, conv.stride)
        self.padding = conv.padding if isinstance(conv.padding, tuple) else (conv.padding, conv.padding)
        self.bias = conv.bias is not None

        # Copy weights and bias
        self.weight = nn.Parameter(conv.weight.data.clone())
        if self.bias:
            self.bias_param = nn.Parameter(conv.bias.data.clone())
        else:
            self.register_parameter('bias_param', None)

    def forward(self, x: torch.Tensor, t: int = 0, seed: Optional[int] = None) -> torch.Tensor:
        """
        Forward pass with memristor non-idealities.

        Args:
            x: Input tensor [batch, in_channels, H, W]
            t: Time/cycle index for drift
            seed: Random seed for reproducibility

        Returns:
            Output tensor [batch, out_channels, H', W']
        """
        batch_size = x.size(0)
        in_h, in_w = x.size(2), x.size(3)
        k_h, k_w = self.kernel_size
        stride_h, stride_w = self.stride
        pad_h, pad_w = self.padding

        # Unfold input into patches
        # x_unfold: [batch, in_channels*k_h*k_w, num_patches]
        x_unfold = F.unfold(
            x,
            kernel_size=self.kernel_size,
            dilation=1,
            padding=self.padding,
            stride=self.stride
        )

        # Flatten conv weight to matrix form
        # W_flat: [out_channels, in_channels*k_h*k_w]
        W_flat = self.weight.view(self.out_channels, -1)

        # Transpose for linear operation: [batch, num_patches, in_channels*k_h*k_w]
        x_flat = x_unfold.transpose(1, 2)

        # Unified hardware forward
        out_flat, _ = hardware_linear_forward_adaptive(
            x_flat, W_flat,
            self.device_model,
            t=t, seed=seed, training=self.training,
            enable_noise=self.enable_noise
        )

        # Transpose back: [batch, out_channels, num_patches]
        out_unfold = out_flat.transpose(1, 2)

        # Calculate output dimensions
        out_h = (in_h + 2 * pad_h - k_h) // stride_h + 1
        out_w = (in_w + 2 * pad_w - k_w) // stride_w + 1

        # Fold back to feature map
        out = F.fold(
            out_unfold,
            output_size=(out_h, out_w),
            kernel_size=1
        )

        # Add bias
        if self.bias:
            out = out + self.bias_param.view(1, -1, 1, 1)

        return out

