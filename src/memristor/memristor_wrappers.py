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
) -> tuple:

    # Map weights to differential conductances adaptively
    Gp, Gn, max_abs = device_model.map_weights_to_conductance_diff_adaptive(W)

    # Apply non-idealities separately (seed can be different or same depending on model)
    Gp_noisy = device_model.apply_nonidealities(Gp, t=t, seed=seed)
    Gn_noisy = device_model.apply_nonidealities(Gn, t=t, seed=(None if seed is None else seed+1))

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

    # Perform linear operation with effective weight
    # NOTE: All operations above maintain gradients - no detach() calls
    # Always use tiling if array_size > 0 (for both comp and no_comp)
    use_tiling = (hasattr(device_model, 'array_size') and device_model.array_size > 0)

    if use_tiling:
        # Compute analog tile outputs (no quant here)
        analog_out = device_model.matmul_with_tiling(
            x, W_eff,
            adc_bits=None,
            per_tile_quant=False  # always false: quant happens after tile-sum
        )
        out = analog_out
    else:
        out = F.linear(x, W_eff)

    # Final ADC quantization AFTER tile-sum
    # Only apply ADC quantization during inference (not during training) to preserve gradients
    if hasattr(device_model, "enable_adc") and device_model.enable_adc and not training:
        out = device_model.adc_quant(out, bits=device_model.adc_bits)

    # Apply new IR-drop correction based on paper equations (16)-(18) if enabled
    # For 'paper' mode, only apply IR-drop during inference (not during training)
    if device_model.ir_drop_mode == "paper" and not training:
        # Compute normalization factors
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
        y_tilde_with_ir = device_model.apply_ir_drop_paper(
            y_tilde,
            W_tilde_normalized,
            x_tilde_normalized
        )
        
        # Convert back to physical scale
        out = y_tilde_with_ir * output_scale

    return out, (Gp_noisy, Gn_noisy, W_eff_conductance, scale)


class MemristorLinear(nn.Module):
    """
    Linear layer with memristor device non-idealities.

    This layer wraps a standard nn.Linear and applies memristor device
    modeling during the forward pass. The weights are mapped to conductance
    and non-idealities are applied before computation.
    """

    def __init__(self, linear: nn.Linear, device_model: MemristorDeviceModel):
        """
        Initialize memristor-aware linear layer.

        Args:
            linear: Standard nn.Linear layer to wrap
            device_model: MemristorDeviceModel instance
        """
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.device_model = device_model

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
        out, _ = hardware_linear_forward_adaptive(x, self.weight, self.device_model, t=t, seed=seed, training=self.training)
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

    def __init__(self, conv: nn.Conv2d, device_model: MemristorDeviceModel):
        """
        Initialize memristor-aware convolutional layer.

        Args:
            conv: Standard nn.Conv2d layer to wrap
            device_model: MemristorDeviceModel instance
        """
        super().__init__()
        self.device_model = device_model

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
            t=t, seed=seed, training=self.training
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

