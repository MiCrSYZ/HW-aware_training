"""
Sanity check script for verifying numerical correctness of memristor simulation.

This script verifies that:
1. Hardware-aware forward pass produces outputs with comparable magnitude to float forward
2. Gradients are properly computed (not detached)
3. The adaptive mapping and scale recovery work correctly
"""

import torch
import torch.nn.functional as F
from typing import Tuple

try:
    from ..memristor.device_model import MemristorDeviceModel
    from ..models.memristor_wrappers import hardware_linear_forward_adaptive
except ImportError:
    import sys
    import os
    # Add project root to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, '../..')
    project_root = os.path.abspath(project_root)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.memristor.device_model import MemristorDeviceModel
    from src.models.memristor_wrappers import hardware_linear_forward_adaptive


def sanity_check_layer(
    batch_size: int = 32,
    in_features: int = 128,
    out_features: int = 64,
    device: str = 'cpu',
    verbose: bool = True,
) -> Tuple[bool, dict]:
    """
    Sanity check for a single linear layer with memristor simulation.
    
    This function:
    1. Creates random input x and weight W
    2. Computes float forward: out_float = F.linear(x, W)
    3. Computes hardware forward using adaptive mapping
    4. Compares output magnitudes (mean, std)
    5. Performs backward pass and verifies gradients exist
    
    Args:
        batch_size: Batch size for input
        in_features: Input feature dimension
        out_features: Output feature dimension
        device: Device to run on ('cpu' or 'cuda')
        verbose: Whether to print detailed information
        
    Returns:
        success: True if all checks pass
        stats: Dictionary with statistics and check results
    """
    device_torch = torch.device(device)
    
    # Create device model with typical parameters
    device_model = MemristorDeviceModel(
        G_min=1e-6,
        G_max=1e-4,
        weight_clip=(-1.0, 1.0),
        variability_sigma=0.05,
        read_noise_sigma=1e-7,
        drift_alpha=1e-4,
        stuck_ratio=0.0,
        ir_drop_beta=0.0,
        mapping='linear',
    )
    
    # Create random input and weights
    torch.manual_seed(42)
    x = torch.randn(batch_size, in_features, device=device_torch, requires_grad=True)
    W = torch.randn(out_features, in_features, device=device_torch, requires_grad=True)
    
    # Clamp weights to reasonable range
    W = torch.clamp(W, -1.0, 1.0)
    
    if verbose:
        print("=" * 60)
        print("Sanity Check: Memristor Layer Numerical Correctness")
        print("=" * 60)
        print(f"Input shape: {x.shape}")
        print(f"Weight shape: {W.shape}")
        print(f"Input stats: mean={x.mean().item():.6f}, std={x.std().item():.6f}")
        print(f"Weight stats: mean={W.mean().item():.6f}, std={W.std().item():.6f}, "
              f"abs_max={W.abs().max().item():.6f}")
        print()
    
    # 1. Float forward pass
    out_float = F.linear(x, W)
    
    if verbose:
        print("Float Forward Pass:")
        print(f"  Output mean: {out_float.mean().item():.6f}")
        print(f"  Output std: {out_float.std().item():.6f}")
        print(f"  Output abs_max: {out_float.abs().max().item():.6f}")
        print()
    
    # 2. Hardware forward pass with adaptive mapping
    out_hw, debug_info = hardware_linear_forward_adaptive(
        x, W, device_model, t=0, seed=42
    )
    Gp_noisy, Gn_noisy, W_eff_conductance, scale = debug_info
    
    if verbose:
        print("Hardware Forward Pass (Adaptive):")
        print(f"  Output mean: {out_hw.mean().item():.6f}")
        print(f"  Output std: {out_hw.std().item():.6f}")
        print(f"  Output abs_max: {out_hw.abs().max().item():.6f}")
        print()
        print("Debug Info:")
        print(f"  G_pos mean: {Gp_noisy.mean().item():.6e}")
        print(f"  G_neg mean: {Gn_noisy.mean().item():.6e}")
        print(f"  W_eff_conductance mean: {W_eff_conductance.mean().item():.6e}")
        print(f"  Scale: {scale.item():.6e}")
        print()
    
    # 3. Compare magnitudes
    mean_ratio = out_hw.abs().mean().item() / (out_float.abs().mean().item() + 1e-12)
    std_ratio = out_hw.std().item() / (out_float.std().item() + 1e-12)
    
    if verbose:
        print("Magnitude Comparison:")
        print(f"  Mean ratio (hw/float): {mean_ratio:.6f}")
        print(f"  Std ratio (hw/float): {std_ratio:.6f}")
        print()
    
    # 4. Backward pass check
    # Create a dummy loss
    loss_float = out_float.sum()
    loss_hw = out_hw.sum()
    
    # Backward on float
    loss_float.backward(retain_graph=True)
    grad_W_float = W.grad.clone() if W.grad is not None else None
    x.grad = None
    W.grad = None
    
    # Backward on hardware
    loss_hw.backward()
    grad_W_hw = W.grad.clone() if W.grad is not None else None
    
    # Check gradients
    grad_exists = grad_W_hw is not None
    grad_nonzero = grad_W_hw is not None and grad_W_hw.abs().max().item() > 1e-10
    
    if verbose:
        print("Gradient Check:")
        print(f"  W.grad exists: {grad_exists}")
        if grad_W_hw is not None:
            print(f"  W.grad mean: {grad_W_hw.mean().item():.6e}")
            print(f"  W.grad std: {grad_W_hw.std().item():.6e}")
            print(f"  W.grad abs_max: {grad_W_hw.abs().max().item():.6e}")
            print(f"  W.grad is non-zero: {grad_nonzero}")
        print()
    
    # 5. Check criteria
    # Output magnitudes should be within reasonable range (not orders of magnitude different)
    # We allow some difference due to non-idealities, but should be within 1-2 orders
    mean_ratio_ok = 0.1 <= mean_ratio <= 10.0
    std_ratio_ok = 0.1 <= std_ratio <= 10.0
    
    # Both outputs should have reasonable magnitude (not tiny)
    float_magnitude_ok = out_float.abs().mean().item() > 1e-6
    hw_magnitude_ok = out_hw.abs().mean().item() > 1e-6
    
    all_checks_pass = (
        mean_ratio_ok and
        std_ratio_ok and
        float_magnitude_ok and
        hw_magnitude_ok and
        grad_exists and
        grad_nonzero
    )
    
    if verbose:
        print("Check Results:")
        print(f"  Mean ratio in range [0.1, 10.0]: {mean_ratio_ok} (ratio={mean_ratio:.6f})")
        print(f"  Std ratio in range [0.1, 10.0]: {std_ratio_ok} (ratio={std_ratio:.6f})")
        print(f"  Float output magnitude OK: {float_magnitude_ok} (mean_abs={out_float.abs().mean().item():.6e})")
        print(f"  Hardware output magnitude OK: {hw_magnitude_ok} (mean_abs={out_hw.abs().mean().item():.6e})")
        print(f"  Gradients exist: {grad_exists}")
        print(f"  Gradients non-zero: {grad_nonzero}")
        print()
        print("=" * 60)
        if all_checks_pass:
            print("✓ ALL CHECKS PASSED")
        else:
            print("✗ SOME CHECKS FAILED")
        print("=" * 60)
    
    stats = {
        'float_mean': out_float.mean().item(),
        'float_std': out_float.std().item(),
        'float_abs_max': out_float.abs().max().item(),
        'hw_mean': out_hw.mean().item(),
        'hw_std': out_hw.std().item(),
        'hw_abs_max': out_hw.abs().max().item(),
        'mean_ratio': mean_ratio,
        'std_ratio': std_ratio,
        'scale': scale.item(),
        'grad_exists': grad_exists,
        'grad_nonzero': grad_nonzero,
        'mean_ratio_ok': mean_ratio_ok,
        'std_ratio_ok': std_ratio_ok,
        'float_magnitude_ok': float_magnitude_ok,
        'hw_magnitude_ok': hw_magnitude_ok,
        'all_checks_pass': all_checks_pass,
    }
    
    return all_checks_pass, stats


if __name__ == '__main__':
    # Run sanity check
    success, stats = sanity_check_layer(
        batch_size=32,
        in_features=128,
        out_features=64,
        device='cpu',
        verbose=True,
    )
    
    # Exit with error code if checks fail
    exit(0 if success else 1)

