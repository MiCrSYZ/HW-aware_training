"""
Diagnostic script to analyze input dependence of hardware errors.

This script:
1. Loads a trained model
2. Selects a representative layer
3. Samples multiple input batches
4. Computes hardware errors for each input
5. Analyzes input dependence metrics
6. Visualizes results
"""

import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

from ..utils.seeds import set_seed
from ..data.dataset import get_dataloaders
from ..models.model_zoo import get_model, wrap_model_with_memristor
from ..memristor.device_model import MemristorDeviceModel
from ..utils.hardware_error_diagnosis import (
    hardware_error,
    diagnose_input_dependence,
)
from ..utils.checkpoint import load_checkpoint

# Import memristor layer classes for type checking
try:
    from ..memristor.memristor_wrappers import MemristorLinear, MemristorConv2d
except ImportError:
    from src.memristor.memristor_wrappers import MemristorLinear, MemristorConv2d

try:
    from ..memristor.learned_weight_mapping import (
        MemristorLinear as LearnedMappingMemristorLinear,
        MemristorConv2d as LearnedMappingMemristorConv2d
    )
except ImportError:
    from src.memristor.learned_weight_mapping import (
        MemristorLinear as LearnedMappingMemristorLinear,
        MemristorConv2d as LearnedMappingMemristorConv2d
    )


def select_representative_layer(
    model: nn.Module,
    layer_name: Optional[str] = None,
) -> Tuple[nn.Module, str, torch.Tensor]:
    """
    Select a representative layer from the model.
    
    Args:
        model: Model to analyze (may be wrapped with MemristorModel)
        layer_name: Optional specific layer name. If None, selects a middle layer.
        
    Returns:
        Tuple of (layer_module, layer_name, weight_tensor)
    """
    # Get base model if wrapped
    base_model = model
    if hasattr(model, 'base_model'):
        base_model = model.base_model
        print(f"Model is wrapped with MemristorModel, using base_model")
    
    # Import memristor layer classes
    try:
        from ..memristor.memristor_wrappers import MemristorLinear, MemristorConv2d
    except ImportError:
        from src.memristor.memristor_wrappers import MemristorLinear, MemristorConv2d
    
    try:
        from ..memristor.learned_weight_mapping import (
            MemristorLinear as LearnedMappingMemristorLinear,
            MemristorConv2d as LearnedMappingMemristorConv2d
        )
    except ImportError:
        from src.memristor.learned_weight_mapping import (
            MemristorLinear as LearnedMappingMemristorLinear,
            MemristorConv2d as LearnedMappingMemristorConv2d
        )
    
    # Get all linear/conv layers (including memristor-wrapped ones)
    layers = []
    all_layers = []  # For debugging
    
    for name, module in base_model.named_modules():
        # Check if it's a linear or conv layer (including memristor-wrapped)
        is_linear = isinstance(module, (nn.Linear, MemristorLinear, LearnedMappingMemristorLinear))
        is_conv = isinstance(module, (nn.Conv2d, MemristorConv2d, LearnedMappingMemristorConv2d))
        
        if is_linear or is_conv:
            all_layers.append((name, type(module).__name__))
            
            # Skip first and last layers (optional filtering)
            if 'fc' in name.lower() or 'classifier' in name.lower() or 'head' in name.lower():
                continue
            if 'conv1' in name or ('layer1' in name and 'conv1' in name) or 'stem' in name:
                continue
            
            # Check if it has weight
            if hasattr(module, 'weight') and module.weight is not None:
                layers.append((name, module))
    
    # Print all available layers for debugging
    print(f"\nAll available layers ({len(all_layers)}):")
    for name, module_type in all_layers[:10]:  # Show first 10
        print(f"  {name}: {module_type}")
    if len(all_layers) > 10:
        print(f"  ... and {len(all_layers) - 10} more")
    
    if not layers:
        # Fallback: get any linear/conv layer with weight
        print("\nNo filtered layers found, using all layers...")
        for name, module in base_model.named_modules():
            is_linear = isinstance(module, (nn.Linear, MemristorLinear, LearnedMappingMemristorLinear))
            is_conv = isinstance(module, (nn.Conv2d, MemristorConv2d, LearnedMappingMemristorConv2d))
            if (is_linear or is_conv) and hasattr(module, 'weight') and module.weight is not None:
                layers.append((name, module))
    
    if not layers:
        raise ValueError(
            f"No suitable layer found in model. "
            f"Available layers: {[n for n, _ in all_layers]}. "
            f"Make sure the model has Linear or Conv2d layers with weights."
        )
    
    # Select middle layer if not specified
    if layer_name is None:
        idx = len(layers) // 2
        name, module = layers[idx]
    else:
        # Find specified layer
        found = False
        for name, module in layers:
            if layer_name in name:
                found = True
                break
        if not found:
            raise ValueError(f"Layer '{layer_name}' not found. Available: {[n for n, _ in layers]}")
    
    # Get weight
    if hasattr(module, 'weight') and module.weight is not None:
        W = module.weight.data.clone()
    else:
        raise ValueError(f"Layer '{name}' has no weight parameter")
    
    print(f"Selected layer: {name}")
    print(f"  Type: {type(module).__name__}")
    print(f"  Weight shape: {W.shape}")
    
    return module, name, W


def sample_inputs(
    loader,
    num_samples: int = 20,
    batch_size: int = 32,
    device: torch.device = torch.device('cpu'),
) -> List[torch.Tensor]:
    """
    Sample multiple input batches from data loader.
    
    Args:
        loader: Data loader
        num_samples: Number of input batches to sample
        batch_size: Batch size for each sample
        device: Device to place inputs on
        
    Returns:
        List of input batches [B, ...]
    """
    inputs = []
    count = 0
    
    for data, _ in loader:
        data = data.to(device)
        
        # For conv layers, we need to flatten/unfold
        # For now, assume we'll handle this in the analysis
        
        inputs.append(data[:batch_size])  # Take first batch_size samples
        count += 1
        
        if count >= num_samples:
            break
    
    if len(inputs) < num_samples:
        print(f"Warning: Only sampled {len(inputs)} batches (requested {num_samples})")
    
    return inputs


def extract_layer_inputs(
    model: nn.Module,
    layer_name: str,
    inputs: List[torch.Tensor],
    device: torch.device,
) -> List[torch.Tensor]:
    """
    Extract inputs to a specific layer by running forward pass up to that layer.
    
    Args:
        model: Model (may be wrapped with MemristorModel)
        layer_name: Name of target layer
        inputs: List of model inputs
        device: Device
        
    Returns:
        List of layer inputs
    """
    layer_inputs = []
    
    # Get base model if wrapped
    base_model = model
    if hasattr(model, 'base_model'):
        base_model = model.base_model
    
    def hook_fn(module, input, output):
        # Store the input to this layer
        if isinstance(input, tuple):
            layer_inputs.append(input[0].detach().clone())
        else:
            layer_inputs.append(input.detach().clone())
    
    # Register hook
    hook = None
    for name, module in base_model.named_modules():
        if name == layer_name:
            hook = module.register_forward_hook(hook_fn)
            break
    
    if hook is None:
        raise ValueError(f"Layer '{layer_name}' not found in model")
    
    # Run forward passes
    model.eval()
    with torch.no_grad():
        for inp in inputs:
            try:
                _ = model(inp, t=0)
            except TypeError:
                _ = model(inp)
    
    # Remove hook
    hook.remove()
    
    return layer_inputs


def analyze_linear_layer(
    x_list: List[torch.Tensor],
    W: torch.Tensor,
    device_model: MemristorDeviceModel,
    t: int = 0,
    seed: Optional[int] = None,
    model: Optional[nn.Module] = None,
    criterion: Optional[nn.Module] = None,
    inputs_list: Optional[List[torch.Tensor]] = None,
    targets_list: Optional[List[torch.Tensor]] = None,
    layer_module: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
) -> Tuple[Dict[str, Any], List[torch.Tensor]]:
    """
    Analyze input dependence for a linear layer.
    
    Args:
        x_list: List of input batches [B, in_dim]
        W: Weight matrix [out_dim, in_dim]
        device_model: Device model
        t: Time/cycle index
        seed: Random seed
        model: (Optional) Neural network model for loss-sensitivity computation
        criterion: (Optional) Loss function for loss-sensitivity computation
        inputs_list: (Optional) List of full input batches for model forward pass
        targets_list: (Optional) List of target labels for loss computation
        layer_module: (Optional) Layer module to compute gradients for
        device: (Optional) Device to run computation on
        
    Returns:
        Tuple of (diagnostic results dictionary, list of error tensors)
    """
    # Compute hardware errors for each input
    eps_list = []
    x_processed = []
    h_ideal_list = []
    h_hw_list = []
    
    for x in x_list:
        # Ensure x is 2D [B, in_dim]
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x_processed.append(x)
        
        eps = hardware_error(x, W, device_model, t=t, seed=seed)
        eps_list.append(eps)
        
        # Compute ideal and hardware outputs for loss-sensitivity computation
        if model is not None and layer_module is not None:
            import torch.nn.functional as F
            # Ideal output: x @ W.T
            h_ideal = F.linear(x, W)
            h_ideal_list.append(h_ideal)
            # Hardware output: ideal + error
            h_hw = h_ideal + eps
            h_hw_list.append(h_hw)
    
    # Run comprehensive diagnosis
    results = diagnose_input_dependence(
        x_processed, eps_list, ref_idx=0,
        model=model,
        criterion=criterion,
        inputs_list=inputs_list,
        targets_list=targets_list,
        h_ideal_list=h_ideal_list if h_ideal_list else None,
        h_hw_list=h_hw_list if h_hw_list else None,
        layer_module=layer_module,
        device=device,
    )
    
    return results, eps_list


def analyze_conv2d_layer(
    x_list: List[torch.Tensor],
    W: torch.Tensor,
    layer_module: nn.Module,
    device_model: MemristorDeviceModel,
    t: int = 0,
    seed: Optional[int] = None,
    model: Optional[nn.Module] = None,
    criterion: Optional[nn.Module] = None,
    inputs_list: Optional[List[torch.Tensor]] = None,
    targets_list: Optional[List[torch.Tensor]] = None,
    device: Optional[torch.device] = None,
) -> Tuple[Dict[str, Any], List[torch.Tensor]]:
    """
    Analyze input dependence for a Conv2d layer.
    
    Args:
        x_list: List of input feature maps [B, C, H, W]
        W: Weight tensor [out_ch, in_ch, k_h, k_w]
        layer_module: Conv2d layer module (to get kernel_size, stride, padding)
        device_model: Device model
        t: Time/cycle index
        seed: Random seed
        model: (Optional) Neural network model for loss-sensitivity computation
        criterion: (Optional) Loss function for loss-sensitivity computation
        inputs_list: (Optional) List of full input batches for model forward pass
        targets_list: (Optional) List of target labels for loss computation
        device: (Optional) Device to run computation on
        
    Returns:
        Tuple of (diagnostic results dictionary, list of error tensors)
    """
    import torch.nn.functional as F
    
    # Filter out invalid inputs (must be 4D)
    x_list_valid = []
    for x in x_list:
        if x.dim() == 4:
            x_list_valid.append(x)
        else:
            print(f"Warning: Skipping input with shape {x.shape} (expected 4D)")
    
    if not x_list_valid:
        raise ValueError("No valid 4D inputs found for Conv2d layer analysis")
    
    x_list = x_list_valid
    
    # Get conv parameters
    if hasattr(layer_module, 'kernel_size'):
        kernel_size = layer_module.kernel_size
        if isinstance(kernel_size, tuple):
            k_h, k_w = kernel_size
        else:
            k_h = k_w = kernel_size
    else:
        # Fallback: infer from weight shape
        _, _, k_h, k_w = W.shape
    
    if hasattr(layer_module, 'stride'):
        stride = layer_module.stride
        if isinstance(stride, tuple):
            stride_h, stride_w = stride
        else:
            stride_h = stride_w = stride
    else:
        stride_h = stride_w = 1
    
    if hasattr(layer_module, 'padding'):
        padding = layer_module.padding
        if isinstance(padding, tuple):
            pad_h, pad_w = padding
        else:
            pad_h = pad_w = padding
    else:
        pad_h = pad_w = 0
    
    # Flatten conv weight: [out_ch, in_ch, k_h, k_w] -> [out_ch, in_ch*k_h*k_w]
    out_ch, in_ch, _, _ = W.shape
    W_flat = W.view(out_ch, -1)  # [out_ch, in_ch*k_h*k_w]
    
    # Process each input
    eps_list = []
    x_processed = []
    h_ideal_list = []
    h_hw_list = []
    
    for x in x_list:
        # Ensure x is 4D [B, C, H, W]
        if x.dim() != 4:
            raise ValueError(f"Conv2d input must be 4D [B, C, H, W], got {x.shape}")
        
        # Unfold input: [B, C, H, W] -> [B, C*k_h*k_w, num_patches]
        x_unfold = F.unfold(
            x,
            kernel_size=(k_h, k_w),
            dilation=1,
            padding=(pad_h, pad_w),
            stride=(stride_h, stride_w)
        )
        
        # Transpose: [B, C*k_h*k_w, num_patches] -> [B, num_patches, C*k_h*k_w]
        x_flat = x_unfold.transpose(1, 2)  # [B, num_patches, in_ch*k_h*k_w]
        
        # Flatten batch and patches: [B*num_patches, in_ch*k_h*k_w]
        B, num_patches, _ = x_flat.shape
        x_flat_2d = x_flat.reshape(-1, x_flat.size(-1))  # [B*num_patches, in_ch*k_h*k_w]
        
        # Compute hardware error for each patch
        eps_flat = hardware_error(x_flat_2d, W_flat, device_model, t=t, seed=seed)
        
        # Reshape back: [B*num_patches, out_ch] -> [B, num_patches, out_ch]
        eps = eps_flat.reshape(B, num_patches, out_ch)
        
        # For diagnosis, we'll average over patches or flatten
        # Let's flatten to [B*num_patches, out_ch] for analysis
        eps_flat_for_analysis = eps.reshape(-1, out_ch)
        x_flat_for_analysis = x_flat_2d
        
        eps_list.append(eps_flat_for_analysis)
        x_processed.append(x_flat_for_analysis)
        
        # Compute ideal and hardware outputs for loss-sensitivity computation
        if model is not None and layer_module is not None:
            try:
                # Ideal output: F.conv2d(x, W, ...)
                # x is 4D [B, C, H, W], W is 4D [out_ch, in_ch, k_h, k_w]
                # Output will be 4D [B, out_ch, H_out, W_out]
                # F.conv2d expects padding as int or (int, int) tuple
                h_ideal_4d = F.conv2d(x, W, stride=(stride_h, stride_w), padding=(pad_h, pad_w))
                
                # Store the 4D version for the forward hook
                # The forward hook needs the exact shape that the layer would normally output
                h_ideal_list.append(h_ideal_4d)  # Store 4D version [B, out_ch, H_out, W_out]
                
                # For matching with eps, we need to flatten
                # eps_flat_for_analysis is [B*num_patches, out_ch]
                # h_ideal_4d is [B, out_ch, H_out, W_out]
                # Reshape to match the patch structure
                h_ideal_flat = h_ideal_4d.permute(0, 2, 3, 1).reshape(-1, out_ch)  # [B*H_out*W_out, out_ch]
                
                # Match dimensions with eps_flat_for_analysis
                if h_ideal_flat.shape[0] == eps_flat_for_analysis.shape[0]:
                    # Hardware output: ideal + error (flattened for matching)
                    h_hw_flat = h_ideal_flat + eps_flat_for_analysis
                else:
                    # Dimensions don't match, create a dummy that matches eps shape
                    h_hw_flat = torch.zeros_like(eps_flat_for_analysis)
                h_hw_list.append(h_hw_flat)
            except Exception as e:
                # If computation fails, skip this sample for sensitivity computation
                # But we still need to store something to maintain list length
                print(f"Warning: Failed to compute ideal output for conv layer: {e}")
                # Store a dummy 4D tensor with the expected output shape
                # We need to infer the output shape from the input and conv parameters
                B = x.shape[0]
                H_out = (x.shape[2] + 2 * pad_h - k_h) // stride_h + 1
                W_out = (x.shape[3] + 2 * pad_w - k_w) // stride_w + 1
                h_ideal_dummy = torch.zeros(B, out_ch, H_out, W_out, device=x.device)
                h_ideal_list.append(h_ideal_dummy)
                h_hw_dummy = torch.zeros_like(eps_flat_for_analysis)
                h_hw_list.append(h_hw_dummy)
    
    # Run comprehensive diagnosis
    results = diagnose_input_dependence(
        x_processed, eps_list, ref_idx=0,
        model=model,
        criterion=criterion,
        inputs_list=inputs_list,
        targets_list=targets_list,
        h_ideal_list=h_ideal_list if h_ideal_list else None,
        h_hw_list=h_hw_list if h_hw_list else None,
        layer_module=layer_module,
        device=device,
    )
    
    # Return both results and eps_list for visualization
    # Note: eps_list contains flattened errors [B*num_patches, out_ch]
    # For visualization, we might want to reshape, but for now keep flattened
    return results, eps_list


def visualize_results(
    results: Dict[str, Any],
    x_list: List[torch.Tensor],
    eps_list: List[torch.Tensor],
    output_dir: Path,
    layer_name: str,
):
    """
    Visualize diagnostic results.
    
    Args:
        results: Diagnostic results dictionary
        x_list: List of input batches
        eps_list: List of error tensors
        output_dir: Output directory for plots
        layer_name: Name of analyzed layer
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Heatmap of errors for different inputs
    # Stack errors: [K, B, out_dim] -> average over batch -> [K, out_dim]
    eps_stack = torch.stack(eps_list)  # [K, B, out_dim]
    eps_mean = eps_stack.mean(dim=1).cpu().numpy()  # [K, out_dim]
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(eps_mean, cmap='RdBu_r', center=0, cbar_kws={'label': 'Error'})
    plt.xlabel('Output Dimension')
    plt.ylabel('Input Batch Index')
    plt.title(f'Hardware Error Heatmap - {layer_name}')
    plt.tight_layout()
    plt.savefig(output_dir / f'{layer_name}_error_heatmap.png', dpi=150)
    plt.close()
    
    # 2. Correlation matrix
    corr_matrix = results['correlation_matrix']
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                vmin=-1, vmax=1, cbar_kws={'label': 'Correlation'})
    plt.xlabel('Input Batch Index')
    plt.ylabel('Input Batch Index')
    plt.title(f'Error Correlation Matrix - {layer_name}')
    plt.tight_layout()
    plt.savefig(output_dir / f'{layer_name}_correlation_matrix.png', dpi=150)
    plt.close()
    
    # 3. Residual norms vs input index
    residuals = results['static_deltaW_residuals']
    residual_ratios = results['static_deltaW_residual_ratios']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(residuals, 'o-', linewidth=2, markersize=8)
    ax1.axhline(y=residuals[0], color='r', linestyle='--', label='Reference (fit)')
    ax1.set_xlabel('Input Batch Index')
    ax1.set_ylabel('Residual Norm')
    ax1.set_title('Static ΔW Residual Norms')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(residual_ratios, 'o-', linewidth=2, markersize=8, color='orange')
    ax2.axhline(y=residual_ratios[0], color='r', linestyle='--', label='Reference (fit)')
    ax2.set_xlabel('Input Batch Index')
    ax2.set_ylabel('Residual / Error Norm Ratio')
    ax2.set_title('Static ΔW Transferability')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{layer_name}_residuals.png', dpi=150)
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")


def analyze_multiple_layers(
    model: nn.Module,
    layer_names: List[str],
    eval_loader,
    device_model: MemristorDeviceModel,
    device: torch.device,
    config: Dict[str, Any],
    args,
) -> Dict[str, Dict[str, Any]]:
    """
    Analyze input dependence for multiple layers.
    
    Returns:
        Dictionary mapping layer_name -> diagnostic results
    """
    # Get base model if wrapped
    base_model = model
    if hasattr(model, 'base_model'):
        base_model = model.base_model
    
    # Find all layers
    all_layers_dict = {}
    for name, module in base_model.named_modules():
        is_linear = isinstance(module, (nn.Linear, MemristorLinear, LearnedMappingMemristorLinear))
        is_conv = isinstance(module, (nn.Conv2d, MemristorConv2d, LearnedMappingMemristorConv2d))
        if (is_linear or is_conv) and hasattr(module, 'weight') and module.weight is not None:
            all_layers_dict[name] = module
    
    # Filter to requested layers
    layers_to_analyze = {}
    for layer_name in layer_names:
        if layer_name in all_layers_dict:
            layers_to_analyze[layer_name] = all_layers_dict[layer_name]
        else:
            print(f"Warning: Layer '{layer_name}' not found, skipping")
    
    if not layers_to_analyze:
        raise ValueError(f"No valid layers found from: {layer_names}")
    
    print(f"\nAnalyzing {len(layers_to_analyze)} layers: {list(layers_to_analyze.keys())}")
    
    # Sample inputs and targets once (shared across all layers)
    print(f"\nSampling {args.num_samples} input batches...")
    model_inputs = []
    model_targets = []
    count = 0
    for data, target in eval_loader:
        model_inputs.append(data[:args.batch_size].to(device))
        model_targets.append(target[:args.batch_size].to(device))
        count += 1
        if count >= args.num_samples:
            break
    
    # Analyze each layer
    all_results = {}
    
    for layer_name, layer_module in layers_to_analyze.items():
        print(f"\n{'='*60}")
        print(f"Analyzing layer: {layer_name}")
        print(f"{'='*60}")
        
        W = layer_module.weight.data.clone()
        
        # Extract layer inputs
        layer_inputs = extract_layer_inputs(model, layer_name, model_inputs, device)
        
        if not layer_inputs:
            print(f"Warning: Could not extract inputs for {layer_name}, skipping")
            continue
        
        # Analyze based on layer type
        is_linear = isinstance(layer_module, (nn.Linear, MemristorLinear, LearnedMappingMemristorLinear))
        is_conv = isinstance(layer_module, (nn.Conv2d, MemristorConv2d, LearnedMappingMemristorConv2d))
        
        # Create criterion for loss-sensitivity computation
        criterion = nn.CrossEntropyLoss()
        
        if is_linear:
            # Process inputs for linear
            x_processed = []
            for x in layer_inputs:
                if x.dim() > 2:
                    x = x.view(x.size(0), -1)
                x_processed.append(x)
            
            results, eps_list = analyze_linear_layer(
                x_processed, W, device_model, t=args.t, seed=args.seed,
                model=model,
                criterion=criterion,
                inputs_list=model_inputs,
                targets_list=model_targets,
                layer_module=layer_module,
                device=device,
            )
        elif is_conv:
            results, eps_list = analyze_conv2d_layer(
                layer_inputs, W, layer_module, device_model, t=args.t, seed=args.seed,
                model=model,
                criterion=criterion,
                inputs_list=model_inputs,
                targets_list=model_targets,
                device=device,
            )
        else:
            print(f"Warning: Unsupported layer type for {layer_name}, skipping")
            continue
        
        # Store results
        all_results[layer_name] = {
            'results': results,
            'eps_list': eps_list,
            'layer_inputs': layer_inputs,
        }
        
        # Print summary
        print(f"  Error variance: {results['error_variance']:.6e}")
        print(f"  Mean correlation: {results['mean_correlation']:.4f}")
        avg_residual = np.mean(results['static_deltaW_residual_ratios'][1:]) if len(results['static_deltaW_residual_ratios']) > 1 else np.nan
        print(f"  Avg residual ratio: {avg_residual:.4f}")
        # Print loss-sensitivity weighted error if available
        if 'loss_sensitivity_weighted_error' in results and not np.isnan(results['loss_sensitivity_weighted_error']):
            print(f"  Loss-sensitivity weighted error (S_ℓ): {results['loss_sensitivity_weighted_error']:.6e}")
        if 'loss_sensitivity_weighted_error_abs' in results and not np.isnan(results['loss_sensitivity_weighted_error_abs']):
            print(f"  Loss-sensitivity weighted error (abs, S_ℓ^abs): {results['loss_sensitivity_weighted_error_abs']:.6e}")
    
    return all_results


def visualize_multi_layer_results(
    all_results: Dict[str, Dict[str, Any]],
    output_dir: Path,
):
    """Generate comparison visualizations for multiple layers."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    layer_names = list(all_results.keys())
    n_layers = len(layer_names)
    
    if n_layers == 0:
        return
    
    # Extract metrics
    error_vars = [all_results[name]['results']['error_variance'] for name in layer_names]
    mean_corrs = [all_results[name]['results']['mean_correlation'] for name in layer_names]
    residual_ratios = []
    for name in layer_names:
        ratios = all_results[name]['results']['static_deltaW_residual_ratios']
        avg_ratio = np.mean(ratios[1:]) if len(ratios) > 1 else np.nan
        residual_ratios.append(avg_ratio)
    
    # 1. Comparison bar plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Error variance
    ax = axes[0]
    valid_mask = ~np.isnan(error_vars)
    if valid_mask.sum() > 0:
        ax.bar(range(n_layers), error_vars, alpha=0.7)
        ax.set_xticks(range(n_layers))
        ax.set_xticklabels([name.replace('.', '\n') for name in layer_names], rotation=45, ha='right')
        ax.set_ylabel('Error Variance')
        ax.set_title('Error Variance Across Layers')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, axis='y')
    
    # Mean correlation
    ax = axes[1]
    valid_mask = ~np.isnan(mean_corrs)
    if valid_mask.sum() > 0:
        ax.bar(range(n_layers), mean_corrs, alpha=0.7, color='orange')
        ax.set_xticks(range(n_layers))
        ax.set_xticklabels([name.replace('.', '\n') for name in layer_names], rotation=45, ha='right')
        ax.set_ylabel('Mean Correlation')
        ax.set_title('Error Correlation Across Layers')
        ax.set_ylim([-1, 1])
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3, axis='y')
    
    # Residual ratio
    ax = axes[2]
    valid_mask = ~np.isnan(residual_ratios)
    if valid_mask.sum() > 0:
        ax.bar(range(n_layers), residual_ratios, alpha=0.7, color='green')
        ax.set_xticks(range(n_layers))
        ax.set_xticklabels([name.replace('.', '\n') for name in layer_names], rotation=45, ha='right')
        ax.set_ylabel('Avg Residual Ratio')
        ax.set_title('Static ΔW Transferability')
        ax.axhline(y=0.5, color='r', linestyle='--', label='Threshold (0.5)')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'multi_layer_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Heatmap comparing layers
    metrics_data = {
        'Error Variance': error_vars,
        'Mean Correlation': mean_corrs,
        'Residual Ratio': residual_ratios,
    }
    
    # Normalize each metric for visualization
    df_metrics = pd.DataFrame(metrics_data, index=layer_names)
    df_metrics_norm = df_metrics.apply(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-12) if x.max() > x.min() else x,
        axis=0
    )
    
    plt.figure(figsize=(10, max(6, n_layers * 0.4)))
    sns.heatmap(df_metrics_norm.T, annot=True, fmt='.2f', cmap='RdYlGn',
               xticklabels=[name.replace('.', '\n') for name in layer_names],
               yticklabels=list(metrics_data.keys()),
               cbar_kws={'label': 'Normalized Value'})
    plt.title('Input Dependence Metrics Comparison Across Layers')
    plt.tight_layout()
    plt.savefig(output_dir / 'multi_layer_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nMulti-layer comparison visualizations saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Diagnose input dependence of hardware errors')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--layer', type=str, default=None, help='Specific layer name (optional, deprecated: use --layers)')
    parser.add_argument('--layers', type=str, nargs='+', default=None, help='List of layer names to analyze (e.g., --layers conv1 layer1.0.conv2)')
    parser.add_argument('--num_samples', type=int, default=20, help='Number of input batches to sample')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for each sample')
    parser.add_argument('--t', type=int, default=0, help='Time/cycle index for drift')
    parser.add_argument('--output_dir', type=str, default='./diagnosis_results', help='Output directory')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    
    args = parser.parse_args()
    
    # Handle backward compatibility: --layer -> --layers
    if args.layer is not None and args.layers is None:
        args.layers = [args.layer]
        print("Warning: --layer is deprecated, use --layers instead")
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set seed
    set_seed(args.seed if args.seed is not None else config.get('seed', 42))
    
    # Device
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Using device: {device}")
    
    # Data loader
    _, val_loader, test_loader = get_dataloaders(
        dataset_name=config['dataset'],
        data_root=config['data_root'],
        batch_size=args.batch_size,
        num_workers=config.get('num_workers', 4),
        val_split=0.0,
        seed=config.get('seed'),
    )
    eval_loader = val_loader if val_loader else test_loader
    
    # Model
    model = get_model(
        name=config['model_name'],
        num_classes=config.get('num_classes', 10),
    )
    
    # Device model
    device_model = None
    if config['experiment']['mode'] != 'baseline':
        memristor_config = config['memristor']
        device_model = MemristorDeviceModel(
            G_min=float(memristor_config['G_min']),
            G_max=float(memristor_config['G_max']),
            weight_clip=memristor_config['weight_clip'],
            variability_sigma=float(memristor_config['variability_sigma']),
            read_noise_sigma=float(memristor_config['read_noise_sigma']),
            drift_alpha=float(memristor_config['drift_alpha']),
            stuck_ratio=float(memristor_config['stuck_ratio']),
            stuck_low_prob=float(memristor_config['stuck_low_prob']),
            ir_drop_mode=str(memristor_config.get('ir_drop_mode', 'none')),
            ir_drop_beta=float(memristor_config.get('ir_drop_beta', 0.01)),
            ir_drop_gamma=float(memristor_config.get('ir_drop_gamma', 0.35)),
            ir_drop_scaling=float(memristor_config.get('ir_drop_scaling', 1.0)),
            ir_drop_eta=float(memristor_config.get('ir_drop_eta', 1.0)),
            ir_drop_cap=float(memristor_config.get('ir_drop_cap', 0.10)),
            ir_drop_norm=str(memristor_config.get('ir_drop_norm', 'mean')),
            drift_time_mode='fixed',
            drift_time_fixed=args.t,
            array_size=int(memristor_config.get('array_size', 128)),
            adc_bits=int(memristor_config.get('adc_bits', 8)),
            enable_adc=bool(memristor_config.get('enable_adc', True)),
            adc_add_noise=bool(memristor_config.get('adc_add_noise', False)),
        )
        model = wrap_model_with_memristor(model, device_model)
    
    model = model.to(device)
    
    # Load checkpoint
    checkpoint = load_checkpoint(args.checkpoint, model, device=device)
    print(f"Loaded checkpoint from {args.checkpoint}")
    
    # Check if analyzing multiple layers
    if args.layers and len(args.layers) > 1:
        # Multi-layer analysis
        print(f"\n{'='*60}")
        print("MULTI-LAYER INPUT DEPENDENCE ANALYSIS")
        print(f"{'='*60}")
        
        all_results = analyze_multiple_layers(
            model, args.layers, eval_loader, device_model, device, config, args
        )
        
        # Print summary table
        print("\n" + "="*60)
        print("SUMMARY TABLE")
        print("="*60)
        print(f"{'Layer':<30} {'Error Var':<15} {'Correlation':<15} {'Residual Ratio':<15} {'S_ℓ':<15} {'S_ℓ^abs':<15}")
        print("-" * 105)
        
        summary_data = []
        for layer_name, layer_data in all_results.items():
            results = layer_data['results']
            avg_residual = np.mean(results['static_deltaW_residual_ratios'][1:]) if len(results['static_deltaW_residual_ratios']) > 1 else np.nan
            s_ell = results.get('loss_sensitivity_weighted_error', np.nan)
            s_ell_abs = results.get('loss_sensitivity_weighted_error_abs', np.nan)
            summary_data.append({
                'layer': layer_name,
                'error_variance': results['error_variance'],
                'mean_correlation': results['mean_correlation'],
                'residual_ratio': avg_residual,
                'loss_sensitivity_weighted_error': s_ell,
                'loss_sensitivity_weighted_error_abs': s_ell_abs,
            })
            s_ell_str = f"{s_ell:.6e}" if not np.isnan(s_ell) else "N/A"
            s_ell_abs_str = f"{s_ell_abs:.6e}" if not np.isnan(s_ell_abs) else "N/A"
            print(f"{layer_name:<30} {results['error_variance']:<15.6e} {results['mean_correlation']:<15.4f} {avg_residual:<15.4f} {s_ell_str:<15} {s_ell_abs_str:<15}")
        
        # Save summary CSV
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_dir / 'multi_layer_summary.csv', index=False)
        print(f"\nSummary saved to {output_dir / 'multi_layer_summary.csv'}")
        
        # Generate comparison visualizations
        print(f"\nGenerating comparison visualizations...")
        visualize_multi_layer_results(all_results, output_dir)
        
        # Also generate individual visualizations for each layer
        for layer_name, layer_data in all_results.items():
            results = layer_data['results']
            eps_list = layer_data['eps_list']
            layer_inputs = layer_data['layer_inputs']
            visualize_results(results, layer_inputs, eps_list, output_dir, layer_name)
        
        print("\nDone!")
        
    else:
        # Single layer analysis (original behavior)
        layer_name_arg = args.layers[0] if args.layers else args.layer
        layer_module, layer_name, W = select_representative_layer(model, layer_name_arg)
        
        # Sample inputs and targets
        print(f"\nSampling {args.num_samples} input batches...")
        model_inputs = []
        model_targets = []
        count = 0
        for data, target in eval_loader:
            model_inputs.append(data[:args.batch_size].to(device))
            model_targets.append(target[:args.batch_size].to(device))
            count += 1
            if count >= args.num_samples:
                break
        
        # Extract layer inputs
        layer_inputs = []
        is_linear = isinstance(layer_module, (nn.Linear, MemristorLinear, LearnedMappingMemristorLinear))
        is_conv = isinstance(layer_module, (nn.Conv2d, MemristorConv2d, LearnedMappingMemristorConv2d))
        
        if is_linear:
            for inp in model_inputs:
                x_layer_list = extract_layer_inputs(model, layer_name, [inp], device)
                if x_layer_list:
                    x_layer = x_layer_list[0]
                    if x_layer.dim() > 2:
                        x_layer = x_layer.view(x_layer.size(0), -1)
                    layer_inputs.append(x_layer)
                else:
                    print(f"Warning: Could not extract input for layer {layer_name}, using model input")
                    if inp.dim() > 2:
                        inp_flat = inp.view(inp.size(0), -1)
                    else:
                        inp_flat = inp
                    layer_inputs.append(inp_flat)
        elif is_conv:
            print(f"Note: Analyzing Conv2d layer {layer_name}")
            print(f"  Extracting 4D feature maps [B, C, H, W]")
            for inp in model_inputs:
                x_layer_list = extract_layer_inputs(model, layer_name, [inp], device)
                if x_layer_list:
                    x_layer = x_layer_list[0]
                    if x_layer.dim() == 4:
                        layer_inputs.append(x_layer)
                    elif x_layer.dim() == 2:
                        print(f"Warning: Input to conv layer is 2D, attempting to reshape")
                        if hasattr(layer_module, 'in_channels'):
                            in_ch = layer_module.in_channels
                            B = x_layer.size(0)
                            total_elements = x_layer.size(1)
                            spatial_size = total_elements // in_ch
                            H = W = int(spatial_size ** 0.5)
                            
                            if H * W * in_ch == total_elements and H > 0 and W > 0:
                                x_layer_4d = x_layer.view(B, in_ch, H, W)
                                layer_inputs.append(x_layer_4d)
                                print(f"  Reshaped to [B={B}, C={in_ch}, H={H}, W={W}]")
                            else:
                                import math
                                factors = []
                                for i in range(1, int(math.sqrt(spatial_size)) + 1):
                                    if spatial_size % i == 0:
                                        factors.append((i, spatial_size // i))
                                
                                found = False
                                for h, w in factors:
                                    if h * w * in_ch == total_elements:
                                        x_layer_4d = x_layer.view(B, in_ch, h, w)
                                        layer_inputs.append(x_layer_4d)
                                        print(f"  Reshaped to [B={B}, C={in_ch}, H={h}, W={w}]")
                                        found = True
                                        break
                                
                                if not found:
                                    print(f"  Cannot reshape, skipping")
                                    continue
                        else:
                            print(f"  Cannot determine in_channels, skipping")
                            continue
                    else:
                        print(f"Warning: Unexpected input dimension {x_layer.dim()}")
                        continue
                else:
                    print(f"Warning: Could not extract input for conv layer {layer_name}")
                    if inp.dim() == 4:
                        layer_inputs.append(inp)
                    else:
                        print(f"  Model input is not 4D, skipping")
                        continue
        else:
            raise NotImplementedError(f"Layer type {type(layer_module)} not supported.")
        
        # Analyze
        print("\nAnalyzing input dependence...")
        
        # Create criterion for loss-sensitivity computation
        criterion = nn.CrossEntropyLoss()
        
        if isinstance(layer_module, (nn.Conv2d, MemristorConv2d, LearnedMappingMemristorConv2d)):
            results, eps_list = analyze_conv2d_layer(
                layer_inputs, W, layer_module, device_model, t=args.t, seed=args.seed,
                model=model,
                criterion=criterion,
                inputs_list=model_inputs,
                targets_list=model_targets,
                device=device,
            )
        else:
            results, eps_list = analyze_linear_layer(
                layer_inputs, W, device_model, t=args.t, seed=args.seed,
                model=model,
                criterion=criterion,
                inputs_list=model_inputs,
                targets_list=model_targets,
                layer_module=layer_module,
                device=device,
            )
        
        # Print results
        print("\n" + "="*60)
        print("DIAGNOSTIC RESULTS")
        print("="*60)
        print(f"Layer: {layer_name}")
        print(f"Number of input batches: {results['num_inputs']}")
        print(f"\n(A) Error variance across inputs: {results['error_variance']:.6e}")
        print(f"(B) Mean error correlation: {results['mean_correlation']:.4f}")
        print(f"\n(C) Static ΔW transferability:")
        print(f"  Residual norms: {[f'{r:.6e}' for r in results['static_deltaW_residuals']]}")
        print(f"  Residual ratios: {[f'{r:.4f}' for r in results['static_deltaW_residual_ratios']]}")
        # Print loss-sensitivity weighted error if available
        if 'loss_sensitivity_weighted_error' in results and not np.isnan(results['loss_sensitivity_weighted_error']):
            print(f"\n(D) Loss-sensitivity weighted error:")
            print(f"  S_ℓ = E_x [⟨∇_{layer_name} L(x), e_ℓ(x)⟩]: {results['loss_sensitivity_weighted_error']:.6e}")
        if 'loss_sensitivity_weighted_error_abs' in results and not np.isnan(results['loss_sensitivity_weighted_error_abs']):
            print(f"  S_ℓ^abs = E_x [||∇_{layer_name} L(x) ⊙ e_ℓ(x)||]: {results['loss_sensitivity_weighted_error_abs']:.6e}")
        
        # Interpretation
        print("\n" + "="*60)
        print("INTERPRETATION")
        print("="*60)
        if results['error_variance'] > 1e-4:
            print(" High error variance → Strong input dependence")
        else:
            print(" Low error variance → Weak input dependence")
        
        if results['mean_correlation'] < 0.5:
            print(" Low correlation → Error structure changes with input")
        else:
            print(" High correlation → Error structure is consistent")
        
        avg_residual_ratio = np.mean(results['static_deltaW_residual_ratios'][1:])
        if avg_residual_ratio > 0.5:
            print(" High residual ratio → Static ΔW does NOT transfer well")
            print("  → Input-dependent calibration may be needed")
        else:
            print(" Low residual ratio → Static ΔW transfers well")
            print("  → Static weight-only mapping may be sufficient")
        
        # Visualize
        output_dir = Path(args.output_dir)
        print(f"\nGenerating visualizations...")
        visualize_results(results, layer_inputs, eps_list, output_dir, layer_name)
        
        print("\nDone!")


if __name__ == '__main__':
    main()
