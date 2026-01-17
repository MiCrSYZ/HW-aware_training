"""
One-layer Learned Mapping Sensitivity Experiment

This diagnostic experiment evaluates whether learned mapping effectiveness
is layer-specific and correlated with input-dependent hardware errors.

For each layer:
1. Enable learned mapping ONLY on that layer
2. Disable learned mapping on all other layers
3. Evaluate test accuracy
4. Compute input-dependent error metrics
5. Compare with HAT-only and full mapping baselines
"""

import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from tqdm import tqdm

from ..utils.seeds import set_seed
from ..data.dataset import get_dataloaders
from ..models.model_zoo import get_model, wrap_model_with_memristor
from ..memristor.device_model import MemristorDeviceModel
from ..utils.hardware_error_diagnosis import diagnose_input_dependence
from ..utils.checkpoint import load_checkpoint
from ..utils.metrics import accuracy

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


def get_all_mappable_layers(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """
    Get all layers that support learned mapping.
    
    Returns:
        List of (layer_name, layer_module) tuples
    """
    layers = []
    
    # Get base model if wrapped
    base_model = model
    if hasattr(model, 'base_model'):
        base_model = model.base_model
    
    for name, module in base_model.named_modules():
        # Check if it's a mappable layer
        is_mappable = isinstance(module, (
            nn.Linear, nn.Conv2d,
            MemristorLinear, MemristorConv2d,
            LearnedMappingMemristorLinear, LearnedMappingMemristorConv2d
        ))
        
        if is_mappable and hasattr(module, 'set_learned_mapping'):
            layers.append((name, module))
    
    return layers


def set_mapping_for_layer(
    model: nn.Module,
    target_layer_name: str,
    mapping_net: Optional[nn.Module],
) -> None:
    """
    Set learned mapping ONLY for the target layer, disable for all others.
    
    Args:
        model: Model to configure
        target_layer_name: Name of layer to enable mapping
        mapping_net: Mapping network (None to disable)
    """
    # Get base model if wrapped
    base_model = model
    if hasattr(model, 'base_model'):
        base_model = model.base_model
    
    # Disable mapping for all layers first
    for name, module in base_model.named_modules():
        if hasattr(module, 'set_learned_mapping'):
            if name == target_layer_name:
                # Enable mapping for target layer
                module.set_learned_mapping(mapping_net)
            else:
                # Disable mapping for all other layers
                module.set_learned_mapping(None)


def evaluate_accuracy(
    model: nn.Module,
    test_loader,
    device: torch.device,
    t: int = 0,
) -> float:
    """
    Evaluate model accuracy on test set.
    
    Returns:
        Top-1 accuracy percentage
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            try:
                output = model(data, t=t)
            except TypeError:
                output = model(data)
            
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    acc = 100.0 * correct / total if total > 0 else 0.0
    return acc


def compute_layer_input_dependence(
    model: nn.Module,
    layer_name: str,
    layer_module: nn.Module,
    data_loader,  # Can be val_loader or test_loader
    device_model: MemristorDeviceModel,
    device: torch.device,
    num_samples: int = 10,
    batch_size: int = 32,
    t: int = 0,
) -> Dict[str, float]:
    """
    Compute input-dependent error metrics for a specific layer.
    
    Args:
        data_loader: Data loader for sampling inputs (val_loader or test_loader)
    
    Returns:
        Dictionary with error variance, correlation, and transferability metrics
    """
    from ..utils.hardware_error_diagnosis import hardware_error
    
    # Check if data_loader is None
    if data_loader is None:
        return {
            'error_variance': np.nan,
            'mean_correlation': np.nan,
            'static_deltaW_residual_ratio_mean': np.nan,
        }
    
    # Sample inputs and targets
    inputs = []
    targets_list = []
    count = 0
    for data, target in data_loader:
        inputs.append(data[:batch_size].to(device))
        targets_list.append(target[:batch_size].to(device))
        count += 1
        if count >= num_samples:
            break
    
    if not inputs:
        return {
            'error_variance': np.nan,
            'mean_correlation': np.nan,
            'static_deltaW_residual_ratio_mean': np.nan,
            'loss_sensitivity_weighted_error': np.nan,
            'loss_sensitivity_weighted_error_abs': np.nan,
        }
    
    # Extract layer inputs and outputs using forward hooks
    layer_inputs = []
    layer_outputs_ideal = []
    layer_outputs_hw = []
    
    def hook_input_fn(module, input):
        if isinstance(input, tuple):
            layer_inputs.append(input[0].detach().clone())
        else:
            layer_inputs.append(input.detach().clone())
    
    def hook_output_fn(module, input, output):
        # Capture layer output (ideal computation)
        layer_outputs_ideal.append(output.detach().clone())
    
    # Get base model if wrapped
    base_model = model
    if hasattr(model, 'base_model'):
        base_model = model.base_model
    
    # Register hooks
    hook_input = None
    hook_output = None
    for name, module in base_model.named_modules():
        if name == layer_name:
            hook_input = module.register_forward_pre_hook(hook_input_fn)
            hook_output = module.register_forward_hook(hook_output_fn)
            break
    
    if hook_input is None or hook_output is None:
        return {
            'error_variance': np.nan,
            'mean_correlation': np.nan,
            'static_deltaW_residual_ratio_mean': np.nan,
            'loss_sensitivity_weighted_error': np.nan,
            'loss_sensitivity_weighted_error_abs': np.nan,
        }
    
    # Run forward passes to get ideal outputs
    model.eval()
    with torch.no_grad():
        for inp in inputs:
            try:
                _ = model(inp, t=t)
            except TypeError:
                _ = model(inp)
    
    # Also compute hardware outputs
    # We need to run forward pass with hardware computation enabled
    # For now, we'll compute h_hw from h_ideal + eps
    # But first, let's get the ideal outputs
    
    hook_input.remove()
    hook_output.remove()
    
    if not layer_inputs:
        return {
            'error_variance': np.nan,
            'mean_correlation': np.nan,
            'static_deltaW_residual_ratio_mean': np.nan,
        }
    
    # Get layer weight
    W = layer_module.weight.data.clone()
    
    # Process inputs based on layer type
    is_linear = isinstance(layer_module, (
        nn.Linear, MemristorLinear, LearnedMappingMemristorLinear
    ))
    is_conv = isinstance(layer_module, (
        nn.Conv2d, MemristorConv2d, LearnedMappingMemristorConv2d
    ))
    
    x_processed = []
    eps_list = []
    h_ideal_list = []
    h_hw_list = []
    
    for i, x_layer in enumerate(layer_inputs):
        # Get ideal output for this sample
        h_ideal = None
        if i < len(layer_outputs_ideal):
            h_ideal = layer_outputs_ideal[i]
        
        if is_linear:
            # Flatten if needed
            if x_layer.dim() > 2:
                x_layer = x_layer.view(x_layer.size(0), -1)
            x_processed.append(x_layer)
            
            # Compute hardware error
            eps = hardware_error(x_layer, W, device_model, t=t, seed=None)
            eps_list.append(eps)
            
            # Get ideal and hardware outputs
            if h_ideal is not None:
                # Flatten if needed
                if h_ideal.dim() > 2:
                    h_ideal = h_ideal.view(h_ideal.size(0), -1)
                # Ensure dimensions match
                if h_ideal.shape[1] == eps.shape[1]:
                    h_ideal_list.append(h_ideal)
                    # Hardware output = ideal + error
                    h_hw = h_ideal + eps
                    h_hw_list.append(h_hw)
        
        elif is_conv:
            # For Conv2d, we need to unfold
            import torch.nn.functional as F
            
            # Get conv parameters
            if hasattr(layer_module, 'kernel_size'):
                kernel_size = layer_module.kernel_size
                if isinstance(kernel_size, tuple):
                    k_h, k_w = kernel_size
                else:
                    k_h = k_w = kernel_size
            else:
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
            
            # Ensure 4D
            if x_layer.dim() != 4:
                continue
            
            # Unfold and flatten
            x_unfold = F.unfold(x_layer, kernel_size=(k_h, k_w), padding=(pad_h, pad_w), stride=(stride_h, stride_w))
            x_flat = x_unfold.transpose(1, 2).reshape(-1, x_unfold.size(1))
            
            W_flat = W.view(W.size(0), -1)
            
            x_processed.append(x_flat)
            eps = hardware_error(x_flat, W_flat, device_model, t=t, seed=None)
            eps_list.append(eps)
            
            # For conv layers, reshape outputs to match eps dimensions
            if h_ideal is not None:
                h_ideal_flat = h_ideal.view(h_ideal.size(0), -1)
                # Match dimensions with eps
                if h_ideal_flat.shape[1] == eps.shape[1]:
                    h_ideal_list.append(h_ideal_flat)
                    h_hw = h_ideal_flat + eps
                    h_hw_list.append(h_hw)
    
    if not eps_list:
        return {
            'error_variance': np.nan,
            'mean_correlation': np.nan,
            'static_deltaW_residual_ratio_mean': np.nan,
            'loss_sensitivity_weighted_error': np.nan,
            'loss_sensitivity_weighted_error_abs': np.nan,
        }
    
    # Prepare for loss-sensitivity computation
    # Create criterion (CrossEntropyLoss)
    criterion = nn.CrossEntropyLoss()
    
    # Ensure inputs_list and targets_list match the length of eps_list
    # (Some samples may have been skipped for conv layers)
    inputs_matched = inputs[:len(eps_list)]
    targets_matched = targets_list[:len(eps_list)]
    
    # Ensure h_ideal_list and h_hw_list match eps_list length
    # If they don't match, we can't compute loss-sensitivity
    can_compute_sensitivity = (
        len(h_ideal_list) == len(eps_list) and
        len(h_hw_list) == len(eps_list) and
        len(h_ideal_list) > 0
    )
    
    # Run diagnosis
    try:
        results = diagnose_input_dependence(
            x_processed, eps_list, ref_idx=0,
            model=model if can_compute_sensitivity else None,
            criterion=criterion if can_compute_sensitivity else None,
            inputs_list=inputs_matched if can_compute_sensitivity else None,
            targets_list=targets_matched if can_compute_sensitivity else None,
            h_ideal_list=h_ideal_list if can_compute_sensitivity else None,
            h_hw_list=h_hw_list if can_compute_sensitivity else None,
            layer_module=layer_module if can_compute_sensitivity else None,
            device=device if can_compute_sensitivity else None,
        )
        
        return {
            'error_variance': results['error_variance'],
            'mean_correlation': results['mean_correlation'],
            'static_deltaW_residual_ratio_mean': np.mean(results['static_deltaW_residual_ratios'][1:]) if len(results['static_deltaW_residual_ratios']) > 1 else np.nan,
            'loss_sensitivity_weighted_error': results.get('loss_sensitivity_weighted_error', np.nan),
            'loss_sensitivity_weighted_error_abs': results.get('loss_sensitivity_weighted_error_abs', np.nan),
        }
    except Exception as e:
        print(f"Warning: Error computing input dependence for {layer_name}: {e}")
        return {
            'error_variance': np.nan,
            'mean_correlation': np.nan,
            'static_deltaW_residual_ratio_mean': np.nan,
            'loss_sensitivity_weighted_error': np.nan,
            'loss_sensitivity_weighted_error_abs': np.nan,
        }


def run_one_layer_sensitivity(
    config_path: str,
    checkpoint_path: str,
    mapping_net_path: Optional[str],
    output_dir: str,
    layer_list: Optional[List[str]] = None,
    num_samples: int = 10,
    t: int = 0,
    compute_input_dependence: bool = True,
) -> pd.DataFrame:
    """
    Run one-layer sensitivity experiment.
    
    Args:
        config_path: Path to config YAML
        checkpoint_path: Path to model checkpoint
        mapping_net_path: Path to mapping net checkpoint (optional)
        output_dir: Output directory for results
        layer_list: List of layer names to test (None = test all)
        num_samples: Number of input samples for input dependence analysis
        t: Time/cycle index for drift
        compute_input_dependence: Whether to compute input dependence metrics
        
    Returns:
        DataFrame with results
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set seed
    set_seed(config.get('seed', 42))
    
    # Device
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Using device: {device}")
    
    # Data loaders
    _, val_loader, test_loader = get_dataloaders(
        dataset_name=config['dataset'],
        data_root=config['data_root'],
        batch_size=config.get('batch_size', 128),
        num_workers=config.get('num_workers', 4),
        val_split=0.0,
        seed=config.get('seed'),
    )
    
    # Use test_loader if val_loader is None
    error_analysis_loader = val_loader if val_loader is not None else test_loader
    
    # Model
    dataset_name = config['dataset'].lower()
    if dataset_name == 'mnist':
        in_channels = 1
        num_classes = config.get('num_classes', 10)
    elif dataset_name == 'cifar10':
        in_channels = 3
        num_classes = config.get('num_classes', 10)
    elif dataset_name == 'cifar100':
        in_channels = 3
        num_classes = config.get('num_classes', 100)
    else:
        in_channels = config.get('in_channels', 3)
        num_classes = config.get('num_classes', 10)
    
    model = get_model(
        name=config['model_name'],
        num_classes=num_classes,
        in_channels=in_channels,
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
            drift_time_fixed=t,
            array_size=int(memristor_config.get('array_size', 128)),
            adc_bits=int(memristor_config.get('adc_bits', 8)),
            enable_adc=bool(memristor_config.get('enable_adc', True)),
            adc_add_noise=bool(memristor_config.get('adc_add_noise', False)),
        )
        model = wrap_model_with_memristor(model, device_model, use_learned_mapping=True)
    
    model = model.to(device)
    
    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path, model, device=device)
    print(f"Loaded checkpoint from {checkpoint_path}")
    
    # Load mapping net
    # Priority: 1) from main checkpoint, 2) from mapping_net_path if provided
    mapping_net = None
    from ..memristor.learned_weight_mapping import WeightMappingNet
    
    # First, check if mapping_net_state_dict is in the main checkpoint
    if 'mapping_net_state_dict' in checkpoint:
        print("Found mapping_net_state_dict in main checkpoint")
        mapping_net = WeightMappingNet(
            hidden_dim=config['experiment'].get('mapping_hidden_dim', 32),
            alpha=float(config['experiment'].get('mapping_alpha', 0.5))
        )
        # Load with strict=False to handle missing 'scale' in old checkpoints
        state_dict = checkpoint['mapping_net_state_dict']
        try:
            mapping_net.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            # Old checkpoint without 'scale' parameter
            print("  Warning: Checkpoint appears to be from before scale parameter was added.")
            print("  Loading with strict=False (missing 'scale' will use default value 1.0)")
            mapping_net.load_state_dict(state_dict, strict=False)
            # Ensure scale is initialized (should already be 1.0 by default)
            if not hasattr(mapping_net, 'scale') or mapping_net.scale is None:
                mapping_net.scale = nn.Parameter(torch.ones(1, device=device))
                mapping_net._use_multiplicative_scale = False  # Old checkpoint = additive_only mode
        mapping_net = mapping_net.to(device)
        mapping_net.eval()
        print(f"Loaded mapping net from main checkpoint")
    else:
        print("Warning: No mapping_net_state_dict found in checkpoint.")
        print("  This checkpoint may not have been saved after post_train learned mapping.")
        print("  Full mapping baseline will be skipped.")
        print("  Available keys in checkpoint:", list(checkpoint.keys())[:10])
    
    # If mapping_net not loaded from main checkpoint, try separate file
    if mapping_net is None and mapping_net_path:
        # If not in main checkpoint, try loading from separate file
        print(f"Loading mapping net from separate file: {mapping_net_path}")
        mapping_net = WeightMappingNet(
            hidden_dim=config['experiment'].get('mapping_hidden_dim', 32),
            alpha=float(config['experiment'].get('mapping_alpha', 0.5))
        )
        mapping_checkpoint = torch.load(mapping_net_path, map_location=device)
        if 'mapping_net_state_dict' in mapping_checkpoint:
            state_dict = mapping_checkpoint['mapping_net_state_dict']
        else:
            # Try loading directly (in case it's a standalone mapping net checkpoint)
            state_dict = mapping_checkpoint
        
        try:
            # Try strict loading first
            mapping_net.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            # Old checkpoint without 'scale' parameter
            print("  Warning: Checkpoint appears to be from before scale parameter was added.")
            print("  Loading with strict=False (missing 'scale' will use default value 1.0)")
            try:
                mapping_net.load_state_dict(state_dict, strict=False)
                # Ensure scale is initialized
                if not hasattr(mapping_net, 'scale') or mapping_net.scale is None:
                    mapping_net.scale = nn.Parameter(torch.ones(1, device=device))
                    mapping_net._use_multiplicative_scale = False  # Old checkpoint = additive_only mode
            except RuntimeError as e:
                print(f"Warning: Could not load mapping net from {mapping_net_path}")
                print(f"Error: {e}")
                print("Continuing without mapping net (HAT-only baseline)")
                mapping_net = None
        
        if mapping_net is not None:
            mapping_net = mapping_net.to(device)
            mapping_net.eval()
            print(f"Loaded mapping net from {mapping_net_path}")
    else:
        print("No mapping net found. Will run HAT-only baseline.")
    
    # Get all mappable layers
    all_layers = get_all_mappable_layers(model)
    print(f"\nFound {len(all_layers)} mappable layers")
    
    # Filter layers if layer_list is provided
    if layer_list:
        layers_to_test = [(name, module) for name, module in all_layers if name in layer_list]
        if not layers_to_test:
            print(f"Warning: No layers from layer_list found. Testing all layers.")
            layers_to_test = all_layers
    else:
        layers_to_test = all_layers
    
    print(f"Testing {len(layers_to_test)} layers")
    
    # Baseline 1: HAT only (no mapping)
    print("\n" + "="*60)
    print("BASELINE 1: HAT only (no mapping)")
    print("="*60)
    set_mapping_for_layer(model, "", None)  # Disable all
    acc_hat_only = evaluate_accuracy(model, test_loader, device, t=t)
    print(f"HAT-only accuracy: {acc_hat_only:.2f}%")
    
    # Baseline 2: Full mapping (if mapping_net available)
    acc_full_mapping = None
    if mapping_net is not None:
        print("\n" + "="*60)
        print("BASELINE 2: Full mapping (all layers)")
        print("="*60)
        for name, module in all_layers:
            if hasattr(module, 'set_learned_mapping'):
                module.set_learned_mapping(mapping_net)
        acc_full_mapping = evaluate_accuracy(model, test_loader, device, t=t)
        print(f"Full mapping accuracy: {acc_full_mapping:.2f}%")
    else:
        print("\n" + "="*60)
        print("BASELINE 2: Full mapping (skipped - no mapping net available)")
        print("="*60)
    
    # Test each layer
    print("\n" + "="*60)
    print("ONE-LAYER SENSITIVITY TEST")
    print("="*60)
    
    results = []
    
    for layer_name, layer_module in tqdm(layers_to_test, desc="Testing layers"):
        print(f"\nTesting layer: {layer_name}")
        
        # Set mapping only for this layer
        if mapping_net is not None:
            set_mapping_for_layer(model, layer_name, mapping_net)
        else:
            set_mapping_for_layer(model, layer_name, None)
        
        # Evaluate accuracy
        acc_one_layer = evaluate_accuracy(model, test_loader, device, t=t)
        
        # Compute input dependence metrics
        error_metrics = {}
        if compute_input_dependence:
            print(f"  Computing input dependence metrics...")
            error_metrics = compute_layer_input_dependence(
                model, layer_name, layer_module, error_analysis_loader,
                device_model, device, num_samples=num_samples, t=t
            )
        
        # Store results
        result = {
            'layer': layer_name,
            'acc_one_layer': acc_one_layer,
            'delta_acc_vs_hat': acc_one_layer - acc_hat_only,
        }
        
        if acc_full_mapping is not None:
            result['delta_acc_vs_full'] = acc_one_layer - acc_full_mapping
        
        result.update(error_metrics)
        results.append(result)
        
        print(f"  Accuracy: {acc_one_layer:.2f}% (Δ vs HAT: {result['delta_acc_vs_hat']:+.2f}%)")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save CSV
    csv_path = output_path / 'one_layer_sensitivity_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    
    # Save JSON (convert numpy types to Python native types)
    def convert_to_native(obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_native(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        elif isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        else:
            return obj
    
    json_path = output_path / 'one_layer_sensitivity_results.json'
    with open(json_path, 'w') as f:
        json.dump({
            'baselines': {
                'hat_only': float(acc_hat_only),
                'full_mapping': float(acc_full_mapping) if acc_full_mapping is not None else None,
            },
            'results': convert_to_native(results),
        }, f, indent=2)
    print(f"JSON saved to {json_path}")
    
    # Generate visualizations
    if len(results) > 0:
        visualize_results(df, output_path, acc_hat_only, acc_full_mapping)
    
    return df


def visualize_results(
    df: pd.DataFrame,
    output_dir: Path,
    acc_hat_only: float,
    acc_full_mapping: Optional[float],
):
    """Generate visualization plots."""
    n_layers = len(df)
    
    # 1. Accuracy comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Accuracy vs layer index
    ax = axes[0, 0]
    ax.plot(df['acc_one_layer'].values, 'o-', linewidth=2, markersize=6, label='One-layer mapping')
    ax.axhline(y=acc_hat_only, color='r', linestyle='--', label='HAT only')
    if acc_full_mapping is not None:
        ax.axhline(y=acc_full_mapping, color='g', linestyle='--', label='Full mapping')
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy: One-layer Mapping')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Delta accuracy vs HAT
    ax = axes[0, 1]
    ax.plot(df['delta_acc_vs_hat'].values, 'o-', linewidth=2, markersize=6, color='orange')
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Δ Accuracy vs HAT (%)')
    ax.set_title('Accuracy Improvement vs HAT')
    ax.grid(True, alpha=0.3)
    
    # Error variance
    if 'error_variance' in df.columns and not df['error_variance'].isna().all():
        ax = axes[1, 0]
        valid_mask = ~df['error_variance'].isna()
        if valid_mask.sum() > 0:
            ax.plot(np.where(valid_mask)[0], df.loc[valid_mask, 'error_variance'].values, 
                   'o-', linewidth=2, markersize=6, color='purple')
            ax.set_xlabel('Layer Index')
            ax.set_ylabel('Error Variance')
            ax.set_title('Input-dependent Error Variance')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
    
    # Error correlation
    if 'mean_correlation' in df.columns and not df['mean_correlation'].isna().all():
        ax = axes[1, 1]
        valid_mask = ~df['mean_correlation'].isna()
        if valid_mask.sum() > 0:
            ax.plot(np.where(valid_mask)[0], df.loc[valid_mask, 'mean_correlation'].values,
                   'o-', linewidth=2, markersize=6, color='teal')
            ax.set_xlabel('Layer Index')
            ax.set_ylabel('Mean Correlation')
            ax.set_title('Error Correlation Across Inputs')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'one_layer_sensitivity_plots.png', dpi=150)
    plt.close()
    
    # 2. Heatmap
    if len(df) > 1:
        metrics_to_plot = ['acc_one_layer', 'delta_acc_vs_hat', 'error_variance', 'mean_correlation']
        metrics_to_plot = [m for m in metrics_to_plot if m in df.columns and not df[m].isna().all()]
        
        if metrics_to_plot:
            fig, ax = plt.subplots(figsize=(12, max(8, len(df) * 0.3)))
            heatmap_data = df[metrics_to_plot].T
            # Normalize each metric to [0, 1] for visualization
            heatmap_data_norm = heatmap_data.apply(
                lambda x: (x - x.min()) / (x.max() - x.min() + 1e-12) if x.max() > x.min() else x,
                axis=1
            )
            sns.heatmap(heatmap_data_norm, annot=True, fmt='.2f', cmap='RdYlGn', 
                       xticklabels=df['layer'].values, yticklabels=metrics_to_plot,
                       cbar_kws={'label': 'Normalized Value'})
            plt.title('One-layer Sensitivity Heatmap')
            plt.tight_layout()
            plt.savefig(output_dir / 'one_layer_sensitivity_heatmap.png', dpi=150)
            plt.close()
    
    print(f"Visualizations saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='One-layer Learned Mapping Sensitivity Experiment')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--mapping_net', type=str, default=None, help='Path to mapping net checkpoint')
    parser.add_argument('--output_dir', type=str, default='./one_layer_sensitivity_results', help='Output directory')
    parser.add_argument('--layers', type=str, nargs='+', default=None, help='Specific layers to test (default: all)')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of input samples for error analysis')
    parser.add_argument('--t', type=int, default=0, help='Time/cycle index for drift')
    parser.add_argument('--no_input_dependence', action='store_true', help='Skip input dependence computation')
    
    args = parser.parse_args()
    
    df = run_one_layer_sensitivity(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        mapping_net_path=args.mapping_net,
        output_dir=args.output_dir,
        layer_list=args.layers,
        num_samples=args.num_samples,
        t=args.t,
        compute_input_dependence=not args.no_input_dependence,
    )
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(df.to_string(index=False))
    print("\nDone!")


if __name__ == '__main__':
    main()
