"""
GRU-specific metrics collection utilities.

This module provides functions to collect tier-based metrics for GRU models,
similar to ViT metrics.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
import numpy as np


def get_gru_tier_name_from_param_name(param_name: str) -> Optional[str]:
    """
    Determine the tier name for a GRU parameter from its name.
    
    Tiers:
    - embedding: Embedding layer
    - gru_weight: GRU weight_ih and weight_hh
    - head: Classification head
    
    Args:
        param_name: Full parameter name (e.g., 'embedding.weight', 'gru.weight_ih_l0', 'head.weight')
        
    Returns:
        Tier name or None if not a tracked layer
    """
    # Embedding layer
    if 'embedding' in param_name:
        return 'embedding'
    
    # GRU weights
    if 'gru' in param_name and ('weight_ih' in param_name or 'weight_hh' in param_name):
        return 'gru_weight'
    
    # Classification head
    if 'head' in param_name:
        return 'head'
    
    return None


def collect_gradient_norms_by_tier(
    model: nn.Module,
    tier_names: List[str] = ['embedding', 'gru_weight', 'head']
) -> Dict[str, float]:
    """
    Collect gradient norms grouped by tier.
    
    Args:
        model: Model to collect gradients from
        tier_names: List of tier names to collect
        
    Returns:
        Dictionary mapping tier names to gradient norms
    """
    tier_grads = {tier: [] for tier in tier_names}
    
    # Get base model if wrapped
    base_model = model
    if hasattr(model, 'base_model'):
        base_model = model.base_model
    
    for name, param in base_model.named_parameters():
        if param.grad is not None:
            tier = get_gru_tier_name_from_param_name(name)
            
            if tier and tier in tier_grads:
                tier_grads[tier].append(param.grad.flatten())
    
    # Compute norms for each tier
    tier_norms = {}
    for tier, grads in tier_grads.items():
        if grads:
            all_grads = torch.cat(grads)
            tier_norms[tier] = all_grads.norm().item()
        else:
            tier_norms[tier] = 0.0
    
    return tier_norms


def collect_activation_stats(
    model: nn.Module,
    hook_registry: Dict[str, List[torch.Tensor]],
    tier_names: List[str] = ['embedding', 'gru_weight', 'head']
) -> Dict[str, Dict[str, float]]:
    """
    Collect activation statistics for tracked layers.
    
    Args:
        model: Model
        hook_registry: Dictionary mapping layer names to their activation tensors
        tier_names: List of tier names
        
    Returns:
        Dictionary mapping tier names to statistics dictionaries
    """
    tier_stats = {tier: {
        'mean': 0.0,
        'std': 0.0,
        'p99': 0.0,
        'max': 0.0,
        'nan_count': 0,
        'inf_count': 0,
        'total_elements': 0
    } for tier in tier_names}
    
    # Get base model if wrapped
    base_model = model
    if hasattr(model, 'base_model'):
        base_model = model.base_model
    
    # Group activations by tier
    tier_activations = {tier: [] for tier in tier_names}
    
    for layer_name, activations in hook_registry.items():
        tier = None
        # Determine tier from layer name
        if 'embedding' in layer_name:
            tier = 'embedding'
        elif 'gru' in layer_name:
            tier = 'gru_weight'
        elif 'head' in layer_name:
            tier = 'head'
        
        if tier and tier in tier_activations:
            tier_activations[tier].extend(activations)
    
    # Compute statistics for each tier
    for tier, acts in tier_activations.items():
        if acts:
            # Concatenate all activations
            all_acts = torch.cat([a.flatten() for a in acts])
            
            # Filter out NaN and Inf
            valid_acts = all_acts[~torch.isnan(all_acts) & ~torch.isinf(all_acts)]
            
            if len(valid_acts) > 0:
                tier_stats[tier]['mean'] = valid_acts.mean().item()
                tier_stats[tier]['std'] = valid_acts.std().item()
                tier_stats[tier]['p99'] = torch.quantile(valid_acts, 0.99).item()
                tier_stats[tier]['max'] = valid_acts.max().item()
                tier_stats[tier]['nan_count'] = torch.isnan(all_acts).sum().item()
                tier_stats[tier]['inf_count'] = torch.isinf(all_acts).sum().item()
                tier_stats[tier]['total_elements'] = all_acts.numel()
            else:
                # All NaN/Inf
                tier_stats[tier]['nan_count'] = all_acts.numel()
                tier_stats[tier]['total_elements'] = all_acts.numel()
    
    return tier_stats


def compute_logit_margin(logits: torch.Tensor) -> Dict[str, float]:
    """
    Compute logit margin statistics (top1 - top2).
    
    Args:
        logits: Logit tensor [batch_size, num_classes]
        
    Returns:
        Dictionary with margin statistics
    """
    if logits.dim() != 2:
        return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
    
    # Get top-2 predictions
    top2_values, _ = torch.topk(logits, k=2, dim=1)
    top1 = top2_values[:, 0]
    top2 = top2_values[:, 1]
    
    # Compute margin
    margins = top1 - top2
    
    # Filter out NaN/Inf
    valid_margins = margins[~torch.isnan(margins) & ~torch.isinf(margins)]
    
    if len(valid_margins) > 0:
        return {
            'mean': valid_margins.mean().item(),
            'std': valid_margins.std().item(),
            'min': valid_margins.min().item(),
            'max': valid_margins.max().item(),
        }
    else:
        return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}


def register_activation_hooks(
    model: nn.Module,
    layer_names: Optional[List[str]] = None
) -> Tuple[Dict[str, List[torch.Tensor]], List[Any]]:
    """
    Register forward hooks to capture activations from specified layers.
    
    Args:
        model: Model to register hooks on
        layer_names: List of layer names to track. If None, tracks key GRU layers.
        
    Returns:
        Tuple of (hook_registry dict, hook_handles list)
    """
    hook_registry = {}
    hook_handles = []
    
    # Get base model if wrapped
    base_model = model
    if hasattr(model, 'base_model'):
        base_model = model.base_model
    
    # Default layers to track for GRU
    if layer_names is None:
        layer_names = []
        for name, module in base_model.named_modules():
            # Track embedding output
            if 'embedding' in name and isinstance(module, nn.Embedding):
                layer_names.append(name)
            # Track GRU outputs (first and last layer)
            elif 'gru' in name and isinstance(module, nn.GRU):
                layer_names.append(name)
            # Track head output
            elif 'head' in name and isinstance(module, nn.Linear):
                layer_names.append(name)
    
    def make_hook_fn(layer_name: str):
        def hook_fn(module, input, output):
            if layer_name not in hook_registry:
                hook_registry[layer_name] = []
            # Store activation (detach to avoid memory issues)
            if isinstance(output, torch.Tensor):
                hook_registry[layer_name].append(output.detach().clone())
            elif isinstance(output, tuple):
                # For GRU, output is (output, hidden)
                hook_registry[layer_name].append(output[0].detach().clone())
        return hook_fn
    
    # Register hooks
    for layer_name in layer_names:
        for name, module in base_model.named_modules():
            if name == layer_name:
                handle = module.register_forward_hook(make_hook_fn(layer_name))
                hook_handles.append(handle)
                break
    
    return hook_registry, hook_handles


def compute_update_norm_by_tier(
    model: nn.Module,
    weights_before: List[torch.Tensor],
    tier_names: List[str] = ['embedding', 'gru_weight', 'head']
) -> Dict[str, float]:
    """
    Compute weight update norms grouped by tier.
    
    Args:
        model: Model
        weights_before: List of weight tensors before update
        tier_names: List of tier names
        
    Returns:
        Dictionary mapping tier names to update norms
    """
    tier_updates = {tier: [] for tier in tier_names}
    
    # Get base model if wrapped
    base_model = model
    if hasattr(model, 'base_model'):
        base_model = model.base_model
    
    param_idx = 0
    for name, param in base_model.named_parameters():
        if param.requires_grad and param_idx < len(weights_before):
            tier = get_gru_tier_name_from_param_name(name)
            
            if tier and tier in tier_updates:
                update = (param.data - weights_before[param_idx]).flatten()
                tier_updates[tier].append(update)
            
            param_idx += 1
    
    # Compute norms for each tier
    tier_norms = {}
    for tier, updates in tier_updates.items():
        if updates:
            all_updates = torch.cat(updates)
            tier_norms[tier] = all_updates.norm().item()
        else:
            tier_norms[tier] = 0.0
    
    return tier_norms
