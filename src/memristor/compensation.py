import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Callable, Tuple
import logging
import numpy as np

from .device_model import MemristorDeviceModel

try:
    from .learned_weight_mapping import (
        WeightMappingNet,
        train_weight_mapping,
    )
except ImportError:
    from src.memristor.learned_weight_mapping import (
        WeightMappingNet,
        train_weight_mapping,
    )

logger = logging.getLogger(__name__)


def hardware_aware_training(
    model: nn.Module,
    device_model: MemristorDeviceModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    num_epochs: int = 1,
    t_step: int = 0,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Hardware-aware training (HAT) with non-idealities injected during forward pass.
    
    This function trains a model where device non-idealities are applied during
    the forward pass, making the model robust to memristor imperfections.
    
    Args:
        model: PyTorch model (should use MemristorLinear/MemristorConv2d layers)
        device_model: MemristorDeviceModel instance
        dataloader: Training data loader
        optimizer: Optimizer instance
        criterion: Loss function
        device: Device to run on
        num_epochs: Number of training epochs
        t_step: Starting time step for drift
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with training statistics
    """
    model.train()
    total_loss = 0.0
    total_samples = 0
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_samples = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with non-idealities (t increases with each batch)
            current_t = t_step + epoch * len(dataloader) + batch_idx
            output = model(data, t=current_t)
            
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * data.size(0)
            epoch_samples += data.size(0)
        
        avg_loss = epoch_loss / epoch_samples
        total_loss += avg_loss
        total_samples += 1
        
        logger.info(f"HAT epoch {epoch+1}/{num_epochs}, loss: {avg_loss:.4f}")
    
    return {
        'avg_loss': total_loss / total_samples,
        'num_epochs': num_epochs,
    }


def _set_mapping_net_for_model(model: nn.Module, mapping_net: Optional[nn.Module]):
    """Set mapping network for all memristor layers in the model."""
    # IMPORTANT: If model is a MemristorModel wrapper, we need to access base_model
    target_model = model
    if hasattr(model, 'base_model'):
        target_model = model.base_model
    
    for module in target_model.modules():
        if hasattr(module, 'set_learned_mapping'):
            module.set_learned_mapping(mapping_net)


def _set_device_model_for_model(model: nn.Module, device_model: MemristorDeviceModel):
    """Temporarily replace device_model in all memristor layers (for float forward)."""
    # IMPORTANT: If model is a MemristorModel wrapper, we need to access base_model
    target_model = model
    if hasattr(model, 'base_model'):
        target_model = model.base_model
    
    for module in target_model.modules():
        if hasattr(module, 'device_model'):
            module.device_model = device_model


def _forward_with_learned_mapping(
    model: nn.Module,
    x: torch.Tensor,
    device_model: MemristorDeviceModel,
    mapping_net: nn.Module,
    t: int = 0,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Forward pass with learned mapping applied to each layer.
    
    The mapping network is already set in each layer via set_learned_mapping(),
    so we just need to run the forward pass normally.
    
    Args:
        model: Model with memristor layers
        x: Input tensor
        device_model: MemristorDeviceModel instance (not used, kept for compatibility)
        mapping_net: LearnedMappingNet instance (not used, kept for compatibility)
        t: Time/cycle index for drift
        seed: Random seed for non-idealities (to ensure different noise from HAT forward)
        
    Returns:
        Output tensor
    """
    try:
        if seed is not None:
            output = model(x, t=t, seed=seed)
        else:
            output = model(x, t=t)
    except TypeError:
        output = model(x)
    
    return output


def _evaluate_with_learned_mapping(
    model: nn.Module,
    val_loader: DataLoader,
    device_model: MemristorDeviceModel,
    mapping_net: nn.Module,
    criterion: nn.Module,
    device: torch.device,
    t: int = 0,
) -> tuple:
    """Evaluate model with learned mapping."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            
            output = _forward_with_learned_mapping(
                model, data, device_model, mapping_net, t=t
            )
            
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)
            
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += data.size(0)
    
    avg_loss = total_loss / total if total > 0 else float('inf')
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    
    return accuracy, avg_loss


