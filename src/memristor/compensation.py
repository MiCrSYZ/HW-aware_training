import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Callable, Tuple
import logging
import numpy as np

from .device_model import MemristorDeviceModel


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


def _set_device_model_for_model(model: nn.Module, device_model: MemristorDeviceModel):
    """Temporarily replace device_model in all memristor layers (for float forward)."""
    # IMPORTANT: If model is a MemristorModel wrapper, we need to access base_model
    target_model = model
    if hasattr(model, 'base_model'):
        target_model = model.base_model
    
    for module in target_model.modules():
        if hasattr(module, 'device_model'):
            module.device_model = device_model


