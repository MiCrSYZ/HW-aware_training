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


def hybrid_compensation_train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device_model: MemristorDeviceModel,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    mapping_net: Optional[nn.Module] = None,
    mapping_optimizer: Optional[torch.optim.Optimizer] = None,
    stage1_epochs: int = 50,
    stage2_epochs: int = 50,
    mapping_lr: float = 1e-4,
    mapping_alpha: float = 0.5,
    mapping_lambda_reg: float = 1e-4,
) -> Dict[str, float]:
    """
    Hybrid compensation strategy with two-stage training.
    
    Stage 1: HAT training (mapping_net=None, train W to adapt to non-idealities)
    Stage 2: Freeze W, train only mapping_net to learn compensation ΔW
    
    Args:
        model: PyTorch model (should be memristor-wrapped)
        train_loader: Training data loader
        val_loader: Validation data loader
        device_model: MemristorDeviceModel instance
        criterion: Loss function
        optimizer: Optimizer for model parameters
        device: Device to run on
        epoch: Current epoch number
        mapping_net: WeightMappingNet instance (created if None)
        mapping_optimizer: Optimizer for mapping network (created if None)
        stage1_epochs: Number of epochs for Stage 1 (HAT training)
        stage2_epochs: Number of epochs for Stage 2 (mapping_net training)
        mapping_lr: Learning rate for mapping_net (default: 1e-4)
        mapping_alpha: Alpha parameter for mapping_net delta scaling (default: 0.5)
        mapping_lambda_reg: Regularization weight for mapping_net parameters ||Δ||^2 (default: 1e-4)
        
    Returns:
        Dictionary with training metrics
    """
    from ..utils.metrics import AverageMeter, accuracy
    
    # Determine current stage
    if epoch < stage1_epochs:
        current_stage = 1
        stage_epoch = epoch
        total_stage_epochs = stage1_epochs
    else:
        current_stage = 2
        stage_epoch = epoch - stage1_epochs
        total_stage_epochs = stage2_epochs
    
    # Create mapping network if not provided (needed for Stage 2)
    if mapping_net is None:
        mapping_net = WeightMappingNet(alpha=mapping_alpha).to(device)
    if mapping_optimizer is None:
        mapping_optimizer = torch.optim.Adam(mapping_net.parameters(), lr=mapping_lr)
    
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    
    if current_stage == 1:
        # Stage 1: HAT training (mapping_net=None, train W)
        logger.info(f"Hybrid Stage 1 (HAT): epoch {stage_epoch+1}/{total_stage_epochs}")
        
        # Ensure model parameters are trainable (in case they were frozen before)
        for param in model.parameters():
            param.requires_grad = True
        
        model.train()
        
        # Ensure mapping_net is disabled
        _set_mapping_net_for_model(model, None)
        target_model = model
        if hasattr(model, 'base_model'):
            target_model = model.base_model
        for module in target_model.modules():
            if hasattr(module, 'mapping_net'):
                module.mapping_net = None
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            t = epoch * len(train_loader) + batch_idx
            seed = None  # Let randomness vary naturally for HAT
            
            try:
                output = model(data, t=t, seed=seed)
            except TypeError:
                output = model(data)
            
            loss = criterion(output, target)
            
            # Check for NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"NaN/Inf detected in loss at batch {batch_idx}, skipping batch")
                continue
            
            loss.backward()
            
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                logger.warning(f"NaN/Inf gradients at batch {batch_idx}, skipping optimizer step")
                optimizer.zero_grad()
                continue
            
            optimizer.step()
            
            acc1 = accuracy(output, target, topk=(1,))[0]
            losses.update(loss.item(), data.size(0))
            top1.update(acc1, data.size(0))
        
        return {
            'loss': losses.avg,
            'acc1': top1.avg,
            'stage': 1,
        }
    
    else:
        # Stage 2: Freeze W, train only mapping_net
        logger.info(f"Hybrid Stage 2 (mapping_net): epoch {stage_epoch+1}/{total_stage_epochs}")
        
        # Freeze model parameters (only do this once at the start of Stage 2)
        if stage_epoch == 0:
            logger.info("Freezing model parameters for Stage 2")
            for param in model.parameters():
                param.requires_grad = False
        
        # Ensure mapping_net parameters require gradients
        for param in mapping_net.parameters():
            param.requires_grad = True
        
        # Set mapping network for all memristor layers
        _set_mapping_net_for_model(model, mapping_net)
        
        # Verify mapping_net is set correctly
        target_model = model
        if hasattr(model, 'base_model'):
            target_model = model.base_model
        
        num_layers_with_mapping = 0
        for module in target_model.modules():
            if hasattr(module, 'mapping_net'):
                mn = getattr(module, 'mapping_net', None)
                if mn is mapping_net:
                    num_layers_with_mapping += 1
        
        if num_layers_with_mapping == 0:
            logger.error("ERROR: No layers found with mapping_net set! Cannot train mapping_net.")
            logger.error("Make sure model was created with use_learned_mapping=True")
            raise RuntimeError("No layers with mapping_net found. Check model creation.")
        
        logger.info(f"Stage 2: Found {num_layers_with_mapping} layers with mapping_net")
        
        model.eval()  # Model is frozen, use eval mode
        mapping_net.train()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            mapping_optimizer.zero_grad()
            
            t = epoch * len(train_loader) + batch_idx
            seed = None  # Let randomness vary naturally
            
            # Forward pass with learned mapping (W is frozen, only mapping_net is trainable)
            try:
                output = model(data, t=t, seed=seed)
            except TypeError:
                output = model(data)
            
            # Task loss
            loss_task = criterion(output, target)
            
            # Regularization loss: ||Δ||^2 to keep mapping_net corrections small
            reg_loss = sum(p.pow(2).sum() for p in mapping_net.parameters())
            
            # Combined loss
            loss = loss_task + mapping_lambda_reg * reg_loss
            
            # Check for NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"NaN/Inf detected in loss at batch {batch_idx}, skipping batch")
                continue
            
            # Verify that loss requires grad (should be True if mapping_net participates in computation)
            if not loss.requires_grad:
                logger.error(f"ERROR: loss does not require grad at batch {batch_idx}!")
                logger.error("This means mapping_net is not participating in the computation graph.")
                logger.error("Checking mapping_net parameters:")
                for name, param in mapping_net.named_parameters():
                    logger.error(f"  {name}: requires_grad={param.requires_grad}")
                raise RuntimeError("Loss does not require grad. mapping_net may not be participating in computation.")
            
            loss.backward()
            
            grad_norm = torch.nn.utils.clip_grad_norm_(mapping_net.parameters(), max_norm=1.0)
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                logger.warning(f"NaN/Inf gradients in mapping_net at batch {batch_idx}, skipping optimizer step")
                mapping_optimizer.zero_grad()
                continue
            
            mapping_optimizer.step()
            
            acc1 = accuracy(output, target, topk=(1,))[0]
            losses.update(loss.item(), data.size(0))
            top1.update(acc1, data.size(0))
        
        return {
            'loss': losses.avg,
            'acc1': top1.avg,
            'stage': 2,
        }


def joint_hat_mapping_train(
    model: nn.Module,
    mapping_net: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device_model: MemristorDeviceModel,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    mapping_optimizer: Optional[torch.optim.Optimizer] = None,
    beta: float = 0.5,
    gamma: float = 1e-5,
    t_step: int = 0,
) -> Dict[str, float]:
    """
    Joint HAT + mapping_net training strategy.
    
    Strategy:
    - Train model and mapping_net simultaneously
    - Apply non-idealities during training
    - Loss = L_task + β * L_hw_mse + γ * L_reg
    
    Where:
    - L_task: Cross-entropy loss
    - L_hw_mse: MSE(f_float(x), f_hw(x)) - helps matching logits
    - L_reg: ||Δ||_2^2 - prevents large corrections
    
    Args:
        model: Model to train (should be memristor-wrapped)
        mapping_net: LearnedMappingNet instance
        train_loader: Training data loader
        val_loader: Validation data loader
        device_model: MemristorDeviceModel instance
        criterion: Loss function (cross-entropy)
        optimizer: Optimizer for model parameters
        device: Device to run on
        epoch: Current epoch number
        mapping_optimizer: Optimizer for mapping_net (created if None)
        beta: Weight for hw_mse loss (default: 0.5)
        gamma: Weight for regularization (default: 1e-5)
        t_step: Starting time step for drift
        
    Returns:
        Dictionary with training metrics
    """
    from ..utils.metrics import AverageMeter, accuracy
    
    model.train()
    mapping_net.train()
    
    # Create mapping optimizer if not provided
    if mapping_optimizer is None:
        mapping_optimizer = torch.optim.Adam(mapping_net.parameters(), lr=1e-4)
    
    # Set mapping network for all memristor layers
    _set_mapping_net_for_model(model, mapping_net)
    
    losses = AverageMeter('Loss', ':.4f')
    task_losses = AverageMeter('TaskLoss', ':.4f')
    hw_losses = AverageMeter('HwMSE', ':.4e')
    reg_losses = AverageMeter('RegLoss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        mapping_optimizer.zero_grad()
        
        t = t_step + epoch * len(train_loader) + batch_idx
        
        # Forward pass 1: Float (no non-idealities, no mapping)
        # Create a temporary device model with all non-idealities disabled
        float_device_model = MemristorDeviceModel(
            G_min=device_model.G_min,
            G_max=device_model.G_max,
            weight_clip=(device_model.wmin, device_model.wmax),
            variability_sigma=0.0,  # Disable all non-idealities
            read_noise_sigma=0.0,
            drift_alpha=0.0,
            stuck_ratio=0.0,
            ir_drop_beta=0.0,
            mapping=device_model.mapping,
        )
        
        # Temporarily replace device_model in all layers for float forward
        _set_device_model_for_model(model, float_device_model)
        _set_mapping_net_for_model(model, None)
        
        # Additional safety: Ensure no mapping_net is set for float forward
        # If model is a MemristorModel wrapper, we need to access base_model
        target_model = model
        if hasattr(model, 'base_model'):
            target_model = model.base_model
        
        for module in target_model.modules():
            if hasattr(module, 'mapping_net'):
                module.mapping_net = None
        
        model.eval()
        with torch.no_grad():
            try:
                float_output = model(data, t=0)
            except TypeError:
                float_output = model(data)
        
        # Restore original device_model
        _set_device_model_for_model(model, device_model)
        model.train()
        
        # Forward pass 2: Hardware (with non-idealities and learned mapping)
        _set_mapping_net_for_model(model, mapping_net)
        try:
            hw_output = model(data, t=t, seed=None)
        except TypeError:
            hw_output = model(data)
        
        # Primary loss: cross-entropy on hardware output
        task_loss = criterion(hw_output, target)
        
        # Auxiliary loss: MSE between float and hardware outputs
        hw_mse = F.mse_loss(float_output, hw_output)
        
        # Regularization loss: ||Δ||^2
        reg_loss = sum(p.pow(2).sum() for p in mapping_net.parameters())
        
        # Total loss
        loss = task_loss + beta * hw_mse + gamma * reg_loss

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(mapping_net.parameters(), max_norm=1.0)
        
        optimizer.step()
        mapping_optimizer.step()
        
        # Metrics
        acc1 = accuracy(hw_output, target, topk=(1,))[0]
        losses.update(loss.item(), data.size(0))
        task_losses.update(task_loss.item(), data.size(0))
        hw_losses.update(hw_mse.item(), data.size(0))
        reg_losses.update(reg_loss.item(), data.size(0))
        top1.update(acc1, data.size(0))
    
    return {
        'loss': losses.avg,
        'acc1': top1.avg,
        'task_loss': task_losses.avg,
        'hw_mse': hw_losses.avg,
        'reg_loss': reg_losses.avg,
    }


