"""
Checkpoint saving and loading utilities.
"""

import torch
import os
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def save_checkpoint(
    state: Dict[str, Any],
    filepath: str,
    is_best: bool = False,
) -> None:
    """
    Save model checkpoint.
    
    Args:
        state: Dictionary containing model state, optimizer state, epoch, etc.
        filepath: Path to save checkpoint
        is_best: Whether this is the best model so far (if True, also saves to model_best.pth)
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)
    logger.info(f"Saved checkpoint to {filepath}")
    
    if is_best:
        best_path = os.path.join(os.path.dirname(filepath), 'model_best.pth')
        torch.save(state, best_path)
        logger.info(f"Saved best model to {best_path}")


def load_checkpoint(
    filepath: str,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        filepath: Path to checkpoint file
        model: Model to load state into (optional)
        optimizer: Optimizer to load state into (optional)
        device: Device to map model to
        
    Returns:
        Checkpoint dictionary
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    checkpoint = torch.load(filepath, map_location=device)
    
    if model is not None:
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        logger.info(f"Loaded model state from {filepath}")
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Loaded optimizer state from {filepath}")
    
    return checkpoint


