"""
Checkpoint saving and loading utilities.
"""

import torch
import os
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def _try_strip_prefix(sd: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """Return a new state_dict with prefix stripped when present."""
    out = {}
    for k, v in sd.items():
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
        else:
            out[k] = v
    return out


def _try_add_prefix(sd: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """Return a new state_dict with prefix added to all keys."""
    return {f"{prefix}{k}": v for k, v in sd.items()}


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

    # PyTorch 2.6+ defaults weights_only=True for torch.load; full training checkpoints
    # (dict with optimizer/config) require weights_only=False.
    try:
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(filepath, map_location=device)
    
    if model is not None:
        if 'model_state_dict' in checkpoint:
            sd = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            sd = checkpoint['state_dict']
        else:
            sd = checkpoint
        model_keys = set(model.state_dict().keys())
        ckpt_keys = set(sd.keys()) if isinstance(sd, dict) else set()
        missing = model_keys - ckpt_keys
        unexpected = ckpt_keys - model_keys
        if missing or unexpected:
            logger.warning(
                "Checkpoint / model key mismatch before load: missing=%d unexpected=%d "
                "(showing up to 5 each): missing=%s unexpected=%s",
                len(missing),
                len(unexpected),
                list(sorted(missing))[:5],
                list(sorted(unexpected))[:5],
            )
        try:
            model.load_state_dict(sd, strict=True)
        except RuntimeError as e:
            # Common when loading between compiled/uncompiled modules:
            # keys may differ by "_orig_mod." prefix.
            tried = False
            last_err = e
            if isinstance(sd, dict):
                if any(k.startswith("_orig_mod.") for k in sd.keys()):
                    tried = True
                    try:
                        model.load_state_dict(_try_strip_prefix(sd, "_orig_mod."), strict=True)
                        logger.warning(
                            "Loaded checkpoint after stripping '_orig_mod.' prefix from keys."
                        )
                    except RuntimeError as e2:
                        last_err = e2
                else:
                    tried = True
                    try:
                        model.load_state_dict(_try_add_prefix(sd, "_orig_mod."), strict=True)
                        logger.warning(
                            "Loaded checkpoint after adding '_orig_mod.' prefix to keys."
                        )
                    except RuntimeError as e2:
                        last_err = e2
            if not tried:
                raise
            # If prefix adaptation also failed, re-raise with original context.
            if last_err is not None and last_err is not e:
                raise last_err
        logger.info(
            "Loaded model state from %s (%d tensors)",
            filepath,
            len(sd) if isinstance(sd, dict) else -1,
        )
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Loaded optimizer state from {filepath}")
    
    return checkpoint


