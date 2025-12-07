"""
Logging, TensorBoard, and Weights & Biases setup.
"""

import logging
import os
from pathlib import Path
from typing import Optional
import torch

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


def setup_logger(
    log_dir: str,
    name: str = 'experiment',
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Set up file and console logging.
    
    Args:
        log_dir: Directory to save log files
        name: Logger name
        level: Logging level
        
    Returns:
        Logger instance
    """
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()  # Remove existing handlers
    
    # File handler
    log_file = os.path.join(log_dir, 'train.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger


def setup_tensorboard(log_dir: str) -> Optional[object]:
    """
    Set up TensorBoard writer.
    
    Args:
        log_dir: Directory to save TensorBoard logs
        
    Returns:
        SummaryWriter instance or None if TensorBoard not available
    """
    if not TENSORBOARD_AVAILABLE:
        logging.warning("TensorBoard not available. Install with: pip install tensorboard")
        return None
    
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    return writer


def setup_wandb(
    project_name: str,
    run_name: Optional[str] = None,
    config: Optional[dict] = None,
    enabled: bool = True,
) -> Optional[object]:
    """
    Set up Weights & Biases logging.
    
    Args:
        project_name: W&B project name
        run_name: Run name (optional)
        config: Configuration dictionary to log
        enabled: Whether to enable W&B (checks for API key)
        
    Returns:
        W&B run object or None if not available/disabled
    """
    if not enabled:
        return None
    
    if not WANDB_AVAILABLE:
        logging.warning("wandb not available. Install with: pip install wandb")
        return None
    
    # Check for API key
    api_key = os.environ.get('WANDB_API_KEY')
    if not api_key:
        logging.warning("WANDB_API_KEY not set. Disabling wandb logging.")
        return None
    
    try:
        wandb.login(key=api_key)
        run = wandb.init(
            project=project_name,
            name=run_name,
            config=config,
        )
        return run
    except Exception as e:
        logging.warning(f"Failed to initialize wandb: {e}. Disabling wandb logging.")
        return None


