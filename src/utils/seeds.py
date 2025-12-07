"""
Deterministic seed setting for reproducibility.
"""

import torch
import numpy as np
import random
from typing import Optional


def set_seed(seed: Optional[int] = None) -> None:
    """
    Set random seeds for reproducibility.
    
    Sets seeds for Python random, NumPy, PyTorch CPU, and PyTorch CUDA.
    
    Args:
        seed: Random seed (None for no seeding)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Ensure deterministic behavior (may reduce performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


