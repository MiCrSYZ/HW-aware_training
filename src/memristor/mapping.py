"""
Weight-to-conductance mapping utilities.

This module provides helper functions for mapping neural network weights
to memristor conductance values, including linear, logarithmic, and
differential pair mappings.
"""

import torch
import torch.nn.functional as F
from typing import Tuple


def map_weights_linear(
    W: torch.Tensor,
    G_min: float,
    G_max: float,
    wmin: float,
    wmax: float,
) -> torch.Tensor:
    """
    Linear mapping from weights to conductance.
    
    Maps weights in range [wmin, wmax] linearly to [G_min, G_max].
    
    Args:
        W: Weight tensor
        G_min: Minimum conductance
        G_max: Maximum conductance
        wmin: Minimum weight value
        wmax: Maximum weight value
        
    Returns:
        G: Conductance tensor with same shape as W
    """
    W_clamped = torch.clamp(W, wmin, wmax)
    scale = (W_clamped - wmin) / (wmax - wmin + 1e-12)
    G = G_min + scale * (G_max - G_min)
    return G


def map_weights_log(
    W: torch.Tensor,
    G_min: float,
    G_max: float,
    wmin: float,
    wmax: float,
) -> torch.Tensor:
    """
    Logarithmic mapping from weights to conductance.
    
    Maps weights to conductance using exponential relationship:
    G = exp(a*W + b) where a and b are chosen to map [wmin, wmax] -> [G_min, G_max].
    
    Args:
        W: Weight tensor
        G_min: Minimum conductance
        G_max: Maximum conductance
        wmin: Minimum weight value
        wmax: Maximum weight value
        
    Returns:
        G: Conductance tensor with same shape as W
    """
    import numpy as np
    W_clamped = torch.clamp(W, wmin, wmax)
    a = (np.log(G_max) - np.log(G_min)) / (wmax - wmin + 1e-12)
    b = np.log(G_min) - a * wmin
    G = torch.exp(a * W_clamped + b)
    return G


def differential_pair_mapping(W: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split weights into positive and negative pairs for differential memristor arrays.
    
    In differential pair mapping, each weight is represented by two memristors:
    one for positive values and one for negative values. The effective weight is
    W_eff = W_pos - W_neg.
    
    Args:
        W: Weight tensor (can be negative)
        
    Returns:
        W_pos: Positive weight tensor (clamped to [0, max(W)])
        W_neg: Negative weight tensor (clamped to [0, -min(W)])
    """
    W_pos = torch.clamp(W, min=0.0)
    W_neg = torch.clamp(-W, min=0.0)
    return W_pos, W_neg


def reshape_conv_to_matrix(weight_conv: torch.Tensor) -> torch.Tensor:
    """
    Reshape convolutional weight tensor to matrix form.
    
    Converts conv weight from shape [out_ch, in_ch, k, k] to [out_ch, in_ch*k*k].
    This is useful for mapping conv layers to memristor crossbar arrays.
    
    Args:
        weight_conv: Convolutional weight tensor [out_ch, in_ch, k, k]
        
    Returns:
        W_flat: Flattened weight matrix [out_ch, in_ch*k*k]
    """
    out_ch, in_ch, k_h, k_w = weight_conv.shape
    W_flat = weight_conv.view(out_ch, in_ch * k_h * k_w)
    return W_flat


def reshape_matrix_to_conv(
    W_flat: torch.Tensor,
    out_ch: int,
    in_ch: int,
    k_h: int,
    k_w: int,
) -> torch.Tensor:
    """
    Reshape matrix back to convolutional weight tensor.
    
    Inverse operation of reshape_conv_to_matrix.
    
    Args:
        W_flat: Flattened weight matrix [out_ch, in_ch*k*k]
        out_ch: Output channels
        in_ch: Input channels
        k_h: Kernel height
        k_w: Kernel width
        
    Returns:
        weight_conv: Convolutional weight tensor [out_ch, in_ch, k, k]
    """
    weight_conv = W_flat.view(out_ch, in_ch, k_h, k_w)
    return weight_conv


