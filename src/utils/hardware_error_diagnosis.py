"""
Hardware error diagnosis utilities.

This module provides tools to diagnose whether hardware-induced errors
are input-dependent, which would indicate that static weight-only mapping
is insufficient.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

try:
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    # Fallback if sklearn not available
    def cosine_similarity(a, b):
        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = np.linalg.norm(b, axis=1, keepdims=True)
        return np.dot(a, b.T) / (a_norm * b_norm.T + 1e-12)


def hardware_error(
    x: torch.Tensor,
    W: torch.Tensor,
    device_model,
    t: int = 0,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Compute hardware error: epsilon = hw(x, W) - x @ W.T
    
    Args:
        x: Input batch [B, in_dim]
        W: Ideal weight [out_dim, in_dim]
        device_model: MemristorDeviceModel instance
        t: Time/cycle index for drift
        seed: Random seed for reproducibility
        
    Returns:
        epsilon: Hardware error [B, out_dim]
    """
    with torch.no_grad():
        # Ideal computation
        y_ideal = F.linear(x, W)
        
        # Hardware computation
        try:
            from ..memristor.memristor_wrappers import hardware_linear_forward_adaptive
        except ImportError:
            from src.memristor.memristor_wrappers import hardware_linear_forward_adaptive
        
        y_hw, _ = hardware_linear_forward_adaptive(
            x, W, device_model, t=t, seed=seed, training=False
        )
        
        epsilon = y_hw - y_ideal
    
    return epsilon


def compute_error_variance(eps_list: List[torch.Tensor]) -> float:
    """
    Compute error variance across inputs.
    
    Args:
        eps_list: List of error tensors, each [B, out_dim]
        
    Returns:
        Mean variance across inputs
    """
    eps_stack = torch.stack(eps_list)  # [K, B, out_dim]
    var_input = eps_stack.var(dim=0).mean().item()  # Variance across K inputs, then mean
    return var_input


def compute_error_correlation(eps_list: List[torch.Tensor]) -> Tuple[float, np.ndarray]:
    """
    Compute correlation between errors from different inputs.
    
    Args:
        eps_list: List of error tensors, each [B, out_dim]
        
    Returns:
        Mean correlation across all (i, j) pairs
        Correlation matrix [K, K]
    """
    K = len(eps_list)
    correlations = []
    corr_matrix = np.zeros((K, K))
    
    for i in range(K):
        for j in range(i + 1, K):
            eps_i_flat = eps_list[i].flatten().cpu().numpy()
            eps_j_flat = eps_list[j].flatten().cpu().numpy()
            
            # Compute cosine similarity
            corr = cosine_similarity(
                eps_i_flat.reshape(1, -1),
                eps_j_flat.reshape(1, -1)
            )[0, 0]
            
            correlations.append(corr)
            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr
    
    # Diagonal is 1.0 (self-correlation)
    np.fill_diagonal(corr_matrix, 1.0)
    
    mean_corr = np.mean(correlations) if correlations else 0.0
    return mean_corr, corr_matrix


def fit_static_deltaW(
    x_ref: torch.Tensor,
    eps_ref: torch.Tensor,
) -> torch.Tensor:
    """
    Fit a static ΔW using least squares on reference input.
    
    Solve: x_ref @ ΔW.T ≈ eps_ref
    
    Args:
        x_ref: Reference input [B, in_dim]
        eps_ref: Reference error [B, out_dim]
        
    Returns:
        DeltaW_hat: Estimated static weight correction [out_dim, in_dim]
    """
    with torch.no_grad():
        # Solve for each output dimension independently
        # eps_ref[:, j] = x_ref @ DeltaW[j, :].T
        # So: DeltaW[j, :] = (x_ref.T @ x_ref)^(-1) @ x_ref.T @ eps_ref[:, j]
        
        out_dim = eps_ref.shape[1]
        in_dim = x_ref.shape[1]
        DeltaW_hat = torch.zeros(out_dim, in_dim, device=x_ref.device)
        
        # Use torch.linalg.lstsq for each output dimension
        for j in range(out_dim):
            # Solve: x_ref @ w_j ≈ eps_ref[:, j]
            solution, _, _, _ = torch.linalg.lstsq(
                x_ref, eps_ref[:, j:j+1]
            )
            DeltaW_hat[j, :] = solution.squeeze()
    
    return DeltaW_hat


def test_static_deltaW_transferability(
    x_list: List[torch.Tensor],
    eps_list: List[torch.Tensor],
    ref_idx: int = 0,
) -> Dict[str, Any]:
    """
    Test whether a static ΔW fitted on one input transfers to others.
    
    Args:
        x_list: List of input batches
        eps_list: List of error tensors
        ref_idx: Index of reference input for fitting
        
    Returns:
        Dictionary with:
            - DeltaW_hat: Fitted static correction
            - residuals: Residual norms for each input
            - residual_ratios: Residual / error norm ratios
    """
    x_ref = x_list[ref_idx]
    eps_ref = eps_list[ref_idx]
    
    # Fit static ΔW on reference input
    DeltaW_hat = fit_static_deltaW(x_ref, eps_ref)
    
    # Test on all inputs
    residuals = []
    residual_ratios = []
    
    for x_k, eps_k in zip(x_list, eps_list):
        # Predict error using static ΔW
        eps_pred = F.linear(x_k, DeltaW_hat)
        
        # Compute residual
        residual = (eps_k - eps_pred).norm().item()
        eps_norm = eps_k.norm().item()
        
        residuals.append(residual)
        residual_ratios.append(residual / (eps_norm + 1e-12))
    
    return {
        'DeltaW_hat': DeltaW_hat,
        'residuals': residuals,
        'residual_ratios': residual_ratios,
        'ref_idx': ref_idx,
    }


def diagnose_input_dependence(
    x_list: List[torch.Tensor],
    eps_list: List[torch.Tensor],
    ref_idx: int = 0,
) -> Dict[str, Any]:
    """
    Comprehensive diagnosis of input dependence.
    
    Args:
        x_list: List of input batches [B, in_dim]
        eps_list: List of error tensors [B, out_dim]
        ref_idx: Index of reference input for static ΔW fitting
        
    Returns:
        Dictionary with all diagnostic metrics
    """
    # (A) Error variance across inputs
    var_input = compute_error_variance(eps_list)
    
    # (B) Correlation between errors from different inputs
    mean_corr, corr_matrix = compute_error_correlation(eps_list)
    
    # (C) Static ΔW transferability test
    transfer_results = test_static_deltaW_transferability(
        x_list, eps_list, ref_idx=ref_idx
    )
    
    return {
        'error_variance': var_input,
        'mean_correlation': mean_corr,
        'correlation_matrix': corr_matrix,
        'static_deltaW_residuals': transfer_results['residuals'],
        'static_deltaW_residual_ratios': transfer_results['residual_ratios'],
        'DeltaW_hat': transfer_results['DeltaW_hat'],
        'num_inputs': len(x_list),
    }
