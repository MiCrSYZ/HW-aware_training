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
            # Solve: x_ref @ w_j ≈ eps_ref[:, j] （求最小二乘解）
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


def compute_loss_sensitivity_weighted_error(
    model: nn.Module,
    criterion: nn.Module,
    inputs_list: List[torch.Tensor],
    targets_list: List[torch.Tensor],
    h_ideal_list: List[torch.Tensor],
    h_hw_list: List[torch.Tensor],
    eps_list: List[torch.Tensor],
    layer_module: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """
    Compute loss-sensitivity weighted error metrics.
    
    Computes:
    - S_ℓ = E_x [⟨∇_{h_ℓ} L(x), e_ℓ(x)⟩]  (inner product)
    - S_ℓ^abs = E_x [||∇_{h_ℓ} L(x) ⊙ e_ℓ(x)||]  (element-wise product norm)
    
    where:
    - e_ℓ(x) = h_ℓ^hw(x) - h_ℓ^ideal(x)  (error)
    - ∇_{h_ℓ} L(x) is the gradient of loss w.r.t. layer output
    
    Args:
        model: Neural network model
        criterion: Loss function
        inputs_list: List of input batches [B, ...]
        targets_list: List of target labels [B]
        h_ideal_list: List of ideal layer outputs [B, out_dim]
        h_hw_list: List of hardware layer outputs [B, out_dim]
        eps_list: List of error tensors [B, out_dim] (e_ℓ(x))
        layer_module: The layer module to compute gradients for
        device: Device to run computation on
        
    Returns:
        Dictionary with:
            - loss_sensitivity_weighted_error: S_ℓ (inner product)
            - loss_sensitivity_weighted_error_abs: S_ℓ^abs (element-wise norm)
    """
    model.eval()
    sensitivity_scores = []
    sensitivity_scores_abs = []
    
    for inputs, targets, h_ideal, h_hw, eps in zip(
        inputs_list, targets_list, h_ideal_list, h_hw_list, eps_list
    ):
        inputs = inputs.to(device)
        targets = targets.to(device)
        h_ideal = h_ideal.to(device)
        h_hw = h_hw.to(device)
        eps = eps.to(device)
        
        # Create a tensor with requires_grad=True for the ideal layer output
        h_ideal_grad = h_ideal.clone().requires_grad_(True)
        
        # Register forward hook to replace layer output with h_ideal_grad
        def forward_hook(module, input, output):
            return h_ideal_grad
        
        hook_handle = layer_module.register_forward_hook(forward_hook)
        
        try:
            # Forward pass (will use h_ideal_grad instead of actual layer output)
            output = model(inputs)
            
            # Compute loss
            loss = criterion(output, targets)
            
            # Backward to compute gradients
            loss.backward()
            
            # Get gradient w.r.t. layer output
            if h_ideal_grad.grad is not None:
                grad_h = h_ideal_grad.grad
            else:
                # Fallback: use zero gradient
                grad_h = torch.zeros_like(h_ideal)
            
            # Handle shape mismatch between grad_h and eps
            # For conv layers, grad_h might be 4D [B, out_ch, H, W] while eps is 2D [B*num_patches, out_ch]
            # For linear layers, both should be 2D [B, out_dim]
            if grad_h.dim() != eps.dim():
                # Reshape grad_h to match eps shape
                if grad_h.dim() == 4 and eps.dim() == 2:
                    # grad_h is [B, out_ch, H, W], eps is [B*num_patches, out_ch]
                    # Reshape grad_h to [B*H*W, out_ch] to match eps structure
                    B, out_ch, H, W = grad_h.shape
                    grad_h_flat = grad_h.permute(0, 2, 3, 1).reshape(-1, out_ch)  # [B*H*W, out_ch]
                    # If dimensions still don't match, try to match the first dimension
                    if grad_h_flat.shape[0] != eps.shape[0]:
                        # Try to reshape to match eps
                        if eps.shape[0] % B == 0:
                            # eps might be [B*num_patches, out_ch] where num_patches = H*W
                            # Try to reshape grad_h to match
                            num_patches = eps.shape[0] // B
                            if num_patches == H * W:
                                grad_h_flat = grad_h.permute(0, 2, 3, 1).reshape(B * num_patches, out_ch)
                            else:
                                # Dimensions don't match, use zero gradient
                                grad_h_flat = torch.zeros_like(eps)
                        else:
                            # Dimensions don't match, use zero gradient
                            grad_h_flat = torch.zeros_like(eps)
                    grad_h = grad_h_flat
                else:
                    # Unsupported shape mismatch, use zero gradient
                    grad_h = torch.zeros_like(eps)
            elif grad_h.shape != eps.shape:
                # Same number of dimensions but different shapes
                # Try to reshape grad_h to match eps
                if grad_h.numel() == eps.numel():
                    grad_h = grad_h.view_as(eps)
                else:
                    # Total number of elements don't match, use zero gradient
                    grad_h = torch.zeros_like(eps)
            
            # Compute S_ℓ = ⟨∇_{h_ℓ} L(x), e_ℓ(x)⟩
            # Inner product: sum over all dimensions
            sensitivity = (grad_h * eps).sum().item()
            
            # Compute S_ℓ^abs = ||∇_{h_ℓ} L(x) ⊙ e_ℓ(x)||
            # Element-wise product then norm
            sensitivity_abs = (grad_h * eps).norm().item()
            
            sensitivity_scores.append(sensitivity)
            sensitivity_scores_abs.append(sensitivity_abs)
            
        except Exception as e:
            # If computation fails, skip this sample
            print(f"Warning: Failed to compute gradient for loss-sensitivity weighted error: {e}")
            sensitivity_scores.append(0.0)
            sensitivity_scores_abs.append(0.0)
        finally:
            # Remove hook
            if hook_handle is not None:
                hook_handle.remove()
            # Clear gradients
            model.zero_grad()
            if h_ideal_grad.grad is not None:
                h_ideal_grad.grad = None
    
    # Compute mean across all samples
    mean_sensitivity = np.mean(sensitivity_scores) if sensitivity_scores else np.nan
    mean_sensitivity_abs = np.mean(sensitivity_scores_abs) if sensitivity_scores_abs else np.nan
    
    return {
        'loss_sensitivity_weighted_error': mean_sensitivity,
        'loss_sensitivity_weighted_error_abs': mean_sensitivity_abs,
    }


def diagnose_input_dependence(
    x_list: List[torch.Tensor],
    eps_list: List[torch.Tensor],
    ref_idx: int = 0,
    model: Optional[nn.Module] = None,
    criterion: Optional[nn.Module] = None,
    inputs_list: Optional[List[torch.Tensor]] = None,
    targets_list: Optional[List[torch.Tensor]] = None,
    h_ideal_list: Optional[List[torch.Tensor]] = None,
    h_hw_list: Optional[List[torch.Tensor]] = None,
    layer_module: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Comprehensive diagnosis of input dependence.
    
    Args:
        x_list: List of input batches [B, in_dim]
        eps_list: List of error tensors [B, out_dim]
        ref_idx: Index of reference input for static ΔW fitting
        model: (Optional) Neural network model for loss-sensitivity computation
        criterion: (Optional) Loss function for loss-sensitivity computation
        inputs_list: (Optional) List of full input batches for model forward pass
        targets_list: (Optional) List of target labels for loss computation
        h_ideal_list: (Optional) List of ideal layer outputs [B, out_dim]
        h_hw_list: (Optional) List of hardware layer outputs [B, out_dim]
        layer_module: (Optional) Layer module to compute gradients for
        device: (Optional) Device to run computation on
        
    Returns:
        Dictionary with all diagnostic metrics, including:
            - error_variance: Error variance across inputs
            - mean_correlation: Mean correlation between errors
            - static_deltaW_residuals: Residuals from static ΔW fitting
            - static_deltaW_residual_ratios: Residual ratios
            - loss_sensitivity_weighted_error: S_ℓ = E_x [⟨∇_{h_ℓ} L(x), e_ℓ(x)⟩]
            - loss_sensitivity_weighted_error_abs: S_ℓ^abs = E_x [||∇_{h_ℓ} L(x) ⊙ e_ℓ(x)||]
    """
    # (A) Error variance across inputs
    var_input = compute_error_variance(eps_list)
    
    # (B) Correlation between errors from different inputs
    mean_corr, corr_matrix = compute_error_correlation(eps_list)
    
    # (C) Static ΔW transferability test
    transfer_results = test_static_deltaW_transferability(
        x_list, eps_list, ref_idx=ref_idx
    )
    
    result = {
        'error_variance': var_input,
        'mean_correlation': mean_corr,
        'correlation_matrix': corr_matrix,
        'static_deltaW_residuals': transfer_results['residuals'],
        'static_deltaW_residual_ratios': transfer_results['residual_ratios'],
        'DeltaW_hat': transfer_results['DeltaW_hat'],
        'num_inputs': len(x_list),
    }
    
    # (D) Loss-sensitivity weighted error (optional)
    if all([model is not None, criterion is not None, inputs_list is not None,
            targets_list is not None, h_ideal_list is not None, h_hw_list is not None,
            layer_module is not None, device is not None]):
        try:
            sensitivity_results = compute_loss_sensitivity_weighted_error(
                model=model,
                criterion=criterion,
                inputs_list=inputs_list,
                targets_list=targets_list,
                h_ideal_list=h_ideal_list,
                h_hw_list=h_hw_list,
                eps_list=eps_list,
                layer_module=layer_module,
                device=device,
            )
            result.update(sensitivity_results)
        except Exception as e:
            print(f"Warning: Failed to compute loss-sensitivity weighted error: {e}")
            result['loss_sensitivity_weighted_error'] = np.nan
            result['loss_sensitivity_weighted_error_abs'] = np.nan
    else:
        # Set to NaN if parameters not provided
        result['loss_sensitivity_weighted_error'] = np.nan
        result['loss_sensitivity_weighted_error_abs'] = np.nan
    
    return result
