"""
Gradient quality metrics for synthetic noise experiments.

- Dead-zone ratio (Jacobian zero proxy): proportion of gradient elements (or channels)
  with magnitude below threshold, or proportion of channels with near-zero gradient norm.
- Effective rank / top-k energy ratio: singular value energy concentration of the
  gradient matrix (batch of gradient vectors) as a proxy for gradient rank.

New metrics (gradient reachability, consistency, variance domination, B_mean, perturbation stability):
- A (gradient reachability / sensitivity): ||g_noisy|| / (||g_clean|| + ||g_noisy|| + eps) ∈ (0,1].
- C (gradient consistency / effective signal alignment): E[<g_noisy, g_clean>] / (E[||g_clean||^2] + eps).
- V (noise dominance / gradient variance domination): variance of g over noise samples / ||E[g]||^2.
- B_mean: ||E[g_noisy]||^2 / (||g_clean||^2 + eps).
- S (perturbation stability / structural stability): ||E[Δ]||^2 / (E[||Δ||^2] + eps), Δ = h_noisy - h_clean.
- P (sign_coupled_scaling): per-batch proportion of sign(v^T z) in positive / negative / zero (only when noise_type is sign_coupled_scaling).
- Layer-wise A, C, V, B_mean: same metrics per tier (ResNet-20: stem, layer1, layer2, layer3; ViT: blocks_1, blocks_3, blocks_5; GRU: embedding, gru_l0, gru_l1, head).
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from contextlib import contextmanager

# Default threshold for "near zero" (relative to global scale or absolute)
DEFAULT_EPS = 1e-7


def dead_zone_ratio(
    grad: torch.Tensor,
    mode: str = "channel",
    eps: float = DEFAULT_EPS,
    relative_eps: bool = False,
) -> float:
    """
    Fraction of gradient structure that is effectively zero (dead zone).

    Args:
        grad: Gradient tensor, shape (B, C, ...) or (B, D). Typically ∂L/∂z̃ (output of noisy layer).
        mode: 'element' = fraction of elements with |g| < eps;
              'channel' = fraction of channels whose Frobenius norm (over batch + spatial) < eps.
        eps: Absolute threshold for "zero". Ignored if relative_eps=True.
        relative_eps: If True, use eps relative to max element: threshold = eps * grad.abs().max().clamp(min=1e-12).

    Returns:
        Ratio in [0, 1]. Higher = more dead zone (worse gradient flow).
    """
    if grad is None or grad.numel() == 0:
        return 0.0
    g = grad.detach().float()
    if relative_eps:
        scale = g.abs().max().clamp(min=1e-12).item()
        thresh = eps * scale
    else:
        thresh = eps

    if mode == "element":
        near_zero = (g.abs() < thresh).float()
        return (near_zero.sum().item() / g.numel())
    elif mode == "channel":
        # Per-channel norm: (B, D) -> norm over B; (B, C, H, W) -> norm over B and spatial per C
        if g.dim() == 2:
            norms = g.norm(dim=0)  # (D,)
        else:
            g_flat = g.reshape(g.size(0), g.size(1), -1)  # (B, C, L)
            norms = g_flat.norm(dim=(0, 2))  # (C,)
        if norms.numel() == 0:
            return 0.0
        dead = (norms < thresh).float()
        return (dead.sum().item() / norms.numel())
    else:
        raise ValueError(f"mode must be 'element' or 'channel', got {mode}")


def gradient_top_k_energy_ratio(
    grad: torch.Tensor,
    k: int = 10,
) -> float:
    """
    Top-k singular value squared energy ratio of the gradient matrix.

    Stack gradient vectors (per sample) as rows: shape (B, D). Then compute
    singular values of this matrix and return (sum of first k s_i^2) / (sum of all s_i^2).
    Measures how concentrated the gradient is in a low-dimensional subspace.

    Args:
        grad: Gradient tensor, shape (B, C, ...) or (B, D). Flattened to (B, D).
        k: Number of leading singular values to sum.

    Returns:
        Ratio in (0, 1]. Near 1 = energy in top-k (low effective rank); lower = more spread.
    """
    if grad is None or grad.numel() == 0:
        return 0.0
    g = grad.detach().float().reshape(grad.size(0), -1)
    B, D = g.shape
    if B == 0 or D == 0:
        return 0.0
    # SVD: G = U S V^T, we need S. For (B, D), min is min(B, D).
    try:
        # Use torch.linalg.svdvals for efficiency (only singular values)
        s = torch.linalg.svdvals(g)
    except Exception:
        return 0.0
    s_sq = s * s
    total = s_sq.sum().item()
    if total <= 0:
        return 0.0
    k_use = min(k, s.numel())
    top_k_sum = s_sq[:k_use].sum().item()
    return top_k_sum / total


def gradient_effective_rank(grad: torch.Tensor) -> float:
    """
    Effective rank of the gradient matrix (exponential of entropy of normalized squared singular values).

    G flattened to (B, D). Let s_i be singular values, p_i = s_i^2 / sum(s_j^2).
    Effective rank = exp(- sum_i p_i log(p_i)). Equals true rank if spectrum is flat;
    lower if spectrum is concentrated.

    Args:
        grad: Gradient tensor, shape (B, C, ...) or (B, D).

    Returns:
        Effective rank (float). Lower = more rank-deficient / concentrated.
    """
    if grad is None or grad.numel() == 0:
        return 0.0
    g = grad.detach().float().reshape(grad.size(0), -1)
    B, D = g.shape
    if B == 0 or D == 0:
        return 0.0
    try:
        s = torch.linalg.svdvals(g)
    except Exception:
        return 0.0
    s_sq = s * s
    total = s_sq.sum().item()
    if total <= 0:
        return 0.0
    p = (s_sq / total).cpu().numpy()
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    entropy = -np.sum(p * np.log(p + 1e-12))
    return float(np.exp(entropy))


# -----------------------------------------------------------------------------
# Helpers for new metrics (A, C, V, S)
# -----------------------------------------------------------------------------


def _get_param_grad_vector(model: nn.Module) -> Optional[torch.Tensor]:
    """Concatenate all parameter gradients into a single vector. Returns None if no grads."""
    grads = []
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            grads.append(p.grad.detach().flatten())
    if not grads:
        return None
    return torch.cat(grads)


# Layer-wise tier definitions: ResNet-20 stem/layer1/2/3, ViT blocks.1/3/5, GRU embedding/gru_l0/gru_l1/head
TIER_KEYS: Dict[str, List[str]] = {
    "resnet20": ["stem", "layer1", "layer2", "layer3"],
    "vit_tiny": ["blocks_1", "blocks_3", "blocks_5"],
    "gru_agnews": ["embedding", "gru_l0", "gru_l1", "head"],
}


def _get_tier_for_param(param_name: str, model_name: str) -> Optional[str]:
    """Map parameter name to tier key for layer-wise metrics. Strips base_model. prefix."""
    name = param_name.replace("base_model.", "", 1) if param_name.startswith("base_model.") else param_name
    if model_name == "resnet20":
        if name.startswith("conv1.") or name.startswith("bn1."):
            return "stem"
        if name.startswith("layer1."):
            return "layer1"
        if name.startswith("layer2."):
            return "layer2"
        if name.startswith("layer3."):
            return "layer3"
        return None
    if model_name == "vit_tiny":
        if name.startswith("blocks.1."):
            return "blocks_1"
        if name.startswith("blocks.3."):
            return "blocks_3"
        if name.startswith("blocks.5."):
            return "blocks_5"
        return None
    if model_name == "gru_agnews":
        if name.startswith("embedding."):
            return "embedding"
        if "gru." in name and "_l0" in name:
            return "gru_l0"
        if "gru." in name and "_l1" in name:
            return "gru_l1"
        if name.startswith("head."):
            return "head"
        return None
    return None


def _get_tier_param_names(model: nn.Module, model_name: str) -> Dict[str, List[str]]:
    """Tier -> list of param names (order preserved for consistent grad concatenation)."""
    base = getattr(model, "base_model", model)
    tier_keys = TIER_KEYS.get(model_name, [])
    tier_params: Dict[str, List[str]] = {t: [] for t in tier_keys}
    for name, param in base.named_parameters():
        if not param.requires_grad:
            continue
        tier = _get_tier_for_param(name, model_name)
        if tier and tier in tier_params:
            tier_params[tier].append(name)
    return tier_params


def _get_param_grad_vector_for_tier(
    model: nn.Module, tier_param_list: List[str]
) -> Optional[torch.Tensor]:
    """Concatenate gradients of listed params (by name) in order. Returns None if no grads."""
    base = getattr(model, "base_model", model)
    name_to_param = dict(base.named_parameters())
    grads = []
    for name in tier_param_list:
        p = name_to_param.get(name)
        if p is not None and p.grad is not None:
            grads.append(p.grad.detach().flatten())
    if not grads:
        return None
    return torch.cat(grads)


@contextmanager
def _noise_toggled(model: nn.Module, synth_layer_types: Tuple[type, ...], enabled: bool):
    """Temporarily set enable_noise on all synth_layer_types layers. Restore on exit."""
    saved = []
    for _name, mod in model.named_modules():
        if isinstance(mod, synth_layer_types):
            if hasattr(mod, "enable_noise"):
                saved.append((mod, mod.enable_noise))
                mod.enable_noise = enabled
    try:
        yield
    finally:
        for mod, old in saved:
            mod.enable_noise = old


def _run_forward_backward(
    model: nn.Module,
    data: torch.Tensor,
    target: torch.Tensor,
    criterion: nn.Module,
    lengths: Optional[torch.Tensor] = None,
    is_gru: bool = False,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """One forward-backward pass. Returns loss value (and fills param.grad)."""
    model.zero_grad()
    if is_gru and lengths is not None:
        out = model(data, lengths=lengths, seed=seed)
    else:
        out = model(data, seed=seed) if hasattr(model, "forward") else model(data)
    loss = criterion(out, target)
    loss.backward()
    return loss.detach()


def compute_layer_gradient_metrics(
    grad: torch.Tensor,
    eps: float = DEFAULT_EPS,
    k: int = 10,
    relative_eps: bool = True,
) -> Dict[str, float]:
    """Compute dead_zone_ratio (element and channel) and top_k_energy_ratio and effective_rank for one gradient tensor.
    relative_eps=True (default) uses threshold eps * max|grad| so the metric scales with gradient magnitude and
    distinguishes different noise types; relative_eps=False can collapse to 0 or 1 when gradient scale varies."""
    out = {}
    out["dead_zone_ratio_element"] = dead_zone_ratio(grad, mode="element", eps=eps, relative_eps=relative_eps)
    out["dead_zone_ratio_channel"] = dead_zone_ratio(grad, mode="channel", eps=eps, relative_eps=relative_eps)
    out["grad_top_k_energy_ratio"] = gradient_top_k_energy_ratio(grad, k=k)
    out["grad_effective_rank"] = gradient_effective_rank(grad)
    return out


# -----------------------------------------------------------------------------
# Gradient reachability (A) and consistency (C)
# -----------------------------------------------------------------------------


def gradient_reachability_and_consistency(
    model: nn.Module,
    batch: Tuple[torch.Tensor, ...],
    criterion: nn.Module,
    device: torch.device,
    synth_layer_types: Tuple[type, ...],
    is_gru: bool = False,
    seed: Optional[int] = None,
    eps: float = DEFAULT_EPS,
) -> Dict[str, float]:
    """
    Gradient reachability (A) and effective signal alignment (C) on the same batch.

    A = ||g_noisy|| / (||g_clean|| + ||g_noisy|| + eps) ∈ (0, 1]: fraction of combined gradient
        magnitude that is in the noisy path; measures whether gradient pathway exists under noise.
        A ≈ 0 ⇒ pathway effectively dead; A appreciable (e.g. > 0.1) ⇒ pathway present.
    C = E[<g_noisy, g_clean>] / (E[||g_clean||^2] + eps): effective projection of noisy gradient
        onto clean direction (alignment).

    Same batch: first backward with noise disabled → g_clean, then with noise enabled → g_noisy.
    """
    if is_gru:
        data, target = batch[1].to(device), batch[0].to(device)
        lengths = batch[2].to(device) if len(batch) > 2 else None
    else:
        data, target = batch[0].to(device), batch[1].to(device)
        lengths = None

    model.train()
    # Clean backward
    with _noise_toggled(model, synth_layer_types, False):
        _run_forward_backward(model, data, target, criterion, lengths=lengths, is_gru=is_gru, seed=seed)
    g_clean = _get_param_grad_vector(model)
    if g_clean is None:
        return {"gradient_reachability": float("nan"), "gradient_consistency": float("nan")}

    # Noisy backward (same batch)
    with _noise_toggled(model, synth_layer_types, True):
        _run_forward_backward(model, data, target, criterion, lengths=lengths, is_gru=is_gru, seed=seed)
    g_noisy = _get_param_grad_vector(model)
    if g_noisy is None:
        return {"gradient_reachability": float("nan"), "gradient_consistency": float("nan")}

    # Ensure same length (same params)
    if g_clean.shape != g_noisy.shape:
        min_len = min(g_clean.numel(), g_noisy.numel())
        g_clean = g_clean.flatten()[:min_len]
        g_noisy = g_noisy.flatten()[:min_len]

    norm_clean = g_clean.norm().item()
    norm_noisy = g_noisy.norm().item()
    inner = (g_noisy * g_clean).sum().item()
    norm_sq_clean = (g_clean * g_clean).sum().item()

    # A = ||g_noisy|| / (||g_clean|| + ||g_noisy|| + eps) ∈ (0,1]: pathway existence
    A = norm_noisy / (norm_clean + norm_noisy + eps)
    C = inner / (norm_sq_clean + eps)
    return {"gradient_reachability": float(A), "gradient_consistency": float(C)}


# -----------------------------------------------------------------------------
# Gradient variance domination (V)
# -----------------------------------------------------------------------------


def gradient_variance_domination(
    model: nn.Module,
    batch: Tuple[torch.Tensor, ...],
    criterion: nn.Module,
    device: torch.device,
    K: int = 8,
    seed_base: Optional[int] = None,
    eps: float = DEFAULT_EPS,
    is_gru: bool = False,
) -> float:
    """
    Noise dominance: V = E[||g - E[g|x,W]||^2] / (||E[g|x,W]||^2 + eps).

    Fix batch and W; sample noise K times (different seeds), get g^(1),...,g^(K).
    mean_g = (1/K) sum g^(k), var = (1/K) sum ||g^(k) - mean_g||^2, V = var / (||mean_g||^2 + eps).
    """
    if is_gru:
        data, target = batch[1].to(device), batch[0].to(device)
        lengths = batch[2].to(device) if len(batch) > 2 else None
    else:
        data, target = batch[0].to(device), batch[1].to(device)
        lengths = None

    model.train()
    grads: List[torch.Tensor] = []
    for k in range(K):
        seed = (seed_base + k) if seed_base is not None else k
        _run_forward_backward(model, data, target, criterion, lengths=lengths, is_gru=is_gru, seed=seed)
        g = _get_param_grad_vector(model)
        if g is not None:
            grads.append(g.detach().clone())

    if len(grads) < 2:
        return float("nan")
    stack = torch.stack(grads)
    mean_g = stack.mean(dim=0)
    var = ((stack - mean_g) ** 2).sum().item() / len(grads)
    signal_sq = (mean_g * mean_g).sum().item()
    V = var / (signal_sq + eps)
    return float(V)


# -----------------------------------------------------------------------------
# Gradient mean bias (B_mean): ||E[g_noisy]||^2 / (||g_clean||^2 + eps)
# -----------------------------------------------------------------------------


def gradient_B_mean(
    model: nn.Module,
    batch: Tuple[torch.Tensor, ...],
    criterion: nn.Module,
    device: torch.device,
    synth_layer_types: Tuple[type, ...],
    K: int = 8,
    seed_base: Optional[int] = None,
    eps: float = DEFAULT_EPS,
    is_gru: bool = False,
) -> float:
    """
    B_mean = ||E[g_noisy]||^2 / (||g_clean||^2 + eps).

    One backward with noise disabled → g_clean; K backwards with noise enabled (different seeds)
    → g_noisy^(1), ..., g_noisy^(K). Then mean_g = (1/K) sum g_noisy^(k), B_mean = ||mean_g||^2 / (||g_clean||^2 + eps).
    """
    if is_gru:
        data, target = batch[1].to(device), batch[0].to(device)
        lengths = batch[2].to(device) if len(batch) > 2 else None
    else:
        data, target = batch[0].to(device), batch[1].to(device)
        lengths = None

    model.train()
    # g_clean
    with _noise_toggled(model, synth_layer_types, False):
        _run_forward_backward(model, data, target, criterion, lengths=lengths, is_gru=is_gru, seed=seed_base)
    g_clean = _get_param_grad_vector(model)
    if g_clean is None:
        return float("nan")
    norm_sq_clean = (g_clean * g_clean).sum().item()

    grads_noisy: List[torch.Tensor] = []
    for k in range(K):
        seed = (seed_base + k) if seed_base is not None else k
        with _noise_toggled(model, synth_layer_types, True):
            _run_forward_backward(model, data, target, criterion, lengths=lengths, is_gru=is_gru, seed=seed)
        g = _get_param_grad_vector(model)
        if g is not None:
            grads_noisy.append(g.detach().clone())

    if not grads_noisy:
        return float("nan")
    stack = torch.stack(grads_noisy)
    mean_g_noisy = stack.mean(dim=0)
    norm_sq_mean = (mean_g_noisy * mean_g_noisy).sum().item()
    B_mean = norm_sq_mean / (norm_sq_clean + eps)
    return float(B_mean)


# -----------------------------------------------------------------------------
# Layer-wise A, C, V, B_mean (by tier: ResNet stem/layer1/2/3, ViT blocks.1/3/5, GRU embedding/gru_l0/gru_l1/head)
# -----------------------------------------------------------------------------


def gradient_reachability_and_consistency_layerwise(
    model: nn.Module,
    batch: Tuple[torch.Tensor, ...],
    criterion: nn.Module,
    device: torch.device,
    synth_layer_types: Tuple[type, ...],
    model_name: str,
    is_gru: bool = False,
    seed: Optional[int] = None,
    eps: float = DEFAULT_EPS,
    return_consistency_denom_numer: bool = False,
) -> Dict[str, float]:
    """
    Layer-wise A and C per tier. Returns gradient_reachability_<tier>, gradient_consistency_<tier>.
    If return_consistency_denom_numer=True, also returns gradient_consistency_<tier>_numer (inner product)
    and gradient_consistency_<tier>_denom (||g_clean||^2 + eps) to debug large C when denom is tiny.
    """
    tier_keys = TIER_KEYS.get(model_name, [])
    if not tier_keys:
        return {}
    tier_param_names = _get_tier_param_names(model, model_name)

    if is_gru:
        data, target = batch[1].to(device), batch[0].to(device)
        lengths = batch[2].to(device) if len(batch) > 2 else None
    else:
        data, target = batch[0].to(device), batch[1].to(device)
        lengths = None

    model.train()
    with _noise_toggled(model, synth_layer_types, False):
        _run_forward_backward(model, data, target, criterion, lengths=lengths, is_gru=is_gru, seed=seed)
    g_clean_by_tier = {
        t: _get_param_grad_vector_for_tier(model, tier_param_names[t])
        for t in tier_keys
        if tier_param_names[t]
    }

    with _noise_toggled(model, synth_layer_types, True):
        _run_forward_backward(model, data, target, criterion, lengths=lengths, is_gru=is_gru, seed=seed)
    g_noisy_by_tier = {
        t: _get_param_grad_vector_for_tier(model, tier_param_names[t])
        for t in tier_keys
        if tier_param_names[t]
    }

    out: Dict[str, float] = {}
    for t in tier_keys:
        g_clean = g_clean_by_tier.get(t)
        g_noisy = g_noisy_by_tier.get(t)
        if g_clean is None or g_noisy is None:
            out[f"gradient_reachability_{t}"] = float("nan")
            out[f"gradient_consistency_{t}"] = float("nan")
            if return_consistency_denom_numer:
                out[f"gradient_consistency_{t}_numer"] = float("nan")
                out[f"gradient_consistency_{t}_denom"] = float("nan")
            continue
        if g_clean.shape != g_noisy.shape:
            min_len = min(g_clean.numel(), g_noisy.numel())
            g_clean = g_clean.flatten()[:min_len]
            g_noisy = g_noisy.flatten()[:min_len]
        norm_clean = g_clean.norm().item()
        norm_noisy = g_noisy.norm().item()
        inner = (g_noisy * g_clean).sum().item()
        norm_sq_clean = (g_clean * g_clean).sum().item()
        denom = norm_sq_clean + eps
        out[f"gradient_reachability_{t}"] = float(norm_noisy / (norm_clean + norm_noisy + eps))
        out[f"gradient_consistency_{t}"] = float(inner / denom)
        if return_consistency_denom_numer:
            out[f"gradient_consistency_{t}_numer"] = float(inner)
            out[f"gradient_consistency_{t}_denom"] = float(denom)
    return out


def gradient_variance_domination_layerwise(
    model: nn.Module,
    batch: Tuple[torch.Tensor, ...],
    criterion: nn.Module,
    device: torch.device,
    model_name: str,
    K: int = 8,
    seed_base: Optional[int] = None,
    eps: float = DEFAULT_EPS,
    is_gru: bool = False,
) -> Dict[str, float]:
    """Layer-wise V per tier. Returns gradient_variance_domination_<tier>."""
    tier_keys = TIER_KEYS.get(model_name, [])
    if not tier_keys:
        return {}
    tier_param_names = _get_tier_param_names(model, model_name)

    if is_gru:
        data, target = batch[1].to(device), batch[0].to(device)
        lengths = batch[2].to(device) if len(batch) > 2 else None
    else:
        data, target = batch[0].to(device), batch[1].to(device)
        lengths = None

    model.train()
    grads_by_tier: Dict[str, List[torch.Tensor]] = {t: [] for t in tier_keys}
    for k in range(K):
        seed = (seed_base + k) if seed_base is not None else k
        _run_forward_backward(model, data, target, criterion, lengths=lengths, is_gru=is_gru, seed=seed)
        for t in tier_keys:
            if tier_param_names[t]:
                g = _get_param_grad_vector_for_tier(model, tier_param_names[t])
                if g is not None:
                    grads_by_tier[t].append(g.detach().clone())

    out: Dict[str, float] = {}
    for t in tier_keys:
        grads = grads_by_tier.get(t, [])
        if len(grads) < 2:
            out[f"gradient_variance_domination_{t}"] = float("nan")
            continue
        stack = torch.stack(grads)
        mean_g = stack.mean(dim=0)
        var = ((stack - mean_g) ** 2).sum().item() / len(grads)
        signal_sq = (mean_g * mean_g).sum().item()
        out[f"gradient_variance_domination_{t}"] = float(var / (signal_sq + eps))
    return out


def gradient_B_mean_layerwise(
    model: nn.Module,
    batch: Tuple[torch.Tensor, ...],
    criterion: nn.Module,
    device: torch.device,
    synth_layer_types: Tuple[type, ...],
    model_name: str,
    K: int = 8,
    seed_base: Optional[int] = None,
    eps: float = DEFAULT_EPS,
    is_gru: bool = False,
) -> Dict[str, float]:
    """Layer-wise B_mean per tier. Returns gradient_B_mean_<tier>."""
    tier_keys = TIER_KEYS.get(model_name, [])
    if not tier_keys:
        return {}
    tier_param_names = _get_tier_param_names(model, model_name)

    if is_gru:
        data, target = batch[1].to(device), batch[0].to(device)
        lengths = batch[2].to(device) if len(batch) > 2 else None
    else:
        data, target = batch[0].to(device), batch[1].to(device)
        lengths = None

    model.train()
    with _noise_toggled(model, synth_layer_types, False):
        _run_forward_backward(model, data, target, criterion, lengths=lengths, is_gru=is_gru, seed=seed_base)
    g_clean_by_tier = {
        t: _get_param_grad_vector_for_tier(model, tier_param_names[t])
        for t in tier_keys
        if tier_param_names[t]
    }

    grads_noisy_by_tier: Dict[str, List[torch.Tensor]] = {t: [] for t in tier_keys}
    for k in range(K):
        seed = (seed_base + k) if seed_base is not None else k
        with _noise_toggled(model, synth_layer_types, True):
            _run_forward_backward(model, data, target, criterion, lengths=lengths, is_gru=is_gru, seed=seed)
        for t in tier_keys:
            if tier_param_names[t]:
                g = _get_param_grad_vector_for_tier(model, tier_param_names[t])
                if g is not None:
                    grads_noisy_by_tier[t].append(g.detach().clone())

    out: Dict[str, float] = {}
    for t in tier_keys:
        g_clean = g_clean_by_tier.get(t)
        grads_noisy = grads_noisy_by_tier.get(t, [])
        if g_clean is None or not grads_noisy:
            out[f"gradient_B_mean_{t}"] = float("nan")
            continue
        norm_sq_clean = (g_clean * g_clean).sum().item()
        stack = torch.stack(grads_noisy)
        mean_g_noisy = stack.mean(dim=0)
        norm_sq_mean = (mean_g_noisy * mean_g_noisy).sum().item()
        out[f"gradient_B_mean_{t}"] = float(norm_sq_mean / (norm_sq_clean + eps))
    return out


# -----------------------------------------------------------------------------
# Perturbation stability / structural stability (S)
# -----------------------------------------------------------------------------


def perturbation_structural_stability(
    model: nn.Module,
    batch: Tuple[torch.Tensor, ...],
    device: torch.device,
    synth_layer_types: Tuple[type, ...],
    K: int = 8,
    seed_base: Optional[int] = None,
    eps: float = DEFAULT_EPS,
    is_gru: bool = False,
) -> Dict[str, float]:
    """
    Structural stability: S = ||E[Δ]||^2 / (E[||Δ||^2] + eps), Δ = h_noisy - h_clean.

    Forward only: one pass with noise disabled → h_clean; K passes with noise enabled → h_noisy^(k).
    Per layer we get Δ^(k), then E[Δ], E[||Δ||^2], S. Return mean S over layers.
    """
    if is_gru:
        data = batch[1].to(device)
        lengths = batch[2].to(device) if len(batch) > 2 else None
    else:
        data = batch[0].to(device)
        lengths = None

    model.train()
    captured_clean: Dict[str, torch.Tensor] = {}
    captured_noisy: List[Dict[str, torch.Tensor]] = []
    handles: List[torch.utils.hooks.RemovableHandle] = []

    def make_forward_hook(name: str, store: Dict[str, torch.Tensor]):
        def hook(_mod: nn.Module, _in: Tuple[torch.Tensor, ...], out: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) -> None:
            o = out[0] if isinstance(out, tuple) else out
            if o is not None:
                store[name] = o.detach().clone()
        return hook

    # Register forward hooks on synth layers
    for mod_name, mod in model.named_modules():
        if isinstance(mod, synth_layer_types):
            h = mod.register_forward_hook(make_forward_hook(mod_name, captured_clean))
            handles.append(h)

    if not handles:
        return {"perturbation_stability": float("nan")}

    try:
        # One forward with noise off
        with _noise_toggled(model, synth_layer_types, False):
            if is_gru and lengths is not None:
                model(data, lengths=lengths, seed=None)
            else:
                model(data, seed=None) if hasattr(model, "forward") else model(data)
        h_clean = {k: v.clone() for k, v in captured_clean.items()}

        # K forwards with noise on (different seeds)
        for k in range(K):
            captured_clean.clear()
            seed = (seed_base + k) if seed_base is not None else k
            with _noise_toggled(model, synth_layer_types, True):
                if is_gru and lengths is not None:
                    model(data, lengths=lengths, seed=seed)
                else:
                    model(data, seed=seed) if hasattr(model, "forward") else model(data)
            captured_noisy.append({k: v.clone() for k, v in captured_clean.items()})
    finally:
        for h in handles:
            h.remove()

    if not h_clean or not captured_noisy:
        return {"perturbation_stability": float("nan")}

    S_list: List[float] = []
    for name in h_clean:
        if name not in captured_noisy[0]:
            continue
        deltas = []
        for c in captured_noisy:
            d = (c[name] - h_clean[name]).flatten().float()
            deltas.append(d)
        stack_d = torch.stack(deltas)
        mean_d = stack_d.mean(dim=0)
        E_delta_sq = (stack_d * stack_d).sum(dim=1).mean().item()
        norm_mean_sq = (mean_d * mean_d).sum().item()
        S = norm_mean_sq / (E_delta_sq + eps)
        S_list.append(S)

    S_mean = float(np.mean(S_list)) if S_list else float("nan")
    return {"perturbation_stability": S_mean}


# -----------------------------------------------------------------------------
# Sign-coupled scaling P: proportion of sign(v^T z) on each side per batch
# -----------------------------------------------------------------------------


def sign_coupled_scaling_P(
    model: nn.Module,
    batch: Tuple[torch.Tensor, ...],
    device: torch.device,
    synth_layer_types: Tuple[type, ...],
    is_gru: bool = False,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """
    For noise_type sign_coupled_scaling only: per-batch proportion of sign(v^T z)
    falling on positive / negative / zero. z' = (1 + α sign(v^T z)) z; we use
    sign(v^T z') = sign(v^T z) and compute proportions from layer outputs z'.

    Returns:
        sign_coupled_P_positive: mean over layers of fraction with v^T z > 0
        sign_coupled_P_negative: mean over layers of fraction with v^T z < 0
        sign_coupled_P_zero: mean over layers of fraction with v^T z == 0
    """
    config = None
    for mod in model.modules():
        if isinstance(mod, synth_layer_types):
            config = getattr(mod, "config", None)
            break
    if config is None or getattr(config, "noise_type", "") != "sign_coupled_scaling":
        return {
            "sign_coupled_P_positive": float("nan"),
            "sign_coupled_P_negative": float("nan"),
            "sign_coupled_P_zero": float("nan"),
        }

    if is_gru:
        data = batch[1].to(device)
        lengths = batch[2].to(device) if len(batch) > 2 else None
    else:
        data = batch[0].to(device)
        lengths = None

    model.train()
    proportions: List[Dict[str, float]] = []
    handles: List[torch.utils.hooks.RemovableHandle] = []
    vec_cache = getattr(config, "_sign_scale_v_vectors", {})

    def make_hook():
        def hook(_mod: nn.Module, _in: Tuple[torch.Tensor, ...], out: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) -> None:
            o = out[0] if isinstance(out, tuple) else out
            if o is None:
                return
            d = o.shape[-1]
            v = vec_cache.get((d,))
            if v is None:
                return
            v = v.to(device=o.device, dtype=o.dtype)
            flat = o.reshape(-1, d)
            vTz = (flat * v.unsqueeze(0)).sum(dim=1)
            n = vTz.numel()
            p_pos = (vTz > 0).float().sum().item() / n
            p_neg = (vTz < 0).float().sum().item() / n
            p_zero = (vTz == 0).float().sum().item() / n
            proportions.append({"positive": p_pos, "negative": p_neg, "zero": p_zero})
        return hook

    for mod in model.modules():
        if isinstance(mod, synth_layer_types) and getattr(mod, "enable_noise", True):
            handles.append(mod.register_forward_hook(make_hook()))

    if not handles:
        return {
            "sign_coupled_P_positive": float("nan"),
            "sign_coupled_P_negative": float("nan"),
            "sign_coupled_P_zero": float("nan"),
        }

    try:
        if is_gru and lengths is not None:
            model(data, lengths=lengths, seed=seed) if hasattr(model, "forward") else model(data, lengths=lengths)
        else:
            model(data, seed=seed) if hasattr(model, "forward") else model(data)
    finally:
        for h in handles:
            h.remove()

    if not proportions:
        return {
            "sign_coupled_P_positive": float("nan"),
            "sign_coupled_P_negative": float("nan"),
            "sign_coupled_P_zero": float("nan"),
        }
    return {
        "sign_coupled_P_positive": float(np.mean([p["positive"] for p in proportions])),
        "sign_coupled_P_negative": float(np.mean([p["negative"] for p in proportions])),
        "sign_coupled_P_zero": float(np.mean([p["zero"] for p in proportions])),
    }


def collect_gradient_quality_metrics(
    model: nn.Module,
    batch: Tuple[torch.Tensor, ...],
    criterion: nn.Module,
    device: torch.device,
    is_gru: bool = False,
    seed: Optional[int] = None,
    eps: float = DEFAULT_EPS,
    k: int = 10,
    synth_layer_types: Tuple[type, ...] = (),
    relative_eps: bool = True,
) -> Dict[str, Union[float, List[Dict[str, float]]]]:
    """
    Run one forward-backward pass with backward hooks on given layer types to capture
    ∂L/∂z̃ (gradient at layer output), then compute dead_zone_ratio and effective_rank
    per layer and aggregated.

    Args:
        model: Model that has noise layers (e.g. SynthNoiseModel or model with SynthNoiseLinear/Conv2d).
        batch: (data, target) or (data, target, lengths) for AG News.
        criterion: Loss criterion.
        device: Device.
        is_gru: If True, unpack batch as (texts, labels, lengths) and call model(..., lengths=lengths).
        seed: Optional seed for model forward (noise).
        eps: Threshold for dead zone ratio.
        k: Top-k for energy ratio.
        synth_layer_types: Tuple of layer classes to hook (e.g. (SynthNoiseLinear, SynthNoiseConv2d)).
        relative_eps: If True (default), dead-zone threshold is eps * max|grad| so the metric scales with
            gradient magnitude and differs across noise types; if False, absolute eps can cause 0/1 saturation.

    Returns:
        Dict with:
          - dead_zone_ratio_element_mean, dead_zone_ratio_channel_mean
          - grad_top_k_energy_ratio_mean, grad_effective_rank_mean
          - per_layer: list of dicts (one per hooked layer) with same keys + layer_name.
    """
    if not synth_layer_types:
        return {
            "dead_zone_ratio_element_mean": 0.0,
            "dead_zone_ratio_channel_mean": 0.0,
            "grad_top_k_energy_ratio_mean": 0.0,
            "grad_effective_rank_mean": 0.0,
            "per_layer": [],
        }

    captured: Dict[str, torch.Tensor] = {}
    handles: List[torch.utils.hooks.RemovableHandle] = []

    def make_hook(name: str):
        def hook(_module: nn.Module, _grad_input: Tuple[torch.Tensor, ...], grad_output: Tuple[torch.Tensor, ...]) -> None:
            if grad_output and grad_output[0] is not None:
                captured[name] = grad_output[0].detach().clone()

        return hook

    # Register hooks on all layers of given types
    for mod_name, mod in model.named_modules():
        if isinstance(mod, synth_layer_types):
            if not getattr(mod, "enable_noise", True):
                continue
            h = mod.register_full_backward_hook(make_hook(mod_name))
            handles.append(h)

    if not handles:
        return {
            "dead_zone_ratio_element_mean": 0.0,
            "dead_zone_ratio_channel_mean": 0.0,
            "grad_top_k_energy_ratio_mean": 0.0,
            "grad_effective_rank_mean": 0.0,
            "per_layer": [],
        }

    model.train()

    # Unpack batch (same convention as _unpack_batch: AG News batch is (labels, texts, lengths))
    if is_gru:
        labels, texts = batch[0], batch[1]
        lengths = batch[2] if len(batch) > 2 else None
        data = texts.to(device)
        target = labels.to(device)
        if lengths is not None:
            lengths = lengths.to(device)
    else:
        data = batch[0].to(device)
        target = batch[1].to(device)
        lengths = None

    try:
        model.zero_grad()
        if hasattr(model, "forward"):
            if is_gru and lengths is not None:
                out = model(data, lengths=lengths, seed=seed)
            else:
                out = model(data, seed=seed)
        else:
            out = model(data)
        loss = criterion(out, target)
        loss.backward()
    finally:
        for h in handles:
            h.remove()

    # Compute metrics per layer
    per_layer: List[Dict[str, float]] = []
    for name, g in captured.items():
        m = compute_layer_gradient_metrics(g, eps=eps, k=k, relative_eps=relative_eps)
        m["layer_name"] = name
        per_layer.append(m)

    if not per_layer:
        return {
            "dead_zone_ratio_element_mean": 0.0,
            "dead_zone_ratio_channel_mean": 0.0,
            "grad_top_k_energy_ratio_mean": 0.0,
            "grad_effective_rank_mean": 0.0,
            "per_layer": [],
        }

    result = {
        "dead_zone_ratio_element_mean": float(np.mean([p["dead_zone_ratio_element"] for p in per_layer])),
        "dead_zone_ratio_channel_mean": float(np.mean([p["dead_zone_ratio_channel"] for p in per_layer])),
        "grad_top_k_energy_ratio_mean": float(np.mean([p["grad_top_k_energy_ratio"] for p in per_layer])),
        "grad_effective_rank_mean": float(np.mean([p["grad_effective_rank"] for p in per_layer])),
        "per_layer": per_layer,
    }
    return result
