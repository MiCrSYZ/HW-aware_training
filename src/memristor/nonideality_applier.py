"""
Offline (weights-only) non-ideality applier.

Goal:
- For any trained model, "damage" its weights *once* before inference:
    model_dmg = apply_non_idealities(model, device_model, t=0, seed=123)

This reproduces the same weight->conductance->nonidealities->effective_weight path used in
hardware_linear_forward_adaptive, but WITHOUT wrapping forward.

Important:
- This only captures weight-side / conductance-side non-idealities (variability/read-noise/stuck/drift/...).
- It does NOT capture activation/output-side effects (ADC quant, some IR-drop modes, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple, Type, Union

import copy
import torch
import torch.nn as nn

try:
    from .device_model import MemristorDeviceModel
except ImportError:
    from src.memristor.device_model import MemristorDeviceModel


@dataclass
class ApplyStats:
    name: str
    module_type: str
    weight_shape: Tuple[int, ...]
    w_abs_max_before: float
    w_abs_max_after: float
    delta_l2: float
    delta_rel_l2: float


def _effective_weight_from_device(
    W: torch.Tensor,
    device_model: MemristorDeviceModel,
    t: int = 0,
    seed: Optional[int] = None,
    use_diff_seeds: bool = True,
    clamp_scale: Tuple[float, float] = (1e-3, 1e6),
    clamp_weight_to_wrange: bool = False,
) -> torch.Tensor:
    """
    Mimic memristor_wrappers.hardware_linear_forward_adaptive weight path:

        Gp, Gn, max_abs = map_weights_to_conductance_diff_adaptive(W)
        Gp_noisy = apply_nonidealities(Gp, seed=seed)
        Gn_noisy = apply_nonidealities(Gn, seed=seed+1)
        W_eff = (Gp_noisy - Gn_noisy) * (max_abs / (G_max - G_min))

    Ref: memristor_wrappers.py lines ~21-40. :contentReference[oaicite:1]{index=1}
    """
    # 1) differential mapping (already clamps W internally)
    Gp, Gn, max_abs = device_model.map_weights_to_conductance_diff_adaptive(W)

    # 2) apply conductance-side nonidealities
    Gp_noisy = device_model.apply_nonidealities(Gp, t=t, seed=seed)
    if seed is None or not use_diff_seeds:
        Gn_seed = seed
    else:
        Gn_seed = seed + 1
    Gn_noisy = device_model.apply_nonidealities(Gn, t=t, seed=Gn_seed)

    # 3) convert back to effective weight scale
    G_range = (device_model.G_max - device_model.G_min)
    scale = max_abs / (G_range + 1e-12)  # broadcastable scalar tensor
    scale = torch.clamp(scale, min=clamp_scale[0], max=clamp_scale[1])
    W_eff = (Gp_noisy - Gn_noisy) * scale

    # Optional: clamp to original weight range if you want strict bounded weights
    if clamp_weight_to_wrange and hasattr(device_model, "wmin") and hasattr(device_model, "wmax"):
        W_eff = torch.clamp(W_eff, device_model.wmin, device_model.wmax)

    return W_eff


def apply_non_idealities_inplace_(
    model: nn.Module,
    device_model: MemristorDeviceModel,
    t: int = 0,
    seed: Optional[int] = None,
    module_types: Tuple[Type[nn.Module], ...] = (nn.Linear, nn.Conv2d),
    include_embeddings: bool = False,
    use_diff_seeds: bool = True,
    clamp_weight_to_wrange: bool = False,
) -> Dict[str, ApplyStats]:
    """
    In-place apply weights-only non-idealities to selected modules.

    Returns a stats dict keyed by parameter name/module name for quick sanity checks.
    """
    stats: Dict[str, ApplyStats] = {}
    seed_i = 0

    for name, module in model.named_modules():
        # Optionally include Embedding as "weight-only" too
        if include_embeddings and isinstance(module, nn.Embedding):
            W = module.weight
            if not isinstance(W, torch.Tensor):
                continue

            s = None if seed is None else seed + seed_i
            seed_i += 1

            W_before = W.detach()
            W_eff = _effective_weight_from_device(
                W=W,
                device_model=device_model,
                t=t,
                seed=s,
                use_diff_seeds=use_diff_seeds,
                clamp_weight_to_wrange=clamp_weight_to_wrange,
            ).detach()

            with torch.no_grad():
                module.weight.copy_(W_eff)

            delta = (W_eff - W_before).reshape(-1)
            denom = W_before.reshape(-1).norm(p=2).clamp(min=1e-12)
            stats[name + ".weight"] = ApplyStats(
                name=name + ".weight",
                module_type="Embedding",
                weight_shape=tuple(W_before.shape),
                w_abs_max_before=float(W_before.abs().max().cpu()),
                w_abs_max_after=float(W_eff.abs().max().cpu()),
                delta_l2=float(delta.norm(p=2).cpu()),
                delta_rel_l2=float((delta.norm(p=2) / denom).cpu()),
            )
            continue

        if not isinstance(module, module_types):
            continue

        if not hasattr(module, "weight") or module.weight is None:
            continue

        W = module.weight
        if not isinstance(W, torch.Tensor):
            continue

        # Different layers can get different seeds, but deterministic overall
        s = None if seed is None else seed + seed_i
        seed_i += 1

        W_before = W.detach()
        W_eff = _effective_weight_from_device(
            W=W,
            device_model=device_model,
            t=t,
            seed=s,
            use_diff_seeds=use_diff_seeds,
            clamp_weight_to_wrange=clamp_weight_to_wrange,
        ).detach()

        with torch.no_grad():
            module.weight.copy_(W_eff)

        delta = (W_eff - W_before).reshape(-1)
        denom = W_before.reshape(-1).norm(p=2).clamp(min=1e-12)
        stats[name + ".weight"] = ApplyStats(
            name=name + ".weight",
            module_type=module.__class__.__name__,
            weight_shape=tuple(W_before.shape),
            w_abs_max_before=float(W_before.abs().max().cpu()),
            w_abs_max_after=float(W_eff.abs().max().cpu()),
            delta_l2=float(delta.norm(p=2).cpu()),
            delta_rel_l2=float((delta.norm(p=2) / denom).cpu()),
        )

    return stats


def apply_non_idealities(
    model: nn.Module,
    device_model: MemristorDeviceModel,
    t: int = 0,
    seed: Optional[int] = None,
    module_types: Tuple[Type[nn.Module], ...] = (nn.Linear, nn.Conv2d),
    include_embeddings: bool = False,
    inplace: bool = False,
    use_diff_seeds: bool = True,
    clamp_weight_to_wrange: bool = False,
) -> Tuple[nn.Module, Dict[str, ApplyStats]]:
    """
    Convenience wrapper.
    - inplace=False: returns a deep-copied damaged model
    - inplace=True: modifies model directly

    Returns (model_damaged, stats)
    """
    if not inplace:
        model = copy.deepcopy(model)

    stats = apply_non_idealities_inplace_(
        model=model,
        device_model=device_model,
        t=t,
        seed=seed,
        module_types=module_types,
        include_embeddings=include_embeddings,
        use_diff_seeds=use_diff_seeds,
        clamp_weight_to_wrange=clamp_weight_to_wrange,
    )
    return model, stats


def apply_non_idealities_to_state_dict(
    state_dict: Dict[str, torch.Tensor],
    device_model: MemristorDeviceModel,
    t: int = 0,
    seed: Optional[int] = None,
    key_filter: Optional[Iterable[str]] = None,
    use_diff_seeds: bool = True,
    clamp_weight_to_wrange: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Offline apply to a plain state_dict (useful if you want to save a "damaged checkpoint").

    - key_filter: if provided, only keys containing any substring in key_filter will be processed.
      Example: key_filter=("weight",) to process all weights; or ("attn", "mlp") for transformers.
    """
    out = {}
    seed_i = 0
    for k, v in state_dict.items():
        if not torch.is_tensor(v):
            out[k] = v
            continue
        if key_filter is not None:
            if not any(s in k for s in key_filter):
                out[k] = v
                continue
        # Typical pattern: only process ".weight" tensors
        if not k.endswith("weight"):
            out[k] = v
            continue

        s = None if seed is None else seed + seed_i
        seed_i += 1

        out[k] = _effective_weight_from_device(
            W=v,
            device_model=device_model,
            t=t,
            seed=s,
            use_diff_seeds=use_diff_seeds,
            clamp_weight_to_wrange=clamp_weight_to_wrange,
        ).detach()
    return out
