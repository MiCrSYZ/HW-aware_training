"""
Synthetic noise wrappers for neural network layers.

This module provides noise injection mechanisms that do NOT use weight-to-conductance mapping.
Instead, noise is applied directly to weights or outputs.

Noise types
-----------
Weight-domain:
  iid_multiplicative       W' = W ⊙ (1 + ε),  ε ~ N(0,σ²)
  heavy_tail               W' = W ⊙ (1 + αε),  ε ~ t_ν,  ν ≤ 2

Output-domain, compensable (forward perturbation + consistent or zero-mean backward):
  heavy_tail_output        z' = z ⊙ (1 + αε),  ε ~ t_ν
  decoupled_consistent     z' = (1+ε)z,  ε ~ N(0,σ²),  ∂z'/∂z = detach(1+ε)
  decoupled_inconsistent   z' = z + ε tanh(z),  ε ~ N(0,σ²),  ∂z'/∂z = 1 + ε sech²(z)
  input_dependent          z' = s(z)⊙z,  s(z) = 1 + α tanh(v^T z),  STE backward
  coupled_consistent       same forward as input_dependent,  ∂z'/∂W = detach(s(z))·∂z/∂W
  coupled_inconsistent     same forward,  ∂z'/∂W = s(z)·∂z/∂W  (bounded bias)

Output-domain, MAYBE NOT compensable — gradient structure broken:
  gradient_degenerate      ADC quantization with gradient blocking
  adversarial_direction_bias  forward=identity, backward: g += β‖g‖d  (frozen d)
  sign_gradient_corruption    forward=identity, backward: flip element signs w.p. p
  saturation_collapse      z' = α tanh(γz),  honest backward → vanishing gradient

Output-domain, MAYBE NOT compensable — forward structure broken:
  frozen_additive_drift    z' = z + β|z|d  (element-wise) or z + β‖z‖d  (norm-scalar)
                           d frozen → systematic Jacobian drift, not zero-mean
  sign_coupled_scaling     z' = (1 + α sign(v^T z)) z
                           sign Jacobian is 0 a.e. → structured dead-neuron pattern
  rank_collapse            z' = P_k z + ε_fill,  P_k frozen rank-k projection
                           gradient in (I-P_k) subspace permanently zeroed
  deterministic_clip       z' = clip(z, [-c, c])
                           dead zone kills gradient for |z| ≥ c

Control pairs (same δ, compensable):
  frozen_additive_drift  with drift_frozen=False  (d re-sampled → zero-mean)
  deterministic_clip     with clip_dither=True     (dithered → stochastic rounding)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)

# Backward-only diagnostic: set SYNTH_NOISE_DIAGNOSTIC=1 to accumulate sign_corrupt call count and flip ratio
_BACKWARD_DIAGNOSTIC: Dict[str, float] = {
    "sign_corrupt_calls": 0.0,
    "sign_corrupt_flip_sum": 0.0,
    "adv_bias_calls": 0.0,
}


def get_and_reset_backward_diagnostic() -> Dict[str, float]:
    """Return current backward diagnostic (sign_corrupt calls, mean flip ratio) and reset counters."""
    out = {
        "sign_corrupt_calls": _BACKWARD_DIAGNOSTIC["sign_corrupt_calls"],
        "sign_corrupt_flip_ratio": (
            _BACKWARD_DIAGNOSTIC["sign_corrupt_flip_sum"] / _BACKWARD_DIAGNOSTIC["sign_corrupt_calls"]
            if _BACKWARD_DIAGNOSTIC["sign_corrupt_calls"] > 0
            else 0.0
        ),
        "adv_bias_calls": _BACKWARD_DIAGNOSTIC["adv_bias_calls"],
    }
    _BACKWARD_DIAGNOSTIC["sign_corrupt_calls"] = 0.0
    _BACKWARD_DIAGNOSTIC["sign_corrupt_flip_sum"] = 0.0
    _BACKWARD_DIAGNOSTIC["adv_bias_calls"] = 0.0
    return out


# =============================================================================
# Config
# =============================================================================

class SynthNoiseConfig:
    """Configuration for synthetic noise injection."""

    def __init__(
        self,
        noise_type: str = 'none',
        # ── weight-domain ──────────────────────────────────────────────────
        variability_sigma: float = 0.05,
        heavy_tail_alpha: float = 0.1,
        heavy_tail_nu: float = 2.0,
        # ── output-domain compensable ──────────────────────────────────────
        input_dependent_alpha: float = 0.1,
        decoupled_consistent_sigma: float = 0.05,
        decoupled_inconsistent_sigma: float = 0.05,
        coupled_consistent_alpha: float = 0.1,
        coupled_inconsistent_alpha: float = 0.1,
        # ── gradient_degenerate ────────────────────────────────────────────
        adc_bits: float = 8.0,
        enable_adc: bool = False,
        adc_backward_mode: Optional[str] = None,  # None = use compensation_in_backward; 'ste' | 'stopgrad' (ablation)
        # ── adversarial_direction_bias (backward-only) ────────────────────
        adv_direction_beta: float = 1.0,
        adv_direction_frozen: bool = True,   # False = resample d each call (zero-mean, ablation)
        adv_direction_random_sign: bool = False,  # True = multiply drift by ±1 each call (zero-mean, ablation)
        # ── sign_gradient_corruption (backward-only) ──────────────────────
        sign_corrupt_p: float = 0.5,
        sign_corrupt_mode: str = 'flip',  # 'flip' | 'noise_positive_align' (ablation: low cos_sim but E⟨g′,g⟩>0)
        sign_corrupt_noise_sigma: float = 1.0,  # for noise_positive_align: g' = g + σ*N(0,I)
        # ── saturation_collapse ────────────────────────────────────────────
        saturation_gamma: float = 5.0,
        saturation_alpha: float = 1.0,
        # ── frozen_additive_drift (new) ────────────────────────────────────
        drift_beta: float = 0.3,
        drift_use_norm: bool = False,   # False=element-wise |z|, True=scalar ‖z‖
        drift_frozen: bool = True,      # False = control (zero-mean, compensable)
        drift_resample_when_eval: bool = False,  # True: train like frozen; val/test resample d each call (A3)
        drift_d_mean: float = 0.0,      # mean bias added to d before normalization (ablation)
        # ── sign_coupled_scaling (new) ─────────────────────────────────────
        sign_scale_alpha: float = 0.5,
        sign_scale_v_resample: bool = False,  # True = resample v each forward (control vs frozen v)
        # ── rank_collapse (new) ────────────────────────────────────────────
        rank_k: int = 4,
        rank_fill_sigma: float = 0.0,   # set by calibration to match δ target
        rank_resample: bool = False,    # True = resample P_k every call (per step/batch); False = fixed projector
        rank_resample_when_eval: bool = False,  # True: train fixed P_k; val/test resample P_k (A3 rank suite)
        # ── deterministic_clip (new) ───────────────────────────────────────
        clip_c: float = 1.0,
        clip_c_eval: Optional[float] = None,  # if set, val/test use this c (train keeps clip_c)
        clip_dither: bool = False,      # True = compensable control
        # ── input_dependent / coupled_* (shared forward path) ─────────────
        input_dependent_v_resample: bool = False,  # True = resample v each forward (coupled_inconsistent ablation)
        # ── shared ────────────────────────────────────────────────────────
        seed: Optional[int] = None,
        compensation_in_backward: bool = True,
        backward_corruption_at: Optional[str] = None,  # None | 'per_layer' | 'logits' (single-point, unbypassable)
    ):
        self.noise_type = noise_type
        self.variability_sigma = variability_sigma
        self.heavy_tail_alpha = heavy_tail_alpha
        self.heavy_tail_nu = heavy_tail_nu
        self.input_dependent_alpha = input_dependent_alpha
        self.decoupled_consistent_sigma = decoupled_consistent_sigma
        self.decoupled_inconsistent_sigma = decoupled_inconsistent_sigma
        self.coupled_consistent_alpha = coupled_consistent_alpha
        self.coupled_inconsistent_alpha = coupled_inconsistent_alpha
        self.adc_bits = adc_bits
        self.enable_adc = enable_adc
        self.adc_backward_mode = adc_backward_mode
        self.adv_direction_beta = adv_direction_beta
        self.adv_direction_frozen = adv_direction_frozen
        self.adv_direction_random_sign = adv_direction_random_sign
        self.sign_corrupt_p = sign_corrupt_p
        self.sign_corrupt_mode = sign_corrupt_mode
        self.sign_corrupt_noise_sigma = sign_corrupt_noise_sigma
        self.saturation_gamma = saturation_gamma
        self.saturation_alpha = saturation_alpha
        self.drift_beta = drift_beta
        self.drift_use_norm = drift_use_norm
        self.drift_frozen = drift_frozen
        self.drift_resample_when_eval = drift_resample_when_eval
        self.drift_d_mean = drift_d_mean
        self.sign_scale_alpha = sign_scale_alpha
        self.sign_scale_v_resample = sign_scale_v_resample
        self.rank_k = rank_k
        self.rank_fill_sigma = rank_fill_sigma
        self.rank_resample = rank_resample
        self.rank_resample_when_eval = rank_resample_when_eval
        self.clip_c = clip_c
        self.clip_c_eval = clip_c_eval
        self.clip_dither = clip_dither
        self.input_dependent_v_resample = input_dependent_v_resample
        self.seed = seed
        self.compensation_in_backward = compensation_in_backward
        self.backward_corruption_at = backward_corruption_at

        # frozen vector / matrix caches
        self._input_dependent_v_vectors: Dict[tuple, torch.Tensor] = {}
        self._adv_dir_vectors:           Dict[tuple, torch.Tensor] = {}
        self._drift_d_vectors:           Dict[tuple, torch.Tensor] = {}
        self._sign_scale_v_vectors:      Dict[tuple, torch.Tensor] = {}
        self._rank_proj:                 Dict[tuple, torch.Tensor] = {}
        # Runtime seed injected by wrapper forward to support models whose
        # forward() does not accept explicit seed argument.
        self._runtime_seed: Optional[int] = None


def clear_synth_noise_template_caches(config: SynthNoiseConfig) -> None:
    """
    Clear frozen-template caches (drift d, rank projectors, etc.) on the shared
    SynthNoiseConfig. Templates are not stored in checkpoints; after loading weights
    from disk, call this so the next forward rebuilds vectors from deterministic
    per-key generators (e.g. _frozen_vec with fixed seed), not stale in-memory tensors.
    """
    config._input_dependent_v_vectors.clear()
    config._adv_dir_vectors.clear()
    config._drift_d_vectors.clear()
    config._sign_scale_v_vectors.clear()
    config._rank_proj.clear()
    config._runtime_seed = None


# =============================================================================
# Helpers
# =============================================================================

def _frozen_vec(cache: Dict, key: tuple, dim: int, device, dtype,
                seed_val: Optional[int], seed_offset: int) -> torch.Tensor:
    """Fetch or create a frozen unit vector of length `dim`."""
    if key not in cache:
        if seed_val is not None:
            gen = torch.Generator(device=device)
            gen.manual_seed((seed_val + seed_offset) % 2**31)
            v = torch.randn(dim, generator=gen, device=device, dtype=dtype)
        else:
            v = torch.randn(dim, device=device, dtype=dtype)
        cache[key] = (v / (v.norm() + 1e-8)).detach()
    return cache[key].to(device=device, dtype=dtype)


# =============================================================================
# Existing perturbations (unchanged)
# =============================================================================

def apply_iid_multiplicative_distortion(W, config, seed=None):
    if config.variability_sigma <= 0:
        return W
    if seed is not None:
        g = torch.Generator(device=W.device); g.manual_seed(seed)
        eps = torch.randn(W.shape, generator=g, device=W.device, dtype=W.dtype, requires_grad=False)
    else:
        eps = torch.randn_like(W)
    return W * (1.0 + eps * config.variability_sigma)


def apply_heavy_tail_forward_noise(W, config, seed=None):
    t = _sample_student_t(W.shape, config.heavy_tail_nu, W.device, W.dtype, seed=seed)
    return W * (1.0 + config.heavy_tail_alpha * t)


def _sample_student_t(shape, nu, device, dtype, seed=None):
    if seed is not None:
        cpu_st = torch.get_rng_state()
        cuda_st = torch.cuda.get_rng_state(device) if device.type == 'cuda' else None
        torch.manual_seed(seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)
    Z    = torch.randn(shape, device=device, dtype=dtype)
    chi2 = torch.clamp(
        torch.distributions.Gamma(
            torch.tensor(nu / 2., device=device, dtype=dtype),
            torch.tensor(0.5,     device=device, dtype=dtype),
        ).sample(shape), min=1e-8)
    t = Z / torch.sqrt(chi2 / nu)
    if seed is not None:
        torch.set_rng_state(cpu_st)
        if cuda_st is not None:
            torch.cuda.set_rng_state(cuda_st, device)
    return t


def apply_heavy_tail_output(out, config, seed=None):
    t     = _sample_student_t(out.shape, config.heavy_tail_nu, out.device, out.dtype, seed=seed)
    scale = 1.0 + config.heavy_tail_alpha * t
    if config.compensation_in_backward:
        sd = scale.detach()
        return out * sd + (out * scale - out * sd).detach()
    return out * scale


def apply_decoupled_consistent(out, config, seed=None):
    s = config.decoupled_consistent_sigma
    if s <= 0: return out
    if seed is not None:
        g = torch.Generator(device=out.device); g.manual_seed(seed)
        eps = torch.randn(out.shape, generator=g, device=out.device, dtype=out.dtype, requires_grad=False)
    else:
        eps = torch.randn_like(out)
    scale = 1.0 + eps * s
    return out * scale.detach() + (out * scale - out * scale.detach()).detach()


def apply_decoupled_inconsistent(out, config, seed=None):
    s = config.decoupled_inconsistent_sigma
    if s <= 0: return out
    if seed is not None:
        g = torch.Generator(device=out.device); g.manual_seed(seed)
        eps = torch.randn(out.shape, generator=g, device=out.device, dtype=out.dtype, requires_grad=False)
    else:
        eps = torch.randn_like(out)
    return out + eps * s * torch.tanh(out)


def apply_input_dependent_scale_bias(out, config, seed=None, compensation_override=None, alpha_override=None):
    alpha = alpha_override if alpha_override is not None else config.input_dependent_alpha
    d     = out.shape[-1]
    key   = (d,)
    sv    = seed if seed is not None else config.seed
    if getattr(config, "input_dependent_v_resample", False):
        if seed is not None:
            gen = torch.Generator(device=out.device)
            gen.manual_seed((seed + hash(key) % 1_000_000 + 0xD0DD) % 2**31)
            v = torch.randn(d, generator=gen, device=out.device, dtype=out.dtype)
        else:
            v = torch.randn(d, device=out.device, dtype=out.dtype)
        v = (v / (v.norm() + 1e-8)).detach()
    else:
        sv_key = sv if sv is not None else 0
        key = (d, sv_key)
        v = _frozen_vec(config._input_dependent_v_vectors, key, d, out.device, out.dtype,
                        sv, hash(key) % 1_000_000)
    vTz   = (out * (v.unsqueeze(0) if out.dim() == 2 else v.view(1, -1))).sum(-1, keepdim=True)
    s_z   = 1.0 + alpha * torch.tanh(vTz)
    use_det = compensation_override if compensation_override is not None else config.compensation_in_backward
    if use_det:
        sd = s_z.detach()
        return out * sd + (out * s_z - out * sd).detach()
    return out * s_z


def apply_gradient_degenerate_perturbations(out, config, training=True):
    if config.noise_type != 'gradient_degenerate':
        return out
    if not config.enable_adc:
        logger.warning("gradient_degenerate requires enable_adc=True. Skipping.")
        return out
    xmin  = out.min(dim=-1, keepdim=True)[0]
    xmax  = out.max(dim=-1, keepdim=True)[0]
    scale = (2.**config.adc_bits - 1.) / torch.clamp(xmax - xmin, min=1e-12)
    q     = torch.round((out - xmin) * scale) / scale + xmin
    if training:
        # Ablation: explicit adc_backward_mode overrides compensation_in_backward
        mode = getattr(config, 'adc_backward_mode', None)
        if mode is not None:
            use_ste = (mode == 'ste')
        else:
            use_ste = config.compensation_in_backward
        q = (out + (q - out).detach()) if use_ste else q.detach()
    return q


class _AdversarialDirBiasFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, out, d, beta, sign_scale=None):
        # sign_scale: tensor scalar +1 or -1 for zero-mean ablation (E[drift]=0)
        ctx.save_for_backward(d)
        ctx.beta = beta
        ctx.sign_scale = 1.0 if sign_scale is None else sign_scale.item()
        return out.clone()
    @staticmethod
    def backward(ctx, g):
        if os.environ.get("SYNTH_NOISE_DIAGNOSTIC", "").strip() in ("1", "true", "yes"):
            _BACKWARD_DIAGNOSTIC["adv_bias_calls"] += 1.0
        (d,) = ctx.saved_tensors
        n  = g.numel()
        df = (d[:n] if d.numel() >= n else d.repeat((n // d.numel()) + 1)[:n]).reshape(g.shape)
        df = df / (df.norm() + 1e-8)
        return g + ctx.beta * g.norm() * (ctx.sign_scale * df), None, None, None

def apply_adversarial_direction_bias(out, config, seed=None):
    beta = config.adv_direction_beta
    if beta == 0. or not out.requires_grad:
        return out
    key = (out.numel(),)
    if getattr(config, 'adv_direction_frozen', True):
        sv = seed if seed is not None else config.seed
        d = _frozen_vec(config._adv_dir_vectors, key, key[0], out.device, out.dtype, sv, 0xDEAD)
    else:
        # Resampled d each call: use seed only when caller passes one; else fresh randomness
        # so resampled is not identical to frozen (same fix as rank_collapse).
        if seed is not None:
            gen = torch.Generator(device=out.device)
            gen.manual_seed((seed + 0xDEAD) % 2**31)
            d = torch.randn(key[0], generator=gen, device=out.device, dtype=out.dtype)
        else:
            d = torch.randn(key[0], device=out.device, dtype=out.dtype)
        d = (d / (d.norm() + 1e-8)).detach()
    sign_scale = None
    if getattr(config, 'adv_direction_random_sign', False):
        # Draw ±1 each call (no seed) so E[drift]=0 over steps — ablation for zero-mean bias
        s = 2 * torch.randint(0, 2, (1,), device=out.device, dtype=out.dtype) - 1.0
        sign_scale = s
    return _AdversarialDirBiasFunction.apply(out, d, beta, sign_scale)


class _SignGradCorruptFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, out, p): ctx.p = p; return out.clone()
    @staticmethod
    def backward(ctx, g):
        flip = torch.bernoulli(torch.full_like(g, ctx.p))
        if os.environ.get("SYNTH_NOISE_DIAGNOSTIC", "").strip() in ("1", "true", "yes"):
            _BACKWARD_DIAGNOSTIC["sign_corrupt_calls"] += 1.0
            _BACKWARD_DIAGNOSTIC["sign_corrupt_flip_sum"] += flip.float().mean().item()
        return g * (1.0 - 2.0 * flip), None


class _SignGradNoisePositiveAlignFunction(torch.autograd.Function):
    """Ablation: g' = g + σ*N(0,I). E[g']=g so E⟨g′,g⟩>0; cos_sim can be low when σ is large."""
    @staticmethod
    def forward(ctx, out, sigma): ctx.sigma = sigma; return out.clone()
    @staticmethod
    def backward(ctx, g):
        if ctx.sigma <= 0.:
            return g, None
        noise = torch.randn_like(g, device=g.device, dtype=g.dtype)
        return g + ctx.sigma * noise, None


def apply_sign_gradient_corruption(out, config, seed=None):
    if not out.requires_grad:
        return out
    mode = getattr(config, 'sign_corrupt_mode', 'flip')
    if mode == 'noise_positive_align':
        sigma = getattr(config, 'sign_corrupt_noise_sigma', 1.0)
        if sigma <= 0.:
            return out
        return _SignGradNoisePositiveAlignFunction.apply(out, sigma)
    # default: flip
    p = config.sign_corrupt_p
    if p <= 0.:
        return out
    return _SignGradCorruptFunction.apply(out, p)


# =============================================================================
# Logits-level backward corruption (single point, cannot be bypassed by residual)
# =============================================================================

class _LogitsSignCorruptFunction(torch.autograd.Function):
    """Apply sign_gradient_corruption at logits only. Forward = identity; backward = flip g w.p. p."""
    @staticmethod
    def forward(ctx, logits, p): ctx.p = p; return logits.clone()
    @staticmethod
    def backward(ctx, g):
        flip = torch.bernoulli(torch.full_like(g, ctx.p))
        if os.environ.get("SYNTH_NOISE_DIAGNOSTIC", "").strip() in ("1", "true", "yes"):
            _BACKWARD_DIAGNOSTIC["sign_corrupt_calls"] += 1.0
            _BACKWARD_DIAGNOSTIC["sign_corrupt_flip_sum"] += flip.float().mean().item()
        return g * (1.0 - 2.0 * flip), None


class _LogitsAdvBiasFunction(torch.autograd.Function):
    """Apply adversarial_direction_bias at logits only. Forward = identity; backward = g + β‖g‖d."""
    @staticmethod
    def forward(ctx, logits, d, beta, sign_scale):
        ctx.save_for_backward(d)
        ctx.beta = beta
        ctx.sign_scale = 1.0 if sign_scale is None else sign_scale.item()
        return logits.clone()
    @staticmethod
    def backward(ctx, g):
        if os.environ.get("SYNTH_NOISE_DIAGNOSTIC", "").strip() in ("1", "true", "yes"):
            _BACKWARD_DIAGNOSTIC["adv_bias_calls"] += 1.0
        (d,) = ctx.saved_tensors
        n = g.numel()
        df = (d[:n] if d.numel() >= n else d.repeat((n // d.numel()) + 1)[:n]).reshape(g.shape)
        df = df / (df.norm() + 1e-8)
        return g + ctx.beta * g.norm() * (ctx.sign_scale * df), None, None, None


def apply_logits_backward_corruption(logits: torch.Tensor, config: SynthNoiseConfig, seed: Optional[int] = None) -> torch.Tensor:
    """
    Apply backward-only corruption at the logits (single point). No residual can bypass this.
    Use with backward_corruption_at='logits' and do NOT apply sign/adv at per-layer.
    """
    if not logits.requires_grad:
        return logits
    nt = config.noise_type
    if nt == "sign_gradient_corruption":
        if getattr(config, "sign_corrupt_mode", "flip") != "flip":
            return logits
        p = config.sign_corrupt_p
        if p <= 0.0:
            return logits
        return _LogitsSignCorruptFunction.apply(logits, p)
    if nt == "adversarial_direction_bias":
        beta = config.adv_direction_beta
        if beta == 0.0:
            return logits
        key = (logits.numel(),)
        if getattr(config, "adv_direction_frozen", True):
            sv = seed if seed is not None else config.seed
            d = _frozen_vec(config._adv_dir_vectors, key, key[0], logits.device, logits.dtype, sv, 0xDEAD)
        else:
            if seed is not None:
                gen = torch.Generator(device=logits.device)
                gen.manual_seed((seed + 0xDEAD) % 2**31)
                d = torch.randn(key[0], generator=gen, device=logits.device, dtype=logits.dtype)
            else:
                d = torch.randn(key[0], device=logits.device, dtype=logits.dtype)
            d = (d / (d.norm() + 1e-8)).detach()
        sign_scale = None
        if getattr(config, "adv_direction_random_sign", False):
            s = 2 * torch.randint(0, 2, (1,), device=logits.device, dtype=logits.dtype) - 1.0
            sign_scale = s
        return _LogitsAdvBiasFunction.apply(logits, d, beta, sign_scale)
    return logits


def apply_saturation_collapse(out, config):
    if config.saturation_gamma <= 0.: return out
    return config.saturation_alpha * torch.tanh(config.saturation_gamma * out)


# =============================================================================
# perturbations — forward structure broken
# =============================================================================

def apply_frozen_additive_drift(out: torch.Tensor, config: SynthNoiseConfig,
                                seed: Optional[int] = None, training: bool = True) -> torch.Tensor:
    """
    z' = z + β · magnitude(z) · d

    magnitude(z):
      drift_use_norm=False (default)  →  element-wise |z|,  drift_i = β|z_i|d_i
      drift_use_norm=True             →  scalar ‖z‖_F,      drift   = β‖z‖d

    drift_frozen=True  (default): d is frozen at first call.
      Jacobian carries a systematic rank-1 term every step → not compensable.

    drift_frozen=False (control):  d is re-sampled i.i.d. each call.
      E[drift term] = 0 over minibatches → compensable.
      Use this as the same-δ control to isolate the frozen-direction effect.

    drift_resample_when_eval: if True, use frozen d during training but resample d each call when not training (eval).
    """
    beta = config.drift_beta
    if beta == 0.: return out

    d_dim = out.shape[-1]
    sv    = seed if seed is not None else config.seed

    eval_resample = bool(getattr(config, "drift_resample_when_eval", False)) and not training
    effective_frozen = bool(config.drift_frozen) and not eval_resample

    if effective_frozen:
        # Cache by (dim, seed) so A2 can use a different fixed template at eval.
        sv_key = sv if sv is not None else 0
        key = (d_dim, sv_key)
        existed_before = key in config._drift_d_vectors
        d = _frozen_vec(config._drift_d_vectors, key, d_dim, out.device, out.dtype, sv, 0xBEEF)
    else:
        # Resampled d each call: use seed only when caller passes one; else fresh randomness.
        # This enables epoch-wise resampling (seed fixed within epoch) while keeping step-wise
        # resampling stochastic when seed=None.
        if seed is not None:
            gen = torch.Generator(device=out.device)
            gen.manual_seed((seed + 0xBEEF) % 2**31)
            d = torch.randn(d_dim, generator=gen, device=out.device, dtype=out.dtype)
        else:
            d = torch.randn(d_dim, device=out.device, dtype=out.dtype)
        # Mean-bias ablation (applied before normalization so |d| is controlled)
        if getattr(config, "drift_d_mean", 0.0) != 0.0:
            d = d + float(config.drift_d_mean)
        d = (d / (d.norm() + 1e-8)).detach()
    
    # Mean-bias for frozen case too (done after _frozen_vec so cache stays "same family")
    if effective_frozen and getattr(config, "drift_d_mean", 0.0) != 0.0:
        d = d + float(config.drift_d_mean)
        d = (d / (d.norm() + 1e-8)).detach()

        # Optional debug: print template stats once per (dim, seed)
        if os.environ.get("SYNTH_DEBUG_DRIFT_D", "").strip() in ("1", "true", "yes") and (not existed_before):
            try:
                print(
                    f"[DEBUG drift_d] dim={d_dim} seed={sv} drift_d_mean={config.drift_d_mean} "
                    f"mean={float(d.mean()):.6f} norm={float(d.norm()):.6f} std={float(d.std()):.6f}"
                )
            except Exception:
                pass
    elif effective_frozen:
        # Frozen but zero drift_d_mean: still allow debug once per (dim, seed)
        if os.environ.get("SYNTH_DEBUG_DRIFT_D", "").strip() in ("1", "true", "yes") and (not existed_before):
            try:
                print(
                    f"[DEBUG drift_d] dim={d_dim} seed={sv} drift_d_mean=0.0 "
                    f"mean={float(d.mean()):.6f} norm={float(d.norm()):.6f} std={float(d.std()):.6f}"
                )
            except Exception:
                pass

    if config.drift_use_norm:
        # scalar ‖z‖: Jacobian = I + β (d ⊗ z/‖z‖)
        z_norm = out.norm(dim=-1, keepdim=True)
        drift  = beta * z_norm * d.unsqueeze(0)
    else:
        # element-wise |z|: Jacobian = diag(1 + β sign(z) d)
        drift = beta * out.abs() * d.unsqueeze(0)

    # No detach: gradient flows through drift → Jacobian I + β diag(sign(z))⊙d (systematic bias in backward)
    return out + drift


def apply_sign_coupled_scaling(out: torch.Tensor, config: SynthNoiseConfig,
                               seed: Optional[int] = None) -> torch.Tensor:
    """
    z' = (1 + α · sign(v^T z)) · z,   v frozen unit vector.

    The "broken" version of coupled_inconsistent: tanh → sign.

    Jacobian ∂z'/∂z  =  diag(1 + α sign(v^T z))  +  α z ⊗ ∂[sign(v^T z)]/∂z
                           ↑ this part flows           ↑ this is 0 a.e. (autograd)

    What happens:
      - Activations are scaled by +(1+α) or +(1-α) depending on which half-space
        v^T z falls in.  Hard threshold, not smooth.
      - At α near 1: the (1-α) side → near-zero scaling → structured dead neurons.
        These dead neurons contribute zero gradient, and the pattern is data-dependent
        (determined by sign(v^T z) per sample), so it shifts every batch.
      - Autograd gives zero gradient for how z moves across the sign boundary,
        creating a systematic mismatch between forward curvature and backward signal.

    Compare to coupled_inconsistent at same α and same δ for a clean ablation:
    same forward variance, tanh vs sign, compensable vs not.
    """
    alpha = config.sign_scale_alpha
    if alpha == 0.: return out

    d   = out.shape[-1]
    sv  = seed if seed is not None else config.seed
    if getattr(config, "sign_scale_v_resample", False):
        if seed is not None:
            gen = torch.Generator(device=out.device)
            gen.manual_seed((seed + 0xCAFE) % 2**31)
            v = torch.randn(d, generator=gen, device=out.device, dtype=out.dtype)
        else:
            v = torch.randn(d, device=out.device, dtype=out.dtype)
        v = (v / (v.norm() + 1e-8)).detach()
    else:
        sv_key = sv if sv is not None else 0
        v_key = (d, sv_key)
        v = _frozen_vec(config._sign_scale_v_vectors, v_key, d, out.device, out.dtype, sv, 0xCAFE)

    vTz = (out * v.unsqueeze(0)).sum(-1, keepdim=True)          # [batch, 1]
    s_z = (1.0 + alpha * torch.sign(vTz))                      # no detach: backward flows through s_z (∂sign/∂z = 0 a.e.)
    return s_z * out


def apply_rank_collapse(out: torch.Tensor, config: SynthNoiseConfig,
                        seed: Optional[int] = None, training: bool = True) -> torch.Tensor:
    """
    z' = P_k · z + ε_fill,   P_k rank-k orthogonal projection.

    - rank_resample=False (default): P_k is fixed at first use (frozen projector).
      Gradient in (d-k)-dim null space of P_k is permanently zero → information barrier.

    - rank_resample=True: P_k is resampled every forward call (per step/batch).
      Each step still uses a rank-k projection, but the subspace changes; gradient
      can flow in different directions over time → control for "fixed direction" effect.

    rank_resample_when_eval: if True, use fixed P_k during training but resample P_k each call when not training.

    Jacobian ∂z'/∂z = P_k  (rank k < d, symmetric).

    ε_fill ~ N(0, σ²) is detached fill noise to match δ target (no gradient).

    Calibration:
      Run with rank_fill_sigma=0 to measure δ_proj = ‖(I-P_k)z‖/‖z‖.
      Then set rank_fill_sigma to top up to desired δ.
    """
    d = out.shape[-1]
    k = min(config.rank_k, d)

    effective_resample = bool(config.rank_resample)
    if bool(getattr(config, "rank_resample_when_eval", False)) and not training:
        effective_resample = True

    if effective_resample:
        # Resampled projector: new P_k every call.
        # When seed is None: do NOT fall back to config.seed — that would yield the same P_k
        # every call and make rank_resample identical to rank_resample=False. Use fresh
        # randomness so each forward gets a different subspace. Only use a fixed seed when
        # the caller explicitly passes one (e.g. for reproducible eval, or epoch-wise same P).
        if seed is not None:
            gen = torch.Generator(device=out.device)
            gen.manual_seed((seed + 0xF00D) % 2**31)
            Q = torch.randn(d, k, generator=gen, device=out.device, dtype=out.dtype)
        else:
            Q = torch.randn(d, k, device=out.device, dtype=out.dtype)
        Q, _ = torch.linalg.qr(Q)
        P_k = (Q @ Q.t()).detach()
    else:
        # Fixed projector: cache P_k per (d, k, seed) so train vs eval can use different P (A2).
        sv = seed if seed is not None else config.seed
        sv = sv if sv is not None else 0
        key = (d, k, sv)
        if key not in config._rank_proj:
            gen = torch.Generator(device=out.device)
            gen.manual_seed((sv + 0xF00D) % 2**31)
            Q = torch.randn(d, k, generator=gen, device=out.device, dtype=out.dtype)
            Q, _ = torch.linalg.qr(Q)
            config._rank_proj[key] = (Q @ Q.t()).detach()
        P_k = config._rank_proj[key].to(device=out.device, dtype=out.dtype)

    z_proj = out @ P_k.t()

    if config.rank_fill_sigma > 0.:
        z_proj = z_proj + (torch.randn_like(out) * config.rank_fill_sigma).detach()

    return z_proj


def apply_deterministic_clip(out: torch.Tensor, config: SynthNoiseConfig) -> torch.Tensor:
    """
    z' = clip(z, [-c, c])                             (clip_dither=False)
    z' = clip(z + u, [-c, c]) - u,  u~U(-c/2, c/2)   (clip_dither=True)

    Hard clip (dither=False) — NOT compensable:
      ∂z'/∂z = 1 if |z| < c,  0 if |z| ≥ c.
      Activations in the dead zone (|z| ≥ c) carry no gradient.
      As weights grow, more neurons saturate, progressively killing gradient flow.
      Unlike ReLU (one-sided), clip creates a symmetric dead zone on both tails.

    Dithered clip (dither=True) — compensable control:
      Adding uniform noise before clipping and subtracting after converts the
      hard step into a stochastic one.  E[z'] ≈ z for |z| < c + Δ/2,
      and the expected gradient is non-zero over a wider range.
      Same δ as hard clip, but gradient signal survives.

    This pair directly instantiates the dithering = compensability argument.
    """
    c = config.clip_c
    if c <= 0.: return out

    if config.clip_dither:
        u = (torch.rand_like(out) - 0.5) * c       # U(-c/2, c/2)
        return torch.clamp(out + u, -c, c) - u
    else:
        return torch.clamp(out, -c, c)


# =============================================================================
# Dispatch
# =============================================================================

def synth_noise_linear_forward(
    x: torch.Tensor,
    W: torch.Tensor,
    config: SynthNoiseConfig,
    training: bool = True,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Forward pass for a linear layer with synthetic noise."""

    # weight-domain
    if config.noise_type == 'iid_multiplicative':
        W_eff = apply_iid_multiplicative_distortion(W, config, seed=seed)
    elif config.noise_type == 'heavy_tail':
        W_eff = apply_heavy_tail_forward_noise(W, config, seed=seed)
    else:
        W_eff = W

    if config.noise_type in ('iid_multiplicative', 'heavy_tail') and config.compensation_in_backward:
        W_eff = W + (W_eff - W).detach()

    out = F.linear(x, W_eff)

    # output-domain
    nt = config.noise_type
    at_logits = getattr(config, "backward_corruption_at", None) == "logits"
    if   nt == 'heavy_tail_output':          out = apply_heavy_tail_output(out, config, seed=seed)
    elif nt == 'decoupled_consistent':       out = apply_decoupled_consistent(out, config, seed=seed)
    elif nt == 'decoupled_inconsistent':     out = apply_decoupled_inconsistent(out, config, seed=seed)
    elif nt == 'input_dependent':            out = apply_input_dependent_scale_bias(out, config, seed=seed)
    elif nt == 'coupled_consistent':         out = apply_input_dependent_scale_bias(out, config, seed=seed, compensation_override=True,  alpha_override=config.coupled_consistent_alpha)
    elif nt == 'coupled_inconsistent':       out = apply_input_dependent_scale_bias(out, config, seed=seed, compensation_override=False, alpha_override=config.coupled_inconsistent_alpha)
    elif nt == 'gradient_degenerate':        out = apply_gradient_degenerate_perturbations(out, config, training=training)
    elif nt == 'adversarial_direction_bias' and not at_logits: out = apply_adversarial_direction_bias(out, config, seed=seed)
    elif nt == 'sign_gradient_corruption' and not at_logits:   out = apply_sign_gradient_corruption(out, config, seed=seed)
    elif nt == 'saturation_collapse':        out = apply_saturation_collapse(out, config)
    elif nt == 'frozen_additive_drift':      out = apply_frozen_additive_drift(out, config, seed=seed, training=training)
    elif nt == 'sign_coupled_scaling':       out = apply_sign_coupled_scaling(out, config, seed=seed)
    elif nt == 'rank_collapse':              out = apply_rank_collapse(out, config, seed=seed, training=training)
    elif nt == 'deterministic_clip':         out = apply_deterministic_clip(out, config)

    return out


# =============================================================================
# Layer wrappers
# =============================================================================

class SynthNoiseLinear(nn.Module):
    """nn.Linear with synthetic noise injection."""

    def __init__(self, linear: nn.Linear, config: SynthNoiseConfig, enable_noise: bool = True):
        super().__init__()
        self.in_features  = linear.in_features
        self.out_features = linear.out_features
        self.config       = config
        self.enable_noise = enable_noise
        self.weight       = nn.Parameter(linear.weight.data.clone())
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.data.clone())
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor, seed: Optional[int] = None) -> torch.Tensor:
        eff_seed = seed if seed is not None else getattr(self.config, "_runtime_seed", None)
        out = (synth_noise_linear_forward(x, self.weight, self.config,
                                          training=self.training, seed=eff_seed)
               if self.enable_noise else F.linear(x, self.weight))
        if self.bias is not None:
            out = out + self.bias
        return out


class SynthNoiseConv2d(nn.Module):
    """nn.Conv2d with synthetic noise injection (im2col path)."""

    def __init__(self, conv: nn.Conv2d, config: SynthNoiseConfig, enable_noise: bool = True):
        super().__init__()
        self.config       = config
        self.enable_noise = enable_noise
        self.in_channels  = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size  = conv.kernel_size if isinstance(conv.kernel_size, tuple) else (conv.kernel_size,) * 2
        self.stride       = conv.stride      if isinstance(conv.stride,      tuple) else (conv.stride,)      * 2
        self.padding      = conv.padding     if isinstance(conv.padding,     tuple) else (conv.padding,)     * 2
        self.has_bias     = conv.bias is not None
        self.weight       = nn.Parameter(conv.weight.data.clone())
        if self.has_bias:
            self.bias = nn.Parameter(conv.bias.data.clone())
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor, seed: Optional[int] = None) -> torch.Tensor:
        eff_seed = seed if seed is not None else getattr(self.config, "_runtime_seed", None)
        in_h, in_w = x.size(2), x.size(3)
        k_h,  k_w  = self.kernel_size
        s_h,  s_w  = self.stride
        p_h,  p_w  = self.padding
        x_unfold   = F.unfold(x, kernel_size=self.kernel_size, dilation=1,
                               padding=self.padding, stride=self.stride)
        W_flat     = self.weight.view(self.out_channels, -1)
        x_flat     = x_unfold.transpose(1, 2)
        out_flat   = (synth_noise_linear_forward(x_flat, W_flat, self.config,
                                                 training=self.training, seed=eff_seed)
                      if self.enable_noise else F.linear(x_flat, W_flat))
        out_h = (in_h + 2*p_h - k_h) // s_h + 1
        out_w = (in_w + 2*p_w - k_w) // s_w + 1
        out   = F.fold(out_flat.transpose(1, 2), output_size=(out_h, out_w), kernel_size=1)
        if self.has_bias:
            out = out + self.bias.view(1, -1, 1, 1)
        return out
