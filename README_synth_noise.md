# synth_noise_wrappers

Synthetic noise injection for neural network layers, without weight-to-conductance mapping.

Designed for studying **compensability under gradient-based training**: given two perturbations with the same forward distortion magnitude δ, does the gradient structure allow the model to learn through the noise?

---

## Perturbation magnitude calibration

**Forward perturbations** use the same normalized distortion on logits:

$$\delta_{\text{logit}} = \sqrt{\frac{\mathbb{E}[\|z_{\text{noisy}} - z_{\text{clean}}\|^2]}{\mathbb{E}[\|z_{\text{clean}}\|^2]}}$$

**Backward-only perturbations** (e.g. `adversarial_direction_bias`, `sign_gradient_corruption`) leave the forward unchanged, so δ_logit = 0. To compare them with forward noise on a common scale, the calibration script defines a **gradient-level** strength with the same formula, on the gradient of the loss w.r.t. all parameters:

$$\delta_{\text{grad}} = \sqrt{\frac{\mathbb{E}[\|g_{\text{noisy}} - g_{\text{clean}}\|^2]}{\mathbb{E}[\|g_{\text{clean}}\|^2]}}$$

Here \(g\) is the flattened gradient from one backward pass. Calibrate with `--noise_type adv_direction_beta` or `--noise_type sign_corrupt_p` and the same `--target_delta` (e.g. 1.0) as for forward types; the script reports θ* and **δ_grad** in the output JSON. You can then compare “δ_logit = 1” (forward) with “δ_grad = 1” (backward) in experiments.

**sign_corrupt_p:** If `noise_injection` only enables sign corruption on a subset of layers, δ_grad is smaller than 2√p (dilution). Use a smaller `--target_delta` (e.g. 0.3) or enable noise on all layers; the script allows p up to 1.0.

Calibrate each (forward) noise type to a target δ by adjusting its magnitude parameter before running comparative experiments. For `rank_collapse`, set `rank_fill_sigma=0` first to measure δ from the projection alone, then increase to reach the target.

**Why ViT’s calibratable δ range is often lower than ResNet (e.g. drift_beta: ResNet can reach δ=10, ViT tops out around δ≈1):**

- **δ_logit** is a *relative* logit deviation: √E[‖z'−z‖²/‖z‖²]. So δ ≈ (typical absolute change) / (typical ‖z_clean‖).
- **ViT** usually has **larger ‖z_clean‖** at logits (no final norm, many layers, larger embed dim). For the same perturbation strength (e.g. same `drift_beta`), ‖z_noisy−z_clean‖ is similar in scale, so δ is **smaller** for ViT. To reach the same δ you need a larger magnitude (e.g. larger β).
- Pushing the magnitude too high in ViT makes the forward **unstable** (drift/scale accumulates over many layers, then NaNs or saturation). So the **maximum achievable \(\delta\)** before instability is lower for ViT (often around 1), while ResNet can be calibrated up to \(\delta=10\) with the same noise types.
- The calibration script uses a larger default \(\theta\) range for `vit_tiny` on several params (e.g. `drift_beta`, `sign_scale_alpha`, `rank_fill_sigma`, `coupled_*`, `clip_c`) so that higher target \(\delta\) can be attempted; if you still hit “\(\delta < \text{target}\)” at the max \(\theta\), that is the model’s \(\delta\) ceiling—use a lower `--target_delta` for ViT when comparing across architectures.

---

## Perturbation taxonomy

### Group 1 — Weight-domain noise

Noise is applied to W before the linear operation. Gradient flows through the noisy weight via STE when `compensation_in_backward=True`.

| `noise_type` | Forward | Compensable |
|---|---|---|
| `iid_multiplicative` | W' = W ⊙ (1+ε), ε~N(0,σ²) | ✓ |
| `heavy_tail` | W' = W ⊙ (1+αε), ε~t_ν | ✓ (ν>1); ✗ (ν≤1, variance diverges) |

**Key parameters:** `variability_sigma`, `heavy_tail_alpha`, `heavy_tail_nu`

---

### Group 2 — Output-domain noise, compensable

Noise is applied to z = Wx. These perturbations have zero-mean gradient bias or use detached scaling (STE), so the optimizer can correct for them given sufficient training.

| `noise_type` | Forward | Backward Jacobian | Why compensable |
|---|---|---|---|
| `heavy_tail_output` | z' = z⊙(1+αε) | detached scale | STE: gradient sees clean z |
| `decoupled_consistent` | z' = (1+ε)z, ε~N(0,σ²) | detach(1+ε) | Gradient scaled by ~1 in expectation |
| `decoupled_inconsistent` | z' = z + ε tanh(z) | 1 + ε sech²(z) | Extra term is zero-mean → averages out |
| `input_dependent` | z' = s(z)⊙z, s(z)=1+α tanh(v^Tz) | STE (detached s) | Gradient sees clean z direction |
| `coupled_consistent` | same as input_dependent | detach(s(z)) | STE |
| `coupled_inconsistent` | same as input_dependent | s(z) (no detach) | s(z) bounded in [1-α, 1+α], bias is small |

**Key parameters:** `decoupled_consistent_sigma`, `decoupled_inconsistent_sigma`, `input_dependent_alpha`, `coupled_consistent_alpha`, `coupled_inconsistent_alpha`

---

### Group 3 — Gradient-structure broken (backward-only, forward = identity)

Forward pass is identical to clean. Gradient is corrupted in the custom backward. **δ_logit = 0** (no forward change). To match strength with forward perturbations, use **δ_grad** (see above): run `calibrate_noise_strength.py` with `--noise_type adv_direction_beta` or `--noise_type sign_corrupt_p` and the same `--target_delta`; the script finds θ* so that δ_grad = target, and writes `delta_grad` in the output JSON.

| `noise_type` | Backward | Why not compensable |
|---|---|---|
| `adversarial_direction_bias` | g_eff = g + β‖g‖d, d frozen | Drift ∝ ‖g‖; doesn't shrink relative to true gradient |
| `sign_gradient_corruption` | flip sign of g[i] w.p. p | E[g_eff] = (1−2p)g; p=0.5 → random walk |

**Key parameters:** `adv_direction_beta`, `adv_direction_frozen`, `adv_direction_random_sign`, `sign_corrupt_p`, `sign_corrupt_mode`, `sign_corrupt_noise_sigma`

**Ablation (Group 3):**
- **adversarial_direction_bias:** `adv_direction_frozen=False` → resample d each call (E[d]=0) → compensable control; `adv_direction_random_sign=True` → drift × ±1 each call (zero-mean) → compensable control.
- **sign_gradient_corruption:** `sign_corrupt_mode='noise_positive_align'` → g′ = g + σ·N(0,I), E[g′]=g so E⟨g′,g⟩>0; tune `sign_corrupt_noise_sigma` so cos_sim is similarly low as sign flip at p=0.5.

**Why ViT can look fine while ResNet collapses (feature, not bug):**  
Per-layer backward corruption is applied on **each layer’s output**. In ViT, each block is `x + Attn(LN(x))` and `x + MLP(LN(x))` — the **identity branch** never passes through the noisy backward, so a clean gradient path always exists. In ResNet, the main path and (where present) the 1×1 shortcut conv are both wrapped, so the residual path can be corrupted too. So “backward-only” here attacks **one branch**; ViT has an untouched branch, ResNet does not. To get a **single-point, unbypassable** probe, set **`backward_corruption_at: 'logits'`**: corruption is applied only at the logits (all paths have merged), and per-layer sign/adv is skipped.

**Diagnostic:** Set env **`SYNTH_NOISE_DIAGNOSTIC=1`** and run training. Each epoch log will include `sign_corrupt_calls`, `sign_corrupt_flip_ratio` (should be ≈ p), and `adv_bias_calls`. Use this to confirm that corruption is actually applied (e.g. on ViT, call count and flip ratio should be non-trivial).

---

### Group 4 — Forward structure broken, gradient killed by honest autograd

The forward perturbation itself causes the honest Jacobian ∂z'/∂z to be near-zero, zero in structured subspaces, or systematically misaligned. No backward hacking needed.

#### `saturation_collapse`
```
z' = α · tanh(γ · z)
```
Large γ compresses the linear region to width ~2/γ. For typical activations |z|~O(1), sech²(γz) ≈ 4e^{-2γ|z|} → gradient death within a few layers. The honest backward passes through this near-zero Jacobian.

**Key parameters:** `saturation_gamma` (3=mild, 5=strong, 10=extreme), `saturation_alpha`

#### `gradient_degenerate`
```
z' = ADC_quantize(z, bits)
```
Min-max quantization to `bits` levels. Without STE (`compensation_in_backward=False`), gradients are blocked entirely. With STE (`=True`), this becomes compensable — use as a control.

**Key parameters:** `adc_bits`, `enable_adc`, `compensation_in_backward`, `adc_backward_mode`

**Ablation (Group 4):** `adc_backward_mode`: `'ste'` (grad flows as identity) vs `'stopgrad'` (grad blocked). When unset, uses `compensation_in_backward` (True→ste, False→stopgrad). Same forward/δ, different backward → isolate Jacobian-structure effect.

---

### Group 5 — Forward structure broken, new additions

All four perturbations in this group have **measurable δ > 0** and are designed for same-δ comparison against the compensable perturbations in Group 2. Each comes with a built-in control experiment.

#### `frozen_additive_drift`
```
z' = z + β · |z| · d          (drift_use_norm=False, default)
z' = z + β · ‖z‖ · d         (drift_use_norm=True)
```
`d` is a frozen unit vector. The Jacobian carries a systematic rank-1 term proportional to `d` every step. Because `d` is fixed, this term never averages to zero over minibatches, causing consistent weight drift away from the loss gradient direction.

**Control experiment:** set `drift_frozen=False`. `d` is re-sampled i.i.d. each call → zero-mean drift → compensable. Identical δ, different compensability. This is the cleanest same-δ isolation experiment in this file.

**Key parameters:** `drift_beta`, `drift_use_norm`, `drift_frozen`

| `drift_frozen` | Compensable | Note |
|---|---|---|
| `True` | ✗ | Systematic direction drift |
| `False` | ✓ | Zero-mean, control |

---

#### `sign_coupled_scaling`
```
z' = (1 + α · sign(v^T z)) · z,   v frozen
```
The "broken" variant of `coupled_inconsistent`. Replacing `tanh` with `sign` makes the scale function a step function over the half-space defined by `v`.

The Jacobian is `diag(1 + α sign(v^T z))` (which autograd computes), plus a term from `∂[sign(v^T z)]/∂z` which is **zero almost everywhere**. This missing term creates a systematic inconsistency between forward curvature and the backward signal.

At `α` near 1, the lower half-space gets near-zero scaling → structured dead neurons whose pattern shifts per batch (input-dependent) but always follows `v`.

**Compare to:** `coupled_inconsistent` at same `α` and δ — tanh vs sign, compensable vs not.

**Key parameters:** `sign_scale_alpha` (range [0.3, 0.9])

---

#### `rank_collapse`
```
z' = P_k · z + ε_fill,   P_k = frozen rank-k projection
```
`P_k` projects `z` onto a random `k`-dimensional subspace (orthonormal basis via QR). The Jacobian is `P_k` (rank `k < d`).

Gradient in the `(d-k)`-dimensional null space of `P_k` is permanently zero — not attenuated, not noisy, but structurally absent. The optimizer is restricted to a `k`-dimensional subspace of the full parameter space regardless of training duration.

`ε_fill` (detached) is optional isotropic noise that contributes to δ without restoring gradient signal — use to match δ with other noise types.

**Key parameters:** `rank_k`, `rank_fill_sigma`

| `rank_k` | Fraction of gradient preserved |
|---|---|
| `d` | 1.0 (identity, no collapse) |
| `d//2` | 0.5 |
| `1` | 1/d (extreme collapse) |

---

#### `deterministic_clip`
```
z' = clip(z, [-c, c])                             (clip_dither=False)
z' = clip(z + u, [-c, c]) - u,  u~U(-c/2, c/2)   (clip_dither=True)
```
Hard clip creates a symmetric dead zone: `∂z'/∂z = 0` for `|z| ≥ c`. Unlike ReLU (one-sided), both tails are killed. As training proceeds and activations grow, more neurons fall into the dead zone — a progressive gradient death spiral.

**Control experiment:** `clip_dither=True` adds uniform noise before clipping and subtracts it after (equivalent to dithered quantization). The expected gradient is non-zero over a wider activation range. Same δ, but compensable.

This pair directly instantiates the classical result that dithering converts a hard quantizer into a compensable one.

**Key parameters:** `clip_c`, `clip_dither`

| `clip_dither` | Compensable | Mechanism |
|---|---|---|
| `False` | ✗ | Hard dead zone |
| `True` | ✓ | Stochastic rounding |

**Architecture note (ViT vs ResNet):** The same `clip_c` has very different impact across architectures. ViT has more linear layers (QKV, proj, MLP × depth) and LayerNorm-normalized inputs, so activations are clipped in many more places and the effect compounds. For the same target δ, ViT typically needs a **larger** `clip_c` than ResNet (e.g. ResNet δ=1 → c≈0.85; ViT may need c in the tens). The calibration script uses a wider search range for `vit_tiny` (`theta_max=80`) so that `clip_c` can be found. Using a ResNet-calibrated or too-small `clip_c` on ViT will over-clip and collapse accuracy; always calibrate per architecture.

---

## Compensability summary

| `noise_type` | δ > 0 | Compensable | Broken condition | Control pair |
|---|---|---|---|---|
| `iid_multiplicative` | ✓ | ✓ | — | — |
| `heavy_tail` | ✓ | ✓ / ✗ | B: SNR (ν≤1) | — |
| `heavy_tail_output` | ✓ | ✓ | — | — |
| `decoupled_consistent` | ✓ | ✓ | — | — |
| `decoupled_inconsistent` | ✓ | ✓ (slow) | — | — |
| `input_dependent` | ✓ | ✓ | — | — |
| `coupled_consistent` | ✓ | ✓ | — | — |
| `coupled_inconsistent` | ✓ | ✓ | — | — |
| `gradient_degenerate` (no STE) | ✓ | ✗ | A: gradient blocked | `compensation_in_backward=True` |
| `adversarial_direction_bias` | — | ✗ | C: systematic drift | — |
| `sign_gradient_corruption` p≥0.5 | — | ✗ | A: E[g]=0 or negative | p=0 |
| `saturation_collapse` | ✓ | ✗ | B: vanishing Jacobian | γ→0 |
| `frozen_additive_drift` (frozen) | ✓ | ✗ | C: frozen rank-1 drift | `drift_frozen=False` |
| `sign_coupled_scaling` | ✓ | ✗ | A: sign Jacobian dead zone | `coupled_inconsistent` |
| `rank_collapse` | ✓ | ✗ | A: null-space zeroing | k=d (identity) |
| `deterministic_clip` | ✓ | ✗ | A/B: hard dead zone | `clip_dither=True` |

Conditions A/B/C from the compensability framework:
- **A** — gradient direction alignment: E[⟨g_eff, g_true⟩] > 0
- **B** — gradient SNR: Var[g_eff] bounded, does not dominate signal
- **C** — no systematic bias growth: drift term does not scale with ‖g‖ or training time

---

## Usage

```python
import torch.nn as nn
from synth_noise_wrappers import SynthNoiseConfig, SynthNoiseLinear

# Wrap an existing linear layer
linear = nn.Linear(256, 128)

# Example: rank_collapse with fill noise calibrated to δ ≈ 0.3
config = SynthNoiseConfig(
    noise_type='rank_collapse',
    rank_k=32,
    rank_fill_sigma=0.05,
    seed=42,
)
noisy_layer = SynthNoiseLinear(linear, config)

# Example: frozen_additive_drift vs its compensable control (same δ)
config_broken  = SynthNoiseConfig(noise_type='frozen_additive_drift', drift_beta=0.3, drift_frozen=True)
config_control = SynthNoiseConfig(noise_type='frozen_additive_drift', drift_beta=0.3, drift_frozen=False)

# Example: deterministic_clip vs dithered control
config_hard    = SynthNoiseConfig(noise_type='deterministic_clip', clip_c=1.0, clip_dither=False)
config_dithered = SynthNoiseConfig(noise_type='deterministic_clip', clip_c=1.0, clip_dither=True)
```

Frozen vectors (`d`, `v`, `P_k`) are initialized on the first forward call and cached in the config object. If you need different random directions per layer, use a separate `SynthNoiseConfig` instance per layer.

---

## Experiment design notes

**Same-δ comparison (main experiment):**
Fix δ using the calibration formula. Compare loss curves across noise types at the same δ. Compensable perturbations should converge; non-compensable ones should stall or diverge.

**Isolation experiments (control pairs):**

| Pair | What it isolates |
|---|---|
| `frozen_additive_drift` frozen=True vs False | Frozen direction vs zero-mean noise (pure condition C) |
| `deterministic_clip` dither=False vs True | Hard dead zone vs stochastic rounding (pure condition A/B) |
| `sign_coupled_scaling` vs `coupled_inconsistent` | sign vs tanh Jacobian at same α (pure condition A) |
| `gradient_degenerate` STE vs stopgrad (`adc_backward_mode`) | Gradient blocking vs STE bypass (pure condition A) |
| `adversarial_direction_bias` frozen vs resampled d (`adv_direction_frozen`) | Systematic drift vs zero-mean direction (pure condition C) |
| `adversarial_direction_bias` with `adv_direction_random_sign=True` | Same magnitude, drift × ±1 → zero-mean (condition C) |
| `sign_gradient_corruption` flip vs `sign_corrupt_mode=noise_positive_align` | E[g]=0 / wrong sign vs E[g]=g, low cos_sim (condition A) |

**Varying δ within a type:**
For compensable types, accuracy should recover as δ increases (more noise, slower but eventual convergence). For non-compensable types, there is a δ threshold above which training fails completely — this threshold is a structural property of the perturbation, not just a magnitude effect.
