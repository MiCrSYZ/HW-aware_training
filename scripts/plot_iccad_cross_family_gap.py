"""
Cross-family summary figure:
same-template vs unseen-template gap across structured perturbation families.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

MODELS = ("gru", "vit", "resnet")
FAMILIES = (
    "frozen_additive_drift",
    "rank_collapse",
    "coupled_inconsistent",
    "sign_coupled_scaling",
)
FAMILY_LABELS = {
    "frozen_additive_drift": "frozen_additive_drift",
    "rank_collapse": "rank_collapse",
    "coupled_inconsistent": "coupled_inconsistent",
    "sign_coupled_scaling": "sign_coupled_scaling",
}

MODEL_LABELS = {"gru": "GRU", "vit": "ViT", "resnet": "ResNet"}

CONDITION_MAP = {
    "frozen_additive_drift": ("frozen_same", "frozen_train_new_test"),
    "rank_collapse": ("frozen_same", "frozen_train_new_test"),
    "coupled_inconsistent": ("A1_fixed_v_same_test", "A2_fixed_v_train_new_v_test"),
    "sign_coupled_scaling": ("A1_fixed_v_same_test", "A2_fixed_v_train_new_v_test"),
    "deterministic_clip": ("A1_same_c_train_and_eval", "A2_train_c_nearby_c_eval"),
}

# Prefer these strengths when available (from previously used/featured settings).
PREFERRED_STRENGTHS = {
    ("gru", "frozen_additive_drift"): 1.0,
    ("vit", "frozen_additive_drift"): 2.0,
    ("resnet", "frozen_additive_drift"): 0.481653254890755,
    ("gru", "rank_collapse"): 2.0,
    ("vit", "rank_collapse"): 98.0,
    ("resnet", "rank_collapse"): 16.0,
    ("gru", "coupled_inconsistent"): 2.0,
    ("vit", "coupled_inconsistent"): 1.2,
    ("resnet", "coupled_inconsistent"): 1.408722,
    ("gru", "sign_coupled_scaling"): 0.9,
    ("vit", "sign_coupled_scaling"): 0.8,
    ("resnet", "sign_coupled_scaling"): 1.0203125,
}

PREFERRED_MIX_PROB = 0.75


def _to_float(x: Any) -> Optional[float]:
    try:
        if x == "NaN" or x is None:
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def _is_tm_nan(x: Any) -> bool:
    if x == "NaN" or x is None:
        return True
    if isinstance(x, float) and math.isnan(x):
        return True
    return False


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"Expected list in {path}")
    return rows


def _collect_strength_candidates(
    rows: List[Dict[str, Any]],
    model: str,
    family: str,
) -> Dict[float, Dict[str, float]]:
    same_cond, unseen_cond = CONDITION_MAP[family]
    out: Dict[float, Dict[str, float]] = {}
    for r in rows:
        if r.get("model") != model:
            continue
        if r.get("perturbation_name") != family:
            continue
        if not _is_tm_nan(r.get("template_mix_prob")):
            continue
        s = _to_float(r.get("strength_param_val"))
        if s is None:
            continue
        cond = r.get("condition")
        if cond not in (same_cond, unseen_cond):
            continue
        d = out.setdefault(s, {})
        if cond == same_cond:
            d["same"] = float(r["test_acc"])
        elif cond == unseen_cond:
            d["unseen"] = float(r["test_acc"])
    return {k: v for k, v in out.items() if "same" in v and "unseen" in v}


def _choose_strength(cands: Dict[float, Dict[str, float]], preferred: Optional[float]) -> float:
    if not cands:
        raise RuntimeError("No valid strength candidates with both same/unseen.")
    if preferred is not None:
        for s in cands.keys():
            if math.isclose(s, preferred, rel_tol=0.0, abs_tol=1e-9):
                return s
    # Fallback: choose strength with highest same-template accuracy (representative operating point).
    return max(cands.keys(), key=lambda s: cands[s]["same"])


def _get_mixed_gap(
    rows: List[Dict[str, Any]],
    model: str,
    family: str,
    strength: float,
) -> Optional[float]:
    mixed_name = family + "_mixed"
    same_cond, unseen_cond = CONDITION_MAP[family]
    buckets: Dict[float, Dict[str, float]] = {}

    for r in rows:
        if r.get("model") != model:
            continue
        if r.get("perturbation_name") != mixed_name:
            continue
        s = _to_float(r.get("strength_param_val"))
        if s is None or not math.isclose(s, strength, rel_tol=0.0, abs_tol=1e-9):
            continue
        tm = _to_float(r.get("template_mix_prob"))
        if tm is None:
            continue
        cond = r.get("condition")
        if cond not in (same_cond, unseen_cond):
            continue
        d = buckets.setdefault(tm, {})
        if cond == same_cond:
            d["same"] = float(r["test_acc"])
        elif cond == unseen_cond:
            d["unseen"] = float(r["test_acc"])

    valid = [(tm, d) for tm, d in buckets.items() if "same" in d and "unseen" in d]
    if not valid:
        return None

    # Prefer p=0.75 if exists; otherwise use the largest available template_mix_prob.
    tm_selected, d_selected = min(valid, key=lambda t: (abs(t[0] - PREFERRED_MIX_PROB), -t[0]))
    _ = tm_selected
    return d_selected["same"] - d_selected["unseen"]


def _build_model_gaps(rows: List[Dict[str, Any]], model: str) -> Dict[str, Tuple[float, Optional[float], float]]:
    """
    Returns family -> (orig_gap, mixed_gap_or_none, selected_strength)
    """
    data: Dict[str, Tuple[float, Optional[float], float]] = {}
    for family in FAMILIES:
        cands = _collect_strength_candidates(rows, model, family)
        pref = PREFERRED_STRENGTHS.get((model, family))
        s = _choose_strength(cands, pref)
        orig_gap = cands[s]["same"] - cands[s]["unseen"]
        mixed_gap = _get_mixed_gap(rows, model, family, s)
        data[family] = (orig_gap, mixed_gap, s)
    return data


def _plot_panel(ax: plt.Axes, model: str, model_data: Dict[str, Tuple[float, Optional[float], float]]) -> None:
    x = np.arange(len(FAMILIES), dtype=float)
    w = 0.34

    def _clean_zero(v: Optional[float]) -> Optional[float]:
        if v is None:
            return None
        return 0.0 if abs(v) < 0.05 else float(v)

    orig_vals = np.array([_clean_zero(model_data[f][0]) for f in FAMILIES], dtype=float)
    mixed_vals = [model_data[f][1] for f in FAMILIES]
    mixed_vals = [_clean_zero(v) for v in mixed_vals]
    mixed_mask = np.array([v is not None for v in mixed_vals], dtype=bool)
    mixed_plot_vals = np.array([v if v is not None else np.nan for v in mixed_vals], dtype=float)

    orig_bars = ax.bar(
        x - w / 2,
        orig_vals,
        w,
        color="#6f7f8f",
        edgecolor="#2c2c2c",
        linewidth=0.55,
        label="Original gap",
        zorder=3,
    )
    mixed_bars = ax.bar(
        x[mixed_mask] + w / 2,
        mixed_plot_vals[mixed_mask],
        w,
        color="#bcc7d2",
        edgecolor="#2c2c2c",
        linewidth=0.55,
        label="Mixed gap",
        zorder=3,
    )

    # Annotate bar values for direct readability.
    for b in orig_bars:
        h = b.get_height()
        x0 = b.get_x() + b.get_width() / 2
        y0 = h + 0.8
        va = "bottom"
        ax.text(x0, y0, f"{h:.1f}", ha="center", va=va, fontsize=6.8, color="#1f1f1f")
    for b in mixed_bars:
        h = b.get_height()
        x0 = b.get_x() + b.get_width() / 2
        y0 = h + 0.8
        va = "bottom"
        ax.text(x0, y0, f"{h:.1f}", ha="center", va=va, fontsize=6.8, color="#1f1f1f")

    ax.axhline(0.0, color="#666666", linewidth=0.6, zorder=2)
    ax.set_title(MODEL_LABELS[model], fontsize=9.5, pad=6)
    ax.set_xticks(x)
    ax.set_xticklabels([FAMILY_LABELS[f] for f in FAMILIES], rotation=20, ha="right")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.45, color="#aaaaaa", zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.55)
        spine.set_color("#2c2c2c")

    handles, labels = ax.get_legend_handles_labels()
    uniq = {}
    for h, l in zip(handles, labels):
        if l not in uniq:
            uniq[l] = h
    ax.legend(
        list(uniq.values()),
        list(uniq.keys()),
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        frameon=True,
        fancybox=False,
        edgecolor="#2c2c2c",
        facecolor="white",
        borderpad=0.3,
        handlelength=1.3,
        fontsize=7.5,
    )


def plot(summary_dir: Path, out_pdf: Path, out_eps: Optional[Path]) -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "Nimbus Roman"],
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.5,
            "legend.fontsize": 8,
            "axes.linewidth": 0.55,
            "xtick.major.width": 0.45,
            "ytick.major.width": 0.45,
        }
    )

    rows_by_model = {
        m: _load_rows(summary_dir / f"{m}_resampled.json")
        for m in MODELS
    }
    data_by_model = {m: _build_model_gaps(rows_by_model[m], m) for m in MODELS}

    # compute y-limits globally for comparability
    vals = []
    for m in MODELS:
        for f in FAMILIES:
            o, mm, _s = data_by_model[m][f]
            vals.append(o)
            if mm is not None:
                vals.append(mm)
    y_min = min(vals)
    y_max = max(vals)
    pad = max(2.0, 0.08 * (y_max - y_min if y_max > y_min else 10.0))

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.05), sharey=True, constrained_layout=True)
    for ax, m in zip(axes, MODELS):
        _plot_panel(ax, m, data_by_model[m])
        ax.set_ylim(y_min - pad, y_max + pad)

    axes[0].set_ylabel("Gap = test_acc(A1) - test_acc(A2) (%)")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    if out_eps is not None:
        fig.savefig(out_eps, format="eps", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--summary-dir", type=Path, default=REPO_ROOT / "outputs" / "summary")
    p.add_argument(
        "--out-pdf",
        type=Path,
        default=REPO_ROOT / "outputs" / "figures" / "iccad_cross_family_gap.pdf",
    )
    p.add_argument("--out-eps", type=Path, default=None)
    args = p.parse_args()

    summary_dir = args.summary_dir if args.summary_dir.is_absolute() else (REPO_ROOT / args.summary_dir)
    out_pdf = args.out_pdf if args.out_pdf.is_absolute() else (REPO_ROOT / args.out_pdf)
    out_eps = args.out_eps if (args.out_eps and args.out_eps.is_absolute()) else ((REPO_ROOT / args.out_eps) if args.out_eps else None)

    plot(summary_dir.resolve(), out_pdf.resolve(), out_eps.resolve() if out_eps else None)
    print("Wrote", out_pdf)
    if out_eps:
        print("Wrote", out_eps)


if __name__ == "__main__":
    main()

