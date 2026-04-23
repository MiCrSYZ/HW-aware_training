"""
Mirror-case supplemental figure for `rank_collapse` family.

Plots (per model) a before/after comparison:
  - Original: `rank_collapse` (template_mix_prob is NaN)
  - Mixed-template: `rank_collapse_mixed` (template_mix_prob == 0.75)

Within each group (Original / Mixed), show:
  - same template:   frozen_same
  - unseen template: frozen_train_new_test
  - resampled family (optional): A3_resampled_same_family_step

Only GRU and ViT are plotted.
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

# Families
PERT_ORIG = "rank_collapse"
PERT_MIXED = "rank_collapse_mixed"
MIX_PROB = 0.75

# Conditions (A1/A2/A3)
COND_SAME = "frozen_same"
COND_UNSEEN = "frozen_train_new_test"
COND_A3 = "A3_resampled_same_family_step"

MODEL_ORDER = ("gru", "vit")

# Colors: muted, print-friendly
EDGE = "#2c2c2c"
COL_SAME = "#8eb4d2"
COL_UNSEEN = "#d4a574"
COL_A3 = "#95b89a"


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}, got {type(data)}")
    return data


def _is_nan_template_mix_prob(v: Any) -> bool:
    # The files use string "NaN" for missing values sometimes.
    if v is None:
        return True
    if v == "NaN":
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    return False


def _tm_ok_pure(rec: Dict[str, Any]) -> bool:
    return _is_nan_template_mix_prob(rec.get("template_mix_prob"))


def _tm_ok_mixed(rec: Dict[str, Any]) -> bool:
    v = rec.get("template_mix_prob")
    if isinstance(v, (int, float)):
        return math.isclose(float(v), MIX_PROB, rel_tol=0.0, abs_tol=1e-6)
    return False


def _strength_ok(val: Any, target: float) -> bool:
    try:
        return math.isclose(float(val), target, rel_tol=0.0, abs_tol=1e-9)
    except (TypeError, ValueError):
        return False


def _pick_strength_and_values(
    rows: List[Dict[str, Any]],
    model: str,
) -> Tuple[float, Dict[str, float], Dict[str, float], Optional[Tuple[float, float]]]:
    """
    Returns:
      strength_val,
      orig_values {same, unseen, maybe a3},
      mixed_values {same, unseen, maybe a3},
      a3_pair (a3_orig, a3_mixed) or None.
    """
    # Candidate strengths from each family.
    strengths_orig = set()
    strengths_mixed = set()
    for r in rows:
        if r.get("model") != model:
            continue
        if r.get("perturbation_name") == PERT_ORIG and _tm_ok_pure(r):
            strengths_orig.add(float(r["strength_param_val"]))
        if r.get("perturbation_name") == PERT_MIXED and _tm_ok_mixed(r):
            strengths_mixed.add(float(r["strength_param_val"]))

    common = strengths_orig & strengths_mixed
    if not common:
        raise RuntimeError(f"No common strength found for {model} between {PERT_ORIG} and {PERT_MIXED}.")

    def collect_for_strength(perturbation: str, tm_ok_fn, strength: float) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for r in rows:
            if r.get("model") != model:
                continue
            if r.get("perturbation_name") != perturbation:
                continue
            if not _strength_ok(r.get("strength_param_val"), strength):
                continue
            if not tm_ok_fn(r):
                continue
            cond = r.get("condition")
            if cond == COND_SAME:
                out["same"] = float(r["test_acc"])
            elif cond == COND_UNSEEN:
                out["unseen"] = float(r["test_acc"])
            elif cond == COND_A3:
                out["a3"] = float(r["test_acc"])
        return out

    best_strength: Optional[float] = None
    best_shrink: float = -1e18
    best_orig: Dict[str, float] = {}
    best_mixed: Dict[str, float] = {}
    best_a3: Optional[Tuple[float, float]] = None

    for s in sorted(common):
        orig = collect_for_strength(PERT_ORIG, _tm_ok_pure, s)
        mixed = collect_for_strength(PERT_MIXED, _tm_ok_mixed, s)
        if "same" not in orig or "unseen" not in orig:
            continue
        if "same" not in mixed or "unseen" not in mixed:
            continue
        gap_orig = orig["same"] - orig["unseen"]
        gap_mixed = mixed["same"] - mixed["unseen"]
        shrink = gap_orig - gap_mixed

        # Prefer stronger evidence: largest gap shrink.
        if shrink > best_shrink:
            best_shrink = shrink
            best_strength = s
            best_orig = orig
            best_mixed = mixed
            if "a3" in orig and "a3" in mixed:
                best_a3 = (orig["a3"], mixed["a3"])
            else:
                best_a3 = None

    if best_strength is None:
        raise RuntimeError(f"Could not select a valid strength for {model}.")

    return best_strength, best_orig, best_mixed, best_a3


def _annotate_bar(ax: plt.Axes, bar, dy: float, fontsize: float) -> None:
    h = bar.get_height()
    x = bar.get_x() + bar.get_width() / 2
    ax.text(
        x,
        h + dy,
        f"{h:.1f}",
        ha="center",
        va="bottom",
        fontsize=fontsize,
        color="black",
    )


def _plot_model(
    ax: plt.Axes,
    rows: List[Dict[str, Any]],
    model: str,
) -> None:
    strength, orig, mixed, a3_pair = _pick_strength_and_values(rows, model)

    y_same_o, y_unseen_o = orig["same"], orig["unseen"]
    y_same_m, y_unseen_m = mixed["same"], mixed["unseen"]

    show_a3 = a3_pair is not None
    y_a3_o, y_a3_m = a3_pair if a3_pair is not None else (0.0, 0.0)

    # Layout mirrors the original intervention plots.
    x_centers = np.array([0.0, 1.45])
    group_distance = x_centers[1] - x_centers[0]
    bar_w = 0.28
    # Offsets within group.
    if show_a3:
        offsets = np.array([-0.28, 0.0, 0.28])  # same, a3, unseen
    else:
        offsets = np.array([-0.28, 0.28])  # same, unseen

    # Colors and labels per drawn bar.
    ax.set_title("GRU" if model == "gru" else "ViT", fontsize=10, pad=10)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.45, color="#999999", zorder=0)
    ax.set_axisbelow(True)

    # Bars: x positions are derived from actual bar widths/offsets.
    bars_legend: List[Any] = []
    labels_legend: List[str] = []

    def add_group(center: float, ys: List[float], labels: List[str]) -> None:
        for i, y in enumerate(ys):
            x = center + offsets[i]
            c = COL_SAME if labels[i] == "same" else (COL_A3 if labels[i] == "a3" else COL_UNSEEN)
            label_str = "Same" if labels[i] == "same" else ("Resampled" if labels[i] == "a3" else "Unseen")
            b = ax.bar(
                [x],
                [y],
                bar_w,
                label=label_str,
                color=c,
                edgecolor=EDGE,
                linewidth=0.55,
                zorder=2,
            )[0]
            _annotate_bar(ax, b, dy=1.0, fontsize=6.5)
            # Keep only first handle for each label.
            if label_str not in labels_legend:
                bars_legend.append(b)
                labels_legend.append(label_str)

    # Original group (left tick).
    if show_a3:
        add_group(x_centers[0], [y_same_o, y_a3_o, y_unseen_o], ["same", "a3", "unseen"])
    else:
        add_group(x_centers[0], [y_same_o, y_unseen_o], ["same", "unseen"])

    # Mixed group (right tick).
    if show_a3:
        add_group(x_centers[1], [y_same_m, y_a3_m, y_unseen_m], ["same", "a3", "unseen"])
    else:
        add_group(x_centers[1], [y_same_m, y_unseen_m], ["same", "unseen"])

    ax.set_xticks(list(x_centers))
    ax.set_xticklabels(["Original", "Mixed"])

    x_min = x_centers.min() + offsets.min() - bar_w * 0.7
    x_max = x_centers.max() + offsets.max() + bar_w * 0.7
    ax.set_xlim(x_min, x_max)

    for spine in ax.spines.values():
        spine.set_linewidth(0.55)
        spine.set_color(EDGE)

    # Legend: inside axis frame, slightly lowered to avoid top clutter.
    if bars_legend:
        ax.legend(
            bars_legend,
            labels_legend,
            loc="lower right",
            bbox_to_anchor=(1.0, 0.001),
            frameon=True,
            fancybox=False,
            edgecolor=EDGE,
            facecolor="white",
            borderpad=0.35,
            handlelength=1.1,
            fontsize=8,
        )

    # Optional tiny note about selected strength (kept subtle; remove if you prefer absolutely no extra text).
    # ax.text(
    #     0.02,
    #     0.02,
    #     f"rank_k={strength:g}",
    #     transform=ax.transAxes,
    #     ha="left",
    #     va="bottom",
    #     fontsize=7,
    #     color=EDGE,
    # )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--summary-dir", type=Path, default=REPO_ROOT / "outputs" / "summary")
    p.add_argument(
        "--out-pdf",
        type=Path,
        default=REPO_ROOT / "outputs" / "figures" / "iccad_rank_collapse_mirror",
        help="Output prefix path without suffix; we will create _gru.pdf and _vit.pdf.",
    )
    p.add_argument("--out-eps", type=Path, default=None)
    args = p.parse_args()

    summary_dir = args.summary_dir if args.summary_dir.is_absolute() else (REPO_ROOT / args.summary_dir)
    out_prefix = args.out_pdf if args.out_pdf.is_absolute() else (REPO_ROOT / args.out_pdf)
    out_dir = out_prefix.parent
    prefix_stem = out_prefix.stem

    for model in MODEL_ORDER:
        json_name = f"{model}_resampled.json"
        rows = _load_rows(summary_dir / json_name)

        fig, ax = plt.subplots(1, 1, figsize=(2.9, 2.55), constrained_layout=True)
        _plot_model(ax, rows, model)
        fig.tight_layout(pad=0.08)

        pdf_path = out_dir / f"{prefix_stem}_{model}.pdf"
        eps_path = None
        if args.out_eps is not None:
            eps_prefix = args.out_eps if args.out_eps.is_absolute() else (REPO_ROOT / args.out_eps)
            eps_dir = eps_prefix.parent
            eps_stem = eps_prefix.stem
            eps_path = eps_dir / f"{eps_stem}_{model}.eps"

        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
        if eps_path is not None:
            fig.savefig(eps_path, format="eps", bbox_inches="tight")
        plt.close(fig)

        print("Wrote", pdf_path)
        if eps_path is not None:
            print("Wrote", eps_path)


if __name__ == "__main__":
    main()

