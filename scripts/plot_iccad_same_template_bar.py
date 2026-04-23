"""
ICCAD-style grouped bar chart: same-template vs unseen-template vs resampled family.
Reads outputs/summary/{gru,vit,resnet}_resampled.json.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

CONDITIONS = ("frozen_same", "frozen_train_new_test", "A3_resampled_same_family_step")
CONDITION_LABELS = (
    "A1: same template",
    "A2: unseen template",
    "A3: resampled family",
)

MODEL_ORDER = ("gru", "vit", "resnet")
DISPLAY_NAMES = {"gru": "GRU", "vit": "ViT", "resnet": "ResNet"}

# Representative drift_beta per model (from paper / config commentary).
STRENGTH_BY_MODEL = {
    "gru": 1.0,
    "vit": 2.0,
    "resnet": 0.481653254890755,
}


def _is_pure_frozen_row(rec: Dict[str, Any]) -> bool:
    if rec.get("perturbation_name") != "frozen_additive_drift":
        return False
    tm = rec.get("template_mix_prob")
    if tm == "NaN" or tm is None:
        return True
    if isinstance(tm, (int, float)) and not (isinstance(tm, float) and math.isnan(tm)):
        return False
    return True


def _strength_matches(model: str, val: Any) -> bool:
    target = STRENGTH_BY_MODEL[model]
    try:
        v = float(val)
    except (TypeError, ValueError):
        return False
    if model == "resnet":
        return math.isclose(v, target, rel_tol=0.0, abs_tol=1e-9)
    return math.isclose(v, target, rel_tol=0.0, abs_tol=1e-6)


def load_slice(summary_dir: Path) -> Dict[str, Dict[str, float]]:
    files = {
        "gru": summary_dir / "gru_resampled.json",
        "vit": summary_dir / "vit_resampled.json",
        "resnet": summary_dir / "resnet_resampled.json",
    }
    out: Dict[str, Dict[str, float]] = {}
    for model, path in files.items():
        with path.open("r", encoding="utf-8") as f:
            rows: List[Dict[str, Any]] = json.load(f)
        by_cond: Dict[str, float] = {}
        for rec in rows:
            if rec.get("model") != model:
                continue
            if not _is_pure_frozen_row(rec):
                continue
            if not _strength_matches(model, rec.get("strength_param_val")):
                continue
            cond = rec.get("condition")
            if cond not in CONDITIONS:
                continue
            acc = float(rec["test_acc"])
            if cond in by_cond:
                raise RuntimeError(
                    f"Duplicate row for {model=} {cond=} at target strength; "
                    "check template_mix / perturbation filters."
                )
            by_cond[cond] = acc
        missing = [c for c in CONDITIONS if c not in by_cond]
        if missing:
            raise RuntimeError(f"{path.name}: missing conditions {missing} for target strength")
        out[model] = by_cond
    return out


def plot_iccad_bar(
    data: Dict[str, Dict[str, float]],
    out_pdf: Path,
    out_eps: Path | None = None,
) -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "Nimbus Roman"],
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "lines.linewidth": 0.6,
        }
    )

    n_groups = len(MODEL_ORDER)
    n_bars = len(CONDITIONS)
    x = np.arange(n_groups, dtype=float)
    width = 0.22
    offsets = np.linspace(-(n_bars - 1) * width / 2, (n_bars - 1) * width / 2, n_bars)

    # Muted colors (low saturation, print-friendly).
    fills = ("#8eb4d2", "#d4a574", "#95b89a")
    edge = "#2c2c2c"

    fig, ax = plt.subplots(figsize=(3.4, 2.4), layout="constrained")

    for i, cond in enumerate(CONDITIONS):
        heights = np.array([data[m][cond] for m in MODEL_ORDER], dtype=float)
        pos = x + offsets[i]
        bars = ax.bar(
            pos,
            heights,
            width,
            label=CONDITION_LABELS[i],
            color=fills[i],
            edgecolor=edge,
            linewidth=0.6,
            zorder=2,
        )
        for b in bars:
            h = b.get_height()
            ax.text(
                b.get_x() + b.get_width() / 2,
                h + 1.2,
                f"{h:.1f}",
                ha="center",
                va="bottom",
                fontsize=7,
                color="black",
            )

    ax.set_ylabel("Test accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_NAMES[m] for m in MODEL_ORDER])
    ax.set_ylim(0, 100)
    ax.set_xlim(x.min() - 0.55, x.max() + 0.55)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, color="#888888", zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color(edge)

    ax.legend(
        frameon=True,
        fancybox=False,
        edgecolor=edge,
        facecolor="white",
        loc="upper right",
        bbox_to_anchor=(1.0, 0.88),
        borderpad=0.4,
        handlelength=1.2,
    )

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    if out_eps is not None:
        fig.savefig(out_eps, format="eps", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--summary-dir",
        type=Path,
        default=Path("outputs/summary"),
    )
    p.add_argument(
        "--out-pdf",
        type=Path,
        default=Path("outputs/figures/iccad_same_template_overestimate.pdf"),
    )
    p.add_argument(
        "--out-eps",
        type=Path,
        default=None,
        help="Optional EPS output (same vector use as PDF).",
    )
    args = p.parse_args()
    data = load_slice(args.summary_dir.resolve())
    plot_iccad_bar(data, args.out_pdf.resolve(), args.out_eps.resolve() if args.out_eps else None)
    print("Wrote", args.out_pdf)
    if args.out_eps:
        print("Wrote", args.out_eps)


if __name__ == "__main__":
    main()
