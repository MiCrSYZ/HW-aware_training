"""
Supporting figure: protocol-dependent diagnostics under frozen_additive_drift.

Data source:
  outputs/summary/resnet_resampled.json
  outputs/summary/vit_resampled.json
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

REPO_ROOT = Path(__file__).resolve().parents[1]

CONDITIONS = ("frozen_same", "frozen_train_new_test", "A3_resampled_same_family_step")
X_LABELS = ("same", "new", "resampled")

METRICS = (
    ("gradient_reachability_A", "Reachability A"),
    ("gradient_consistency_C", "Consistency C"),
    ("gradient_variance_domination_V", "Variance domination V"),
    ("perturbation_stability_S", "Stability S"),
)

MODEL_CONFIG = {
    "resnet": {
        "file": "resnet_resampled.json",
        "strength": 0.481653254890755,
        "label": "ResNet",
        "color": "#4c78a8",
        "marker": "o",
    },
    "vit": {
        "file": "vit_resampled.json",
        "strength": 2.0,
        "label": "ViT",
        "color": "#f58518",
        "marker": "s",
    },
}


def _is_pure_row(rec: Dict[str, Any]) -> bool:
    if rec.get("perturbation_name") != "frozen_additive_drift":
        return False
    tm = rec.get("template_mix_prob")
    if tm == "NaN" or tm is None:
        return True
    if isinstance(tm, float) and math.isnan(tm):
        return True
    return False


def _match_strength(v: Any, target: float, abs_tol: float) -> bool:
    try:
        return math.isclose(float(v), target, rel_tol=0.0, abs_tol=abs_tol)
    except (TypeError, ValueError):
        return False


def _to_float(v: Any) -> float:
    if v == "NaN" or v is None:
        return float("nan")
    return float(v)


def _load_model_data(summary_dir: Path, model: str) -> Dict[str, Dict[str, float]]:
    cfg = MODEL_CONFIG[model]
    fp = summary_dir / cfg["file"]
    rows: List[Dict[str, Any]] = json.loads(fp.read_text(encoding="utf-8"))

    by_cond: Dict[str, Dict[str, float]] = {}
    for rec in rows:
        if rec.get("model") != model:
            continue
        if not _is_pure_row(rec):
            continue
        tol = 1e-9 if model == "resnet" else 1e-6
        if not _match_strength(rec.get("strength_param_val"), cfg["strength"], tol):
            continue
        cond = rec.get("condition")
        if cond not in CONDITIONS:
            continue
        if cond in by_cond:
            raise RuntimeError(f"Duplicate row found: {model=} {cond=}")
        by_cond[cond] = {m: _to_float(rec.get(m)) for m, _ in METRICS}

    missing = [c for c in CONDITIONS if c not in by_cond]
    if missing:
        raise RuntimeError(f"{fp.name}: missing conditions for {model}: {missing}")
    return by_cond


def _plot_panel(ax: plt.Axes, metric_key: str, metric_label: str, all_data: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    x = np.arange(len(CONDITIONS))
    for model in ("resnet", "vit"):
        cfg = MODEL_CONFIG[model]
        y = np.array([all_data[model][cond][metric_key] for cond in CONDITIONS], dtype=float)
        ax.plot(
            x,
            y,
            color=cfg["color"],
            marker=cfg["marker"],
            linewidth=1.0,
            markersize=3.8,
            label=cfg["label"],
            zorder=3,
        )

    ax.set_title(metric_label, fontsize=9, pad=4)
    ax.set_xticks(x)
    ax.set_xticklabels(X_LABELS)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.45, color="#aaaaaa", zorder=0)
    for spine in ax.spines.values():
        spine.set_linewidth(0.55)
        spine.set_color("#2c2c2c")


def plot_figure(summary_dir: Path, out_pdf: Path, out_eps: Path | None) -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "Nimbus Roman"],
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "axes.linewidth": 0.55,
            "xtick.major.width": 0.45,
            "ytick.major.width": 0.45,
        }
    )

    all_data = {
        "resnet": _load_model_data(summary_dir, "resnet"),
        "vit": _load_model_data(summary_dir, "vit"),
    }

    fig, axes = plt.subplots(2, 2, figsize=(5.6, 3.6), constrained_layout=True)
    axes = axes.flatten()

    for ax, (metric_key, metric_label) in zip(axes, METRICS):
        _plot_panel(ax, metric_key, metric_label, all_data)
        if metric_key == "gradient_variance_domination_V":
            vals = []
            for model in ("resnet", "vit"):
                for cond in CONDITIONS:
                    v = all_data[model][cond][metric_key]
                    if np.isfinite(v) and v > 0:
                        vals.append(v)
            if vals:
                vmax = max(vals)
                vmin = min(vals)
                if vmax / max(vmin, 1e-12) >= 20:
                    ax.set_yscale("log")
                    ax.set_ylabel("log scale")
        if metric_key in ("gradient_reachability_A", "gradient_consistency_C", "perturbation_stability_S"):
            ax.set_ylabel("value")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=True,
        fancybox=False,
        edgecolor="#2c2c2c",
        facecolor="white",
        bbox_to_anchor=(0.5, 1.01),
        borderpad=0.3,
        handlelength=1.4,
    )

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
        default=REPO_ROOT / "outputs" / "figures" / "iccad_supporting_protocol_metrics.pdf",
    )
    p.add_argument("--out-eps", type=Path, default=None)
    args = p.parse_args()

    summary_dir = args.summary_dir if args.summary_dir.is_absolute() else (REPO_ROOT / args.summary_dir)
    out_pdf = args.out_pdf if args.out_pdf.is_absolute() else (REPO_ROOT / args.out_pdf)
    out_eps = None
    if args.out_eps is not None:
        out_eps = args.out_eps if args.out_eps.is_absolute() else (REPO_ROOT / args.out_eps)

    plot_figure(summary_dir.resolve(), out_pdf.resolve(), out_eps.resolve() if out_eps else None)
    print("Wrote", out_pdf)
    if out_eps:
        print("Wrote", out_eps)


if __name__ == "__main__":
    main()

