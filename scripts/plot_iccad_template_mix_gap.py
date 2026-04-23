"""
ICCAD-style before/after: original frozen_additive_drift vs frozen_additive_drift_mixed.
Main panels: GRU and ViT (strongest gap shrink); ResNet omitted (optional tiny inset off by default).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

CONDITIONS = ("frozen_same", "frozen_train_new_test")

STRENGTH = {"gru": 1.0, "vit": 2.0}
MIX_PROB = 0.75
REPO_ROOT = Path(__file__).resolve().parents[1]


def _tm_ok_pure(rec: Dict[str, Any]) -> bool:
    tm = rec.get("template_mix_prob")
    return tm == "NaN" or tm is None


def _tm_ok_mixed(rec: Dict[str, Any]) -> bool:
    tm = rec.get("template_mix_prob")
    if isinstance(tm, (int, float)):
        return math.isclose(float(tm), MIX_PROB, rel_tol=0.0, abs_tol=1e-6)
    return False


def _strength_ok(model: str, val: Any) -> bool:
    t = STRENGTH[model]
    try:
        v = float(val)
    except (TypeError, ValueError):
        return False
    return math.isclose(v, t, rel_tol=0.0, abs_tol=1e-6)


def _load_model_block(path: Path, model: str) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Returns (original_same_new, mixed_same_new) each dict keys frozen_same, frozen_train_new_test."""
    with path.open("r", encoding="utf-8") as f:
        rows: List[Dict[str, Any]] = json.load(f)

    def pick(
        perturbation: str, tm_pred, out: Dict[str, float]
    ) -> None:
        seen: Dict[str, int] = {}
        for rec in rows:
            if rec.get("model") != model:
                continue
            if rec.get("perturbation_name") != perturbation:
                continue
            if not _strength_ok(model, rec.get("strength_param_val")):
                continue
            if not tm_pred(rec):
                continue
            cond = rec.get("condition")
            if cond not in CONDITIONS:
                continue
            seen[cond] = seen.get(cond, 0) + 1
            if seen[cond] > 1:
                raise RuntimeError(f"{path.name}: duplicate {model=} {perturbation=} {cond=}")
            out[cond] = float(rec["test_acc"])

    orig: Dict[str, float] = {}
    mixed: Dict[str, float] = {}
    pick("frozen_additive_drift", _tm_ok_pure, orig)
    pick("frozen_additive_drift_mixed", _tm_ok_mixed, mixed)
    for label, d in ("original", orig), ("mixed", mixed):
        for c in CONDITIONS:
            if c not in d:
                raise RuntimeError(f"{path.name}: missing {label} {c} for {model=}")
    return orig, mixed

def _panel(
    ax: plt.Axes,
    orig: Dict[str, float],
    mixed: Dict[str, float],
    model_name: str,
    edge: str,
    same_color: str,
    unseen_color: str,
) -> None:
    y_same_o, y_new_o = orig["frozen_same"], orig["frozen_train_new_test"]
    y_same_m, y_new_m = mixed["frozen_same"], mixed["frozen_train_new_test"]

    x_centers = np.array([0.0, 1.45])
    delta = 0.24
    w = 0.4
    same_x = x_centers - delta
    new_x = x_centers + delta

    bars_same = ax.bar(
        same_x,
        [y_same_o, y_same_m],
        w,
        label="Same",
        color=same_color,
        edgecolor=edge,
        linewidth=0.55,
        zorder=2,
    )
    bars_unseen = ax.bar(
        new_x,
        [y_new_o, y_new_m],
        w,
        label="Unseen",
        color=unseen_color,
        edgecolor=edge,
        linewidth=0.55,
        zorder=2,
    )

    # Annotate using actual bar centers to avoid any x-offset drift.
    for bar in list(bars_same) + list(bars_unseen):
        h = bar.get_height()
        x = bar.get_x() + bar.get_width() / 2
        ax.text(
            x,
            h + 1.0,
            f"{h:.1f}",
            ha="center",
            va="bottom",
            fontsize=6.5,
            color="black",
        )

    ax.set_title(model_name, fontsize=10, pad=10)
    ax.set_xticks(list(x_centers))
    ax.set_xticklabels(["Original", "Mixed"])
    ax.set_ylabel("Test accuracy (%)")
    ax.set_ylim(0, 100)
    x_min = min(same_x.min(), new_x.min()) - w * 0.6
    x_max = max(same_x.max(), new_x.max()) + w * 0.6
    ax.set_xlim(x_min, x_max)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.45, color="#999999", zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.55)
        spine.set_color(edge)

    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.0, 0.95),
        frameon=True,
        fancybox=False,
        edgecolor=edge,
        facecolor="white",
        borderpad=0.35,
        handlelength=1.1,
        fontsize=8,
    )


def plot_figure_one(
    summary_dir: Path,
    out_pdf: Path,
    out_eps: Path | None,
    model: str,
) -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "Nimbus Roman"],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "axes.linewidth": 0.55,
            "xtick.major.width": 0.45,
            "ytick.major.width": 0.45,
        }
    )

    json_name = {"gru": "gru_resampled.json", "vit": "vit_resampled.json"}[model]
    model_json = summary_dir / json_name
    o, m = _load_model_block(model_json, model)

    edge = "#2c2c2c"
    same_c = "#8eb4d2"
    unseen_c = "#d4a574"

    fig, ax = plt.subplots(1, 1, figsize=(2.9, 2.55), constrained_layout=True)
    _panel(
        ax,
        o,
        m,
        "GRU" if model == "gru" else "ViT",
        edge,
        same_c,
        unseen_c,
    )
    fig.tight_layout(pad=0.1)

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
        default=REPO_ROOT / "outputs" / "figures" / "iccad_template_mix_gap",
        help="Output prefix path (without extension). We will create _gru.pdf and _vit.pdf.",
    )
    p.add_argument("--out-eps", type=Path, default=None)
    args = p.parse_args()
    summary_dir = args.summary_dir if args.summary_dir.is_absolute() else (REPO_ROOT / args.summary_dir)
    out_prefix = args.out_pdf if args.out_pdf.is_absolute() else (REPO_ROOT / args.out_pdf)
    out_dir = out_prefix.parent
    out_prefix_stem = out_prefix.stem
    # If user accidentally passed something like ".../foo.pdf", keep stem stable.
    out_prefix_pdf_stem = out_prefix_stem

    for model, suffix in (("gru", "gru"), ("vit", "vit")):
        pdf_path = out_dir / f"{out_prefix_pdf_stem}_{suffix}.pdf"
        eps_path = None
        if args.out_eps is not None:
            out_eps_arg = args.out_eps if args.out_eps.is_absolute() else (REPO_ROOT / args.out_eps)
            out_eps_stem = out_eps_arg.stem
            out_eps_dir = out_eps_arg.parent
            eps_path = out_eps_dir / f"{out_eps_stem}_{suffix}.eps"
        plot_figure_one(
            summary_dir.resolve(),
            pdf_path.resolve(),
            eps_path.resolve() if eps_path else None,
            model=model,
        )
        print("Wrote", pdf_path)


if __name__ == "__main__":
    main()
