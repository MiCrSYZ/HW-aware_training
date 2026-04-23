import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COL_ALIASES = {
    "comp": ["comp", "condition"],
    "test_acc": ["test_acc"],
    "cos_sim": ["cos_sim"],
    "dead_zone_ratio_element_mean": ["dead_zone_ratio_element_mean"],
    "grad_top_k_energy_ratio_mean": ["grad_top_k_energy_ratio_mean"],
    "model_name": ["model_name", "model"],
    "noise_type": ["noise_type", "perturbation_name", "condition_name"],
}

METRICS = [
    ("cos_sim", "Cosine Similarity"),
    ("dead_zone_ratio_element_mean", "Dead-Zone Ratio"),
    ("grad_top_k_energy_ratio_mean", "Top-k Gradient Energy Ratio"),
]


def _first_existing(df: pd.DataFrame, keys: List[str]) -> Optional[str]:
    for k in keys:
        if k in df.columns:
            return k
    return None


def _parse_numeric_or_mean_pm(value):
    """Parse float or 'mean±std' text; return mean."""
    if value is None:
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    s = str(value).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return np.nan
    # Handle "0.123±0.045" or "+1.2e-3 ± 9e-4"
    m = re.match(r"^\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*(?:±.*)?$", s)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return np.nan
    return np.nan


def load_summary(summary_dir: Path, file_pattern: str) -> pd.DataFrame:
    files = sorted(summary_dir.glob(file_pattern))
    if not files:
        raise FileNotFoundError(f"No files matched '{file_pattern}' in {summary_dir}")

    frames = []
    for fp in files:
        with fp.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            continue
        raw = pd.DataFrame(data)
        out = pd.DataFrame(index=raw.index)
        for std, aliases in COL_ALIASES.items():
            src = _first_existing(raw, aliases)
            out[std] = raw[src] if src is not None else np.nan
        out["source_file"] = fp.name
        frames.append(out)

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if df.empty:
        return df

    # Normalize fields.
    df["comp"] = df["comp"].where(df["comp"].notna(), "").astype(str).str.strip().str.lower()
    df["model_name"] = df["model_name"].where(df["model_name"].notna(), "unknown").astype(str).str.strip()
    df["test_acc"] = pd.to_numeric(df["test_acc"], errors="coerce")
    for m, _ in METRICS:
        df[m] = df[m].apply(_parse_numeric_or_mean_pm)

    # Keep only explicit comp/no_comp to match intended comparison.
    df = df[df["comp"].isin(["comp", "no_comp"])].copy()
    df = df.dropna(subset=["test_acc"]).copy()
    return df


def overlap_stats(df: pd.DataFrame, metric: str) -> Dict[str, float]:
    d = df.dropna(subset=[metric]).copy()
    c = d[d["comp"] == "comp"][metric].to_numpy()
    n = d[d["comp"] == "no_comp"][metric].to_numpy()
    if len(c) == 0 or len(n) == 0:
        return {"comp_n": float(len(c)), "no_comp_n": float(len(n)), "overlap_len": np.nan, "overlap_ratio": np.nan}

    c_min, c_max = float(np.min(c)), float(np.max(c))
    n_min, n_max = float(np.min(n)), float(np.max(n))
    lo = max(c_min, n_min)
    hi = min(c_max, n_max)
    overlap_len = max(0.0, hi - lo)
    union_len = max(c_max, n_max) - min(c_min, n_min)
    overlap_ratio = (overlap_len / union_len) if union_len > 0 else np.nan
    return {
        "comp_n": float(len(c)),
        "no_comp_n": float(len(n)),
        "comp_min": c_min,
        "comp_max": c_max,
        "no_comp_min": n_min,
        "no_comp_max": n_max,
        "overlap_len": overlap_len,
        "overlap_ratio": overlap_ratio,
    }


def should_split_rows(df: pd.DataFrame, threshold: int) -> bool:
    return df["model_name"].nunique() >= 2 and len(df) > threshold


def _scatter_two_groups(ax: plt.Axes, d: pd.DataFrame, x: str, xlabel: str, y_limits: Tuple[float, float]) -> None:
    colors = {"comp": "#1f77b4", "no_comp": "#d62728"}
    for group in ["comp", "no_comp"]:
        g = d[d["comp"] == group].dropna(subset=[x, "test_acc"])
        ax.scatter(
            g[x],
            g["test_acc"],
            s=28,
            alpha=0.75,
            c=colors[group],
            label=group,
            edgecolors="white",
            linewidths=0.4,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylim(*y_limits)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)


def plot_figure(df: pd.DataFrame, out_pdf: Path, out_png: Path, dpi: int, split_threshold: int) -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    )

    y_min = float(np.nanmin(df["test_acc"]))
    y_max = float(np.nanmax(df["test_acc"]))
    pad = 0.05 * max(1e-6, y_max - y_min)
    y_limits = (y_min - pad, y_max + pad)

    if should_split_rows(df, split_threshold):
        models = sorted(df["model_name"].unique().tolist())[:2]
        fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.8), sharey=True, constrained_layout=True)
        handles = None
        labels = None
        for r, model in enumerate(models):
            dm = df[df["model_name"] == model]
            for c, (metric, xlabel) in enumerate(METRICS):
                ax = axes[r, c]
                _scatter_two_groups(ax, dm, metric, xlabel, y_limits)
                if c == 0:
                    ax.set_ylabel("Test Accuracy (%)")
                if r == 0:
                    ax.set_title(f"{model}")
                if handles is None:
                    handles, labels = ax.get_legend_handles_labels()
                ax.legend_.remove() if ax.legend_ else None
        if handles and labels:
            fig.legend(handles[:2], labels[:2], loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.06))
    else:
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True, constrained_layout=True)
        handles = None
        labels = None
        for ax, (metric, xlabel) in zip(axes, METRICS):
            _scatter_two_groups(ax, df, metric, xlabel, y_limits)
            if handles is None:
                handles, labels = ax.get_legend_handles_labels()
            ax.legend_.remove() if ax.legend_ else None
        axes[0].set_ylabel("Test Accuracy (%)")
        if handles and labels:
            fig.legend(handles[:2], labels[:2], loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.10))

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def print_overlap_report(df: pd.DataFrame) -> None:
    print("X-range overlap stats (comp vs no_comp):")
    for metric, xlabel in METRICS:
        s = overlap_stats(df, metric)
        if np.isnan(s.get("overlap_ratio", np.nan)):
            print(f"- {xlabel}: insufficient data for overlap.")
            continue
        print(
            f"- {xlabel}: comp[{s['comp_min']:.4g}, {s['comp_max']:.4g}] "
            f"vs no_comp[{s['no_comp_min']:.4g}, {s['no_comp_max']:.4g}], "
            f"overlap_len={s['overlap_len']:.4g}, overlap_ratio={s['overlap_ratio']:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot existing single-score metrics vs test accuracy for comp/no_comp comparison."
    )
    parser.add_argument("--summary_dir", type=Path, default=Path("outputs/summary"))
    parser.add_argument("--file_pattern", type=str, default="*.json")
    parser.add_argument("--out_pdf", type=Path, default=Path("outputs/summary/existing_metrics_scatter.pdf"))
    parser.add_argument("--out_png", type=Path, default=Path("outputs/summary/existing_metrics_scatter.png"))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--split_threshold", type=int, default=140, help="Auto split by model when crowded.")
    args = parser.parse_args()

    df = load_summary(args.summary_dir, args.file_pattern)
    if df.empty:
        raise ValueError("No valid records after filtering comp/no_comp and numeric test_acc.")

    plot_figure(df, args.out_pdf, args.out_png, dpi=args.dpi, split_threshold=args.split_threshold)
    print_overlap_report(df)
    print(f"Saved PDF: {args.out_pdf}")
    print(f"Saved PNG: {args.out_png}")


if __name__ == "__main__":
    main()

