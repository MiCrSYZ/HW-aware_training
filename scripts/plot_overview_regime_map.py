import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Column aliases let you reuse this script across differently named JSON schemas.
COL_ALIASES = {
    "model_name": ["model_name", "model"],
    "comp": ["comp", "condition"],
    "test_acc": ["test_acc"],
    "A": ["A", "gradient_reachability_A", "gradient_reachability"],
    "C": ["C", "gradient_consistency_C", "gradient_consistency"],
    "V": ["V", "gradient_variance_domination_V", "gradient_variance_domination"],
    "noise_type": ["noise_type", "perturbation_name"],
}


def pick_first_existing(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    for name in names:
        if name in df.columns:
            return name
    return None


def load_json_records(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, got {type(data)}")
    return pd.DataFrame(data)


def normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    for std_name, candidates in COL_ALIASES.items():
        src = pick_first_existing(df, candidates)
        out[std_name] = df[src] if src is not None else np.nan

    # Ensure expected dtypes.
    for col in ["test_acc", "A", "C", "V"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["comp"] = out["comp"].where(out["comp"].notna(), "").astype(str).str.strip().str.lower()
    out["model_name"] = out["model_name"].astype(str).str.strip()
    out["noise_type"] = out["noise_type"].astype(str)
    return out


def filter_by_comp_mode(df: pd.DataFrame, comp_mode: str) -> pd.DataFrame:
    """
    comp_mode:
      - strict: only explicit 'comp'
      - include_missing: include explicit 'comp' OR missing/empty comp flag
    """
    comp = df["comp"].astype(str).str.strip().str.lower()
    is_comp = comp == "comp"
    is_missing = comp.isin(["", "nan", "none", "null"])

    if comp_mode == "strict":
        return df[is_comp].copy()
    if comp_mode == "include_missing":
        return df[is_comp | is_missing].copy()
    raise ValueError(f"Unknown comp_mode: {comp_mode}")


def read_summary_dir(summary_dir: Path, file_pattern: str) -> pd.DataFrame:
    files = sorted(summary_dir.glob(file_pattern))
    if not files:
        raise FileNotFoundError(f"No files matched '{file_pattern}' in {summary_dir}")

    parts = []
    for fp in files:
        df = load_json_records(fp)
        df = normalize_schema(df)
        df["source_file"] = fp.name
        parts.append(df)

    merged = pd.concat(parts, ignore_index=True)
    # Drop complete duplicates to avoid double-counting when multiple JSONs overlap.
    dedup_cols = ["model_name", "comp", "test_acc", "A", "C", "V", "noise_type"]
    merged = merged.drop_duplicates(subset=dedup_cols, keep="first").reset_index(drop=True)
    return merged


def choose_layout(df_comp: pd.DataFrame, crowded_threshold: int) -> str:
    model_count = df_comp["model_name"].nunique()
    n_points = len(df_comp)
    if model_count >= 2 and n_points > crowded_threshold:
        return "2x2"
    return "1x2"


def style_axes(ax: plt.Axes) -> None:
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def plot_overview(
    df_comp: pd.DataFrame,
    out_pdf: Path,
    out_png: Path,
    dpi: int = 300,
    crowded_threshold: int = 90,
) -> None:
    if df_comp.empty:
        raise ValueError("No 'comp' records found after filtering.")

    layout = choose_layout(df_comp, crowded_threshold=crowded_threshold)
    cmap = "viridis"
    marker_size = 42
    edge_color = "#f2f2f2"
    edge_width = 0.5

    vmin = float(df_comp["test_acc"].min())
    vmax = float(df_comp["test_acc"].max())

    if layout == "1x2":
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)
        ax_left, ax_right = axes

        sc = ax_left.scatter(
            df_comp["A"],
            df_comp["V"],
            c=df_comp["test_acc"],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            s=marker_size,
            edgecolors=edge_color,
            linewidths=edge_width,
            alpha=0.95,
        )
        ax_right.scatter(
            df_comp["C"],
            df_comp["V"],
            c=df_comp["test_acc"],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            s=marker_size,
            edgecolors=edge_color,
            linewidths=edge_width,
            alpha=0.95,
        )

        ax_left.set_xlabel("A (Pathway Reachability)")
        ax_left.set_ylabel("V (Gradient Variance Ratio)")
        ax_right.set_xlabel("C (Signal Alignment)")
        ax_right.set_ylabel("V (Gradient Variance Ratio)")
        style_axes(ax_left)
        style_axes(ax_right)

        cbar = fig.colorbar(sc, ax=[ax_left, ax_right], shrink=0.95, pad=0.02)
        cbar.set_label("Test Accuracy (%)")

    else:
        models = sorted(df_comp["model_name"].dropna().unique().tolist())[:2]
        model_a, model_b = models[0], models[1]
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), constrained_layout=True)

        for row_idx, model in enumerate([model_a, model_b]):
            d = df_comp[df_comp["model_name"] == model]
            ax_l = axes[row_idx, 0]
            ax_r = axes[row_idx, 1]

            sc = ax_l.scatter(
                d["A"],
                d["V"],
                c=d["test_acc"],
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                s=marker_size,
                edgecolors=edge_color,
                linewidths=edge_width,
                alpha=0.95,
            )
            ax_r.scatter(
                d["C"],
                d["V"],
                c=d["test_acc"],
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                s=marker_size,
                edgecolors=edge_color,
                linewidths=edge_width,
                alpha=0.95,
            )
            ax_l.set_title(f"{model}: A-V", fontsize=10)
            ax_r.set_title(f"{model}: C-V", fontsize=10)
            ax_l.set_xlabel("A (Pathway Reachability)")
            ax_l.set_ylabel("V (Gradient Variance Ratio)")
            ax_r.set_xlabel("C (Signal Alignment)")
            ax_r.set_ylabel("V (Gradient Variance Ratio)")
            style_axes(ax_l)
            style_axes(ax_r)

        cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.95, pad=0.01)
        cbar.set_label("Test Accuracy (%)")

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def print_point_summary(df_comp: pd.DataFrame) -> None:
    counts = df_comp.groupby("model_name").size().sort_values(ascending=False)
    print("Plotted points per model (comp only):")
    for model, n in counts.items():
        print(f"  - {model}: {int(n)}")
    print(f"Total points: {int(len(df_comp))}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate ICCAD-style overview regime map from summary JSONs."
    )
    parser.add_argument("--summary_dir", type=Path, default=Path("outputs/summary"))
    parser.add_argument("--file_pattern", type=str, default="*_data.json")
    parser.add_argument("--out_pdf", type=Path, default=Path("outputs/summary/overview_regime_map.pdf"))
    parser.add_argument("--out_png", type=Path, default=Path("outputs/summary/overview_regime_map.png"))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--crowded_threshold", type=int, default=90)
    parser.add_argument(
        "--comp_mode",
        type=str,
        default="include_missing",
        choices=["strict", "include_missing"],
        help="How to filter for 'comp' experiments.",
    )
    args = parser.parse_args()

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

    df = read_summary_dir(args.summary_dir, args.file_pattern)
    df_comp = filter_by_comp_mode(df, args.comp_mode)
    df_comp = df_comp.dropna(subset=["A", "C", "V", "test_acc"])

    print(f"comp_mode: {args.comp_mode}")
    print_point_summary(df_comp)
    plot_overview(
        df_comp=df_comp,
        out_pdf=args.out_pdf,
        out_png=args.out_png,
        dpi=args.dpi,
        crowded_threshold=args.crowded_threshold,
    )
    print(f"Saved PDF: {args.out_pdf}")
    print(f"Saved PNG: {args.out_png}")


if __name__ == "__main__":
    main()

