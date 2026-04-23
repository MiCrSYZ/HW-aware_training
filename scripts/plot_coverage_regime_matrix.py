"""
Compact coverage figure: degeneration regime (rows) vs dominant diagnostic category (columns).
Uses matplotlib + pandas only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Baselines for severe degradation: test_acc < 0.5 * acc_baseline
ACC_BASELINE: Dict[str, float] = {
    "resnet": 90.91,
    "vit": 80.64,
}

COL_ALIASES = {
    "model": ["model", "model_name"],
    "test_acc": ["test_acc"],
    "A": ["gradient_reachability", "gradient_reachability_A", "A"],
    "C": ["gradient_consistency", "gradient_consistency_C", "C"],
    "V": ["gradient_variance_domination", "gradient_variance_domination_V", "V"],
}

MECH_LABELS = ["low-A", "low/negative-C", "high-V", "uncovered"]
ROW_LABELS = ["Severely degraded", "Not severely degraded"]

RULE_LOW_A = 0.55
RULE_LOW_C = 0.0
RULE_HIGH_V = 2.0


def _first_existing(df: pd.DataFrame, keys: List[str]) -> Optional[str]:
    for k in keys:
        if k in df.columns:
            return k
    return None


def load_all_records(summary_dir: Path, file_pattern: str) -> pd.DataFrame:
    files = sorted(summary_dir.glob(file_pattern))
    if not files:
        raise FileNotFoundError(f"No files matched '{file_pattern}' in {summary_dir}")

    parts: List[pd.DataFrame] = []
    for fp in files:
        with fp.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            continue
        raw = pd.DataFrame(data)
        row = pd.DataFrame(index=raw.index)
        for std, aliases in COL_ALIASES.items():
            src = _first_existing(raw, aliases)
            row[std] = raw[src] if src is not None else np.nan
        row["source_file"] = fp.name
        parts.append(row)

    df = pd.concat(parts, ignore_index=True)
    df["test_acc"] = pd.to_numeric(df["test_acc"], errors="coerce")
    for c in ["A", "C", "V"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["model_key"] = (
        df["model"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"resnet-20": "resnet", "vit-tiny": "vit"})
    )
    return df


def normalize_model(m: str) -> Optional[str]:
    s = str(m).strip().lower()
    if "resnet" in s:
        return "resnet"
    if "vit" in s:
        return "vit"
    return None


def rule_flags(A: float, C: float, V: float) -> Tuple[bool, bool, bool]:
    low_a = np.isfinite(A) and A < RULE_LOW_A
    low_c = np.isfinite(C) and C < RULE_LOW_C
    high_v = np.isfinite(V) and V > RULE_HIGH_V
    return low_a, low_c, high_v


def dominant_mechanism(low_a: bool, low_c: bool, high_v: bool) -> str:
    if low_a:
        return "low-A"
    if low_c:
        return "low/negative-C"
    if high_v:
        return "high-V"
    return "uncovered"


def assign_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["model_key"] = out["model_key"].apply(normalize_model)
    out = out[out["model_key"].notna()].copy()

    baselines = out["model_key"].map(ACC_BASELINE)
    out["acc_baseline"] = baselines
    out["threshold_severe"] = 0.5 * out["acc_baseline"]
    out["severely_degraded"] = out["test_acc"] < out["threshold_severe"]

    lows = []
    doms = []
    multi_flags = []
    for _, r in out.iterrows():
        la, lc, hv = rule_flags(r["A"], r["C"], r["V"])
        lows.append((la, lc, hv))
        doms.append(dominant_mechanism(la, lc, hv))
        n_rules = int(la) + int(lc) + int(hv)
        multi_flags.append(n_rules > 1)
    out["dominant_mech"] = doms
    out["multi_rule"] = multi_flags
    return out


def build_contingency(df: pd.DataFrame) -> pd.DataFrame:
    """Rows: outcome regime; columns: mechanism labels."""
    rows = []
    for severe in [True, False]:
        sub = df[df["severely_degraded"] == severe]
        row_name = "Severely degraded" if severe else "Not severely degraded"
        counts = sub["dominant_mech"].value_counts()
        row = {m: int(counts.get(m, 0)) for m in MECH_LABELS}
        row["row_name"] = row_name
        rows.append(row)
    mat = pd.DataFrame(rows).set_index("row_name")
    mat = mat[[c for c in MECH_LABELS]]
    return mat


def row_percentages(mat: pd.DataFrame) -> pd.DataFrame:
    sums = mat.sum(axis=1).replace(0, np.nan)
    return mat.div(sums, axis=0) * 100.0


def plot_heatmap(
    mat: pd.DataFrame,
    out_pdf: Path,
    out_png: Path,
    title: str,
    dpi: int,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
        }
    )
    data = mat.values.astype(float)
    row_pct = row_percentages(mat).values

    fig, ax = plt.subplots(figsize=(5.2, 2.0), constrained_layout=True)
    im = ax.imshow(data, cmap="Blues", aspect="auto", vmin=0, vmax=max(data.max(), 1))

    ax.set_xticks(np.arange(len(MECH_LABELS)))
    ax.set_yticks(np.arange(len(ROW_LABELS)))
    ax.set_xticklabels(MECH_LABELS, rotation=25, ha="right")
    ax.set_yticklabels(ROW_LABELS)
    ax.set_xlabel("Dominant Diagnostic Category")
    ax.set_ylabel("Degeneration Regime")
    ax.set_title(title, fontsize=9.5)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            c = int(data[i, j])
            p = row_pct[i, j]
            if np.isnan(p):
                txt = f"{c}\n(—)"
            else:
                txt = f"{c}\n({p:.0f}%)"
            ax.text(j, i, txt, ha="center", va="center", color="#111111" if data[i, j] < data.max() * 0.55 else "white", fontsize=8)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Count")

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def print_stats(name: str, df: pd.DataFrame, mat: pd.DataFrame) -> None:
    n = len(df)
    n_uncovered = int((df["dominant_mech"] == "uncovered").sum())
    n_multi = int(df["multi_rule"].sum())

    print(f"\n=== {name} (n={n}) ===")
    print("Full contingency table (rows = degeneration regime, cols = dominant mechanism):")
    print(mat.to_string())
    print("\nRow-normalized percentages (within each row):")
    rp = row_percentages(mat)
    print(rp.round(2).to_string())

    print(f"\nUncovered (dominant label): {n_uncovered} / {n} ({100.0 * n_uncovered / max(n, 1):.2f}%)")
    print(f"Multiple rules satisfied simultaneously: {n_multi} / {n} ({100.0 * n_multi / max(n, 1):.2f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Coverage regime matrix heatmap (ResNet / ViT separate).")
    parser.add_argument("--summary_dir", type=Path, default=Path("outputs/summary"))
    parser.add_argument("--file_pattern", type=str, default="*.json")
    parser.add_argument("--out_pdf_resnet", type=Path, default=Path("outputs/summary/coverage_regime_matrix_resnet.pdf"))
    parser.add_argument("--out_png_resnet", type=Path, default=Path("outputs/summary/coverage_regime_matrix_resnet.png"))
    parser.add_argument("--out_pdf_vit", type=Path, default=Path("outputs/summary/coverage_regime_matrix_vit.pdf"))
    parser.add_argument("--out_png_vit", type=Path, default=Path("outputs/summary/coverage_regime_matrix_vit.png"))
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    df_raw = load_all_records(args.summary_dir, args.file_pattern)
    df = assign_rows(df_raw)
    df = df.dropna(subset=["test_acc"]).copy()

    # Combined
    mat_all = build_contingency(df)
    print_stats("ALL MODELS", df, mat_all)

    for model_key, label, pdf, png in [
        ("resnet", "ResNet", args.out_pdf_resnet, args.out_png_resnet),
        ("vit", "ViT", args.out_pdf_vit, args.out_png_vit),
    ]:
        dm = df[df["model_key"] == model_key]
        mat = build_contingency(dm)
        print_stats(f"{label} only", dm, mat)
        plot_heatmap(mat, pdf, png, title=f"Diagnostic coverage — {label}", dpi=args.dpi)
        print(f"Saved: {pdf}")
        print(f"Saved: {png}")


if __name__ == "__main__":
    main()
