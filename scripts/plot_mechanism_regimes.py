import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COL_ALIASES = {
    "perturbation_name": ["perturbation_name", "noise_type", "condition_name"],
    "experiment_name": ["experiment_name", "exp_name", "name"],
    "model": ["model", "model_name"],
    "test_acc": ["test_acc"],
    "A": ["gradient_reachability", "gradient_reachability_A", "A"],
    "C": ["gradient_consistency", "gradient_consistency_C", "C"],
    "V": ["gradient_variance_domination", "gradient_variance_domination_V", "V"],
    "stability": ["perturbation_stability", "perturbation_stability_S", "stability"],
}


def _first_existing(df: pd.DataFrame, keys: List[str]) -> Optional[str]:
    for k in keys:
        if k in df.columns:
            return k
    return None


def _load_json(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a list of records.")
    return pd.DataFrame(data)


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for std, aliases in COL_ALIASES.items():
        src = _first_existing(df, aliases)
        out[std] = df[src] if src is not None else np.nan

    for c in ["test_acc", "A", "C", "V", "stability"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["perturbation_name"] = out["perturbation_name"].where(out["perturbation_name"].notna(), "unknown").astype(str)
    out["experiment_name"] = out["experiment_name"].where(out["experiment_name"].notna(), "unnamed").astype(str)
    out["model"] = out["model"].where(out["model"].notna(), "unknown").astype(str)
    return out


def load_summary_dir(summary_dir: Path, file_pattern: str) -> pd.DataFrame:
    files = sorted(summary_dir.glob(file_pattern))
    if not files:
        raise FileNotFoundError(f"No JSON matched '{file_pattern}' in {summary_dir}")

    frames = []
    for fp in files:
        d = _normalize(_load_json(fp))
        d["source_file"] = fp.name
        frames.append(d)

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.dropna(subset=["A", "C", "V", "test_acc"]).copy()
    dedup_cols = ["perturbation_name", "experiment_name", "model", "test_acc", "A", "C", "V"]
    merged = merged.drop_duplicates(subset=dedup_cols, keep="first").reset_index(drop=True)
    return merged


def _rank01(series: pd.Series, ascending: bool) -> pd.Series:
    return series.rank(method="average", pct=True, ascending=ascending)


def _score_cases(df: pd.DataFrame) -> Dict[str, pd.Series]:
    # Lower score means more representative for target regime.
    score_low_a = _rank01(df["A"], ascending=True) + _rank01(df["test_acc"], ascending=True)

    c_neg_bonus = np.where(df["C"] < 0, 0.15, 0.0)
    score_low_c = _rank01(df["C"], ascending=True) + _rank01(df["test_acc"], ascending=True) - c_neg_bonus

    score_high_v = _rank01(df["V"], ascending=False) + _rank01(df["test_acc"], ascending=True)
    return {
        "low-A": score_low_a,
        "low/negative-C": score_low_c,
        "high-V": score_high_v,
    }


def _select_one(
    df: pd.DataFrame,
    scores: pd.Series,
    used_indices: set,
    used_families: set,
) -> int:
    order = scores.sort_values().index.tolist()
    # Prefer new perturbation families.
    for idx in order:
        if idx in used_indices:
            continue
        fam = df.at[idx, "perturbation_name"]
        if fam not in used_families:
            return idx
    for idx in order:
        if idx not in used_indices:
            return idx
    raise RuntimeError("Unable to select representative case.")


def select_representatives(df: pd.DataFrame) -> Dict[str, pd.Series]:
    scores = _score_cases(df)
    used_indices = set()
    used_families = set()
    picked: Dict[str, pd.Series] = {}

    for regime in ["low-A", "low/negative-C", "high-V"]:
        idx = _select_one(df, scores[regime], used_indices, used_families)
        used_indices.add(idx)
        used_families.add(df.at[idx, "perturbation_name"])
        picked[regime] = df.loc[idx]

    return picked


def _find_nearest_row(
    df: pd.DataFrame,
    *,
    model: str,
    perturbation: str,
    experiment_name: Optional[str],
    a: Optional[float],
    c: Optional[float],
    v: Optional[float],
    test_acc: Optional[float],
) -> Optional[pd.Series]:
    d = df[
        (df["model"].astype(str).str.lower() == model.lower())
        & (df["perturbation_name"].astype(str) == perturbation)
    ].copy()
    if experiment_name:
        d = d[d["experiment_name"].astype(str) == experiment_name]
    if d.empty:
        return None

    # Weighted distance on provided fields only.
    dist = np.zeros(len(d), dtype=float)
    wsum = np.zeros(len(d), dtype=float)
    for col, target, w in [("A", a, 1.0), ("C", c, 1.0), ("V", v, 0.7), ("test_acc", test_acc, 0.8)]:
        if target is None:
            continue
        vals = pd.to_numeric(d[col], errors="coerce").to_numpy()
        mask = np.isfinite(vals)
        if not mask.any():
            continue
        # robust scale to avoid V dominating
        scale = np.nanstd(vals)
        if not np.isfinite(scale) or scale < 1e-8:
            scale = max(abs(float(target)), 1.0)
        dd = np.zeros_like(vals, dtype=float) + 1e6
        dd[mask] = np.abs(vals[mask] - float(target)) / scale
        dist += w * dd
        wsum += w
    score = dist / np.maximum(wsum, 1e-12)
    best_idx = d.index[int(np.argmin(score))]
    return df.loc[best_idx]


def select_fixed_cases(df: pd.DataFrame) -> Dict[str, pd.Series]:
    # Cases specified by user for paper figure.
    requested = {
        "low-A": {
            "model": "resnet",
            "perturbation_name": "rank_collapse",
            "experiment_name": "C3_step_wise_resampled",
            "A": 0.4989,
            "C": 0.8032,
            "V": 0.2511,
            "test_acc": 26.45,
        },
        "low/negative-C": {
            "model": "vit",
            "perturbation_name": "deterministic_clip",
            "experiment_name": None,
            "A": 0.83,
            "C": -0.396,
            "V": None,  # auto-filled from nearest matched record
            "test_acc": 31.6,
        },
        "high-V": {
            "model": "vit",
            "perturbation_name": "frozen_additive_drift_resampled",
            "experiment_name": None,
            "A": 0.975,
            "C": None,  # auto-filled from nearest matched record
            "V": 5.59,
            "test_acc": 17.13,
        },
    }

    selected: Dict[str, pd.Series] = {}
    for regime, spec in requested.items():
        r = _find_nearest_row(
            df,
            model=spec["model"],
            perturbation=spec["perturbation_name"],
            experiment_name=spec["experiment_name"],
            a=spec["A"],
            c=spec["C"],
            v=spec["V"],
            test_acc=spec["test_acc"],
        )
        if r is None:
            # fallback: create row directly from provided values
            selected[regime] = pd.Series(
                {
                    "model": spec["model"],
                    "perturbation_name": spec["perturbation_name"],
                    "experiment_name": spec["experiment_name"] or "manual_case",
                    "A": spec["A"],
                    "C": spec["C"],
                    "V": spec["V"],
                    "test_acc": spec["test_acc"],
                }
            )
        else:
            selected[regime] = r
    return selected


def _short_title(perturbation: str, experiment: str, max_len: int = 38) -> str:
    t = f"{perturbation} | {experiment}"
    return t if len(t) <= max_len else t[: max_len - 1] + "..."


def make_figure(
    selected: Dict[str, pd.Series],
    out_pdf: Path,
    out_png: Path,
    dpi: int = 300,
) -> Tuple[float, str]:
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )

    regimes = ["low-A", "low/negative-C", "high-V"]
    rows = [selected[r] for r in regimes]

    a_vals = [float(r["A"]) for r in rows]
    c_vals = [float(r["C"]) for r in rows]
    v_vals = [float(r["V"]) for r in rows]
    ac_max = max([abs(x) for x in a_vals + c_vals] + [1e-12])
    v_max = max([abs(x) for x in v_vals] + [1e-12])

    v_scale = 1.0
    scale_note = "V shown in original scale."
    if v_max > 5.0 * ac_max:
        p = int(np.floor(np.log10(v_max / max(ac_max, 1e-12))))
        v_scale = float(10**max(p, 1))
        scale_note = f"V bars are scaled by 1/{v_scale:g} for display."

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), constrained_layout=True)
    bar_colors = ["#4c78a8", "#f58518", "#54a24b"]

    y_values_all = []
    for r in rows:
        y_values_all.extend([float(r["A"]), float(r["C"]), float(r["V"]) / v_scale])
    ymin = min(y_values_all + [0.0])
    ymax = max(y_values_all + [0.0])
    ypad = 0.12 * max(ymax - ymin, 1e-6)

    for ax, regime, row in zip(axes, regimes, rows):
        values = [float(row["A"]), float(row["C"]), float(row["V"]) / v_scale]
        bars = ax.bar(["A", "C", "V"], values, color=bar_colors, width=0.62, edgecolor="#f4f4f4", linewidth=0.6)
        ax.axhline(0.0, color="#666666", linewidth=0.8)
        ax.set_ylim(ymin - ypad, ymax + ypad)
        ax.set_title(f"{regime}\n{_short_title(str(row['perturbation_name']), str(row['experiment_name']))}")
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
        acc = float(row["test_acc"])
        ax.text(
            0.03,
            0.96,
            f"Acc: {acc:.2f}%",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#dddddd", linewidth=0.5, alpha=0.9),
        )
        for b in bars:
            h = b.get_height()
            ax.text(
                b.get_x() + b.get_width() / 2.0,
                h + (0.015 * (ymax - ymin + 1e-12) if h >= 0 else -0.015 * (ymax - ymin + 1e-12)),
                f"{h:.2g}",
                ha="center",
                va="bottom" if h >= 0 else "top",
                fontsize=7,
                color="#333333",
            )

    axes[0].set_ylabel("Metric Value")
    for ax in axes[1:]:
        ax.set_ylabel("")

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return v_scale, scale_note


def print_selected(selected: Dict[str, pd.Series], scale_note: str) -> None:
    print("Selected representative experiments:")
    for regime in ["low-A", "low/negative-C", "high-V"]:
        r = selected[regime]
        print(
            f"[{regime}] model={r['model']} | perturbation={r['perturbation_name']} | "
            f"experiment={r['experiment_name']} | acc={float(r['test_acc']):.2f}% | "
            f"A={float(r['A']):.4g}, C={float(r['C']):.4g}, V={float(r['V']):.4g}"
        )
    print(scale_note)
    print("Regime threshold check:")
    r1 = selected["low-A"]
    r2 = selected["low/negative-C"]
    r3 = selected["high-V"]
    print(f"  low-A: A={float(r1['A']):.4g} (<0.55: {float(r1['A']) < 0.55})")
    print(f"  low/negative-C: C={float(r2['C']):.4g} (<0: {float(r2['C']) < 0.0})")
    print(f"  high-V: V={float(r3['V']):.4g} (>2: {float(r3['V']) > 2.0})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create compact ICCAD-style mechanism figure with representative degeneration regimes."
    )
    parser.add_argument("--summary_dir", type=Path, default=Path("outputs/summary"))
    parser.add_argument("--file_pattern", type=str, default="*.json")
    parser.add_argument("--out_pdf", type=Path, default=Path("outputs/summary/mechanism_regimes.pdf"))
    parser.add_argument("--out_png", type=Path, default=Path("outputs/summary/mechanism_regimes.png"))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--use_fixed_cases",
        action="store_true",
        help="Use the paper-specified 3 representative cases instead of automatic selection.",
    )
    args = parser.parse_args()

    df = load_summary_dir(args.summary_dir, args.file_pattern)
    selected = select_fixed_cases(df) if args.use_fixed_cases else select_representatives(df)
    _, scale_note = make_figure(selected, args.out_pdf, args.out_png, dpi=args.dpi)
    print_selected(selected, scale_note)
    print(f"Saved PDF: {args.out_pdf}")
    print(f"Saved PNG: {args.out_png}")


if __name__ == "__main__":
    main()

