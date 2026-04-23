import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _load_json_allow_nan(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError(f"Expected top-level list in {path}, got {type(data)}")
    return data  # type: ignore[return-value]


def _fmt_strength(name: str, val: Any) -> str:
    try:
        f = float(val)
        if math.isfinite(f):
            return f"{name}={f:g}"
    except Exception:
        pass
    return f"{name}={val}"


def _row_label(entry: Dict[str, Any]) -> str:
    noise = str(entry.get("perturbation_name", "unknown"))
    cond = str(entry.get("condition", "unknown"))
    spn = str(entry.get("strength_param_name", "strength"))
    spv = entry.get("strength_param_val", np.nan)
    return f"{noise} | {cond} | {_fmt_strength(spn, spv)}"


def _sort_key(entry: Dict[str, Any]) -> Tuple[str, int, float]:
    noise = str(entry.get("perturbation_name", ""))
    cond = str(entry.get("condition", ""))
    cond_rank = 0 if cond == "no_comp" else 1
    try:
        strength = float(entry.get("strength_param_val", np.nan))
    except Exception:
        strength = float("nan")
    if not np.isfinite(strength):
        strength = float("inf")
    return (noise, cond_rank, strength)


def _extract_table(
    data: Sequence[Dict[str, Any]],
    layers: Sequence[str],
    metric_prefix: str,
    key_template: str,
    *,
    unstable_consistency_abs_gt: Optional[float] = None,
) -> pd.DataFrame:
    rows = []
    index = []
    for e in sorted(data, key=_sort_key):
        row = {}
        for layer in layers:
            k = key_template.format(prefix=metric_prefix, layer=layer)
            v = e.get(k, np.nan)
            try:
                v = float(v)
            except Exception:
                v = np.nan
            if unstable_consistency_abs_gt is not None and np.isfinite(v) and abs(v) > unstable_consistency_abs_gt:
                v = np.nan
            row[layer] = v
        rows.append(row)
        index.append(_row_label(e))
    df = pd.DataFrame(rows, index=index, columns=list(layers))
    return df


def _build_debug_csv(
    data: Sequence[Dict[str, Any]],
    *,
    layers: Sequence[str],
    a_key_template: str,
    c_key_template: str,
    b_key_template: str,
) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for e in sorted(data, key=_sort_key):
        rec: Dict[str, Any] = {
            "model": e.get("model"),
            "perturbation_name": e.get("perturbation_name"),
            "condition": e.get("condition"),
            "strength_param_name": e.get("strength_param_name"),
            "strength_param_val": e.get("strength_param_val"),
            "test_acc": e.get("test_acc"),
            "gradient_reachability_A": e.get("gradient_reachability_A"),
            "gradient_consistency_C": e.get("gradient_consistency_C"),
            "gradient_B_mean": e.get("gradient_B_mean"),
            "row_label": _row_label(e),
        }
        for layer in layers:
            rec[f"A_{layer}"] = e.get(a_key_template.format(prefix="gradient_reachability", layer=layer))
            rec[f"C_{layer}"] = e.get(c_key_template.format(prefix="gradient_consistency", layer=layer))
            rec[f"B_mean_{layer}"] = e.get(b_key_template.format(prefix="gradient_B_mean", layer=layer))
        records.append(rec)
    return pd.DataFrame.from_records(records)


def _heatmap(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    title: str,
    cmap: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    center: Optional[float] = None,
    cbar_label: Optional[str] = None,
) -> None:
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="lightgray")

    sns.heatmap(
        df,
        ax=ax,
        cmap=cmap_obj,
        vmin=vmin,
        vmax=vmax,
        center=center,
        mask=df.isna(),
        linewidths=0.25,
        linecolor="white",
        cbar=True,
        cbar_kws={"label": cbar_label} if cbar_label else None,
    )
    ax.set_title(title)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Condition")
    ax.tick_params(axis="y", labelsize=7)
    ax.tick_params(axis="x", labelrotation=0)


def make_arch_figure(
    *,
    json_path: Path,
    arch_name: str,
    layers: Sequence[str],
    a_key_template: str,
    c_key_template: str,
    b_key_template: str,
    out_png: Path,
    out_csv: Path,
) -> None:
    data = _load_json_allow_nan(json_path)

    a_df = _extract_table(data, layers, "gradient_reachability", a_key_template)
    c_df = _extract_table(
        data,
        layers,
        "gradient_consistency",
        c_key_template,
        unstable_consistency_abs_gt=5.0,
    )
    b_df_raw = _extract_table(data, layers, "gradient_B_mean", b_key_template)
    b_df = np.log10(b_df_raw + 1e-8)

    debug_df = _build_debug_csv(
        data,
        layers=layers,
        a_key_template=a_key_template,
        c_key_template=c_key_template,
        b_key_template=b_key_template,
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    debug_df.to_csv(out_csv, index=False)

    sns.set_theme(style="white", font_scale=0.9)
    fig, axes = plt.subplots(1, 3, figsize=(18, max(6, 0.25 * len(a_df) + 2)), constrained_layout=True)
    fig.suptitle(f"{arch_name}: Layer-wise Gradient Diagnostics", y=1.02, fontsize=14)

    _heatmap(
        axes[0],
        a_df,
        title="Gradient Reachability (A) — layer-wise",
        cmap="coolwarm",
        vmin=0.0,
        vmax=1.0,
        center=0.5,
        cbar_label="A",
    )
    _heatmap(
        axes[1],
        c_df,
        title="Gradient Consistency (C) — layer-wise",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        center=0.0,
        cbar_label="C",
    )
    _heatmap(
        axes[2],
        b_df,
        title="log10(Gradient B_mean + 1e-8) — layer-wise",
        cmap="magma",
        vmin=np.nanmin(b_df.values) if np.isfinite(np.nanmin(b_df.values)) else None,
        vmax=np.nanmax(b_df.values) if np.isfinite(np.nanmax(b_df.values)) else None,
        center=None,
        cbar_label="log10(B_mean + 1e-8)",
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def maybe_make_scatter(
    *,
    resnet_json: Path,
    vit_json: Path,
    out_png: Path,
) -> None:
    def _prep(path: Path, label: str) -> pd.DataFrame:
        data = _load_json_allow_nan(path)
        rows = []
        for e in data:
            try:
                a = float(e.get("gradient_reachability_A", np.nan))
            except Exception:
                a = np.nan
            try:
                b = float(e.get("gradient_B_mean", np.nan))
            except Exception:
                b = np.nan
            try:
                acc = float(e.get("test_acc", np.nan))
            except Exception:
                acc = np.nan
            rows.append(
                {
                    "arch": label,
                    "A": a,
                    "B_mean": b,
                    "log10_B_mean": np.log10(b + 1e-8) if np.isfinite(b) else np.nan,
                    "test_acc": acc,
                }
            )
        return pd.DataFrame(rows)

    df = pd.concat([_prep(resnet_json, "resnet20"), _prep(vit_json, "vit_tiny")], ignore_index=True)
    df = df[np.isfinite(df["A"]) & np.isfinite(df["log10_B_mean"])]
    if df.empty or not np.isfinite(df["test_acc"]).any():
        return

    sns.set_theme(style="whitegrid", font_scale=0.95)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
    sc = ax.scatter(
        df["A"],
        df["log10_B_mean"],
        c=df["test_acc"],
        cmap="viridis",
        s=30,
        alpha=0.85,
        edgecolors="none",
    )
    ax.set_xlabel("gradient_reachability_A")
    ax.set_ylabel("log10(gradient_B_mean + 1e-8)")
    ax.set_title("Global A vs B_mean (colored by test_acc)")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("test_acc")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Visualize layer-wise gradient diagnostics heatmaps for ResNet and ViT from JSON tables."
    )
    p.add_argument("--resnet_json", type=Path, default=Path("outputs/resnet_data.json"))
    p.add_argument("--vit_json", type=Path, default=Path("outputs/vit_data.json"))
    p.add_argument("--out_dir", type=Path, default=Path("outputs"))
    p.add_argument("--make_scatter", action="store_true", help="Also generate optional global scatter plot.")
    args = p.parse_args()

    out_dir: Path = args.out_dir
    make_arch_figure(
        json_path=args.resnet_json,
        arch_name="ResNet-20",
        layers=["stem", "layer1", "layer2", "layer3"],
        a_key_template="{prefix}_{layer}",
        c_key_template="{prefix}_{layer}",
        b_key_template="{prefix}_{layer}",
        out_png=out_dir / "resnet_layerwise_heatmap.png",
        out_csv=out_dir / "resnet_layerwise_metrics.csv",
    )

    # ViT JSON uses "blocks_1/3/5" (not "block1/3/5") in current outputs.
    make_arch_figure(
        json_path=args.vit_json,
        arch_name="ViT-Tiny",
        layers=["blocks_1", "blocks_3", "blocks_5"],
        a_key_template="{prefix}_{layer}",
        c_key_template="{prefix}_{layer}",
        b_key_template="{prefix}_{layer}",
        out_png=out_dir / "vit_layerwise_heatmap.png",
        out_csv=out_dir / "vit_layerwise_metrics.csv",
    )

    if args.make_scatter:
        maybe_make_scatter(
            resnet_json=args.resnet_json,
            vit_json=args.vit_json,
            out_png=out_dir / "global_A_vs_Bmean_scatter.png",
        )


if __name__ == "__main__":
    main()

