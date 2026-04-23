"""
将 outputs/metrics 下所有 JSON 文件整理成 CSV 表。
- resnet 系列写入 metrics_resnet.csv
- vit 系列写入 metrics_vit.csv
列统一包含：
filename, gradient_reachability, gradient_consistency, gradient_variance_domination,
gradient_B_mean, perturbation_stability, sign_coupled_P_positive, sign_coupled_P_negative,
sign_coupled_P_zero 以及各层/blocks 的指标（若无则留空）。
"""
import json
import csv
from pathlib import Path

METRICS_DIR = Path(__file__).resolve().parent.parent / "outputs" / "metrics"
OUT_CSV_RESNET = METRICS_DIR / "metrics_resnet.csv"
OUT_CSV_VIT = METRICS_DIR / "metrics_vit.csv"

BASE_COLUMNS = [
    "filename",
    "gradient_reachability",
    "gradient_consistency",
    "gradient_variance_domination",
    "gradient_B_mean",
    "perturbation_stability",
    "sign_coupled_P_positive",
    "sign_coupled_P_negative",
    "sign_coupled_P_zero",
]

# resnet 额外关心的分层指标
RESNET_EXTRA = [
    "gradient_reachability_stem",
    "gradient_consistency_stem",
    "gradient_reachability_layer1",
    "gradient_consistency_layer1",
    "gradient_reachability_layer2",
    "gradient_consistency_layer2",
    "gradient_reachability_layer3",
    "gradient_consistency_layer3",
    "gradient_variance_domination_stem",
    "gradient_variance_domination_layer1",
    "gradient_variance_domination_layer2",
    "gradient_variance_domination_layer3",
    "gradient_B_mean_stem",
    "gradient_B_mean_layer1",
    "gradient_B_mean_layer2",
    "gradient_B_mean_layer3",
]

# vit 额外关心的 blocks 指标
VIT_EXTRA = [
    "gradient_reachability_blocks_1",
    "gradient_consistency_blocks_1",
    "gradient_reachability_blocks_3",
    "gradient_consistency_blocks_3",
    "gradient_reachability_blocks_5",
    "gradient_consistency_blocks_5",
    "gradient_variance_domination_blocks_1",
    "gradient_variance_domination_blocks_3",
    "gradient_variance_domination_blocks_5",
    "gradient_B_mean_blocks_1",
    "gradient_B_mean_blocks_3",
    "gradient_B_mean_blocks_5",
]

RESNET_COLUMNS = BASE_COLUMNS + RESNET_EXTRA
VIT_COLUMNS = BASE_COLUMNS + VIT_EXTRA

def build_rows(columns, file_filter):
    rows = []
    for jpath in sorted(METRICS_DIR.rglob("*.json")):
        rel = jpath.relative_to(METRICS_DIR)
        name = str(rel).replace("\\", "/")
        if not file_filter(name):
            continue
        with open(jpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        row = [name]
        for col in columns[1:]:
            row.append(data.get(col, ""))
        rows.append(row)
    return rows


def main():
    # resnet: 目录名里包含 resnet_20
    resnet_rows = build_rows(
        RESNET_COLUMNS, lambda fname: "resnet_20" in fname
    )
    with open(OUT_CSV_RESNET, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(RESNET_COLUMNS)
        w.writerows(resnet_rows)

    # vit: 目录名里包含 vit_tiny
    vit_rows = build_rows(
        VIT_COLUMNS, lambda fname: "vit_tiny" in fname
    )
    with open(OUT_CSV_VIT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(VIT_COLUMNS)
        w.writerows(vit_rows)

    print(f"已写入 resnet 行数: {len(resnet_rows)} 到 {OUT_CSV_RESNET}")
    print(f"已写入 vit 行数: {len(vit_rows)} 到 {OUT_CSV_VIT}")

if __name__ == "__main__":
    main()
