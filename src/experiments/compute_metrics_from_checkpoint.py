"""
从 checkpoint 直接计算梯度质量指标（A, C, V, S）及原有 dead-zone / effective rank，无需重跑训练。

用法:
  python -m src.experiments.compute_metrics_from_checkpoint --checkpoint path/to/model_final.pth
  python -m src.experiments.compute_metrics_from_checkpoint --checkpoint path/to/model_final.pth --device cpu
  python -m src.experiments.compute_metrics_from_checkpoint --checkpoint path/to/model_final.pth --K 16 --output metrics.json
"""

import argparse
import json
import logging
import os
import sys

import torch

# 保证能导入项目模块
if __name__ == "__main__" and __package__ is None:
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _root not in sys.path:
        sys.path.insert(0, _root)

from src.experiments.run_experiment_synth import build_grad_model_and_loader_from_config
from src.utils.checkpoint import load_checkpoint
from src.utils.gradient_quality_metrics import (
    collect_gradient_quality_metrics,
    gradient_reachability_and_consistency,
    gradient_variance_domination,
    gradient_B_mean,
    perturbation_structural_stability,
    sign_coupled_scaling_P,
    gradient_reachability_and_consistency_layerwise,
    gradient_variance_domination_layerwise,
    gradient_B_mean_layerwise,
)
from src.memristor.synth_noise_wrappers import SynthNoiseLinear, SynthNoiseConv2d

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Compute gradient metrics from checkpoint")
    parser.add_argument("--checkpoint", "-c", required=True, help="Path to checkpoint (.pth)")
    parser.add_argument("--device", "-d", default=None, help="Device (cuda/cpu). Default: cuda if available")
    parser.add_argument("--K", type=int, default=8, help="Number of noise samples for V and S (default: 8)")
    parser.add_argument("--seed", type=int, default=0, help="Seed for metric computation (default: 0)")
    parser.add_argument("--output", "-o", default=None, help="Optional JSON file to write metrics")
    parser.add_argument("--no-gq", action="store_true", help="Skip dead-zone / effective rank (gradient quality)")
    args = parser.parse_args()

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    if not os.path.isfile(args.checkpoint):
        logger.error("Checkpoint not found: %s", args.checkpoint)
        sys.exit(1)

    logger.info("Loading checkpoint: %s", args.checkpoint)
    checkpoint = load_checkpoint(args.checkpoint, model=None, optimizer=None, device=device)
    config = checkpoint.get("config")
    if config is None:
        logger.error("Checkpoint has no 'config'. Cannot rebuild model.")
        sys.exit(1)

    state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
    if state_dict is None:
        state_dict = {k: v for k, v in checkpoint.items() if isinstance(v, torch.Tensor)}
    if not state_dict:
        logger.error("No model state_dict in checkpoint.")
        sys.exit(1)

    logger.info("Building model and dataloader from config...")
    grad_model, train_loader, criterion, is_gru = build_grad_model_and_loader_from_config(
        config, device, state_dict=state_dict
    )
    batch = next(iter(train_loader))
    synth_layer_types = (SynthNoiseLinear, SynthNoiseConv2d)

    metrics = {}
    experiment_mode = config.get("experiment", {}).get("mode", "baseline")
    if experiment_mode == "baseline":
        logger.warning("Checkpoint is baseline (no noise). A/C/V/S require synth_no_comp or synth_with_comp.")

    if experiment_mode in ("synth_no_comp", "synth_with_comp"):
        logger.info("Computing gradient reachability (A) and consistency (C)...")
        rc = gradient_reachability_and_consistency(
            grad_model, batch, criterion, device,
            synth_layer_types=synth_layer_types,
            is_gru=is_gru, seed=args.seed,
        )
        metrics["gradient_reachability"] = rc["gradient_reachability"]
        metrics["gradient_consistency"] = rc["gradient_consistency"]

        logger.info("Computing gradient variance domination (V)...")
        v = gradient_variance_domination(
            grad_model, batch, criterion, device,
            K=args.K, seed_base=args.seed, is_gru=is_gru,
        )
        metrics["gradient_variance_domination"] = v

        logger.info("Computing B_mean (||E[g_noisy]||^2 / ||g_clean||^2)...")
        b_mean = gradient_B_mean(
            grad_model, batch, criterion, device,
            synth_layer_types=synth_layer_types,
            K=args.K, seed_base=args.seed, is_gru=is_gru,
        )
        metrics["gradient_B_mean"] = b_mean

        logger.info("Computing perturbation stability (S)...")
        ps = perturbation_structural_stability(
            grad_model, batch, device,
            synth_layer_types=synth_layer_types,
            K=args.K, seed_base=args.seed, is_gru=is_gru,
        )
        metrics["perturbation_stability"] = ps["perturbation_stability"]

        logger.info("Computing sign_coupled_scaling P (proportion each side)...")
        p_sc = sign_coupled_scaling_P(
            grad_model, batch, device,
            synth_layer_types=synth_layer_types,
            is_gru=is_gru, seed=args.seed,
        )
        metrics["sign_coupled_P_positive"] = p_sc["sign_coupled_P_positive"]
        metrics["sign_coupled_P_negative"] = p_sc["sign_coupled_P_negative"]
        metrics["sign_coupled_P_zero"] = p_sc["sign_coupled_P_zero"]

        model_name = config.get("model_name", "")
        if model_name in ("resnet20", "vit_tiny", "gru_agnews"):
            logger.info("Computing layer-wise A, C, V, B_mean...")
            try:
                rc_lw = gradient_reachability_and_consistency_layerwise(
                    grad_model, batch, criterion, device,
                    synth_layer_types=synth_layer_types,
                    model_name=model_name, is_gru=is_gru, seed=args.seed,
                )
                metrics.update(rc_lw)
                v_lw = gradient_variance_domination_layerwise(
                    grad_model, batch, criterion, device,
                    model_name=model_name, K=args.K, seed_base=args.seed, is_gru=is_gru,
                )
                metrics.update(v_lw)
                b_lw = gradient_B_mean_layerwise(
                    grad_model, batch, criterion, device,
                    synth_layer_types=synth_layer_types,
                    model_name=model_name, K=args.K, seed_base=args.seed, is_gru=is_gru,
                )
                metrics.update(b_lw)
            except Exception as elw:
                logger.warning("Layer-wise gradient metrics failed: %s", elw)

    if not args.no_gq and experiment_mode in ("synth_no_comp", "synth_with_comp"):
        logger.info("Computing dead-zone / effective rank...")
        gq = collect_gradient_quality_metrics(
            grad_model, batch, criterion, device,
            is_gru=is_gru, seed=args.seed,
            synth_layer_types=synth_layer_types,
        )
        metrics["dead_zone_ratio_element_mean"] = gq["dead_zone_ratio_element_mean"]
        metrics["dead_zone_ratio_channel_mean"] = gq["dead_zone_ratio_channel_mean"]
        metrics["grad_top_k_energy_ratio_mean"] = gq["grad_top_k_energy_ratio_mean"]
        metrics["grad_effective_rank_mean"] = gq["grad_effective_rank_mean"]

    for k, v in metrics.items():
        if isinstance(v, float):
            logger.info("  %s = %.6g", k, v)
        else:
            logger.info("  %s = %s", k, v)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Wrote metrics to %s", args.output)

    return metrics


if __name__ == "__main__":
    main()
