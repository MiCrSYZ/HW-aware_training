"""
验证 “BN 放大扰动” 与 “优化目标一致性” 的两种观点。

用法:
  # 1. BN 失配统计（no_comp eval，ResNet）
  python -m src.experiments.verify_drift_hypotheses --mode bn_mismatch \\
    --config configs/synth/resnet20_synth_no_comp.yaml --drift_frozen true \\
    --checkpoint outputs/.../model_best.pth --num_batches 50

  # 2. 同一 batch 两次前向的 loss 差（comp，frozen vs resampled）
  python -m src.experiments.verify_drift_hypotheses --mode loss_variance \\
    --config configs/synth/resnet20_synth_comp.yaml --drift_frozen true \\
    --num_batches 100
"""

import argparse
import copy
import os
import sys
import yaml
import torch
import torch.nn as nn
import numpy as np

# 项目根目录
if __name__ == "__main__":
    _ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

from src.models.model_zoo import get_model, wrap_model_with_synth_noise
from src.memristor.synth_noise_wrappers import SynthNoiseConfig
from src.data.dataset import get_dataloaders
from src.utils.checkpoint import load_checkpoint


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_synth_config_from_config(config: dict, drift_frozen: bool) -> SynthNoiseConfig:
    """从 config 构建 SynthNoiseConfig（与 run_experiment_synth 对齐），并覆盖 drift_frozen."""
    s = config.get("synth_noise", {})
    noise_type = s.get("noise_type", "frozen_additive_drift")
    noise_type_map = {
        "full_variability": "iid_multiplicative",
        "cond1_variance_bounded": "heavy_tail",
        "cond2_gradient_unbiased": "input_dependent",
        "cond3_adc_direct": "gradient_degenerate",
    }
    noise_type = noise_type_map.get(noise_type, noise_type)

    def _float(key, default):
        v = s.get(key)
        return float(v) if v is not None else default

    def _bool(key, default):
        v = s.get(key)
        return bool(v) if v is not None else default

    return SynthNoiseConfig(
        noise_type=noise_type,
        variability_sigma=_float("variability_sigma", 0.05),
        heavy_tail_alpha=_float("heavy_tail_alpha", _float("cond1_alpha", 0.1)),
        heavy_tail_nu=_float("heavy_tail_nu", _float("cond1_nu", 2.0)),
        input_dependent_alpha=_float("input_dependent_alpha", _float("cond2_alpha", 0.1)),
        decoupled_consistent_sigma=_float("decoupled_consistent_sigma", 0.1),
        decoupled_inconsistent_sigma=_float("decoupled_inconsistent_sigma", 0.1),
        coupled_consistent_alpha=_float("coupled_consistent_alpha", 0.1),
        coupled_inconsistent_alpha=_float("coupled_inconsistent_alpha", 0.1),
        drift_beta=_float("drift_beta", 0.3),
        drift_use_norm=_bool("drift_use_norm", False),
        drift_frozen=drift_frozen,
        sign_scale_alpha=_float("sign_scale_alpha", 0.5),
        rank_k=int(s.get("rank_k", 4)),
        rank_fill_sigma=_float("rank_fill_sigma", 0.0),
        rank_resample=_bool("rank_resample", False),
        clip_c=_float("clip_c", 1.0),
        clip_dither=_bool("clip_dither", False),
        seed=config.get("seed"),
        compensation_in_backward=_bool("compensation_in_backward", True),
        backward_corruption_at=s.get("backward_corruption_at"),
    )


# -----------------------------------------------------------------------------
# 1. BN 失配统计（验证 “BN 放大扰动”）
# -----------------------------------------------------------------------------

def _collect_bn_mismatch_stats_impl(model: nn.Module, device: torch.device, dataloader, num_batches: int):
    """
    在 model 的 BN 层上注册 hook，跑 num_batches 个 batch，记录每个 BN 的
    batch_mean - running_mean（及 L2 范数）。
    model 应为已包装的 synth_noise_model（带 drift），eval 模式。
    """
    results = {}  # name -> list of (diff_norm, diff_mean_tensor)
    hooks = []

    def make_hook(name):
        def hook(module, input, output):
            x = input[0]
            if not isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                return
            # 当前 batch 在 (N, H, W) 上求均值，得到 (C,) 与 running_mean 一致
            dim = [d for d in range(x.dim()) if d != 1]  # 对 BN2d: (N,C,H,W) -> mean over N,H,W
            batch_mean = x.mean(dim=dim)
            rm = module.running_mean
            if rm is None:
                return
            diff = (batch_mean.detach() - rm.detach()).float()
            diff_norm = diff.norm().item()
            if name not in results:
                results[name] = []
            results[name].append((diff_norm, diff.detach().cpu()))

        return hook

    for name, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            h = m.register_forward_hook(make_hook(name))
            hooks.append(h)

    if not results:
        for h in hooks:
            h.remove()
        return None

    model.eval()
    n = 0
    with torch.no_grad():
        for batch in dataloader:
            if n >= num_batches:
                break
            x = batch[0].to(device)
            _ = model(x)
            n += 1

    for h in hooks:
        h.remove()

    # 汇总：每层 (batch_mean - running_mean) 的均值向量、范数均值/标准差
    summary = {}
    for name, lst in results.items():
        norms = [t[0] for t in lst]
        diffs = [t[1] for t in lst]
        mean_diff = torch.stack(diffs).mean(dim=0)
        summary[name] = {
            "diff_norm_mean": float(np.mean(norms)),
            "diff_norm_std": float(np.std(norms)),
            "mean_diff_l2": float(mean_diff.norm()),
            "mean_diff_mean_over_ch": float(mean_diff.abs().mean()),
        }
    return summary


def collect_bn_mismatch_stats(
    config_path: str,
    checkpoint_path: str,
    drift_frozen: bool,
    num_batches: int = 50,
    device: str = "cuda",
):
    """
    验证 “BN 放大扰动”：no_comp 下用 synth_noise_model（带 drift）做 eval，
    统计各 BN 层 (batch_mean - running_mean) 的范数与方向。
    """
    config = _load_config(config_path)
    if config["model_name"] != "resnet20":
        print("bn_mismatch 当前仅支持 resnet20（含 BN）。跳过。")
        return

    # 数据
    dataset_name = config.get("dataset", "cifar10")
    data_root = config.get("data_root", "./datasets/cifar-10")
    batch_size = config.get("batch_size", 128)
    num_workers = config.get("num_workers", 4)
    val_split = config.get("val_split", 0.1)
    seed = config.get("seed", 42)
    if dataset_name == "agnews":
        train_loader, val_loader, test_loader, vocab = get_dataloaders(
            dataset_name, data_root, batch_size, num_workers, val_split, seed
        )
    else:
        train_loader, val_loader, test_loader = get_dataloaders(
            dataset_name, data_root, batch_size, num_workers, val_split, seed
        )

    num_classes = config.get("num_classes", 10)
    base_model = get_model(
        name=config["model_name"],
        num_classes=num_classes,
        in_channels=3,
    )
    load_checkpoint(checkpoint_path, model=base_model, device=device)
    base_model = base_model.to(device)

    synth_config = _build_synth_config_from_config(config, drift_frozen=drift_frozen)
    noise_injection = config.get("synth_noise", {}).get("noise_injection")
    base_for_noisy = copy.deepcopy(base_model)
    synth_noise_model = wrap_model_with_synth_noise(
        base_for_noisy, synth_config, noise_config=noise_injection
    )
    synth_noise_model.load_state_dict(base_model.state_dict(), strict=True)
    synth_noise_model = synth_noise_model.to(device)
    synth_noise_model.eval()

    summary = _collect_bn_mismatch_stats_impl(
        synth_noise_model, torch.device(device), val_loader, num_batches
    )
    if summary is None:
        print("未找到 BN 层（或非 ResNet）。")
        return

    print(f"\n[BN mismatch] drift_frozen={drift_frozen}, num_batches={num_batches}")
    print("layer | diff_norm_mean | diff_norm_std | mean_diff_l2 | mean_diff_mean_over_ch")
    for name, s in summary.items():
        print(f"{name} | {s['diff_norm_mean']:.6f} | {s['diff_norm_std']:.6f} | {s['mean_diff_l2']:.6f} | {s['mean_diff_mean_over_ch']:.6f}")
    return summary


# -----------------------------------------------------------------------------
# 2. 同一 batch 两次前向的 loss 差（验证 “优化目标一致性”）
# -----------------------------------------------------------------------------

def measure_loss_variance_same_batch(
    config_path: str,
    drift_frozen: bool,
    num_batches: int = 100,
    device: str = "cuda",
    checkpoint_path: str = None,
):
    """
    验证 “优化目标一致性”：comp 下对同一 batch 做两次前向（不同 seed），
    frozen 时两次 loss 应相同，resampled 时应有差异。
    """
    config = _load_config(config_path)
    s = config.get("synth_noise", {})
    if s.get("noise_type") != "frozen_additive_drift":
        config = copy.deepcopy(config)
        config["synth_noise"] = dict(config["synth_noise"])
        config["synth_noise"]["noise_type"] = "frozen_additive_drift"

    dataset_name = config.get("dataset", "cifar10")
    data_root = config.get("data_root", "./datasets/cifar-10")
    batch_size = config.get("batch_size", 128)
    num_workers = config.get("num_workers", 4)
    val_split = config.get("val_split", 0.1)
    seed = config.get("seed", 42)
    if dataset_name == "agnews":
        train_loader, val_loader, test_loader, vocab = get_dataloaders(
            dataset_name, data_root, batch_size, num_workers, val_split, seed
        )
    else:
        train_loader, val_loader, test_loader = get_dataloaders(
            dataset_name, data_root, batch_size, num_workers, val_split, seed
        )

    num_classes = config.get("num_classes", 10)
    model_kwargs = {}
    if config["model_name"] == "vit_tiny":
        model_kwargs = {
            "patch_size": config.get("patch_size", 4),
            "embed_dim": config.get("embed_dim", 192),
            "depth": config.get("depth", 6),
            "num_heads": config.get("num_heads", 3),
            "mlp_ratio": config.get("mlp_ratio", 4.0),
            "qkv_bias": config.get("qkv_bias", False),
        }

    base_model = get_model(
        name=config["model_name"],
        num_classes=num_classes,
        in_channels=3,
        **model_kwargs
    )
    synth_config = _build_synth_config_from_config(config, drift_frozen=drift_frozen)
    noise_injection = config.get("synth_noise", {}).get("noise_injection")
    synth_noise_model = wrap_model_with_synth_noise(
        base_model, synth_config, noise_config=noise_injection
    )
    if checkpoint_path and os.path.isfile(checkpoint_path):
        load_checkpoint(checkpoint_path, model=synth_noise_model, device=device)
    synth_noise_model = synth_noise_model.to(device)
    synth_noise_model.eval()

    criterion = nn.CrossEntropyLoss()
    loss_diffs = []
    dev = torch.device(device)

    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            if i >= num_batches:
                break
            x, y = batch[0].to(dev), batch[1].to(dev)
            # 第一次前向（seed = base）
            out1 = synth_noise_model(x, seed=seed + i * 10000)
            L1 = criterion(out1, y).item()
            # 第二次前向（不同 seed → resampled 会得到不同 d）
            out2 = synth_noise_model(x, seed=seed + i * 10000 + 9999)
            L2 = criterion(out2, y).item()
            loss_diffs.append(abs(L1 - L2))

    loss_diffs = np.array(loss_diffs)
    print(f"\n[Loss variance same batch] drift_frozen={drift_frozen}, num_batches={num_batches}")
    print(f"  |L1-L2|: mean={loss_diffs.mean():.6f}, std={loss_diffs.std():.6f}, max={loss_diffs.max():.6f}")
    if drift_frozen:
        print("  (frozen: 预期 |L1-L2|≈0，仅数值误差)")
    else:
        print("  (resampled: 预期 |L1-L2|>0，不同 seed 导致不同 d → 不同 loss)")
    return {"mean": float(loss_diffs.mean()), "std": float(loss_diffs.std()), "max": float(loss_diffs.max())}


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Verify BN amplification and optimization consistency hypotheses")
    parser.add_argument("--mode", choices=["bn_mismatch", "loss_variance"], required=True)
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML")
    parser.add_argument("--drift_frozen", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--checkpoint", type=str, default=None, help="For bn_mismatch: no_comp checkpoint; for loss_variance: optional")
    parser.add_argument("--num_batches", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    drift_frozen = args.drift_frozen.lower() == "true"

    if args.mode == "bn_mismatch":
        if not args.checkpoint:
            print("bn_mismatch 需要 --checkpoint（no_comp 训练得到的 checkpoint）")
            return 1
        collect_bn_mismatch_stats(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            drift_frozen=drift_frozen,
            num_batches=args.num_batches,
            device=args.device,
        )
    else:
        measure_loss_variance_same_batch(
            config_path=args.config,
            drift_frozen=drift_frozen,
            num_batches=args.num_batches,
            device=args.device,
            checkpoint_path=args.checkpoint,
        )
    return 0


if __name__ == "__main__":
    exit(main())
