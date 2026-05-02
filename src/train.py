"""
Main training script.

This script provides the CLI interface for training models with different
experiment modes (baseline, memristor_no_comp, memristor_with_comp).
"""

import argparse
import yaml
import os
import json
import hashlib
from pathlib import Path
from datetime import datetime
import logging

# Set OpenMP threads to avoid libgomp warning (only if not already set)
if 'OMP_NUM_THREADS' not in os.environ:
    # Default to number of CPU cores, but cap at 8 to avoid oversubscription
    import multiprocessing
    num_threads = min(multiprocessing.cpu_count(), 8)
    os.environ['OMP_NUM_THREADS'] = str(num_threads)

try:
    from .experiments.run_experiment import run_experiment
    from .experiments.visualize import plot_accuracy_curve
    from .utils.seeds import set_seed
except ImportError:
    from src.experiments.run_experiment import run_experiment
    from src.experiments.visualize import plot_accuracy_curve
    from src.utils.seeds import set_seed


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def _deep_update(base: dict, upd: dict) -> dict:
    """Recursively merge `upd` into `base` (returns new dict)."""
    out = dict(base)
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _canonical_suite_cfg(cfg: dict) -> dict:
    """Build a canonical config for deduplication (drop per-variant naming only)."""
    c = _deep_update({}, cfg)
    if isinstance(c.get('logging'), dict):
        c['logging'] = dict(c['logging'])
        c['logging'].pop('run_name', None)
    c.pop('experiment_name', None)
    c.pop('suite', None)
    return c


def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser(description='Train memristor-aware neural network')
        parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
        parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
        parser.add_argument('--compensation', type=str, default=None, choices=['hat'],
                           help='Compensation method (overrides config)')
        parser.add_argument('--output-dir', type=str, default=None, help='Output directory (overrides config)')
        parser.add_argument('--suite-groups', type=str, default=None,
                           help='Only run these suite groups (e.g. A or A,B or T). Groups: A/B/C are noise ablations, T is template-training strategies')
        parser.add_argument('--suite-variants', type=str, default=None,
                           help='Only run these suite variant names (exact match), e.g. A2_frozen_train_new_test or A2_frozen_train_new_test,C2_epoch_wise_resampled')
        args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override compensation method if provided
    if args.compensation:
        config['experiment']['compensation_method'] = args.compensation
    
    # Suite runner: if config contains `suite`, run multiple variants sequentially.
    suite = config.get('suite', None)
    if suite is not None:
        # Filter by explicit variant names first (exact match)
        if getattr(args, 'suite_variants', None):
            wanted_names = [n.strip() for n in args.suite_variants.split(',') if n.strip()]
            wanted_set = set(wanted_names)
            suite = [v for v in suite if v.get('name', '') in wanted_set]
            if not suite:
                raise ValueError(f"No suite variants matched --suite-variants={wanted_names}.")

        # Filter by group(s): A, B, C (from config suite_groups or CLI --suite-groups)
        suite_groups = config.get('suite_groups', None)
        if getattr(args, 'suite_groups', None):
            suite_groups = [g.strip().upper() for g in args.suite_groups.split(',') if g.strip()]
        if suite_groups:
            allowed = set(suite_groups)
            def _variant_group(v):
                name = v.get('name', '')
                group = v.get('group', None)
                if group is not None:
                    return str(group).upper()
                if name.startswith('A1_') or name.startswith('A2_') or name.startswith('A3_'):
                    return 'A'
                if name.startswith('B1_') or name.startswith('B2_') or name.startswith('B3_'):
                    return 'B'
                if name.startswith('C1_') or name.startswith('C2_') or name.startswith('C3_'):
                    return 'C'
                return None
            suite = [v for v in suite if _variant_group(v) in allowed]
            if not suite:
                raise ValueError(f"No suite variants in groups {allowed}. Check suite_groups or --suite-groups.")

        dataset_name = config.get('dataset', 'cifar10').lower()
        seed = config.get('seed', 42)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = config.get('experiment_name', os.path.splitext(os.path.basename(args.config))[0])
        if args.output_dir:
            base_output_dir = args.output_dir
        else:
            if dataset_name == 'cifar100':
                config_name = os.path.splitext(os.path.basename(args.config))[0]
                base_output_dir = os.path.join('output', 'CIFAR-100', config_name, f'seed_{seed}', timestamp)
            else:
                base_output_dir = os.path.join('outputs', base_name, f'seed_{seed}', timestamp)
        os.makedirs(base_output_dir, exist_ok=True)

        # Set up logging once for the suite
        try:
            from .utils.logger import setup_logger
        except ImportError:
            from utils.logger import setup_logger
        logger = setup_logger(os.path.join(base_output_dir, 'logs'), name='train')
        logger.info(f"Starting SUITE with config: {args.config}")
        logger.info(f"Base output directory: {base_output_dir}")
        logger.info(f"Suite variants: {len(suite)}")
        dedup_enabled = bool(config.get('suite_deduplicate', True))
        logger.info(f"Suite deduplicate: {dedup_enabled}")
        seen_fingerprints = {}

        for i, variant in enumerate(suite):
            v_name = variant.get('name', f'variant_{i:02d}')
            v_overrides = variant.get('overrides', {})
            v_cfg = _deep_update(config, v_overrides)
            # Keep suite key out of child config to avoid nesting recursion
            v_cfg.pop('suite', None)
            v_cfg['experiment_name'] = f"{base_name}__{v_name}"
            if 'logging' not in v_cfg:
                v_cfg['logging'] = {}
            v_cfg['logging']['run_name'] = v_cfg['experiment_name']

            v_out = os.path.join(base_output_dir, v_name)
            # Avoid mixing artifacts across repeated runs in the same output-dir.
            # If variant dir already exists and is non-empty, create a timestamped rerun dir.
            if os.path.exists(v_out):
                try:
                    has_existing = len(os.listdir(v_out)) > 0
                except OSError:
                    has_existing = False
                if has_existing:
                    rerun_tag = datetime.now().strftime('%Y%m%d_%H%M%S')
                    v_out = os.path.join(base_output_dir, f"{v_name}__rerun_{rerun_tag}")
            os.makedirs(v_out, exist_ok=True)

            # Save per-variant config
            with open(os.path.join(v_out, 'config.yaml'), 'w') as f:
                yaml.dump(v_cfg, f)

            # Deduplicate variants with identical effective config
            if dedup_enabled:
                canonical = _canonical_suite_cfg(v_cfg)
                fp = hashlib.sha256(
                    json.dumps(canonical, sort_keys=True, default=str).encode('utf-8')
                ).hexdigest()
                if fp in seen_fingerprints:
                    src = seen_fingerprints[fp]
                    logger.info(f"[{i+1}/{len(suite)}] Skipping duplicate variant: {v_name} (same as {src})")
                    with open(os.path.join(v_out, 'DUPLICATE_OF.txt'), 'w', encoding='utf-8') as f:
                        f.write(src)
                    continue
                seen_fingerprints[fp] = v_name

            logger.info(f"[{i+1}/{len(suite)}] Running variant: {v_name}")
            experiment_mode = v_cfg.get('experiment', {}).get('mode', 'baseline')
            if experiment_mode in ('synth_no_comp', 'synth_with_comp'):
                raise ValueError(
                    f"Unsupported experiment mode on memristor branch: {experiment_mode}"
                )
            run_experiment(v_cfg, v_out, resume=args.resume)
        return

    # Single-run path
    if args.output_dir:
        output_dir = args.output_dir
    else:
        dataset_name = config.get('dataset', 'cifar10').lower()
        seed = config.get('seed', 42)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if dataset_name == 'cifar100':
            config_name = os.path.splitext(os.path.basename(args.config))[0]
            output_dir = os.path.join('output', 'CIFAR-100', config_name, f'seed_{seed}', timestamp)
        else:
            experiment_name = config.get('experiment_name', 'experiment')
            output_dir = os.path.join('outputs', experiment_name, f'seed_{seed}', timestamp)

    os.makedirs(output_dir, exist_ok=True)

    config_save_path = os.path.join(output_dir, 'config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f)
    
    # Set up logging
    try:
        from .utils.logger import setup_logger
    except ImportError:
        from utils.logger import setup_logger
    logger = setup_logger(os.path.join(output_dir, 'logs'), name='train')
    
    logger.info(f"Starting training with config: {args.config}")
    logger.info(f"Output directory: {output_dir}")
    experiment_mode = config.get('experiment', {}).get('mode', 'baseline')
    logger.info(f"Experiment mode: {experiment_mode}")

    if experiment_mode in ('synth_no_comp', 'synth_with_comp'):
        raise ValueError(
            f"Unsupported experiment mode on memristor branch: {experiment_mode}"
        )
    results = run_experiment(config, output_dir, resume=args.resume)
    
    # Generate plots
    import pandas as pd
    metrics_path = os.path.join(output_dir, 'metrics.csv')
    if os.path.exists(metrics_path):
        metrics_df = pd.read_csv(metrics_path)
        plot_accuracy_curve(
            metrics_df,
            os.path.join(output_dir, 'accuracy_curve.png'),
            title=f"Training Curves - {config.get('experiment_name', 'Experiment')}",
        )
        logger.info("Generated accuracy curve plot")
    
    logger.info(f"Training completed. Best val acc: {results['best_val_acc']:.2f}%")
    logger.info(f"Test acc: {results['test_acc']:.2f}%")


if __name__ == '__main__':
    main()
