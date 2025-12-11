"""
Main training script.

This script provides the CLI interface for training models with different
experiment modes (baseline, memristor_no_comp, memristor_with_comp).
"""

import argparse
import yaml
import os
from pathlib import Path
from datetime import datetime
import logging

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


def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser(description='Train memristor-aware neural network')
        parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
        parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
        parser.add_argument('--compensation', type=str, default=None, choices=['hat', 'learned_mapping'],
                           help='Compensation method (overrides config)')
        parser.add_argument('--output-dir', type=str, default=None, help='Output directory (overrides config)')
        args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override compensation method if provided
    if args.compensation:
        config['experiment']['compensation_method'] = args.compensation
    
    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        experiment_name = config.get('experiment_name', 'experiment')
        seed = config.get('seed', 42)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join('outputs', experiment_name, f'seed_{seed}', timestamp)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config to output directory
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
    logger.info(f"Experiment mode: {config['experiment']['mode']}")
    
    # Run experiment
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

