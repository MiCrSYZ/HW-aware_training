"""
CLI entrypoint for the memristor neural network framework.

This module provides the main CLI interface. Run with:
    python -m src train --config configs/baseline/default.yaml
    python -m src eval --config configs/baseline/default.yaml --checkpoint model.pth
"""

import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description='Memristor Neural Network Framework')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    train_parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    train_parser.add_argument('--compensation', type=str, default=None, choices=['hat'],
                              help='Compensation method (overrides config)')
    train_parser.add_argument('--output-dir', type=str, default=None, help='Output directory (overrides config)')
    
    # Eval command
    eval_parser = subparsers.add_parser('eval', help='Evaluate a trained model')
    eval_parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    eval_parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    eval_parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                           help='Dataset split to evaluate on')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        from .train import main as train_main
        train_main(args)
    elif args.command == 'eval':
        from .eval import main as eval_main
        eval_main(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()

