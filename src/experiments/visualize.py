"""
Visualization utilities for experiment results.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import os


def plot_accuracy_curve(
    metrics_df: pd.DataFrame,
    output_path: str,
    title: str = 'Training and Validation Accuracy',
) -> None:
    """
    Plot training and validation accuracy curves.
    
    Args:
        metrics_df: DataFrame with columns 'epoch', 'train_acc1', 'val_acc1'
        output_path: Path to save plot
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = metrics_df['epoch'].values
    train_acc = metrics_df['train_acc1'].values
    val_acc = metrics_df['val_acc1'].values
    
    ax.plot(epochs, train_acc, label='Train Accuracy', marker='o', markersize=3)
    ax.plot(epochs, val_acc, label='Validation Accuracy', marker='s', markersize=3)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_parameter_sweep(
    results: List[Dict],
    param_name: str,
    output_path: str,
    title: Optional[str] = None,
) -> None:
    """
    Plot accuracy vs parameter sweep (e.g., variability, drift).
    
    Args:
        results: List of dictionaries with 'param_value' and 'accuracy' keys
        param_name: Name of parameter being swept
        output_path: Path to save plot
        title: Plot title (auto-generated if None)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    param_values = [r['param_value'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    
    ax.plot(param_values, accuracies, marker='o', markersize=6, linewidth=2)
    
    ax.set_xlabel(param_name)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(title or f'Accuracy vs {param_name}')
    ax.grid(True, alpha=0.3)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_energy_vs_accuracy(
    results: List[Dict],
    output_path: str,
    title: str = 'Energy vs Accuracy Trade-off',
) -> None:
    """
    Plot energy consumption vs accuracy trade-off.
    
    Args:
        results: List of dictionaries with 'energy' and 'accuracy' keys
        output_path: Path to save plot
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    energies = [r['energy'] * 1e9 for r in results]  # Convert to nJ
    accuracies = [r['accuracy'] for r in results]
    
    ax.scatter(energies, accuracies, s=100, alpha=0.6)
    
    ax.set_xlabel('Energy (nJ)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


