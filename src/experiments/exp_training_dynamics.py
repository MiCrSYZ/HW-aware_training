"""
实验B: 训练动态的对比诊断实验

对比可学习噪声和不可学习噪声下的训练动态:
- 梯度范数与梯度方差随时间的变化
- 训练损失曲线和验证精度曲线
- 权重更新轨迹的可视化（可选：PCA降维）
"""

import argparse
import yaml
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

try:
    from .run_experiment import run_experiment
except ImportError:
    from src.experiments.run_experiment import run_experiment


def extract_weight_trajectory(checkpoint_dir: str, num_epochs: int) -> List[np.ndarray]:
    """
    从检查点目录提取权重轨迹。
    
    Args:
        checkpoint_dir: 检查点目录
        num_epochs: 总epoch数
    
    Returns:
        权重轨迹列表，每个元素是一个epoch的权重向量（展平后）
    """
    weight_trajectory = []
    
    for epoch in range(0, num_epochs + 1):
        checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch}.pth')
        if not os.path.exists(checkpoint_path):
            # 尝试其他可能的文件名
            if epoch == 0:
                checkpoint_path = os.path.join(checkpoint_dir, 'model_initial.pth')
            elif epoch == num_epochs:
                checkpoint_path = os.path.join(checkpoint_dir, 'model_final.pth')
            else:
                continue
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # 提取所有权重并展平
            weights = []
            for key, value in state_dict.items():
                if 'weight' in key and value.numel() > 0:
                    weights.append(value.flatten().cpu().numpy())
            
            if weights:
                weight_vector = np.concatenate(weights)
                weight_trajectory.append(weight_vector)
    
    return weight_trajectory


def visualize_weight_trajectory_pca(
    weight_trajectory: List[np.ndarray],
    output_path: str,
    title: str = "Weight Trajectory (PCA)",
):
    """
    使用PCA将权重轨迹降维到2D并可视化。
    
    Args:
        weight_trajectory: 权重轨迹列表
        output_path: 输出图片路径
        title: 图片标题
    """
    if len(weight_trajectory) < 2:
        print(f"Warning: Not enough checkpoints for PCA visualization. Skipping.")
        return
    
    # 转换为numpy数组
    weights_matrix = np.array(weight_trajectory)  # [num_epochs, weight_dim]
    
    # 使用PCA降维到2D
    pca = PCA(n_components=2)
    weights_2d = pca.fit_transform(weights_matrix)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制轨迹
    ax.plot(weights_2d[:, 0], weights_2d[:, 1], 'o-', linewidth=2, markersize=8, alpha=0.7)
    
    # 标记起点和终点
    ax.scatter(weights_2d[0, 0], weights_2d[0, 1], s=200, marker='s', 
              color='green', label='Start', zorder=5, edgecolors='black', linewidths=2)
    ax.scatter(weights_2d[-1, 0], weights_2d[-1, 1], s=200, marker='*', 
              color='red', label='End', zorder=5, edgecolors='black', linewidths=2)
    
    # 添加箭头指示方向
    for i in range(len(weights_2d) - 1):
        dx = weights_2d[i+1, 0] - weights_2d[i, 0]
        dy = weights_2d[i+1, 1] - weights_2d[i, 1]
        ax.arrow(weights_2d[i, 0], weights_2d[i, 1], dx*0.8, dy*0.8,
                head_width=0.01, head_length=0.01, fc='blue', ec='blue', alpha=0.3)
    
    ax.set_xlabel(f'PC1 (explained variance: {pca.explained_variance_ratio_[0]:.2%})', fontsize=12)
    ax.set_ylabel(f'PC2 (explained variance: {pca.explained_variance_ratio_[1]:.2%})', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"PCA visualization saved to: {output_path}")
    plt.close()


def plot_training_dynamics(
    results: Dict[str, Any],
    output_dir: str,
    noise_name: str,
):
    """
    绘制训练动态曲线。
    
    Args:
        results: 实验结果字典（包含metrics_history）
        output_dir: 输出目录
        noise_name: 噪声名称（用于文件名）
    """
    if 'metrics_history' not in results or len(results['metrics_history']) == 0:
        print(f"Warning: No metrics history found. Skipping plots.")
        return
    
    metrics_history = results['metrics_history']
    epochs = [m['epoch'] for m in metrics_history]
    
    # 1. 损失曲线
    train_losses = [m.get('train_loss', 0.0) for m in metrics_history]
    val_losses = [m.get('val_loss', 0.0) for m in metrics_history]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(epochs, train_losses, 'o-', label='Train Loss', linewidth=2, markersize=4)
    ax1.plot(epochs, val_losses, 's-', label='Val Loss', linewidth=2, markersize=4)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. 准确率曲线
    train_accs = [m.get('train_acc1', 0.0) for m in metrics_history]
    val_accs = [m.get('val_acc1', 0.0) for m in metrics_history]
    
    ax2.plot(epochs, train_accs, 'o-', label='Train Acc', linewidth=2, markersize=4)
    ax2.plot(epochs, val_accs, 's-', label='Val Acc', linewidth=2, markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'{noise_name}_loss_acc_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Loss/Accuracy curves saved to: {plot_path}")
    plt.close()
    
    # 3. 梯度范数和方差
    if any('grad_norm' in m for m in metrics_history):
        grad_norms = [m.get('grad_norm', 0.0) for m in metrics_history]
        grad_norm_stds = [m.get('grad_norm_std', 0.0) for m in metrics_history]
        grad_vars = [m.get('grad_var', 0.0) for m in metrics_history]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # 梯度范数
        ax1.plot(epochs, grad_norms, 'o-', label='Grad Norm (mean)', linewidth=2, markersize=4)
        if any(std > 0 for std in grad_norm_stds):
            ax1.fill_between(epochs, 
                            [n - s for n, s in zip(grad_norms, grad_norm_stds)],
                            [n + s for n, s in zip(grad_norms, grad_norm_stds)],
                            alpha=0.3, label='Grad Norm (std)')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Gradient Norm', fontsize=12)
        ax1.set_title('Gradient Norm Over Time', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')  # 使用对数刻度，因为梯度范数可能变化很大
        
        # 梯度方差
        ax2.plot(epochs, grad_vars, 'o-', label='Grad Variance', linewidth=2, markersize=4)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Gradient Variance', fontsize=12)
        ax2.set_title('Gradient Variance Over Time', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'{noise_name}_gradient_stats.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Gradient statistics saved to: {plot_path}")
        plt.close()


def run_training_dynamics_experiment(
    base_config_path: str,
    noise_config: Dict[str, Any],
    output_dir: str,
    save_checkpoints: bool = True,
    extract_weights: bool = True,
) -> Dict[str, Any]:
    """
    运行训练动态诊断实验。
    
    Args:
        base_config_path: 基础配置文件路径
        noise_config: 噪声配置字典（用于修改基础配置）
        output_dir: 输出目录
        save_checkpoints: 是否保存每个epoch的检查点（用于权重轨迹分析）
        extract_weights: 是否提取权重轨迹
    
    Returns:
        实验结果字典
    """
    # 加载基础配置
    with open(base_config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 应用噪声配置
    for key, value in noise_config.items():
        if key.startswith('memristor.'):
            memristor_key = key.replace('memristor.', '')
            config['memristor'][memristor_key] = value
        else:
            config[key] = value
    
    # 确保保存检查点
    if save_checkpoints:
        config['save_interval'] = 1  # 每个epoch都保存
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 运行实验
    print(f"\n{'='*60}")
    print(f"Running training dynamics experiment")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")
    
    try:
        results = run_experiment(config, output_dir)
        
        # 提取权重轨迹（如果启用）
        weight_trajectory = None
        if extract_weights and save_checkpoints:
            try:
                weight_trajectory = extract_weight_trajectory(
                    output_dir,
                    config.get('epochs', 100)
                )
                if weight_trajectory:
                    results['weight_trajectory_length'] = len(weight_trajectory)
                    print(f"Extracted weight trajectory with {len(weight_trajectory)} checkpoints")
            except Exception as e:
                print(f"Warning: Failed to extract weight trajectory: {e}")
        
        return results, weight_trajectory
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def main():
    parser = argparse.ArgumentParser(description='实验B: 训练动态的对比诊断实验')
    parser.add_argument('--config', type=str, required=True, help='基础配置文件路径')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--noise_name', type=str, required=True, help='噪声名称（用于标识实验）')
    parser.add_argument('--enable_adc_during_training', action='store_true',
                       help='在训练时启用ADC量化')
    parser.add_argument('--adc_training_mode', type=str, default='ste',
                       choices=['ste', 'direct'],
                       help='ADC训练模式：ste（使用Straight-Through Estimator）或direct（直接量化，梯度会消失）')
    parser.add_argument('--enable_ir_drop_paper_during_training', action='store_true',
                       help='在训练时启用paper版IR-drop')
    parser.add_argument('--adc_bits', type=int, default=None, help='ADC位数（如果启用ADC）')
    parser.add_argument('--ir_drop_scaling', type=float, default=None, help='IR-drop缩放因子（如果启用IR-drop）')
    parser.add_argument('--save_checkpoints', action='store_true', default=True,
                       help='保存每个epoch的检查点（用于权重轨迹分析）')
    parser.add_argument('--extract_weights', action='store_true', default=True,
                       help='提取权重轨迹（用于PCA可视化）')
    
    args = parser.parse_args()
    
    # 构建噪声配置
    noise_config = {}
    if args.enable_adc_during_training:
        noise_config['memristor.enable_adc'] = True
        noise_config['memristor.enable_adc_during_training'] = True
        noise_config['memristor.adc_training_mode'] = args.adc_training_mode
        if args.adc_bits is not None:
            noise_config['memristor.adc_bits'] = args.adc_bits
    
    if args.enable_ir_drop_paper_during_training:
        noise_config['memristor.ir_drop_mode'] = 'paper'
        noise_config['memristor.enable_ir_drop_paper_during_training'] = True
        if args.ir_drop_scaling is not None:
            noise_config['memristor.ir_drop_scaling'] = args.ir_drop_scaling
    
    # 运行实验
    results, weight_trajectory = run_training_dynamics_experiment(
        args.config,
        noise_config,
        args.output_dir,
        save_checkpoints=args.save_checkpoints,
        extract_weights=args.extract_weights,
    )
    
    if results is None:
        print("Experiment failed!")
        return
    
    # 保存结果
    results_path = os.path.join(args.output_dir, f'{args.noise_name}_results.json')
    # 移除weight_trajectory（如果存在）因为太大，单独保存
    results_to_save = {k: v for k, v in results.items() if k != 'weight_trajectory'}
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_to_save, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to: {results_path}")
    
    # 绘制训练动态曲线
    plot_training_dynamics(results, args.output_dir, args.noise_name)
    
    # 可视化权重轨迹（如果提取成功）
    if weight_trajectory and len(weight_trajectory) >= 2:
        pca_path = os.path.join(args.output_dir, f'{args.noise_name}_weight_trajectory_pca.png')
        visualize_weight_trajectory_pca(
            weight_trajectory,
            pca_path,
            title=f'Weight Trajectory (PCA) - {args.noise_name}',
        )


if __name__ == '__main__':
    main()

