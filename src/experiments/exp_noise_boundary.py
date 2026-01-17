"""
实验A: 噪声强度渐增的边界验证实验

针对各类非理想性分别绘制"噪声强度 vs 最终准确率"曲线,
预期展示可学习与不可学习两类不同的趋势。
"""

import argparse
import yaml
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List
import torch

try:
    from .run_experiment import run_experiment
except ImportError:
    from src.experiments.run_experiment import run_experiment


def create_config_with_noise_strength(
    base_config: Dict[str, Any],
    noise_type: str,
    noise_strength: float,
    enable_adc_during_training: bool = False,
    adc_training_mode: str = 'ste',
    enable_ir_drop_paper_during_training: bool = False,
) -> Dict[str, Any]:
    """
    创建带有指定噪声强度的配置。
    
    Args:
        base_config: 基础配置
        noise_type: 噪声类型 ('adc_bits', 'ir_drop_scaling', 'ir_drop_cap', 'ir_drop_beta', 'variability_sigma', 'read_noise_sigma', 'drift_alpha', 'stuck_ratio')
        noise_strength: 噪声强度值
        enable_adc_during_training: 是否在训练时启用ADC
        enable_ir_drop_paper_during_training: 是否在训练时启用paper版IR-drop
    
    Returns:
        修改后的配置
    """
    config = base_config.copy()
    
    if noise_type == 'adc_bits':
        # ADC位数: 越小噪声越大 (2-16 bits)
        config['memristor']['adc_bits'] = int(noise_strength)
        config['memristor']['enable_adc'] = True
        config['memristor']['enable_adc_during_training'] = enable_adc_during_training
        config['memristor']['adc_training_mode'] = adc_training_mode
    elif noise_type == 'ir_drop_scaling':
        # IR-drop缩放因子: 越大噪声越大 (0.0-2.0)
        config['memristor']['ir_drop_mode'] = 'paper'
        config['memristor']['ir_drop_scaling'] = float(noise_strength)
        config['memristor']['enable_ir_drop_paper_during_training'] = enable_ir_drop_paper_during_training
    elif noise_type == 'ir_drop_cap':
        # IR-drop衰减上限: 越大噪声越大 (0.0-1.0)
        config['memristor']['ir_drop_mode'] = 'crossbar'
        config['memristor']['ir_drop_cap'] = float(noise_strength)
    elif noise_type == 'ir_drop_beta':
        # IR-drop简单模式系数: 越大噪声越大 (0.0-1.0)
        config['memristor']['ir_drop_mode'] = 'simple'
        config['memristor']['ir_drop_beta'] = float(noise_strength)
    elif noise_type == 'variability_sigma':
        # 器件变异: 越大噪声越大 (0.0-0.5)
        config['memristor']['variability_sigma'] = float(noise_strength)
    elif noise_type == 'read_noise_sigma':
        # 读噪声: 越大噪声越大 (0.0-1e-5)
        config['memristor']['read_noise_sigma'] = float(noise_strength)
    elif noise_type == 'drift_alpha':
        # 电导漂移: 越大噪声越大 (0.0-1e-3)
        config['memristor']['drift_alpha'] = float(noise_strength)
    elif noise_type == 'stuck_ratio':
        # 卡位故障: 越大噪声越大 (0.0-0.5)
        config['memristor']['stuck_ratio'] = float(noise_strength)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    return config


def run_noise_sweep(
    base_config_path: str,
    noise_type: str,
    noise_strengths: List[float],
    output_dir: str,
    enable_adc_during_training: bool = False,
    adc_training_mode: str = 'ste',
    enable_ir_drop_paper_during_training: bool = False,
) -> Dict[str, Any]:
    """
    运行噪声强度扫描实验。
    
    Args:
        base_config_path: 基础配置文件路径
        noise_type: 噪声类型
        noise_strengths: 噪声强度值列表
        output_dir: 输出目录
        enable_adc_during_training: 是否在训练时启用ADC
        enable_ir_drop_paper_during_training: 是否在训练时启用paper版IR-drop
    
    Returns:
        实验结果字典
    """
    # 加载基础配置
    with open(base_config_path, 'r', encoding='utf-8') as f:
        base_config = yaml.safe_load(f)
    
    results = {
        'noise_type': noise_type,
        'noise_strengths': noise_strengths,
        'final_accuracies': [],
        'final_losses': [],
        'training_successful': [],  # 是否成功完成训练（没有NaN/崩溃）
    }
    
    for noise_strength in noise_strengths:
        print(f"\n{'='*60}")
        print(f"Running experiment: {noise_type} = {noise_strength}")
        print(f"{'='*60}\n")
        
        # 创建配置
        config = create_config_with_noise_strength(
            base_config,
            noise_type,
            noise_strength,
            enable_adc_during_training=enable_adc_during_training,
            adc_training_mode=adc_training_mode,
            enable_ir_drop_paper_during_training=enable_ir_drop_paper_during_training,
        )
        
        # 修改实验名称
        config['experiment_name'] = f"{base_config['experiment_name']}_{noise_type}_{noise_strength}"
        
        # 创建输出目录
        exp_output_dir = os.path.join(output_dir, f"{noise_type}_{noise_strength}")
        os.makedirs(exp_output_dir, exist_ok=True)
        
        try:
            # 运行实验
            exp_results = run_experiment(config, exp_output_dir)
            
            # 提取最终准确率和损失
            if 'metrics_history' in exp_results and len(exp_results['metrics_history']) > 0:
                final_metrics = exp_results['metrics_history'][-1]
                final_acc = final_metrics.get('val_acc1', 0.0)
                final_loss = final_metrics.get('val_loss', float('inf'))
                training_successful = True
            else:
                final_acc = 0.0
                final_loss = float('inf')
                training_successful = False
            
            results['final_accuracies'].append(final_acc)
            results['final_losses'].append(final_loss)
            results['training_successful'].append(training_successful)
            
            print(f"Final accuracy: {final_acc:.2f}%")
            print(f"Training successful: {training_successful}")
            
        except Exception as e:
            print(f"Error during training: {e}")
            results['final_accuracies'].append(0.0)
            results['final_losses'].append(float('inf'))
            results['training_successful'].append(False)
    
    return results


def plot_noise_boundary(results: Dict[str, Any], output_path: str):
    """
    绘制噪声强度 vs 最终准确率曲线。
    
    Args:
        results: 实验结果字典
        output_path: 输出图片路径
    """
    noise_type = results['noise_type']
    noise_strengths = results['noise_strengths']
    final_accuracies = results['final_accuracies']
    training_successful = results['training_successful']
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制成功和失败的实验点
    successful_strengths = [s for s, success in zip(noise_strengths, training_successful) if success]
    successful_accs = [a for a, success in zip(final_accuracies, training_successful) if success]
    failed_strengths = [s for s, success in zip(noise_strengths, training_successful) if not success]
    failed_accs = [a for a, success in zip(final_accuracies, training_successful) if not success]
    
    # 绘制成功实验的曲线
    if successful_strengths:
        ax.plot(successful_strengths, successful_accs, 'o-', label='Training Successful', 
                linewidth=2, markersize=8, color='blue')
    
    # 绘制失败实验的点
    if failed_strengths:
        ax.scatter(failed_strengths, failed_accs, marker='x', s=100, color='red', 
                  label='Training Failed', linewidths=2)
    
    ax.set_xlabel(f'{noise_type} (Noise Strength)', fontsize=12)
    ax.set_ylabel('Final Validation Accuracy (%)', fontsize=12)
    ax.set_title(f'Noise Boundary: {noise_type} vs Final Accuracy', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='实验A: 噪声强度渐增的边界验证实验')
    parser.add_argument('--config', type=str, required=True, help='基础配置文件路径')
    parser.add_argument('--noise_type', type=str, required=True,
                       choices=['adc_bits', 'ir_drop_scaling', 'ir_drop_cap', 'ir_drop_beta', 'variability_sigma', 
                               'read_noise_sigma', 'drift_alpha', 'stuck_ratio'],
                       help='噪声类型')
    parser.add_argument('--noise_strengths', type=float, nargs='+', required=True,
                       help='噪声强度值列表')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--enable_adc_during_training', action='store_true',
                       help='在训练时启用ADC量化')
    parser.add_argument('--adc_training_mode', type=str, default='ste',
                       choices=['ste', 'direct'],
                       help='ADC训练模式：ste（使用Straight-Through Estimator）或direct（直接量化，梯度会消失）')
    parser.add_argument('--enable_ir_drop_paper_during_training', action='store_true',
                       help='在训练时启用paper版IR-drop')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 运行噪声扫描实验
    results = run_noise_sweep(
        args.config,
        args.noise_type,
        args.noise_strengths,
        args.output_dir,
        enable_adc_during_training=args.enable_adc_during_training,
        adc_training_mode=args.adc_training_mode,
        enable_ir_drop_paper_during_training=args.enable_ir_drop_paper_during_training,
    )
    
    # 保存结果
    results_path = os.path.join(args.output_dir, f'{args.noise_type}_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {results_path}")
    
    # 绘制曲线
    plot_path = os.path.join(args.output_dir, f'{args.noise_type}_boundary.png')
    plot_noise_boundary(results, plot_path)


if __name__ == '__main__':
    main()

