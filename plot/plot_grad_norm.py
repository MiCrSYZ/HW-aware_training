#!/usr/bin/env python3
"""
绘制梯度范数图表
"""

import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import glob
import re

# 设置字体 - 使用Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 20
plt.rcParams['axes.unicode_minus'] = False

# 使用低饱和度的配色方案（与之前保持一致）
colors = [
    '#2E86AB',  # 蓝色
    '#A23B72',  # 紫红色
    '#F18F01',  # 橙色
    '#C73E1D'   # 红色
]

base_dir = Path("output")

# 噪声类型列表
noise_types = [
    'noise_boundary_adc_direct',
    'noise_boundary_adc_ste',
    'noise_boundary_drift_alpha',
    'noise_boundary_ir_drop',
    'noise_boundary_ir_drop_beta',
    'noise_boundary_read_noise_sigma',
    'noise_boundary_stuck_ratio',
    'noise_boundary_variability_sigma'
]

# 噪声类型的中文名称（用于标题）
noise_names = {
    'noise_boundary_adc_direct': 'ADC Direct',
    'noise_boundary_adc_ste': 'ADC STE',
    'noise_boundary_drift_alpha': 'Drift Alpha',
    'noise_boundary_ir_drop': 'IR Drop (Paper)',
    'noise_boundary_ir_drop_beta': 'IR Drop (Crossbar)',
    'noise_boundary_read_noise_sigma': 'Read Noise Sigma',
    'noise_boundary_stuck_ratio': 'Stuck Ratio',
    'noise_boundary_variability_sigma': 'Variability Sigma'
}

def extract_noise_strength(folder_name, noise_type):
    """从文件夹名称中提取噪声强度值"""
    # 提取数值部分
    match = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', folder_name)
    if match:
        value_str = match.group()
        try:
            # 尝试转换为浮点数
            if 'e' in value_str.lower():
                return float(value_str)
            else:
                return float(value_str)
        except:
            return value_str
    return folder_name

def read_metrics(csv_path):
    """读取metrics.csv文件"""
    epochs = []
    grad_norms = []
    grad_norm_stds = []
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs.append(int(row['epoch']))
                grad_norms.append(float(row['grad_norm']))
                grad_norm_stds.append(float(row['grad_norm_std']))
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None, None, None
    
    return np.array(epochs), np.array(grad_norms), np.array(grad_norm_stds)

def format_noise_strength_label(value, noise_type):
    """格式化噪声强度标签"""
    if noise_type == 'noise_boundary_read_noise_sigma':
        # 使用科学计数法格式
        if value == 0:
            return '0'
        exp = int(np.floor(np.log10(abs(value))))
        coeff = value / (10 ** exp)
        if abs(coeff - round(coeff)) < 1e-10:
            coeff_str = str(int(round(coeff)))
        else:
            coeff_str = f'{coeff:.1f}'.rstrip('0').rstrip('.')
        return f'${coeff_str} \\times 10^{{{exp}}}$'
    elif noise_type == 'noise_boundary_drift_alpha':
        # 使用科学计数法格式
        if value == 0:
            return '0'
        exp = int(np.floor(np.log10(abs(value))))
        coeff = value / (10 ** exp)
        if abs(coeff - round(coeff)) < 1e-10:
            coeff_str = str(int(round(coeff)))
        else:
            coeff_str = f'{coeff:.1f}'.rstrip('0').rstrip('.')
        return f'${coeff_str} \\times 10^{{{exp}}}$'
    else:
        # 其他情况：保留适当的小数位
        if value == int(value):
            return str(int(value))
        elif value < 0.01:
            return f'{value:.4f}'
        else:
            return f'{value:.2f}'

# 为每个噪声类型生成图表
for noise_type in noise_types:
    noise_dir = base_dir / noise_type
    
    if not noise_dir.exists():
        print(f"Warning: {noise_dir} does not exist")
        continue
    
    # 找到所有子文件夹（排除结果文件）
    subdirs = [d for d in noise_dir.iterdir() if d.is_dir()]
    
    if len(subdirs) == 0:
        print(f"Warning: No subdirectories found in {noise_dir}")
        continue
    
    # 读取每个子文件夹的数据
    data_list = []
    for subdir in subdirs:
        metrics_file = subdir / 'metrics.csv'
        if not metrics_file.exists():
            continue
        
        epochs, grad_norms, grad_norm_stds = read_metrics(metrics_file)
        if epochs is None:
            continue
        
        # 提取噪声强度
        noise_strength = extract_noise_strength(subdir.name, noise_type)
        
        data_list.append({
            'noise_strength': noise_strength,
            'epochs': epochs,
            'grad_norms': grad_norms,
            'grad_norm_stds': grad_norm_stds
        })
    
    if len(data_list) == 0:
        print(f"Warning: No valid data found for {noise_type}")
        continue
    
    # 按噪声强度排序
    data_list.sort(key=lambda x: x['noise_strength'])
    
    # 创建图表
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # 绘制每条曲线
    for idx, data in enumerate(data_list):
        epochs = data['epochs']
        grad_norms = data['grad_norms']
        grad_norm_stds = data['grad_norm_stds']
        noise_strength = data['noise_strength']
        
        color = colors[idx % len(colors)]
        label = format_noise_strength_label(noise_strength, noise_type)
        
        # 绘制误差带
        ax.fill_between(epochs, grad_norms - grad_norm_stds, grad_norms + grad_norm_stds,
                        alpha=0.2, color=color, label='_nolegend_')
        
        # 绘制折线
        ax.plot(epochs, grad_norms, linewidth=2, color=color, label=label)
    
    # 对于ir_drop使用对数坐标
    if noise_type == 'noise_boundary_ir_drop':
        ax.set_yscale('log')
    
    # 设置标题和标签
    #ax.set_title(noise_names[noise_type], fontsize=20, fontweight='bold', fontfamily='serif')
    ax.set_xlabel('Epoch', fontsize=18, fontfamily='serif')
    ax.set_ylabel('Gradient Norm', fontsize=18, fontfamily='serif')
    
    # 设置网格
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # 设置刻度字体
    ax.tick_params(labelsize=16)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily('serif')
    
    # 添加图例
    legend = ax.legend(loc='best', fontsize=14, framealpha=0.9, title='Noise Strength')
    legend.get_title().set_fontfamily('serif')
    legend.get_title().set_fontsize(14)
    for text in legend.get_texts():
        text.set_fontfamily('serif')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    output_name = noise_type.replace('noise_boundary_', 'grad_norm_')
    output_path = base_dir / f"{output_name}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存到: {output_path}")
    
    output_path_pdf = base_dir / f"{output_name}.pdf"
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"图表已保存到: {output_path_pdf}")
    
    plt.close()

print("所有图表生成完成！")

