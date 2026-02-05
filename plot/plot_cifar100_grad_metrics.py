#!/usr/bin/env python3
"""
绘制CIFAR-100的grad_metrics数据
每个噪声类型两个图：grad_norm+grad_norm_std（误差带），grad_var
同一噪声的3种强度曲线放在一个图里
"""

import csv
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path
from collections import defaultdict

# 设置matplotlib使用Times New Roman字体
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 20
matplotlib.rcParams['pdf.fonttype'] = 42  # 确保PDF中的文字可编辑

# 设置低饱和度的配色方案（3种强度用3种颜色）
colors = ['#4472C4', '#70AD47', '#ED7D31']  # 蓝色、绿色、红橙色

def load_data(csv_path):
    """加载CSV数据"""
    data = defaultdict(lambda: defaultdict(lambda: {
        'epochs': [],
        'grad_norm': [],
        'grad_norm_std': [],
        'grad_var': []
    }))
    
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            noise_type = row['noise_type']
            intensity = row['intensity']
            epoch = int(row['epoch'])
            grad_norm = float(row['grad_norm'])
            grad_norm_std = float(row['grad_norm_std'])
            grad_var = float(row['grad_var'])
            
            data[noise_type][intensity]['epochs'].append(epoch)
            data[noise_type][intensity]['grad_norm'].append(grad_norm)
            data[noise_type][intensity]['grad_norm_std'].append(grad_norm_std)
            data[noise_type][intensity]['grad_var'].append(grad_var)
    
    return data

def extract_intensity_value(intensity_str):
    """从intensity字符串中提取数值用于排序"""
    # 例如 "adc_bits_2.0" -> 2.0
    try:
        parts = intensity_str.split('_')
        if len(parts) >= 2:
            return float(parts[-1])
        return float(intensity_str)
    except:
        return 0.0

def plot_grad_norm_and_std(noise_type, intensities_data, output_dir):
    """绘制grad_norm和grad_norm_std图（std用误差带）"""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # 按强度值排序
    sorted_intensities = sorted(intensities_data.items(), 
                               key=lambda x: extract_intensity_value(x[0]))
    
    for idx, (intensity, data) in enumerate(sorted_intensities):
        epochs = np.array(data['epochs'])
        grad_norm = np.array(data['grad_norm'])
        grad_norm_std = np.array(data['grad_norm_std'])
        
        # 按epoch排序
        sort_idx = np.argsort(epochs)
        epochs = epochs[sort_idx]
        grad_norm = grad_norm[sort_idx]
        grad_norm_std = grad_norm_std[sort_idx]
        
        color = colors[idx % len(colors)]
        label = intensity.replace('_', ' ')
        
        # 绘制误差带（grad_norm ± grad_norm_std）
        ax.fill_between(epochs, grad_norm - grad_norm_std, grad_norm + grad_norm_std,
                       alpha=0.2, color=color, label='_nolegend_')
        
        # 绘制grad_norm折线
        ax.plot(epochs, grad_norm, marker='o', linestyle='-', 
               color=color, label=label, linewidth=2, markersize=4, markevery=5)
    
    ax.set_xlabel('Epoch', fontsize=20, fontname='Times New Roman')
    ax.set_ylabel('Gradient Norm', fontsize=20, fontname='Times New Roman')
    ax.tick_params(axis='both', labelsize=18)
    
    # 添加图例（不使用阴影）
    ax.legend(loc='best', fontsize=16, frameon=True, fancybox=True, shadow=False)
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # 设置标题
    noise_type_display = noise_type.replace('noise_boundary_', '').replace('_', ' ').title()
    # ax.set_title(f'{noise_type_display} - Grad Norm & Std', fontsize=22, fontname='Times New Roman', pad=15)
    
    plt.tight_layout()
    
    # 保存为PDF
    output_path = output_dir / f"{noise_type}_grad_norm_CIFAR100.pdf"
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"已保存: {output_path}")
    
    plt.close()

def plot_grad_var(noise_type, intensities_data, output_dir):
    """绘制grad_var图"""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # 按强度值排序
    sorted_intensities = sorted(intensities_data.items(), 
                               key=lambda x: extract_intensity_value(x[0]))
    
    for idx, (intensity, data) in enumerate(sorted_intensities):
        epochs = np.array(data['epochs'])
        grad_var = np.array(data['grad_var'])
        
        # 按epoch排序
        sort_idx = np.argsort(epochs)
        epochs = epochs[sort_idx]
        grad_var = grad_var[sort_idx]
        
        color = colors[idx % len(colors)]
        label = intensity.replace('_', ' ')
        
        # 绘制grad_var折线
        ax.plot(epochs, grad_var, marker='s', linestyle='-', 
               color=color, label=label, linewidth=2, markersize=4, markevery=5)
    
    ax.set_xlabel('Epoch', fontsize=20, fontname='Times New Roman')
    ax.set_ylabel('Gradient Variance', fontsize=20, fontname='Times New Roman')
    ax.tick_params(axis='both', labelsize=18)
    
    # 使用科学计数法格式化Y轴
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    ax.yaxis.major.formatter._useMathText = True
    
    # 添加图例（不使用阴影）
    ax.legend(loc='best', fontsize=16, frameon=True, fancybox=True, shadow=False)
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # 设置标题
    noise_type_display = noise_type.replace('noise_boundary_', '').replace('_', ' ').title()
    # ax.set_title(f'{noise_type_display} - Grad Var', fontsize=22, fontname='Times New Roman', pad=15)
    
    plt.tight_layout()
    
    # 保存为PDF
    output_path = output_dir / f"{noise_type}_grad_var_CIFAR100.pdf"
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"已保存: {output_path}")
    
    plt.close()

def main():
    """主函数"""
    # 文件路径
    csv_path = Path("output/CIFAR-100/grad_metrics.csv")
    output_dir = Path("output/CIFAR-100")
    
    if not csv_path.exists():
        print(f"错误: 找不到文件 {csv_path}")
        return
    
    # 加载数据
    print("正在加载数据...")
    data = load_data(csv_path)
    
    print(f"找到 {len(data)} 种噪声类型: {list(data.keys())}")
    
    # 为每种噪声类型绘制两个图表
    for noise_type, intensities_data in sorted(data.items()):
        print(f"正在绘制 {noise_type}...")
        
        # 绘制grad_norm和grad_norm_std图
        plot_grad_norm_and_std(noise_type, intensities_data, output_dir)
        
        # 绘制grad_var图
        plot_grad_var(noise_type, intensities_data, output_dir)
    
    print(f"\n所有图表已保存到 {output_dir} 目录")

if __name__ == "__main__":
    main()
