#!/usr/bin/env python3
"""
绘制噪声类型准确率折线图
"""

import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.ticker as ticker

# 设置字体 - 使用Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 20
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 启用LaTeX渲染（用于上标）
plt.rcParams['text.usetex'] = False  # 不使用完整LaTeX，使用matplotlib的内置数学渲染
plt.rcParams['mathtext.default'] = 'regular'  # 使用常规数学字体

# 读取CSV数据
def read_csv_data(filepath):
    """读取CSV文件并返回按noise_name组织的数据"""
    data = defaultdict(list)
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            noise_name = row['noise_name']
            data[noise_name].append({
                'noise_strength': float(row['noise_strength']),
                'accuracy_mean': float(row['accuracy_mean']),
                'accuracy_std': float(row['accuracy_std'])
            })
    # 对每个噪声类型按noise_strength排序
    for noise_name in data:
        data[noise_name].sort(key=lambda x: x['noise_strength'])
    return data

def format_scientific_label(value):
    """将数值转换为LaTeX科学计数法格式的字符串（带真正上标）"""
    if value == 0:
        return '0'
    
    # 计算科学计数法的指数
    exp = int(np.floor(np.log10(abs(value))))
    coeff = value / (10 ** exp)
    
    # 格式化系数，保留适当的小数位
    if abs(coeff - round(coeff)) < 1e-10:
        coeff_str = str(int(round(coeff)))
    else:
        coeff_str = f'{coeff:.1f}'.rstrip('0').rstrip('.')
    
    if exp == 0:
        return coeff_str
    else:
        # 使用LaTeX格式，上标用^表示
        return f'${coeff_str} \\times 10^{{{exp}}}$'

base_dir = Path("output")
data_comp = read_csv_data(base_dir / "noise_statistics.csv")
data_no_comp = read_csv_data(base_dir / "noise_statistics_no_comp.csv")

# 噪声类型列表
noise_types = ['IR_drop_crossbar', 'drift_alpha', 'read_noise_sigma', 'stuck_ratio', 'variability_sigma']

# 噪声类型的英文名称（用于标题）
noise_names_en = {
    'IR_drop_crossbar': 'IR Drop (Crossbar)',
    'drift_alpha': 'Drift Alpha',
    'read_noise_sigma': 'Read Noise Sigma',
    'stuck_ratio': 'Stuck Ratio',
    'variability_sigma': 'Variability Sigma'
}

# 使用低饱和度的配色方案
colors = {
    'comp': '#2E86AB',      # 蓝色（低饱和度）
    'no_comp': '#A23B72'    # 紫红色（低饱和度）
}

# 为每个噪声类型生成单独的图片
for noise_type in noise_types:
    # 获取数据
    comp_data = data_comp.get(noise_type, [])
    no_comp_data = data_no_comp.get(noise_type, [])
    
    if not comp_data or not no_comp_data:
        continue
    
    # 创建单独的图表
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    
    x_comp_orig = np.array([d['noise_strength'] for d in comp_data])
    y_comp = np.array([d['accuracy_mean'] for d in comp_data])
    std_comp = np.array([d['accuracy_std'] for d in comp_data])
    
    x_no_comp_orig = np.array([d['noise_strength'] for d in no_comp_data])
    y_no_comp = np.array([d['accuracy_mean'] for d in no_comp_data])
    std_no_comp = np.array([d['accuracy_std'] for d in no_comp_data])
    
    # 对于read_noise_sigma，使用等间距的X轴位置
    if noise_type == 'read_noise_sigma':
        # 创建等间距的位置（0, 1, 2, 3, ...）
        all_x_orig = sorted(set(list(x_comp_orig) + list(x_no_comp_orig)))
        x_positions = np.arange(len(all_x_orig))
        # 创建映射字典
        x_map = {orig: pos for orig, pos in zip(all_x_orig, x_positions)}
        x_comp = np.array([x_map[x] for x in x_comp_orig])
        x_no_comp = np.array([x_map[x] for x in x_no_comp_orig])
        x_ticks = x_positions
        x_tick_labels = [format_scientific_label(x) for x in all_x_orig]
    else:
        # 其他噪声类型使用原始X值
        x_comp = x_comp_orig
        x_no_comp = x_no_comp_orig
        x_ticks = None
        x_tick_labels = None
    
    # 绘制误差带
    ax.fill_between(x_comp, y_comp - std_comp, y_comp + std_comp, 
                    alpha=0.2, color=colors['comp'], label='_nolegend_')
    ax.fill_between(x_no_comp, y_no_comp - std_no_comp, y_no_comp + std_no_comp, 
                    alpha=0.2, color=colors['no_comp'], label='_nolegend_')
    
    # 绘制折线图
    line1 = ax.plot(x_comp, y_comp, marker='s', markersize=8, linewidth=2, 
                    color=colors['comp'], label='Hardware-Aware Training', linestyle='-')
    line2 = ax.plot(x_no_comp, y_no_comp, marker='o', markersize=8, linewidth=2, 
                    color=colors['no_comp'], label='No Compensation Inference', linestyle='-')

    
    # 在标记上方显示数值（调整位置避免重叠）
    for i, (x, y) in enumerate(zip(x_comp, y_comp)):
        # 计算合适的偏移量
        offset = max(std_comp[i] + 1.5, 2.0)
        ax.text(x, y - offset, f'{y:.2f}',
                ha='center', va='top', fontsize=16, fontfamily='serif',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))
    
    for i, (x, y) in enumerate(zip(x_no_comp, y_no_comp)):
        # 计算合适的偏移量
        offset = max(std_no_comp[i] + 1.5, 2.0)
        
        # 对于stuck_ratio，前两个标签放在下方，其他放在上方
        if noise_type == 'stuck_ratio' and i < 2:
            # 前两个标签放在下方
            ax.text(x, y - offset, f'{y:.2f}',
                    ha='center', va='top', fontsize=16, fontfamily='serif',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))
        else:
            # 其他标签放在上方
            ax.text(x, y + offset, f'{y:.2f}',
                    ha='center', va='bottom', fontsize=16, fontfamily='serif',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))


    # 添加Y=90.05%的虚线
    ax.axhline(y=90.05, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='_nolegend_')
    
    # 设置标题和标签
    #ax.set_title(noise_names_en[noise_type], fontsize=20, fontweight='bold', fontfamily='serif')
    ax.set_xlabel('Noise Strength', fontsize=18, fontfamily='serif')
    ax.set_ylabel('Accuracy (%)', fontsize=18, fontfamily='serif')
    
    # 设置网格
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # 设置刻度字体（在设置X轴标签之后）
    ax.tick_params(labelsize=16)
    
    # 设置Y轴范围（根据数据自动调整，但确保能看到90.05%的线）
    y_min = min(y_comp.min() - std_comp.max() - 5, y_no_comp.min() - std_no_comp.max() - 5)
    y_max = max(y_comp.max() + std_comp.max() + 5, y_no_comp.max() + std_no_comp.max() + 5, 100)
    # 确保Y轴范围至少包含20-100
    y_min = min(y_min, 20)
    y_max = max(y_max, 100)
    ax.set_ylim(y_min, y_max)
    
    # 设置X轴刻度
    if noise_type == 'read_noise_sigma':
        # read_noise_sigma使用等间距位置
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_tick_labels)
    else:
        # 其他噪声类型：只显示数据点的扰动强度值
        all_x_values = sorted(set(list(x_comp_orig) + list(x_no_comp_orig)))
        ax.set_xticks(all_x_values)
        
        # 格式化X轴标签
        if noise_type == 'drift_alpha':
            # drift_alpha使用 n×10^{-m} 格式
            labels = [format_scientific_label(x) for x in all_x_values]
            ax.set_xticklabels(labels)
        else:
            # 其他噪声类型：保留2位小数
            ax.ticklabel_format(style='plain', axis='x')
            labels = [f'{x:.2f}' for x in all_x_values]
            ax.set_xticklabels(labels)
    
    # 设置Y轴刻度：从20到100，每20一个，加上90.05
    y_ticks = [20, 40, 60, 80, 90.05, 100]
    # 只保留在Y轴范围内的刻度
    y_ticks = [tick for tick in y_ticks if y_min <= tick <= y_max]
    ax.set_yticks(y_ticks)
    
    # 格式化Y轴标签：整数显示为整数，90.05显示为90.05
    y_tick_labels = []
    for tick in y_ticks:
        if tick == 90.05:
            y_tick_labels.append('90.05')
        elif tick == int(tick):
            y_tick_labels.append(str(int(tick)))
        else:
            y_tick_labels.append(f'{tick:.2f}')
    ax.set_yticklabels(y_tick_labels)
    
    # 设置刻度字体
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily('serif')
    
    # 添加图例
    legend = ax.legend(loc='best', fontsize=14, framealpha=0.9)
    for text in legend.get_texts():
        text.set_fontfamily('serif')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    output_path = base_dir / f"noise_accuracy_{noise_type}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存到: {output_path}")
    
    # 也保存为PDF（矢量图）
    output_path_pdf = base_dir / f"noise_accuracy_{noise_type}.pdf"
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"图表已保存到: {output_path_pdf}")
    
    plt.close()  # 关闭图形以释放内存

