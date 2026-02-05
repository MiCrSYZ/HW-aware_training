#!/usr/bin/env python3
"""
绘制seed42中特殊噪声类型的图表
- ADC_direct 和 ADC_STE 在一张图
- IR_drop_paper (comp 和 no_comp) 在一张图
"""

import csv
import json
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

# 设置字体 - 使用Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 20
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 启用LaTeX渲染（用于上标）
plt.rcParams['text.usetex'] = False
plt.rcParams['mathtext.default'] = 'regular'

# 使用低饱和度的配色方案（与之前保持一致）
colors = {
    'comp': '#2E86AB',      # 蓝色（低饱和度）
    'no_comp': '#A23B72'    # 紫红色（低饱和度）
}

def parse_file(filepath):
    """解析数据文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 按 '---' 分割不同的噪声类型
    sections = content.split('---')
    results = {}
    
    for section in sections:
        if not section.strip():
            continue
        
        # 提取噪声类型名称（第一行）
        lines = section.strip().split('\n')
        if not lines:
            continue
        
        noise_name = lines[0].strip().rstrip(':')
        
        # 提取JSON部分
        json_start = section.find('{')
        if json_start == -1:
            continue
        
        json_str = section[json_start:]
        # 移除尾随逗号（在}之前）
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        try:
            data = json.loads(json_str)
            results[noise_name] = data
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse JSON for {noise_name} in {filepath}: {e}")
            continue
    
    return results

base_dir = Path("output")

# 读取seed42的数据
data_comp_seed42 = parse_file(base_dir / "noise_boundary_sweep_seed42.txt")
data_no_comp_seed42 = parse_file(base_dir / "noise_boundary_sweep_no_comp_seed42.txt")

# ========== 图1: ADC_direct 和 ADC_STE ==========
if 'ADC_direct' in data_comp_seed42 and 'ADC_STE' in data_comp_seed42:
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    
    # 获取数据
    adc_direct = data_comp_seed42['ADC_direct']
    adc_ste = data_comp_seed42['ADC_STE']
    
    x_direct = np.array(adc_direct['noise_strengths'])
    y_direct = np.array(adc_direct['val_accuracies'])
    
    x_ste = np.array(adc_ste['noise_strengths'])
    y_ste = np.array(adc_ste['val_accuracies'])
    
    # 绘制折线图
    ax.plot(x_direct, y_direct, marker='s', markersize=8, linewidth=2, 
            color=colors['comp'], label='ADC Direct', linestyle='-')
    ax.plot(x_ste, y_ste, marker='o', markersize=8, linewidth=2, 
            color=colors['no_comp'], label='ADC STE', linestyle='-')
    
    # 在标记上方显示数值
    for i, (x, y) in enumerate(zip(x_direct, y_direct)):
        offset = 2.0
        ax.text(x, y + offset, f'{y:.2f}',
                ha='center', va='bottom', fontsize=16, fontfamily='serif',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))
    
    for i, (x, y) in enumerate(zip(x_ste, y_ste)):
        offset = 2.0
        ax.text(x, y - offset, f'{y:.2f}',
                ha='center', va='top', fontsize=16, fontfamily='serif',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))
    
    # 添加Y=90.05%的虚线
    ax.axhline(y=91.04, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='_nolegend_')
    
    # 设置标题和标签
    #ax.set_title('ADC Direct vs ADC STE', fontsize=20, fontweight='bold', fontfamily='serif')
    ax.set_xlabel('Noise Strength', fontsize=18, fontfamily='serif')
    ax.set_ylabel('Accuracy (%)', fontsize=18, fontfamily='serif')
    
    # 设置网格
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # 设置刻度字体
    ax.tick_params(labelsize=16)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily('serif')
    
    # 设置Y轴范围
    y_min = min(y_direct.min(), y_ste.min()) - 5
    y_max = max(y_direct.max(), y_ste.max(), 92) + 5
    ax.set_ylim(y_min, y_max)
    
    # 设置X轴刻度：只显示数据点的值
    all_x_values = sorted(set(list(x_direct) + list(x_ste)))
    ax.set_xticks(all_x_values)
    labels = [f'{x:.0f}' for x in all_x_values]
    ax.set_xticklabels(labels)
    
    # 设置Y轴刻度：从20到100，每20一个，加上90.05
    y_ticks = [20, 40, 60, 80, 91.04, 100]
    y_ticks = [tick for tick in y_ticks if y_min <= tick <= y_max]
    ax.set_yticks(y_ticks)
    
    # 格式化Y轴标签
    y_tick_labels = []
    for tick in y_ticks:
        if tick == 91.04:
            y_tick_labels.append('91.04')
        elif tick == int(tick):
            y_tick_labels.append(str(int(tick)))
        else:
            y_tick_labels.append(f'{tick:.2f}')
    ax.set_yticklabels(y_tick_labels)
    
    # 添加图例
    legend = ax.legend(loc='best', fontsize=14, framealpha=0.9)
    for text in legend.get_texts():
        text.set_fontfamily('serif')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    output_path = base_dir / "noise_accuracy_ADC_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存到: {output_path}")
    
    output_path_pdf = base_dir / "noise_accuracy_ADC_comparison.pdf"
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"图表已保存到: {output_path_pdf}")
    
    plt.close()

# ========== 图2: IR_drop_paper (comp 和 no_comp) ==========
if 'IR_drop_paper' in data_comp_seed42 and 'IR_drop_paper' in data_no_comp_seed42:
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    
    # 获取数据
    ir_drop_comp = data_comp_seed42['IR_drop_paper']
    ir_drop_no_comp = data_no_comp_seed42['IR_drop_paper']
    
    x_comp = np.array(ir_drop_comp['noise_strengths'])
    y_comp = np.array(ir_drop_comp['val_accuracies'])
    
    x_no_comp = np.array(ir_drop_no_comp['noise_strengths'])
    y_no_comp = np.array(ir_drop_no_comp['val_accuracies'])
    
    # 绘制折线图
    ax.plot(x_comp, y_comp, marker='s', markersize=8, linewidth=2, 
            color=colors['comp'], label='Hardware-Aware Training', linestyle='-')
    ax.plot(x_no_comp, y_no_comp, marker='o', markersize=8, linewidth=2, 
            color=colors['no_comp'], label='No Compensation Inference', linestyle='-')
    
    # 在标记上方显示数值
    for i, (x, y) in enumerate(zip(x_comp, y_comp)):
        offset = 2.0
        ax.text(x, y - offset, f'{y:.2f}',
                ha='center', va='top', fontsize=16, fontfamily='serif',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))
    
    for i, (x, y) in enumerate(zip(x_no_comp, y_no_comp)):
        offset = 2.0
        ax.text(x, y + offset, f'{y:.2f}',
                ha='center', va='bottom', fontsize=16, fontfamily='serif',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))
    
    # 添加Y=90.05%的虚线
    ax.axhline(y=91.04, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='_nolegend_')
    
    # 设置标题和标签
    #ax.set_title('IR Drop (Paper)', fontsize=20, fontweight='bold', fontfamily='serif')
    ax.set_xlabel('Noise Strength', fontsize=18, fontfamily='serif')
    ax.set_ylabel('Accuracy (%)', fontsize=18, fontfamily='serif')
    
    # 设置网格
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # 设置刻度字体
    ax.tick_params(labelsize=16)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily('serif')
    
    # 设置Y轴范围
    y_min = min(y_comp.min(), y_no_comp.min()) - 5
    y_max = max(y_comp.max(), y_no_comp.max(), 92) + 5
    ax.set_ylim(y_min, y_max)
    
    # 设置X轴刻度：只显示数据点的值，保留2位小数
    all_x_values = sorted(set(list(x_comp) + list(x_no_comp)))
    ax.set_xticks(all_x_values)
    labels = [f'{x:.2f}' for x in all_x_values]
    ax.set_xticklabels(labels)
    
    # 设置Y轴刻度：从20到100，每20一个，加上90.05
    y_ticks = [20, 40, 60, 80, 91.04, 100]
    y_ticks = [tick for tick in y_ticks if y_min <= tick <= y_max]
    ax.set_yticks(y_ticks)
    
    # 格式化Y轴标签
    y_tick_labels = []
    for tick in y_ticks:
        if tick == 91.04:
            y_tick_labels.append('91.04')
        elif tick == int(tick):
            y_tick_labels.append(str(int(tick)))
        else:
            y_tick_labels.append(f'{tick:.2f}')
    ax.set_yticklabels(y_tick_labels)
    
    # 添加图例
    legend = ax.legend(loc='best', fontsize=14, framealpha=0.9)
    for text in legend.get_texts():
        text.set_fontfamily('serif')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    output_path = base_dir / "noise_accuracy_IR_drop_paper.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存到: {output_path}")
    
    output_path_pdf = base_dir / "noise_accuracy_IR_drop_paper.pdf"
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"图表已保存到: {output_path_pdf}")
    
    plt.close()

print("所有图表生成完成！")

