#!/usr/bin/env python3
"""
绘制sweep统计数据的折线图
显示h-mean和std误差带，包含comp/no_comp和test_acc/val_acc
"""

import csv
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path

# 设置matplotlib使用Times New Roman字体
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 20
matplotlib.rcParams['pdf.fonttype'] = 42  # 确保PDF中的文字可编辑

# 设置低饱和度的配色方案
colors = {
    'comp_val': '#4472C4',      # 蓝色（低饱和度）
    'comp_test': '#70AD47',     # 绿色（低饱和度）
    'no_comp_val': '#FFC000',   # 橙色（低饱和度）
    'no_comp_test': '#ED7D31',  # 红橙色（低饱和度）
}

def number_to_superscript(num):
    """将数字转换为Unicode上标字符"""
    superscript_map = {
        '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
        '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
        '-': '⁻'
    }
    return ''.join(superscript_map.get(char, char) for char in str(num))

def format_noise_strength(value, use_original=False):
    """格式化噪声强度值：保留2位小数，超过3位的用科学计数法（使用×而不是e）
    
    Args:
        value: 噪声强度值
        use_original: 如果为True，返回原始字符串表示（用于ir_drop_beta）
    """
    if use_original:
        return str(value)
    
    if abs(value) >= 1000 or (abs(value) < 0.001 and value != 0):
        # 转换为科学计数法字符串，然后替换e为×
        sci_str = f"{value:.2e}"
        # 将格式从 "1.23e-08" 转换为 "1.23×10⁻⁸"（上标形式）
        if 'e' in sci_str.lower():
            base, exp = sci_str.lower().split('e')
            exp = int(exp)
            # 使用×和上标格式，保持Times New Roman字体
            exp_superscript = number_to_superscript(exp)
            return f"{base}×10{exp_superscript}"
        return sci_str
    else:
        return f"{value:.2f}"

def load_data(csv_path):
    """加载CSV数据"""
    data = {}
    
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            noise_name = row['noise_name']
            version = row['version']
            noise_strength = float(row['noise_strength'])
            
            # 初始化数据结构
            if noise_name not in data:
                data[noise_name] = {
                    'comp_val': {'strengths': [], 'hmeans': [], 'stds': []},
                    'comp_test': {'strengths': [], 'hmeans': [], 'stds': []},
                    'no_comp_val': {'strengths': [], 'hmeans': [], 'stds': []},
                    'no_comp_test': {'strengths': [], 'hmeans': [], 'stds': []},
                }
            
            # val_acc数据
            val_key = f"{version}_val"
            val_hmean = float(row['val_acc_hmean']) if row['val_acc_hmean'] else None
            val_std = float(row['val_acc_std']) if row['val_acc_std'] else None
            if val_hmean is not None:
                data[noise_name][val_key]['strengths'].append(noise_strength)
                data[noise_name][val_key]['hmeans'].append(val_hmean)
                data[noise_name][val_key]['stds'].append(val_std if val_std is not None else 0)
            
            # test_acc数据
            test_key = f"{version}_test"
            test_hmean = float(row['test_acc_hmean']) if row['test_acc_hmean'] else None
            test_std = float(row['test_acc_std']) if row['test_acc_std'] else None
            if test_hmean is not None:
                data[noise_name][test_key]['strengths'].append(noise_strength)
                data[noise_name][test_key]['hmeans'].append(test_hmean)
                data[noise_name][test_key]['stds'].append(test_std if test_std is not None else 0)
    
    return data

def plot_noise_type(noise_name, data, output_dir):
    """为单个噪声类型绘制图表"""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # 获取该噪声类型的所有数据
    noise_data = data[noise_name]
    
    # 先收集所有数据以确定Y轴范围
    all_hmeans = []
    all_stds = []
    all_strengths = []
    for key in noise_data.keys():
        if len(noise_data[key]['hmeans']) > 0:
            all_hmeans.extend(noise_data[key]['hmeans'])
            all_stds.extend(noise_data[key]['stds'])
            all_strengths.extend(noise_data[key]['strengths'])
    
    # 设置Y轴范围
    if all_hmeans:
        y_min_data = min(all_hmeans) - max(all_stds) if all_stds else min(all_hmeans)
        y_max_data = max(all_hmeans) + max(all_stds) if all_stds else max(all_hmeans)
        
        # 设置Y轴范围，留出一些边距用于标注
        y_margin = (y_max_data - y_min_data) * 0.15
        y_min = max(0, y_min_data - y_margin)
        y_max = min(100, y_max_data + y_margin)
        
        # 设置Y轴刻度，每20%一个标记
        y_start = int(y_min // 20) * 20
        y_end = int((y_max // 20) + 1) * 20
        y_ticks = np.arange(y_start, y_end + 1, 20)
        
        # 确保包含0和100如果数据接近
        if y_min < 20:
            y_ticks = np.concatenate([[0], y_ticks[y_ticks > 0]])
        if y_max > 80:
            y_ticks = np.concatenate([y_ticks[y_ticks < 100], [100]])
        
        ax.set_yticks(y_ticks)
        ax.set_ylim(y_min, y_max)
    
    # 确定X轴处理方式
    unique_strengths = sorted(set(all_strengths)) if all_strengths else []
    use_uniform_x = (noise_name == 'read_noise_sigma' and len(unique_strengths) > 0)
    
    # 定义要绘制的4条线
    lines_to_plot = [
        ('comp_val', 'Comp Val Acc', colors['comp_val'], 'o', 'solid'),
        ('comp_test', 'Comp Test Acc', colors['comp_test'], 's', 'solid'),
        ('no_comp_val', 'No-Comp Val Acc', colors['no_comp_val'], 'o', 'dashed'),
        ('no_comp_test', 'No-Comp Test Acc', colors['no_comp_test'], 's', 'dashed'),
    ]
    
    # 绘制所有线条
    for key, label, color, marker, linestyle in lines_to_plot:
        if key in noise_data and len(noise_data[key]['strengths']) > 0:
            strengths = np.array(noise_data[key]['strengths'])
            hmeans = np.array(noise_data[key]['hmeans'])
            stds = np.array(noise_data[key]['stds'])
            
            # 按strength排序
            sort_idx = np.argsort(strengths)
            strengths = strengths[sort_idx]
            hmeans = hmeans[sort_idx]
            stds = stds[sort_idx]
            
            # 确定X轴位置
            if use_uniform_x:
                # 将strength值映射到均匀分布的X轴位置
                x_positions = np.array([unique_strengths.index(s) for s in strengths])
            else:
                x_positions = strengths
            
            # 绘制误差带
            ax.fill_between(x_positions, hmeans - stds, hmeans + stds, 
                           alpha=0.2, color=color, label='_nolegend_')
            
            # 绘制折线
            ax.plot(x_positions, hmeans, marker=marker, linestyle=linestyle, 
                   color=color, label=label, linewidth=2, markersize=8)
    
    # 标注准确率值（在所有线条绘制后）
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    for key, label, color, marker, linestyle in lines_to_plot:
        if key in noise_data and len(noise_data[key]['strengths']) > 0:
            strengths = np.array(noise_data[key]['strengths'])
            hmeans = np.array(noise_data[key]['hmeans'])
            stds = np.array(noise_data[key]['stds'])
            
            # 按strength排序
            sort_idx = np.argsort(strengths)
            strengths = strengths[sort_idx]
            hmeans = hmeans[sort_idx]
            stds = stds[sort_idx]
            
            # 确定X轴位置
            if use_uniform_x:
                x_positions = np.array([unique_strengths.index(s) for s in strengths])
            else:
                x_positions = strengths
            
            # 确定标注位置：test acc标下方，val acc标上方
            is_test = 'test' in key
            offset_y = -y_range * 0.05 if is_test else y_range * 0.05
            
            # 标注准确率值
            for x_pos, h, std in zip(x_positions, hmeans, stds):
                ax.annotate(f'{h:.2f}', 
                           xy=(x_pos, h), 
                           xytext=(x_pos, h + offset_y),
                           fontsize=16,
                           ha='center',
                           va='top' if is_test else 'bottom',
                           color=color,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                   edgecolor=color, alpha=0.8, linewidth=1.5))
    
    # 设置X轴
    ax.set_xlabel('Noise Strength', fontsize=20, fontname='Times New Roman')
    ax.tick_params(axis='x', labelsize=18)
    
    if unique_strengths:
        if use_uniform_x:
            # 使用均匀分布的X轴位置
            x_positions = np.arange(len(unique_strengths))
            ax.set_xticks(x_positions)
            # 对于read_noise_sigma，使用科学计数法格式
            labels = [format_noise_strength(s) for s in unique_strengths]
            ax.set_xticklabels(labels, rotation=45, ha='right', fontfamily='Times New Roman')
            ax.set_xlim(-0.3, len(unique_strengths) - 1 + 0.3)
        else:
            # 其他噪声类型使用原始值
            ax.set_xticks(unique_strengths)
            # ir_drop_beta使用原始小数位，其他使用格式化
            if noise_name == 'ir_drop_beta':
                labels = [format_noise_strength(s, use_original=True) for s in unique_strengths]
            else:
                labels = [format_noise_strength(s) for s in unique_strengths]
            ax.set_xticklabels(labels, rotation=45, ha='right', fontfamily='Times New Roman')
    
    # 设置Y轴标签
    ax.set_ylabel('Accuracy (%)', fontsize=20, fontname='Times New Roman')
    ax.tick_params(axis='y', labelsize=18)
    
    # 设置标题
    noise_type_display = noise_name.replace('_', ' ').title()
    #ax.set_title(noise_type_display, fontsize=22, fontname='Times New Roman', pad=15)
    
    # 添加图例（不使用阴影）
    ax.legend(loc='best', fontsize=16, frameon=True, fancybox=True, shadow=False)
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存为PDF
    output_path = output_dir / f"{noise_name}_statistics.pdf"
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"已保存: {output_path}")
    
    plt.close()

def main():
    """主函数"""
    # 文件路径
    csv_path = Path("output/sweep_statistics.csv")
    output_dir = Path("output")
    
    if not csv_path.exists():
        print(f"错误: 找不到文件 {csv_path}")
        return
    
    # 加载数据
    print("正在加载数据...")
    data = load_data(csv_path)
    
    # 获取所有噪声类型
    noise_types = sorted(data.keys())
    print(f"找到 {len(noise_types)} 种噪声类型: {noise_types}")
    
    # 为每种噪声类型绘制图表
    for noise_name in noise_types:
        print(f"正在绘制 {noise_name}...")
        plot_noise_type(noise_name, data, output_dir)
    
    print(f"\n所有图表已保存到 {output_dir} 目录")

if __name__ == "__main__":
    main()

