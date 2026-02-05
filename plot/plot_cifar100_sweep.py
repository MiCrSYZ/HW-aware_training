#!/usr/bin/env python3
"""
绘制CIFAR-100的noise_boundary_sweep数据折线图
每个噪声类型一张图
"""

import json
import re
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path

# 设置matplotlib使用Times New Roman字体
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 20
matplotlib.rcParams['pdf.fonttype'] = 42  # 确保PDF中的文字可编辑

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
        use_original: 如果为True，返回原始字符串表示
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

def normalize_noise_name(name):
    """标准化噪声类型名称"""
    name_mapping = {
        "ADC_direct": "adc_direct",
        "ADC_STE": "adc_ste",
        "IR-drop_paper": "ir_drop",
        "IR-drop_crossbar": "ir_drop_beta",
        "read_noise": "read_noise_sigma",
        "variability": "variability_sigma",
    }
    return name_mapping.get(name, name.lower())

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
        
        noise_name_raw = lines[0].strip().rstrip(':')
        noise_name = normalize_noise_name(noise_name_raw)
        
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
            results[noise_name] = {
                'raw_name': noise_name_raw,
                'data': data
            }
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse JSON for {noise_name} in {filepath}: {e}")
            continue
    
    return results

def plot_noise_type(noise_name, noise_info, output_dir):
    """为单个噪声类型绘制图表"""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    data = noise_info['data']
    raw_name = noise_info['raw_name']
    
    noise_strengths = data.get('noise_strengths', [])
    # 尝试获取val_accuracies，如果没有则尝试final_accuracies
    val_accuracies = data.get('val_accuracies', [])
    if not val_accuracies:
        val_accuracies = data.get('final_accuracies', [])
    
    if not noise_strengths or not val_accuracies:
        print(f"Warning: {noise_name} 没有有效数据")
        return
    
    # 转换为numpy数组并排序
    strengths = np.array(noise_strengths)
    accuracies = np.array(val_accuracies)
    
    # 按strength排序
    sort_idx = np.argsort(strengths)
    strengths = strengths[sort_idx]
    accuracies = accuracies[sort_idx]
    
    # 确定X轴处理方式（read_noise_sigma使用均匀分布）
    use_uniform_x = (noise_name == 'read_noise_sigma' and len(strengths) > 0)
    
    if use_uniform_x:
        x_positions = np.arange(len(strengths))
    else:
        x_positions = strengths
    
    # 绘制折线
    ax.plot(x_positions, accuracies, marker='o', linestyle='-', 
           color='#4472C4', linewidth=2, markersize=10)
    
    # 标注准确率值
    y_range = max(accuracies) - min(accuracies) if len(accuracies) > 1 else 10
    y_min = min(accuracies) - y_range * 0.1
    y_max = max(accuracies) + y_range * 0.1
    
    for i, (x_pos, acc) in enumerate(zip(x_positions, accuracies)):
        offset_y = y_range * 0.05
        ax.annotate(f'{acc:.2f}', 
                   xy=(x_pos, acc), 
                   xytext=(x_pos, acc + offset_y),
                   fontsize=16,
                   ha='center',
                   va='bottom',
                   color='#4472C4',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                           edgecolor='#4472C4', alpha=0.8, linewidth=1.5))
    
    # 设置X轴
    ax.set_xlabel('Noise Strength', fontsize=20, fontname='Times New Roman')
    ax.tick_params(axis='x', labelsize=18)
    
    if use_uniform_x:
        # 使用均匀分布的X轴位置
        x_positions_labels = np.arange(len(strengths))
        ax.set_xticks(x_positions_labels)
        labels = [format_noise_strength(s) for s in strengths]
        ax.set_xticklabels(labels, rotation=45, ha='right', fontfamily='Times New Roman')
        ax.set_xlim(-0.3, len(strengths) - 1 + 0.3)
    else:
        # 使用原始值
        ax.set_xticks(strengths)
        # ir_drop_beta使用原始小数位
        if noise_name == 'ir_drop_beta':
            labels = [format_noise_strength(s, use_original=True) for s in strengths]
        else:
            labels = [format_noise_strength(s) for s in strengths]
        ax.set_xticklabels(labels, rotation=45, ha='right', fontfamily='Times New Roman')
    
    # 设置Y轴
    ax.set_ylabel('Accuracy (%)', fontsize=20, fontname='Times New Roman')
    ax.tick_params(axis='y', labelsize=18)
    
    # 设置Y轴范围
    if len(accuracies) > 1:
        y_margin = (y_max - y_min) * 0.1
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
    
    # 设置标题（使用原始名称）
    # ax.set_title(raw_name, fontsize=22, fontname='Times New Roman', pad=15)
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存为PDF
    output_path = output_dir / f"{noise_name}_CIFAR100.pdf"
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"已保存: {output_path}")
    
    plt.close()

def main():
    """主函数"""
    # 文件路径
    file_path = Path("output/CIFAR-100/noise_boundary_sweep_seed42_CIFAR-100.txt")
    output_dir = Path("output/CIFAR-100")
    
    if not file_path.exists():
        print(f"错误: 找不到文件 {file_path}")
        return
    
    # 解析文件
    print("正在解析文件...")
    data = parse_file(file_path)
    
    print(f"找到 {len(data)} 种噪声类型: {list(data.keys())}")
    
    # 为每种噪声类型绘制图表
    for noise_name, noise_info in sorted(data.items()):
        print(f"正在绘制 {noise_name}...")
        plot_noise_type(noise_name, noise_info, output_dir)
    
    print(f"\n所有图表已保存到 {output_dir} 目录")

if __name__ == "__main__":
    main()

