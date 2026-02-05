#!/usr/bin/env python3
"""
分析output目录中三组seeds数据的统计分析
计算h-mean和std，排除只在seed42中存在的噪声类型
"""

import json
import re
import os
from collections import defaultdict
from pathlib import Path
import numpy as np
import csv

def harmonic_mean(values):
    """计算调和平均数"""
    values = [v for v in values if not (np.isnan(v) or np.isinf(v))]
    if len(values) == 0:
        return np.nan
    return len(values) / sum(1.0 / v for v in values if v != 0)

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

def main(version="comp"):
    """
    分析output目录中三组seeds数据的统计分析
    version: "comp" 或 "no_comp"
    """
    # 文件路径
    base_dir = Path("output")
    seeds = [26, 3407, 42]
    
    # 根据版本选择文件名模式
    if version == "no_comp":
        file_pattern = "noise_boundary_sweep_no_comp_seed{}.txt"
        output_suffix = "_no_comp"
    else:
        file_pattern = "noise_boundary_sweep_seed{}.txt"
        output_suffix = ""
    
    print(f"正在分析 {version} 版本的数据...")
    
    # 读取所有文件
    all_data = {}
    for seed in seeds:
        filepath = base_dir / file_pattern.format(seed)
        if filepath.exists():
            all_data[seed] = parse_file(filepath)
        else:
            print(f"Warning: {filepath} not found")
    
    # 找出所有噪声类型
    all_noise_types = set()
    for seed_data in all_data.values():
        all_noise_types.update(seed_data.keys())
    
    # 找出只在seed42中存在的噪声类型
    seed26_types = set(all_data.get(26, {}).keys())
    seed3407_types = set(all_data.get(3407, {}).keys())
    seed42_types = set(all_data.get(42, {}).keys())
    
    only_in_seed42 = seed42_types - seed26_types - seed3407_types
    print(f"只在seed42中存在的噪声类型: {only_in_seed42}")
    
    # 收集需要统计的噪声类型（排除只在seed42中的）
    valid_noise_types = all_noise_types - only_in_seed42
    
    # 收集数据：按噪声类型和noise_strength组织
    stats_data = []
    
    for noise_name in sorted(valid_noise_types):
        # 收集所有seeds中该噪声类型的数据
        noise_data_by_seed = {}
        for seed in seeds:
            if seed in all_data and noise_name in all_data[seed]:
                noise_data_by_seed[seed] = all_data[seed][noise_name]
        
        if len(noise_data_by_seed) < 2:  # 至少需要2个seeds的数据
            continue
        
        # 获取noise_strengths（使用第一个seed的作为参考）
        first_seed_data = list(noise_data_by_seed.values())[0]
        noise_strengths = first_seed_data.get('noise_strengths', [])
        
        # 对每个noise_strength收集所有seeds的准确率和损失
        for i, strength in enumerate(noise_strengths):
            accuracies = []
            losses = []
            
            for seed in seeds:
                if seed in noise_data_by_seed:
                    seed_data = noise_data_by_seed[seed]
                    if i < len(seed_data.get('final_accuracies', [])):
                        acc = seed_data['final_accuracies'][i]
                        loss = seed_data['final_losses'][i]
                        accuracies.append(acc)
                        losses.append(loss)
            
            if len(accuracies) >= 2:  # 至少需要2个数据点
                # 计算统计量
                acc_hmean = harmonic_mean(accuracies)
                acc_std = np.std(accuracies) if len(accuracies) > 1 else 0.0
                acc_mean = np.mean(accuracies)
                
                # 过滤掉NaN和Inf的损失值
                valid_losses = [l for l in losses if not (np.isnan(l) or np.isinf(l))]
                if len(valid_losses) >= 2:
                    loss_hmean = harmonic_mean(valid_losses)
                    loss_std = np.std(valid_losses) if len(valid_losses) > 1 else 0.0
                    loss_mean = np.mean(valid_losses)
                else:
                    loss_hmean = np.nan
                    loss_std = np.nan
                    loss_mean = np.nan
                
                stats_data.append({
                    'noise_name': noise_name,
                    'noise_type': first_seed_data.get('noise_type', ''),
                    'noise_strength': strength,
                    'num_seeds': len(accuracies),
                    'accuracy_mean': acc_mean,
                    'accuracy_hmean': acc_hmean,
                    'accuracy_std': acc_std,
                    'loss_mean': loss_mean,
                    'loss_hmean': loss_hmean,
                    'loss_std': loss_std,
                })
    
    # 保存为CSV
    csv_path = base_dir / f"noise_statistics{output_suffix}.csv"
    if stats_data:
        fieldnames = stats_data[0].keys()
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in stats_data:
                # 转换numpy类型为Python原生类型
                csv_row = {}
                for k, v in row.items():
                    if isinstance(v, (np.integer, np.floating)):
                        if np.isnan(v):
                            csv_row[k] = ''
                        elif np.isinf(v):
                            csv_row[k] = 'Infinity'
                        else:
                            csv_row[k] = float(v)
                    else:
                        csv_row[k] = v
                writer.writerow(csv_row)
    print(f"\n统计结果已保存到: {csv_path}")
    
    # 同时保存为JSON
    json_path = base_dir / f"noise_statistics{output_suffix}.json"
    # 将numpy类型转换为Python原生类型以便JSON序列化
    json_data = []
    for row in stats_data:
        json_row = {}
        for k, v in row.items():
            if isinstance(v, (np.integer, np.floating)):
                if np.isnan(v):
                    json_row[k] = None
                elif np.isinf(v):
                    json_row[k] = "Infinity"
                else:
                    json_row[k] = float(v)
            else:
                json_row[k] = v
        json_data.append(json_row)
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"统计结果已保存到: {json_path}")
    
    # 打印摘要
    print(f"\n统计摘要:")
    print(f"共分析了 {len(valid_noise_types)} 种噪声类型")
    print(f"排除了 {len(only_in_seed42)} 种只在seed42中存在的噪声类型")
    print(f"共 {len(stats_data)} 个数据点")
    
    # 按噪声类型汇总
    print(f"\n按噪声类型汇总:")
    noise_counts = defaultdict(int)
    for row in stats_data:
        noise_counts[row['noise_name']] += 1
    for noise_name in sorted(noise_counts.keys()):
        print(f"  {noise_name}: {noise_counts[noise_name]} 个数据点")

if __name__ == "__main__":
    import sys
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        version = sys.argv[1]
        if version not in ["comp", "no_comp"]:
            print("Usage: python analyze_output_stats.py [comp|no_comp]")
            print("默认处理 comp 版本")
            sys.exit(1)
    else:
        version = "comp"
    
    # 如果指定了no_comp，只处理no_comp；否则处理comp
    if version == "no_comp":
        main("no_comp")
    else:
        main("comp")

