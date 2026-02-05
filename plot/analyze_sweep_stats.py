#!/usr/bin/env python3
"""
分析output目录中三组seeds数据的统计分析
计算test acc和val acc的h-mean和std，排除只在seed42中存在的噪声类型
"""

import json
import re
from collections import defaultdict
from pathlib import Path
import numpy as np
import csv

def harmonic_mean(values):
    """计算调和平均数"""
    values = [v for v in values if not (np.isnan(v) or np.isinf(v))]
    if len(values) == 0:
        return np.nan
    # 过滤掉0值
    non_zero_values = [v for v in values if v != 0]
    if len(non_zero_values) == 0:
        return np.nan
    return len(non_zero_values) / sum(1.0 / v for v in non_zero_values)

def normalize_noise_name(name):
    """标准化噪声类型名称，处理不同格式"""
    # 将不同格式的名称统一
    name_mapping = {
        "ADC_direct": "adc_direct",
        "ADC_STE": "adc_ste",
        "IR_drop_paper": "ir_drop",
        "IR_drop_crossbar": "ir_drop_beta",
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
        # 修复常见的JSON格式错误
        # 1. 移除尾随逗号（在}之前）
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        # 2. 修复缺少右方括号的情况：在数字后直接跟着"test_accuracies"
        # 模式：数字 + 换行/空格 + "test_accuracies"
        json_str = re.sub(r'(\d+\.?\d*)\s+\n\s*"test_accuracies"', r'\1\n  ],\n  "test_accuracies"', json_str)
        json_str = re.sub(r'(\d+\.?\d*)\s+"test_accuracies"', r'\1\n  ],\n  "test_accuracies"', json_str)
        
        try:
            data = json.loads(json_str)
            results[noise_name] = data
        except json.JSONDecodeError as e:
            # 如果还是失败，尝试更精确的修复
            try:
                # 查找 "val_accuracies": [ ... 数字 "test_accuracies" 的模式
                # 在最后一个数字和"test_accuracies"之间插入 ],\n  "
                pattern = r'("val_accuracies":\s*\[[^\]]*?)(\d+\.?\d*)\s+("test_accuracies")'
                def fix_match(m):
                    return m.group(1) + m.group(2) + '\n  ],\n  ' + m.group(3)
                fixed_json = re.sub(pattern, fix_match, json_str, flags=re.DOTALL)
                data = json.loads(fixed_json)
                results[noise_name] = data
            except Exception as e2:
                print(f"Warning: Failed to parse JSON for {noise_name} in {filepath}: {e}")
                continue
    
    return results

def main():
    """
    分析output目录中三组seeds数据的统计分析
    处理所有6个文件（comp和no_comp版本）
    """
    # 文件路径
    base_dir = Path("output")
    seeds = [26, 3407, 42]
    versions = ["comp", "no_comp"]
    
    all_results = {}
    
    for version in versions:
        print(f"\n正在分析 {version} 版本的数据...")
        
        # 根据版本选择文件名模式
        if version == "no_comp":
            file_pattern = "noise_boundary_sweep_no_comp_seed{}.txt"
        else:
            file_pattern = "noise_boundary_sweep_seed{}.txt"
        
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
            noise_type = first_seed_data.get('noise_type', '')
            
            # 对每个noise_strength收集所有seeds的val_acc和test_acc
            for i, strength in enumerate(noise_strengths):
                val_accs = []
                test_accs = []
                
                for seed in seeds:
                    if seed in noise_data_by_seed:
                        seed_data = noise_data_by_seed[seed]
                        val_acc_list = seed_data.get('val_accuracies', [])
                        test_acc_list = seed_data.get('test_accuracies', [])
                        
                        if i < len(val_acc_list):
                            val_accs.append(val_acc_list[i])
                        if i < len(test_acc_list):
                            test_accs.append(test_acc_list[i])
                
                if len(val_accs) >= 2:  # 至少需要2个数据点
                    # 计算val_acc统计量
                    val_acc_hmean = harmonic_mean(val_accs)
                    val_acc_std = np.std(val_accs) if len(val_accs) > 1 else 0.0
                    val_acc_mean = np.mean(val_accs)
                else:
                    val_acc_hmean = np.nan
                    val_acc_std = np.nan
                    val_acc_mean = np.nan
                
                if len(test_accs) >= 2:  # 至少需要2个数据点
                    # 计算test_acc统计量
                    test_acc_hmean = harmonic_mean(test_accs)
                    test_acc_std = np.std(test_accs) if len(test_accs) > 1 else 0.0
                    test_acc_mean = np.mean(test_accs)
                else:
                    test_acc_hmean = np.nan
                    test_acc_std = np.nan
                    test_acc_mean = np.nan
                
                stats_data.append({
                    'version': version,
                    'noise_name': noise_name,
                    'noise_type': noise_type,
                    'noise_strength': strength,
                    'num_seeds': max(len(val_accs), len(test_accs)),
                    'val_acc_mean': val_acc_mean,
                    'val_acc_hmean': val_acc_hmean,
                    'val_acc_std': val_acc_std,
                    'test_acc_mean': test_acc_mean,
                    'test_acc_hmean': test_acc_hmean,
                    'test_acc_std': test_acc_std,
                })
        
        all_results[version] = {
            'stats_data': stats_data,
            'only_in_seed42': list(only_in_seed42),
            'valid_noise_types': list(valid_noise_types),
        }
        
        print(f"共分析了 {len(valid_noise_types)} 种噪声类型")
        print(f"排除了 {len(only_in_seed42)} 种只在seed42中存在的噪声类型")
        print(f"共 {len(stats_data)} 个数据点")
    
    # 合并所有版本的数据
    all_stats_data = []
    for version in versions:
        all_stats_data.extend(all_results[version]['stats_data'])
    
    # 保存为CSV
    csv_path = base_dir / "sweep_statistics.csv"
    if all_stats_data:
        fieldnames = all_stats_data[0].keys()
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_stats_data:
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
    json_path = base_dir / "sweep_statistics.json"
    # 将numpy类型转换为Python原生类型以便JSON序列化
    json_data = {
        'summary': {
            'comp': {
                'valid_noise_types': all_results['comp']['valid_noise_types'],
                'only_in_seed42': all_results['comp']['only_in_seed42'],
                'num_data_points': len(all_results['comp']['stats_data']),
            },
            'no_comp': {
                'valid_noise_types': all_results['no_comp']['valid_noise_types'],
                'only_in_seed42': all_results['no_comp']['only_in_seed42'],
                'num_data_points': len(all_results['no_comp']['stats_data']),
            },
        },
        'data': []
    }
    
    for row in all_stats_data:
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
        json_data['data'].append(json_row)
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"统计结果已保存到: {json_path}")
    
    # 打印摘要
    print(f"\n总体统计摘要:")
    print(f"comp版本: {len(all_results['comp']['stats_data'])} 个数据点")
    print(f"no_comp版本: {len(all_results['no_comp']['stats_data'])} 个数据点")
    print(f"总计: {len(all_stats_data)} 个数据点")
    
    # 按噪声类型汇总
    print(f"\n按噪声类型汇总 (comp版本):")
    noise_counts = defaultdict(int)
    for row in all_results['comp']['stats_data']:
        noise_counts[row['noise_name']] += 1
    for noise_name in sorted(noise_counts.keys()):
        print(f"  {noise_name}: {noise_counts[noise_name]} 个数据点")
    
    print(f"\n按噪声类型汇总 (no_comp版本):")
    noise_counts = defaultdict(int)
    for row in all_results['no_comp']['stats_data']:
        noise_counts[row['noise_name']] += 1
    for noise_name in sorted(noise_counts.keys()):
        print(f"  {noise_name}: {noise_counts[noise_name]} 个数据点")

if __name__ == "__main__":
    main()

