import os
import csv
import json
from pathlib import Path
from collections import defaultdict

def extract_grad_metrics(output_dir="output/CIFAR-100"):
    """
    从output/CIFAR-100目录下的所有metrics.csv文件中提取grad_norm, grad_norm_std, grad_var
    """
    base_dir = Path(output_dir)
    all_data = []
    
    # 遍历所有噪声类型目录
    for noise_type_dir in sorted(base_dir.iterdir()):
        if not noise_type_dir.is_dir():
            continue
        
        noise_type = noise_type_dir.name
        
        # 遍历每个噪声类型的强度子目录
        for intensity_dir in sorted(noise_type_dir.iterdir()):
            if not intensity_dir.is_dir():
                continue
            
            intensity = intensity_dir.name
            metrics_file = intensity_dir / "metrics.csv"
            
            if not metrics_file.exists():
                print(f"警告: {metrics_file} 不存在，跳过")
                continue
            
            # 读取metrics.csv文件
            try:
                with open(metrics_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        data_entry = {
                            'noise_type': noise_type,
                            'intensity': intensity,
                            'epoch': int(row['epoch']),
                            'grad_norm': float(row['grad_norm']),
                            'grad_norm_std': float(row['grad_norm_std']),
                            'grad_var': float(row['grad_var'])
                        }
                        all_data.append(data_entry)
            except Exception as e:
                print(f"错误: 读取 {metrics_file} 时出错: {e}")
                continue
    
    return all_data

def save_to_csv(data, output_file="grad_metrics.csv"):
    """保存数据到CSV文件"""
    if not data:
        print("没有数据可保存")
        return
    
    fieldnames = ['noise_type', 'intensity', 'epoch', 'grad_norm', 'grad_norm_std', 'grad_var']
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    
    print(f"已保存 {len(data)} 条记录到 {output_file}")

def save_to_json(data, output_file="grad_metrics.json"):
    """保存数据到JSON文件"""
    if not data:
        print("没有数据可保存")
        return
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"已保存 {len(data)} 条记录到 {output_file}")

if __name__ == "__main__":
    print("开始提取梯度指标数据...")
    data = extract_grad_metrics()
    
    if data:
        print(f"共提取了 {len(data)} 条记录")
        
        # 保存为CSV和JSON格式
        save_to_csv(data, "grad_metrics.csv")
        save_to_json(data, "grad_metrics.json")
        
        # 打印统计信息
        noise_types = set(d['noise_type'] for d in data)
        print(f"\n找到的噪声类型: {len(noise_types)}")
        for noise_type in sorted(noise_types):
            intensities = set(d['intensity'] for d in data if d['noise_type'] == noise_type)
            print(f"  {noise_type}: {len(intensities)} 个强度级别")
    else:
        print("未找到任何数据")

