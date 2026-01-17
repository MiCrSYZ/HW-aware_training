"""
提取所有noise_boundary文件夹中的grad_norm和grad_norm_std数据
"""
import os
import csv
import json
from pathlib import Path

# 定义要处理的文件夹列表
NOISE_FOLDERS = [
    "noise_boundary_adc_direct",
    "noise_boundary_adc_ste",
    "noise_boundary_drift_alpha",
    "noise_boundary_ir_drop",
    "noise_boundary_ir_drop_beta",
    "noise_boundary_read_noise_sigma",
    "noise_boundary_stuck_ratio",
    "noise_boundary_variability_sigma",
]

OUTPUT_DIR = Path("output")

def extract_noise_strength(folder_name, noise_type):
    """从文件夹名中提取噪声强度值"""
    # 移除噪声类型前缀
    prefix_map = {
        "noise_boundary_adc_direct": "adc_bits_",
        "noise_boundary_adc_ste": "adc_bits_",
        "noise_boundary_drift_alpha": "drift_alpha_",
        "noise_boundary_ir_drop": "ir_drop_scaling_",
        "noise_boundary_ir_drop_beta": "ir_drop_beta_",
        "noise_boundary_read_noise_sigma": "read_noise_sigma_",
        "noise_boundary_stuck_ratio": "stuck_ratio_",
        "noise_boundary_variability_sigma": "variability_sigma_",
    }
    
    prefix = prefix_map.get(noise_type, "")
    if prefix and folder_name.startswith(prefix):
        value_str = folder_name[len(prefix):]
        try:
            # 尝试转换为浮点数
            return float(value_str)
        except:
            return value_str
    return folder_name

def read_metrics_csv(csv_path):
    """读取metrics.csv文件，返回grad_norm和grad_norm_std的列表"""
    try:
        grad_norms = []
        grad_norm_stds = []
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'grad_norm' in row and 'grad_norm_std' in row:
                    try:
                        grad_norms.append(float(row['grad_norm']))
                        grad_norm_stds.append(float(row['grad_norm_std']))
                    except ValueError:
                        continue
        
        if not grad_norms:
            print(f"Warning: {csv_path} 没有有效数据")
            return None, None
        
        return grad_norms, grad_norm_stds
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None, None

def collect_all_data():
    """收集所有文件夹中的数据"""
    all_data = []
    
    for noise_folder in NOISE_FOLDERS:
        folder_path = OUTPUT_DIR / noise_folder
        
        if not folder_path.exists():
            print(f"Warning: {folder_path} 不存在")
            continue
        
        # 遍历该文件夹下的所有子文件夹
        for subfolder in folder_path.iterdir():
            if not subfolder.is_dir():
                continue
            
            # 跳过一些特殊文件夹
            if subfolder.name.endswith('.png') or subfolder.name.endswith('.json'):
                continue
            
            metrics_file = subfolder / "metrics.csv"
            if not metrics_file.exists():
                print(f"Warning: {metrics_file} 不存在")
                continue
            
            # 提取噪声强度
            noise_strength = extract_noise_strength(subfolder.name, noise_folder)
            
            # 读取metrics数据
            grad_norms, grad_norm_stds = read_metrics_csv(metrics_file)
            
            if grad_norms is None or grad_norm_stds is None:
                continue
            
            # 计算平均值
            avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else None
            avg_grad_norm_std = sum(grad_norm_stds) / len(grad_norm_stds) if grad_norm_stds else None
            
            # 计算标准差
            if len(grad_norms) > 1:
                mean_grad_norm = avg_grad_norm
                mean_grad_norm_std = avg_grad_norm_std
                variance_grad_norm = sum((x - mean_grad_norm) ** 2 for x in grad_norms) / len(grad_norms)
                variance_grad_norm_std = sum((x - mean_grad_norm_std) ** 2 for x in grad_norm_stds) / len(grad_norm_stds)
                std_grad_norm = variance_grad_norm ** 0.5
                std_grad_norm_std = variance_grad_norm_std ** 0.5
            else:
                std_grad_norm = 0.0
                std_grad_norm_std = 0.0
            
            all_data.append({
                'noise_type': noise_folder,
                'noise_strength': noise_strength,
                'subfolder': subfolder.name,
                'avg_grad_norm': avg_grad_norm,
                'avg_grad_norm_std': avg_grad_norm_std,
                'std_grad_norm': std_grad_norm,
                'std_grad_norm_std': std_grad_norm_std,
                'num_epochs': len(grad_norms),
                # 也保存所有epoch的数据（作为列表）
                'grad_norm_all': grad_norms,
                'grad_norm_std_all': grad_norm_stds,
            })
    
    return all_data

def save_to_csv(data, output_path):
    """保存为CSV格式（不包含列表列）"""
    if not data:
        return
    
    # 准备CSV数据（排除列表列）
    csv_data = []
    for item in data:
        csv_item = {k: v for k, v in item.items() if not k.endswith('_all')}
        csv_data.append(csv_item)
    
    # 获取所有字段名
    fieldnames = csv_data[0].keys()
    
    with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_data:
            # 转换浮点数为字符串，处理None值
            csv_row = {}
            for k, v in row.items():
                if v is None:
                    csv_row[k] = ''
                elif isinstance(v, float):
                    csv_row[k] = v
                else:
                    csv_row[k] = v
            writer.writerow(csv_row)
    
    print(f"CSV数据已保存到: {output_path}")

def save_to_json(data, output_path):
    """保存为JSON格式（包含所有数据）"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"JSON数据已保存到: {output_path}")

def main():
    print("开始提取grad_norm和grad_norm_std数据...")
    
    # 收集所有数据
    all_data = collect_all_data()
    
    if not all_data:
        print("未找到任何数据！")
        return
    
    print(f"共收集到 {len(all_data)} 条数据")
    
    # 保存为CSV
    csv_path = OUTPUT_DIR / "grad_norm_summary.csv"
    save_to_csv(all_data, csv_path)
    
    # 保存为JSON
    json_path = OUTPUT_DIR / "grad_norm_summary.json"
    save_to_json(all_data, json_path)
    
    # 打印摘要
    print("\n数据摘要:")
    print(f"共 {len(NOISE_FOLDERS)} 种噪声类型")
    for noise_type in NOISE_FOLDERS:
        count = sum(1 for item in all_data if item['noise_type'] == noise_type)
        print(f"  {noise_type}: {count} 个噪声强度")

if __name__ == "__main__":
    main()

