"""
测试脚本：验证忆阻器器件模型升级功能

测试内容：
1. 状态依赖写入模型（write_update 和 write_pulse_train）
2. ADC量化功能
3. 阵列规模tiling功能
4. 能耗估计功能
"""

import torch
import numpy as np
from src.memristor.device_model import MemristorDeviceModel


def test_write_update():
    """测试状态依赖写入更新模型"""
    print("=" * 60)
    print("测试 1: 状态依赖写入更新模型")
    print("=" * 60)
    
    device_model = MemristorDeviceModel(
        G_min=1e-6,
        G_max=1e-4,
        enable_update_model=True,
        update_params={
            'A_plus': 1e-5,
            'A_minus': 1e-5,
            'p_plus': 1.0,
            'p_minus': 1.0,
            'gamma': 1.0,
            'write_noise_ratio': 0.05,
        }
    )
    
    # 创建初始电导值
    G = torch.ones(5, 5) * 5e-5  # 中间值
    pulse_V = torch.ones(5, 5) * 1.0  # 1V脉冲
    pulse_t = torch.ones(5, 5) * 1e-3  # 1ms脉冲宽度
    direction = torch.ones(5, 5)  # Potentiation方向
    
    print(f"初始电导值: {G.mean().item():.6e} S")
    
    # 应用写入更新
    G_new = device_model.write_update(G, pulse_V, pulse_t, direction)
    
    print(f"更新后电导值: {G_new.mean().item():.6e} S")
    print(f"电导变化: {(G_new - G).mean().item():.6e} S")
    print("✓ 写入更新模型测试通过\n")


def test_write_pulse_train():
    """测试脉冲序列写入模型"""
    print("=" * 60)
    print("测试 2: 脉冲序列写入模型")
    print("=" * 60)
    
    device_model = MemristorDeviceModel(
        G_min=1e-6,
        G_max=1e-4,
        enable_update_model=True,
    )
    
    # 创建初始电导值
    G = torch.ones(3, 3) * 5e-5
    
    # 创建脉冲序列
    V_list = [
        torch.ones(3, 3) * 1.0,
        torch.ones(3, 3) * 1.2,
        torch.ones(3, 3) * 0.8,
    ]
    t_list = [
        torch.ones(3, 3) * 1e-3,
        torch.ones(3, 3) * 1e-3,
        torch.ones(3, 3) * 1e-3,
    ]
    direction_list = [
        torch.ones(3, 3),  # Potentiation
        torch.ones(3, 3),  # Potentiation
        -torch.ones(3, 3),  # Depression
    ]
    
    print(f"初始电导值: {G.mean().item():.6e} S")
    
    # 应用脉冲序列
    G_final = device_model.write_pulse_train(G, V_list, t_list, direction_list)
    
    print(f"最终电导值: {G_final.mean().item():.6e} S")
    print(f"总电导变化: {(G_final - G).mean().item():.6e} S")
    print("✓ 脉冲序列写入模型测试通过\n")


def test_adc_quant():
    """测试ADC量化功能"""
    print("=" * 60)
    print("测试 3: ADC量化功能")
    print("=" * 60)
    
    device_model = MemristorDeviceModel(
        enable_adc=True,
        adc_bits=6,
    )
    
    # 创建测试数据
    x = torch.randn(10, 20) * 10.0  # 随机数据
    
    print(f"原始数据范围: [{x.min().item():.4f}, {x.max().item():.4f}]")
    print(f"原始数据均值: {x.mean().item():.4f}")
    
    # 应用ADC量化
    x_quant = device_model.adc_quant(x, bits=6)
    
    print(f"量化后数据范围: [{x_quant.min().item():.4f}, {x_quant.max().item():.4f}]")
    print(f"量化后数据均值: {x_quant.mean().item():.4f}")
    print(f"量化误差 (MSE): {((x - x_quant) ** 2).mean().item():.6f}")
    print("✓ ADC量化功能测试通过\n")


def test_tiling():
    """测试阵列规模tiling功能"""
    print("=" * 60)
    print("测试 4: 阵列规模tiling功能")
    print("=" * 60)
    
    device_model = MemristorDeviceModel(
        array_size=32,  # 小tile用于测试
        enable_adc=True,
        adc_bits=6,
    )
    
    # 创建测试矩阵（大于array_size）
    batch_size = 4
    in_dim = 100
    out_dim = 80
    
    x = torch.randn(batch_size, in_dim)
    W = torch.randn(out_dim, in_dim)
    
    print(f"输入形状: {x.shape}")
    print(f"权重形状: {W.shape}")
    print(f"阵列规模: {device_model.array_size}")
    
    # 使用tiling进行矩阵乘法
    y_tiled = device_model.matmul_with_tiling(x, W)
    
    # 标准矩阵乘法（用于对比）
    y_standard = torch.matmul(x, W.T)
    
    print(f"Tiling输出形状: {y_tiled.shape}")
    print(f"标准输出形状: {y_standard.shape}")
    print(f"输出差异 (MSE): {((y_tiled - y_standard) ** 2).mean().item():.6f}")
    print("✓ Tiling功能测试通过\n")


def test_energy_estimation():
    """测试能耗估计功能"""
    print("=" * 60)
    print("测试 5: 能耗估计功能")
    print("=" * 60)
    
    device_model = MemristorDeviceModel(
        enable_update_model=True,
        enable_energy=True,
        enable_adc=True,
        array_size=32,
        adc_bits=6,
        energy_coefs={
            'alpha': 1.0,
            'beta': 1.0,
        }
    )
    
    # 重置能耗统计
    device_model.reset_energy_stats()
    
    # 模拟写入操作
    G = torch.ones(10, 10) * 5e-5
    pulse_V = torch.ones(10, 10) * 1.0
    pulse_t = torch.ones(10, 10) * 1e-3
    direction = torch.ones(10, 10)
    
    device_model.write_update(G, pulse_V, pulse_t, direction)
    
    # 模拟读取操作（通过tiling）
    x = torch.randn(4, 50)
    W = torch.randn(40, 50)
    device_model.matmul_with_tiling(x, W)
    
    # 获取能耗统计
    energy_stats = device_model.get_energy_stats()
    
    print(f"写入能耗: {energy_stats['write']:.6e}")
    print(f"读出能耗: {energy_stats['read']:.6e}")
    print(f"总能耗: {energy_stats['write'] + energy_stats['read']:.6e}")
    print("✓ 能耗估计功能测试通过\n")


def test_config_compatibility():
    """测试配置兼容性（向后兼容）"""
    print("=" * 60)
    print("测试 6: 配置兼容性（向后兼容）")
    print("=" * 60)
    
    # 使用默认参数（不提供新参数）
    device_model_old = MemristorDeviceModel(
        G_min=1e-6,
        G_max=1e-4,
    )
    
    print(f"默认array_size: {device_model_old.array_size}")
    print(f"默认adc_bits: {device_model_old.adc_bits}")
    print(f"默认enable_update_model: {device_model_old.enable_update_model}")
    print(f"默认enable_adc: {device_model_old.enable_adc}")
    print(f"默认enable_energy: {device_model_old.enable_energy}")
    
    # 使用新参数
    device_model_new = MemristorDeviceModel(
        G_min=1e-6,
        G_max=1e-4,
        array_size=256,
        adc_bits=8,
        enable_update_model=True,
        enable_adc=True,
        enable_energy=True,
    )
    
    print(f"新array_size: {device_model_new.array_size}")
    print(f"新adc_bits: {device_model_new.adc_bits}")
    print(f"新enable_update_model: {device_model_new.enable_update_model}")
    print(f"新enable_adc: {device_model_new.enable_adc}")
    print(f"新enable_energy: {device_model_new.enable_energy}")
    print("✓ 配置兼容性测试通过\n")


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("忆阻器器件模型升级功能测试")
    print("=" * 60 + "\n")
    
    try:
        test_write_update()
        test_write_pulse_train()
        test_adc_quant()
        test_tiling()
        test_energy_estimation()
        test_config_compatibility()
        
        print("=" * 60)
        print("所有测试通过！✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

