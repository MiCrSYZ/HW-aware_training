# 忆阻器器件模型升级说明

## 概述

本次升级在 `src/memristor/device_model.py` 中添加了以下新功能：

1. **状态依赖写入更新模型** - 实现真实的器件写入行为
2. **脉冲序列写入模型** - 支持多脉冲累积更新
3. **ADC量化** - 模拟模数转换器的量化效应
4. **阵列规模tiling** - 支持大规模矩阵的分块计算
5. **能耗估计** - 统计写入和读出能耗

所有新功能都通过配置开关控制，保持与现有代码的完全兼容。

## 新增功能详解

### 1. 状态依赖写入更新模型

#### 函数：`write_update()`

实现状态依赖的电导更新：

- **Potentiation（增强）**：
  ```
  ΔG_pot = A⁺ · (1 − G/G_max)^(p⁺) · (1 − exp(−γ V t)) + noise
  ```

- **Depression（减弱）**：
  ```
  ΔG_dep = −A⁻ · ((G/G_min − 1))^(p⁻) · (1 − exp(−γ V t)) + noise
  ```

**参数**：
- `G`: 当前电导值
- `pulse_V`: 脉冲电压
- `pulse_t`: 脉冲宽度
- `direction`: 方向（>0为potentiation，<=0为depression）

#### 函数：`write_pulse_train()`

支持脉冲序列写入，逐个应用脉冲并累积更新。

**参数**：
- `G`: 初始电导值
- `V_list`: 脉冲电压列表
- `t_list`: 脉冲宽度列表
- `direction_list`: 方向列表（可选）

### 2. ADC量化

#### 函数：`adc_quant()`

实现ADC量化：
1. 计算min-max范围
2. 缩放到 [0, 2^bits - 1]
3. 四舍五入
4. 缩放回原范围

**参数**：
- `x`: 输入张量
- `bits`: ADC位数（默认使用 `self.adc_bits`）
- `add_noise`: 是否添加量化噪声

### 3. 阵列规模Tiling

#### 函数：`matmul_with_tiling()`

将大矩阵按 `array_size` 分块计算：
- 每个tile独立计算
- Tile输出经过ADC量化
- 多tile结果累加

**参数**：
- `x`: 输入张量 [batch, in_dim]
- `W`: 权重矩阵 [out_dim, in_dim]
- `adc_bits`: ADC位数（可选）

### 4. 能耗估计

#### 写入能耗
```
E_write = Σ (α · V² · t)
```
每个写入脉冲累加。

#### 读出能耗
```
E_read = num_tiles · β · 2^bits
```
每次矩阵乘法累加。

#### 能耗统计
- `self.energy_stats['write']`: 累积写入能耗
- `self.energy_stats['read']`: 累积读出能耗
- `reset_energy_stats()`: 重置统计
- `get_energy_stats()`: 获取当前统计

## 配置参数

### 新增配置项

在 `configs/*.yaml` 的 `memristor` 部分添加：

```yaml
memristor:
  # 阵列规模和ADC
  array_size: 128  # 忆阻器阵列规模（tile大小）
  adc_bits: 6  # ADC量化位数
  
  # 功能开关
  enable_update_model: true  # 启用状态依赖写入模型
  enable_adc: true  # 启用ADC量化
  enable_energy: true  # 启用能耗估计
  
  # 写入更新模型参数
  update_params:
    A_plus: 1e-5  # Potentiation幅度系数
    A_minus: 1e-5  # Depression幅度系数
    p_plus: 1.0  # Potentiation非线性指数
    p_minus: 1.0  # Depression非线性指数
    gamma: 1.0  # 电压-时间耦合系数
    write_noise_ratio: 0.05  # 写入噪声比例
  
  # 能耗系数
  energy_coefs:
    alpha: 1.0  # 写入能耗系数
    beta: 1.0  # 读出能耗系数
```

### 默认值

如果不提供新参数，使用以下默认值：
- `array_size`: 128
- `adc_bits`: 6
- `enable_update_model`: False
- `enable_adc`: False
- `enable_energy`: False
- `update_params`: 见代码中的默认值
- `energy_coefs`: `{'alpha': 1.0, 'beta': 1.0}`

## 使用示例

### 1. 基本使用（向后兼容）

```python
from src.memristor.device_model import MemristorDeviceModel

# 使用默认参数（新功能关闭）
device_model = MemristorDeviceModel(
    G_min=1e-6,
    G_max=1e-4,
)
```

### 2. 启用新功能

```python
# 启用所有新功能
device_model = MemristorDeviceModel(
    G_min=1e-6,
    G_max=1e-4,
    array_size=128,
    adc_bits=6,
    enable_update_model=True,
    enable_adc=True,
    enable_energy=True,
    update_params={
        'A_plus': 1e-5,
        'A_minus': 1e-5,
        'p_plus': 1.0,
        'p_minus': 1.0,
        'gamma': 1.0,
        'write_noise_ratio': 0.05,
    },
    energy_coefs={
        'alpha': 1.0,
        'beta': 1.0,
    }
)
```

### 3. 写入更新示例

```python
# 创建初始电导值
G = torch.ones(10, 10) * 5e-5

# 单个脉冲更新
pulse_V = torch.ones(10, 10) * 1.0  # 1V
pulse_t = torch.ones(10, 10) * 1e-3  # 1ms
direction = torch.ones(10, 10)  # Potentiation

G_new = device_model.write_update(G, pulse_V, pulse_t, direction)

# 脉冲序列更新
V_list = [torch.ones(10, 10) * 1.0, torch.ones(10, 10) * 1.2]
t_list = [torch.ones(10, 10) * 1e-3, torch.ones(10, 10) * 1e-3]
direction_list = [torch.ones(10, 10), -torch.ones(10, 10)]

G_final = device_model.write_pulse_train(G, V_list, t_list, direction_list)
```

### 4. ADC量化示例

```python
# 量化数据
x = torch.randn(10, 20) * 10.0
x_quant = device_model.adc_quant(x, bits=6)
```

### 5. Tiling示例

```python
# 大矩阵乘法（自动使用tiling）
x = torch.randn(4, 1000)
W = torch.randn(800, 1000)
y = device_model.matmul_with_tiling(x, W)
```

### 6. 能耗统计示例

```python
# 重置统计
device_model.reset_energy_stats()

# 执行写入和读取操作
# ...

# 获取能耗统计
energy_stats = device_model.get_energy_stats()
print(f"写入能耗: {energy_stats['write']:.6e}")
print(f"读出能耗: {energy_stats['read']:.6e}")
```

## 集成到训练流程

新功能已自动集成到训练和评估流程中：

1. **配置文件**：在 `configs/*.yaml` 中添加新参数
2. **自动加载**：`run_experiment.py` 和 `eval.py` 会自动读取新参数
3. **自动应用**：在 `learned_weight_mapping.py` 中自动使用tiling和ADC

## 测试

运行测试脚本验证功能：

```bash
python test_device_model_upgrade.py
```

测试内容包括：
1. 状态依赖写入更新模型
2. 脉冲序列写入模型
3. ADC量化功能
4. 阵列规模tiling功能
5. 能耗估计功能
6. 配置兼容性

## 修改的文件

1. **src/memristor/device_model.py**
   - 添加新参数到 `__init__`
   - 添加 `write_update()` 函数
   - 添加 `write_pulse_train()` 函数
   - 添加 `adc_quant()` 函数
   - 添加 `matmul_with_tiling()` 函数
   - 添加能耗统计功能
   - 更新 `save_state()` 和 `load_state()`

2. **src/memristor/learned_weight_mapping.py**
   - 在 `hardware_linear_forward_with_weight_mapping()` 中集成tiling和ADC

3. **src/experiments/run_experiment.py**
   - 更新 `MemristorDeviceModel` 初始化以支持新参数

4. **src/eval.py**
   - 更新 `MemristorDeviceModel` 初始化以支持新参数

5. **configs/default.yaml**
   - 添加新配置参数示例

6. **configs/resnet20_memristor_enhanced.yaml**（新建）
   - 完整的新功能配置示例

## 向后兼容性

- 所有新参数都有默认值
- 现有配置文件无需修改即可继续使用
- 新功能默认关闭，不影响现有行为
- `save_state()` 和 `load_state()` 支持旧格式

## 注意事项

1. **性能影响**：
   - Tiling会增加计算开销（特别是小矩阵）
   - ADC量化会引入数值误差
   - 能耗统计会增加少量开销

2. **数值稳定性**：
   - 写入更新模型包含指数函数，注意数值范围
   - ADC量化使用min-max scaling，注意极端值

3. **梯度流**：
   - 写入更新和ADC量化保持梯度流（如果输入需要梯度）
   - 能耗统计使用 `detach()`，不影响梯度

## 未来改进方向

1. 支持更复杂的写入模型（如多状态模型）
2. 支持更精细的ADC模型（如非均匀量化）
3. 支持更详细的能耗模型（如考虑工艺节点）
4. 支持并行tiling（多GPU）

