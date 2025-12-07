# 数值正确性修复总结

## 问题描述

原始代码存在严重的数值尺度问题：
- 权重 W ∈ [-1, 1] 被映射到电导范围 [G_min, G_max]，其中 G_min ~ 1e-6，G_max ~ 1e-4
- 这导致映射后的电导值非常小（~1e-6 到 ~1e-5），比浮点权重小很多数量级
- 结果：线性输出约为 ~1e-5，而浮点输出约为 ~1e-1 或更高
- Softmax 变得几乎均匀 → 准确率 ≈ 1/num_classes ≈ 10%
- 硬件感知训练（HAT）无法修复，因为梯度信号太小

## 修复内容

### 1. 自适应差分映射 (`map_weights_to_conductance_diff_adaptive`)

**文件**: `src/memristor/device_model.py`

实现了每层自适应归一化的差分对映射：
- 将权重限制到 [wmin, wmax]
- 计算每层的 max_abs = max(|W|)
- 归一化正部分：W_pos = max(W, 0) / max_abs
- 归一化负部分：W_neg = max(-W, 0) / max_abs
- 将两者映射到完整电导范围：G = G_min + (W_norm * (G_max - G_min))
- 返回 (G_pos, G_neg, max_abs)

这确保了完整的权重范围映射到完整的电导范围，避免了小电导值导致输出幅度过小的问题。

### 2. 硬件前向传播（自适应）(`hardware_linear_forward_adaptive`)

**文件**: `src/models/memristor_wrappers.py`

实现了带自适应差分映射和尺度恢复的硬件感知线性前向传播：
- 使用 `map_weights_to_conductance_diff_adaptive` 进行映射
- 对正负电导分别应用非理想性
- 计算有效电导差：W_eff_conductance = G_pos_noisy - G_neg_noisy
- 使用尺度恢复：scale = max_abs / (G_max - G_min)，限制在 [1e-3, 1e6]
- 计算有效权重：W_eff = W_eff_conductance * scale
- 执行线性操作：F.linear(x, W_eff)
- **确保所有操作都保持梯度（无 detach 调用）**

### 3. 更新包装器类

**文件**: `src/models/memristor_wrappers.py`

- `MemristorLinear.forward()`: 现在使用 `hardware_linear_forward_adaptive`
- `MemristorConv2d.forward()`: 现在使用 `hardware_linear_forward_adaptive`

### 4. 数值正确性检查脚本

**文件**: `src/utils/sanity_check.py`

创建了 `sanity_check_layer()` 函数来验证：
1. 硬件感知前向传播产生的输出与浮点前向传播的输出幅度相当
2. 梯度被正确计算（未分离）
3. 自适应映射和尺度恢复正常工作

检查包括：
- 输出均值/标准差比较
- 输出幅度合理性检查
- 梯度存在性和非零性验证

## 修改的文件

1. **src/memristor/device_model.py**
   - 完善了 `map_weights_to_conductance_diff_adaptive` 的文档和实现

2. **src/models/memristor_wrappers.py**
   - 重写并完善了 `hardware_linear_forward_adaptive` 函数
   - 更新了 `MemristorLinear.forward()` 使用新的自适应函数
   - 更新了 `MemristorConv2d.forward()` 使用新的自适应函数

3. **src/utils/sanity_check.py** (新建)
   - 实现了 `sanity_check_layer()` 函数用于数值正确性验证

## 关键改进

1. **尺度恢复**: 通过 `scale = max_abs / (G_max - G_min)` 恢复正确的权重幅度
2. **自适应归一化**: 每层使用 max_abs 进行归一化，确保完整权重范围映射到完整电导范围
3. **梯度保持**: 所有操作都保持梯度，确保硬件感知训练可以正常工作
4. **数值稳定性**: scale 被限制在 [1e-3, 1e6] 范围内，避免数值问题

## 预期效果

修复后：
- 硬件前向传播的输出幅度应该与浮点前向传播的输出幅度相当（在 1-2 个数量级内）
- Softmax 应该能够正常工作，准确率应该显著提高（不再是 ~10%）
- 硬件感知训练应该能够有效工作，因为梯度信号现在具有正确的幅度
- 补偿和非补偿配置应该产生不同的结果（之前它们产生相同的结果，因为映射被破坏）

## 使用方法

运行数值正确性检查：
```bash
python run_sanity_check.py
```

或者直接导入使用：
```python
from src.utils.sanity_check import sanity_check_layer
success, stats = sanity_check_layer(verbose=True)
```

## 技术细节

### 尺度恢复公式

```
W_eff = (G_pos_noisy - G_neg_noisy) * scale
其中 scale = max_abs / (G_max - G_min)
```

这确保了：
- 当 max_abs = 1.0，G_range = 1e-4 - 1e-6 ≈ 1e-4 时，scale ≈ 1e4
- W_eff 的幅度约为 1e-4 * 1e4 = 1.0，与原始权重幅度匹配

### 映射流程

```
原始权重 W → 限制到 [wmin, wmax] → 计算 max_abs
→ 归一化 W_pos, W_neg → 映射到 [G_min, G_max]
→ 应用非理想性 → 计算电导差 → 应用尺度恢复 → 线性操作
```

## 注意事项

- 旧的 `hardware_linear_forward` 函数仍然存在，但不再被使用（它调用了不存在的 `map_weights_to_conductance_diff` 方法）
- 所有新的代码路径都使用 `hardware_linear_forward_adaptive`
- 确保在训练和评估中都使用修复后的代码

