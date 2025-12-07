# Learned Mapping 数值修复

## 问题描述

在 learned mapping 实现中，当对权重应用 `learned_scale` 和 `learned_offset` 后，`max_abs` 会改变，导致 scale recovery 基于错误的量级，最终导致输出量级错误，准确率降至 ~10%。

## 根本原因

1. **错误的 scale recovery**：
   - 在 `MemristorLinear.forward()` 中，我们先对权重应用 `W = W * learned_scale + learned_offset`
   - 然后传递给 `hardware_linear_forward_adaptive`
   - `hardware_linear_forward_adaptive` 计算 `max_abs = W.abs().max()`（基于修改后的权重）
   - 使用 `scale = max_abs / (G_max - G_min)` 进行 scale recovery
   - **问题**：如果 `learned_scale` 改变了权重量级，`max_abs` 也会改变，scale recovery 就会基于错误的量级

2. **输出量级错误**：
   - 如果 `learned_scale` 很大，`max_abs` 会变大，`scale` 也会变大
   - 如果 `learned_scale` 很小，`max_abs` 会变小，`scale` 也会变小
   - 这导致 `W_eff = W_eff_conductance * scale` 的量级错误
   - 最终输出量级错误，softmax 失效，准确率降至 ~10%

## 修复方案

### 核心思想

**Learned mapping 应该影响权重到电导的映射过程，但不应该破坏 scale recovery。**

### 实现细节

1. **保存原始 max_abs**：
   - 在应用 learned mapping 之前，计算并保存原始权重的 `max_abs_original`
   - 这确保了 scale recovery 始终基于原始权重的量级

2. **应用 learned mapping**：
   - 对权重应用 `learned_scale` 和 `learned_offset`，得到 `W_mapped`
   - `W_mapped` 用于电导映射，影响非理想性的应用

3. **使用原始 max_abs 进行 scale recovery**：
   - 在 `hardware_linear_forward_adaptive_with_learned_mapping` 中
   - 使用 `max_abs_original` 而不是 `max_abs_mapped` 来计算 scale
   - 这确保了输出量级始终正确

### 代码修改

**文件**: `src/models/memristor_wrappers.py`

1. **新增函数**: `hardware_linear_forward_adaptive_with_learned_mapping`
   - 接受 `max_abs_original` 参数
   - 使用原始 max_abs 进行 scale recovery

2. **修改 `MemristorLinear.forward()`**:
   ```python
   # 计算原始 max_abs（在应用 learned mapping 之前）
   max_abs_original = W_original.abs().max().clamp_min(1e-12)
   
   # 应用 learned mapping
   W_mapped = W_original * learned_scale + learned_offset
   
   # 使用原始 max_abs 进行 scale recovery
   out = hardware_linear_forward_adaptive_with_learned_mapping(
       x, W_mapped, device_model, max_abs_original=max_abs_original
   )
   ```

3. **修改 `MemristorConv2d.forward()`**:
   - 同样的修复逻辑

## 修复效果

修复后：
- ✅ Learned mapping 可以影响权重到电导的映射（通过 `W_mapped`）
- ✅ Scale recovery 始终基于原始权重量级（通过 `max_abs_original`）
- ✅ 输出量级保持正确，不会因为 learned_scale 而改变
- ✅ Softmax 正常工作，准确率不再降至 ~10%

## 技术细节

### Scale Recovery 公式

```
W_eff = (G_pos_noisy - G_neg_noisy) * scale
其中 scale = max_abs_original / (G_max - G_min)
```

**关键点**：
- `max_abs_original` 是原始权重的 max_abs，不受 learned mapping 影响
- `G_pos_noisy - G_neg_noisy` 是基于 `W_mapped` 计算的（受 learned mapping 影响）
- 最终的 `W_eff` 量级正确，因为 scale 基于原始权重

### 映射流程

```
原始权重 W_original
  ↓
计算 max_abs_original = |W_original|.max()
  ↓
应用 learned mapping: W_mapped = W_original * scale + offset
  ↓
映射到电导: G_pos, G_neg = map(W_mapped)
  ↓
应用非理想性: G_pos_noisy, G_neg_noisy
  ↓
计算电导差: W_eff_conductance = G_pos_noisy - G_neg_noisy
  ↓
Scale recovery: W_eff = W_eff_conductance * (max_abs_original / G_range)
  ↓
线性操作: output = F.linear(x, W_eff)
```

## 验证

修复后，learned mapping 应该：
1. 能够学习到有效的映射参数（scale, offset）
2. 输出量级与 baseline 相当（在 1-2 个数量级内）
3. 准确率显著高于 ~10%
4. 能够有效补偿非理想性的影响

## 注意事项

- 这个修复确保了 learned mapping 不会破坏数值正确性
- Learned mapping 仍然可以通过改变权重到电导的映射来优化性能
- 但输出量级始终基于原始权重，保持数值稳定性

