# 噪声注入顺序说明

本文档详细说明训练和推理阶段中各种非理想性噪声的注入顺序。

## 训练阶段（training=True）

### Direct模式（enable_adc_during_training=True, adc_training_mode='direct'）

噪声注入顺序如下：

```
1. 权重映射到电导
   W (权重) → Gp, Gn (正负电导对)
   
2. 应用非理想性到电导层（apply_nonidealities）
   顺序：
   a) Stuck-at faults (卡位故障)
      - 固定部分单元到 G_min 或 G_max
      - 基于seed固定，保护这些单元不被后续噪声影响
   
   b) Variability (器件变异，乘性噪声)
      - G' = G * (1 + ε_v), ε_v ~ N(0, σ_v²)
      - 只应用于非stuck单元
   
   c) Read noise (读噪声，加性噪声)
      - G' = G + η, η ~ N(0, σ_r²)
      - 只应用于非stuck单元
   
   d) Drift (电导漂移，时间相关)
      - G_t = G * (1 - α * log(1 + t))
      - 只应用于非stuck单元
   
   e) IR-drop (simple模式，如果启用)
      - G' = G * (1 - β)
      - 只应用于非stuck单元
   
3. 计算有效权重
   W_eff = (Gp_noisy - Gn_noisy) * scale
   
4. 矩阵乘法（或tiling）
   out = W_eff @ x  (或使用 matmul_with_tiling)
   
5. ADC量化（direct模式，梯度会消失）
   out = adc_quant(out)
   - 使用 min-max scaling
   - 四舍五入量化
   - 梯度几乎为零（round()操作）
   
6. Paper版IR-drop（如果启用 enable_ir_drop_paper_during_training=True）
   - 归一化权重和输入
   - 应用IR-drop校正（基于论文方程16-18）
   - 转换回物理尺度
   - 注意：如果产生NaN/Inf，训练时会跳过以避免崩溃
```

### STE模式（enable_adc_during_training=True, adc_training_mode='ste'）

与direct模式相同，但第5步不同：

```
5. ADC量化（STE模式，保持梯度流）
   out_quantized = adc_quant(out)
   out = out + (out_quantized - out).detach()
   - 前向传播使用量化值
   - 反向传播使用原始梯度（通过detach实现）
```

### 不注入ADC（enable_adc_during_training=False）

```
1-4. 同direct模式（权重映射、非理想性、有效权重、矩阵乘法）

5. 跳过ADC量化（训练时不应用）

6. Paper版IR-drop（如果启用 enable_ir_drop_paper_during_training=True）
   - 同direct模式
```

## 推理阶段（training=False）

### 标准推理（训练时不注入ADC）

```
1-4. 同训练阶段（权重映射、非理想性、有效权重、矩阵乘法）

5. ADC量化（推理时总是应用，如果enable_adc=True）
   out = adc_quant(out)
   - 直接量化，不使用STE
   - 推理时不需要梯度

6. Paper版IR-drop（如果启用 ir_drop_mode='paper'）
   - 同训练阶段
   - 如果产生NaN/Inf，会进行数值稳定性处理（clamp等）
```

## 关键点说明

### 1. 非理想性应用顺序（apply_nonidealities）

在电导层应用的非理想性顺序是固定的：
1. **Stuck-at faults** 最先应用，保护stuck单元
2. **Variability** 和 **Read noise** 只影响非stuck单元
3. **Drift** 是时间相关的，只影响非stuck单元
4. **IR-drop (simple)** 是均匀缩放，只影响非stuck单元

### 2. ADC量化位置

- **位置**：在矩阵乘法之后，tile-sum之后
- **原因**：ADC在硬件中位于阵列输出之后，对tile求和后的结果进行量化
- **训练时**：根据 `enable_adc_during_training` 和 `adc_training_mode` 决定是否应用和如何应用
- **推理时**：如果 `enable_adc=True`，总是应用

### 3. Paper版IR-drop位置

- **位置**：在ADC量化之后
- **原因**：IR-drop影响的是模拟计算过程，但paper版的实现是在输出端进行校正
- **训练时**：根据 `enable_ir_drop_paper_during_training` 决定是否应用
- **推理时**：如果 `ir_drop_mode='paper'`，总是应用

### 4. 梯度流

- **Direct模式**：ADC量化使用 `round()`，梯度几乎为零，导致梯度消失
- **STE模式**：ADC量化使用 `detach()`，前向用量化值，反向用原始梯度，保持梯度流
- **不注入ADC**：训练时跳过ADC，梯度正常流动

## 示例对比

### 训练阶段 - Direct模式
```
输入 x → 权重映射 → 非理想性 → 有效权重 → 矩阵乘法 → ADC量化(direct) → IR-drop(可选) → 输出
                                                                    ↓
                                                              梯度消失
```

### 训练阶段 - STE模式
```
输入 x → 权重映射 → 非理想性 → 有效权重 → 矩阵乘法 → ADC量化(STE) → IR-drop(可选) → 输出
                                                                    ↓
                                                              梯度保持
```

### 推理阶段
```
输入 x → 权重映射 → 非理想性 → 有效权重 → 矩阵乘法 → ADC量化 → IR-drop(可选) → 输出
                                                          ↓
                                                    直接量化（无梯度）
```

## 注意事项

1. **数值稳定性**：Paper版IR-drop在训练时如果产生NaN/Inf会自动跳过，避免训练崩溃
2. **梯度保护**：所有非理想性应用都保持梯度流，只有direct模式的ADC量化会导致梯度消失
3. **Tiling**：如果使用tiling（array_size > 0），矩阵乘法在tile级别进行，ADC量化在tile-sum之后
4. **Stuck单元保护**：Stuck单元只受stuck故障影响，不受其他噪声影响
