# Hardware Error Input Dependence Diagnosis

## 概述

这个诊断工具用于分析硬件误差是否依赖于输入。如果误差强烈依赖于输入，那么静态权重映射 `W_final = f(W)` 可能不足以补偿硬件非理想性，特别是在 IR-drop、ADC 量化和强漂移的情况下。

## 使用方法

### 基本用法

```bash
python -m src.experiments.diagnose_input_dependence \
    --config configs/resnet20_memristor_learned_mapping.yaml \
    --checkpoint checkpoints/resnet20_memristor_learned_mapping/best.pth \
    --num_samples 20 \
    --batch_size 32 \
    --t 0 \
    --output_dir ./diagnosis_results
```

### 参数说明

- `--config`: 配置文件路径（YAML格式）
- `--checkpoint`: 训练好的模型检查点路径
- `--layer`: （可选）指定要分析的层名称。如果不指定，会自动选择一个中间层
- `--num_samples`: 采样的输入批次数量（默认：20）
- `--batch_size`: 每个批次的样本数（默认：32）
- `--t`: 漂移时间索引（默认：0）
- `--output_dir`: 输出目录（默认：./diagnosis_results）
- `--seed`: 随机种子（可选）

### 示例：分析特定层

```bash
python -m src.experiments.diagnose_input_dependence \
    --config configs/resnet20_memristor_learned_mapping.yaml \
    --checkpoint checkpoints/resnet20_memristor_learned_mapping/best.pth \
    --layer layer2.0.conv2 \
    --num_samples 20 \
    --output_dir ./diagnosis_results
```

### 示例：分析强漂移情况

```bash
python -m src.experiments.diagnose_input_dependence \
    --config configs/resnet20_memristor_learned_mapping.yaml \
    --checkpoint checkpoints/resnet20_memristor_learned_mapping/best.pth \
    --t 1000 \
    --num_samples 20 \
    --output_dir ./diagnosis_results_t1000
```

## 输出结果

### 1. 控制台输出

诊断工具会在控制台打印：

- **(A) 误差方差**：不同输入之间的误差方差
  - 高方差 → 强输入依赖性
  - 低方差 → 弱输入依赖性

- **(B) 误差相关性**：不同输入误差之间的平均相关性
  - 低相关性 (< 0.5) → 误差结构随输入变化
  - 高相关性 (> 0.5) → 误差结构一致

- **(C) 静态 ΔW 可转移性**：
  - 在一个输入上拟合静态 ΔW
  - 测试在其他输入上的残差
  - 高残差比 (> 0.5) → 静态 ΔW 不能很好地转移，可能需要输入相关的校准
  - 低残差比 (< 0.5) → 静态权重映射可能足够

### 2. 可视化结果

在输出目录中会生成以下可视化文件：

1. **`{layer_name}_error_heatmap.png`**
   - 不同输入的错误热图
   - X轴：输出维度
   - Y轴：输入批次索引
   - 颜色：误差值

2. **`{layer_name}_correlation_matrix.png`**
   - 误差相关性矩阵
   - 显示不同输入批次之间的误差相关性
   - 值范围：-1 到 1

3. **`{layer_name}_residuals.png`**
   - 静态 ΔW 残差分析
   - 左图：残差范数 vs 输入索引
   - 右图：残差/误差范数比率 vs 输入索引
   - 红色虚线：参考输入（用于拟合 ΔW）

## 诊断指标解释

### (A) 误差方差 (Error Variance)

```python
var_input = eps_stack.var(dim=0).mean()
```

- **高方差** (> 1e-4)：不同输入产生的误差差异很大，表明强输入依赖性
- **低方差** (< 1e-4)：不同输入的误差相似，表明弱输入依赖性

### (B) 误差相关性 (Error Correlation)

```python
corr = cosine_similarity(eps_i.flatten(), eps_j.flatten())
```

- **低相关性** (< 0.5)：误差模式随输入变化，静态映射可能不足
- **高相关性** (> 0.5)：误差模式一致，静态映射可能足够

### (C) 静态 ΔW 可转移性 (Static ΔW Transferability)

```python
# 在参考输入上拟合：x_ref @ ΔW.T ≈ eps_ref
DeltaW_hat = fit_static_deltaW(x_ref, eps_ref)

# 在其他输入上测试
eps_pred = F.linear(x_k, DeltaW_hat)
residual = (eps_k - eps_pred).norm() / eps_k.norm()
```

- **高残差比** (> 0.5)：静态 ΔW 不能很好地解释其他输入的误差
  - 表明需要输入相关的校准
  - 静态权重映射可能不足

- **低残差比** (< 0.5)：静态 ΔW 可以很好地解释其他输入的误差
  - 表明静态权重映射可能足够
  - 不需要输入相关的校准

## 解释结果

### 情况 1：弱输入依赖性
- 低误差方差
- 高误差相关性
- 低残差比

**结论**：静态权重映射 `W_final = f(W)` 可能足够。

### 情况 2：强输入依赖性
- 高误差方差
- 低误差相关性
- 高残差比

**结论**：可能需要输入相关的校准策略，静态权重映射可能不足。

## 注意事项

1. **层选择**：建议选择中间层（不是第一层或最后一层），因为这些层最能代表网络的典型行为。

2. **样本数量**：`num_samples` 建议设置为 10-20，足够进行统计分析但不会太慢。

3. **漂移时间**：使用 `--t` 参数可以测试不同漂移时间下的输入依赖性。强漂移（高 t 值）可能增加输入依赖性。

4. **设备配置**：确保配置文件中的设备模型参数（IR-drop、ADC 等）与评估时一致。

## 依赖项

- PyTorch
- NumPy
- Matplotlib
- Seaborn
- scikit-learn (用于相关性计算)

## 故障排除

### 问题：找不到指定的层

**解决方案**：使用 `--layer` 参数时，确保层名称完全匹配。可以先运行不带 `--layer` 的版本，查看自动选择的层名称。

### 问题：内存不足

**解决方案**：减少 `--batch_size` 或 `--num_samples`。

### 问题：可视化文件未生成

**解决方案**：确保安装了 matplotlib 和 seaborn：
```bash
pip install matplotlib seaborn
```
