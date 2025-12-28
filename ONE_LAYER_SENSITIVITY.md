# One-layer Learned Mapping Sensitivity Experiment

## 概述

这个诊断实验用于评估 learned weight mapping 的有效性是否强烈依赖具体层，以及输入依赖误差是否在某些层被显著放大。

## 实验设计

### 核心假设
如果 learned mapping 的失败是由于输入依赖的误差导致的，那么仅在特定层应用 learned mapping 应该显示出强烈的层依赖性能和误差统计。

### 实验模式
- **One-layer mapping**：仅在一个指定层启用 learned mapping，其他层强制关闭
- 支持逐层扫描（sweep），测试所有层或指定层列表
- 使用已经训练好的 mapping net，不重新训练
- Inference-only 实验，每次只改变 mapping 作用层

## 使用方法

### 基本用法

```bash
python -m src.experiments.one_layer_sensitivity \
    --config configs/resnet20_memristor_learned_mapping.yaml \
    --checkpoint checkpoints/resnet20_memristor_learned_mapping/best.pth \
    --mapping_net checkpoints/resnet20_memristor_learned_mapping/best.pth \
    --output_dir ./one_layer_sensitivity_results \
    --t 0
```

### 参数说明

- `--config`: 配置文件路径（YAML格式）
- `--checkpoint`: 训练好的模型检查点路径
- `--mapping_net`: Mapping net 检查点路径（可选，如果 checkpoint 中包含 mapping_net_state_dict 则不需要）
- `--output_dir`: 输出目录（默认：./one_layer_sensitivity_results）
- `--layers`: （可选）指定要测试的层名称列表，例如：`--layers layer1.0.conv1 layer1.0.conv2`
- `--num_samples`: 输入依赖误差分析的样本数量（默认：10）
- `--t`: 漂移时间索引（默认：0）
- `--no_input_dependence`: 跳过输入依赖误差计算（加快速度）

### 示例：测试特定层

```bash
python -m src.experiments.one_layer_sensitivity \
    --config configs/resnet20_memristor_learned_mapping.yaml \
    --checkpoint checkpoints/resnet20_memristor_learned_mapping/best.pth \
    --mapping_net checkpoints/resnet20_memristor_learned_mapping/best.pth \
    --layers layer1.0.conv1 layer1.0.conv2 layer2.0.conv1 \
    --output_dir ./results_layer1
```

### 示例：测试强漂移情况

```bash
python -m src.experiments.one_layer_sensitivity \
    --config configs/resnet20_memristor_learned_mapping.yaml \
    --checkpoint checkpoints/resnet20_memristor_learned_mapping/best.pth \
    --mapping_net checkpoints/resnet20_memristor_learned_mapping/best.pth \
    --t 1000 \
    --output_dir ./results_t1000
```

## 输出结果

### 1. CSV 结果表

`one_layer_sensitivity_results.csv` 包含以下列：

| Column | Description |
|--------|-------------|
| `layer` | 层名称 |
| `acc_one_layer` | 仅在该层启用 mapping 的测试准确率 |
| `delta_acc_vs_hat` | 相对于 HAT-only 的准确率变化 |
| `delta_acc_vs_full` | 相对于 Full mapping 的准确率变化（如果提供了 mapping_net） |
| `error_variance` | 输入依赖误差方差 |
| `mean_correlation` | 不同输入误差之间的平均相关性 |
| `static_deltaW_residual_ratio_mean` | 静态 ΔW 可转移性的平均残差比 |

### 2. JSON 结果

`one_layer_sensitivity_results.json` 包含：
- `baselines`: HAT-only 和 Full mapping 的基准准确率
- `results`: 每个层的详细结果

### 3. 可视化图表

- `one_layer_sensitivity_plots.png`: 
  - 准确率对比图
  - 准确率改进图
  - 误差方差图
  - 误差相关性图

- `one_layer_sensitivity_heatmap.png`: 
  - 层索引 vs 指标的归一化热图

## 评估指标解释

### (A) 性能指标

- **acc_one_layer**: 仅在该层启用 mapping 时的测试准确率
- **delta_acc_vs_hat**: 相对于 HAT-only 的准确率变化
  - 正值：mapping 在该层有效
  - 负值：mapping 在该层有害
- **delta_acc_vs_full**: 相对于 Full mapping 的准确率变化
  - 接近 0：该层对整体性能贡献大
  - 负值大：该层单独应用 mapping 效果差

### (B) 输入依赖误差指标

- **error_variance**: 不同输入之间的误差方差
  - 高方差 → 强输入依赖性
  - 低方差 → 弱输入依赖性

- **mean_correlation**: 不同输入误差之间的平均相关性
  - 低相关性 (< 0.5) → 误差结构随输入变化
  - 高相关性 (> 0.5) → 误差结构一致

- **static_deltaW_residual_ratio_mean**: 静态 ΔW 可转移性的平均残差比
  - 高残差比 (> 0.5) → 静态 ΔW 不能很好地转移
  - 低残差比 (< 0.5) → 静态 ΔW 可以很好地转移

## 结果解释

### Mapping-effective layers
- 高 `delta_acc_vs_hat`（显著提升准确率）
- 低 `error_variance`（弱输入依赖性）
- 高 `mean_correlation`（误差结构一致）

**结论**：这些层适合应用 learned mapping，静态权重映射可能足够。

### Mapping-fragile layers
- 低或负 `delta_acc_vs_hat`（mapping 无效或有害）
- 高 `error_variance`（强输入依赖性）
- 低 `mean_correlation`（误差结构随输入变化）
- 高 `static_deltaW_residual_ratio_mean`（静态 ΔW 不能转移）

**结论**：这些层可能需要输入相关的校准策略，静态权重映射可能不足。

## 注意事项

1. **这是诊断实验，不是优化实验**
   - 不引入新的 non-ideality
   - 不调整 mapping 超参数
   - 不进行 joint training
   - 使用已经训练好的 mapping net

2. **Mapping net 加载**
   - 如果 checkpoint 中包含 `mapping_net_state_dict`，会自动加载
   - 否则需要单独提供 `--mapping_net` 参数

3. **性能考虑**
   - 输入依赖误差计算可能较慢
   - 可以使用 `--no_input_dependence` 跳过以加快速度
   - 减少 `--num_samples` 可以加快误差分析

4. **层名称**
   - 层名称必须完全匹配模型中的实际名称
   - 运行时会打印所有可用的层名称

## 依赖项

- PyTorch
- NumPy
- Pandas
- Matplotlib
- Seaborn
- tqdm

## 故障排除

### 问题：找不到指定的层

**解决方案**：先运行不带 `--layers` 参数的版本，查看所有可用层名称。

### 问题：Mapping net 加载失败

**解决方案**：
1. 确保 checkpoint 中包含 `mapping_net_state_dict`
2. 或者提供单独的 `--mapping_net` 参数指向 mapping net 的 checkpoint

### 问题：内存不足

**解决方案**：
1. 减少 `--num_samples`
2. 使用 `--no_input_dependence` 跳过误差分析
3. 减少测试的层数量
