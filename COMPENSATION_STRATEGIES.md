# 补偿策略说明文档

本文档说明如何使用不同的补偿策略来减少忆阻器非理想性的影响。

## 可用的补偿策略

### 1. HAT (Hardware-Aware Training)
**方法**: 在训练时应用非理想性，使模型对忆阻器缺陷具有鲁棒性。

**配置文件**: `configs/resnet20_memristor_comp.yaml`

```yaml
experiment:
  mode: memristor_with_comp
  compensation_method: hat
```

**运行命令**:
```bash
python -m src.train --config configs/resnet20_memristor_comp.yaml
```

### 2. Learned Mapping (学习映射)
**方法**: 训练一个小的神经网络来学习每层的最优权重到电导映射参数，自适应地优化映射函数。

**工作原理**:
- 使用 `LearnedMappingNet` 网络，输入每层权重的统计信息（均值、标准差、最小值、最大值）
- 输出每层的映射参数（scale, offset）
- 这些参数用于修改权重到电导的映射，以最小化非理想性的影响

**配置文件**: `configs/resnet20_memristor_learned_mapping.yaml`

```yaml
experiment:
  mode: memristor_with_comp
  compensation_method: learned_mapping
  mapping_lr: 1e-3                    # 映射网络的学习率
  mapping_epochs_per_main_epoch: 1     # 每个主训练epoch中映射网络的训练轮数
```

**运行命令**:
```bash
python -m src.train --config configs/resnet20_memristor_learned_mapping.yaml
```

**实现细节**:
- `LearnedMappingNet`: 一个小的MLP网络，学习每层的映射参数
- `learned_mapping_train`: 训练映射网络的函数
- `_forward_with_learned_mapping`: 应用学习到的映射参数进行前向传播
- `MemristorLinear` 和 `MemristorConv2d` 支持 `set_learned_mapping()` 方法

### 3. Hybrid (混合策略)
**方法**: 同时使用 HAT 和 learned mapping，通过加权损失函数结合两种策略。

**工作原理**:
- 对每个batch，同时进行 HAT 前向传播和 learned mapping 前向传播
- 计算两个损失：`hat_loss` 和 `mapping_loss`
- 组合损失：`loss = hat_weight * hat_loss + mapping_weight * mapping_loss`
- 同时更新模型参数和映射网络参数

**配置文件**: `configs/resnet20_memristor_hybrid.yaml`

```yaml
experiment:
  mode: memristor_with_comp
  compensation_method: hybrid
  hat_weight: 0.5                      # HAT 损失的权重
  mapping_weight: 0.5                  # Learned mapping 损失的权重
  mapping_lr: 1e-3                     # 映射网络的学习率
```

**运行命令**:
```bash
python -m src.train --config configs/resnet20_memristor_hybrid.yaml
```

**实现细节**:
- `hybrid_compensation_train`: 混合训练函数
- 同时维护模型优化器和映射网络优化器
- 每个batch都进行两次前向传播（HAT和learned mapping）

## 代码结构

### 核心模块

1. **`src/memristor/compensation.py`**:
   - `LearnedMappingNet`: 学习映射的网络
   - `learned_mapping_train`: 训练映射网络
   - `hybrid_compensation_train`: 混合训练
   - `_forward_with_learned_mapping`: 应用学习映射的前向传播

2. **`src/models/memristor_wrappers.py`**:
   - `MemristorLinear.set_learned_mapping()`: 设置线性层的映射参数
   - `MemristorConv2d.set_learned_mapping()`: 设置卷积层的映射参数

3. **`src/memristor/device_model.py`**:
   - `map_weights_to_conductance()`: 支持 `learned_scale` 和 `learned_offset` 参数

4. **`src/experiments/run_experiment.py`**:
   - `_train_learned_mapping()`: 训练循环中的 learned mapping 支持
   - `_train_hybrid()`: 训练循环中的 hybrid 支持

## 使用建议

### 选择补偿策略

1. **HAT**: 
   - 最简单，计算开销最小
   - 适合快速实验和基线对比
   - 通常能提供良好的鲁棒性

2. **Learned Mapping**:
   - 需要额外的训练时间（训练映射网络）
   - 可能在某些情况下提供更好的性能
   - 适合对精度要求较高的场景

3. **Hybrid**:
   - 结合两种方法的优点
   - 计算开销最大（每个batch两次前向传播）
   - 适合追求最佳性能的场景

### 超参数调优

**Learned Mapping**:
- `mapping_lr`: 通常使用 1e-3 到 1e-4
- `mapping_epochs_per_main_epoch`: 1-3 通常足够

**Hybrid**:
- `hat_weight` 和 `mapping_weight`: 可以尝试不同的权重组合
  - 默认 0.5/0.5 是平衡的选择
  - 如果 HAT 效果更好，可以增加 `hat_weight`
  - 如果 learned mapping 效果更好，可以增加 `mapping_weight`

## 预期结果

根据非理想性参数的不同，不同策略的效果可能不同：

- **Baseline**: 最高准确率（无非理想性）
- **No Compensation**: 最低准确率（非理想性在评估时应用）
- **HAT**: 通常介于 baseline 和 no_comp 之间
- **Learned Mapping**: 可能接近或超过 HAT
- **Hybrid**: 通常是最好的，可能接近 baseline

## 故障排除

如果 learned mapping 或 hybrid 方法没有改善性能：

1. **检查映射网络是否在训练**:
   - 查看日志中是否有 "Learned mapping epoch" 的输出
   - 检查映射损失是否在下降

2. **调整学习率**:
   - 如果映射损失不下降，尝试降低 `mapping_lr`
   - 如果训练不稳定，尝试降低 `mapping_lr`

3. **增加映射训练轮数**:
   - 增加 `mapping_epochs_per_main_epoch` 可能有助于学习更好的映射

4. **检查非理想性参数**:
   - 如果非理想性参数太小，所有策略的效果可能不明显
   - 参考 `TROUBLESHOOTING.md` 中的建议

## 示例输出

运行 learned mapping 训练时，你会看到类似以下的日志：

```
Learned mapping epoch 1/1, train_loss: 0.8234, val_loss: 0.7891, val_acc: 72.34%
Epoch 0: train_loss=0.8234, train_acc=72.34%, val_loss=0.7891, val_acc=72.34%
```

运行 hybrid 训练时，你会看到：

```
Epoch 0: train_loss=0.8456, train_acc=71.23%, val_loss=0.8123, val_acc=71.23%
```

其中 `train_loss` 是 HAT 和 learned mapping 损失的加权组合。

