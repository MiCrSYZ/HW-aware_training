# Matched-Distortion Learnability Test

## 实验目的

验证 paper IR-drop 的"不可学习性"是否源于其结构特性，而非仅仅因为"太激进"。

**核心思想**：将 paper IR-drop 和 crossbar IR-drop 调整到相同的平均相对输出扰动 δ，然后观察训练是否仍然崩溃。如果同等 δ 下只有 paper IR-drop 崩溃，那就说明是结构不可学，而非"太激进"。

## 实验设计

### A 组：paper IR-drop
- `ir_drop_mode="paper"`
- 用 `ir_drop_scaling` 控制强度

### B 组：crossbar IR-drop（温和替代）
- `ir_drop_mode="crossbar"`
- 用 `ir_drop_cap` 控制强度

## 实验步骤

### 步骤 1：校准强度

对每个目标 δ 值（Low: 0.05, Mid: 0.10, High: 0.20），分别找到：
- paper 模式：对应的 `ir_drop_scaling` 值
- crossbar 模式：对应的 `ir_drop_cap` 值

使得两种模式达到相同的 δ。

### 步骤 2：正式训练对照

每个 δ 档位运行两次训练：
- A(δ): paper IR-drop，强度已校准
- B(δ): crossbar IR-drop，强度已校准

## 使用方法

### 1. 准备 baseline checkpoint

首先训练一个 baseline 模型：

```bash
python -m src.experiments.run_experiment --config configs/resnet20_baseline.yaml
```

训练完成后，checkpoint 会保存在 `outputs/resnet20_baseline/seed_42/.../model_best.pth`

### 2. 运行匹配失真实验

```bash
python -m src.experiments.exp_matched_distortion \
    --config configs/resnet20_baseline.yaml \
    --checkpoint outputs/resnet20_baseline/seed_42/.../model_best.pth \
    --output_dir ./outputs/matched_distortion \
    --calibration_samples 512 \
    --target_deltas 0.05 0.10 0.20
```

### 参数说明

- `--config`: 基础配置文件路径
- `--checkpoint`: baseline checkpoint 路径
- `--output_dir`: 输出目录
- `--calibration_samples`: 用于校准的样本数（默认 512）
- `--target_deltas`: 目标 δ 值列表（默认 [0.05, 0.10, 0.20]）
- `--skip_calibration`: 跳过校准，直接使用提供的强度值
- `--paper_strengths`: paper 模式强度值（仅在 `--skip_calibration` 时使用）
- `--crossbar_strengths`: crossbar 模式强度值（仅在 `--skip_calibration` 时使用）
- `--seed`: 随机种子（默认 42）

### 3. 跳过校准（使用已有校准结果）

如果已经完成校准，可以跳过校准步骤：

```bash
python -m src.experiments.exp_matched_distortion \
    --config configs/resnet20_baseline.yaml \
    --checkpoint outputs/resnet20_baseline/seed_42/.../model_best.pth \
    --output_dir ./outputs/matched_distortion \
    --skip_calibration \
    --paper_strengths 0.5 1.0 1.5 \
    --crossbar_strengths 0.05 0.10 0.15 \
    --target_deltas 0.05 0.10 0.20
```

## 输出结果

### 校准结果

保存在 `{output_dir}/calibration_results.json`：

```json
{
  "paper_strengths": {
    "0.05": 0.234,
    "0.10": 0.456,
    "0.20": 0.789
  },
  "crossbar_strengths": {
    "0.05": 0.012,
    "0.10": 0.025,
    "0.20": 0.050
  },
  "target_deltas": [0.05, 0.10, 0.20]
}
```

### 训练结果

每个实验的结果保存在：
- `{output_dir}/paper_delta{delta:.2f}/` - paper 模式训练结果
- `{output_dir}/crossbar_delta{delta:.2f}/` - crossbar 模式训练结果

所有结果汇总在 `{output_dir}/all_results.json`。

### 记录的指标

1. **训练指标**：
   - train loss / val acc
   - 梯度统计（grad_norm/grad_var）
   - 训练是否成功完成

2. **δ 指标**：
   - 训练后的实际 δ 值
   - applied_ratio（IR-drop 被应用的比例，用于检查是否真的注入了）

## 注意事项

### 1. 训练时 IR-drop 注入

**重要**：代码会确保训练时也应用 IR-drop：
- paper 模式：设置 `enable_ir_drop_paper_during_training=True`
- crossbar 模式：设置 `ir_drop_train_enabled=True`

### 2. NaN/Inf 处理

当前实现中，如果 IR-drop 产生 NaN/Inf，训练时会静默跳过（使用原始输出）。这可能会影响实验结论。

**建议**：
- 检查 `applied_ratio`，如果太低（< 0.9），说明 IR-drop 经常被跳过
- 如果 paper IR-drop 导致训练崩溃，这可能正是我们想要看到的结果（证明其不可学性）

### 3. 其他噪声

实验会自动禁用其他噪声（variability, read_noise, drift, stuck），避免混淆。

## 实验解读

### 预期结果

如果实验成功，应该看到：

1. **同等 δ 下**：
   - paper IR-drop：训练崩溃或准确率极低
   - crossbar IR-drop：训练正常，准确率可接受

2. **这证明了**：
   - paper IR-drop 的"不可学习性"不是因为它"太激进"
   - 而是因为其结构特性（输入相关、非平稳、跨维度耦合）导致不可学

### 如果两种模式都崩溃

如果同等 δ 下两种模式都崩溃，可能说明：
- δ 值设置过高
- 或者两种模式都有不可学的问题

### 如果两种模式都不崩溃

如果同等 δ 下两种模式都能正常训练，可能说明：
- δ 值设置过低
- 或者需要更高的 δ 值才能看到差异

## 故障排除

### 1. 校准失败

如果校准时找不到合适的强度值，可以：
- 扩大搜索范围（修改 `calibrate_strength` 函数中的 `strength_range`）
- 增加搜索次数（修改 `num_trials` 参数）

### 2. 训练崩溃

如果训练时崩溃，检查：
- 日志文件 `{output_dir}/matched_distortion.log`
- 是否是因为 NaN/Inf 导致的崩溃
- 如果是，这可能是预期的（证明 paper IR-drop 不可学）

### 3. δ 值不匹配

如果训练后的实际 δ 与目标 δ 差异较大，可能原因：
- 校准时使用的样本与训练时不同
- 模型权重在训练过程中发生变化
- 可以尝试使用更多样本进行校准

