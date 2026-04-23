# 验证 “BN 放大扰动” 与 “优化目标一致性” 的两种观点

本文档给出可操作的实验设计，用于验证：

1. **BN 放大扰动**：no_comp 下，ResNet 的 BN 用干净训练得到的 running stats 去归一化带 drift 的激活，使 **frozen** 方向的系统性偏置无法被吸收，从而放大 frozen vs resampled 的差距。
2. **优化目标一致性**：comp 下，**frozen** 提供稳定、可适应的扰动（同一 d），**resampled** 每步 d 不同、目标在变，导致 frozen comp > resampled comp。

---

## 一、验证 “BN 放大扰动”

### 1.1 直接度量：BN 输入与 running stats 的失配（推荐）

**思路**：在 no_comp 的 **eval** 阶段，对每个 BN 层记录：当前 batch 的输入均值 `batch_mean` 与 BN 的 `running_mean`。若 “BN 用干净统计量去吃带 drift 的激活” 成立，则：

- **frozen**：drift 方向固定 → 每个 batch 的 `(batch_mean - running_mean)` 应沿**同一方向**、均值非零（系统性偏置）。
- **resampled**：drift 零均值 → `(batch_mean - running_mean)` 在不同 batch 间应近似零均值、无稳定方向。

**做法**：

- 用 hook 在 `BatchNorm2d.forward` 里记录：
  - `batch_mean` = 当前 batch 在 channel 维的均值（与 BN 内部计算一致）；
  - `running_mean` = 当前 BN 的 `running_mean`；
  - 可选：`(batch_mean - running_mean)` 的 L2 范数、或沿 channel 维的均值。
- 在 **no_comp** 下，用 **synth_noise_model**（带 frozen_additive_drift）在 val/test 上跑若干 batch（不更新权重、不更新 BN 的 running stats）。
- 分两轮跑：一轮 `drift_frozen=True`，一轮 `drift_frozen=False`（resampled）。
- **预期**：
  - frozen：各层 `mean(batch_mean - running_mean)` 明显非零，且方向在多次运行/不同 batch 间稳定（或与各层 cached 的 d 相关）；
  - resampled：`mean(batch_mean - running_mean)` 接近 0，或其范数显著小于 frozen。

**实现**：见 `src/experiments/verify_drift_hypotheses.py` 中的 `collect_bn_mismatch_stats()`。

---

### 1.2 消融：ResNet 用 LayerNorm 替代 BN

**思路**：若 “BN 用固定统计量放大 frozen 的失配” 是主因，则把 ResNet 的 BN 换成 LayerNorm（或 GroupNorm 等不依赖 running stats 的归一化）后，no_comp 下 **frozen vs resampled 的准确率差** 应明显缩小，更接近 ViT 的差距。

**做法**：

- 实现一个 `ResNet20LayerNorm`（或通过 config 在现有 ResNet 上 swap BN→LN），保持其他（conv、shortcut、注入点）一致。
- 用同一套 `frozen_additive_drift` 配置（frozen / resampled 各跑 no_comp），对比：
  - ResNet20 + BN：frozen no_comp 与 resampled no_comp 的准确率差 Δ_BN；
  - ResNet20 + LN：frozen no_comp 与 resampled no_comp 的准确率差 Δ_LN。
- **预期**：Δ_LN < Δ_BN（LN 后差距缩小），则支持 “BN 放大扰动” 的解释。

---

### 1.3 消融：eval 时用 “带噪统计量” 的 BN（可选）

**思路**：若 BN 的 running stats 来自干净训练是问题所在，则若在 eval 时**临时**用 “在带 drift 数据上重新估计的均值/方差” 做归一化，frozen no_comp 的准确率应有所恢复。

**做法**：

- 在 no_comp 的 eval 阶段，先对 synth_noise_model 在若干 batch 上跑一遍 **eval 模式但打开 BN 的 momentum 更新**（或手动用当前 batch 的 mean/var 做一次归一化），得到一组 “带噪的 running_mean/var”；
- 用这组统计量替换 BN 的 running_mean/var，再在 test 上评估 frozen no_comp。
- **预期**：frozen no_comp 准确率相对 “用干净 running stats” 的 baseline 有提升，则与 “BN 统计量与输入分布失配” 一致。

---

## 二、验证 “优化目标一致性”

### 2.1 同一 batch、不同 d 下的 loss 方差（resampled 特有）

**思路**：comp 时，**resampled** 每步 d 不同，同一 batch 若用不同 seed 得到不同 d，前向 loss 会不同；**frozen** 同一 batch 多次前向 loss 相同。因此 “同一 batch、不同 d 的 loss 方差” 在 resampled 上应 > 0，在 frozen 上应 ≈ 0。

**做法**：

- 在 **comp** 训练中，每隔 K 步（如每 100 step）取当前 batch，在 **不更新参数** 的前提下：
  - **frozen**：对该 batch 做 2 次前向，loss 记为 L1, L2（理论上 L1=L2）；
  - **resampled**：对该 batch 做 2 次前向，用不同 `seed` 使各层 d 不同，loss 记为 L1, L2。
- 记录 `|L1 - L2|` 或 `var(L1, L2)`。多步、多 batch 后：
  - **预期**：frozen 的 `|L1-L2|` 近似 0（或仅数值误差）；resampled 的 `|L1-L2|` 明显 > 0，且方差随 drift_beta 增大而增大。

**实现**：见 `verify_drift_hypotheses.py` 中的 `measure_loss_variance_same_batch()`。

---

### 2.2 训练过程中 loss / 梯度范数的方差（frozen vs resampled）

**思路**：若 resampled 的 “目标在变”，则 step 间 loss 和梯度应更波动；frozen 更稳定。

**做法**：

- 用相同数据顺序、相同超参，分别跑 **comp + frozen** 与 **comp + resampled**（可只跑少量 epoch 或 1 个 epoch）。
- 每步记录：当前 batch 的 loss；可选地，某几层参数的 gradient norm。
- 对 “同一 epoch 内、同一 data order” 的 step 序列，计算：
  - loss 的方差（或滑动方差）；
  - 梯度范数的方差。
- **预期**：resampled 的 loss 方差、梯度范数方差 **大于** frozen，则与 “resampled 目标不一致、优化更抖” 一致。

---

### 2.3 收敛速度与最终精度

**思路**：一致性有利于收敛 → frozen comp 应更快收敛、且最终准确率更高。

**做法**：

- 已有现象：frozen comp 90% vs resampled comp 62%（ResNet）；frozen comp 80% vs resampled 68%（ViT）。可补充：
  - 画 loss 曲线：frozen comp 应在前期就低于 resampled comp，且最终 plateau 更低；
  - 画 “accuracy vs epoch”：frozen comp 更早达到高准确率。
- **预期**：frozen 收敛更快、最终 acc 更高，与 “一致目标更易优化” 一致。

---

### 2.4 中途切换 frozen ↔ resampled（可选）

**思路**：若 “一致性” 是关键，则训练中途从 resampled 改为 frozen（固定住当前的 d）应有利于后续收敛；反之，从 frozen 改为 resampled 可能使 loss 上升或震荡。

**做法**：

- **A**：comp 前 50% step 用 resampled，后 50% 用 frozen（例如固定每层 d 为 “当前 step 的 d” 或某次采样的 d）。对比 “全程 resampled” 的 final acc。
- **B**：comp 前 50% 用 frozen 训到收敛，后 50% 改为 resampled。看 loss/acc 是否变差或震荡。
- **预期**：A 中 “后半段 frozen” 比 “全程 resampled” 更好；B 中 “后半段改 resampled” 可能变差或更抖。则支持 “稳定目标有助于 comp” 的说法。

---

## 三、脚本使用方式（概要）

```bash
# 1. BN 失配统计（no_comp eval，需已有 no_comp 训练的 checkpoint）
python -m src.experiments.verify_drift_hypotheses --mode bn_mismatch \
  --config configs/synth/resnet20_synth_no_comp.yaml \
  --drift_frozen true \
  --checkpoint outputs/xxx/model_best.pth \
  --num_batches 50
# 再跑一遍 --drift_frozen false，对比两轮输出：frozen 时 mean_diff_l2 应明显更大

# 2. 同一 batch 两次前向的 loss 差（comp，验证优化目标一致性）
python -m src.experiments.verify_drift_hypotheses --mode loss_variance \
  --config configs/synth/resnet20_synth_comp.yaml \
  --drift_frozen true --num_batches 100
# 再跑一遍 --drift_frozen false：frozen 时 |L1-L2|≈0，resampled 时 |L1-L2|>0
```

详细参数见脚本内 `--help`。

---

## 四、预期结论汇总

| 假设 | 验证方法 | 若假设成立，预期 |
|------|----------|------------------|
| BN 放大扰动 | BN 输入与 running_mean 的失配 | frozen 时 (batch_mean - running_mean) 系统性非零；resampled 时近似零均值 |
| BN 放大扰动 | ResNet BN → LN 消融 | 换 LN 后 no_comp 下 frozen vs resampled 准确率差缩小 |
| 优化目标一致性 | 同 batch 不同 d 的 loss 差 | resampled 下 \|L1-L2\| > 0；frozen 下 ≈ 0 |
| 优化目标一致性 | loss/梯度方差 | resampled comp 的 step 间方差 > frozen comp |
| 优化目标一致性 | 收敛曲线 | frozen comp 收敛更快、最终 acc 更高 |

若上述预期均与实验结果一致，则 “BN 放大扰动” 与 “优化目标一致性” 两个观点都得到支持。
