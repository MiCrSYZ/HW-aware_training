# 扰动强度标定系统

## 概述

扰动强度标定系统用于统一不同噪声类型的扰动程度。给定目标扰动强度δ*，系统会自动找到对应的噪声参数θ*使得δ(θ*) = δ*。

## 扰动强度定义

### 主标尺：δ_logit（logits级别）

[
\delta_{\text{logit}} = \sqrt{
\frac{\mathbb{E}[\lVert z_{\text{noisy}}-z_{\text{clean}} \rVert^2]}
{\mathbb{E}[\lVert z_{\text{clean}}\rVert^2]}
}
]

### 诊断标尺：δ_block（block级别）

[
\delta_{\text{block}(k)}=
\sqrt{
\frac{\mathbb{E}\left[|h^{(k)}_{\text{noisy}}-h^{(k)}_{\text{clean}}|^2\right]}
{\mathbb{E}\left[|h^{(k)}_{\text{clean}}|^2\right]}
}
]

## 支持的噪声类型

- `variability_sigma`: 器件变异噪声（全满足条件）
- `cond1_alpha`: 方差有界噪声（cond.1）
- `cond2_alpha`: 梯度无偏噪声（cond.2）
- `adc_bits`: ADC量化位数（cond.3）

## 使用方法

### 基本用法

```bash
python -m src.experiments.calibrate_noise_strength \
    --config configs/resnet20_memristor_comp.yaml \
    --target_delta 0.1 \
    --noise_type variability_sigma \
    --output calibration_results.json
```

### 完整参数

```bash
python -m src.experiments.calibrate_noise_strength \
    --config <config_file> \
    --checkpoint <checkpoint_path> \  # 可选，使用训练好的模型
    --target_delta <target_delta> \    # 目标扰动强度δ*
    --noise_type <noise_type> \       # 噪声类型
    --theta_min <min_value> \         # 可选，参数搜索下界
    --theta_max <max_value> \         # 可选，参数搜索上界
    --calibration_size 512 \          # 标定数据集大小（默认512）
    --output <output_json> \          # 输出JSON文件路径
    --seed 42                          # 随机种子（默认42）
```

### 参数说明

- `--config`: 配置文件路径（必须）
- `--checkpoint`: 模型checkpoint路径（可选，如果不提供则使用随机初始化的模型）
- `--target_delta`: 目标扰动强度δ*（必须）
- `--noise_type`: 噪声类型，可选值：
  - `variability_sigma`: 器件变异
  - `cond1_alpha`: 方差有界噪声
  - `cond2_alpha`: 梯度无偏噪声
  - `adc_bits`: ADC量化位数
- `--theta_min` / `--theta_max`: 参数搜索范围（可选，有默认值）
- `--calibration_size`: 标定数据集大小（默认512）
- `--output`: 输出JSON文件路径（必须）
- `--seed`: 随机种子（默认42）

### 默认参数搜索范围

- `variability_sigma`: [0.001, 0.5]
- `cond1_alpha`: [0.001, 1.0]
- `cond2_alpha`: [0.001, 1.0]
- `adc_bits`: [2.0, 16.0]

## 输出格式

标定完成后，会生成一个JSON文件，包含以下信息：

```json
{
  "config_path": "configs/resnet20_memristor_comp.yaml",
  "checkpoint_path": "checkpoints/model.pth",
  "noise_type": "variability_sigma",
  "target_delta": 0.1,
  "theta_star": 0.0523,
  "delta_logit": 0.1001,
  "delta_blocks": {
    "0": 0.085,
    "1": 0.092,
    "2": 0.098
  },
  "error": 0.0001,
  "calibration_size": 512,
  "seed": 42,
  "theta_bounds": [0.001, 0.5]
}
```

## 标定流程

1. **固定基线模型权重**：使用提供的checkpoint或随机初始化
2. **固定注入位置**：根据配置文件中的`noise_injection`设置
3. **固定eval模式**：模型始终在eval模式下运行
4. **固定标定数据**：使用固定的512个样本（可通过`--calibration_size`调整）
5. **固定随机种子**：确保可重复性
6. **不训练，只forward**：只进行前向传播，不更新权重
7. **二分搜索**：使用scipy的brentq算法找到θ*使得δ(θ*) = δ*

## 示例

### 示例1：标定ResNet20的variability_sigma

```bash
python -m src.experiments.calibrate_noise_strength \
    --config configs/resnet20_memristor_comp.yaml \
    --checkpoint checkpoints/resnet20_baseline.pth \
    --target_delta 0.1 \
    --noise_type variability_sigma \
    --output calibration/resnet20_variability_delta0.1.json
```

### 示例2：标定ViT的cond1_alpha

```bash
python -m src.experiments.calibrate_noise_strength \
    --config configs/vit_tiny_memristor_comp.yaml \
    --checkpoint checkpoints/vit_tiny_baseline.pth \
    --target_delta 0.15 \
    --noise_type cond1_alpha \
    --theta_min 0.01 \
    --theta_max 0.5 \
    --output calibration/vit_cond1_delta0.15.json
```

### 示例3：标定ADC位数

```bash
python -m src.experiments.calibrate_noise_strength \
    --config configs/resnet20_memristor_comp.yaml \
    --checkpoint checkpoints/resnet20_baseline.pth \
    --target_delta 0.2 \
    --noise_type adc_bits \
    --theta_min 4.0 \
    --theta_max 12.0 \
    --output calibration/resnet20_adc_delta0.2.json
```

## 注意事项

1. **计算成本**：标定过程需要多次前向传播，可能需要一些时间
2. **内存使用**：标定过程中会存储所有block特征，注意内存使用
3. **模型结构**：目前支持ResNet20和ViT-Tiny，其他模型可能需要修改`extract_block_features`函数
4. **精度**：二分搜索的容差默认是1e-3，可以通过修改代码调整

## 技术细节

- 使用scipy的`brentq`算法进行二分搜索
- 如果brentq失败，会自动回退到手动二分搜索
- 对于`adc_bits`，会自动取整（因为必须是整数）
- 所有计算都在eval模式下进行，确保BN/LN行为一致

