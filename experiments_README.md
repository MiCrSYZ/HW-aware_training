# 噪声可学习性边界研究实验

本目录包含用于研究噪声"可学习与不可学习"边界的实验脚本。

## 概述

我们已经发现ADC量化噪声和paper版的IR-drop不适合接入训练过程。现在通过配置文件控制是否在训练时注入这些噪声，然后进行以下实验：

### 实验A: 噪声强度渐增的边界验证实验

针对各类非理想性分别绘制"噪声强度 vs 最终准确率"曲线，预期展示可学习与不可学习两类不同的趋势。

### 实验B: 训练动态的对比诊断实验

对比可学习噪声和不可学习噪声下的训练动态：
- (a) 梯度范数与梯度方差随时间的变化
- (b) 训练损失曲线和验证精度曲线
- (c) 权重更新轨迹的可视化（PCA降维）

## 配置说明

在配置文件中添加了以下新参数：

```yaml
memristor:
  # ... 其他参数 ...
  enable_adc_during_training: false  # 是否在训练时启用ADC量化（默认false）
  adc_training_mode: ste  # ADC训练模式：'ste'（使用Straight-Through Estimator）或'direct'（直接量化，梯度会消失）
  enable_ir_drop_paper_during_training: false  # 是否在训练时启用paper版IR-drop（默认false）
```

**注意**: 
- `enable_adc_during_training`默认是`false`，因为ADC量化会导致梯度消失（量化函数几乎处处零梯度）
- `adc_training_mode`控制训练时ADC的行为：
  - `ste`（默认）：使用Straight-Through Estimator，前向使用量化值，反向使用原始梯度，保持梯度流
  - `direct`：直接量化，不使用STE，梯度会消失，用于观察训练崩溃时的loss和梯度行为
- `enable_ir_drop_paper_during_training`默认是`false`，因为paper版IR-drop可能导致梯度不稳定、梯度爆炸或数值异常

## 实验A: 噪声强度渐增的边界验证实验

### 使用方法

```bash
python -m src.experiments.exp_noise_boundary \
    --config configs/memristor/resnet20_memristor_comp.yaml \
    --noise_type adc_bits \
    --noise_strengths 16 12 10 8 6 4 2 \
    --output_dir outputs/noise_boundary_adc \
    --enable_adc_during_training
```

### 参数说明

- `--config`: 基础配置文件路径
- `--noise_type`: 噪声类型，可选值：
  - `adc_bits`: ADC位数（越小噪声越大，范围2-16）
  - `ir_drop_scaling`: IR-drop缩放因子（越大噪声越大，范围0.0-2.0）
  - `variability_sigma`: 器件变异（越大噪声越大，范围0.0-0.5）
  - `read_noise_sigma`: 读噪声（越大噪声越大，范围0.0-1e-5）
  - `drift_alpha`: 电导漂移（越大噪声越大，范围0.0-1e-3）
  - `stuck_ratio`: 卡位故障（越大噪声越大，范围0.0-0.5）
- `--noise_strengths`: 噪声强度值列表
- `--output_dir`: 输出目录
- `--enable_adc_during_training`: 如果测试ADC，需要添加此标志
- `--enable_ir_drop_paper_during_training`: 如果测试paper版IR-drop，需要添加此标志

### 输出

- `{noise_type}_results.json`: 实验结果JSON文件
- `{noise_type}_boundary.png`: 噪声强度 vs 最终准确率曲线图

### 示例：测试ADC量化噪声

```bash
# 测试ADC位数从16到2（噪声逐渐增大），使用STE模式
python -m src.experiments.exp_noise_boundary \
    --config configs/memristor/resnet20_memristor_comp.yaml \
    --noise_type adc_bits \
    --noise_strengths 16 14 12 10 8 6 4 2 \
    --output_dir outputs/noise_boundary_adc_ste \
    --enable_adc_during_training

# 测试ADC量化噪声，使用direct模式（观察梯度消失导致的训练崩溃）
# 注意：需要在配置文件中设置 adc_training_mode: direct
python -m src.experiments.exp_noise_boundary \
    --config configs/memristor/resnet20_memristor_comp.yaml \
    --noise_type adc_bits \
    --noise_strengths 16 14 12 10 8 6 4 2 \
    --output_dir outputs/noise_boundary_adc_direct \
    --enable_adc_during_training
```

### 示例：测试paper版IR-drop

```bash
# 测试IR-drop缩放因子从0.0到2.0（噪声逐渐增大）
python -m src.experiments.exp_noise_boundary \
    --config configs/memristor/resnet20_memristor_comp.yaml \
    --noise_type ir_drop_scaling \
    --noise_strengths 0.0 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0 \
    --output_dir outputs/noise_boundary_ir_drop \
    --enable_ir_drop_paper_during_training
```

## 实验B: 训练动态的对比诊断实验

### 使用方法

```bash
python -m src.experiments.exp_training_dynamics \
    --config configs/memristor/resnet20_memristor_comp.yaml \
    --output_dir outputs/training_dynamics_adc \
    --noise_name adc_8bits_during_training \
    --enable_adc_during_training \
    --adc_bits 8 \
    --save_checkpoints \
    --extract_weights
```

### 参数说明

- `--config`: 基础配置文件路径
- `--output_dir`: 输出目录
- `--noise_name`: 噪声名称（用于标识实验，会出现在文件名中）
- `--enable_adc_during_training`: 在训练时启用ADC量化
- `--enable_ir_drop_paper_during_training`: 在训练时启用paper版IR-drop
- `--adc_bits`: ADC位数（如果启用ADC）
- `--ir_drop_scaling`: IR-drop缩放因子（如果启用IR-drop）
- `--save_checkpoints`: 保存每个epoch的检查点（用于权重轨迹分析，默认启用）
- `--extract_weights`: 提取权重轨迹（用于PCA可视化，默认启用）

### 输出

- `{noise_name}_results.json`: 实验结果JSON文件（包含metrics_history）
- `{noise_name}_loss_acc_curves.png`: 训练损失和准确率曲线
- `{noise_name}_gradient_stats.png`: 梯度范数和梯度方差随时间的变化
- `{noise_name}_weight_trajectory_pca.png`: 权重轨迹的PCA降维可视化（如果提取成功）

### 示例：对比可学习噪声和不可学习噪声

#### 1. 可学习噪声（例如：器件变异）

```bash
# 使用正常的HAT训练（只注入可学习噪声）
python -m src.experiments.exp_training_dynamics \
    --config configs/memristor/resnet20_memristor_comp.yaml \
    --output_dir outputs/training_dynamics_learnable \
    --noise_name learnable_variability \
    --save_checkpoints \
    --extract_weights
```

#### 2. 不可学习噪声 - ADC量化（STE模式）

```bash
# 在训练时注入ADC量化噪声，使用STE模式（保持梯度流）
python -m src.experiments.exp_training_dynamics \
    --config configs/memristor/resnet20_memristor_comp.yaml \
    --output_dir outputs/training_dynamics_adc_ste \
    --noise_name unlearnable_adc_8bits_ste \
    --enable_adc_during_training \
    --adc_bits 8 \
    --save_checkpoints \
    --extract_weights
```

#### 2b. 不可学习噪声 - ADC量化（Direct模式，观察梯度消失）

```bash
# 在训练时注入ADC量化噪声，使用direct模式（梯度会消失，观察训练崩溃）
# 注意：需要在配置文件中设置 adc_training_mode: direct
python -m src.experiments.exp_training_dynamics \
    --config configs/memristor/resnet20_memristor_comp.yaml \
    --output_dir outputs/training_dynamics_adc_direct \
    --noise_name unlearnable_adc_8bits_direct \
    --enable_adc_during_training \
    --adc_bits 8 \
    --save_checkpoints \
    --extract_weights
```

#### 3. 不可学习噪声 - paper版IR-drop

```bash
# 在训练时注入paper版IR-drop
python -m src.experiments.exp_training_dynamics \
    --config configs/memristor/resnet20_memristor_comp.yaml \
    --output_dir outputs/training_dynamics_ir_drop \
    --noise_name unlearnable_ir_drop_paper \
    --enable_ir_drop_paper_during_training \
    --ir_drop_scaling 1.0 \
    --save_checkpoints \
    --extract_weights
```

## 预期结果

### 可学习噪声场景

- 梯度范数保持平稳适度
- 不同批次梯度方向相对一致
- 损失下降曲线大体平滑（或呈抖动渐降趋势）
- 最终收敛到较高精度
- 权重轨迹相对平滑，朝向某一收敛区域

### 不可学习噪声场景

- 梯度范数出现异常峰值或剧烈震荡
- 可能出现梯度爆炸
- 参数更新发生数值异常
- 训练损失曲线上下剧烈波动甚至无法单调下降
- 验证精度停滞不前或随机波动
- 如果噪声足够严重，训练过程可能提前崩溃（出现NaN，使训练中断）
- 权重轨迹杂乱无章甚至反复来回，难以靠近最优点

## 注意事项

1. **数值稳定性保护**: 代码中已经添加了数值稳定性保护，如果IR-drop产生NaN或Inf值，会自动跳过以避免训练崩溃。

2. **ADC量化的训练模式**:
   - **STE模式**（默认）：使用Straight-Through Estimator，前向传播使用量化值，反向传播使用原始（未量化）梯度，保持梯度流。loss大概率不会炸，但可以观察量化对训练的影响。
   - **Direct模式**：直接量化，不使用STE，梯度会消失（因为`round()`操作几乎处处零梯度）。用于观察梯度消失导致的训练崩溃，以及loss和梯度的异常行为。

3. **训练时间**: 启用不可学习噪声可能导致训练时间显著增加，因为需要处理更多的数值异常情况。使用direct模式的ADC可能导致训练提前崩溃。

4. **资源需求**: 权重轨迹提取和PCA可视化需要保存每个epoch的检查点，会占用较多磁盘空间。

## 故障排除

如果训练过程中出现NaN或训练崩溃：

1. 检查配置中的噪声强度是否过大
2. 检查是否启用了数值稳定性保护（代码中已默认启用）
3. 尝试降低学习率
4. 检查梯度裁剪是否正常工作

