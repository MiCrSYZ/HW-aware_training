# Memristor-Aware Neural Network Evaluation Framework

PyTorch framework for evaluating neural networks under memristor device
non-idealities. The code supports baseline training, post-training memristor
evaluation, and hardware-aware training with compensation.

## Features

- Baseline, memristor no-compensation, and memristor compensation modes
- Device-level non-idealities: conductance variability, read noise, drift,
  stuck devices, ADC quantization, and IR drop
- Weight-to-conductance mapping utilities, including differential-pair mapping
- Model support for ResNet-20, ViT-Tiny, and GRU AG News
- Checkpointing, TensorBoard logging, optional Weights & Biases logging
- Energy-estimation hook for NeuroSim/MNSIM-style integration

## Installation

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## Quick Start

Run a clean baseline:

```bash
python -m src.train --config configs/baseline/resnet20_baseline.yaml
```

Evaluate with memristor non-idealities after clean training:

```bash
python -m src.train --config configs/memristor/resnet20_memristor_no_comp.yaml
```

Train with memristor-aware compensation:

```bash
python -m src.train --config configs/memristor/resnet20_memristor_comp.yaml --compensation hat
```

Resume from a checkpoint:

```bash
python -m src.train --config configs/baseline/resnet20_baseline.yaml --resume outputs/exp_name/seed_42/model_best.pth
```

Evaluate a checkpoint:

```bash
python -m src.eval --config configs/baseline/resnet20_baseline.yaml --checkpoint outputs/exp_name/seed_42/model_final.pth
```

## Configurations

YAML presets live under `configs/`:

| Folder | Purpose |
|--------|---------|
| `configs/baseline/` | Clean floating-point baselines |
| `configs/memristor/` | Memristor device-model experiments |

Common examples:

- `configs/baseline/resnet20_baseline.yaml`
- `configs/baseline/vit_tiny_baseline.yaml`
- `configs/memristor/resnet20_memristor_no_comp.yaml`
- `configs/memristor/resnet20_memristor_comp.yaml`
- `configs/memristor/vit_tiny_memristor_comp.yaml`

Important config sections:

- `experiment.mode`: `baseline`, `memristor_no_comp`, or `memristor_with_comp`
- `memristor`: device ranges, variability, read noise, drift, ADC, IR drop,
  mapping, and layer noise-injection controls
- `optimizer` and `scheduler`: training hyperparameters
- `logging`: TensorBoard and optional W&B settings

## Project Layout

```text
.
|-- README.md
|-- requirements.txt
|-- configs/
|   |-- baseline/
|   `-- memristor/
|-- docker/
|-- examples/
|-- src/
|   |-- data/
|   |-- experiments/
|   |-- memristor/
|   |-- models/
|   |-- utils/
|   |-- train.py
|   `-- eval.py
`-- tests/
```

## Adding a Model

1. Add the model implementation under `src/models/`.
2. Register it in `src/models/model_zoo.py`.
3. Add a baseline config and, if needed, a memristor config.
4. Run with `python -m src.train --config <config-path>`.

## Energy Estimation Hook

The energy-estimation interface is in `src/memristor/energy_estimator.py`.
Replace the stub implementation with calls to your simulator, using the model,
device model, dataloader, subarray parameters, and technology-node settings from
the config.

Expected return keys:

- `energy_joules`
- `latency_seconds`
- `power_watts`

## Outputs

Training artifacts are written under `outputs/{experiment_name}/seed_{seed}/`:

- `model_best.pth`
- `model_final.pth`
- `metrics.csv`
- `accuracy_curve.png`
- `tensorboard_logs/`

## Testing

```bash
python -m pytest tests/
```

Run selected tests:

```bash
python -m pytest tests/test_device_model.py
python -m pytest tests/test_mapping.py
python -m pytest tests/test_model_forward.py
```

## Docker

```bash
docker build -t memristor-nn -f docker/Dockerfile .
docker run --gpus all -v $(pwd)/outputs:/app/outputs memristor-nn \
  python -m src.train --config configs/baseline/resnet20_baseline.yaml
```

## Notes

- A GPU is recommended for CIFAR-scale training.
- CPU runs are supported but can be slow; reduce `batch_size` and `epochs` for
  quick checks.
- Generated datasets, outputs, local notes, scripts, and paper material are
  intentionally ignored by Git.
