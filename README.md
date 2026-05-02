# Memristor-Aware Neural Network Evaluation Framework

A production-quality PyTorch experiment framework for evaluating neural networks with memristor device non-idealities. Supports ResNet-20 and ViT-Tiny models on CIFAR-10 and MNIST, with three experiment modes: baseline, memristor without compensation, and memristor with compensation (HAT or learned mapping).

## Features

- **Modular Design**: Clean separation between device models, neural networks, and experiment orchestration
- **Three Experiment Modes**:
  - `baseline`: Standard float training and evaluation
  - `memristor_no_comp`: Weights mapped to memristor device model at evaluation time
  - `memristor_with_comp`: Hardware-aware training (HAT) compensation
- **Reproducibility**: Deterministic seeds, checkpointing, and comprehensive logging
- **Visualization**: TensorBoard integration and local PNG plots
- **Energy Estimation**: Hook for NeuroSim/MNSIM integration (stub provided)
- **CLI Interface**: Easy-to-use command-line tools for training and evaluation

## Quick Start

### Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Baseline Experiment

```bash
# CIFAR-10 baseline
python -m src train --config configs/baseline/resnet20_baseline.yaml

# MNIST baseline
python -m src train --config configs/baseline/resnet20_mnist_baseline.yaml
```

### Run Memristor Experiment (No Compensation)

```bash
python -m src.train --config configs/memristor/resnet20_memristor_no_comp.yaml
```

### Run Memristor Experiment (With HAT Compensation)

```bash
# ResNet-20 with compensation
python -m src train --config configs/memristor/resnet20_memristor_comp.yaml --compensation hat

# ViT-Tiny with compensation
python -m src train --config configs/memristor/vit_tiny_memristor_comp.yaml --compensation hat
```

### Resume Training

```bash
python -m src train --config configs/baseline/resnet20_baseline.yaml --resume outputs/exp_name/seed_42/model_best.pth
```

### Evaluate Checkpoint

```bash
python -m src.eval --config configs/baseline/resnet20_baseline.yaml --checkpoint outputs/exp_name/seed_42/model_final.pth
```

## Available Configurations

YAML presets live under `configs/` in subfolders by role:

| Folder | Purpose |
|--------|---------|
| `configs/baseline/` | Float training/eval (no memristor forward); includes `default.yaml` (full option reference) |
| `configs/memristor/` | Real device-model runs: `*_memristor_no_comp.yaml`, `*_memristor_comp.yaml` (HAT) |

**Examples (paths):**
- `configs/baseline/resnet20_baseline.yaml`, `configs/baseline/resnet20_mnist_baseline.yaml`, `configs/baseline/vit_tiny_baseline.yaml`
- `configs/memristor/resnet20_memristor_no_comp.yaml`, `configs/memristor/resnet20_memristor_comp.yaml`
- `configs/memristor/vit_tiny_memristor_comp.yaml`

## Configuration

Configuration files are YAML-based and located under `configs/<subfolder>/`. Key sections:

- **General**: `seed`, `device`, `dataset`, `data_root`, `model_name`, `batch_size`, `epochs` (default: 100)
- **Optimizer**: `optimizer.type`, `optimizer.lr`, `optimizer.weight_decay`
- **Scheduler**: `scheduler.type`, `scheduler.params`
- **Memristor**: `memristor.G_min`, `memristor.G_max`, `memristor.variability_sigma`, `memristor.read_noise_sigma`, `memristor.drift_alpha`, `memristor.stuck_ratio`, `memristor.mapping`
- **Experiment**: `experiment.mode`, `experiment.compensation_method`, `experiment.energy_estimation`
- **Logging**: `logging.use_wandb`, `logging.project_name`, `logging.run_name`

See `configs/baseline/default.yaml` for a complete example with all available options.

## Project Structure

```
.
├── README.md
├── requirements.txt
├── configs/              # YAML presets: baseline/, memristor/
├── src/
│   ├── __main__.py      # CLI entrypoint
│   ├── train.py         # Training script
│   ├── eval.py          # Evaluation script
│   ├── utils/           # Utilities (logging, checkpointing, seeds, metrics)
│   ├── data/            # Data loading and preprocessing
│   ├── models/          # Model definitions (ResNet-20, ViT-Tiny, wrappers)
│   ├── memristor/       # Device model, mapping, compensation, energy estimation
│   └── experiments/     # Experiment orchestration and visualization
├── tests/               # Unit tests
├── examples/            # Example scripts
└── docker/              # Dockerfile for reproducibility
```

## Adding a New Model

1. Create model definition in `src/models/` (e.g., `my_model.py`)
2. Register in `src/models/model_zoo.py`:
   ```python
   def get_model(name: str, **kwargs):
       if name == "my_model":
           return MyModel(**kwargs)
       ...
   ```
3. Create a config file under the appropriate `configs/baseline/`, `configs/memristor/`, etc., referencing the model name
4. Run: `python -m src.train --config configs/baseline/my_model.yaml` (or place under the appropriate `configs/` subfolder)

## Integrating Real Energy Estimator

The energy estimation hook is located in `src/memristor/energy_estimator.py`. To integrate NeuroSim/MNSIM:

1. Replace the `estimate_energy()` function in `EnergyEstimator.estimate()` method
2. The function receives:
   - `model`: PyTorch model
   - `device_model`: MemristorDeviceModel instance
   - `dataloader`: DataLoader for inference
   - `subarray_size`, `num_subarrays`, `technology_node_nm`: Hardware parameters
3. Return a dictionary with keys: `energy_joules`, `latency_seconds`, `power_watts`

Example:
```python
def estimate_energy(self, model, device_model, dataloader, **kwargs):
    # Call your NeuroSim/MNSIM API here
    energy = call_neurosim(model, device_model, dataloader, **kwargs)
    return {
        "energy_joules": energy,
        "latency_seconds": latency,
        "power_watts": power
    }
```

## Expected Outputs

After running an experiment, check `outputs/{experiment_name}/{seed}/`:

- `model_best.pth`: Best model checkpoint (highest validation accuracy)
- `model_final.pth`: Final model state (after all epochs)
- `metrics.csv`: Per-epoch metrics (loss, accuracy, etc.)
- `accuracy_curve.png`: Training/validation accuracy curves
- `accuracy_vs_variability.png`: Parameter sweep results (if applicable)
- `tensorboard_logs/`: TensorBoard event files

## Testing

Run unit tests:

```bash
python -m pytest tests/
```

Or run individual test files:

```bash
python -m pytest tests/test_device_model.py
python -m pytest tests/test_mapping.py
python -m pytest tests/test_model_forward.py
```

## Docker

Build and run with Docker:

```bash
docker build -t memristor-nn -f docker/Dockerfile .
docker run --gpus all -v $(pwd)/outputs:/app/outputs memristor-nn python -m src.train --config configs/baseline/resnet20_baseline.yaml
```

## Runtime Notes

- **GPU Recommended**: ResNet-20 training on CIFAR-10 benefits from GPU acceleration
- **CPU Fallback**: Works on CPU but will be slower; reduce `batch_size` and `epochs` for quick tests
- **Memory**: ~2-4GB GPU memory for ResNet-20 with batch_size=128
- **Time**: Baseline training ~30-60 minutes on GPU, memristor experiments may take longer due to non-ideality computation

## License

This project is provided as-is for research purposes.

## Citation

If you use this framework in your research, please cite appropriately.


