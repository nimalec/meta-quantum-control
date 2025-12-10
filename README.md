# Meta Learning for Adaptive Quantum Control 
## Quick Start

### Prerequisites
You must have uv installed:
```bash
# Mac/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Installation
```bash
# Install dependencies (creates .venv and installs all packages from pyproject.toml)
export UV_NATIVE_TLS=true         
uv sync --active

# Activate the environment
source .venv/bin/activate  # Mac/Linux
.venv\Scripts\activate     # Windows
```

## Running Experiments

### 1. Trainin the Meta Policy 

Train a FOMAML policy for 1-qubit quantum control:

```bash
cd experiments/fig_5_meta_training 
python train_meta.py --config ../configs/experiment_config.yaml
```

This will:
- Train a meta-learned policy using MAML
- Save checkpoints to `checkpoints/`
- Save training metrics to `checkpoints/training_history.json`
- Display training progress with validation metrics


### 2. Configuration

Edit `configs/experiment_config.yaml` to customize:
- Task distribution parameters (noise PSD ranges)
- MAML hyperparameters (inner/meta learning rates)
- Training iterations and validation intervals
- Policy architecture (hidden layers, dimensions)

Example config:
```yaml
# MAML hyperparameters
inner_lr: 0.01
inner_steps: 5
meta_lr: 0.001
n_iterations: 1000
val_interval: 50

# Task distribution
task_dist_type: 'uniform'
alpha_range: [0.5, 2.0]
A_range: [0.05, 0.3]
omega_c_range: [2.0, 8.0]
```


## Testing

```bash
# Run all tests
make test

# Or directly with uv
uv run pytest tests/

# Run tests with verbose output
make test-verbose

# Run tests with coverage report
make test-coverage
```

## Citation

```bibtex
@inproceedings{leclerc2025meta,
  title={Meta-Reinforcement Learning for Quantum Control},
  author={Leclerc, Nima; Miller, Chris; Brawand, Nicholas},
  year={2025}
}
```

## License

MIT License - see LICENSE file
