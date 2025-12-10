# Meta-RL for Quantum Control
Implementation of "Meta-Reinforcement Learning for Quantum Control: Generalization and Robustness under Noise Shifts"

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

### 1. Training a Meta-Learned Policy

Train a MAML policy for 1-qubit quantum control:

```bash
cd experiments
python train_meta.py --config ../configs/experiment_config.yaml
```

This will:
- Train a meta-learned policy using MAML
- Save checkpoints to `checkpoints/`
- Save training metrics to `checkpoints/training_history.json`
- Display training progress with validation metrics


This generates three publication-quality figures:
- `training_validation_curves.png` - Training loss and validation metrics
- `validation_fidelity.png` - Validation fidelity with error bars
- `complete_training_summary.png` - Comprehensive 2×2 grid with all metrics

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


## Project Structure

```
meta-quantum-control/
├── metaqctrl/              # Main package
│   ├── quantum/            # Quantum simulation (Lindblad, gates, noise)
│   ├── meta_rl/            # MAML implementation (policy, training)
│   ├── theory/             # Theoretical tools (environment, bounds)
│   └── utils/              # Plotting and utilities
├── experiments/            # Training scripts
│   ├── train_meta.py       # 1-qubit training
│   └── train_meta_two_qubit.py  # 2-qubit training
├── configs/                # Configuration files
├── tests/                  # Unit tests
└── results/                # Generated figures
```

## Contributing
```bash
# Adding packages
uv add <package>
```

## Citation

```bibtex
@inproceedings{leclerc2025meta,
  title={Meta-Reinforcement Learning for Quantum Control},
  author={Leclerc, Nima; Brawand, Nicholas},
  year={2025}
}
```

## License

MIT License - see LICENSE file
