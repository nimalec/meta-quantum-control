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

**For 2-qubit systems (e.g., CNOT gate):**
```bash
python train_meta_two_qubit.py --config ../configs/two_qubit_experiment.yaml
```

### 2. Generating Publication-Quality Plots

After training, generate plots from the saved training history:

**Option 1: Command line (recommended)**
```bash
cd metaqctrl
python -m utils.plot_training checkpoints/training_history.json results/figures/
```

**Option 2: Python script**
```python
from metaqctrl.utils.plot_training import plot_complete_summary

plot_complete_summary(
    history_path="checkpoints/training_history.json",
    save_dir="results/figures",
    smooth_window=10,
    dpi=300
)
```

**Option 3: Use example script**
```bash
cd examples
python plot_training_example.py
```

This generates three publication-quality figures:
- `training_validation_curves.png` - Training loss and validation metrics
- `validation_fidelity.png` - Validation fidelity with error bars
- `complete_training_summary.png` - Comprehensive 2×2 grid with all metrics

### 3. Configuration

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

### 4. Understanding Training Output

During training, you'll see output like:

```
Iter 0/1000 | Meta Loss: 0.8234 | Task Loss: 0.8234 ± 0.0123 | Range: [0.7891, 0.8567] | Grad Norm: 0.0234

[Validation] Iter 50
  Pre-adapt loss:  0.7892
  Post-adapt loss: 0.4567
  Val Fidelity: 0.5433 ± 0.0234
  Val Error: 0.4567
  Adaptation gain: 0.3325
```

**Key metrics:**
- **Meta Loss**: Average loss across all tasks (lower is better)
- **Val Fidelity**: 1 - validation error (higher is better, max = 1.0)
- **Val Error**: Validation loss after adaptation (lower is better)
- **Adaptation Gain**: How much the policy improves with adaptation

**Outputs saved:**
- `checkpoints/maml_TIMESTAMP.pt` - Final model checkpoint
- `checkpoints/maml_TIMESTAMP_best.pt` - Best validation checkpoint
- `checkpoints/training_history.json` - All training metrics

### 5. Typical Workflow

Complete workflow from training to publication figures:

```bash
# 1. Install dependencies
uv sync
source .venv/bin/activate

# 2. Train the model
cd experiments
python train_meta.py --config ../configs/experiment_config.yaml

# 3. Generate plots for your paper
cd ../metaqctrl
python -m utils.plot_training ../checkpoints/training_history.json ../results/figures/

# 4. Find your figures
ls ../results/figures/
# Output:
#   training_validation_curves.png
#   validation_fidelity.png
#   complete_training_summary.png
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

## Documentation
See `docs/` directory for complete documentation.

## Troubleshooting

**Issue: "No module named 'metaqctrl'"**
- Solution: Make sure you've activated the virtual environment with `source .venv/bin/activate`

**Issue: Training is very slow**
- Solution: The code will automatically use GPU if available. Check `torch.cuda.is_available()` to verify GPU access.

**Issue: "FileNotFoundError" when generating plots**
- Solution: Make sure you've run training first to generate `training_history.json`

**Issue: Validation fidelity is not improving**
- Try: Reduce `inner_lr` or increase `inner_steps` in your config
- Try: Increase `n_support` (more support examples per task)
- Check: Task distribution variance might be too high

**Issue: NaN losses during training**
- Try: Reduce learning rates (`inner_lr`, `meta_lr`)
- Try: Enable `first_order: true` in config for stability
- Check: Gradient clipping is enabled (happens automatically)

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
