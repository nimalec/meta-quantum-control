# Meta-Reinforcement Learning for Adaptive Quantum Control
Author: Nima Leclerc (nleclerc@mitre.org)-- Quantum Research Scientist at MITRE

A research implementation of first-order Model-Agnostic Meta-Learning (MAML) for quantum state control under noise. This framework trains a meta-learned policy initialization that rapidly adapts to new quantum noise environments with minimal gradient steps.

## Overview

This project combines meta-reinforcement learning with quantum control theory, enabling control policies to adapt quickly to different colored noise profiles (1/f, Lorentzian, etc.) by leveraging task-specific structure. The key innovation is a fully differentiable Lindblad master equation simulator that allows end-to-end gradient-based meta learned optimization of optimal pulse sequences to calculate an "adaptation gap" quantity.   


## Project Structure

```
meta-quantum-control/
├── metaqctrl/                    # Main package
│   ├── meta_rl/                  # Meta-learning algorithms
│   │   ├── maml.py               # MAML implementation
│   │   └── policy.py             # Neural network policies
│   ├── quantum/                  # Quantum simulation
│   │   ├── lindblad.py           # NumPy Lindblad simulator
│   │   ├── lindblad_torch.py     # Differentiable PyTorch simulator
│   │   ├── gates.py              # Fidelity computation
│   │   ├── noise_models_v2.py    # Noise PSD models
│   │   └── noise_adapter.py      # PSD to Lindblad conversion
│   ├── theory/                   # Theoretical analysis
│   │   ├── quantum_environment.py    # Unified simulation interface
│   │   ├── optimality_gap.py         # Gap analysis
│   │   └── physics_constants.py      # Spectral gap computation
│   └── utils/                    # Utilities
│       ├── checkpoint_utils.py   # Model saving/loading
│       └── plot_training.py      # Visualization
├── experiments/                  # Reproducible experiment scripts
├── configs/                      # Configuration files
├── checkpoints/                  # Saved model weights
└── tests/                        # Unit tests
```

## Installation

### Prerequisites

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Install uv (Mac/Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install uv (Windows)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Setup

```bash
# Clone the repository 
cd meta-quantum-control

# Install dependencies
export UV_NATIVE_TLS=true
uv sync --all-groups

# Activate the virtual environment
source .venv/bin/activate  # Mac/Linux
.venv\Scripts\activate     # Windows
```

### Dependencies

Core dependencies (automatically installed):
- PyTorch >= 2.0.0
- NumPy >= 1.24.0, SciPy >= 1.10.0
- JAX >= 0.4.0 (optional GPU acceleration)
- higher >= 0.2.1 (differentiable inner loop)
- QuTiP >= 4.7.0 (quantum utilities)
- Hydra >= 1.3.0 (configuration management)

## Quick Start

### Training a Meta-Policy

```bash
cd experiments/fig_5_meta_training
python train_meta.py --config ../../configs/experiment_config.yaml
```

This will:
1. Train a FOMAML policy for single-qubit quantum control
2. Save checkpoints to `checkpoints/`
3. Log training metrics to `checkpoints/training_history.json`
4. Display training progress with pre/post-adaptation validation metrics

### Evaluating a Trained Policy

```python
from metaqctrl.utils.checkpoint_utils import load_policy_from_checkpoint
from metaqctrl.theory.quantum_environment import QuantumEnvironment

# Load trained policy
policy = load_policy_from_checkpoint("checkpoints/best_policy.pt")

# Create environment with specific noise parameters
env = QuantumEnvironment(
    alpha=1.0,      # Spectral exponent
    A=0.1,          # Noise amplitude
    omega_c=50.0    # Cutoff frequency
)

# Generate control pulses and evaluate
task_features = torch.tensor([[1.0, 0.1, 50.0]])
controls = policy(task_features)
fidelity = env.evaluate_controls(controls)
```

## Configuration

Edit `configs/experiment_config.yaml` to customize training:

```yaml
# Random seed for reproducibility
seed: 42

# Quantum System
psd_model: 'one_over_f'       # Noise model: 'one_over_f', 'lorentzian', 'double_exp'
horizon: 1                     # Evolution time (arbitrary units)
target_gate: 'pauli_x'         # Target: 'pauli_x', 'pauli_y', 'hadamard'
num_qubits: 1

# Task Distribution (noise parameter ranges)
task_dist_type: 'uniform'
alpha_range: [0.1, 2.0]        # Spectral exponent range
A_range: [0.01, 10.0]          # Noise amplitude range
omega_c_range: [1.0, 300.0]    # Cutoff frequency range

# Policy Network Architecture
task_feature_dim: 3            # Input: (alpha, A, omega_c)
hidden_dim: 128
n_hidden_layers: 2
n_segments: 60                 # Number of pulse segments
n_controls: 2                  # X and Y control channels
activation: 'tanh'

# MAML Hyperparameters
inner_lr: 0.01                 # Inner loop learning rate
inner_steps: 5                 # Adaptation steps (K)
meta_lr: 0.001                 # Meta learning rate
first_order: true              # Use FOMAML (recommended)

# Training
n_iterations: 2000
tasks_per_batch: 32
n_support: 10                  # Support set size per task
n_query: 10                    # Query set size per task
val_interval: 10
val_tasks: 20

# Quantum Simulation
dt_training: 0.01              # Integration time step
use_rk4_training: true         # Use RK4 (recommended for stability)

# Checkpointing
save_dir: 'checkpoints'
```

## Reproducing Paper Figures

All experiments are organized by figure number. Each script uses fixed random seeds for reproducibility.

### Main Figures

| Figure | Script | Description |
|--------|--------|-------------|
| Fig. 1 | `experiments/fig_1_overview/fig1.py` | System overview schematic |
| Fig. 2 | `experiments/fig_2_lemma_validation/lemma_validation.py` | Theoretical lemma validation |
| Fig. 3 | `experiments/fig_3_adaptation_gap_analysis/generate_adaptation_gap_figure.py` | Adaptation gap analysis |
| Fig. 4 | `experiments/fig_4_adaptation_dynamics/adapt_plots/adaptation_dynamics_figure.py` | Adaptation dynamics over K steps |
| Fig. 5 | `experiments/fig_5_meta_training/train_meta.py` | Meta-training results |

### Appendix Figures

| Figure | Script | Description |
|--------|--------|-------------|
| Fig. A1-A2 | `experiments/figs_appendix_classical/fig_a1_a2_lqr.py` | Classical LQR baseline |
| Fig. A3 | `experiments/figs_appendix_ablations/fig_a3_adaptation_lr/generate_lr_sensitivity_figure.py` | Inner learning rate sensitivity |
| Fig. A4 | `experiments/figs_appendix_ablations/fig_a4_adaptation_steps/generate_adaptation_steps_figure.py` | Adaptation steps (K) analysis |
| Fig. A5 | `experiments/figs_appendix_ablations/fig_a5_baselines/generate_baseline_comparison_figure.py` | Baseline comparisons |


## Algorithm Overview

The MAML training loop:

```
for iteration in 1..N:
    # Sample batch of tasks (noise configurations)
    tasks = sample_tasks(task_distribution, batch_size)

    for task in tasks:
        # Clone meta-parameters
        adapted_policy = clone(meta_policy)

        # Inner loop: K gradient steps on support set
        for k in 1..K:
            loss = compute_loss(adapted_policy, task.support)
            adapted_policy = gradient_step(adapted_policy, loss, inner_lr)

        # Evaluate adapted policy on query set
        query_loss = compute_loss(adapted_policy, task.query)
        accumulate_meta_gradient(meta_policy, query_loss)

    # Meta-update
    meta_policy = gradient_step(meta_policy, meta_loss, meta_lr)
```

Where `compute_loss`:
1. Generates control pulses from policy
2. Simulates Lindblad dynamics with noise
3. Computes gate fidelity
4. Returns `loss = 1 - fidelity`


## Citation

```bibtex
@inproceedings{leclerc2025meta,
  title={Meta-Reinforcement Learning for Quantum Control},
  author={Leclerc, Nima and Miller, Chris and Brawand, Nicholas},
  year={2025}
}
```


## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Run tests (`make test`)
4. Submit a pull request

