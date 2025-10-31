## Overview

The codebase validates the central theoretical prediction:

```
Gap(P, K) ≥ c_quantum * σ²_S * (1 - e^(-μηK))
```

Where:
- **Gap(P, K)**: Optimality gap between meta-learning and robust control
- **σ²_S**: Control-relevant task variance
- **K**: Number of adaptation steps
- **μ**: PL constant (related to spectral gap Δ)
- **c_quantum**: System-dependent constant

## Quick Start

### 1. Train Policies (Prerequisites)

First, train the meta-learned and robust baseline policies:

```bash
# Train meta-learned policy (MAML)
cd experiments
python train_meta.py --config ../configs/experiment_config.yaml --output checkpoints/maml_best.pt

# Train robust baseline
python train_robust.py --config ../configs/experiment_config.yaml --output checkpoints/robust_best.pt
```

### 2. Generate All Results

Run the master script to generate all figures and tables:

```bash
cd experiments/paper_results
python generate_all_results.py --meta_path ../checkpoints/maml_best.pt \
                                --robust_path ../checkpoints/robust_best.pt \
                                --n_tasks 100
```

## Individual Experiments

You can also run experiments individually:

### Experiment 1: Gap vs Adaptation Steps K

Validates: **Gap(P, K) ∝ (1 - e^(-μηK))**

```bash
python experiment_gap_vs_k.py
```

### Experiment 2: Gap vs Task Variance σ²_S

Validates: **Gap(P, K) ∝ σ²_S**

```bash
python experiment_gap_vs_variance.py
```

### Experiment 3: Physics Constants Validation

Validates: **Empirical constants within 2× of theoretical predictions**

```bash
python experiment_constants_validation.py
```

## Configuration

Modify parameters in the scripts or pass via config:

### System Parameters
- `num_qubits`: Qubit dimension (default: 1, d=2)
- `num_controls`: Number of control Hamiltonians (default: 2)
- `num_segments`: Control discretization (default: 20)
- `evolution_time`: Gate time T (default: 1.0)

### Task Distribution
- `alpha_range`: Spectral exponent [0.5, 2.0]
- `A_range`: Noise amplitude [0.05, 0.3]
- `omega_c_range`: Cutoff frequency [2.0, 8.0] rad/s

### Meta-Learning
- `inner_lr`: Inner loop learning rate η (default: 0.01)
- `k_values`: Adaptation steps to test (default: [1, 2, 3, 5, 7, 10, 15, 20])

