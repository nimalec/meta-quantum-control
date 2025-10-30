# Meta-RL for Quantum Control: Robust Experiments Structure & Analysis

## Executive Summary

The codebase implements a meta-learning framework for quantum control with robust baseline comparisons. It uses MAML (Model-Agnostic Meta-Learning) as the primary approach, with multiple baseline implementations including GRAPE (Gradient Ascent Pulse Engineering). The paper results validate theoretical optimality gap bounds through systematic experiments.

---

## PAPER RESULTS GENERATION PIPELINE


#### Experiment 1: Gap vs Adaptation Steps K
**File:** `experiment_gap_vs_k.py`

- **Theory:** Gap(K) = gap_max × (1 - exp(-μηK))
- **Validates:** Exponential decay with adaptation steps
- **K values tested:** [1, 2, 3, 5, 7, 10, 15, 20]
- **Target R²:** 0.96 for exponential fit
- **Key metrics:**
  - gap_max: Maximum achievable gap
  - mu_eta: Decay rate constant
  - Confidence intervals

#### Experiment 2: Gap vs Task Variance σ²_S
**File:** `experiment_gap_vs_variance.py`

- **Theory:** Gap(σ²_S) ∝ σ²_S (linear relationship)
- **Validates:** Linear dependency on task variance
- **Variance levels:** [0.001, 0.002, 0.004, 0.008, 0.016]
- **Fixed K:** 5 adaptation steps
- **Target R²:** 0.94 for linear fit
- **Key metrics:**
  - slope: Proportionality constant
  - intercept: Baseline gap
  - Worst-case performance

#### Experiment 3: Constants Validation
**File:** `experiment_constants_validation.py`

- **Estimates:** C_sep, μ, L, L_F, c_quantum
- **Validates:** Theory vs empirical constants
- **Ratio bounds:** 0.5-2.0× for acceptance
- **Distribution analysis:** Histogram plots
- **Sample size:** 50 tasks

#### Experiment 4: System Scaling
**File:** `experiment_system_scaling.py`

- Tests performance scaling with system size
- Multiple qubit systems (1, 2 qubits)
- Gate fidelity analysis

---

##  CURRENT OPTIMIZATION METHODS

### A. Primary Method: MAML (Model-Agnostic Meta-Learning)

**Location:** `metaqctrl/meta_rl/maml.py`

**Architecture:**
```
Meta-initialization π₀ → Fast adaptation K steps → Task-specific policy π_task
```

**Inner Loop (Adaptation):**
- K gradient steps on support set
- Learning rate: α = 0.01
- Optimizer: SGD

**Outer Loop (Meta-update):**
- Meta-learning rate: β = 0.001-0.01
- Optimizer: Adam
- Loss: Task-specific fidelity
- Second-order available via `higher` library

**Implementation Details:**
- First-order MAML (FOMAML) by default for stability
- Supports second-order MAML with `higher` library
- Deepcopy-based task cloning for separation

**Training:**
```bash
python experiments/train_meta.py --config configs/experiment_config.yaml
```

### B. Baseline 1: Robust Control (Average & Minimax)

**Location:** `metaqctrl/baselines/robust_control.py` (lines 26-323)

**RobustPolicy Class:**
- No task-specific adaptation
- Three variants:

**1. Average Robust (minimax loss):**
```
π_rob = argmin_π E_θ[L(π, θ)]
```

**2. Minimax Robust:**
```
π_rob = argmin_π max_θ L(π, θ)
```
Implementation: Smooth max via LogSumExp with β=10.0

**3. CVaR Robust (Conditional Value at Risk):**
```
π_rob = argmin_π CVaR_α[L(π, θ)]
```
Minimizes worst α fraction of losses (α=0.1 default)

**Trainer:** `RobustTrainer` class
- Batch training over tasks
- Validation with best model selection
- Early stopping capability

**Training:**
```bash
python experiments/train_robust.py --config configs/experiment_config.yaml
```

### C. Baseline 2: GRAPE (Gradient Ascent Pulse Engineering)

**Location:** `metaqctrl/baselines/robust_control.py` (lines 435-735)

**GRAPEOptimizer Class:**

**Single-Task Optimization:** `optimize()`
- Direct pulse sequence optimization (no policy network)
- Finite-difference gradients (non-differentiable simulation)
- Control bounds enforcement via tanh transformation
- Methods: Adam, LBFGS, SGD

**Robust Optimization:** `optimize_robust()`
- Multi-task optimization with aggregated objectives
- Average fidelity or worst-case minimization
- Gradient averaging across task distribution

**Parameters:**
- n_segments: 20 (pulse discretization)
- n_controls: 2 (X, Y channels)
- control_bounds: (-5.0, 5.0)
- learning_rate: 0.1
- T: 1.0 (evolution time)

**Gradient Computation:**
```
∂F/∂u_i ≈ (F(u + ε*e_i) - F(u)) / ε
```
- epsilon: 1e-4 (finite difference step)
- Full-dimensional gradient per element
- O(n_segments × n_controls) evaluations per iteration




### D. Classical Optimal Control: H₂/H∞

**Location:** `metaqctrl/baselines/robust_control.py` (lines 326-385)

- Riccati equation solver via scipy
- LQR as approximation to H∞
- Classical control theory baseline
- Note: Requires system linearization (limited for quantum)

---

## EXPERIMENT CONFIGURATION

### A. Master Configuration: `configs/experiment_config.yaml`

```yaml
# System Setup
psd_model: 'one_over_f'        # Noise model type
horizon: 1.0                    # Evolution time
target_gate: 'pauli_x'          # Target quantum gate

# Task Distribution
task_dist_type: 'uniform'
alpha_range: [0.5, 2.0]         # Spectral exponent
A_range: [0.001, 0.01]          # PSD amplitude
omega_c_range: [2.0, 10.0]      # Cutoff frequency

# Policy Network
task_feature_dim: 3             # (α, A, ωc)
hidden_dim: 128                 # Network width
n_hidden_layers: 1              # Depth
n_segments: 20                  # Pulse discretization
n_controls: 2                   # X, Y channels
output_scale: 2.0               # Control amplitude scale

# MAML Hyperparameters
inner_lr: 0.01                  # Adaptation learning rate
inner_steps: 1                  # K (adaptation steps)
meta_lr: 0.01                   # Meta-learning rate
first_order: true               # Use FOMAML

# Training
n_iterations: 50
tasks_per_batch: 5
n_support: 2
n_query: 2

# Robust Baseline
robust_type: 'minimax'
robust_iterations: 1000
robust_tasks_per_batch: 16

# Gap Experiments
gap_n_samples: 100
gap_K_values: [1, 3, 5, 10, 20]
gap_variance_range: [0.01, 0.05, 0.1, 0.2, 0.5]
```

### B. Configuration Usage

**In Training Scripts:**
```python
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Access parameters
inner_lr = config.get('inner_lr', 0.01)
robust_type = config.get('robust_type', 'minimax')
```

**Dynamic Configuration in Experiments:**
```python
# experiment_gap_vs_k.py
config = {
    'num_qubits': 1,
    'num_controls': 2,
    'num_segments': 20,
    'evolution_time': 1.0,
    'target_gate': 'hadamard',
    'policy_hidden_dims': [128, 128, 128],
    'inner_lr': 0.01,
    'task_dist': {
        'alpha_range': [0.5, 2.0],
        'A_range': [0.05, 0.3],
        'omega_c_range': [2.0, 8.0]
    }
}
```

---

## METHOD SELECTION & SWITCHING

### A. Training Pipeline

**Step 1: Train Meta-Learner**
```python
# train_meta.py
from metaqctrl.meta_rl.maml import MAML, MAMLTrainer

maml = MAML(policy, inner_lr=0.01, meta_lr=0.001)
trainer = MAMLTrainer(maml, ...)
trainer.train(n_iterations=5000)
```

**Step 2: Train Robust Baseline**
```python
# train_robust.py
from metaqctrl.baselines.robust_control import RobustPolicy, RobustTrainer

robust_policy = RobustPolicy(policy, robust_type='minimax')
trainer = RobustTrainer(robust_policy, ...)
trainer.train(n_iterations=1000)
```

**Step 3: Evaluate Gap**
```python
# Paper results experiments
gap_computer = OptimalityGapComputer(env, fidelity_fn)
gaps = gap_computer.compute_gap(
    meta_policy=meta_model,
    robust_policy=robust_model,
    task_distribution=test_tasks,
    K=5  # adaptation steps
)
```

### B. Robust Type Selection

**In `train_robust.py` (lines 100-107):**
```python
robust_type = config.get('robust_type', 'minimax')
robust_policy = RobustPolicy(
    policy=policy,
    learning_rate=config.get('meta_lr', 0.001),
    robust_type=robust_type,
    device=device
)
```

**Available Options:**
- `'average'`: Standard empirical risk minimization
- `'minimax'`: Worst-case optimization (LogSumExp)
- `'cvar'`: Conditional value at risk

**How It's Implemented:**
- `RobustPolicy.train_step()` dispatches to:
  - `_average_robust_loss()` → simple mean
  - `_minimax_robust_loss()` → LogSumExp smoothing
  - `_cvar_robust_loss()` → Top-α quantile

### C. Method Comparison in Paper Results

**`generate_all_results.py` Structure:**
```python
# Load both models
meta_policy = load_model(args.meta_path)
robust_policy = load_model(args.robust_path)

# Experiment 1: Gap vs K (uses both)
gap_vs_k = run_gap_vs_k_experiment(
    meta_policy, robust_policy, config, ...
)

# Experiment 2: Gap vs Variance (uses both)
gap_vs_var = run_gap_vs_variance_experiment(
    meta_policy, robust_policy, config, ...
)

# Generates comparison table
generate_results_table(gap_vs_k, gap_vs_var, constants)
```

