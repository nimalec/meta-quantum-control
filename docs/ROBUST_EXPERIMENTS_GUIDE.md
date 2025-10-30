# Meta-RL for Quantum Control: Robust Experiments Structure & Analysis

## Executive Summary

The codebase implements a meta-learning framework for quantum control with robust baseline comparisons. It uses MAML (Model-Agnostic Meta-Learning) as the primary approach, with multiple baseline implementations including GRAPE (Gradient Ascent Pulse Engineering). The paper results validate theoretical optimality gap bounds through systematic experiments.

---

## 1. DIRECTORY STRUCTURE & KEY FILES

### A. Robust Experiment Files

**Location:** `/Users/nimalec/Documents/metarl_project/meta-quantum-control/`

```
.
├── experiments/
│   ├── train_meta.py                    # Main meta-learning training script
│   ├── train_robust.py                  # Robust baseline training
│   ├── eval_gap.py                      # Gap evaluation utilities
│   └── paper_results/                   # Paper figure generation
│       ├── generate_all_results.py      # Master orchestration script
│       ├── experiment_gap_vs_k.py       # Figure 1: Gap vs adaptation steps
│       ├── experiment_gap_vs_variance.py # Figure 2: Gap vs task variance
│       ├── experiment_constants_validation.py # Table 1: Constants
│       ├── experiment_system_scaling.py # System size scaling experiments
│       ├── plot_fidelity_vs_k.py        # Advanced fidelity plotting
│       └── plot_validation_metrics.py   # Validation metric plots
│
├── metaqctrl/
│   ├── baselines/
│   │   └── robust_control.py            # **KEY**: All baseline implementations
│   ├── meta_rl/
│   │   ├── maml.py                      # MAML meta-learning algorithm
│   │   └── policy.py                    # PulsePolicy neural network
│   ├── theory/
│   │   ├── optimality_gap.py            # Gap computation & theory
│   │   ├── quantum_environment.py       # Unified simulation interface
│   │   └── physics_constants.py         # Constant estimation
│   └── quantum/
│       ├── lindblad.py                  # Lindblad equation simulator
│       ├── lindblad_torch.py            # Differentiable simulator
│       ├── noise_models.py              # Task & noise distributions
│       └── gates.py                     # Quantum gates & fidelity
│
├── configs/
│   └── experiment_config.yaml           # Master configuration file
│
└── examples/
    └── grape_example.py                 # GRAPE usage demonstration
```

---

## 2. PAPER RESULTS GENERATION PIPELINE

### A. Master Script: `generate_all_results.py`

**Purpose:** Orchestrates all paper result experiments

**Input:** Trained models (meta-policy and robust-policy checkpoints)

**Outputs:** 
- Figure 1: Gap vs K (exponential fit validation, R² ≈ 0.96)
- Figure 2: Gap vs σ²_S (linear fit validation, R² ≈ 0.94)
- Table 1: Physics constants and validation metrics
- Summary statistics and validation checks

**Command:**
```bash
python experiments/paper_results/generate_all_results.py \
    --meta_path checkpoints/maml_best.pt \
    --robust_path checkpoints/robust_best.pt \
    --output_dir results/paper \
    --n_tasks 100
```

### B. Individual Experiments

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

## 3. CURRENT OPTIMIZATION METHODS

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

**Example Usage:**
```python
grape = GRAPEOptimizer(n_segments=20, n_controls=2)
optimal_controls = grape.optimize(
    simulate_fn=simulate_fidelity,
    task_params=task_params,
    max_iterations=100
)
```

**See:** `examples/grape_example.py` for complete demo

### D. Baseline 3: Domain Randomization

**Location:** `metaqctrl/baselines/robust_control.py` (lines 388-432)

- Policy training with random task perturbations
- Randomization strength: 0.1 (10% multiplicative noise)
- Similar to robust average but with explicit augmentation
- Implementation: Add random perturbations to task parameters during training

### E. Classical Optimal Control: H₂/H∞

**Location:** `metaqctrl/baselines/robust_control.py` (lines 326-385)

- Riccati equation solver via scipy
- LQR as approximation to H∞
- Classical control theory baseline
- Note: Requires system linearization (limited for quantum)

---

## 4. EXPERIMENT CONFIGURATION

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

## 5. METHOD SELECTION & SWITCHING

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

---

## 6. EXISTING GRAPE IMPLEMENTATION

### A. GRAPE Baseline Structure

**File:** `/Users/nimalec/Documents/metarl_project/meta-quantum-control/metaqctrl/baselines/robust_control.py` (lines 435-735)

**Key Features:**
1. **Direct Pulse Optimization:** No policy network, just control sequences
2. **Finite-Difference Gradients:** Non-differentiable simulator support
3. **Multiple Optimizers:** Adam (default), LBFGS, SGD
4. **Bound Enforcement:** tanh-based control magnitude scaling
5. **Single & Robust Modes:** Single-task and multi-task optimization

### B. GRAPE Integration Points

**Simulation Hook:**
```python
def optimize(
    self,
    simulate_fn: Callable,           # User-provided simulator
    task_params,                      # Task specification
    max_iterations: int = 100,
    tolerance: float = 1e-6
) -> np.ndarray:
    # simulate_fn takes (controls, task_params) → float (fidelity)
```

**Example Simulator:**
```python
def simulate_fidelity(controls, task_params):
    L_ops = psd_to_lindblad.get_lindblad_operators(task_params)
    sim = LindbladSimulator(H0, H_controls, L_ops)
    rho_final, _ = sim.evolve(rho0, controls, T=1.0)
    return state_fidelity(rho_final, target_state)
```

### C. Current Limitations

1. **Finite Differences Only:** No automatic differentiation through simulator
2. **No Policy Learning:** Optimizes controls directly, not learned initialization
3. **No Meta-Learning Integration:** Treats each task independently
4. **Computational Cost:** O(n_segments × n_controls) evals/iteration

### D. Example Usage

**From `examples/grape_example.py`:**
```python
grape = GRAPEOptimizer(
    n_segments=20,
    n_controls=2,
    T=1.0,
    control_bounds=(-3.0, 3.0),
    learning_rate=0.1,
    method='adam'
)

optimal_controls = grape.optimize(
    simulate_fn=simulate_fidelity,
    task_params=task_params,
    max_iterations=50,
    verbose=True
)

# Test robust optimization
robust_controls = grape.optimize_robust(
    simulate_fn=simulate_fidelity,
    task_distribution=[task1, task2, task3],
    max_iterations=30,
    robust_type='average'
)
```

---

## 7. HOW TO ADD GRAPE BASELINE TO PAPER RESULTS

### A. Integration Strategy

To integrate GRAPE baseline into paper results experiments:

**1. Modify `generate_all_results.py`:**

```python
# Add GRAPE optimization section before paper generation
from metaqctrl.baselines.robust_control import GRAPEOptimizer

# For Experiment 1: Gap vs K
grape_controls_per_k = {}
for K in k_values:
    grape = GRAPEOptimizer(...)
    optimal_controls = grape.optimize(
        simulate_fn=create_grape_simulator(config),
        task_params=test_tasks[0],
        max_iterations=K * 10  # Scale iterations with K
    )
    grape_controls_per_k[K] = optimal_controls
```

**2. Create GRAPE-Specific Experiment:**

New file: `experiment_grape_baseline.py`

```python
def run_grape_baseline_experiment(
    config: Dict,
    k_values: List[int] = [1, 3, 5, 10, 20],
    n_test_tasks: int = 100,
    output_dir: str = "results/grape_baseline"
) -> Dict:
    """
    Evaluate GRAPE as baseline for comparison.
    
    Returns:
        results: Dict with GRAPE fidelities per K value
    """
    # Create simulator function
    def simulate_fn(controls, task_params):
        # Get Lindblad operators
        L_ops = psd_to_lindblad.get_lindblad_operators(task_params)
        sim = LindbladSimulator(H0, H_controls, L_ops)
        rho_final, _ = sim.evolve(rho0, controls, T=config['horizon'])
        return state_fidelity(rho_final, target_state)
    
    # For each K value, do GRAPE optimization
    results = {}
    for K in k_values:
        grape_fidelities = []
        for task_params in test_tasks:
            grape = GRAPEOptimizer(
                n_segments=config['n_segments'],
                n_controls=config['n_controls'],
                T=config['horizon'],
                learning_rate=0.1
            )
            optimal_controls = grape.optimize(
                simulate_fn=simulate_fn,
                task_params=task_params,
                max_iterations=K * 20,  # More iterations for worse K
                tolerance=1e-6
            )
            final_fidelity = simulate_fn(optimal_controls, task_params)
            grape_fidelities.append(final_fidelity)
        
        results[K] = {
            'fidelities': grape_fidelities,
            'mean': np.mean(grape_fidelities),
            'std': np.std(grape_fidelities),
            'min': np.min(grape_fidelities)
        }
    
    return results
```

**3. Update Master Script:**

```python
# In generate_all_results.py main()
grape_results = run_grape_baseline_experiment(
    config=config,
    k_values=[1, 2, 3, 5, 7, 10, 15, 20],
    n_test_tasks=args.n_tasks,
    output_dir=f"{args.output_dir}/grape_baseline"
)

# Compare all three methods
plot_method_comparison(
    meta=results_gap_k,
    robust=robust_results,
    grape=grape_results,
    output_path=f"{args.output_dir}/method_comparison.pdf"
)
```

### B. Performance Metrics to Track

**For Each K Value:**
- Mean fidelity: E[F]
- Std deviation: σ(F)
- Min fidelity: min(F)
- Max fidelity: max(F)
- Convergence iterations
- Wall-clock time

**Comparison Ratios:**
- GRAPE vs Robust baseline
- GRAPE vs Meta-learned (K adaptation steps)
- Adaptation cost per additional K

### C. Files That Need Modification

| File | Changes | Reason |
|------|---------|--------|
| `generate_all_results.py` | Add GRAPE experiment call | Master orchestration |
| `paper_results/experiment_grape_baseline.py` | NEW FILE | Dedicated GRAPE experiment |
| `paper_results/plot_validation_metrics.py` | Add GRAPE comparison plots | Visualization |
| `metaqctrl/baselines/robust_control.py` | Enhance GRAPE docs | Already has implementation |

---

## 8. SUMMARY TABLE: OPTIMIZATION METHODS

| Method | Location | Single/Multi-Task | Adaptation | Learnable | Used in Paper |
|--------|----------|------------------|-----------|-----------|---------------|
| **MAML** | `meta_rl/maml.py` | Multi | Yes (K steps) | Yes (π₀) | Yes (main) |
| **Robust Avg** | `baselines/robust_control.py:90-113` | Multi | No | Yes | Yes (baseline) |
| **Robust Minimax** | `baselines/robust_control.py:115-144` | Multi | No | Yes | Yes (baseline) |
| **Robust CVaR** | `baselines/robust_control.py:146-176` | Multi | No | Yes | Tested |
| **GRAPE Single** | `baselines/robust_control.py:492-586` | Single | No (direct opt) | No | Example only |
| **GRAPE Robust** | `baselines/robust_control.py:635-718` | Multi | No (direct opt) | No | Example only |
| **Domain Randomization** | `baselines/robust_control.py:388-432` | Multi | No | Yes | Not in paper |
| **H2/H∞ Control** | `baselines/robust_control.py:326-385` | Single | No | No | Not used |

---

## 9. EXECUTION FLOW FOR PAPER RESULTS

```
1. Train Models (Sequential)
   ├─ train_meta.py → checkpoints/maml_best.pt
   └─ train_robust.py → checkpoints/robust_best.pt

2. Generate Paper Results (Parallel in generate_all_results.py)
   ├─ experiment_gap_vs_k.py
   │  ├─ Load meta & robust policies
   │  ├─ For K in [1,2,3,5,7,10,15,20]:
   │  │  └─ Evaluate both on 100 test tasks
   │  └─ Fit exponential model
   │
   ├─ experiment_gap_vs_variance.py
   │  ├─ Create task distributions with varying σ²_S
   │  ├─ Evaluate both on each distribution (K=5 fixed)
   │  └─ Fit linear model
   │
   └─ experiment_constants_validation.py
      ├─ Estimate physics constants
      ├─ Compare theory vs empirical
      └─ Generate validation metrics

3. Generate Summary Table
   └─ Compare all metrics, check R² > 0.90

4. Output Artifacts
   ├─ Figures: gap_vs_k.pdf, gap_vs_variance.pdf
   ├─ Data: results.json for each experiment
   └─ Table: summary_table.txt
```

---

## 10. KEY ARCHITECTURAL INSIGHTS

### A. Loss Function Architecture

**Unified Loss (all methods):**
```python
def loss_fn(policy, task_data):
    """
    1. Policy outputs controls: π(θ) → u(t)
    2. Simulator evolves: ρ(t) = evolve(H, u, t)
    3. Fidelity computed: F = ⟨ψ_target|ρ|ψ_target⟩
    4. Loss returned: L = 1 - F
    """
    task_params = task_data['task_params']
    loss = env.compute_loss_differentiable(policy, task_params, device)
    return loss
```

### B. Task Sampling Strategy

**Three-Way Split for Reproducibility:**
```python
def task_sampler(n_tasks, split, task_dist, rng):
    if split == 'train':
        seed_offset = 0
    elif split == 'val':
        seed_offset = 100000
    else:  # test
        seed_offset = 200000
    
    local_rng = np.random.default_rng(rng.integers(0, 1000000) + seed_offset)
    return task_dist.sample(n_tasks, local_rng)
```

### C. Gap Computation

**Empirical Definition:**
```
Gap(P, K) = E_θ~P[F(AdaptK(π_meta; θ), θ)] - E_θ~P[F(π_robust, θ)]
```

**Adaptive Policy:**
```python
def adapt_policy(policy, task_params, K, lr):
    adapted = deepcopy(policy)
    optimizer = optim.SGD(adapted.parameters(), lr=lr)
    for _ in range(K):
        loss = env.compute_loss_differentiable(adapted, task_params, device)
        loss.backward()
        optimizer.step()
    return adapted
```

---

## 11. QUICK START: WHERE TO ADD GRAPE

### Location in Codebase

**Already Implemented:** `metaqctrl/baselines/robust_control.py` lines 435-735
- No new code needed for basic GRAPE
- Just needs integration into paper results pipeline

### Where to Hook GRAPE Results

**Option 1: Add to existing figure generation**
```python
# In experiment_gap_vs_k.py, add:
from metaqctrl.baselines.robust_control import GRAPEOptimizer

# After computing meta & robust gaps:
grape_gaps = []
for k_val in k_values:
    grape_fidelities = compute_grape_fidelities(
        k_val, test_tasks, config
    )
    grape_gaps.append(mean_fidelity - robust_baseline)

# Plot all three curves
plt.plot(k_values, meta_gaps, label='MAML')
plt.plot(k_values, robust_gaps, label='Robust')
plt.plot(k_values, grape_gaps, label='GRAPE')
```

**Option 2: Dedicated GRAPE experiment**
```python
# New file: experiment_grape_baseline.py
# Parallel to existing experiments
# Run independently and compare

# In generate_all_results.py:
grape_results = run_grape_baseline_experiment(...)
compare_all_methods(meta_results, robust_results, grape_results)
```

### Required Simulator Function

```python
def create_grape_simulator(config, psd_to_lindblad, H0, H_controls, target_state):
    """Factory for GRAPE simulate_fn"""
    def simulate_fidelity(controls, task_params):
        L_ops = psd_to_lindblad.get_lindblad_operators(task_params)
        sim = LindbladSimulator(H0, H_controls, L_ops, method='RK45')
        rho0 = np.eye(2) / 2  # Initial state
        rho_final, _ = sim.evolve(rho0, controls, T=config['horizon'])
        return state_fidelity(rho_final, target_state)
    return simulate_fidelity
```

