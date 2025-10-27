# Meta-RL for Quantum Control: Documentation Index

This repository contains comprehensive documentation for understanding the robust experiments and optimization methods used in the meta-learning quantum control research.

## Documentation Files

### 1. **ROBUST_EXPERIMENTS_GUIDE.md** (23 KB - COMPREHENSIVE)
The definitive guide to robust experiments structure and methodology.

**Contains:**
- Complete directory structure and file locations
- Paper results generation pipeline (sections 1-2)
- Detailed breakdown of all 8 optimization methods (section 3)
- Experiment configuration and setup (section 4)
- Method selection and switching mechanisms (section 5)
- Existing GRAPE implementation details (section 6)
- Integration guide for adding GRAPE to paper results (section 7)
- Method comparison table (section 8)
- Execution flow diagrams (section 9)
- Architectural insights (section 10)
- Quick start for GRAPE integration (section 11)

**Best for:** Understanding the complete system architecture and methodology

---

### 2. **QUICK_REFERENCE.txt** (9.4 KB - QUICK LOOKUP)
Fast reference guide for common operations and configurations.

**Contains:**
- Key files at a glance
- Paper results workflow (step-by-step)
- Output file structure
- Optimization methods comparison table
- Configuration parameters with values
- Robust type selection guide
- Experiment details summary
- GRAPE baseline integration options
- Code snippets for common tasks
- Debugging tips

**Best for:** Quick lookups and day-to-day reference

---

### 3. **CODE_LOCATIONS.txt** (14 KB - PRECISE LOCATIONS)
Exact line-by-line code locations for all components.

**Contains:**
- Robust experiment file locations
- Paper results pipeline structure
- All optimization method implementations with line numbers
- RobustPolicy class detailed breakdown
- GRAPEOptimizer class structure (lines 435-735)
- MAML implementation details
- Core component file locations
- Configuration file structure
- Example file breakdown
- Key imports and dependencies
- Usage flow and modification points

**Best for:** Precise code navigation and implementation details

---

## Quick Navigation

### For Understanding Paper Results
1. Start: **QUICK_REFERENCE.txt** → "PAPER RESULTS WORKFLOW" section
2. Deep dive: **ROBUST_EXPERIMENTS_GUIDE.md** → Section 2 "PAPER RESULTS GENERATION PIPELINE"
3. Code details: **CODE_LOCATIONS.txt** → "PAPER RESULTS PIPELINE" section

### For Understanding Optimization Methods
1. Overview: **QUICK_REFERENCE.txt** → "OPTIMIZATION METHODS COMPARISON" table
2. Details: **ROBUST_EXPERIMENTS_GUIDE.md** → Section 3 "CURRENT OPTIMIZATION METHODS"
3. Code: **CODE_LOCATIONS.txt** → "OPTIMIZATION METHOD IMPLEMENTATIONS"

### For Adding GRAPE to Paper Results
1. Quick version: **QUICK_REFERENCE.txt** → "GRAPE BASELINE INTEGRATION"
2. Full guide: **ROBUST_EXPERIMENTS_GUIDE.md** → Section 7 "HOW TO ADD GRAPE BASELINE"
3. Code locations: **CODE_LOCATIONS.txt** → GRAPEOptimizer class details (lines 435-735)

### For Changing Configurations
1. Parameters: **QUICK_REFERENCE.txt** → "CONFIGURATION PARAMETERS"
2. How to use: **ROBUST_EXPERIMENTS_GUIDE.md** → Section 4 "EXPERIMENT CONFIGURATION"
3. File locations: **CODE_LOCATIONS.txt** → "CONFIGURATION FILES" and "MODIFICATION POINTS"

### For Running Experiments
1. Workflow: **QUICK_REFERENCE.txt** → "PAPER RESULTS WORKFLOW"
2. Detailed steps: **CODE_LOCATIONS.txt** → "USAGE FLOW"
3. System info: **ROBUST_EXPERIMENTS_GUIDE.md** → Section 9 "EXECUTION FLOW FOR PAPER RESULTS"

---

## Key File Locations Reference

### Training Scripts
- **Meta-learning:** `/experiments/train_meta.py`
- **Robust baseline:** `/experiments/train_robust.py`
- **Gap evaluation:** `/experiments/eval_gap.py`

### Paper Results Pipeline
- **Master script:** `/experiments/paper_results/generate_all_results.py`
- **Gap vs K:** `/experiments/paper_results/experiment_gap_vs_k.py`
- **Gap vs variance:** `/experiments/paper_results/experiment_gap_vs_variance.py`
- **Constants validation:** `/experiments/paper_results/experiment_constants_validation.py`
- **System scaling:** `/experiments/paper_results/experiment_system_scaling.py`
- **Plotting:** `/experiments/paper_results/plot_fidelity_vs_k.py`, `plot_validation_metrics.py`

### Optimization Methods
- **All baselines:** `/metaqctrl/baselines/robust_control.py`
  - RobustPolicy & RobustTrainer (lines 26-323)
  - GRAPE (lines 435-735)
  - Domain Randomization (lines 388-432)
  - H2/H∞ Control (lines 326-385)

### Core Components
- **MAML:** `/metaqctrl/meta_rl/maml.py`
- **Policy Network:** `/metaqctrl/meta_rl/policy.py`
- **Gap Computation:** `/metaqctrl/theory/optimality_gap.py`
- **Quantum Environment:** `/metaqctrl/theory/quantum_environment.py`
- **Quantum Simulation:** `/metaqctrl/quantum/lindblad.py`, `lindblad_torch.py`
- **Task Distributions:** `/metaqctrl/quantum/noise_models.py`

### Configuration
- **Master config:** `/configs/experiment_config.yaml`
- **Test config:** `/configs/test_config.yaml`

### Examples
- **GRAPE example:** `/examples/grape_example.py`

---

## Optimization Methods at a Glance

| Method | Type | Adaptation | File | Lines |
|--------|------|-----------|------|-------|
| MAML | Meta-learning | K steps | `meta_rl/maml.py` | 24-150+ |
| Robust Average | Baseline | None | `baselines/robust_control.py` | 90-113 |
| Robust Minimax | Baseline | None | `baselines/robust_control.py` | 115-144 |
| Robust CVaR | Baseline | None | `baselines/robust_control.py` | 146-176 |
| GRAPE Single | Direct optimization | None | `baselines/robust_control.py` | 492-586 |
| GRAPE Robust | Direct optimization | None | `baselines/robust_control.py` | 635-718 |
| Domain Randomization | Data augmentation | None | `baselines/robust_control.py` | 388-432 |
| H2/H∞ Control | Classical optimal | None | `baselines/robust_control.py` | 326-385 |

**Paper uses:** MAML (main) vs Robust Minimax (baseline)

---

## Common Tasks

### Train Models
```bash
# Train meta-learner
python experiments/train_meta.py --config configs/experiment_config.yaml

# Train robust baseline
python experiments/train_robust.py --config configs/experiment_config.yaml
```

### Generate Paper Results
```bash
python experiments/paper_results/generate_all_results.py \
    --meta_path checkpoints/maml_best.pt \
    --robust_path checkpoints/robust_best.pt \
    --output_dir results/paper \
    --n_tasks 100
```

### Run GRAPE Example
```bash
python examples/grape_example.py
```

### Modify Configuration
Edit `/configs/experiment_config.yaml`:
- Task distribution: `alpha_range`, `A_range`, `omega_c_range`
- MAML parameters: `inner_lr`, `inner_steps`, `meta_lr`
- Robust type: `robust_type` (options: average, minimax, cvar)
- Training: `n_iterations`, `tasks_per_batch`

---

## Key Concepts

### Optimization Gap
The performance difference between meta-learned (with K adaptation steps) and robust non-adaptive policies:
```
Gap(P, K) = E_θ~P[F(AdaptK(π_meta; θ), θ)] - E_θ~P[F(π_robust, θ)]
```

### Theoretical Validations
1. **Gap vs K:** Exponential decay: Gap(K) ∝ (1 - exp(-μηK))
   - Target: R² ≈ 0.96
   - File: `experiment_gap_vs_k.py`

2. **Gap vs σ²_S:** Linear relationship: Gap(σ²_S) ∝ σ²_S
   - Target: R² ≈ 0.94
   - File: `experiment_gap_vs_variance.py`

3. **Constants validation:** Theory vs empirical ratio bounds (0.5-2.0×)
   - Estimates: C_sep, μ, L, L_F, c_quantum
   - File: `experiment_constants_validation.py`

---

## Integration Guide: Adding GRAPE to Paper Results

**Current status:** GRAPE fully implemented in `robust_control.py` (lines 435-735)

**To integrate with paper results:**

Option A: Dedicated experiment
```python
# Create: experiments/paper_results/experiment_grape_baseline.py
def run_grape_baseline_experiment(config, k_values, n_test_tasks, output_dir):
    # Evaluates GRAPE for each K value
    # Returns: gaps per K for comparison
```

Option B: Add to existing figures
```python
# Modify: experiment_gap_vs_k.py
grape_fidelities = compute_grape_fidelities(k_val, test_tasks, config)
# Plot alongside MAML and Robust
```

See **ROBUST_EXPERIMENTS_GUIDE.md** Section 7 for complete implementation guide.

---

## Architecture Overview

```
Meta-Learning Framework
├─ MAML (inner + outer loop)
│  ├─ Inner: K gradient steps on support set
│  └─ Outer: Meta-update on validation set
│
├─ Baselines
│  ├─ Robust Control (average, minimax, CVaR)
│  ├─ GRAPE (direct pulse optimization)
│  ├─ Domain Randomization (data augmentation)
│  └─ Classical H2/H∞
│
├─ Evaluation
│  ├─ Optimality Gap Computation
│  └─ Theory Validation
│
└─ Quantum Simulation
   ├─ Lindblad Master Equation (numpy)
   ├─ Differentiable (PyTorch)
   └─ Noise Models (PSD to Lindblad)
```

---

## Documentation Statistics

- **ROBUST_EXPERIMENTS_GUIDE.md:** 11 major sections, 23 KB
- **QUICK_REFERENCE.txt:** 11 sections, 9.4 KB
- **CODE_LOCATIONS.txt:** 13 sections, 14 KB
- **Total:** ~46 KB of comprehensive documentation

---

## Getting Started Recommendations

**New to the codebase?**
1. Read: **QUICK_REFERENCE.txt** (5 min overview)
2. Understand: **ROBUST_EXPERIMENTS_GUIDE.md** Section 1-2 (15 min)
3. Explore: Code locations from **CODE_LOCATIONS.txt**

**Want to run experiments?**
1. Check: **QUICK_REFERENCE.txt** → "PAPER RESULTS WORKFLOW"
2. Configure: Edit `/configs/experiment_config.yaml`
3. Execute: Follow commands in **CODE_LOCATIONS.txt** → "USAGE FLOW"

**Want to add GRAPE baseline?**
1. Read: **ROBUST_EXPERIMENTS_GUIDE.md** Section 7
2. Implement: **CODE_LOCATIONS.txt** → "GRAPE BASELINE INTEGRATION"
3. Execute: Follow **QUICK_REFERENCE.txt** → "GRAPE BASELINE INTEGRATION"

---

**Last Updated:** October 26, 2024

**Questions?** See:
- **DEBUGGING & COMMON ISSUES** in QUICK_REFERENCE.txt
- **KEY ARCHITECTURAL INSIGHTS** in ROBUST_EXPERIMENTS_GUIDE.md
- **MODIFICATION POINTS** in CODE_LOCATIONS.txt
