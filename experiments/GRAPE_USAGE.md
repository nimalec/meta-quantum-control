# GRAPE Baseline Usage Guide

## Overview

GRAPE (Gradient Ascent Pulse Engineering) is a **task-specific optimization method**, fundamentally different from policy-based approaches (Meta-learning, Robust policies).

## Key Differences

| Feature | Policy-based (MAML, Robust) | GRAPE |
|---------|----------------------------|-------|
| **Training** | Learns a parameterized policy π(θ) | Optimizes controls u(t) directly per task |
| **Generalization** | Can generalize to new tasks | Must re-optimize for each new task |
| **Adaptation** | Fast adaptation (few gradient steps) | Full optimization required (100+ iterations) |
| **Storage** | Single policy network | Must store controls for each task |
| **Use case** | Few-shot learning, fast deployment | Oracle baseline, maximum performance |

## Why train_robust.py Doesn't Use GRAPE

`train_robust.py` trains a **robust neural network policy**, not GRAPE:
- **RobustPolicy**: Trains a policy network to minimize worst-case or average loss across task distribution
- **No adaptation**: Policy is fixed at test time (unlike MAML which adapts)
- **Comparison baseline**: Used to measure the "adaptation gap" that MAML provides

## Training GRAPE Baseline

### 1. Train GRAPE on Tasks

```bash
cd experiments
python train_grape.py --config configs/experiment_config.yaml
```

**What this does:**
- Samples N training tasks from the task distribution
- For each task, runs GRAPE optimization (100-200 iterations)
- Stores optimized control sequences
- Saves results to `checkpoints/grape_best.pkl`

**Important:** GRAPE is trained PER TASK, not as a generalizable policy.

### 2. Using GRAPE in Paper Results

The `paper_results/` scripts (like `experiment_gap_vs_k.py`) use GRAPE as an **online optimization baseline**:

```python
# For each test task:
grape = GRAPEOptimizer(n_segments=20, n_controls=2)
optimal_controls = grape.optimize(
    simulate_fn=simulate_fn,
    task_params=test_task,
    max_iterations=100
)
fidelity = env.compute_fidelity(optimal_controls, test_task)
```

This is different from loading pre-trained policies:
- **MAML/Robust**: Load weights from `checkpoints/*.pt`, then evaluate
- **GRAPE**: Run fresh optimization for each test task (no pre-training helps)

## Experiment Workflow

### Step 1: Train Policies
```bash
# Train meta-learning policy
python experiments/train_meta.py --config configs/experiment_config.yaml

# Train robust baseline policy
python experiments/train_robust.py --config configs/experiment_config.yaml

# GRAPE doesn't need "training" - it's run per-task in experiments
```

### Step 2: Run Paper Experiments
```bash
cd experiments/paper_results

# Run all experiments (GRAPE is computed on-the-fly)
python generate_all_results.py \
    --meta_path ../checkpoints/maml_best.pt \
    --robust_path ../checkpoints/robust_best.pt \
    --include_grape \
    --grape_iterations 100
```

**What happens:**
1. Loads pre-trained MAML and Robust policies from checkpoints
2. For each test task:
   - Evaluates MAML with K adaptation steps
   - Evaluates Robust policy (no adaptation)
   - **Runs GRAPE optimization from scratch** (100 iterations)
3. Compares all three methods

## GRAPE in Context

### Performance Hierarchy (Expected)
```
GRAPE (100+ iters) ≥ MAML (K=10) > MAML (K=1) > Robust (K=0)
     ^^^^                ^^^^                      ^^^^
   Oracle            Meta-learned              No adaptation
   baseline          few-shot                  baseline
```

### Why GRAPE is the "Upper Bound"
- Has full task information (α, A, ω_c)
- Can optimize for 100+ iterations per task
- No generalization requirement

### Why We Still Want MAML
- **Speed**: 5-10 gradient steps vs 100+ for GRAPE
- **Generalization**: Single policy works across tasks
- **Deployment**: Lighter weight (network vs full optimization)

## Fixing the n_trajectories Issue

The `n_trajectories` parameter is used correctly in the codebase:

```python
# train_meta.py:86
def data_generator(
    task_params: NoiseParameters,
    n_trajectories: int,  # ✓ Parameter is defined
    split: str,
    quantum_system: dict,
    config: dict,
    device: torch.device
):
```

**If you're seeing errors:**
1. Check you're passing the parameter in function calls
2. Verify the function signature matches where it's called
3. Look for places where positional arguments might be out of order

## Checkpoint Structure

After training, your `experiments/checkpoints/` should contain:

```
checkpoints/
├── maml_20241027_123456.pt          # Meta-policy checkpoint
├── maml_best.pt                      # Best meta-policy (validation)
├── robust_minimax_20241027_124500.pt # Robust policy checkpoint
├── robust_best.pt                    # Best robust policy
└── grape_best.pkl                    # (Optional) Pre-optimized GRAPE controls
```

**Note:** `grape_best.pkl` is optional - the paper_results scripts will run GRAPE on-the-fly if `--include_grape` is set.

## Summary

1. ✗ **train_robust.py does NOT train GRAPE** - it trains a robust neural network policy
2. ✓ **GRAPE is used in paper_results/ as an online optimization baseline**
3. ✓ **Use `--include_grape` flag in generate_all_results.py to enable GRAPE comparison**
4. ✓ **Checkpoints directory now exists at `experiments/checkpoints/`**
5. ✓ **Config keys are now consistent across all scripts**

## Quick Start

```bash
# 1. Train policies
cd experiments
python train_meta.py --config configs/experiment_config.yaml
python train_robust.py --config configs/experiment_config.yaml

# 2. Generate results with GRAPE baseline
cd paper_results
python generate_all_results.py \
    --meta_path ../checkpoints/maml_best.pt \
    --robust_path ../checkpoints/robust_best.pt \
    --include_grape \
    --n_tasks 100
```
