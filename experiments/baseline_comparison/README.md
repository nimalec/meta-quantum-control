# Baseline Comparison for Meta-Learning

This folder contains a clean implementation for comparing meta-learned policies against non-adaptive baselines.

## Overview

We compare three approaches:

1. **Meta K=0** (Non-adaptive meta-initialization)
   - Use the meta-learned initialization without any adaptation
   - Tests the quality of the meta-initialization itself
   - Expected to perform worse than adapted policies

2. **Average Task Policy** (Single-task baseline)
   - Train a standard policy on the mean task (average noise parameters)
   - Represents the classical approach: train on average environment
   - Non-adaptive: does not adapt to individual tasks at test time

3. **Meta K>0** (Adaptive meta-learning)
   - Use the meta-learned initialization with K gradient adaptation steps
   - The full meta-learning approach
   - Expected to outperform both baselines

## Key Metrics

### Gap vs K (Adaptation Steps)

Measures how the performance gap increases with more adaptation steps:
- **Gap vs Meta K=0**: Shows the benefit of adaptation from meta-initialization
- **Gap vs Average Task**: Shows the benefit of meta-learning over single-task training

Expected behavior:
- Gap should increase with K (more adaptation → better performance)
- Follows theoretical prediction: `Gap(K) ∝ (1 - e^(-μηK))`

### Gap vs Variance (Task Diversity)

Measures how the performance gap scales with task distribution variance:
- As variance increases, tasks become more diverse
- Meta-learning should show larger gains for higher variance
- Expected: `Gap(σ²) ∝ σ²` (linear scaling)

## Usage

### Step 1: Train Meta Policy

First, train your meta-learning policy using the standard training script:

```bash
cd experiments/train_scripts
python train_meta.py --config ../../configs/experiment_config.yaml
```

This will save the meta policy to `checkpoints/maml_best_pauli_x_best_policy.pt`.

### Step 2: Train Average Task Baseline

Train a policy on the mean task from the task distribution:

```bash
cd experiments/baseline_comparison
python train_average_task.py --config ../../configs/experiment_config.yaml
```

This will:
1. Sample tasks from the distribution to compute mean parameters
2. Train a policy on the single mean task
3. Save to `checkpoints/baseline_comparison/average_task_policy.pt`

Expected output:
```
Computing mean task from 10000 samples...
  Mean α: 2.0500
  Mean A: 505.0050
  Mean ω_c: 50.5000

Training policy on average task...
  Iterations: 5000
  Learning rate: 0.001

  Iter 500/5000: Loss = 0.234567, Fidelity = 0.765433
  Iter 1000/5000: Loss = 0.123456, Fidelity = 0.876544
  ...

  Training complete!
  Best Fidelity: 0.923456
```

### Step 3: Evaluate All Baselines

Run the evaluation script to compare all three methods:

```bash
python evaluate_baselines.py \
  --meta_policy ../train_scripts/checkpoints/maml_best_pauli_x_best_policy.pt \
  --average_policy checkpoints/baseline_comparison/average_task_policy.pt \
  --config ../../configs/experiment_config.yaml \
  --output_dir results/baseline_comparison \
  --n_test_tasks 100 \
  --k_values 0 1 2 3 5 7 10 \
  --variance_scales 0.1 0.3 0.5 0.7 1.0 \
  --evaluate_gap_vs_k \
  --evaluate_gap_vs_variance
```

This will:
1. Load both policies
2. Sample test tasks
3. Evaluate all three methods (Meta K=0, Average Task, Meta K>0)
4. Compute gaps for different K values
5. Compute gaps for different variance levels
6. Save results to JSON files

Expected output:
```
EVALUATING GAP VS K
========================================

[1/3] Evaluating Meta K=0 (no adaptation)...
  Mean fidelity: 0.756234 ± 0.012345

[2/3] Evaluating Average Task Policy...
  Mean fidelity: 0.743567 ± 0.013456

[3/3] Evaluating Meta with K adaptation steps...

  K = 1:
    Mean fidelity: 0.782345 ± 0.011234
    Gap vs Meta K=0: 0.026111 ± 0.016579
    Gap vs Average: 0.038778 ± 0.017691

  K = 5:
    Mean fidelity: 0.845678 ± 0.009876
    Gap vs Meta K=0: 0.089444 ± 0.015121
    Gap vs Average: 0.102111 ± 0.016579
```

### Step 4: Plot Results

Generate publication-quality figures:

```bash
python plot_gap_analysis.py --results_dir results/baseline_comparison
```

This creates:
1. `gap_vs_k.pdf` - Gap as a function of adaptation steps
2. `gap_vs_variance.pdf` - Gap as a function of task diversity
3. `combined_summary.pdf` - 2×2 panel with all key results

## Configuration Options

### Evaluation Parameters

```bash
--n_test_tasks 100              # Number of tasks for gap vs K evaluation
--k_values 0 1 2 3 5 7 10      # Adaptation steps to evaluate
--k_fixed 5                     # Fixed K for variance evaluation
--variance_scales 0.1 0.5 1.0   # Variance scale factors (0-1)
--n_tasks_per_variance 50       # Tasks per variance level
--seed 42                       # Random seed for reproducibility
```

### Training Parameters

In `train_average_task.py`, you can adjust:
- `n_iterations`: Number of training iterations (default: 5000)
- `lr`: Learning rate (default: 0.001)
- `log_interval`: How often to print progress (default: 500)

## Expected Results

### Gap vs K

You should observe:
- **Positive gap**: Meta K>0 outperforms both baselines
- **Increasing with K**: More adaptation → larger gap
- **Exponential saturation**: Follows `Gap(K) ∝ (1 - e^(-μηK))`
- **Meta K=0 baseline**: Shows meta-initialization quality
- **Average Task baseline**: Shows benefit over single-task training

Example:
```
K=1:  Gap vs K=0 = +0.026, Gap vs Avg = +0.039
K=5:  Gap vs K=0 = +0.089, Gap vs Avg = +0.102
K=10: Gap vs K=0 = +0.125, Gap vs Avg = +0.138
```

### Gap vs Variance

You should observe:
- **Linear scaling**: Gap ∝ σ² (variance)
- **Larger gaps at high variance**: Meta-learning helps more when tasks are diverse
- **Both baselines converge at low variance**: When tasks are similar, meta-learning less critical

Example:
```
σ²=0.01: Gap vs K=0 = +0.045, Gap vs Avg = +0.052
σ²=0.10: Gap vs K=0 = +0.089, Gap vs Avg = +0.102
σ²=0.50: Gap vs K=0 = +0.178, Gap vs Avg = +0.195
```

## Interpreting Results

### Positive Gap

**Gap vs Meta K=0 > 0**: Adaptation is beneficial
- Shows that gradient-based adaptation improves over initialization alone
- Validates the value of few-shot learning

**Gap vs Average Task > Gap vs Meta K=0**: Meta-learning is beneficial
- Shows that meta-initialization outperforms training on average task
- Validates the value of learning a good initialization

### Gap Trends

**Gap increasing with K**:
- Confirms theoretical prediction
- More adaptation steps → better task-specific performance

**Gap increasing with variance**:
- Confirms theoretical prediction: Gap ∝ σ²
- Meta-learning most valuable for diverse task distributions

## File Structure

```
baseline_comparison/
├── README.md                      # This file
├── train_average_task.py          # Train policy on mean task
├── evaluate_baselines.py          # Compare all methods
├── plot_gap_analysis.py           # Visualize results
├── checkpoints/                   # Saved models
│   └── baseline_comparison/
│       └── average_task_policy.pt
└── results/                       # Evaluation results
    └── baseline_comparison/
        ├── gap_vs_k_results.json
        ├── gap_vs_variance_results.json
        ├── gap_vs_k.pdf
        ├── gap_vs_variance.pdf
        └── combined_summary.pdf
```

## Implementation Details

### Non-Adaptive vs Adaptive

**Non-adaptive methods** (Meta K=0, Average Task):
- Evaluate policy directly on test tasks
- No gradient steps at test time
- Fast but potentially suboptimal

**Adaptive method** (Meta K>0):
- Take K gradient steps on each test task
- Requires backpropagation through quantum simulator
- Slower but achieves better task-specific performance

### Why This Design?

1. **Clean separation**: Core algorithms in `metaqctrl/`, experiments in `experiments/`
2. **No modification of core code**: Uses existing `compute_loss_differentiable()`, `compute_fidelity()`
3. **Modular**: Each script does one thing well
4. **Reproducible**: Fixed seeds, saved configs, JSON results
5. **Publication-ready**: High-quality figures with error bars

## Troubleshooting

### Issue: Gap vs Average is negative

**Cause**: Average task policy may have trained better than meta-policy
**Solution**:
- Check meta-policy training converged properly
- Increase `n_iterations` in `train_average_task.py`
- Ensure both use same config/architecture

### Issue: Gap doesn't increase with K

**Cause**: Learning rate may be too high or too low
**Solution**:
- Adjust `inner_lr` in config
- Try K values [1, 2, 3, 5, 10, 20] to see full curve

### Issue: High variance in results

**Cause**: Not enough test tasks
**Solution**: Increase `--n_test_tasks` to 200 or more

## Citation

If you use this baseline comparison framework, please cite:

```
@article{your-paper,
  title={Meta-Reinforcement Learning for Quantum Control},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## References

- MAML paper: Finn et al., "Model-Agnostic Meta-Learning for Fast Adaptation", ICML 2017
- Quantum control: Nielsen & Chuang, "Quantum Computation and Quantum Information"
