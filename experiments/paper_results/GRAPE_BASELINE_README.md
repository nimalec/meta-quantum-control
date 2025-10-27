# GRAPE Baseline Integration

This document describes the GRAPE (Gradient Ascent Pulse Engineering) baseline integration into the paper results experiments.

## Overview

GRAPE has been added as a baseline comparison method for the robust experiments that generate paper results. GRAPE is a classical pulse optimization technique that directly optimizes control pulses for each task independently, providing a strong baseline for comparison with meta-learning approaches (MAML) and robust policy baselines.

## What Changed

### Modified Files

1. **`experiment_gap_vs_k.py`**
   - Added `include_grape` and `grape_iterations` parameters to `run_gap_vs_k_experiment()`
   - GRAPE baseline is computed once across all K values (independent of adaptation)
   - Results include GRAPE fidelity statistics
   - Updated plotting to show GRAPE as a horizontal reference line

2. **`experiment_gap_vs_variance.py`**
   - Added `include_grape` and `grape_iterations` parameters to `run_gap_vs_variance_experiment()`
   - GRAPE is computed for each variance level to show performance across task distributions
   - Results include GRAPE fidelity per variance level
   - Updated plotting to show GRAPE performance trend

3. **`generate_all_results.py`**
   - Added `--include_grape` flag (default: True)
   - Added `--no_grape` flag to disable GRAPE baseline
   - Added `--grape_iterations` parameter (default: 100)
   - Passes GRAPE parameters to both experiments

## Usage

### Basic Usage (GRAPE Enabled by Default)

```bash
cd experiments/paper_results

python generate_all_results.py \
    --meta_path checkpoints/maml_best.pt \
    --robust_path checkpoints/robust_best.pt \
    --output_dir results/paper \
    --n_tasks 100
```

### Disable GRAPE Baseline

```bash
python generate_all_results.py \
    --meta_path checkpoints/maml_best.pt \
    --robust_path checkpoints/robust_best.pt \
    --no_grape
```

### Adjust GRAPE Iterations

```bash
python generate_all_results.py \
    --meta_path checkpoints/maml_best.pt \
    --robust_path checkpoints/robust_best.pt \
    --grape_iterations 200
```

### Run Individual Experiments

#### Gap vs K with GRAPE
```bash
python experiment_gap_vs_k.py
```
Edit the script to set `include_grape=True` and `grape_iterations=100`.

#### Gap vs Variance with GRAPE
```bash
python experiment_gap_vs_variance.py
```
Edit the script to set `include_grape=True` and `grape_iterations=100`.

## Output Structure

### Results JSON

Both experiments now include a `grape` section in their output JSON:

**experiment_gap_vs_k results:**
```json
{
  "k_values": [1, 2, 3, 5, 7, 10, 15, 20],
  "gaps_mean": [...],
  "gaps_std": [...],
  "fit": {...},
  "grape": {
    "included": true,
    "fidelities": [0.95, 0.97, ...],
    "mean": 0.96,
    "std": 0.01,
    "iterations": 100
  }
}
```

**experiment_gap_vs_variance results:**
```json
{
  "variances": [0.001, 0.002, ...],
  "gaps_mean": [...],
  "gaps_std": [...],
  "fit": {...},
  "grape": {
    "included": true,
    "fidelities_per_variance": [
      {"mean": 0.95, "std": 0.01, "values": [...]},
      {"mean": 0.94, "std": 0.02, "values": [...]}
    ],
    "iterations": 100
  }
}
```

### Figures

Updated plots include GRAPE baseline:

- **Gap vs K Figure**: Shows GRAPE as a horizontal reference line with uncertainty band
- **Gap vs Variance Figure**: Shows GRAPE fidelity at each variance level

## GRAPE Implementation Details

### Location
The GRAPE optimizer is implemented in:
```
/metaqctrl/baselines/robust_control.py
```
Class: `GRAPEOptimizer` (lines 435-735)

### Key Features
- **Optimization Method**: Adam (default), LBFGS, or SGD
- **Gradient Computation**: Finite differences (not backprop through quantum simulation)
- **Per-Task Optimization**: Each task gets independent pulse optimization
- **Convergence**: Early stopping based on fidelity improvement tolerance

### Parameters
- `n_segments`: Number of time segments (20 by default)
- `n_controls`: Number of control channels (2 for single qubit)
- `learning_rate`: 0.1 (Adam default)
- `max_iterations`: 100 (configurable via `--grape_iterations`)
- `control_bounds`: (-5.0, 5.0)

## Performance Considerations

### Computational Cost

GRAPE is **significantly slower** than policy-based methods:

- **MAML**: O(K) gradient steps per task (K typically ≤ 20)
- **Robust Policy**: O(1) forward pass (no adaptation)
- **GRAPE**: O(iterations × n_segments × n_controls) evaluations

Example timing (1 qubit, 20 segments, 2 controls, 100 iterations):
- MAML (K=5): ~0.1s per task
- Robust Policy: ~0.01s per task
- GRAPE: ~2-5s per task

### Recommendations

For quick testing:
```bash
--n_tasks 10 --grape_iterations 50
```

For paper results:
```bash
--n_tasks 100 --grape_iterations 100
```

For high-quality baselines:
```bash
--n_tasks 100 --grape_iterations 200
```

## Interpreting Results

### Expected Outcomes

1. **Gap vs K Experiment**:
   - GRAPE should show **high fidelity** (typically 0.95-0.99)
   - GRAPE fidelity is **independent of K** (no meta-learning)
   - MAML with sufficient K should **approach or exceed** GRAPE fidelity
   - Robust policy (K=0) should have **lower fidelity** than GRAPE

2. **Gap vs Variance Experiment**:
   - GRAPE fidelity may **degrade** with higher task variance
   - GRAPE optimizes each task independently (no transfer)
   - MAML gap should **increase** with variance (theory validation)
   - GRAPE provides upper bound on achievable per-task fidelity

### Interpretation Guide

- **MAML > GRAPE**: Meta-learning successfully transfers knowledge across tasks
- **MAML ≈ GRAPE**: Meta-learning reaches optimal per-task performance
- **MAML < GRAPE**: Either insufficient adaptation steps K or sub-optimal meta-policy

## Troubleshooting

### GRAPE Not Converging

If GRAPE fidelities are low (<0.9):
1. Increase `--grape_iterations` (try 200-500)
2. Check control bounds are appropriate
3. Verify quantum simulator settings

### GRAPE Too Slow

If experiments take too long:
1. Reduce `--n_tasks` (e.g., 50 instead of 100)
2. Reduce `--grape_iterations` (e.g., 50 instead of 100)
3. Use `--no_grape` for quick testing

### Import Errors

If you get import errors:
```bash
# From repository root
export PYTHONPATH=/Users/nimalec/Documents/metarl_project/meta-quantum-control:$PYTHONPATH
cd experiments/paper_results
python generate_all_results.py
```

## Scientific Context

### Why GRAPE as a Baseline?

GRAPE is a widely-used classical optimal control method for quantum systems:
- **Task-Specific**: Optimizes pulses for each task independently
- **No Meta-Learning**: Provides comparison against pure optimization
- **Established**: Well-studied in quantum control literature (Khaneja et al., 2005)

### Comparison with Other Baselines

The robust experiments now include **three baselines**:

1. **Robust Policy** (no adaptation): Trained on task distribution, K=0
2. **MAML** (meta-learning): Pre-trained for fast adaptation, K>0
3. **GRAPE** (classical optimal control): Per-task optimization, no transfer

This provides a comprehensive evaluation:
- **Robust vs MAML**: Value of meta-learning over robust averaging
- **MAML vs GRAPE**: Efficiency of few-shot adaptation vs full optimization
- **All three**: Pareto frontier of sample efficiency vs performance

## References

- Khaneja et al., "Optimal control of coupled spin dynamics: design of NMR pulse sequences by gradient ascent algorithms", J. Magn. Reson. 172, 296-305 (2005)
- See also: `/metaqctrl/baselines/robust_control.py` for implementation details

## Next Steps

For further customization:

1. **Add GRAPE to other experiments**: Use the same pattern as shown here
2. **Try different optimizers**: Modify `method='adam'` to `'lbfgs'` or `'gradient'`
3. **Multi-task GRAPE**: Use `RobustPolicy` with `robust_type='grape'` for robust multi-task optimization
4. **Compare with H2/H∞**: See `/metaqctrl/baselines/robust_control.py` for classical control baselines

---

**Last Updated**: 2025-10-26
**Author**: Claude Code
**Contact**: See main repository README for questions
