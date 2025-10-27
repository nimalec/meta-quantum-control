# Issues Found and Fixed

## Issue 1: train_robust.py Does NOT Train GRAPE ✓ CLARIFIED

**Status:** This is NOT a bug - it's by design

**What you thought:**
- `train_robust.py` should train a GRAPE baseline

**What it actually does:**
- Trains a **RobustPolicy** - a neural network policy optimized for worst-case/average performance
- Uses the same policy architecture as MAML, but with different training objective
- No adaptation at test time (unlike MAML which adapts in K steps)

**Why this is correct:**
- GRAPE is not a "trainable policy" - it's a **per-task optimization algorithm**
- GRAPE must be run separately for each test task (see paper_results scripts)
- `train_robust.py` provides a different baseline: "best fixed policy without adaptation"

**Comparison:**
```
MAML:    Learn π_init → Adapt to task with K steps → Evaluate
Robust:  Learn π_robust → Evaluate directly (no adaptation)
GRAPE:   For each task → Optimize controls from scratch → Evaluate
```

**Action taken:**
- ✓ Created `train_grape.py` for reference (optional pre-optimization)
- ✓ Created `GRAPE_USAGE.md` explaining the differences
- ✓ No changes needed to `train_robust.py`

---

## Issue 2: n_trajectories Not Recognized ✓ INVESTIGATED

**Status:** No actual bug found - parameter exists and is used correctly

**Investigation:**
```python
# train_meta.py:86-91
def data_generator(
    task_params: NoiseParameters,
    n_trajectories: int,  # ✓ Parameter exists
    split: str,
    quantum_system: dict,
    config: dict,
    device: torch.device
):
```

**Where it's used:**
- `train_meta.py:225-238`: Called with `n_trajectories` parameter
- `train_robust.py:114,119,273`: Called with positional argument
- `metaqctrl/meta_rl/maml.py`: Uses it in MAMLTrainer

**Potential causes if you saw an error:**
1. Calling with wrong argument order
2. Old cached bytecode (`.pyc` files)
3. Import from wrong module

**Action taken:**
- ✓ Verified parameter exists in all correct locations
- ✓ Confirmed function signatures match call sites
- No changes needed - working as intended

**If you still see errors:**
```bash
# Clear Python cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete
```

---

## Issue 3: experiments/checkpoints Directory Missing ✓ FIXED

**Problem:**
- Scripts expected `experiments/checkpoints/` to exist
- Directory was not created by default
- Paper results scripts would fail looking for checkpoint files

**Files affected:**
- `paper_results/experiment_gap_vs_k.py:362-363`
- `paper_results/experiment_gap_vs_variance.py:391-392`
- `paper_results/generate_all_results.py:279,285`

**Fix applied:**
```bash
mkdir -p experiments/checkpoints/
```

**Verification:**
```bash
$ ls -la experiments/checkpoints/
total 0
drwxr-xr-x  2 user  staff   64 Oct 27 14:39 .
drwxr-xr-x 10 user  staff  320 Oct 27 14:39 ..
```

✓ Directory now exists and will be populated during training

---

## Issue 4: Config Key Inconsistencies ✓ FIXED

**Problem:**
Different scripts used different config key names:

| Script | Segments | Controls | Time | Hidden |
|--------|----------|----------|------|--------|
| `train_meta.py` | `n_segments` | `n_controls` | `horizon` | `hidden_dim` |
| `train_robust.py` | `n_segments` | `n_controls` | `horizon` | `hidden_dim` |
| `experiment_gap_vs_k.py` | `n_segments` | `n_controls` | - | `hidden_dim` |
| `generate_all_results.py` (OLD) | `num_segments` ❌ | `num_controls` ❌ | `evolution_time` ❌ | `policy_hidden_dims` ❌ |

**Fix applied:**
Updated `generate_all_results.py:129-142` to use consistent keys:
```python
config = {
    'n_controls': 2,        # Was: 'num_controls'
    'n_segments': 20,       # Was: 'num_segments'
    'horizon': 1.0,         # Was: 'evolution_time'
    'hidden_dim': 128,      # Was: 'policy_hidden_dims'
    'n_hidden_layers': 2,   # Added
    'alpha_range': [0.5, 2.0],  # Flattened from nested 'task_dist'
    'A_range': [0.05, 0.3],
    'omega_c_range': [2.0, 8.0],
    # ... rest of config
}
```

**Files modified:**
- ✓ `experiments/paper_results/generate_all_results.py`

**Verification needed:**
You may also want to check your `configs/experiment_config.yaml` uses these consistent keys.

---

## Issue 5: GRAPE Baseline Not Used in paper_results ✓ CLARIFIED

**Status:** Working as intended - GRAPE is computed on-the-fly

**What the scripts do:**
```python
# experiment_gap_vs_k.py:108-145
if include_grape:
    for task in test_tasks:
        # Create fresh GRAPE optimizer
        grape = GRAPEOptimizer(...)

        # Optimize from scratch for THIS task
        optimal_controls = grape.optimize(
            simulate_fn=simulate_fn,
            task_params=task,
            max_iterations=100
        )

        # Evaluate
        fidelity = env.compute_fidelity(optimal_controls, task)
```

**This is correct because:**
- GRAPE must optimize per-task (cannot generalize)
- Pre-training GRAPE on different tasks doesn't help
- It serves as an "oracle" baseline (given unlimited optimization budget)

**To enable GRAPE in experiments:**
```bash
python paper_results/generate_all_results.py \
    --meta_path checkpoints/maml_best.pt \
    --robust_path checkpoints/robust_best.pt \
    --include_grape \           # ← Enable GRAPE
    --grape_iterations 100      # ← Set optimization budget
```

**Action taken:**
- ✓ Created `train_grape.py` (optional, for standalone GRAPE testing)
- ✓ Documented correct usage in `GRAPE_USAGE.md`
- No changes needed to paper_results scripts - they work correctly

---

## Summary of Changes

### Files Created:
1. ✓ `experiments/train_grape.py` - Optional GRAPE training script
2. ✓ `experiments/GRAPE_USAGE.md` - Documentation of GRAPE usage
3. ✓ `experiments/ISSUES_FIXED.md` - This file

### Files Modified:
1. ✓ `experiments/paper_results/generate_all_results.py` - Fixed config keys

### Directories Created:
1. ✓ `experiments/checkpoints/` - Directory for model weights

### No Changes Needed:
1. ✓ `train_robust.py` - Working as intended (trains RobustPolicy, not GRAPE)
2. ✓ `train_meta.py` - No issues found
3. ✓ `experiment_gap_vs_k.py` - GRAPE usage is correct
4. ✓ `experiment_gap_vs_variance.py` - GRAPE usage is correct

---

## How to Use the Fixed Codebase

### 1. Train Policies
```bash
cd experiments

# Train meta-learning policy (MAML)
python train_meta.py --config configs/experiment_config.yaml

# Train robust baseline policy
python train_robust.py --config configs/experiment_config.yaml
```

**Expected output:**
- `checkpoints/maml_TIMESTAMP.pt`
- `checkpoints/maml_best.pt`
- `checkpoints/robust_minimax_TIMESTAMP.pt`
- `checkpoints/robust_best.pt`

### 2. Generate Paper Results
```bash
cd experiments/paper_results

# Run all experiments with GRAPE baseline
python generate_all_results.py \
    --meta_path ../checkpoints/maml_best.pt \
    --robust_path ../checkpoints/robust_best.pt \
    --include_grape \
    --grape_iterations 100 \
    --n_tasks 100
```

**Expected output:**
- `results/paper/gap_vs_k/` - Gap vs adaptation steps
- `results/paper/gap_vs_variance/` - Gap vs task diversity
- `results/paper/constants_validation/` - Physics constants
- `results/paper/summary_table.txt` - Combined results

### 3. Individual Experiments
```bash
# Run just Gap vs K
python experiment_gap_vs_k.py

# Run just Gap vs Variance
python experiment_gap_vs_variance.py

# Run just constants validation
python experiment_constants_validation.py
```

---

## Remaining Questions

If you're still encountering specific errors, please provide:

1. **Exact error message** with traceback
2. **Which script** you're running
3. **Command line arguments** used
4. **Python version** and package versions

This will help diagnose any remaining issues.

---

## Quick Diagnostic

Run this to check your setup:

```bash
# Check directory structure
ls -la experiments/checkpoints/

# Check for cached bytecode issues
find experiments -name "*.pyc" | wc -l

# Check Python can import modules
cd experiments
python -c "from train_meta import create_quantum_system; print('✓ Imports OK')"

# Check config file exists
cat configs/experiment_config.yaml | head -20
```
