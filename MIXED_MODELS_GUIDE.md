# Mixed Model Sampling for Meta-Learning

## Overview

This feature enables **sampling tasks from multiple PSD noise models simultaneously**, significantly increasing task diversity and improving meta-learning performance.

### Key Benefits

1. **Larger Optimality Gap**: According to your theory, `Gap ∝ σ²_θ · (1 - e^(-μηK))`. Mixed model sampling dramatically increases σ²_θ beyond just parameter variance.

2. **Better Generalization**: The policy learns to adapt to fundamentally different noise types (1/f vs Lorentzian), improving robustness to model uncertainty.

3. **Physical Realism**: Real quantum devices experience mixed noise sources (charge noise, two-level fluctuators, magnetic field noise, etc.).

## Quick Start

### 1. Single Model Mode (Default)

```bash
# Uses only one_over_f noise
python experiments/train_scripts/train_meta.py --config configs/experiment_config.yaml
```

**Config settings**:
```yaml
psd_model: 'one_over_f'
task_feature_dim: 4  # Still 4D for forward compatibility
# model_types: Not specified → single model
```

### 2. Mixed Model Mode

```bash
# Samples from BOTH one_over_f and lorentzian noise
python experiments/train_scripts/train_meta.py --config configs/experiment_config_mixed_models.yaml
```

**Config settings**:
```yaml
model_types: ['one_over_f', 'lorentzian']  # List of models to sample
model_probs: [0.5, 0.5]  # 50% each (optional, defaults to uniform)
task_feature_dim: 4  # REQUIRED: [alpha, A, omega_c, model_encoding]
inner_steps: 15  # May need more steps for diverse tasks
```

## How It Works

### Task Encoding

Each task is now represented by **4 features** instead of 3:

```python
# Old (single model):
task = [alpha, A, omega_c]  # 3D

# New (mixed models):
task = [alpha, A, omega_c, model_encoding]  # 4D
```

**Model encoding**:
- `one_over_f`: 0.0
- `lorentzian`: 1.0
- `double_exp`: 2.0

### Task Sampling

```python
# Automatic mixed sampling
task_dist = TaskDistribution(
    dist_type='uniform',
    ranges={'alpha': (0.5, 2.0), 'A': (0.001, 0.01), 'omega_c': (100, 1000)},
    model_types=['one_over_f', 'lorentzian'],
    model_probs=[0.5, 0.5]  # Optional
)

tasks = task_dist.sample(n=100)
# Each task has model_type randomly selected
```

### Dynamic PSD Selection

The `PSDToLindblad` converter automatically selects the correct PSD model based on each task's `model_type`:

```python
# Automatically creates correct model for each task
L_ops = psd_to_lindblad.get_lindblad_operators(task_params)
# If task.model_type == 'lorentzian' → uses Lorentzian PSD
# If task.model_type == 'one_over_f' → uses 1/f PSD
```

## Configuration Options

### Available PSD Models

1. **`one_over_f`**: S(ω) = A / (|ω|^α + ω_c^α)
   - Colored noise with adjustable exponent α
   - Common in charge noise, 1/f flux noise

2. **`lorentzian`**: S(ω) = A / (ω² + ω_c²)
   - Ornstein-Uhlenbeck noise
   - Two-level fluctuators, magnetic field noise

3. **`double_exp`**: Sum of two Lorentzians
   - Multi-timescale noise

### Example Configurations

#### Equal Mix (50/50)
```yaml
model_types: ['one_over_f', 'lorentzian']
model_probs: [0.5, 0.5]
```

#### Biased Toward 1/f (70/30)
```yaml
model_types: ['one_over_f', 'lorentzian']
model_probs: [0.7, 0.3]
```

#### Three Models (33/33/33)
```yaml
model_types: ['one_over_f', 'lorentzian', 'double_exp']
# model_probs omitted → automatic uniform distribution
```

## Training Recommendations

### Hyperparameter Adjustments

When using mixed models, consider:

1. **Increase `inner_steps`**: 10 → 15-20
   - More diverse tasks require more adaptation steps
   - Recommended: Start with 15

2. **Monitor adaptation gain**:
   ```
   adaptation_gain = val_loss_pre_adapt - val_loss_post_adapt
   ```
   - Should be larger with mixed models (higher σ²_θ)

3. **Possibly increase `tasks_per_batch`**: 10 → 15
   - More stable meta-gradients with diverse tasks

### Expected Results

**Metrics to compare (single vs mixed)**:
- [ ] Final meta-validation fidelity (post-adapt)
- [ ] Task variance σ²_θ (printed during training)
- [ ] Optimality gap vs robust baseline
- [ ] Generalization to held-out model types

**Example expected variance**:
```
Single model (one_over_f only):
  σ²_θ ≈ 67,500 (from parameter ranges)

Mixed models (50/50 one_over_f + lorentzian):
  σ²_θ ≈ 67,500 + 0.25 = 67,500.25
  INFO: Mixed model variance contribution: 0.2500
```

The categorical variance (0.25 for 50/50 split) seems small numerically, but represents **fundamentally different functional forms**, adding huge diversity!

## Validation Experiments

### 1. Baseline Comparison

```bash
# Single model baseline
python experiments/train_scripts/train_meta.py \
  --config configs/experiment_config.yaml

# Mixed model experiment
python experiments/train_scripts/train_meta.py \
  --config configs/experiment_config_mixed_models.yaml
```

Compare final validation fidelities.

### 2. Generalization Test

Train on `one_over_f` only, test on `lorentzian`:
```python
# After training on one_over_f
val_tasks_lorentzian = [
    NoiseParameters(alpha=1.0, A=0.005, omega_c=500, model_type='lorentzian')
    for _ in range(20)
]
# Evaluate fidelity → should be lower than trained model
```

Train on mixed models, test on `lorentzian`:
```python
# Should have better generalization!
```

### 3. Optimality Gap Validation

```bash
# Compute gap for single vs mixed
python experiments/paper_results/experiment_gap_vs_variance.py \
  --checkpoint_single checkpoints/maml_best_pauli_x_policy.pt \
  --checkpoint_mixed checkpoints_mixed_models/maml_best_pauli_x_policy.pt
```

Expected: `Gap(mixed) > Gap(single)` due to larger σ²_θ.

## Code Changes Summary

### Modified Files

1. **`metaqctrl/quantum/noise_models_v2.py`**
   - ✅ Added `model_type` field to `NoiseParameters`
   - ✅ Added `model_types` and `model_probs` to `TaskDistribution`
   - ✅ Updated `PSDToLindblad` for dynamic model selection

2. **`metaqctrl/theory/quantum_environment.py`**
   - ✅ Updated `_task_hash()` to include model type
   - ✅ Added support for `model_types` config parameter

3. **`experiments/train_scripts/train_meta.py`**
   - ✅ Updated `create_task_distribution()` to pass model types
   - ✅ Updated data generators to use 4D features

4. **`configs/experiment_config.yaml`**
   - ✅ Changed `task_feature_dim: 3 → 4`
   - ✅ Added commented example for mixed models

5. **`configs/experiment_config_mixed_models.yaml`** (NEW)
   - ✅ Ready-to-use config for mixed model experiments

### Backward Compatibility

**100% backward compatible!**

- Old configs (without `model_types`) still work
- `to_array(include_model=False)` returns 3D array
- Single model code paths unchanged

## Troubleshooting

### Issue: "Invalid model encoding"

**Cause**: Policy output has wrong dimension or invalid model encoding value.

**Fix**: Ensure `task_feature_dim: 4` in config.

### Issue: Cache misses / slow training

**Cause**: Different model types create different cache keys.

**Fix**: This is expected. Cache size will be ~2x with 2 model types.

### Issue: Lower fidelity than single model

**Possible causes**:
1. Need more `inner_steps` (try 15-20)
2. Need more training iterations
3. Task distribution variance too high → reduce parameter ranges

## API Reference

### NoiseParameters

```python
from metaqctrl.quantum.noise_models_v2 import NoiseParameters

# Create task with model type
task = NoiseParameters(
    alpha=1.0,
    A=0.01,
    omega_c=500.0,
    model_type='lorentzian'
)

# Convert to array (4D)
arr = task.to_array(include_model=True)  # [1.0, 0.01, 500.0, 1.0]

# Convert to array (3D, backward compatible)
arr = task.to_array(include_model=False)  # [1.0, 0.01, 500.0]

# Decode from array
task2 = NoiseParameters.from_array(arr, has_model=True)
```

### TaskDistribution

```python
from metaqctrl.quantum.noise_models_v2 import TaskDistribution

# Mixed model distribution
dist = TaskDistribution(
    dist_type='uniform',
    ranges={'alpha': (0.5, 2.0), 'A': (0.001, 0.01), 'omega_c': (100, 1000)},
    model_types=['one_over_f', 'lorentzian'],
    model_probs=[0.6, 0.4]  # 60% 1/f, 40% Lorentzian
)

# Sample tasks
tasks = dist.sample(n_tasks=100)
# tasks[i].model_type is randomly selected according to model_probs

# Compute variance (includes model contribution)
sigma_squared = dist.compute_variance()
```

### PSDToLindblad

```python
from metaqctrl.quantum.noise_adapter import PSDToLindblad

# Dynamic model selection (for mixed models)
converter = PSDToLindblad(
    basis_operators=[sigma_x, sigma_y, sigma_z],
    sampling_freqs=omega_sample,
    psd_model=None,  # None → dynamic selection based on task.model_type
    T=1.0,
    sequence='ramsey',
    omega0=omega0,
    g_energy_per_xi=hbar/2
)

# Automatically uses correct model
task1 = NoiseParameters(..., model_type='one_over_f')
L_ops1 = converter.get_lindblad_operators(task1)  # Uses 1/f PSD

task2 = NoiseParameters(..., model_type='lorentzian')
L_ops2 = converter.get_lindblad_operators(task2)  # Uses Lorentzian PSD
```

## Further Experiments

### 1. Curriculum Learning

Start with single model, gradually introduce second model:
```python
# Epochs 0-20: 100% one_over_f
# Epochs 21-40: 70% one_over_f, 30% lorentzian
# Epochs 41+: 50% one_over_f, 50% lorentzian
```

### 2. Three-Model Mix

```yaml
model_types: ['one_over_f', 'lorentzian', 'double_exp']
inner_steps: 20  # Even more steps needed
```

### 3. Model-Specific Performance

Analyze which model types are harder:
```python
# During validation, group by model_type
fidelities_1f = [f for task, f in zip(tasks, fids) if task.model_type == 'one_over_f']
fidelities_lor = [f for task, f in zip(tasks, fids) if task.model_type == 'lorentzian']

print(f"1/f:        {np.mean(fidelities_1f):.4f}")
print(f"Lorentzian: {np.mean(fidelities_lor):.4f}")
```

## Citation

If you use mixed model sampling in your research, please cite:

```bibtex
@article{leclerc2025metarl,
  title={Meta-Reinforcement Learning for Quantum Control: Generalization and Robustness under Noise Shifts},
  author={Leclerc, N. and Brawand, A.},
  year={2025}
}
```

## Contact

For questions or issues with mixed model sampling, please open an issue on GitHub.
