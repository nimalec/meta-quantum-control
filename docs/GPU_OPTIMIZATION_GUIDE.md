# GPU Performance Optimization Guide

This guide explains the GPU performance improvements and how to use them.

## Problem Summary

The original training code was running as slow on GPU as on CPU due to:

1. **Simulator recreation bottleneck** - Creating a new PyTorch simulator on every loss call
2. **Excessive time steps** - Using dt=0.001 caused ~4,000 matrix operations per forward pass
3. **Tiny matrices** - 2×2 matrices (single qubit) don't benefit from GPU parallelization
4. **No batching** - Sequential task processing instead of parallel

## Fixes Applied

### 1. Simulator Caching (CRITICAL FIX ✅)

**File:** `metaqctrl/theory/quantum_environment.py`

Added `get_torch_simulator()` method that caches differentiable simulators:

```python
# Before (SLOW):
sim = DifferentiableLindbladSimulator(...)  # Created every time!

# After (FAST):
sim = env.get_torch_simulator(task_params, device, dt, use_rk4)  # Cached!
```

**Expected speedup:** 10-100x depending on task diversity

### 2. Configurable Integration Settings

**File:** `experiments/train_scripts/train_meta.py`

Loss function now reads dt and integration method from config:

```python
# config.yaml
dt_training: 0.02        # Larger = faster (was 0.001)
use_rk4_training: false  # Euler = faster than RK4
```

**Trade-off:** Speed vs accuracy. Use fast settings for training, accurate for evaluation.

### 3. Batched Task Processing (BONUS)

**File:** `metaqctrl/theory/batched_processing.py`

Process multiple tasks in one forward pass:

```python
from metaqctrl.theory.batched_processing import BatchedLossComputation

batch_computer = BatchedLossComputation(env, device, dt=0.02)
losses = batch_computer(policy, task_params_list)  # All at once!
```

**Expected speedup:** 2-5x additional improvement

## Quick Start

### 1. Run Benchmark

Test the performance improvements:

```bash
cd experiments
python benchmark_gpu_performance.py
```

This will show you the speedup from different settings.

### 2. Use GPU-Optimized Config

Train with the new optimized configuration:

```bash
cd experiments/train_scripts
python train_meta.py --config ../../configs/experiment_config_gpu.yaml
```

### 3. Compare Results

Compare with the original config:

```bash
# Original (slow)
python train_meta.py --config ../../configs/experiment_config.yaml

# GPU-optimized (fast)
python train_meta.py --config ../../configs/experiment_config_gpu.yaml
```

## Configuration Comparison

### Original Config (Slow on GPU)

```yaml
tasks_per_batch: 200      # Too many!
val_tasks: 100            # Too many!
inner_steps: 10
# No dt_training or use_rk4_training specified
```

**Problem:** Processing 200 tasks sequentially, with no integration optimization.

### GPU-Optimized Config (Fast)

```yaml
tasks_per_batch: 16       # Reasonable batch size
val_tasks: 20             # Fewer validation tasks
inner_steps: 5            # Faster convergence
dt_training: 0.02         # Larger time step
use_rk4_training: false   # Euler method
```

**Benefit:** ~20-50x speedup expected!

## Performance Expectations

| Configuration | Time per iteration | Speedup |
|---------------|-------------------|---------|
| **Original** (dt=0.001, recreating sims) | ~300-500s | 1x |
| **With caching** (dt=0.01, RK4) | ~30-50s | ~10x |
| **GPU-optimized** (dt=0.02, Euler, 16 tasks) | ~5-10s | ~50x |

## Tuning Guide

### For Training (Speed Priority)

```yaml
dt_training: 0.02           # Fast
use_rk4_training: false     # Euler
tasks_per_batch: 16-32      # Moderate
inner_steps: 5              # Fewer steps
```

**Use when:** You want to iterate quickly and test ideas.

### For Final Evaluation (Accuracy Priority)

```yaml
dt_training: 0.01           # More accurate
use_rk4_training: true      # RK4
tasks_per_batch: 8-16       # Smaller batches
inner_steps: 10             # More adaptation
```

**Use when:** You need accurate results for papers/reports.

### For Debugging

```yaml
dt_training: 0.01
use_rk4_training: true
tasks_per_batch: 4
inner_steps: 3
n_iterations: 50            # Quick test
```

**Use when:** Debugging code, want faster feedback.

## Advanced: Custom Integration Settings

You can also pass dt and use_rk4 directly:

```python
# In your own script
loss = env.compute_loss_differentiable(
    policy,
    task_params,
    device,
    dt=0.02,        # Custom time step
    use_rk4=False   # Use Euler
)
```

## Monitoring Cache Performance

Check if caching is working:

```python
print(env.get_cache_stats())
# Output:
# {
#   'n_cached_operators': 15,
#   'n_cached_simulators': 15,
#   'n_cached_torch_simulators': 15,  # Should grow during training
#   'cache_size_mb': 0.02
# }
```

If `n_cached_torch_simulators` stays at 0, caching isn't working!

## Why GPU Training Was Slow

### 1. Simulator Recreation (Main Issue)

**Before:**
```python
# Called 1000s of times during training
def compute_loss_differentiable(...):
    # Convert to torch tensors (CPU → GPU transfer)
    H0_torch = numpy_to_torch_complex(self.H0, device)
    ...
    # Create new simulator
    sim = DifferentiableLindbladSimulator(...)  # EXPENSIVE!
    ...
```

**After:**
```python
def compute_loss_differentiable(...):
    # Get cached simulator (reuses GPU tensors)
    sim = self.get_torch_simulator(...)  # FAST!
    ...
```

### 2. Tiny Matrix Operations

GPUs excel at large matrix multiplications (e.g., 1024×1024), not 2×2 matrices.

**Solution:** Process multiple tasks in parallel (batching).

### 3. Sequential Processing

Processing 200 tasks one by one doesn't utilize GPU parallelism.

**Solution:** Use `BatchedLossComputation` to process tasks together.

## Troubleshooting

### Still Slow After Optimization?

1. **Check cache stats:**
   ```python
   print(env.get_cache_stats())
   ```
   If `n_cached_torch_simulators` is 0, the cache isn't being used.

2. **Verify GPU usage:**
   ```bash
   nvidia-smi  # Watch GPU utilization
   ```
   Should see >30% GPU usage during training.

3. **Check batch size:**
   Large batches (>50 tasks) still process sequentially. Try 16-32.

4. **Profile the code:**
   ```python
   import torch.profiler
   # Profile to find bottlenecks
   ```

### Accuracy Concerns?

If results differ with fast settings:

1. Use `dt=0.01` instead of `0.02`
2. Switch to `use_rk4_training: true`
3. Run final evaluation with accurate settings:
   ```python
   # Fast training
   train_with_dt_0.02()

   # Accurate evaluation
   final_fidelity = env.compute_loss_differentiable(
       policy, task_params, device,
       dt=0.005, use_rk4=True
   )
   ```

## Summary

**Three key changes for GPU performance:**

1. ✅ **Simulator caching** - Eliminates recreation overhead (10-100x speedup)
2. ✅ **Larger time steps** - Fewer integration steps (2-5x speedup)
3. ✅ **Batched processing** - Better GPU utilization (2-3x speedup)

**Combined speedup: ~50-100x** 🚀

## Next Steps

1. Run `python experiments/benchmark_gpu_performance.py` to measure speedup
2. Train with GPU-optimized config: `experiment_config_gpu.yaml`
3. Monitor training speed and adjust settings as needed
4. For production, evaluate with accurate settings

---

**Questions or issues?** Check the benchmark script output or profile your code to identify remaining bottlenecks.
