# Step-by-Step Debugging Protocol

## Phase 1: Component Testing (Day 1)

### Step 1.1: Test Imports
```bash
python -c "
from src.theory.quantum_environment import QuantumEnvironment
from src.theory.physics_constants import compute_spectral_gap
from src.quantum.noise_models import NoisePSDModel, TaskDistribution
print('✓ All imports successful')
"
```

**Expected:** No errors

**If fails:** Check requirements are installed: `pip install -r requirements.txt`

---

### Step 1.2: Test PSD Mapping
```bash
python -c "
import numpy as np
from src.quantum.noise_models import NoisePSDModel, NoiseParameters, PSDToLindblad

# Pauli matrices
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)

psd_model = NoisePSDModel('one_over_f')
psd_to_lindblad = PSDToLindblad(
    [sx, sy, sz],
    np.array([1.0, 5.0, 10.0]),
    psd_model
)

task = NoiseParameters(1.0, 0.1, 5.0)
L_ops = psd_to_lindblad.get_lindblad_operators(task)

print(f'✓ PSD mapping works: {len(L_ops)} operators')
for i, L in enumerate(L_ops):
    print(f'  ||L{i}|| = {np.linalg.norm(L):.4f}')
"
```

**Expected:** 
```
✓ PSD mapping works: 3 operators
  ||L0|| = 0.XXXX
  ||L1|| = 0.XXXX
  ||L2|| = 0.XXXX
```

**If fails:** Check the fixed `get_lindblad_operators()` in `src/quantum/noise_models.py`

---

### Step 1.3: Test Environment
```bash
python -c "
import numpy as np
from src.theory.quantum_environment import create_quantum_environment
from src.quantum.gates import TargetGates
from src.quantum.noise_models import NoiseParameters

config = {'psd_model': 'one_over_f', 'horizon': 1.0, 'n_segments': 20}
U = TargetGates.hadamard()
ket_0 = np.array([1, 0], dtype=complex)
target = np.outer(U @ ket_0, (U @ ket_0).conj())

env = create_quantum_environment(config, target)
task = NoiseParameters(1.0, 0.1, 5.0)
controls = np.random.randn(20, 2) * 0.1

fidelity = env.evaluate_controls(controls, task)
print(f'✓ Environment works: fidelity = {fidelity:.4f}')
"
```

**Expected:** `✓ Environment works: fidelity = 0.XXXX`

**If fails:** Check `quantum_environment.py` imports and methods

---

### Step 1.4: Test Spectral Gap
```bash
python -c "
from src.theory.quantum_environment import create_quantum_environment
from src.theory.physics_constants import compute_spectral_gap
from src.quantum.gates import TargetGates
from src.quantum.noise_models import NoiseParameters
import numpy as np

config = {'psd_model': 'one_over_f', 'horizon': 1.0, 'n_segments': 20}
U = TargetGates.hadamard()
target = np.outer(U @ np.array([1, 0]), (U @ np.array([1, 0])).conj())
env = create_quantum_environment(config, target)

task = NoiseParameters(1.0, 0.1, 5.0)
gap = compute_spectral_gap(env, task)
print(f'✓ Spectral gap: Δ = {gap:.6f}')
"
```

**Expected:** `✓ Spectral gap: Δ = 0.XXXXXX` (positive number)

**If fails:** Check eigenvalue computation in `compute_spectral_gap()`

---

## Phase 2: Integration Testing (Day 2)

### Step 2.1: Run Minimal Working Example
```bash
python test_minimal_working.py
```

**Expected output:**
```
============================================================
MINIMAL WORKING EXAMPLE - INTEGRATION TEST
============================================================

============================================================
TEST 1: Environment Creation
============================================================
✓ Environment created
  Hilbert dim: 2
  Controls: 2
  Horizon: 1.0
✓ Lindblad operators: 3 operators
...
✓✓✓ ALL TESTS PASSED ✓✓✓
============================================================
```

**If fails:** Read error message carefully. Common issues:

1. **ImportError:** Missing dependency → `pip install <package>`
2. **ShapeError:** Check array dimensions in simulation
3. **NaN/Inf:** Check numerical stability (add clipping)
4. **Slow:** Normal for first run (JIT compilation)

---

### Step 2.2: Test Training Script (10 iterations)
```bash
# Create test config
cat > configs/test_config.yaml << EOF
seed: 42
psd_model: 'one_over_f'
horizon: 1.0
target_gate: 'hadamard'

task_dist_type: 'uniform'
alpha_range: [0.5, 2.0]
A_range: [0.05, 0.3]
omega_c_range: [2.0, 8.0]

task_feature_dim: 3
hidden_dim: 32
n_hidden_layers: 2
n_segments: 10
n_controls: 2
output_scale: 0.5

inner_lr: 0.01
inner_steps: 3
meta_lr: 0.001
first_order: true

n_iterations: 10
tasks_per_batch: 2
n_support: 5
n_query: 5
log_interval: 5
val_interval: 10

save_dir: 'checkpoints_test'
EOF

# Run training
python experiments/train_meta.py --config configs/test_config.yaml
```

**Expected:** Should complete 10 iterations in ~1-2 minutes

**If fails:**
- **CUDA out of memory:** Set `first_order: true` or reduce `hidden_dim`
- **Slow (>10 min):** Check caching is working, reduce `n_segments`
- **Loss = NaN:** Reduce `output_scale` to 0.1, reduce `inner_lr`

---

## Phase 3: Validation Testing (Day 3)

### Step 3.1: Estimate Constants
```bash
python -c "
import torch
from src.theory.quantum_environment import create_quantum_environment
from src.theory.physics_constants import estimate_all_constants
from src.quantum.gates import TargetGates
from src.quantum.noise_models import TaskDistribution, NoiseParameters
from src.meta_rl.policy import PulsePolicy
import numpy as np

# Setup
config = {'psd_model': 'one_over_f', 'horizon': 1.0, 'n_segments': 20}
U = TargetGates.hadamard()
target = np.outer(U @ np.array([1, 0]), (U @ np.array([1, 0])).conj())
env = create_quantum_environment(config, target)

# Policy
policy = PulsePolicy(3, 32, 2, 20, 2)

# Tasks
task_dist = TaskDistribution('uniform', {
    'alpha': (0.5, 2.0), 'A': (0.05, 0.3), 'omega_c': (2.0, 8.0)
})
tasks = task_dist.sample(5, np.random.default_rng(42))

# Estimate (minimal samples)
constants = estimate_all_constants(env, policy, tasks, n_samples_gap=3, n_samples_mu=2)

print(f'\\n✓ Constants estimated:')
print(f'  Δ_min = {constants[\"Delta_min\"]:.6f}')
print(f'  C_filter = {constants[\"C_filter\"]:.6f}')
print(f'  σ²_S = {constants[\"sigma_S_sq\"]:.8f}')
"
```

**Expected:** Should complete in ~2-5 minutes

**If fails:** Check convergence fitting in `estimate_PL_constant_from_convergence()`

---

### Step 3.2: Test Gap Computation
```bash
python -c "
import torch
from src.theory.optimality_gap import OptimalityGapComputer
# ... (setup as above)
# TODO: This will need the fixed OptimalityGapComputer
"
```

**Note:** `src/theory/optimality_gap.py` needs updating to use QuantumEnvironment. For now, skip this or manually adapt.

---

## Phase 4: Full Run (Day 4)

### Step 4.1: Full Training (100 iterations)
```bash
# Update config
cat > configs/full_test_config.yaml << EOF
seed: 42
psd_model: 'one_over_f'
horizon: 1.0
target_gate: 'hadamard'

task_dist_type: 'uniform'
alpha_range: [0.5, 2.0]
A_range: [0.05, 0.3]
omega_c_range: [2.0, 8.0]

task_feature_dim: 3
hidden_dim: 128
n_hidden_layers: 2
n_segments: 20
n_controls: 2
output_scale: 0.5

inner_lr: 0.01
inner_steps: 5
meta_lr: 0.001
first_order: false  # Second-order for better results

n_iterations: 100
tasks_per_batch: 4
n_support: 10
n_query: 10
log_interval: 10
val_interval: 25

save_dir: 'checkpoints'
EOF

python experiments/train_meta.py --config configs/full_test_config.yaml
```

**Expected:** ~30-60 minutes, final meta_loss < 0.3

---

### Step 4.2: Train Robust Baseline
```bash
python experiments/train_robust.py --config configs/full_test_config.yaml
```

**Expected:** ~30-60 minutes

---

### Step 4.3: Evaluate Gap (if eval_gap.py works with environment)
```bash
# This will need fixing OptimalityGapComputer first
# python experiments/eval_gap.py --meta_path ... --robust_path ...
```

---

## Common Issues & Fixes

### Issue 1: Slow simulation
**Symptom:** Each iteration takes >10 seconds

**Fix:**
```python
# In quantum_environment.py, verify caching:
def get_simulator(self, task_params):
    key = self._task_hash(task_params)
    if key not in self._sim_cache:
        print(f"Cache miss for {key}")  # Should only see this once per task
    # ...
```

If you see many cache misses, task hashing is broken.

---

### Issue 2: NaN losses
**Symptom:** Loss becomes NaN after a few iterations

**Fix:**
```yaml
# In config, reduce:
output_scale: 0.1  # Instead of 0.5
inner_lr: 0.005    # Instead of 0.01

# And add gradient clipping in maml.py (already there)
```

---

### Issue 3: No adaptation gain
**Symptom:** Pre-adapt and post-adapt losses are identical

**Fix:**
- Check inner loop is actually running: add print in `inner_loop()`
- Verify gradients are non-zero
- Increase `inner_steps` to 10
- Increase task variance ranges

---

### Issue 4: Import errors
**Symptom:** `ModuleNotFoundError: No module named 'src'`

**Fix:**
```bash
# Install in editable mode
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

---

## Performance Benchmarks

### Expected speeds (on CPU):

| Component | Time | Notes |
|-----------|------|-------|
| Single simulation | 50-100ms | RK45, 20 segments |
| Policy forward pass | 1-5ms | 128 hidden units |
| Inner loop (K=5) | 0.5-1s | With caching |
| Meta-step (4 tasks) | 5-10s | Second-order |
| Meta-step (4 tasks, FOMAML) | 2-4s | First-order |
| 100 iterations | 30-60 min | FOMAML |
| 1000 iterations | 5-10 hrs | FOMAML |

### On GPU:
- Policy forward: 2-3× faster
- Simulation: Same (CPU-bound in numpy)
- Overall: ~1.5× faster

---

## Success Criteria

### ✅ Phase 1 Complete When:
- All component tests pass
- No import errors
- Spectral gap computes correctly

### ✅ Phase 2 Complete When:
- Minimal working example passes
- 10-iteration training completes
- Losses are finite and decreasing

### ✅ Phase 3 Complete When:
- All constants estimated successfully
- Values are in reasonable ranges:
  - Δ ∈ [0.01, 1.0]
  - C_filter ∈ [0.001, 0.1]
  - σ²_S ∈ [1e-4, 1e-2]

### ✅ Phase 4 Complete When:
- 100 iterations complete successfully
- Meta-loss < 0.3
- Validation shows adaptation gain > 0.05
- Robust baseline trains
- Gap evaluation produces plots

---

## Emergency Fallbacks

### If nothing works:

1. **Use even simpler system:**
```python
# 2-segment, 1-control, no noise
config = {
    'n_segments': 2,
    'n_controls': 1,
    'alpha_range': [1.0, 1.0],  # Fixed
    'A_range': [0.0, 0.0],      # No noise
}
```

2. **Test with pure states (no Lindblad):**
```python
# Set all L_ops to zero
L_ops = [np.zeros((2, 2)) for _ in range(3)]
```

3. **Use pre-computed controls:**
```python
# Test fidelity computation only
controls = np.array([[0.1, 0.0]] * 20)
fidelity = env.evaluate_controls(controls, task)
```

---

## Checklist Before Full Experiments

- [ ] `test_minimal_working.py` passes
- [ ] 10-iteration training completes
- [ ] Caching works (check cache stats)
- [ ] Constants estimated (even if rough)
- [ ] Losses are decreasing
- [ ] No NaN/Inf values
- [ ] Validation shows adaptation gain
- [ ] GPU works (if available)
- [ ] Save/load checkpoints works

**Only proceed to full experiments when ALL boxes checked!**
