# Differentiable Quantum Simulation

## Overview

This document explains the **critical differentiable simulation capability** that enables proper gradient flow through quantum dynamics for meta-learning.

### Architecture

```python
class DifferentiableLindbladSimulator(nn.Module):
    """
    Fully differentiable Lindblad master equation solver.

    Implements: ρ̇(t) = -i[H(t), ρ(t)] + Σⱼ (Lⱼ ρ L†ⱼ - ½{L†ⱼLⱼ, ρ})
    """

    def __init__(self, H0, H_controls, L_operators, dt=0.05, method='rk4'):
        # All operators are PyTorch tensors (complex64)
        self.H0 = H0.to(torch.complex64)
        self.H_controls = [H.to(torch.complex64) for H in H_controls]
        self.L_operators = [L.to(torch.complex64) for L in L_operators]
```

### Key Features

1. **RK4 Integration**: 4th-order Runge-Kutta for accurate time evolution
   ```python
   def step_rk4(self, rho, u, dt):
       k1 = self.lindbladian(rho, u)
       k2 = self.lindbladian(rho + 0.5*dt*k1, u)
       k3 = self.lindbladian(rho + 0.5*dt*k2, u)
       k4 = self.lindbladian(rho + dt*k3, u)
       return rho + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
   ```

2. **Full Gradient Support**: All operations use PyTorch's autograd
   ```python
   def lindbladian(self, rho, u):
       # Build Hamiltonian (differentiable)
       H_total = self.H0.clone()
       for k, u_k in enumerate(u):
           H_total = H_total + u_k * self.H_controls[k]

       # Hamiltonian evolution (differentiable)
       hamiltonian_term = -1j * (H_total @ rho - rho @ H_total)

       # Dissipation (differentiable)
       dissipation_term = ...

       return hamiltonian_term + dissipation_term
   ```

3. **GPU Compatible**: Runs on CPU or GPU
   ```python
   sim = DifferentiableLindbladSimulator(..., device=torch.device('cuda'))
   ```

### Usage

```python
from metaqctrl.theory.quantum_environment import QuantumEnvironment

env = QuantumEnvironment(H0, H_controls, psd_to_lindblad, target_state, T=1.0)

# NEW: Differentiable loss (use this for meta-learning!)
loss = env.compute_loss_differentiable(
    policy,
    task_params,
    device=torch.device('cpu'),
    use_rk4=True  # More accurate but slower
)

# Gradients flow all the way through!
loss.backward()
```

### Comparison

| Feature | `compute_loss()` (OLD) | `compute_loss_differentiable()` (NEW) |
|---------|------------------------|---------------------------------------|
| Gradient flow | ❌ Broken (detach) | ✅ Full gradient flow |
| Backend | NumPy/SciPy | PyTorch |
| Meta-learning | ⚠️ First-order only | ✅ Second-order MAML |
| Speed | Faster | Slightly slower |
| Accuracy | High (adaptive RK45) | High (RK4) |
| GPU support | ❌ No | ✅ Yes |

### When to Use Which

**Use `compute_loss_differentiable()`** (NEW):
- ✅ Training meta-learning policies
- ✅ Any gradient-based optimization
- ✅ Second-order MAML
- ✅ When you need gradients through quantum dynamics

**Use `compute_loss()`** (OLD):
- ✅ Evaluation only (no training)
- ✅ Zeroth-order optimization (e.g., evolution strategies)
- ✅ When speed is critical and gradients not needed

## Implementation Details

### Fidelity Computation

For differentiability, we use a simplified fidelity formula:

```python
def _torch_state_fidelity(self, rho, sigma):
    """
    Differentiable fidelity: F(ρ,σ) ≈ |tr(ρσ)|²

    This approximation works well for high-fidelity control.
    The full formula F = tr(√(√ρ σ √ρ))² requires matrix square roots
    which are difficult to differentiate numerically.
    """
    overlap = torch.trace(rho @ sigma)
    fidelity = torch.abs(overlap) ** 2
    return torch.clamp(fidelity.real, 0.0, 1.0)
```

### Numerical Stability

- **Complex arithmetic**: All quantum operators use `torch.complex64`
- **Hermiticity preservation**: Dissipation operators are properly anti-Hermitian
- **Time stepping**: Adaptive substeps within each control segment
- **Clamping**: Fidelity clamped to [0, 1] for numerical stability

## Testing

Comprehensive test suite in `tests/test_differentiable_simulation.py`:

1. **Test 1**: Basic gradient flow through simulator
   - Verifies gradients computed correctly
   - Checks gradient magnitude is non-zero

2. **Test 2**: Policy → Simulation gradient flow
   - End-to-end: policy network → controls → quantum dynamics → fidelity
   - Verifies all policy parameters receive gradients

3. **Test 3**: QuantumEnvironment integration
   - Tests `compute_loss_differentiable()` method
   - Compares with non-differentiable version

4. **Test 4**: Training step
   - Simulates gradient descent
   - Verifies loss decreases with optimization

Run tests:
```bash
python tests/test_differentiable_simulation.py
# OR
pytest tests/test_differentiable_simulation.py -v
```

## Migration Guide

### For Existing Code

If you're using the old non-differentiable loss:

```python
# OLD
loss = env.compute_loss(policy, task_params)
```

Simply replace with:

```python
# NEW
loss = env.compute_loss_differentiable(policy, task_params)
```

### For MAML Training

Update your meta-learning loops:

```python
# Inner loop (adaptation)
for task in batch_tasks:
    # Use differentiable loss for adaptation
    support_loss = env.compute_loss_differentiable(
        adapted_policy,
        task.support_params
    )
    support_loss.backward()  # Gradients flow!
    optimizer.step()

# Outer loop (meta-update)
meta_loss = 0
for task in batch_tasks:
    query_loss = env.compute_loss_differentiable(
        adapted_policy,
        task.query_params
    )
    meta_loss += query_loss

meta_loss.backward()  # Second-order gradients!
meta_optimizer.step()
```

## Performance Considerations

### Speed

- **RK4 mode** (`use_rk4=True`): More accurate, ~2x slower
- **Euler mode** (`use_rk4=False`): Faster, less accurate
- **Recommendation**: Use RK4 for final training, Euler for debugging

### Memory

Differentiable simulation stores computation graph:
- Memory usage: O(n_segments × d²)
- For large systems, consider gradient checkpointing

### GPU Acceleration

For significant speedup on GPU:

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
policy = policy.to(device)

loss = env.compute_loss_differentiable(
    policy,
    task_params,
    device=device
)
```

## Validation

### Gradient Flow Verification

```python
# Check gradient flow
controls = policy(task_features)
rho_final = sim(rho0, controls, T=1.0)
fidelity = compute_fidelity(rho_final, target)
loss = 1.0 - fidelity

loss.backward()

# Verify gradients
for name, param in policy.named_parameters():
    assert param.grad is not None, f"No gradient for {name}!"
    assert torch.norm(param.grad) > 0, f"Zero gradient for {name}!"
    print(f"✓ {name}: grad_norm = {torch.norm(param.grad):.4e}")
```

### Accuracy Comparison

Compare differentiable vs non-differentiable results:

```python
# Should give similar fidelity values
fid_old = env.evaluate_policy(policy, task_params)
loss_new = env.compute_loss_differentiable(policy, task_params)
fid_new = 1.0 - loss_new.item()

print(f"Old fidelity: {fid_old:.4f}")
print(f"New fidelity: {fid_new:.4f}")
print(f"Difference: {abs(fid_old - fid_new):.4e}")
```

## Limitations

1. **Approximations**:
   - Simplified fidelity formula (works well for high fidelity)
   - Piecewise-constant controls (not infinitely smooth)

2. **Speed**:
   - Slower than NumPy/SciPy for single forward passes
   - Much faster overall due to batching and GPU support

3. **Memory**:
   - Computation graph storage for long simulations
   - Mitigate with gradient checkpointing if needed

## Future Improvements

Potential enhancements:

1. **JAX backend**: For even faster differentiation and JIT compilation
2. **Gradient checkpointing**: Reduce memory for long trajectories
3. **Batched simulation**: Parallel task evaluation on GPU
4. **Adjoint method**: More memory-efficient gradients for very long simulations

## References

- **Lindblad Master Equation**: Lindblad (1976), Gorini-Kossakowski-Sudarshan (1976)
- **GRAPE**: Khaneja et al., JMR 2005
- **Differentiable Physics**: Degrave et al., arXiv:1611.01652
- **PyTorch Autograd**: Paszke et al., NeurIPS 2017

## Citation

If you use this differentiable simulation capability, please cite:

```bibtex
@article{meta-quantum-control-2025,
  title={Meta-Reinforcement Learning for Adaptive Quantum Control},
  author={...},
  journal={ICML},
  year={2025}
}
```
