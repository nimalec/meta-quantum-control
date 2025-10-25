"""
Quick Test Script for Training Fixes

Tests the key components to ensure:
1. Gradient flow works correctly
2. Numerical stability is improved
3. Loss values are reasonable
"""

import torch
import numpy as np
from metaqctrl.quantum.lindblad_torch import DifferentiableLindbladSimulator, numpy_to_torch_complex
from metaqctrl.meta_rl.policy import PulsePolicy
from metaqctrl.quantum.noise_models import NoiseParameters, NoisePSDModel, PSDToLindblad
from metaqctrl.theory.quantum_environment import create_quantum_environment
from metaqctrl.quantum.gates import TargetGates

print("=" * 70)
print("Testing Training Fixes")
print("=" * 70)

# Setup
device = torch.device('cpu')
torch.manual_seed(42)
np.random.seed(42)

# 1. Test Differentiable Lindblad Simulator
print("\n1. Testing Differentiable Lindblad Simulator...")
print("-" * 70)

# Pauli matrices
sigma_x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
sigma_y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)
sigma_z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)

# System setup
H0 = 0.0 * sigma_z
H_controls = [sigma_x, sigma_y]
L_ops = [torch.sqrt(torch.tensor(0.05)) * sigma_z]

# Create simulator with improved settings
sim = DifferentiableLindbladSimulator(
    H0=H0,
    H_controls=H_controls,
    L_operators=L_ops,
    dt=0.01,  # FIXED: Smaller time step
    method='rk4',
    device=device
)

# Initial state |0⟩
rho0 = torch.tensor([[1, 0], [0, 0]], dtype=torch.complex64)

# Control sequence
n_segments = 20
controls = torch.nn.Parameter(torch.randn(n_segments, 2) * 0.5)

print(f"  Simulator dt: {sim.dt}")
print(f"  Control shape: {controls.shape}")
print(f"  Requires grad: {controls.requires_grad}")

# Forward pass
rho_final, _ = sim.evolve(rho0, controls, T=1.0, return_trajectory=False, normalize=True)

# Check trace
trace = torch.trace(rho_final).real
print(f"  Final trace: {trace.item():.6f} (should be ~1.0)")

# Test gradient flow
purity = torch.trace(rho_final @ rho_final).real
loss = 1.0 - purity
loss.backward()

print(f"  Loss value: {loss.item():.4f}")
print(f"  Gradients exist: {controls.grad is not None}")
if controls.grad is not None:
    grad_norm = torch.norm(controls.grad).item()
    print(f"  Gradient norm: {grad_norm:.4e}")
    print(f"  ✓ Gradient flow WORKS!")
else:
    print(f"  ✗ Gradient flow FAILED!")

# 2. Test Policy Network
print("\n2. Testing Policy Network...")
print("-" * 70)

policy = PulsePolicy(
    task_feature_dim=3,
    hidden_dim=128,
    n_hidden_layers=2,
    n_segments=20,
    n_controls=2,
    output_scale=1.0,  # FIXED: Increased from 0.5
    activation='tanh'
)
policy = policy.to(device)

print(f"  Parameters: {policy.count_parameters():,}")
print(f"  Output scale: {policy.output_scale}")

# Test forward pass
task_features = torch.randn(3, device=device)
controls_out = policy(task_features)
print(f"  Output shape: {controls_out.shape}")
print(f"  Output range: [{controls_out.min().item():.2f}, {controls_out.max().item():.2f}]")
print(f"  Policy works!")

# 3. Test Quantum Environment with Differentiable Loss
print("\n3. Testing Quantum Environment with Differentiable Loss...")
print("-" * 70)

# Setup config
config = {
    'psd_model': 'one_over_f',
    'horizon': 1.0,
    'n_segments': 20,
    'integration_method': 'RK45'
}

# Target gate
U_target = TargetGates.hadamard()
ket_0 = np.array([1, 0], dtype=complex)
target_state = np.outer(U_target @ ket_0, (U_target @ ket_0).conj())

# Create environment
env = create_quantum_environment(config, target_state)
print(f"  Environment created: {env.d}D system")

# Create task
task_params = NoiseParameters(alpha=1.0, A=0.05, omega_c=5.0)
print(f"  Task: α={task_params.alpha}, A={task_params.A}, ωc={task_params.omega_c}")

# Test differentiable loss
policy.train()
loss = env.compute_loss_differentiable(policy, task_params, device)

print(f"  Loss value: {loss.item():.4f}")
print(f"  Loss requires grad: {loss.requires_grad}")
print(f"  Loss is finite: {torch.isfinite(loss).item()}")

# Test backward pass
loss.backward()

# Check gradients
has_grad = all(p.grad is not None for p in policy.parameters())
print(f"  All parameters have gradients: {has_grad}")

if has_grad:
    total_grad_norm = torch.sqrt(sum(torch.sum(p.grad**2) for p in policy.parameters())).item()
    print(f"  Total gradient norm: {total_grad_norm:.4e}")
    print(f"  Differentiable loss works")
else:
    print(f"  Some gradients missing")

# 4. Test MAML Inner Loop
print("\n4. Testing MAML Inner Loop...")
print("-" * 70)

from metaqctrl.meta_rl.maml import MAML

policy2 = PulsePolicy(
    task_feature_dim=3,
    hidden_dim=64,  # Smaller for faster test
    n_hidden_layers=2,
    n_segments=20,
    n_controls=2,
    output_scale=1.0,
    activation='tanh'
).to(device)

maml = MAML(
    policy=policy2,
    inner_lr=0.005,  # FIXED: Lower learning rate
    inner_steps=3,   # FIXED: Fewer steps initially
    meta_lr=0.001,
    first_order=False,
    device=device
)

print(f"  Inner LR: {maml.inner_lr}")
print(f"  Inner steps: {maml.inner_steps}")
print(f"  Meta LR: {maml.meta_lr}")

# Create dummy task
def loss_fn(policy, data):
    task_params = data['task_params']
    return env.compute_loss_differentiable(policy, task_params, device)

task_data = {
    'support': {
        'task_params': task_params
    },
    'query': {
        'task_params': task_params
    }
}

# Test inner loop
print("  Running inner loop adaptation...")
adapted_policy, inner_losses = maml.inner_loop(task_data, loss_fn)
print(f"  Inner loop losses: {[f'{l:.4f}' for l in inner_losses]}")

improvement = inner_losses[0] - inner_losses[-1]
print(f"  Loss improvement: {improvement:.4f}")

if improvement > 0:
    print(f"  Inner loop adaptation WORKS!")
else:
    print(f"  No improvement (may need more iterations or tuning)")

# 5. Test Meta-Training Step
print("\n5. Testing Meta-Training Step...")
print("-" * 70)

task_batch = [task_data for _ in range(2)]  # Small batch for quick test
metrics = maml.meta_train_step(task_batch, loss_fn, use_higher=True)

print(f"  Meta loss: {metrics['meta_loss']:.4f}")
print(f"  Mean task loss: {metrics['mean_task_loss']:.4f}")
print(f"  Gradient norm: {metrics['grad_norm']:.4f}")

if 'error' in metrics:
    print(f"  Error: {metrics['error']}")
else:
    print(f"  Meta-training step WORKS!")

print("\n" + "=" * 70)
print("Test Complete!")
print("=" * 70)
print("\nSummary:")
print("  If all tests passed, the training fixes are working correctly.")
print("  You can now run the full training with:")
print("    python experiments/train_meta.py --config configs/experiment_config.yaml")
print("=" * 70)
