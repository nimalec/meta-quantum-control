"""
Test gradient flow through the entire pipeline.
"""

import torch
import numpy as np
import yaml

from metaqctrl.quantum.noise_models import TaskDistribution, NoiseParameters
from metaqctrl.quantum.gates import TargetGates
from metaqctrl.meta_rl.policy import PulsePolicy
from metaqctrl.theory.quantum_environment import create_quantum_environment

print("=" * 70)
print("GRADIENT FLOW TEST")
print("=" * 70)

# Load config
with open('configs/experiment_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Device
device = torch.device('cpu')

# Target state
U_target = TargetGates.hadamard()
ket_0 = np.array([1, 0], dtype=complex)
target_state = np.outer(U_target @ ket_0, (U_target @ ket_0).conj())

# Create quantum environment
env = create_quantum_environment(config, target_state)

# Create policy with requires_grad
policy = PulsePolicy(
    task_feature_dim=3,
    hidden_dim=128,
    n_hidden_layers=2,
    n_segments=20,
    n_controls=2,
    output_scale=2.0,
    activation='tanh'
).to(device)

# Verify parameters require grad
print("\nPolicy parameters:")
for name, param in policy.named_parameters():
    print(f"  {name}: requires_grad={param.requires_grad}, shape={param.shape}")

# Sample a task
task_params = NoiseParameters(alpha=1.0, A=0.01, omega_c=5.0)
print(f"\nTask: α={task_params.alpha}, A={task_params.A}, ωc={task_params.omega_c}")

# Compute loss
print("\nComputing loss...")
policy.train()
loss = env.compute_loss_differentiable(policy, task_params, device)
print(f"Loss: {loss.item():.6f}")
print(f"Loss requires_grad: {loss.requires_grad}")
print(f"Loss grad_fn: {loss.grad_fn}")

# Backward pass
print("\nComputing gradients...")
loss.backward()

# Check gradients
print("\nGradient statistics:")
total_grad_norm = 0.0
n_params_with_grad = 0
n_params_total = 0

for name, param in policy.named_parameters():
    n_params_total += 1
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        total_grad_norm += grad_norm ** 2
        n_params_with_grad += 1
        print(f"  {name}: grad_norm={grad_norm:.6f}")
    else:
        print(f"  {name}: NO GRADIENT!")

total_grad_norm = np.sqrt(total_grad_norm)

print(f"\nSummary:")
print(f"  Total gradient norm: {total_grad_norm:.6f}")
print(f"  Parameters with gradients: {n_params_with_grad}/{n_params_total}")

if total_grad_norm > 0:
    print("\n✓ GRADIENTS ARE FLOWING!")
else:
    print("\n✗ GRADIENT FLOW BROKEN!")

print("=" * 70)
