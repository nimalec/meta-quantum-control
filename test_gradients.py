"""
Test script to verify gradient flow through the entire pipeline.
"""

import torch
import numpy as np
import yaml

# Add parent directory to path
import sys
sys.path.insert(0, '.')

from metaqctrl.meta_rl.policy import PulsePolicy
from metaqctrl.theory.quantum_environment import create_quantum_environment
from metaqctrl.quantum.noise_models import NoiseParameters

print("=" * 70)
print("Testing Gradient Flow")
print("=" * 70)

# Load config
config_path = 'configs/experiment_config.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

# Create quantum environment
print("\n1. Creating quantum environment...")
from metaqctrl.quantum.gates import TargetGates
ket_0 = np.array([1, 0], dtype=complex)
U_target = TargetGates.pauli_x()
target_state = np.outer(U_target @ ket_0, (U_target @ ket_0).conj())

env = create_quantum_environment(config, target_state, U_target)
print(f"   ✓ Environment created")

# Create policy
print("\n2. Creating policy...")
policy = PulsePolicy(
    task_feature_dim=3,
    hidden_dim=64,
    n_hidden_layers=1,
    n_segments=20,
    n_controls=2,
    output_scale=2.0
).to(device)
print(f"   ✓ Policy created with {policy.count_parameters():,} parameters")

# Create a sample task
print("\n3. Creating sample task...")
task_params = NoiseParameters(alpha=1.0, A=0.05, omega_c=500.0)
print(f"   ✓ Task: α={task_params.alpha}, A={task_params.A}, ω_c={task_params.omega_c}")

# Forward pass with gradient tracking
print("\n4. Testing gradient flow...")
policy.train()
policy.zero_grad()

# Generate controls
task_features = torch.tensor(
    task_params.to_array(),
    dtype=torch.float32,
    device=device,
    requires_grad=False
)

controls = policy(task_features)
print(f"   Controls shape: {controls.shape}")
print(f"   Controls require grad: {controls.requires_grad}")

# Compute loss
loss = env.compute_loss_differentiable(
    policy,
    task_params,
    device,
    use_rk4=True,
    dt=0.01
)
print(f"   Loss: {loss.item():.4f}")
print(f"   Loss requires grad: {loss.requires_grad}")

# Backward pass
print("\n5. Computing gradients...")
loss.backward()

# Check gradients
print("\n6. Checking gradients...")
grad_stats = []
zero_grad_count = 0
total_params = 0

for name, param in policy.named_parameters():
    total_params += 1
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        grad_max = param.grad.abs().max().item()
        grad_mean = param.grad.abs().mean().item()

        if grad_max < 1e-10:
            zero_grad_count += 1
            status = "❌ ZERO"
        else:
            status = "✓ OK"

        grad_stats.append({
            'name': name,
            'norm': grad_norm,
            'max': grad_max,
            'mean': grad_mean,
            'status': status
        })
    else:
        zero_grad_count += 1
        grad_stats.append({
            'name': name,
            'norm': 0.0,
            'max': 0.0,
            'mean': 0.0,
            'status': "❌ NONE"
        })

# Print results
print(f"\n{'Parameter':<30} {'Norm':>12} {'Max':>12} {'Mean':>12} {'Status':>10}")
print("-" * 80)
for stat in grad_stats:
    print(f"{stat['name']:<30} {stat['norm']:>12.6e} {stat['max']:>12.6e} {stat['mean']:>12.6e} {stat['status']:>10}")

print("\n" + "=" * 70)
if zero_grad_count == 0:
    print("✓✓✓ SUCCESS! All gradients are non-zero!")
    print("=" * 70)
    exit(0)
else:
    print(f"❌❌❌ FAILURE! {zero_grad_count}/{total_params} parameters have zero/no gradients")
    print("=" * 70)
    exit(1)
