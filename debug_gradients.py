"""
Diagnostic Script for Zero Gradient Issue

This script tests gradient flow through each component:
1. Policy network
2. Differentiable simulator
3. Fidelity computation
4. Full loss function
5. MAML inner/outer loop
"""

import torch
import numpy as np
import yaml

print("=" * 70)
print("GRADIENT FLOW DIAGNOSTIC")
print("=" * 70)

# Load config
with open('configs/experiment_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

device = torch.device('cpu')
print(f"\nDevice: {device}")

# ============================================================================
# TEST 1: Policy Network Gradients
# ============================================================================
print("\n" + "=" * 70)
print("TEST 1: Policy Network")
print("=" * 70)

from metaqctrl.meta_rl.policy import PulsePolicy

policy = PulsePolicy(
    task_feature_dim=3,
    hidden_dim=128,
    n_hidden_layers=2,
    n_segments=20,
    n_controls=2,
    output_scale=2.0
).to(device)

print(f"Policy parameters: {policy.count_parameters():,}")

# Forward pass
task_features = torch.tensor([1.0, 0.05, 5.0], device=device, dtype=torch.float32)
controls = policy(task_features)

print(f"Controls shape: {controls.shape}")
print(f"Controls range: [{controls.min().item():.3f}, {controls.max().item():.3f}]")
print(f"Controls requires_grad: {controls.requires_grad}")

# Dummy loss
loss = controls.abs().mean()
loss.backward()

# Check gradients
grad_count = 0
zero_grad_count = 0
for name, param in policy.named_parameters():
    if param.grad is not None:
        grad_count += 1
        if param.grad.abs().max() < 1e-10:
            zero_grad_count += 1

print(f"\n✓ Parameters with gradients: {grad_count}")
print(f"✗ Parameters with zero gradients: {zero_grad_count}")

if zero_grad_count == 0:
    print("✓ TEST 1 PASSED: Policy gradients flow correctly")
else:
    print("✗ TEST 1 FAILED: Some policy parameters have zero gradients")

# ============================================================================
# TEST 2: Differentiable Simulator
# ============================================================================
print("\n" + "=" * 70)
print("TEST 2: Differentiable Simulator")
print("=" * 70)

from metaqctrl.quantum.lindblad_torch import DifferentiableLindbladSimulator, numpy_to_torch_complex

# Create simple 1-qubit system
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

H0 = 0.0 * sigma_z
H_controls = [sigma_x, sigma_y]

# Simple dephasing noise
L_ops = [0.1 * sigma_z]

# Convert to torch
H0_torch = numpy_to_torch_complex(H0, device)
H_controls_torch = [numpy_to_torch_complex(H, device) for H in H_controls]
L_ops_torch = [numpy_to_torch_complex(L, device) for L in L_ops]

sim = DifferentiableLindbladSimulator(
    H0=H0_torch,
    H_controls=H_controls_torch,
    L_operators=L_ops_torch,
    dt=0.001,
    method='rk4',
    device=device
)

print(f"Simulator created: d={sim.d}, n_controls={sim.n_controls}")

# Initial state
rho0 = torch.zeros((2, 2), dtype=torch.complex64, device=device)
rho0[0, 0] = 1.0

# Control sequence (with gradients)
controls_test = torch.nn.Parameter(torch.randn(20, 2, device=device) * 0.5)
print(f"Test controls requires_grad: {controls_test.requires_grad}")
print(f"Test controls is_leaf: {controls_test.is_leaf}")

# Simulate
rho_final = sim(rho0, controls_test, T=1.0)
print(f"Final state shape: {rho_final.shape}")
print(f"Final state trace: {torch.trace(rho_final).real.item():.4f}")

# Compute purity loss
purity = torch.trace(rho_final @ rho_final).real
loss_sim = 1.0 - purity
print(f"Purity: {purity.item():.4f}")
print(f"Loss: {loss_sim.item():.4f}")

# Backward
loss_sim.backward()

print(f"Controls gradient exists: {controls_test.grad is not None}")
if controls_test.grad is not None:
    grad_norm = torch.norm(controls_test.grad).item()
    print(f"Controls gradient norm: {grad_norm:.4e}")
    if grad_norm > 1e-10:
        print("✓ TEST 2 PASSED: Simulator gradients flow correctly")
    else:
        print("✗ TEST 2 FAILED: Simulator gradients are zero")
else:
    print("✗ TEST 2 FAILED: No gradients computed")

# ============================================================================
# TEST 3: Fidelity Computation
# ============================================================================
print("\n" + "=" * 70)
print("TEST 3: Fidelity Computation")
print("=" * 70)

from metaqctrl.quantum.gates import TargetGates

# Target state (Hadamard|0⟩)
U_target = TargetGates.hadamard()
ket_0 = np.array([1, 0], dtype=complex)
target_state_np = np.outer(U_target @ ket_0, (U_target @ ket_0).conj())
target_state_torch = numpy_to_torch_complex(target_state_np, device)

print(f"Target state shape: {target_state_torch.shape}")

# Create test state (parameterized)
theta = torch.nn.Parameter(torch.tensor([0.5], device=device))

def create_test_state(angle):
    """Create |ψ⟩ = cos(θ)|0⟩ + sin(θ)|1⟩"""
    ket = torch.zeros(2, dtype=torch.complex64, device=device)
    ket[0] = torch.cos(angle)
    ket[1] = torch.sin(angle)
    # Density matrix
    rho = torch.outer(ket, ket.conj())
    return rho

test_state = create_test_state(theta[0])

# Compute fidelity using environment's method
from metaqctrl.theory.quantum_environment import QuantumEnvironment

# Create a dummy environment just to use the fidelity function
env_dummy = QuantumEnvironment(
    H0=H0,
    H_controls=H_controls,
    psd_to_lindblad=None,  # Won't use this
    target_state=target_state_np,
    T=1.0
)

fidelity = env_dummy._torch_state_fidelity(test_state, target_state_torch)
loss_fid = 1.0 - fidelity

print(f"Fidelity: {fidelity.item():.4f}")
print(f"Loss: {loss_fid.item():.4f}")
print(f"Fidelity requires_grad: {fidelity.requires_grad}")

# Backward
loss_fid.backward()

print(f"Theta gradient exists: {theta.grad is not None}")
if theta.grad is not None:
    print(f"Theta gradient: {theta.grad.item():.4e}")
    if abs(theta.grad.item()) > 1e-10:
        print("✓ TEST 3 PASSED: Fidelity gradients flow correctly")
    else:
        print("✗ TEST 3 FAILED: Fidelity gradients are zero")
else:
    print("✗ TEST 3 FAILED: No gradients computed")

# ============================================================================
# TEST 4: Full Loss Function
# ============================================================================
print("\n" + "=" * 70)
print("TEST 4: Full Loss Function (Policy → Simulator → Fidelity)")
print("=" * 70)

from metaqctrl.theory.quantum_environment import create_quantum_environment
from metaqctrl.quantum.noise_models import NoiseParameters

# Create environment
env = create_quantum_environment(config, target_state_np)
print(f"Environment created: {env.get_cache_stats()}")

# Create fresh policy
policy_test = PulsePolicy(
    task_feature_dim=3,
    hidden_dim=64,  # Smaller for faster testing
    n_hidden_layers=2,
    n_segments=20,
    n_controls=2,
    output_scale=2.0
).to(device)

# Zero gradients
for param in policy_test.parameters():
    if param.grad is not None:
        param.grad.zero_()

# Create task
task_params = NoiseParameters(alpha=1.0, A=0.05, omega_c=5.0)

print(f"Task parameters: alpha={task_params.alpha}, A={task_params.A}, omega_c={task_params.omega_c}")

# Compute loss
policy_test.train()
loss_full = env.compute_loss_differentiable(policy_test, task_params, device)

print(f"Full loss: {loss_full.item():.4f}")
print(f"Loss requires_grad: {loss_full.requires_grad}")

# Backward
loss_full.backward()

# Check policy gradients
grad_count_full = 0
zero_grad_count_full = 0
total_grad_norm = 0.0

for name, param in policy_test.named_parameters():
    if param.grad is not None:
        grad_count_full += 1
        param_grad_norm = param.grad.norm().item()
        total_grad_norm += param_grad_norm ** 2

        if param_grad_norm < 1e-10:
            zero_grad_count_full += 1

total_grad_norm = np.sqrt(total_grad_norm)

print(f"\n✓ Parameters with gradients: {grad_count_full}")
print(f"✗ Parameters with zero gradients: {zero_grad_count_full}")
print(f"Total gradient norm: {total_grad_norm:.4e}")

if total_grad_norm > 1e-10 and zero_grad_count_full == 0:
    print("✓ TEST 4 PASSED: Full pipeline gradients flow correctly")
elif total_grad_norm > 1e-10:
    print("⚠ TEST 4 PARTIAL: Some gradients flow but some are zero")
else:
    print("✗ TEST 4 FAILED: No gradients flow through full pipeline")

# ============================================================================
# TEST 5: MAML Inner Loop
# ============================================================================
print("\n" + "=" * 70)
print("TEST 5: MAML Inner Loop (Adaptation)")
print("=" * 70)

try:
    import higher
    print("✓ 'higher' library is installed")
    HIGHER_AVAILABLE = True
except ImportError:
    print("✗ 'higher' library NOT installed - this is likely the problem!")
    print("  Install with: pip install higher")
    HIGHER_AVAILABLE = False

from metaqctrl.meta_rl.maml import MAML

# Create MAML instance
maml = MAML(
    policy=policy_test,
    inner_lr=0.001,
    inner_steps=3,
    meta_lr=0.001,
    first_order=config.get('first_order', True),
    device=device
)

print(f"MAML settings: inner_lr={maml.inner_lr}, inner_steps={maml.inner_steps}, first_order={maml.first_order}")

# Create dummy task data
def create_dummy_task_data(task_params):
    task_features = torch.tensor(
        task_params.to_array(),
        dtype=torch.float32,
        device=device
    )
    task_features_batch = task_features.unsqueeze(0).repeat(5, 1)

    return {
        'task_features': task_features_batch,
        'task_params': task_params
    }

support_data = create_dummy_task_data(task_params)
query_data = create_dummy_task_data(task_params)

task_data = {
    'support': support_data,
    'query': query_data
}

# Loss function
def loss_fn(policy, data):
    return env.compute_loss_differentiable(policy, data['task_params'], device)

# Test inner loop
print("\nTesting inner loop adaptation...")
if HIGHER_AVAILABLE and not maml.first_order:
    fmodel, inner_losses = maml.inner_loop_higher(task_data, loss_fn)
    print(f"Inner losses: {[f'{l:.4f}' for l in inner_losses]}")

    # Query loss
    query_loss = loss_fn(fmodel, query_data)
    print(f"Query loss: {query_loss.item():.4f}")
    print(f"Query loss requires_grad: {query_loss.requires_grad}")

else:
    adapted_policy, inner_losses = maml.inner_loop(task_data, loss_fn)
    print(f"Inner losses: {[f'{l:.4f}' for l in inner_losses]}")

    query_loss = loss_fn(adapted_policy, query_data)
    print(f"Query loss: {query_loss.item():.4f}")
    print(f"Query loss requires_grad: {query_loss.requires_grad}")

# ============================================================================
# TEST 6: MAML Meta-Update
# ============================================================================
print("\n" + "=" * 70)
print("TEST 6: MAML Meta-Update (Outer Loop)")
print("=" * 70)

# Zero meta-parameters gradients
maml.meta_optimizer.zero_grad()

# Create task batch
task_batch = [task_data for _ in range(2)]

# Meta-training step
print("Performing meta-training step...")
metrics = maml.meta_train_step(task_batch, loss_fn, use_higher=HIGHER_AVAILABLE)

print(f"\nMeta-training metrics:")
for key, value in metrics.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.4e}")
    else:
        print(f"  {key}: {value}")

# Check meta-parameter gradients
meta_grad_count = 0
meta_zero_grad_count = 0
meta_grad_norm = 0.0

for name, param in maml.policy.named_parameters():
    if param.grad is not None:
        meta_grad_count += 1
        param_grad_norm = param.grad.norm().item()
        meta_grad_norm += param_grad_norm ** 2

        if param_grad_norm < 1e-10:
            meta_zero_grad_count += 1

meta_grad_norm = np.sqrt(meta_grad_norm)

print(f"\nMeta-parameter gradients:")
print(f"  Parameters with gradients: {meta_grad_count}")
print(f"  Parameters with zero gradients: {meta_zero_grad_count}")
print(f"  Total gradient norm: {meta_grad_norm:.4e}")

if meta_grad_norm > 1e-10 and meta_zero_grad_count == 0:
    print("✓ TEST 6 PASSED: MAML meta-gradients flow correctly")
elif meta_grad_norm > 1e-10:
    print("⚠ TEST 6 PARTIAL: Some meta-gradients flow but some are zero")
else:
    print("✗ TEST 6 FAILED: No meta-gradients - THIS IS YOUR PROBLEM!")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("DIAGNOSTIC SUMMARY")
print("=" * 70)

print("\nKey Findings:")
if not HIGHER_AVAILABLE:
    print("  ✗ CRITICAL: 'higher' library not installed!")
    print("    → This is required for proper MAML gradient flow")
    print("    → Install with: pip install higher")
else:
    print("  ✓ 'higher' library is installed")

if meta_grad_norm < 1e-10:
    print("  ✗ CRITICAL: Meta-gradients are zero!")
    print("    → Check if first_order=True is causing issues")
    print("    → Verify 'higher' library is being used correctly")
elif meta_zero_grad_count > 0:
    print(f"  ⚠ WARNING: {meta_zero_grad_count} parameters have zero gradients")
else:
    print("  ✓ Meta-gradients are flowing correctly!")

print("\n" + "=" * 70)
print("END OF DIAGNOSTIC")
print("=" * 70)
