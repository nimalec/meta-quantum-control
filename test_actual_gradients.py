"""
Test actual gradient flow with the real quantum environment and policy.
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

from metaqctrl.meta_rl.policy import PulsePolicy
from metaqctrl.quantum.noise_adapter import NoiseParameters, NoisePSDModel, PSDToLindblad2
from metaqctrl.quantum.gates import TargetGates
from metaqctrl.theory.quantum_environment import create_quantum_environment

def test_loss_gradient():
    """Test if the loss function actually produces gradients."""

    print("="*70)
    print("TESTING ACTUAL QUANTUM ENVIRONMENT GRADIENT FLOW")
    print("="*70)

    # Minimal config
    config = {
        'psd_model': 'one_over_f',
        'horizon': 1.0,
        'target_gate': 'pauli_x',
        'noise_type': 'frequency',
        'task_feature_dim': 3,
        'hidden_dim': 64,
        'n_hidden_layers': 1,
        'n_segments': 20,
        'n_controls': 2,
        'output_scale': 1.0,
        'activation': 'tanh',
        'drift_strength': 0.5,
        'sequence': 'ramsey',
        'dt_training': 0.01,
        'use_rk4_training': True,
        'omega0': 6.28,
        'Gamma_h': 100,
        'num_qubits': 1,
        'integration_method': 'RK45'
    }

    device = torch.device('cpu')

    # Create environment
    print("\n1. Creating quantum environment...")
    U_target = TargetGates.pauli_x()
    ket_0 = np.array([1, 0], dtype=complex)
    target_state = np.outer(U_target @ ket_0, (U_target @ ket_0).conj())

    try:
        env = create_quantum_environment(config, target_state, U_target)
        print(f"   ✓ Environment created")
    except Exception as e:
        print(f"   ✗ Failed to create environment: {e}")
        return False

    # Create policy
    print("\n2. Creating policy...")
    policy = PulsePolicy(
        task_feature_dim=3,
        hidden_dim=64,
        n_hidden_layers=1,
        n_segments=20,
        n_controls=2,
        output_scale=1.0
    ).to(device)

    policy.train()  # CRITICAL: Must be in train mode

    # Check parameters
    param_count = sum(1 for _ in policy.parameters())
    requires_grad_count = sum(1 for p in policy.parameters() if p.requires_grad)
    print(f"   ✓ Policy created: {param_count} params, {requires_grad_count} require grad")

    # Create task
    print("\n3. Creating task...")
    task_params = NoiseParameters(alpha=1.0, A=1000.0, omega_c=50.0, model_type='one_over_f')
    print(f"   ✓ Task: alpha={task_params.alpha}, A={task_params.A}, omega_c={task_params.omega_c}")

    # Test loss computation
    print("\n4. Computing loss...")
    try:
        loss = env.compute_loss_differentiable(
            policy,
            task_params,
            device,
            use_rk4=True,
            dt=0.01
        )

        print(f"   Loss value: {loss.item():.6f}")
        print(f"   Loss dtype: {loss.dtype}")
        print(f"   Loss device: {loss.device}")
        print(f"   Loss requires_grad: {loss.requires_grad}")
        print(f"   Loss grad_fn: {loss.grad_fn}")
        print(f"   Loss is_leaf: {loss.is_leaf}")

        if loss.grad_fn is None and not loss.requires_grad:
            print("\n   ✗ PROBLEM: Loss has no gradient connection!")
            return False

        print(f"   ✓ Loss has gradient connection")

    except Exception as e:
        print(f"   ✗ Failed to compute loss: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test backward
    print("\n5. Testing backward pass...")
    try:
        policy.zero_grad()
        loss.backward()

        grad_norms = []
        for name, param in policy.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)

        non_zero_grads = sum(1 for g in grad_norms if g > 1e-10)

        print(f"   Parameters with gradients: {len(grad_norms)}/{param_count}")
        print(f"   Parameters with non-zero gradients: {non_zero_grads}/{param_count}")

        if non_zero_grads > 0:
            print(f"   Max gradient norm: {max(grad_norms):.6e}")
            print(f"   ✓ SUCCESS! Gradients are flowing!")
            return True
        else:
            print(f"   ✗ WARNING: All gradients are zero (may indicate a problem)")
            return False

    except Exception as e:
        print(f"   ✗ Backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_loss_gradient()

    print("\n" + "="*70)
    if success:
        print("✓ GRADIENT FLOW TEST PASSED")
        print("\nThe loss function IS differentiable.")
        print("The issue must be elsewhere (likely in how MAML calls it).")
    else:
        print("✗ GRADIENT FLOW TEST FAILED")
        print("\nThe loss function is NOT differentiable.")
        print("This is the root cause of your MAML error.")
    print("="*70)

    sys.exit(0 if success else 1)
