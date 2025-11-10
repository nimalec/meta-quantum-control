"""
Test gradient flow through the entire pipeline.

This diagnoses where gradients are breaking in the meta-learning setup.
"""

import torch
import numpy as np
import yaml
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from metaqctrl.quantum.noise_adapter import TaskDistribution, NoiseParameters
from metaqctrl.meta_rl.policy import PulsePolicy
from metaqctrl.theory.quantum_environment import create_quantum_environment


def test_gradient_flow():
    """Test if gradients flow through the loss function."""

    print("="*70)
    print("GRADIENT FLOW DIAGNOSTIC TEST")
    print("="*70)

    # Load config
    config_path = Path(__file__).parent.parent / "configs" / "experiment_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"\n1. Config loaded:")
    print(f"   first_order: {config.get('first_order')}")
    print(f"   first_order type: {type(config.get('first_order'))}")

    # Device
    device = torch.device('cpu')  # Use CPU for debugging

    # Create environment
    print(f"\n2. Creating quantum environment...")
    from metaqctrl.quantum.gates import TargetGates

    target_gate_name = config.get('target_gate')
    if target_gate_name == 'pauli_x':
        U_target = TargetGates.pauli_x()
    else:
        U_target = TargetGates.hadamard()

    ket_0 = np.array([1, 0], dtype=complex)
    target_state = np.outer(U_target @ ket_0, (U_target @ ket_0).conj())

    env = create_quantum_environment(config, target_state)
    print(f"   Environment created: {env.get_cache_stats()}")

    # Create policy
    print(f"\n3. Creating policy...")
    policy = PulsePolicy(
        task_feature_dim=config.get('task_feature_dim'),
        hidden_dim=config.get('hidden_dim'),
        n_hidden_layers=config.get('n_hidden_layers'),
        n_segments=config.get('n_segments'),
        n_controls=config.get('n_controls'),
        output_scale=config.get('output_scale'),
        activation=config.get('activation')
    ).to(device)

    print(f"   Policy created with {policy.count_parameters()} parameters")

    # Check policy parameters
    print(f"\n4. Checking policy parameters...")
    policy.train()  # Ensure training mode
    param_count = 0
    for name, param in policy.named_parameters():
        param_count += 1
        if param_count <= 2:  # Print first 2
            print(f"   {name}: requires_grad={param.requires_grad}, shape={param.shape}")
    print(f"   Total parameters: {param_count}")

    # Sample a task
    print(f"\n5. Sampling task...")
    task_dist = TaskDistribution(
        dist_type=config.get('task_dist_type'),
        ranges={
            'alpha': tuple(config.get('alpha_range')),
            'A': tuple(config.get('A_range')),
            'omega_c': tuple(config.get('omega_c_range'))
        }
    )
    rng = np.random.default_rng(42)
    tasks = task_dist.sample(1, rng)
    task_params = tasks[0]
    print(f"   Task: alpha={task_params.alpha:.3f}, A={task_params.A:.3f}, omega_c={task_params.omega_c:.3f}")

    # Test loss computation
    print(f"\n6. Testing loss computation...")

    try:
        loss = env.compute_loss_differentiable(
            policy,
            task_params,
            device,
            use_rk4=config.get('use_rk4_training'),
            dt=config.get('dt_training')
        )

        print(f"   Loss value: {loss.item():.6f}")
        print(f"   Loss dtype: {loss.dtype}")
        print(f"   Loss requires_grad: {loss.requires_grad}")
        print(f"   Loss grad_fn: {loss.grad_fn}")
        print(f"   Loss is_leaf: {loss.is_leaf}")

        # Test backward pass
        print(f"\n7. Testing backward pass...")

        # Zero gradients
        policy.zero_grad()

        # Backward
        loss.backward()

        # Check gradients
        print(f"   Backward pass completed!")
        grad_count = 0
        zero_grad_count = 0
        for name, param in policy.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                if grad_norm > 1e-10:
                    grad_count += 1
                else:
                    zero_grad_count += 1

        print(f"   Parameters with gradients: {grad_count}/{param_count}")
        print(f"   Parameters with zero gradients: {zero_grad_count}/{param_count}")

        if grad_count > 0:
            print(f"\n{'='*70}")
            print(f"✓ SUCCESS! Gradients are flowing correctly!")
            print(f"{'='*70}")
            return True
        else:
            print(f"\n{'='*70}")
            print(f"✗ FAILURE! No gradients detected!")
            print(f"{'='*70}")
            return False

    except Exception as e:
        print(f"\n{'='*70}")
        print(f"✗ ERROR during loss computation!")
        print(f"{'='*70}")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_gradient_flow()
    sys.exit(0 if success else 1)
