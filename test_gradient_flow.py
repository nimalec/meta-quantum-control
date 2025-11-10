"""
Test gradient flow through the differentiable loss function.
This script diagnoses where gradients are breaking.
"""

import torch
import numpy as np
import yaml
from pathlib import Path

# Add parent to path
import sys
sys.path.insert(0, str(Path(__file__).parent / "metaqctrl"))

from metaqctrl.theory.quantum_environment import create_quantum_environment
from metaqctrl.meta_rl.policy import PulsePolicy
from metaqctrl.quantum.noise_adapter import NoiseParameters

def test_gradient_flow():
    """Test that gradients flow through the entire computation."""

    print("=" * 70)
    print("Testing Gradient Flow")
    print("=" * 70)

    # Load config
    config_path = Path(__file__).parent / "configs" / "experiment_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Create environment
    print("\n1. Creating quantum environment...")
    env = create_quantum_environment(config, target_state=None)
    print(f"   ✓ Environment created")

    # Create policy
    print("\n2. Creating policy...")
    policy = PulsePolicy(
        task_feature_dim=config.get('task_feature_dim'),
        hidden_dim=config.get('hidden_dim'),
        n_hidden_layers=config.get('n_hidden_layers'),
        n_segments=config.get('n_segments'),
        n_controls=config.get('n_controls'),
        output_scale=config.get('output_scale'),
        activation=config.get('activation')
    ).to(device)

    # Verify policy parameters have requires_grad
    for name, param in policy.named_parameters():
        if not param.requires_grad:
            print(f"   ✗ Parameter {name} does NOT require grad!")
        else:
            print(f"   ✓ Parameter {name} requires grad")

    print(f"   ✓ Policy created with {policy.count_parameters():,} parameters")

    # Create sample task
    print("\n3. Creating sample task...")
    task_params = NoiseParameters(
        alpha=1.0,
        A=100.0,
        omega_c=50.0,
        model_type='one_over_f'
    )
    print(f"   ✓ Task params: α={task_params.alpha}, A={task_params.A}, ωc={task_params.omega_c}")

    # Compute loss
    print("\n4. Computing loss with differentiable simulator...")
    policy.train()  # Ensure training mode

    loss = env.compute_loss_differentiable(
        policy,
        task_params,
        device,
        use_rk4=config.get('use_rk4_training', True),
        dt=config.get('dt_training', 0.01)
    )

    print(f"   Loss value: {loss.item():.6f}")
    print(f"   Loss dtype: {loss.dtype}")
    print(f"   Loss device: {loss.device}")
    print(f"   Loss requires_grad: {loss.requires_grad}")
    print(f"   Loss grad_fn: {loss.grad_fn}")

    # Check if loss has computational graph
    if loss.grad_fn is None:
        print("   ✗ ERROR: Loss has no grad_fn! Gradients will not flow!")
        return False
    else:
        print("   ✓ Loss has grad_fn, computational graph exists")

    # Try backward pass
    print("\n5. Testing backward pass...")
    try:
        loss.backward()
        print("   ✓ Backward pass successful!")

        # Check gradients
        grad_found = False
        for name, param in policy.named_parameters():
            if param.grad is not None and param.grad.abs().max() > 0:
                grad_found = True
                print(f"   ✓ Gradient for {name}: norm={param.grad.norm().item():.6f}")
                break

        if not grad_found:
            print("   ✗ WARNING: No non-zero gradients found!")
            return False

        print("\n" + "=" * 70)
        print("✓ GRADIENT FLOW TEST PASSED!")
        print("=" * 70)
        return True

    except Exception as e:
        print(f"   ✗ ERROR during backward pass: {e}")
        print("\n" + "=" * 70)
        print("✗ GRADIENT FLOW TEST FAILED!")
        print("=" * 70)
        return False

if __name__ == "__main__":
    success = test_gradient_flow()
    sys.exit(0 if success else 1)
