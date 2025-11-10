"""
Comprehensive gradient flow diagnostic for MAML training.

This will test each component to find where gradients break.
"""

import torch
import numpy as np
import yaml
from pathlib import Path
import sys

# Load the actual training setup
from metaqctrl.quantum.lindblad import LindbladSimulator
from metaqctrl.quantum.noise_adapter import (
    TaskDistribution, NoisePSDModel, PSDToLindblad2, NoiseParameters,
    estimate_qubit_frequency_from_hamiltonian
)
from metaqctrl.quantum.gates import GateFidelityComputer, TargetGates
from metaqctrl.meta_rl.policy import PulsePolicy
from metaqctrl.meta_rl.maml import MAML, MAMLTrainer
from metaqctrl.theory.quantum_environment import create_quantum_environment


def diagnostic_test(config_path='configs/experiment_config.yaml'):
    """Run comprehensive gradient flow diagnostic."""

    print("\n" + "="*80)
    print(" "*20 + "GRADIENT FLOW DIAGNOSTIC TEST")
    print("="*80)

    # Load config
    print("\n[1/8] Loading configuration...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"      Config loaded from: {config_path}")
    print(f"      first_order: {config.get('first_order')}")
    print(f"      inner_lr: {config.get('inner_lr')}")
    print(f"      inner_steps: {config.get('inner_steps')}")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"      Device: {device}")

    # Create target state
    print("\n[2/8] Creating target state...")
    target_gate_name = config.get('target_gate')
    if target_gate_name == 'hadamard':
        U_target = TargetGates.hadamard()
    elif target_gate_name == 'pauli_x':
        U_target = TargetGates.pauli_x()
    else:
        raise ValueError(f"Unknown target gate: {target_gate_name}")

    ket_0 = np.array([1, 0], dtype=complex)
    target_state = np.outer(U_target @ ket_0, (U_target @ ket_0).conj())
    print(f"      Target gate: {target_gate_name}")

    # Create environment
    print("\n[3/8] Creating quantum environment...")
    env = create_quantum_environment(config, target_state, U_target)
    print(f"      Environment created: {env.get_cache_stats()}")

    # Create policy
    print("\n[4/8] Creating policy...")
    policy = PulsePolicy(
        task_feature_dim=config.get('task_feature_dim'),
        hidden_dim=config.get('hidden_dim'),
        n_hidden_layers=config.get('n_hidden_layers'),
        n_segments=config.get('n_segments'),
        n_controls=config.get('n_controls'),
        output_scale=config.get('output_scale'),
        activation=config.get('activation')
    ).to(device)

    policy.train()  # CRITICAL: Must be in training mode

    param_count = sum(1 for _ in policy.parameters())
    requires_grad_count = sum(1 for p in policy.parameters() if p.requires_grad)
    print(f"      Policy parameters: {param_count}")
    print(f"      Parameters requiring grad: {requires_grad_count}")
    print(f"      Policy in training mode: {policy.training}")

    # Sample a task
    print("\n[5/8] Sampling task...")
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
    print(f"      Task sampled: alpha={task_params.alpha:.3f}, A={task_params.A:.3f}, omega_c={task_params.omega_c:.3f}")

    # Test loss computation
    print("\n[6/8] Testing loss computation...")
    print("      Computing loss with compute_loss_differentiable()...")

    try:
        loss = env.compute_loss_differentiable(
            policy,
            task_params,
            device,
            use_rk4=config.get('use_rk4_training', True),
            dt=config.get('dt_training', 0.01)
        )

        print(f"      ✓ Loss computed successfully")
        print(f"      Loss value: {loss.item():.6f}")
        print(f"      Loss dtype: {loss.dtype}")
        print(f"      Loss device: {loss.device}")
        print(f"      Loss requires_grad: {loss.requires_grad}")
        print(f"      Loss grad_fn: {loss.grad_fn}")
        print(f"      Loss is_leaf: {loss.is_leaf}")

        # Critical check
        if loss.grad_fn is None and not loss.requires_grad:
            print("\n" + "!"*80)
            print("      ✗ PROBLEM FOUND: Loss has NO gradient connection!")
            print("!"*80)
            print("\n      This is the root cause of your MAML error.")
            print("      The loss tensor is not connected to the policy parameters.")
            print("\n      Possible causes:")
            print("        1. There's a .detach() call in compute_loss_differentiable()")
            print("        2. The quantum simulator breaks gradient flow")
            print("        3. The policy is in eval mode (but we checked it's in train mode)")
            print("        4. The controls are detached somewhere")
            return False
        else:
            print(f"      ✓ Loss HAS gradient connection")

    except Exception as e:
        print(f"      ✗ Error computing loss: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test backward pass
    print("\n[7/8] Testing backward pass...")
    print("      Calling loss.backward()...")

    try:
        policy.zero_grad()
        loss.backward()

        print(f"      ✓ Backward pass completed")

        # Check gradients
        grad_stats = []
        for name, param in policy.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_stats.append((name, grad_norm))

        non_zero_grads = sum(1 for _, norm in grad_stats if norm > 1e-10)

        print(f"      Parameters with gradients: {len(grad_stats)}/{param_count}")
        print(f"      Parameters with non-zero gradients: {non_zero_grads}/{param_count}")

        if non_zero_grads > 0:
            max_grad = max(norm for _, norm in grad_stats)
            print(f"      Max gradient norm: {max_grad:.6e}")
            print(f"      ✓ Gradients are flowing correctly!")
        else:
            print(f"      ⚠ WARNING: All gradients are zero")
            print(f"         This could mean:")
            print(f"           - Loss is constant w.r.t. policy params")
            print(f"           - Vanishing gradient problem")
            return False

    except Exception as e:
        print(f"      ✗ Backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test with MAML inner loop
    print("\n[8/8] Testing MAML inner loop (first-order)...")

    try:
        # Create MAML
        maml = MAML(
            policy=policy,
            inner_lr=config.get('inner_lr'),
            inner_steps=config.get('inner_steps'),
            meta_lr=config.get('meta_lr'),
            first_order=config.get('first_order'),
            device=device
        )

        print(f"      MAML created:")
        print(f"        first_order: {maml.first_order}")
        print(f"        inner_steps: {maml.inner_steps}")
        print(f"        inner_lr: {maml.inner_lr}")

        # Create task data
        task_features = torch.tensor(
            task_params.to_array(),
            dtype=torch.float32,
            device=device
        )

        task_data = {
            'support': {
                'task_features': task_features.unsqueeze(0).repeat(5, 1),
                'task_params': task_params
            },
            'query': {
                'task_features': task_features.unsqueeze(0).repeat(5, 1),
                'task_params': task_params
            }
        }

        # Loss function
        def loss_fn(policy_net, data):
            task_p = data['task_params']
            return env.compute_loss_differentiable(
                policy_net,
                task_p,
                device,
                use_rk4=config.get('use_rk4_training'),
                dt=config.get('dt_training')
            )

        # Test inner loop
        print(f"      Running inner loop adaptation...")
        adapted_policy, inner_losses = maml.inner_loop(task_data, loss_fn)

        print(f"      ✓ Inner loop completed")
        print(f"      Inner losses: {[f'{l:.4f}' for l in inner_losses]}")

        # Test query loss and gradients
        print(f"      Computing query loss...")
        query_loss = loss_fn(adapted_policy, task_data['query'])

        print(f"      Query loss: {query_loss.item():.6f}")
        print(f"      Query loss has grad_fn: {query_loss.grad_fn is not None}")

        if query_loss.grad_fn is None and not query_loss.requires_grad:
            print(f"      ✗ Query loss has no gradients!")
            return False

        print(f"      ✓ MAML inner loop works correctly!")

    except Exception as e:
        print(f"      ✗ MAML inner loop failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    print("\n" + "="*80)
    print(" "*25 + "STARTING DIAGNOSTIC")
    print("="*80)

    # Run diagnostic
    success = diagnostic_test()

    # Results
    print("\n" + "="*80)
    print(" "*30 + "RESULTS")
    print("="*80)

    if success:
        print("\n✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("\nGradient flow is working correctly!")
        print("\nIf MAML training still fails, the issue is likely:")
        print("  1. How the trainer calls the loss function")
        print("  2. The use of higher library vs plain inner_loop")
        print("  3. Numerical instability during training")
        print("\nNext steps:")
        print("  - Run train_meta.py and check the MAML mode printout")
        print("  - Ensure it says 'First-Order (FOMAML) mode'")
        print("  - Check for NaN/Inf losses during training")
    else:
        print("\n✗✗✗ DIAGNOSTIC FAILED ✗✗✗")
        print("\nGradient flow is BROKEN somewhere.")
        print("\nThe diagnostic above shows where the problem is.")
        print("Look for the ✗ marks to see what failed.")

    print("\n" + "="*80 + "\n")

    sys.exit(0 if success else 1)
