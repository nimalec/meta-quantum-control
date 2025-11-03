"""
Test Gradient Flow in Meta-Learning Setup

This script diagnoses where gradients might be getting blocked.
"""

import torch
import numpy as np
import yaml

# Add parent to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from metaqctrl.meta_rl.policy import PulsePolicy
from metaqctrl.quantum.noise_models import NoiseParameters
from metaqctrl.theory.quantum_environment import create_quantum_environment
from metaqctrl.quantum.gates import TargetGates


def test_basic_gradient_flow():
    """Test 1: Basic policy gradient flow"""
    print("=" * 70)
    print("TEST 1: Basic Policy Gradient Flow")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    policy = PulsePolicy(
        task_feature_dim=3,
        hidden_dim=128,
        n_hidden_layers=2,
        n_segments=20,
        n_controls=2,
        output_scale=2.0
    ).to(device)

    # Create dummy input
    task_features = torch.randn(3, device=device, requires_grad=False)

    # Forward pass
    controls = policy(task_features)

    # Simple loss
    loss = controls.pow(2).sum()

    # Backward
    loss.backward()

    # Check gradients
    has_grad = any(p.grad is not None and p.grad.abs().max() > 1e-10
                   for p in policy.parameters())
    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), float('inf'))

    print(f"✓ Policy has parameters: {policy.count_parameters():,}")
    print(f"✓ Loss: {loss.item():.6f}")
    print(f"✓ Gradients exist: {has_grad}")
    print(f"✓ Gradient norm: {grad_norm:.6e}")

    if not has_grad:
        print("✗ FAILED: No gradients in policy!")
        return False

    print("✓ PASSED: Basic gradient flow works\n")
    return True


def test_environment_gradient_flow(config):
    """Test 2: Gradient flow through quantum environment"""
    print("=" * 70)
    print("TEST 2: Environment Gradient Flow")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create environment
    ket_0 = np.array([1, 0], dtype=complex)
    U_target = TargetGates.pauli_x()
    target_state = np.outer(U_target @ ket_0, (U_target @ ket_0).conj())

    env = create_quantum_environment(config, target_state)

    # Create policy
    policy = PulsePolicy(
        task_feature_dim=3,
        hidden_dim=128,
        n_hidden_layers=2,
        n_segments=20,
        n_controls=2,
        output_scale=2.0
    ).to(device)

    # Create task
    task_params = NoiseParameters(alpha=1.0, A=0.005, omega_c=500)

    # Forward pass through environment
    loss = env.compute_loss_differentiable(
        policy,
        task_params,
        device,
        use_rk4=False,  # Use Euler for speed
        dt=0.02
    )

    print(f"✓ Loss computed: {loss.item():.6f}")
    print(f"✓ Loss requires grad: {loss.requires_grad}")
    print(f"✓ Loss grad_fn: {loss.grad_fn}")

    if loss.grad_fn is None:
        print("✗ FAILED: Loss has no gradient function!")
        print("  This means gradients cannot flow through the loss.")
        return False

    # Backward pass
    loss.backward()

    # Check gradients
    grad_norms = []
    zero_grad_params = 0
    total_params = 0

    for name, param in policy.named_parameters():
        total_params += 1
        if param.grad is not None:
            g_norm = param.grad.abs().max().item()
            grad_norms.append(g_norm)
            if g_norm < 1e-10:
                zero_grad_params += 1
        else:
            zero_grad_params += 1

    overall_grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), float('inf'))

    print(f"✓ Overall gradient norm: {overall_grad_norm:.6e}")
    print(f"✓ Params with zero/no gradients: {zero_grad_params}/{total_params}")

    if overall_grad_norm < 1e-10:
        print("✗ FAILED: Gradients are numerically zero!")
        print("  This means gradients exist but are vanishing.")
        return False

    if zero_grad_params > total_params * 0.5:
        print(f"⚠ WARNING: {zero_grad_params}/{total_params} parameters have zero gradients")

    print("✓ PASSED: Environment gradient flow works\n")
    return True


def test_maml_gradient_flow(config):
    """Test 3: Gradient flow in MAML setup"""
    print("=" * 70)
    print("TEST 3: MAML Gradient Flow (FOMAML)")
    print("=" * 70)

    try:
        import higher
    except ImportError:
        print("✗ FAILED: 'higher' library not installed!")
        print("  Install with: pip install higher")
        return False

    from torch import optim, autograd

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create environment
    ket_0 = np.array([1, 0], dtype=complex)
    U_target = TargetGates.pauli_x()
    target_state = np.outer(U_target @ ket_0, (U_target @ ket_0).conj())

    env = create_quantum_environment(config, target_state)

    # Create policy (meta-parameters)
    policy = PulsePolicy(
        task_feature_dim=3,
        hidden_dim=128,
        n_hidden_layers=2,
        n_segments=20,
        n_controls=2,
        output_scale=2.0
    ).to(device)

    # Create task
    task_params = NoiseParameters(alpha=1.0, A=0.005, omega_c=500)

    # Loss function
    def loss_fn(pol, task_p):
        return env.compute_loss_differentiable(
            pol, task_p, device, use_rk4=False, dt=0.02
        )

    # Simulate FOMAML inner loop
    inner_lr = 0.01
    inner_steps = 3

    inner_opt = optim.SGD(policy.parameters(), lr=inner_lr)

    with higher.innerloop_ctx(
        policy,
        inner_opt,
        copy_initial_weights=True,
        track_higher_grads=False  # FOMAML: first-order only
    ) as (fmodel, diffopt):

        # Inner loop (adaptation)
        print(f"  Running {inner_steps} inner loop steps...")
        for step in range(inner_steps):
            support_loss = loss_fn(fmodel, task_params)
            print(f"    Step {step}: loss = {support_loss.item():.6f}")
            diffopt.step(support_loss)

        # Query loss
        query_loss = loss_fn(fmodel, task_params)
        print(f"  Query loss: {query_loss.item():.6f}")
        print(f"  Query loss requires_grad: {query_loss.requires_grad}")
        print(f"  Query loss grad_fn: {query_loss.grad_fn}")

        # Get parameters
        meta_params = list(policy.parameters())
        adapted_params = list(fmodel.parameters())

        print(f"  Meta params: {len(meta_params)}")
        print(f"  Adapted params: {len(adapted_params)}")

        # Compute gradients w.r.t. adapted parameters
        print("  Computing gradients...")
        adapted_grads = autograd.grad(
            query_loss,
            adapted_params,
            create_graph=False,
            allow_unused=True
        )

        # Check gradients
        non_none_grads = sum(1 for g in adapted_grads if g is not None)
        zero_grads = sum(1 for g in adapted_grads if g is not None and g.abs().max() < 1e-10)

        print(f"  Non-None gradients: {non_none_grads}/{len(adapted_grads)}")
        print(f"  Zero gradients: {zero_grads}/{len(adapted_grads)}")

        if non_none_grads == 0:
            print("✗ FAILED: No gradients computed!")
            return False

        if zero_grads == len(adapted_grads):
            print("✗ FAILED: All gradients are zero!")
            return False

        # Compute gradient norm
        grad_norm = torch.sqrt(sum(
            (g ** 2).sum() for g in adapted_grads if g is not None
        ))

        print(f"  Gradient norm: {grad_norm.item():.6e}")

        if grad_norm.item() < 1e-10:
            print("✗ FAILED: Gradient norm is numerically zero!")
            return False

    print("✓ PASSED: MAML gradient flow works\n")
    return True


def main():
    # Load config
    config_path = Path(__file__).parent.parent / "configs" / "experiment_config_gpu.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("\n" + "=" * 70)
    print("GRADIENT FLOW DIAGNOSTIC TEST")
    print("=" * 70 + "\n")

    results = {}

    # Run tests
    results['basic'] = test_basic_gradient_flow()
    results['environment'] = test_environment_gradient_flow(config)
    results['maml'] = test_maml_gradient_flow(config)

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:20s}: {status}")

    all_passed = all(results.values())
    if all_passed:
        print("\n✓ All tests passed! Gradient flow is working.")
    else:
        print("\n✗ Some tests failed. Check output above for details.")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
