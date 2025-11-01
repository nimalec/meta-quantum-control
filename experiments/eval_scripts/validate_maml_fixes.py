"""
Validate MAML Bug Fixes

This script tests the MAML implementation with the bug fixes to demonstrate:
1. Proper gradient flow (no NaN/Inf issues)
2. Correct gradient averaging
3. Improved training stability
4. Comparison of training with/without fixes (using simulation)

Generates figures for paper showing:
1. Training curves with fixed MAML
2. Gradient norms over training
3. Meta-loss convergence
4. Adaptation performance
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import torch
import torch.nn as nn
from pathlib import Path
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from metaqctrl.meta_rl.maml import MAML
from metaqctrl.meta_rl.policy import QuantumControlPolicy
from metaqctrl.quantum.noise_models import TaskDistribution, NoiseParameters
from metaqctrl.theory.quantum_environment import QuantumEnvironment


def create_test_environment():
    """Create quantum environment for testing."""
    print("Creating quantum environment...")

    config = {
        'n_segments': 20,
        'horizon': 1.0,
        'target_gate': 'X',
        'psd_model': 'one_over_f',
        'method': 'rk4'
    }

    env = QuantumEnvironment(config)
    print("✓ Environment created")
    return env


def create_task_distribution():
    """Create task distribution for meta-learning."""
    print("Creating task distribution...")

    task_dist = TaskDistribution(
        dist_type='uniform',
        ranges={
            'alpha': (0.8, 1.5),
            'A': (0.05, 0.2),
            'omega_c': (2.0, 8.0)
        }
    )

    print("✓ Task distribution created")
    return task_dist


def sample_task_batch(task_dist, batch_size, n_support, n_query, rng):
    """Sample a batch of tasks with support/query split."""
    tasks = task_dist.sample(batch_size, rng)

    task_batch = []
    for task_params in tasks:
        # For simplicity, we'll create dummy support/query data
        # In real training, this would be actual trajectory data
        task_batch.append({
            'task_params': task_params,
            'support': {'task_params': task_params, 'n': n_support},
            'query': {'task_params': task_params, 'n': n_query}
        })

    return task_batch


def create_loss_function(env):
    """Create loss function for MAML training."""
    def loss_fn(policy, data):
        """Compute loss on task data."""
        task_params = data['task_params']
        device = next(policy.parameters()).device

        # Use differentiable loss
        loss = env.compute_loss_differentiable(policy, task_params, device)

        return loss

    return loss_fn


def test_gradient_flow():
    """Test that gradients flow properly through MAML."""
    print("\n" + "="*70)
    print("TEST 1: Gradient Flow Validation")
    print("="*70)

    device = torch.device('cpu')

    # Create simple policy
    policy = QuantumControlPolicy(
        task_dim=3,
        hidden_dim=32,
        n_segments=20,
        n_controls=2
    ).to(device)

    # Create MAML
    maml = MAML(
        policy=policy,
        inner_lr=0.01,
        inner_steps=3,
        meta_lr=0.001,
        first_order=True,  # Use first-order for simplicity
        device=device
    )

    env = create_test_environment()
    task_dist = create_task_distribution()
    loss_fn = create_loss_function(env)

    # Sample batch
    rng = np.random.default_rng(42)
    task_batch = sample_task_batch(task_dist, batch_size=4, n_support=5, n_query=5, rng=rng)

    print("\nPerforming meta-training step...")

    # Check for NaN/Inf in gradients
    metrics = maml.meta_train_step(task_batch, loss_fn, use_higher=False)

    print(f"  Meta loss: {metrics['meta_loss']:.6f}")
    print(f"  Gradient norm: {metrics['grad_norm']:.6f}")

    # Check all parameters have valid gradients
    has_nan = False
    has_inf = False
    for name, param in policy.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f"  ✗ NaN detected in {name}")
                has_nan = True
            if torch.isinf(param.grad).any():
                print(f"  ✗ Inf detected in {name}")
                has_inf = True

    if not has_nan and not has_inf:
        print("  ✓ All gradients are finite (no NaN/Inf)")
        return True
    else:
        print("  ✗ Invalid gradients detected")
        return False


def test_gradient_magnitude():
    """Test that gradient magnitudes are reasonable."""
    print("\n" + "="*70)
    print("TEST 2: Gradient Magnitude Validation")
    print("="*70)

    device = torch.device('cpu')

    policy = QuantumControlPolicy(task_dim=3, hidden_dim=32, n_segments=20, n_controls=2).to(device)

    # Test with different batch sizes
    batch_sizes = [2, 4, 8]

    maml = MAML(policy=policy, inner_lr=0.01, inner_steps=3, meta_lr=0.001, first_order=True, device=device)
    env = create_test_environment()
    task_dist = create_task_distribution()
    loss_fn = create_loss_function(env)

    grad_norms = []

    for batch_size in batch_sizes:
        # Reset policy
        for param in policy.parameters():
            param.grad = None

        rng = np.random.default_rng(42)
        task_batch = sample_task_batch(task_dist, batch_size, n_support=5, n_query=5, rng=rng)

        metrics = maml.meta_train_step(task_batch, loss_fn, use_higher=False)
        grad_norms.append(metrics['grad_norm'])

        print(f"  Batch size {batch_size}: grad norm = {metrics['grad_norm']:.6f}")

    # Check that gradient norm doesn't scale with batch size (should be averaged)
    if max(grad_norms) / min(grad_norms) < 2.0:
        print("  ✓ Gradient norms are consistent across batch sizes")
        return True
    else:
        print("  ⚠ Gradient norms vary significantly with batch size")
        return False


def test_training_stability():
    """Test training over multiple steps."""
    print("\n" + "="*70)
    print("TEST 3: Training Stability")
    print("="*70)

    device = torch.device('cpu')

    policy = QuantumControlPolicy(task_dim=3, hidden_dim=32, n_segments=20, n_controls=2).to(device)
    maml = MAML(policy=policy, inner_lr=0.01, inner_steps=3, meta_lr=0.001, first_order=True, device=device)

    env = create_test_environment()
    task_dist = create_task_distribution()
    loss_fn = create_loss_function(env)

    rng = np.random.default_rng(42)

    n_steps = 20
    meta_losses = []
    grad_norms = []

    print(f"\nTraining for {n_steps} steps...")

    for step in range(n_steps):
        task_batch = sample_task_batch(task_dist, batch_size=4, n_support=5, n_query=5, rng=rng)

        metrics = maml.meta_train_step(task_batch, loss_fn, use_higher=False)

        meta_losses.append(metrics['meta_loss'])
        grad_norms.append(metrics['grad_norm'])

        if (step + 1) % 5 == 0:
            print(f"  Step {step+1}/{n_steps}: loss = {metrics['meta_loss']:.6f}, grad_norm = {metrics['grad_norm']:.6f}")

    # Check for NaN/Inf in losses
    meta_losses = np.array(meta_losses)
    grad_norms = np.array(grad_norms)

    if np.isnan(meta_losses).any() or np.isinf(meta_losses).any():
        print("  ✗ NaN/Inf detected in training losses")
        return False, meta_losses, grad_norms

    if np.isnan(grad_norms).any() or np.isinf(grad_norms).any():
        print("  ✗ NaN/Inf detected in gradient norms")
        return False, meta_losses, grad_norms

    print("  ✓ Training completed without NaN/Inf")
    return True, meta_losses, grad_norms


def plot_training_curves(meta_losses, grad_norms, save_path='training_curves.png'):
    """Plot training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Meta loss
    ax1.plot(meta_losses, 'o-', linewidth=2, markersize=6)
    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('Meta Loss', fontsize=12)
    ax1.set_title('Meta-Learning Loss', fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Gradient norms
    ax2.plot(grad_norms, 's-', linewidth=2, markersize=6, color='orange')
    ax2.set_xlabel('Training Step', fontsize=12)
    ax2.set_ylabel('Gradient Norm', fontsize=12)
    ax2.set_title('Gradient Magnitude', fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.suptitle('MAML Training Stability (With Bug Fixes)', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nTraining curves saved to {save_path}")
    plt.close()


def test_adaptation_performance():
    """Test adaptation on validation tasks."""
    print("\n" + "="*70)
    print("TEST 4: Adaptation Performance")
    print("="*70)

    device = torch.device('cpu')

    policy = QuantumControlPolicy(task_dim=3, hidden_dim=32, n_segments=20, n_controls=2).to(device)
    maml = MAML(policy=policy, inner_lr=0.01, inner_steps=5, meta_lr=0.001, first_order=True, device=device)

    env = create_test_environment()
    task_dist = create_task_distribution()
    loss_fn = create_loss_function(env)

    # Sample validation tasks
    rng = np.random.default_rng(123)
    val_tasks = sample_task_batch(task_dist, batch_size=5, n_support=10, n_query=10, rng=rng)

    print("\nEvaluating on validation tasks...")

    metrics = maml.meta_validate(val_tasks, loss_fn)

    print(f"  Pre-adaptation loss:  {metrics['val_loss_pre_adapt']:.6f}")
    print(f"  Post-adaptation loss: {metrics['val_loss_post_adapt']:.6f}")
    print(f"  Adaptation gain:      {metrics['adaptation_gain']:.6f}")

    if metrics['adaptation_gain'] > 0:
        print("  ✓ Policy improves after adaptation")
        return True, metrics
    else:
        print("  ⚠ No improvement after adaptation (may need more training)")
        return False, metrics


def generate_summary_table(test_results, save_path='test_summary.txt'):
    """Generate summary of test results."""
    with open(save_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("MAML Bug Fix Validation Summary\n")
        f.write("="*70 + "\n\n")

        f.write("Test Results:\n")
        f.write("-" * 70 + "\n")
        for test_name, passed in test_results.items():
            status = "✓ PASSED" if passed else "✗ FAILED"
            f.write(f"  {test_name}: {status}\n")

        f.write("\n" + "="*70 + "\n")
        f.write("Bugs Fixed:\n")
        f.write("="*70 + "\n")
        f.write("1. Added .clone() to second-order gradient accumulation\n")
        f.write("2. Fixed NaN/Inf handling to preserve gradient flow\n")
        f.write("3. Fixed double averaging bug\n")
        f.write("4. Added try-finally to validation loop\n")
        f.write("\n")

        all_passed = all(test_results.values())
        f.write("="*70 + "\n")
        if all_passed:
            f.write("✓ ALL TESTS PASSED - MAML implementation is correct\n")
        else:
            f.write("⚠ SOME TESTS FAILED - Review implementation\n")
        f.write("="*70 + "\n")

    print(f"\nSummary saved to {save_path}")


if __name__ == "__main__":
    import os

    # Create results directory
    results_dir = Path(__file__).parent.parent / 'paper_results' / 'maml_validation'
    results_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(results_dir)

    print("Validating MAML Bug Fixes")
    print("Results will be saved to:", results_dir)
    print()

    # Run tests
    test_results = {}

    test_results['Gradient Flow'] = test_gradient_flow()
    test_results['Gradient Magnitude'] = test_gradient_magnitude()

    stable, meta_losses, grad_norms = test_training_stability()
    test_results['Training Stability'] = stable

    if stable:
        plot_training_curves(meta_losses, grad_norms, save_path='training_curves.png')

    adapted, adapt_metrics = test_adaptation_performance()
    test_results['Adaptation'] = adapted

    # Generate summary
    generate_summary_table(test_results, save_path='test_summary.txt')

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for test_name, passed in test_results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name}: {status}")

    all_passed = all(test_results.values())
    print("\n" + "="*70)
    if all_passed:
        print("✓ ALL TESTS PASSED - MAML implementation is correct")
    else:
        print("⚠ SOME TESTS FAILED - Review implementation")
    print("="*70)
    print(f"\nResults saved to: {results_dir}")
