"""
Minimal Working Example

Tests that all components work together end-to-end.
Run this BEFORE running full experiments.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import numpy as np
import torch
from src.theory.quantum_environment import create_quantum_environment
from src.quantum.noise_models import TaskDistribution, NoiseParameters
from src.quantum.gates import TargetGates
from src.meta_rl.policy import PulsePolicy
from src.theory.physics_constants import (
    compute_spectral_gap,
    estimate_filter_constant,
    compute_control_relevant_variance,
    estimate_all_constants
)


def test_environment():
    """Test 1: Environment creation and evaluation."""
    print("\n" + "="*60)
    print("TEST 1: Environment Creation")
    print("="*60)
    
    # Config
    config = {
        'psd_model': 'one_over_f',
        'horizon': 1.0,
        'n_segments': 20,
        'n_controls': 2
    }
    
    # Target
    U_target = TargetGates.hadamard()
    ket_0 = np.array([1, 0], dtype=complex)
    target_state = np.outer(U_target @ ket_0, (U_target @ ket_0).conj())
    
    # Create environment
    env = create_quantum_environment(config, target_state)
    
    print(f"✓ Environment created")
    print(f"  Hilbert dim: {env.d}")
    print(f"  Controls: {env.n_controls}")
    print(f"  Horizon: {env.T}")
    
    # Test task
    task = NoiseParameters(alpha=1.0, A=0.1, omega_c=5.0)
    
    # Test Lindblad operators
    L_ops = env.get_lindblad_operators(task)
    print(f"✓ Lindblad operators: {len(L_ops)} operators")
    for i, L in enumerate(L_ops):
        print(f"  L{i} norm: {np.linalg.norm(L):.4f}")
    
    # Test simulation
    controls = np.random.randn(20, 2) * 0.1
    fidelity = env.evaluate_controls(controls, task)
    print(f"✓ Simulation works: fidelity = {fidelity:.4f}")
    
    # Test caching
    fidelity2 = env.evaluate_controls(controls, task)
    assert abs(fidelity - fidelity2) < 1e-10, "Caching broken!"
    print(f"✓ Caching works: {env.get_cache_stats()}")
    
    return env


def test_policy_integration(env):
    """Test 2: Policy network integration."""
    print("\n" + "="*60)
    print("TEST 2: Policy Integration")
    print("="*60)
    
    # Create policy
    policy = PulsePolicy(
        task_feature_dim=3,
        hidden_dim=32,
        n_segments=20,
        n_controls=2,
        output_scale=0.5
    )
    
    print(f"✓ Policy created: {policy.count_parameters()} params")
    
    # Test forward pass
    task = NoiseParameters(alpha=1.0, A=0.1, omega_c=5.0)
    fidelity = env.evaluate_policy(policy, task)
    print(f"✓ Policy evaluation: fidelity = {fidelity:.4f}")
    
    # Test loss computation
    device = torch.device('cpu')
    loss = env.compute_loss(policy, task, device)
    print(f"✓ Loss computation: loss = {loss.item():.4f}")
    
    # Test gradient
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in policy.parameters() if p.grad is not None)
    print(f"✓ Gradients work: ||∇|| = {grad_norm:.4f}")
    
    return policy


def test_spectral_gap(env):
    """Test 3: Spectral gap computation."""
    print("\n" + "="*60)
    print("TEST 3: Spectral Gap")
    print("="*60)
    
    task = NoiseParameters(alpha=1.0, A=0.1, omega_c=5.0)
    
    gap = compute_spectral_gap(env, task)
    print(f"✓ Spectral gap: Δ = {gap:.6f}")
    
    # Test on multiple tasks
    tasks = [
        NoiseParameters(alpha=0.5, A=0.05, omega_c=2.0),
        NoiseParameters(alpha=1.5, A=0.2, omega_c=8.0),
        NoiseParameters(alpha=1.0, A=0.1, omega_c=5.0),
    ]
    
    gaps = [compute_spectral_gap(env, t) for t in tasks]
    print(f"✓ Multiple gaps: {[f'{g:.4f}' for g in gaps]}")
    print(f"  Min: {min(gaps):.4f}, Max: {max(gaps):.4f}")
    
    return gaps


def test_constants(env, policy):
    """Test 4: All constants estimation."""
    print("\n" + "="*60)
    print("TEST 4: Constants Estimation")
    print("="*60)
    
    # Create task distribution
    task_dist = TaskDistribution(
        dist_type='uniform',
        ranges={
            'alpha': (0.5, 2.0),
            'A': (0.05, 0.3),
            'omega_c': (2.0, 8.0)
        }
    )
    
    rng = np.random.default_rng(42)
    tasks = task_dist.sample(5, rng)  # Small sample for speed
    
    print("Testing individual constants...")
    
    # Filter constant
    C_filter = estimate_filter_constant(env)
    print(f"✓ C_filter = {C_filter:.6f}")
    
    # Control-relevant variance
    var_result = compute_control_relevant_variance(env, tasks)
    print(f"✓ σ²_S = {var_result['sigma_S_sq']:.8f}")
    print(f"  σ²_out = {var_result['sigma_out_sq']:.8f}")
    print(f"  Ratio = {var_result['ratio_in_to_out']:.2f}")
    
    # Full constants (this is slow, use minimal samples)
    print("\nEstimating all constants (minimal samples)...")
    constants = estimate_all_constants(
        env, policy, tasks,
        device=torch.device('cpu'),
        n_samples_gap=3,
        n_samples_mu=2
    )
    
    print(f"\n✓ All constants estimated successfully")
    
    return constants


def test_training_loop(env, policy):
    """Test 5: Minimal training loop."""
    print("\n" + "="*60)
    print("TEST 5: Training Loop (10 iterations)")
    print("="*60)
    
    from src.meta_rl.maml import MAML
    from src.quantum.noise_models import TaskDistribution
    
    # Create MAML
    maml = MAML(
        policy=policy,
        inner_lr=0.01,
        inner_steps=3,
        meta_lr=0.001,
        first_order=True,  # Faster for testing
        device=torch.device('cpu')
    )
    
    print(f"✓ MAML created")
    
    # Task distribution
    task_dist = TaskDistribution(
        dist_type='uniform',
        ranges={
            'alpha': (0.5, 2.0),
            'A': (0.05, 0.3),
            'omega_c': (2.0, 8.0)
        }
    )
    
    # Loss function
    def loss_fn(policy, data):
        task_params = data['task_params']
        return env.compute_loss(policy, task_params, torch.device('cpu'))
    
    # Training loop
    rng = np.random.default_rng(42)
    
    for iteration in range(10):
        # Sample tasks
        tasks = task_dist.sample(2, rng)  # 2 tasks per batch
        
        # Create task batch
        task_batch = []
        for task in tasks:
            task_batch.append({
                'task_params': task,
                'support': {'task_params': task},
                'query': {'task_params': task}
            })
        
        # Meta-training step
        metrics = maml.meta_train_step(task_batch, loss_fn, use_higher=False)
        
        if iteration % 5 == 0:
            print(f"  Iter {iteration}: loss = {metrics['meta_loss']:.4f}")
    
    print(f"✓ Training loop works")
    
    # Test validation
    val_tasks = task_dist.sample(3, rng)
    val_batch = [
        {'task_params': t, 'support': {'task_params': t}, 'query': {'task_params': t}}
        for t in val_tasks
    ]
    val_metrics = maml.meta_validate(val_batch, loss_fn)
    
    print(f"✓ Validation works")
    print(f"  Pre-adapt: {val_metrics['val_loss_pre_adapt']:.4f}")
    print(f"  Post-adapt: {val_metrics['val_loss_post_adapt']:.4f}")
    print(f"  Gain: {val_metrics['adaptation_gain']:.4f}")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("MINIMAL WORKING EXAMPLE - INTEGRATION TEST")
    print("="*70)
    print("\nThis tests all components work together.")
    print("Run this BEFORE running full experiments.\n")
    
    try:
        # Test 1: Environment
        env = test_environment()
        
        # Test 2: Policy
        policy = test_policy_integration(env)
        
        # Test 3: Spectral gap
        gaps = test_spectral_gap(env)
        
        # Test 4: Constants
        constants = test_constants(env, policy)
        
        # Test 5: Training
        test_training_loop(env, policy)
        
        # Summary
        print("\n" + "="*70)
        print("✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("="*70)
        print("\nYou can now run full experiments:")
        print("  python experiments/train_meta.py --config configs/experiment_config.yaml")
        print("\nEstimated constants summary:")
        print(f"  Δ_min = {constants['Delta_min']:.6f}")
        print(f"  C_filter = {constants['C_filter']:.6f}")
        print(f"  μ_empirical = {constants['mu_empirical']:.6f}")
        print(f"  σ²_S = {constants['sigma_S_sq']:.8f}")
        print(f"  c_quantum = {constants['c_quantum']:.8f}")
        
        return True
        
    except Exception as e:
        print("\n" + "="*70)
        print("❌ TEST FAILED ❌")
        print("="*70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
