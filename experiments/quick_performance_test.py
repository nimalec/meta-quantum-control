"""
Quick Performance Test

Simple script to verify GPU optimizations are working.
Should take <5 minutes to run.
"""

import torch
import numpy as np
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from metaqctrl.theory.quantum_environment import create_quantum_environment
from metaqctrl.quantum.noise_models import TaskDistribution
from metaqctrl.meta_rl.policy import PulsePolicy


def test_simulator_caching(env, policy, task_params_list, device):
    """Test that simulator caching is working."""
    print("\n" + "="*60)
    print("Test 1: Simulator Caching")
    print("="*60)

    # Check initial cache
    initial_stats = env.get_cache_stats()
    print(f"\nInitial cache: {initial_stats['n_cached_torch_simulators']} simulators")

    # Process a few tasks
    for task_params in task_params_list[:5]:
        _ = env.compute_loss_differentiable(policy, task_params, device, dt=0.01)

    # Check cache after
    final_stats = env.get_cache_stats()
    print(f"After processing: {final_stats['n_cached_torch_simulators']} simulators")

    if final_stats['n_cached_torch_simulators'] > 0:
        print("✅ PASS: Simulator caching is working!")
        return True
    else:
        print("❌ FAIL: Simulators not being cached!")
        return False


def test_dt_speedup(env, policy, task_params, device):
    """Test speedup from larger dt."""
    print("\n" + "="*60)
    print("Test 2: Time Step Optimization")
    print("="*60)

    configs = [
        ('dt=0.005 (accurate)', {'dt': 0.005, 'use_rk4': True}),
        ('dt=0.01 (balanced)', {'dt': 0.01, 'use_rk4': True}),
        ('dt=0.02 (fast)', {'dt': 0.02, 'use_rk4': False}),
    ]

    times = {}

    for name, config in configs:
        if device.type == 'cuda':
            torch.cuda.synchronize()

        t0 = time.time()

        for _ in range(10):  # Run 10 iterations
            loss = env.compute_loss_differentiable(
                policy, task_params, device, **config
            )
            loss.backward()
            policy.zero_grad()

        if device.type == 'cuda':
            torch.cuda.synchronize()

        elapsed = time.time() - t0
        times[name] = elapsed
        print(f"  {name}: {elapsed:.3f}s")

    # Calculate speedup
    baseline = times['dt=0.005 (accurate)']
    fast = times['dt=0.02 (fast)']
    speedup = baseline / fast

    print(f"\n  Speedup (fast vs accurate): {speedup:.2f}x")

    if speedup > 2.0:
        print("✅ PASS: Time step optimization working!")
        return True
    else:
        print("⚠️  WARNING: Speedup less than expected")
        return False


def test_batch_timing(env, policy, task_params_list, device):
    """Test timing for a typical batch."""
    print("\n" + "="*60)
    print("Test 3: Batch Processing Speed")
    print("="*60)

    batch_sizes = [4, 8, 16]

    for batch_size in batch_sizes:
        tasks = task_params_list[:batch_size]

        if device.type == 'cuda':
            torch.cuda.synchronize()

        t0 = time.time()

        total_loss = 0.0
        for task_params in tasks:
            loss = env.compute_loss_differentiable(
                policy, task_params, device,
                dt=0.02, use_rk4=False
            )
            total_loss += loss

        avg_loss = total_loss / len(tasks)
        avg_loss.backward()

        if device.type == 'cuda':
            torch.cuda.synchronize()

        elapsed = time.time() - t0
        per_task = elapsed / batch_size

        print(f"  Batch size {batch_size}: {elapsed:.3f}s total, {per_task:.3f}s per task")

    print("\n✅ Batch timing test complete")
    return True


def main():
    print("\n" + "="*70)
    print("Quick GPU Performance Test")
    print("="*70)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    if device.type == 'cpu':
        print("⚠️  WARNING: No GPU detected! Running on CPU.")
        print("   Performance improvements will be less dramatic.\n")

    # Create environment
    print("\nSetting up...")
    config = {
        'target_gate': 'pauli_x',
        'num_qubits': 1,
        'horizon': 1.0,
        'n_segments': 20,
        'psd_model': 'one_over_f',
        'drift_strength': 0.1,
    }

    env = create_quantum_environment(config)

    # Create policy
    policy = PulsePolicy(
        task_feature_dim=3,
        hidden_dim=128,
        n_hidden_layers=2,
        n_segments=20,
        n_controls=2,
        output_scale=2.0,
    ).to(device)

    # Sample tasks
    task_dist = TaskDistribution(
        dist_type='uniform',
        ranges={
            'alpha': (0.5, 2.0),
            'A': (0.001, 0.01),
            'omega_c': (100, 1000)
        }
    )
    rng = np.random.default_rng(42)
    task_params_list = task_dist.sample(16, rng)

    # Run tests
    results = []

    results.append(test_simulator_caching(env, policy, task_params_list, device))
    results.append(test_dt_speedup(env, policy, task_params_list[0], device))
    results.append(test_batch_timing(env, policy, task_params_list, device))

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    passed = sum(results)
    total = len(results)

    print(f"\nTests passed: {passed}/{total}")

    if passed == total:
        print("\n✅ All optimizations working correctly!")
        print("\nYou can now train with GPU-optimized settings:")
        print("  python train_meta.py --config ../../configs/experiment_config_gpu.yaml")
    else:
        print("\n⚠️  Some tests failed. Check the output above.")

    print("\nCache final stats:", env.get_cache_stats())
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
