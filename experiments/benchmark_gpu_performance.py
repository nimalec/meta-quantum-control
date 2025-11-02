"""
GPU Performance Benchmark Script

Test the performance improvements from:
1. Simulator caching
2. Optimized time steps (dt)
3. Euler vs RK4 integration
4. Batched task processing

Run this to compare GPU performance before and after optimizations.
"""

import torch
import numpy as np
import time
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from metaqctrl.theory.quantum_environment import create_quantum_environment
from metaqctrl.quantum.noise_models import TaskDistribution
from metaqctrl.meta_rl.policy import PulsePolicy
from metaqctrl.theory.batched_processing import BatchedLossComputation


def benchmark_scenario(
    env,
    policy,
    task_params_list,
    device,
    dt,
    use_rk4,
    name,
    n_repeats=3
):
    """
    Benchmark a specific scenario.

    Args:
        env: QuantumEnvironment
        policy: Policy network
        task_params_list: List of tasks
        device: torch device
        dt: Integration time step
        use_rk4: Whether to use RK4
        name: Scenario name
        n_repeats: Number of timing runs

    Returns:
        avg_time: Average time per iteration
    """
    print(f"\n{'='*60}")
    print(f"Scenario: {name}")
    print(f"  dt={dt}, method={'RK4' if use_rk4 else 'Euler'}, device={device}")
    print(f"{'='*60}")

    times = []
    policy.eval()

    for repeat in range(n_repeats):
        # Clear gradients
        policy.zero_grad()

        # Synchronize GPU if using CUDA
        if device.type == 'cuda':
            torch.cuda.synchronize()

        t0 = time.time()

        # Process all tasks
        total_loss = 0.0
        for task_params in task_params_list:
            loss = env.compute_loss_differentiable(
                policy,
                task_params,
                device,
                use_rk4=use_rk4,
                dt=dt
            )
            total_loss += loss

        # Backward pass
        avg_loss = total_loss / len(task_params_list)
        avg_loss.backward()

        # Synchronize GPU
        if device.type == 'cuda':
            torch.cuda.synchronize()

        elapsed = time.time() - t0
        times.append(elapsed)

        print(f"  Run {repeat+1}/{n_repeats}: {elapsed:.3f}s")

    avg_time = np.mean(times)
    std_time = np.std(times)

    print(f"\n  Average: {avg_time:.3f} ± {std_time:.3f}s")
    print(f"  Tasks per second: {len(task_params_list) / avg_time:.2f}")

    return avg_time


def main():
    """Run GPU performance benchmarks."""

    print("\n" + "="*70)
    print("GPU Performance Benchmark")
    print("="*70)

    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Configuration
    config = {
        'target_gate': 'pauli_x',
        'num_qubits': 1,
        'horizon': 1.0,
        'n_segments': 20,
        'psd_model': 'one_over_f',
        'drift_strength': 0.1,
        'integration_method': 'RK45'
    }

    # Create environment
    print("\nCreating quantum environment...")
    env = create_quantum_environment(config)
    print(f"  Cache stats: {env.get_cache_stats()}")

    # Create policy
    print("\nCreating policy...")
    policy = PulsePolicy(
        task_feature_dim=3,
        hidden_dim=128,
        n_hidden_layers=2,
        n_segments=20,
        n_controls=2,
        output_scale=2.0,
        activation='tanh'
    ).to(device)
    print(f"  Parameters: {policy.count_parameters():,}")

    # Sample tasks
    print("\nSampling tasks...")
    task_dist = TaskDistribution(
        dist_type='uniform',
        ranges={
            'alpha': (0.5, 2.0),
            'A': (0.001, 0.01),
            'omega_c': (100, 1000)
        }
    )
    rng = np.random.default_rng(42)
    n_tasks = 16  # Typical batch size
    task_params_list = task_dist.sample(n_tasks, rng)
    print(f"  Number of tasks: {n_tasks}")

    # Warm-up (populate cache)
    print("\nWarming up (populating simulator cache)...")
    for task_params in task_params_list[:3]:
        _ = env.compute_loss_differentiable(policy, task_params, device, dt=0.01)
    print(f"  Cache stats after warm-up: {env.get_cache_stats()}")

    # Benchmark different scenarios
    results = {}

    # Scenario 1: Original (slow) - dt=0.001, RK4
    if input("\nRun slow scenario (dt=0.001, RK4)? This will take a while. [y/N]: ").lower() == 'y':
        results['slow'] = benchmark_scenario(
            env, policy, task_params_list, device,
            dt=0.001, use_rk4=True,
            name="Slow (dt=0.001, RK4)",
            n_repeats=2
        )

    # Scenario 2: Medium - dt=0.01, RK4
    results['medium_rk4'] = benchmark_scenario(
        env, policy, task_params_list, device,
        dt=0.01, use_rk4=True,
        name="Medium (dt=0.01, RK4)",
        n_repeats=3
    )

    # Scenario 3: Medium - dt=0.01, Euler
    results['medium_euler'] = benchmark_scenario(
        env, policy, task_params_list, device,
        dt=0.01, use_rk4=False,
        name="Medium (dt=0.01, Euler)",
        n_repeats=3
    )

    # Scenario 4: Fast - dt=0.02, Euler
    results['fast'] = benchmark_scenario(
        env, policy, task_params_list, device,
        dt=0.02, use_rk4=False,
        name="Fast (dt=0.02, Euler)",
        n_repeats=3
    )

    # Scenario 5: Very Fast - dt=0.05, Euler
    results['very_fast'] = benchmark_scenario(
        env, policy, task_params_list, device,
        dt=0.05, use_rk4=False,
        name="Very Fast (dt=0.05, Euler)",
        n_repeats=3
    )

    # Summary
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)

    print(f"\n{'Scenario':<30} {'Time (s)':<15} {'Speedup':<10} {'Tasks/s':<10}")
    print("-"*70)

    baseline = results.get('slow', results['medium_rk4'])
    for name, time_val in results.items():
        speedup = baseline / time_val
        tasks_per_sec = n_tasks / time_val
        print(f"{name:<30} {time_val:>10.3f}     {speedup:>7.2f}x    {tasks_per_sec:>7.2f}")

    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)

    print("""
For TRAINING (speed > accuracy):
  - Use dt=0.02, Euler
  - tasks_per_batch=16-32
  - This gives ~{fast_speedup}x speedup

For FINAL EVALUATION (accuracy > speed):
  - Use dt=0.01, RK4
  - Smaller batches if needed
  - More accurate results

Cache stats: {cache_stats}

The simulator caching eliminates the main bottleneck!
Each task now reuses its cached simulator instead of
recreating it every time.
    """.format(
        fast_speedup=baseline / results['fast'],
        cache_stats=env.get_cache_stats()
    ))

    # Test batched processing if available
    print("\n" + "="*70)
    print("Testing Batched Processing")
    print("="*70)

    try:
        batch_computer = BatchedLossComputation(env, device, dt=0.02, use_rk4=False)

        # Time batched processing
        times_batched = []
        for _ in range(3):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.time()

            losses = batch_computer(policy, task_params_list)
            losses.mean().backward()

            if device.type == 'cuda':
                torch.cuda.synchronize()
            times_batched.append(time.time() - t0)

        avg_batched = np.mean(times_batched)
        print(f"\nBatched processing: {avg_batched:.3f}s")
        print(f"Speedup vs sequential: {results['fast'] / avg_batched:.2f}x")

    except Exception as e:
        print(f"\nBatched processing test failed: {e}")

    print("\n" + "="*70)
    print("Benchmark complete!")
    print("="*70)


if __name__ == "__main__":
    main()
