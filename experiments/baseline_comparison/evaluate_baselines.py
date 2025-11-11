"""
Evaluate and compare baseline methods for meta-learning.

This script compares:
1. Meta K=0: Meta-initialized policy without adaptation
2. Average Task: Policy trained on the mean task
3. Meta K>0: Meta-initialized policy with K adaptation steps

It computes the gap vs K and gap vs variance for all methods.

Usage:
    python evaluate_baselines.py --meta_policy <path> --average_policy <path>
"""

import torch
import numpy as np
import json
from pathlib import Path
import argparse
import copy
from typing import Dict, List, Tuple
import yaml

from metaqctrl.quantum.noise_adapter import TaskDistribution, NoiseParameters
from metaqctrl.quantum.gates import TargetGates
from metaqctrl.meta_rl.policy import PulsePolicy
from metaqctrl.theory.quantum_environment import create_quantum_environment
from metaqctrl.utils.checkpoint_utils import load_policy_from_checkpoint


def load_average_task_policy(
    checkpoint_path: str,
    config: dict,
    device: torch.device
) -> Tuple[torch.nn.Module, NoiseParameters]:
    """
    Load the average task policy and its mean task parameters.

    Returns:
        policy: Trained policy
        mean_task: Mean task parameters
    """
    print(f"\nLoading average task policy from {checkpoint_path}...")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create policy with same architecture
    policy = PulsePolicy(
        task_feature_dim=config.get('task_feature_dim', 3),
        hidden_dim=config.get('hidden_dim', 128),
        n_hidden_layers=config.get('n_hidden_layers', 2),
        n_segments=config.get('n_segments', 100),
        n_controls=config.get('n_controls', 2),
        output_scale=config.get('output_scale', 1.0),
        activation=config.get('activation', 'tanh')
    )

    policy.load_state_dict(checkpoint['policy_state_dict'])
    policy = policy.to(device)
    policy.eval()

    # Extract mean task
    mean_task_dict = checkpoint['mean_task']
    mean_task = NoiseParameters(
        alpha=mean_task_dict['alpha'],
        A=mean_task_dict['A'],
        omega_c=mean_task_dict['omega_c'],
        model_type=mean_task_dict['model_type']
    )

    print(f"  Policy loaded successfully")
    print(f"  Mean task: α={mean_task.alpha:.4f}, A={mean_task.A:.4f}, ω_c={mean_task.omega_c:.4f}")

    return policy, mean_task


def evaluate_policy_on_tasks(
    policy: torch.nn.Module,
    tasks: List[NoiseParameters],
    env,
    device: torch.device,
    adapt_steps: int = 0,
    inner_lr: float = 0.01,
    config: dict = None
) -> List[float]:
    """
    Evaluate a policy on a list of tasks with optional adaptation.

    Args:
        policy: Policy to evaluate
        tasks: List of tasks
        env: Quantum environment
        device: Device
        adapt_steps: Number of adaptation steps (0 for no adaptation)
        inner_lr: Learning rate for adaptation
        config: Configuration dict

    Returns:
        List of fidelities for each task
    """
    fidelities = []

    for i, task in enumerate(tasks):
        if i % 10 == 0 and i > 0:
            print(f"    Evaluating task {i}/{len(tasks)}...", end='\r')

        task_features = torch.tensor(
            task.to_array(),
            dtype=torch.float32,
            device=device
        )

        if adapt_steps == 0:
            # No adaptation - just evaluate
            policy.eval()
            with torch.no_grad():
                controls = policy(task_features).cpu().numpy()
            fidelity = env.compute_fidelity(controls, task)
        else:
            # Adapt policy for this task
            adapted_policy = copy.deepcopy(policy)
            adapted_policy.train()

            for k in range(adapt_steps):
                loss = env.compute_loss_differentiable(
                    adapted_policy, task, device=device
                )

                # Gradient step
                loss.backward()
                with torch.no_grad():
                    for param in adapted_policy.parameters():
                        if param.grad is not None:
                            param -= inner_lr * param.grad
                            param.grad.zero_()

            # Evaluate adapted policy
            adapted_policy.eval()
            with torch.no_grad():
                controls = adapted_policy(task_features).cpu().numpy()
            fidelity = env.compute_fidelity(controls, task)

        fidelities.append(fidelity)

    print(f"    Completed {len(tasks)} tasks" + " " * 20)
    return fidelities


def evaluate_gap_vs_k(
    meta_policy: torch.nn.Module,
    average_policy: torch.nn.Module,
    tasks: List[NoiseParameters],
    env,
    device: torch.device,
    k_values: List[int],
    config: dict
) -> Dict:
    """
    Evaluate gap vs K for both baselines.

    Returns:
        Dictionary with results for each K value
    """
    print("\n" + "=" * 70)
    print("EVALUATING GAP VS K")
    print("=" * 70)

    results = {
        'k_values': k_values,
        'meta_k0_fidelities': [],
        'meta_k0_mean': [],
        'meta_k0_std': [],
        'average_task_fidelities': [],
        'average_task_mean': [],
        'average_task_std': [],
        'meta_adapted_fidelities': {k: [] for k in k_values},
        'meta_adapted_mean': [],
        'meta_adapted_std': [],
        'gap_vs_meta_k0_mean': [],
        'gap_vs_meta_k0_std': [],
        'gap_vs_average_mean': [],
        'gap_vs_average_std': []
    }

    inner_lr = config.get('inner_lr', 0.01)

    # Baseline 1: Meta K=0 (no adaptation)
    print("\n[1/3] Evaluating Meta K=0 (no adaptation)...")
    meta_k0_fidelities = evaluate_policy_on_tasks(
        meta_policy, tasks, env, device, adapt_steps=0
    )
    results['meta_k0_fidelities'] = meta_k0_fidelities
    results['meta_k0_mean'] = [np.mean(meta_k0_fidelities)]
    results['meta_k0_std'] = [np.std(meta_k0_fidelities) / np.sqrt(len(tasks))]
    print(f"  Mean fidelity: {results['meta_k0_mean'][0]:.6f} ± {results['meta_k0_std'][0]:.6f}")

    # Baseline 2: Average Task Policy (no adaptation)
    print("\n[2/3] Evaluating Average Task Policy...")
    avg_task_fidelities = evaluate_policy_on_tasks(
        average_policy, tasks, env, device, adapt_steps=0
    )
    results['average_task_fidelities'] = avg_task_fidelities
    results['average_task_mean'] = [np.mean(avg_task_fidelities)]
    results['average_task_std'] = [np.std(avg_task_fidelities) / np.sqrt(len(tasks))]
    print(f"  Mean fidelity: {results['average_task_mean'][0]:.6f} ± {results['average_task_std'][0]:.6f}")

    # Meta with K>0 adaptation
    print("\n[3/3] Evaluating Meta with K adaptation steps...")
    for K in k_values:
        print(f"\n  K = {K}:")
        adapted_fidelities = evaluate_policy_on_tasks(
            meta_policy, tasks, env, device, adapt_steps=K, inner_lr=inner_lr, config=config
        )

        results['meta_adapted_fidelities'][K] = adapted_fidelities
        mean_fid = np.mean(adapted_fidelities)
        std_fid = np.std(adapted_fidelities) / np.sqrt(len(tasks))

        results['meta_adapted_mean'].append(mean_fid)
        results['meta_adapted_std'].append(std_fid)

        # Compute gaps
        gap_vs_k0 = mean_fid - results['meta_k0_mean'][0]
        gap_vs_avg = mean_fid - results['average_task_mean'][0]

        # Error propagation for gap
        gap_vs_k0_std = np.sqrt(std_fid**2 + results['meta_k0_std'][0]**2)
        gap_vs_avg_std = np.sqrt(std_fid**2 + results['average_task_std'][0]**2)

        results['gap_vs_meta_k0_mean'].append(gap_vs_k0)
        results['gap_vs_meta_k0_std'].append(gap_vs_k0_std)
        results['gap_vs_average_mean'].append(gap_vs_avg)
        results['gap_vs_average_std'].append(gap_vs_avg_std)

        print(f"    Mean fidelity: {mean_fid:.6f} ± {std_fid:.6f}")
        print(f"    Gap vs Meta K=0: {gap_vs_k0:.6f} ± {gap_vs_k0_std:.6f}")
        print(f"    Gap vs Average: {gap_vs_avg:.6f} ± {gap_vs_avg_std:.6f}")

    return results


def create_task_distributions_with_varying_variance(
    config: dict,
    variance_scales: List[float]
) -> List[Tuple[TaskDistribution, float]]:
    """
    Create task distributions with different variances.

    Args:
        config: Configuration dict
        variance_scales: List of variance scale factors (e.g., [0.1, 0.5, 1.0, 2.0])

    Returns:
        List of (TaskDistribution, expected_variance) tuples
    """
    distributions = []

    # Base ranges from config
    alpha_range = config.get('alpha_range', [0.1, 4.0])
    A_range = config.get('A_range', [10, 1000])
    omega_c_range = config.get('omega_c_range', [1, 100])

    # Centers
    alpha_center = np.mean(alpha_range)
    A_center = np.mean(A_range)
    omega_c_center = np.mean(omega_c_range)

    # Maximum half-widths
    alpha_hw = (alpha_range[1] - alpha_range[0]) / 2
    A_hw = (A_range[1] - A_range[0]) / 2
    omega_c_hw = (omega_c_range[1] - omega_c_range[0]) / 2

    for scale in variance_scales:
        # Scale ranges proportionally
        task_dist = TaskDistribution(
            dist_type='uniform',
            ranges={
                'alpha': (alpha_center - scale * alpha_hw, alpha_center + scale * alpha_hw),
                'A': (A_center - scale * A_hw, A_center + scale * A_hw),
                'omega_c': (omega_c_center - scale * omega_c_hw, omega_c_center + scale * omega_c_hw)
            },
            model_types=config.get('model_types'),
            model_probs=config.get('model_probs')
        )

        # Compute variance
        expected_var = task_dist.compute_variance()
        distributions.append((task_dist, expected_var))

        print(f"  Scale {scale:.2f}: σ² = {expected_var:.6f}")

    return distributions


def evaluate_gap_vs_variance(
    meta_policy: torch.nn.Module,
    average_policy: torch.nn.Module,
    env,
    device: torch.device,
    config: dict,
    variance_scales: List[float],
    K_fixed: int = 5,
    n_tasks_per_variance: int = 50
) -> Dict:
    """
    Evaluate gap vs variance for both baselines.

    Returns:
        Dictionary with results for each variance level
    """
    print("\n" + "=" * 70)
    print("EVALUATING GAP VS VARIANCE")
    print("=" * 70)
    print(f"  Fixed K = {K_fixed}")
    print(f"  Tasks per variance: {n_tasks_per_variance}")

    # Create distributions with varying variance
    print("\nCreating task distributions...")
    task_dists = create_task_distributions_with_varying_variance(config, variance_scales)

    results = {
        'variance_scales': variance_scales,
        'variances': [],
        'meta_k0_mean': [],
        'meta_k0_std': [],
        'average_task_mean': [],
        'average_task_std': [],
        'meta_adapted_mean': [],
        'meta_adapted_std': [],
        'gap_vs_meta_k0_mean': [],
        'gap_vs_meta_k0_std': [],
        'gap_vs_average_mean': [],
        'gap_vs_average_std': [],
        'K_fixed': K_fixed
    }

    inner_lr = config.get('inner_lr', 0.01)

    for idx, (task_dist, expected_var) in enumerate(task_dists):
        print(f"\n[{idx+1}/{len(task_dists)}] Variance scale = {variance_scales[idx]:.2f}")
        print(f"  Expected σ² = {expected_var:.6f}")

        # Sample tasks
        rng = np.random.default_rng(42 + idx)
        tasks = task_dist.sample(n_tasks_per_variance, rng)

        results['variances'].append(expected_var)

        # Evaluate Meta K=0
        print("  Evaluating Meta K=0...")
        meta_k0_fidelities = evaluate_policy_on_tasks(
            meta_policy, tasks, env, device, adapt_steps=0
        )
        mean_k0 = np.mean(meta_k0_fidelities)
        std_k0 = np.std(meta_k0_fidelities) / np.sqrt(len(tasks))
        results['meta_k0_mean'].append(mean_k0)
        results['meta_k0_std'].append(std_k0)

        # Evaluate Average Task
        print("  Evaluating Average Task Policy...")
        avg_fidelities = evaluate_policy_on_tasks(
            average_policy, tasks, env, device, adapt_steps=0
        )
        mean_avg = np.mean(avg_fidelities)
        std_avg = np.std(avg_fidelities) / np.sqrt(len(tasks))
        results['average_task_mean'].append(mean_avg)
        results['average_task_std'].append(std_avg)

        # Evaluate Meta with K_fixed adaptation
        print(f"  Evaluating Meta K={K_fixed}...")
        adapted_fidelities = evaluate_policy_on_tasks(
            meta_policy, tasks, env, device, adapt_steps=K_fixed,
            inner_lr=inner_lr, config=config
        )
        mean_adapted = np.mean(adapted_fidelities)
        std_adapted = np.std(adapted_fidelities) / np.sqrt(len(tasks))
        results['meta_adapted_mean'].append(mean_adapted)
        results['meta_adapted_std'].append(std_adapted)

        # Compute gaps
        gap_vs_k0 = mean_adapted - mean_k0
        gap_vs_avg = mean_adapted - mean_avg

        gap_vs_k0_std = np.sqrt(std_adapted**2 + std_k0**2)
        gap_vs_avg_std = np.sqrt(std_adapted**2 + std_avg**2)

        results['gap_vs_meta_k0_mean'].append(gap_vs_k0)
        results['gap_vs_meta_k0_std'].append(gap_vs_k0_std)
        results['gap_vs_average_mean'].append(gap_vs_avg)
        results['gap_vs_average_std'].append(gap_vs_avg_std)

        print(f"  Results:")
        print(f"    Meta K=0: {mean_k0:.6f} ± {std_k0:.6f}")
        print(f"    Average Task: {mean_avg:.6f} ± {std_avg:.6f}")
        print(f"    Meta K={K_fixed}: {mean_adapted:.6f} ± {std_adapted:.6f}")
        print(f"    Gap vs K=0: {gap_vs_k0:.6f} ± {gap_vs_k0_std:.6f}")
        print(f"    Gap vs Average: {gap_vs_avg:.6f} ± {gap_vs_avg_std:.6f}")

    return results


def main(args):
    """Main evaluation function."""

    print("=" * 70)
    print("BASELINE COMPARISON EVALUATION")
    print("=" * 70)

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Create quantum environment
    print("\nSetting up quantum environment...")
    target_gate_name = config.get('target_gate', 'pauli_x')
    if target_gate_name == 'hadamard':
        U_target = TargetGates.hadamard()
    elif target_gate_name == 'pauli_x':
        U_target = TargetGates.pauli_x()
    else:
        raise ValueError(f"Unknown target gate: {target_gate_name}")

    ket_0 = np.array([1, 0], dtype=complex)
    target_state = np.outer(U_target @ ket_0, (U_target @ ket_0).conj())

    env = create_quantum_environment(config, target_state)

    # Load policies
    print("\nLoading policies...")
    print(f"  Meta policy: {args.meta_policy}")
    meta_policy = load_policy_from_checkpoint(
        args.meta_policy, config, eval_mode=True, verbose=True
    )
    meta_policy = meta_policy.to(device)

    print(f"  Average task policy: {args.average_policy}")
    average_policy, mean_task = load_average_task_policy(
        args.average_policy, config, device
    )

    # Create task distribution for evaluation
    task_dist = TaskDistribution(
        dist_type=config.get('task_dist_type', 'uniform'),
        ranges={
            'alpha': tuple(config.get('alpha_range')),
            'A': tuple(config.get('A_range')),
            'omega_c': tuple(config.get('omega_c_range'))
        },
        model_types=config.get('model_types'),
        model_probs=config.get('model_probs')
    )

    # Sample test tasks
    print(f"\nSampling {args.n_test_tasks} test tasks...")
    rng = np.random.default_rng(args.seed)
    test_tasks = task_dist.sample(args.n_test_tasks, rng)

    # Evaluate Gap vs K
    if args.evaluate_gap_vs_k:
        gap_vs_k_results = evaluate_gap_vs_k(
            meta_policy=meta_policy,
            average_policy=average_policy,
            tasks=test_tasks,
            env=env,
            device=device,
            k_values=args.k_values,
            config=config
        )

        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / 'gap_vs_k_results.json', 'w') as f:
            json.dump(gap_vs_k_results, f, indent=2)

        print(f"\nGap vs K results saved to {output_dir / 'gap_vs_k_results.json'}")

    # Evaluate Gap vs Variance
    if args.evaluate_gap_vs_variance:
        gap_vs_var_results = evaluate_gap_vs_variance(
            meta_policy=meta_policy,
            average_policy=average_policy,
            env=env,
            device=device,
            config=config,
            variance_scales=args.variance_scales,
            K_fixed=args.k_fixed,
            n_tasks_per_variance=args.n_tasks_per_variance
        )

        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / 'gap_vs_variance_results.json', 'w') as f:
            json.dump(gap_vs_var_results, f, indent=2)

        print(f"\nGap vs Variance results saved to {output_dir / 'gap_vs_variance_results.json'}")

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluate baseline comparison for meta-learning'
    )
    parser.add_argument(
        '--meta_policy',
        type=str,
        required=True,
        help='Path to trained meta-policy checkpoint'
    )
    parser.add_argument(
        '--average_policy',
        type=str,
        required=True,
        help='Path to trained average task policy checkpoint'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='../../configs/experiment_config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/baseline_comparison',
        help='Output directory for results'
    )
    parser.add_argument(
        '--n_test_tasks',
        type=int,
        default=100,
        help='Number of test tasks for gap vs K evaluation'
    )
    parser.add_argument(
        '--k_values',
        type=int,
        nargs='+',
        default=[0, 1, 2, 3, 5, 7, 10],
        help='K values for gap vs K evaluation'
    )
    parser.add_argument(
        '--k_fixed',
        type=int,
        default=5,
        help='Fixed K for gap vs variance evaluation'
    )
    parser.add_argument(
        '--variance_scales',
        type=float,
        nargs='+',
        default=[0.1, 0.3, 0.5, 0.7, 1.0],
        help='Variance scale factors'
    )
    parser.add_argument(
        '--n_tasks_per_variance',
        type=int,
        default=50,
        help='Number of tasks per variance level'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--evaluate_gap_vs_k',
        action='store_true',
        default=True,
        help='Evaluate gap vs K'
    )
    parser.add_argument(
        '--evaluate_gap_vs_variance',
        action='store_true',
        default=True,
        help='Evaluate gap vs variance'
    )

    args = parser.parse_args()
    main(args)
