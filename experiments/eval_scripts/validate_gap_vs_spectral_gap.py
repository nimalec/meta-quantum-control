"""
Validation Script 2: Adaptation Gap vs Spectral Gap (Δ)

This script computes the adaptation gap as a function of adaptation steps K
for different values of spectral gap Δ. The spectral gap Δ(θ) is the gap
between the largest two eigenvalues of the Lindblad superoperator L_θ.

Theory predicts that larger spectral gaps should lead to faster convergence
during adaptation (since the system dynamics are less susceptible to noise).

Expected output: Plot showing Gap(K) curves for different Δ values
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Tuple
from typer import Typer
import copy
from tqdm import tqdm

from metaqctrl.quantum.noise_models import TaskDistribution, NoiseParameters
from metaqctrl.meta_rl.policy import PulsePolicy
from metaqctrl.theory.quantum_environment import create_quantum_environment, get_target_state_from_config
from metaqctrl.theory.physics_constants import compute_spectral_gap
from metaqctrl.utils.checkpoint_utils import load_policy_from_checkpoint

app = Typer()


def sample_tasks_by_spectral_gap(
    env,
    config: Dict,
    target_gaps: List[float],
    n_tasks_per_gap: int = 20,
    gap_tolerance: float = 0.05,
    max_samples: int = 1000
) -> Dict[float, List]:
    """
    Sample tasks with specified spectral gap values.

    Args:
        env: QuantumEnvironment
        config: Experiment configuration
        target_gaps: List of target spectral gap values
        n_tasks_per_gap: Number of tasks to collect per gap value
        gap_tolerance: Tolerance for gap matching (relative)
        max_samples: Maximum number of tasks to sample before giving up

    Returns:
        tasks_by_gap: Dict mapping gap value to list of tasks
    """
    print(f"\nSampling tasks by spectral gap...")
    print(f"  Target gaps: {target_gaps}")
    print(f"  Tasks per gap: {n_tasks_per_gap}")
    print(f"  Tolerance: ±{gap_tolerance*100}%")

    # Create task distribution
    task_dist = TaskDistribution(
        dist_type=config.get('task_dist_type', 'uniform'),
        ranges={
            'alpha': tuple(config['alpha_range']),
            'A': tuple(config['A_range']),
            'omega_c': tuple(config['omega_c_range'])
        }
    )

    tasks_by_gap = {gap: [] for gap in target_gaps}

    # Sample tasks and bin them by spectral gap
    sampled_tasks = task_dist.sample(max_samples)

    for task in tqdm(sampled_tasks, desc="Computing spectral gaps"):
        gap = compute_spectral_gap(env, task)

        # Find closest target gap
        for target_gap in target_gaps:
            if len(tasks_by_gap[target_gap]) < n_tasks_per_gap:
                # Check if gap is within tolerance
                if abs(gap - target_gap) / target_gap <= gap_tolerance:
                    tasks_by_gap[target_gap].append((task, gap))
                    break

    # Check if we got enough tasks for each gap
    print("\nTasks collected:")
    for target_gap in target_gaps:
        n_collected = len(tasks_by_gap[target_gap])
        print(f"  Δ ≈ {target_gap:.4f}: {n_collected}/{n_tasks_per_gap} tasks")

        if n_collected < n_tasks_per_gap:
            print(f"    WARNING: Only collected {n_collected} tasks for Δ={target_gap}")

    return tasks_by_gap


def compute_gap_vs_k_for_spectral_gap(
    meta_policy_path: str,
    robust_policy_path: str,
    config: Dict,
    tasks_with_gap: List[Tuple],
    k_values: List[int],
    spectral_gap_value: float
) -> Dict:
    """
    Compute adaptation gap vs K for tasks with a specific spectral gap.

    Args:
        meta_policy_path: Path to meta policy
        robust_policy_path: Path to robust policy
        config: Configuration dict
        tasks_with_gap: List of (task, gap) tuples
        k_values: List of K values to evaluate
        spectral_gap_value: The spectral gap value for these tasks

    Returns:
        results: Dict with gaps for each K
    """
    print(f"\nComputing Gap vs K for Δ = {spectral_gap_value:.4f}...")

    # Load policies
    meta_policy_template = load_policy_from_checkpoint(
        meta_policy_path, config, eval_mode=False, verbose=False
    )
    robust_policy = load_policy_from_checkpoint(
        robust_policy_path, config, eval_mode=True, verbose=False
    )

    # Create environment
    target_state = get_target_state_from_config(config)
    env = create_quantum_environment(config, target_state)

    # Extract tasks
    tasks = [task for task, gap in tasks_with_gap]

    gaps_mean = []
    gaps_std = []

    for K in k_values:
        print(f"  K = {K}...", end=' ')
        fidelities_meta = []
        fidelities_robust = []

        for task in tasks:
            task_features = torch.tensor(
                [task.alpha, task.A, task.omega_c],
                dtype=torch.float32
            )

            # Robust baseline (no adaptation)
            robust_policy.eval()
            with torch.no_grad():
                controls_robust = robust_policy(task_features).detach().numpy()
            fid_robust = env.compute_fidelity(controls_robust, task)
            fidelities_robust.append(fid_robust)

            # Meta policy with K adaptation steps
            adapted_policy = copy.deepcopy(meta_policy_template)
            adapted_policy.train()

            for k in range(K):
                loss = env.compute_loss_differentiable(
                    adapted_policy, task, device=torch.device('cpu')
                )
                loss.backward()
                with torch.no_grad():
                    for param in adapted_policy.parameters():
                        if param.grad is not None:
                            param -= config['inner_lr'] * param.grad
                            param.grad.zero_()

            adapted_policy.eval()
            with torch.no_grad():
                controls_meta = adapted_policy(task_features).detach().numpy()
            fid_meta = env.compute_fidelity(controls_meta, task)
            fidelities_meta.append(fid_meta)

        # Compute gap statistics
        gap_tasks = np.array(fidelities_meta) - np.array(fidelities_robust)
        gap_mean = np.mean(gap_tasks)
        gap_std = np.std(gap_tasks) / np.sqrt(len(tasks))

        gaps_mean.append(gap_mean)
        gaps_std.append(gap_std)

        print(f"Gap = {gap_mean:.4f} ± {gap_std:.4f}")

    return {
        'spectral_gap': spectral_gap_value,
        'k_values': k_values,
        'gaps_mean': gaps_mean,
        'gaps_std': gaps_std,
        'n_tasks': len(tasks)
    }


def run_gap_vs_spectral_gap_experiment(
    meta_policy_path: str,
    robust_policy_path: str,
    config: Dict,
    spectral_gap_values: List[float] = None,
    k_values: List[int] = None,
    n_tasks_per_gap: int = 20,
    output_dir: str = "results/gap_vs_spectral_gap"
) -> Dict:
    """
    Main experiment: Gap vs K for different spectral gaps.

    Args:
        meta_policy_path: Path to trained meta policy
        robust_policy_path: Path to robust baseline policy
        config: Experiment configuration
        spectral_gap_values: List of target spectral gap values
        k_values: List of K values to evaluate
        n_tasks_per_gap: Number of tasks per gap value
        output_dir: Output directory

    Returns:
        all_results: Dict with results for each spectral gap
    """
    print("=" * 80)
    print("EXPERIMENT: Adaptation Gap vs Spectral Gap")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)

    # Default spectral gap values (you may need to adjust based on your system)
    if spectral_gap_values is None:
        spectral_gap_values = [0.5, 1.0, 2.0, 4.0, 8.0]

    # Default K values
    if k_values is None:
        k_values = [1, 2, 3, 5, 7, 10]

    # Create environment for computing spectral gaps
    print("\n[1/3] Setting up environment...")
    target_state = get_target_state_from_config(config)
    env = create_quantum_environment(config, target_state)

    # Sample tasks by spectral gap
    print("\n[2/3] Sampling tasks by spectral gap...")
    tasks_by_gap = sample_tasks_by_spectral_gap(
        env=env,
        config=config,
        target_gaps=spectral_gap_values,
        n_tasks_per_gap=n_tasks_per_gap,
        gap_tolerance=0.1,  # 10% tolerance
        max_samples=2000
    )

    # Compute gap vs K for each spectral gap
    print("\n[3/3] Computing adaptation gaps...")
    all_results = {}

    for spectral_gap in spectral_gap_values:
        tasks_with_gap = tasks_by_gap[spectral_gap]

        if len(tasks_with_gap) == 0:
            print(f"\nWARNING: No tasks found for Δ = {spectral_gap}, skipping...")
            continue

        print(f"\n{'='*60}")
        print(f"Processing Δ = {spectral_gap:.4f} ({len(tasks_with_gap)} tasks)")
        print(f"{'='*60}")

        results = compute_gap_vs_k_for_spectral_gap(
            meta_policy_path=meta_policy_path,
            robust_policy_path=robust_policy_path,
            config=config,
            tasks_with_gap=tasks_with_gap,
            k_values=k_values,
            spectral_gap_value=spectral_gap
        )

        all_results[spectral_gap] = results

    # Save results
    print(f"\nSaving results to {output_dir}...")
    results_path = f"{output_dir}/results.json"

    # Convert dict keys to strings for JSON serialization
    results_to_save = {
        str(k): v for k, v in all_results.items()
    }
    results_to_save['config'] = config
    results_to_save['spectral_gap_values'] = spectral_gap_values
    results_to_save['k_values'] = k_values

    with open(results_path, 'w') as f:
        json.dump(results_to_save, f, indent=2)

    print(f"Results saved to {results_path}")

    return all_results


def plot_gap_vs_spectral_gap(
    results: Dict,
    output_path: str = None
):
    """
    Plot Gap vs K curves for different spectral gaps.

    Args:
        results: Results dict from run_gap_vs_spectral_gap_experiment
        output_path: Path to save figure
    """
    sns.set_style("whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Color map for different spectral gaps
    spectral_gaps = sorted(results.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(spectral_gaps)))

    # Plot 1: Gap vs K for different Δ
    for i, spectral_gap in enumerate(spectral_gaps):
        res = results[spectral_gap]
        k_vals = np.array(res['k_values'])
        gaps = np.array(res['gaps_mean'])
        gaps_err = np.array(res['gaps_std'])

        ax1.errorbar(
            k_vals, gaps, yerr=gaps_err,
            fmt='o-', linewidth=2.5, markersize=8, capsize=4,
            label=f'Δ = {spectral_gap:.2f}',
            color=colors[i]
        )

    ax1.set_xlabel('Adaptation Steps (K)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Adaptation Gap\n(Meta - Robust)', fontsize=14, fontweight='bold')
    ax1.set_title('(a) Adaptation Gap vs K for Different Spectral Gaps',
                  fontsize=15, fontweight='bold')
    ax1.legend(fontsize=11, title='Spectral Gap (Δ)', title_fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=12)

    # Plot 2: Gap at fixed K vs Δ
    K_fixed = 5  # Choose a representative K value
    gaps_at_K = []
    gaps_err_at_K = []
    deltas = []

    for spectral_gap in spectral_gaps:
        res = results[spectral_gap]
        k_vals = res['k_values']

        if K_fixed in k_vals:
            k_idx = k_vals.index(K_fixed)
            gaps_at_K.append(res['gaps_mean'][k_idx])
            gaps_err_at_K.append(res['gaps_std'][k_idx])
            deltas.append(spectral_gap)

    if len(deltas) > 0:
        ax2.errorbar(
            deltas, gaps_at_K, yerr=gaps_err_at_K,
            fmt='o-', linewidth=2.5, markersize=10, capsize=5,
            color='steelblue'
        )

        ax2.set_xlabel(r'Spectral Gap ($\Delta$)', fontsize=14, fontweight='bold')
        ax2.set_ylabel(f'Adaptation Gap at K={K_fixed}', fontsize=14, fontweight='bold')
        ax2.set_title(f'(b) Gap vs Spectral Gap (K={K_fixed})',
                      fontsize=15, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(labelsize=12)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to {output_path}")

    plt.close()


@app.command()
def main(
    meta_path: Path = Path("experiments/train_scripts/checkpoints/maml_best_policy.pt"),
    robust_path: Path = Path("experiments/train_scripts/checkpoints/robust_best_policy.pt"),
    output_dir: Path = Path("results/gap_vs_spectral_gap"),
    n_tasks_per_gap: int = 15
):
    """
    Run adaptation gap vs spectral gap experiment.

    Args:
        meta_path: Path to trained meta policy checkpoint
        robust_path: Path to robust baseline policy checkpoint
        output_dir: Directory to save results and figures
        n_tasks_per_gap: Number of tasks to collect per spectral gap value
    """
    # Configuration matching training setup
    config = {
        'num_qubits': 1,
        'n_controls': 2,
        'n_segments': 100,
        'horizon': 1.0,
        'target_gate': 'pauli_x',
        'hidden_dim': 256,
        'n_hidden_layers': 2,
        'inner_lr': 0.01,
        'alpha_range': [0.5, 2.0],
        'A_range': [0.05, 0.3],
        'omega_c_range': [2.0, 8.0],
        'noise_frequencies': [1.0, 5.0, 10.0]
    }

    # Check if policy paths exist
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta policy not found: {meta_path}")
    if not robust_path.exists():
        raise FileNotFoundError(f"Robust policy not found: {robust_path}")

    print(f"Using meta policy: {meta_path}")
    print(f"Using robust policy: {robust_path}")

    # Spectral gap values to test (adjust based on your system)
    spectral_gap_values = [0.5, 1.0, 2.0, 4.0, 8.0]
    k_values = [1, 2, 3, 5, 7, 10]

    # Run experiment
    results = run_gap_vs_spectral_gap_experiment(
        meta_policy_path=str(meta_path),
        robust_policy_path=str(robust_path),
        config=config,
        spectral_gap_values=spectral_gap_values,
        k_values=k_values,
        n_tasks_per_gap=n_tasks_per_gap,
        output_dir=str(output_dir)
    )

    # Generate plot
    if len(results) > 0:
        plot_path = f"{output_dir}/gap_vs_spectral_gap.pdf"
        plot_gap_vs_spectral_gap(results, output_path=plot_path)

        plot_path_png = f"{output_dir}/gap_vs_spectral_gap.png"
        plot_gap_vs_spectral_gap(results, output_path=plot_path_png)

        print("\n" + "=" * 80)
        print("VALIDATION COMPLETE")
        print("=" * 80)
    else:
        print("\nERROR: No results generated. Check spectral gap sampling.")


if __name__ == "__main__":
    app()
