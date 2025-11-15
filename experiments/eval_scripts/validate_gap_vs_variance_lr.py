"""
Validation Script 3: Adaptation Gap vs K for Different Task Variance and Learning Rates

This script performs a comprehensive study of how the adaptation gap depends on:
1. Number of adaptation steps (K)
2. Task distribution variance (σ²_θ)
3. Inner loop learning rate (η)

Theory predicts:
- Gap should increase with variance (more diverse tasks → bigger advantage)
- Gap should saturate exponentially with K: Gap(K) ∝ (1 - e^(-μηK))
- Learning rate η should modulate convergence speed

Expected output: Multi-panel plots showing Gap(K) for different (variance, lr) combinations
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
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

from metaqctrl.quantum.noise_models import TaskDistribution, NoiseParameters
from metaqctrl.meta_rl.policy import PulsePolicy
from metaqctrl.theory.quantum_environment import create_quantum_environment, get_target_state_from_config
from metaqctrl.utils.checkpoint_utils import load_policy_from_checkpoint

app = Typer()


def exponential_model(K, gap_max, mu_eta):
    """Theoretical model: Gap(K) = gap_max * (1 - exp(-μηK))"""
    return gap_max * (1 - np.exp(-mu_eta * K))


def compute_gap_vs_k(
    meta_policy_path: str,
    robust_policy_path: str,
    config: Dict,
    task_distribution: TaskDistribution,
    k_values: List[int],
    inner_lr: float,
    n_test_tasks: int = 30
) -> Dict:
    """
    Compute adaptation gap vs K for a given task distribution and learning rate.

    Args:
        meta_policy_path: Path to meta policy
        robust_policy_path: Path to robust policy
        config: Configuration dict
        task_distribution: TaskDistribution to sample from
        k_values: List of K values
        inner_lr: Inner loop learning rate
        n_test_tasks: Number of test tasks

    Returns:
        results: Dict with gaps, fitted parameters, etc.
    """
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

    # Sample test tasks
    test_tasks = task_distribution.sample(n_test_tasks)

    # Compute variance of task distribution
    task_variance = task_distribution.compute_variance()

    gaps_mean = []
    gaps_std = []

    for K in k_values:
        fidelities_meta = []
        fidelities_robust = []

        for task in test_tasks:
            task_features = torch.tensor(
                [task.alpha, task.A, task.omega_c],
                dtype=torch.float32
            )

            # Robust baseline
            robust_policy.eval()
            with torch.no_grad():
                controls_robust = robust_policy(task_features).detach().numpy()
            fid_robust = env.compute_fidelity(controls_robust, task)
            fidelities_robust.append(fid_robust)

            # Meta policy with K steps and specified learning rate
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
                            param -= inner_lr * param.grad  # Use specified lr
                            param.grad.zero_()

            adapted_policy.eval()
            with torch.no_grad():
                controls_meta = adapted_policy(task_features).detach().numpy()
            fid_meta = env.compute_fidelity(controls_meta, task)
            fidelities_meta.append(fid_meta)

        # Compute gap statistics
        gap_tasks = np.array(fidelities_meta) - np.array(fidelities_robust)
        gap_mean = np.mean(gap_tasks)
        gap_std = np.std(gap_tasks) / np.sqrt(n_test_tasks)

        gaps_mean.append(gap_mean)
        gaps_std.append(gap_std)

    # Fit exponential model
    k_array = np.array(k_values)
    gaps_array = np.array(gaps_mean)

    try:
        popt, pcov = curve_fit(
            exponential_model,
            k_array,
            gaps_array,
            p0=[gaps_array[-1], 0.1],
            sigma=gaps_std,
            absolute_sigma=True,
            maxfev=10000
        )
        gap_max_fit, mu_eta_fit = popt
        gaps_predicted = exponential_model(k_array, *popt)
        r2 = r2_score(gaps_array, gaps_predicted)
        fit_success = True
    except Exception as e:
        gap_max_fit = mu_eta_fit = r2 = None
        fit_success = False

    return {
        'k_values': k_values,
        'gaps_mean': gaps_mean,
        'gaps_std': gaps_std,
        'task_variance': task_variance,
        'inner_lr': inner_lr,
        'n_test_tasks': n_test_tasks,
        'fit': {
            'success': fit_success,
            'gap_max': float(gap_max_fit) if fit_success else None,
            'mu_eta': float(mu_eta_fit) if fit_success else None,
            'r2': float(r2) if fit_success else None
        }
    }


def run_variance_lr_experiment(
    meta_policy_path: str,
    robust_policy_path: str,
    config: Dict,
    variance_multipliers: List[float] = None,
    learning_rates: List[float] = None,
    k_values: List[int] = None,
    n_test_tasks: int = 30,
    output_dir: str = "results/gap_vs_variance_lr"
) -> Dict:
    """
    Main experiment: Gap vs K for different variance levels and learning rates.

    Args:
        meta_policy_path: Path to trained meta policy
        robust_policy_path: Path to robust baseline policy
        config: Experiment configuration
        variance_multipliers: Multipliers for base task variance
        learning_rates: List of inner learning rates to test
        k_values: List of K values
        n_test_tasks: Number of test tasks per condition
        output_dir: Output directory

    Returns:
        all_results: Nested dict with results for each (variance, lr) pair
    """
    print("=" * 80)
    print("EXPERIMENT: Adaptation Gap vs K, Variance, and Learning Rate")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)

    # Default parameters
    if variance_multipliers is None:
        variance_multipliers = [0.5, 1.0, 1.5]  # Low, medium, high variance

    if learning_rates is None:
        learning_rates = [0.001, 0.005, 0.01, 0.05]  # Different learning rates

    if k_values is None:
        k_values = [1, 2, 3, 5, 7, 10]

    print(f"\nExperiment parameters:")
    print(f"  Variance multipliers: {variance_multipliers}")
    print(f"  Learning rates: {learning_rates}")
    print(f"  K values: {k_values}")
    print(f"  Test tasks per condition: {n_test_tasks}")

    # Base task distribution ranges
    base_alpha = config.get('alpha_range', [0.5, 2.0])
    base_A = config.get('A_range', [0.05, 0.3])
    base_omega = config.get('omega_c_range', [2.0, 8.0])

    alpha_center = np.mean(base_alpha)
    A_center = np.mean(base_A)
    omega_center = np.mean(base_omega)

    # Run experiments for each combination
    all_results = {}
    total_experiments = len(variance_multipliers) * len(learning_rates)
    experiment_idx = 0

    for var_mult in variance_multipliers:
        all_results[var_mult] = {}

        # Scale ranges around center
        alpha_range = [
            alpha_center - (alpha_center - base_alpha[0]) * var_mult,
            alpha_center + (base_alpha[1] - alpha_center) * var_mult
        ]
        A_range = [
            A_center - (A_center - base_A[0]) * var_mult,
            A_center + (base_A[1] - A_center) * var_mult
        ]
        omega_range = [
            omega_center - (omega_center - base_omega[0]) * var_mult,
            omega_center + (base_omega[1] - omega_center) * var_mult
        ]

        # Create task distribution with this variance
        task_dist = TaskDistribution(
            dist_type='uniform',
            ranges={
                'alpha': tuple(alpha_range),
                'A': tuple(A_range),
                'omega_c': tuple(omega_range)
            }
        )

        for lr in learning_rates:
            experiment_idx += 1
            print(f"\n{'='*70}")
            print(f"Experiment {experiment_idx}/{total_experiments}")
            print(f"  Variance multiplier: {var_mult}")
            print(f"  Inner learning rate: {lr}")
            print(f"{'='*70}")

            results = compute_gap_vs_k(
                meta_policy_path=meta_policy_path,
                robust_policy_path=robust_policy_path,
                config=config,
                task_distribution=task_dist,
                k_values=k_values,
                inner_lr=lr,
                n_test_tasks=n_test_tasks
            )

            all_results[var_mult][lr] = results

            # Print summary
            print(f"\n  Results:")
            print(f"    Task variance: {results['task_variance']:.4f}")
            if results['fit']['success']:
                print(f"    Fitted gap_max: {results['fit']['gap_max']:.4f}")
                print(f"    Fitted μη: {results['fit']['mu_eta']:.4f}")
                print(f"    R²: {results['fit']['r2']:.4f}")
            print(f"    Final gap (K={k_values[-1]}): {results['gaps_mean'][-1]:.4f}")

    # Save results
    print(f"\nSaving results to {output_dir}...")
    results_path = f"{output_dir}/results.json"

    # Convert dict for JSON serialization
    results_to_save = {
        'results': {
            str(var_mult): {
                str(lr): res for lr, res in lr_results.items()
            }
            for var_mult, lr_results in all_results.items()
        },
        'config': config,
        'variance_multipliers': variance_multipliers,
        'learning_rates': learning_rates,
        'k_values': k_values
    }

    with open(results_path, 'w') as f:
        json.dump(results_to_save, f, indent=2)

    print(f"Results saved to {results_path}")

    return all_results


def plot_variance_lr_results(
    results: Dict,
    variance_multipliers: List[float],
    learning_rates: List[float],
    output_dir: str
):
    """
    Generate comprehensive plots for variance-lr experiment.

    Creates:
    1. Grid of Gap vs K plots (rows=variance, cols=learning rate)
    2. Heatmap of gap_max vs (variance, lr)
    3. Heatmap of convergence rate (μη) vs (variance, lr)
    """
    sns.set_style("whitegrid")

    # Plot 1: Grid of Gap vs K curves
    n_vars = len(variance_multipliers)
    n_lrs = len(learning_rates)

    fig, axes = plt.subplots(n_vars, n_lrs, figsize=(5*n_lrs, 4*n_vars))
    if n_vars == 1 and n_lrs == 1:
        axes = np.array([[axes]])
    elif n_vars == 1:
        axes = axes.reshape(1, -1)
    elif n_lrs == 1:
        axes = axes.reshape(-1, 1)

    for i, var_mult in enumerate(variance_multipliers):
        for j, lr in enumerate(learning_rates):
            ax = axes[i, j]
            res = results[var_mult][lr]

            k_vals = np.array(res['k_values'])
            gaps = np.array(res['gaps_mean'])
            gaps_err = np.array(res['gaps_std'])

            # Plot empirical data
            ax.errorbar(k_vals, gaps, yerr=gaps_err, fmt='o',
                       markersize=8, capsize=4, label='Empirical',
                       color='steelblue', linewidth=2)

            # Plot fit
            if res['fit']['success']:
                k_fine = np.linspace(0, max(k_vals), 100)
                gap_pred = exponential_model(
                    k_fine,
                    res['fit']['gap_max'],
                    res['fit']['mu_eta']
                )
                ax.plot(k_fine, gap_pred, '--', linewidth=2,
                       label=f"Fit (R²={res['fit']['r2']:.2f})",
                       color='darkred')

            ax.set_xlabel('Adaptation Steps (K)', fontsize=11, fontweight='bold')
            if j == 0:
                ax.set_ylabel('Adaptation Gap', fontsize=11, fontweight='bold')

            ax.set_title(f'Var={var_mult:.1f}x, η={lr:.3f}',
                        fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

    plt.suptitle('Adaptation Gap vs K for Different Variance and Learning Rates',
                 fontsize=16, fontweight='bold', y=1.0)
    plt.tight_layout()

    grid_path = f"{output_dir}/gap_vs_k_grid.pdf"
    plt.savefig(grid_path, dpi=300, bbox_inches='tight')
    print(f"\nGrid plot saved to {grid_path}")

    grid_path_png = f"{output_dir}/gap_vs_k_grid.png"
    plt.savefig(grid_path_png, dpi=300, bbox_inches='tight')
    print(f"Grid plot saved to {grid_path_png}")

    plt.close()

    # Plot 2: Heatmap of gap_max vs (variance, lr)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    gap_max_matrix = np.zeros((n_vars, n_lrs))
    mu_eta_matrix = np.zeros((n_vars, n_lrs))

    for i, var_mult in enumerate(variance_multipliers):
        for j, lr in enumerate(learning_rates):
            res = results[var_mult][lr]
            if res['fit']['success']:
                gap_max_matrix[i, j] = res['fit']['gap_max']
                mu_eta_matrix[i, j] = res['fit']['mu_eta']
            else:
                gap_max_matrix[i, j] = np.nan
                mu_eta_matrix[i, j] = np.nan

    # Heatmap 1: gap_max
    sns.heatmap(gap_max_matrix, annot=True, fmt='.3f', cmap='viridis',
                xticklabels=[f'{lr:.3f}' for lr in learning_rates],
                yticklabels=[f'{v:.1f}x' for v in variance_multipliers],
                ax=ax1, cbar_kws={'label': 'Maximum Gap'})
    ax1.set_xlabel('Inner Learning Rate (η)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Variance Multiplier', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Maximum Adaptation Gap', fontsize=13, fontweight='bold')

    # Heatmap 2: μη (convergence rate)
    sns.heatmap(mu_eta_matrix, annot=True, fmt='.3f', cmap='plasma',
                xticklabels=[f'{lr:.3f}' for lr in learning_rates],
                yticklabels=[f'{v:.1f}x' for v in variance_multipliers],
                ax=ax2, cbar_kws={'label': 'Convergence Rate (μη)'})
    ax2.set_xlabel('Inner Learning Rate (η)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Variance Multiplier', fontsize=12, fontweight='bold')
    ax2.set_title('(b) Adaptation Convergence Rate', fontsize=13, fontweight='bold')

    plt.tight_layout()

    heatmap_path = f"{output_dir}/heatmaps.pdf"
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to {heatmap_path}")

    heatmap_path_png = f"{output_dir}/heatmaps.png"
    plt.savefig(heatmap_path_png, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to {heatmap_path_png}")

    plt.close()


@app.command()
def main(
    meta_path: Path = Path("experiments/train_scripts/checkpoints/maml_best_policy.pt"),
    robust_path: Path = Path("experiments/train_scripts/checkpoints/robust_best_policy.pt"),
    output_dir: Path = Path("results/gap_vs_variance_lr"),
    n_test_tasks: int = 25
):
    """
    Run adaptation gap vs variance and learning rate experiment.

    Args:
        meta_path: Path to trained meta policy checkpoint
        robust_path: Path to robust baseline policy checkpoint
        output_dir: Directory to save results and figures
        n_test_tasks: Number of test tasks per condition
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
        'inner_lr': 0.01,  # Default (will be varied in experiment)
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

    # Experiment parameters
    variance_multipliers = [0.5, 1.0, 1.5]
    learning_rates = [0.001, 0.005, 0.01, 0.05]
    k_values = [1, 2, 3, 5, 7, 10]

    # Run experiment
    results = run_variance_lr_experiment(
        meta_policy_path=str(meta_path),
        robust_policy_path=str(robust_path),
        config=config,
        variance_multipliers=variance_multipliers,
        learning_rates=learning_rates,
        k_values=k_values,
        n_test_tasks=n_test_tasks,
        output_dir=str(output_dir)
    )

    # Generate plots
    plot_variance_lr_results(
        results=results,
        variance_multipliers=variance_multipliers,
        learning_rates=learning_rates,
        output_dir=str(output_dir)
    )

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    app()
