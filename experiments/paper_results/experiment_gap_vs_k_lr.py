"""
Experiment: Optimality Gap vs Adaptation Steps K for Different Learning Rates

This script generates plots showing how the adaptation gap varies with K (x-axis)
for different inner learning rates, validating:
    Gap(P, K, η) ∝ (1 - e^(-μηK))

Expected result: Faster convergence (steeper curves) for higher learning rates
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
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from tqdm import tqdm
import copy

from metaqctrl.quantum.lindblad import LindbladSimulator
from metaqctrl.quantum.noise_models import TaskDistribution, NoiseParameters, PSDToLindblad
from metaqctrl.quantum.gates import state_fidelity
from metaqctrl.meta_rl.policy import PulsePolicy
from metaqctrl.meta_rl.maml import MAML
from metaqctrl.baselines.robust_control import RobustPolicy, GRAPEOptimizer
from metaqctrl.theory.quantum_environment import create_quantum_environment, get_target_state_from_config
from metaqctrl.theory.optimality_gap import OptimalityGapComputer, GapConstants
from metaqctrl.theory.physics_constants import estimate_all_constants
from metaqctrl.utils.checkpoint_utils import load_policy_from_checkpoint


def exponential_model(K, gap_max, mu_eta):
    """Theoretical model: Gap(K) = gap_max * (1 - exp(-μηK))"""
    return gap_max * (1 - np.exp(-mu_eta * K))


def adapt_and_evaluate(
    meta_policy_template,
    robust_policy,
    task,
    env,
    K: int,
    inner_lr: float,
    task_features: torch.Tensor,
    device: torch.device
) -> Tuple[float, float]:
    """
    Adapt meta policy for K steps and evaluate both policies.

    Returns:
        (fidelity_meta, fidelity_robust): Tuple of fidelities
    """
    # Evaluate robust policy (no adaptation)
    robust_policy.eval()
    with torch.no_grad():
        controls_robust = robust_policy(task_features).detach().numpy()
    fid_robust = env.compute_fidelity(controls_robust, task)

    # Adapt meta policy
    adapted_policy = copy.deepcopy(meta_policy_template)
    adapted_policy.train()

    for k in range(K):
        # Compute loss with gradients
        loss = env.compute_loss_differentiable(
            adapted_policy, task, device=device
        )

        # Manual gradient descent step
        loss.backward()
        with torch.no_grad():
            for param in adapted_policy.parameters():
                if param.grad is not None:
                    param -= inner_lr * param.grad
                    param.grad.zero_()

    # Evaluate adapted policy
    adapted_policy.eval()
    with torch.no_grad():
        controls_meta = adapted_policy(task_features).detach().numpy()
    fid_meta = env.compute_fidelity(controls_meta, task)

    return fid_meta, fid_robust


def run_gap_vs_k_lr_experiment(
    meta_policy_path: str,
    robust_policy_path: str,
    config: Dict,
    k_values: List[int] = [1, 2, 3, 5, 7, 10, 15, 20],
    learning_rates: List[float] = [0.001, 0.005, 0.01, 0.05, 0.1],
    n_test_tasks: int = 100,
    output_dir: str = "results/gap_vs_k_lr"
) -> Dict:
    """
    Main experiment: measure optimality gap as function of K for different learning rates

    Args:
        meta_policy_path: Path to trained meta policy checkpoint
        robust_policy_path: Path to trained robust policy checkpoint
        config: Experiment configuration dictionary
        k_values: List of adaptation step values to test
        learning_rates: List of inner learning rates to test
        n_test_tasks: Number of test tasks to sample
        output_dir: Directory to save results

    Returns:
        results: Dictionary containing gaps, fits, and validation metrics
    """
    print("=" * 80)
    print("EXPERIMENT: Optimality Gap vs K for Different Learning Rates")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load trained policies
    print("\n[1/5] Loading trained policies...")
    print("Loading meta policy...")
    meta_policy_template = load_policy_from_checkpoint(
        meta_policy_path, config, eval_mode=False, verbose=True
    )

    print("Loading robust policy...")
    robust_policy = load_policy_from_checkpoint(
        robust_policy_path, config, eval_mode=True, verbose=True
    )

    # Create quantum environment
    print("\n[2/5] Creating quantum environment...")
    target_state = get_target_state_from_config(config)
    env = create_quantum_environment(config, target_state)
    print(f"Environment created: {env.get_cache_stats()}")

    # Sample test tasks
    print(f"\n[3/5] Sampling {n_test_tasks} test tasks...")
    task_dist = TaskDistribution(
        dist_type=config.get('task_dist_type', 'uniform'),
        ranges={
            'alpha': tuple(config['alpha_range']),
            'A': tuple(config['A_range']),
            'omega_c': tuple(config['omega_c_range'])
        }
    )
    test_tasks = task_dist.sample(n_test_tasks)
    print(f"Task distribution variance: {task_dist.compute_variance():.4f}")

    # Measure gap for each (K, lr) pair
    print(f"\n[4/5] Computing gaps for K = {k_values} and lr = {learning_rates}...")

    results = {
        'k_values': k_values,
        'learning_rates': learning_rates,
        'gaps': {},  # gaps[lr] = {'mean': [...], 'std': [...]}
        'fits': {},  # fits[lr] = {'gap_max': ..., 'mu_eta': ..., 'r2': ...}
        'fidelities': {}  # fidelities[lr] = {'meta': [...], 'robust': [...]}
    }

    # Robust baseline (computed once, independent of K and lr)
    print("\nComputing robust baseline...")
    fidelities_robust_baseline = []
    for i, task in enumerate(tqdm(test_tasks, desc="Robust baseline")):
        task_features = torch.tensor(
            [task.alpha, task.A, task.omega_c],
            dtype=torch.float32,
            device=device
        )
        with torch.no_grad():
            controls_robust = robust_policy(task_features).detach().numpy()
        fid_robust = env.compute_fidelity(controls_robust, task)
        fidelities_robust_baseline.append(fid_robust)

    robust_mean = np.mean(fidelities_robust_baseline)
    robust_std = np.std(fidelities_robust_baseline)
    print(f"Robust baseline: {robust_mean:.4f} ± {robust_std:.4f}")

    # For each learning rate
    for lr in learning_rates:
        print(f"\n{'='*60}")
        print(f"Learning Rate: {lr}")
        print(f"{'='*60}")

        gaps_mean = []
        gaps_std = []
        fidelities_meta_all = []

        # For each K value
        for K in tqdm(k_values, desc=f"lr={lr}"):
            fidelities_meta = []

            for i, task in enumerate(test_tasks):
                task_features = torch.tensor(
                    [task.alpha, task.A, task.omega_c],
                    dtype=torch.float32,
                    device=device
                )

                # Adapt and evaluate
                fid_meta, _ = adapt_and_evaluate(
                    meta_policy_template,
                    robust_policy,
                    task,
                    env,
                    K,
                    lr,
                    task_features,
                    device
                )
                fidelities_meta.append(fid_meta)

            # Compute gap for this (K, lr)
            gap_tasks = np.array(fidelities_meta) - np.array(fidelities_robust_baseline)
            gap_mean = np.mean(gap_tasks)
            gap_std = np.std(gap_tasks) / np.sqrt(n_test_tasks)  # Standard error

            gaps_mean.append(gap_mean)
            gaps_std.append(gap_std)
            fidelities_meta_all.append(fidelities_meta)

        # Store results for this lr
        results['gaps'][lr] = {
            'mean': gaps_mean,
            'std': gaps_std
        }
        results['fidelities'][lr] = {
            'meta': fidelities_meta_all,
            'robust': fidelities_robust_baseline
        }

        # Fit theoretical model
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

            # Compute R²
            gaps_predicted = exponential_model(k_array, *popt)
            r2 = r2_score(gaps_array, gaps_predicted)

            results['fits'][lr] = {
                'gap_max': float(gap_max_fit),
                'mu_eta': float(mu_eta_fit),
                'r2': float(r2),
                'success': True
            }

            print(f"  Fitted: gap_max={gap_max_fit:.4f}, μη={mu_eta_fit:.4f}, R²={r2:.4f}")

        except Exception as e:
            print(f"  Fitting failed: {e}")
            results['fits'][lr] = {'success': False}

    # Save results
    print(f"\n[5/5] Saving results to {output_dir}...")
    results['config'] = config
    results['n_test_tasks'] = n_test_tasks
    results['robust_baseline'] = {
        'mean': float(robust_mean),
        'std': float(robust_std)
    }

    # Convert numpy types for JSON serialization
    results_serializable = json.loads(
        json.dumps(results, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    )

    with open(f"{output_dir}/results.json", 'w') as f:
        json.dump(results_serializable, f, indent=2)

    print(f"\nFinal cache stats: {env.get_cache_stats()}")
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)

    return results


def plot_gap_vs_k_lr(results: Dict, output_path: str = "results/gap_vs_k_lr/figure.pdf"):
    """
    Generate publication-quality figure with multiple learning rates
    """
    sns.set_style("whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    k_values = np.array(results['k_values'])
    learning_rates = results['learning_rates']

    # Color palette
    colors = sns.color_palette("viridis", len(learning_rates))

    # Plot 1: Gap vs K for different learning rates
    for i, lr in enumerate(learning_rates):
        gaps_mean = np.array(results['gaps'][lr]['mean'])
        gaps_std = np.array(results['gaps'][lr]['std'])

        ax1.errorbar(
            k_values, gaps_mean, yerr=gaps_std,
            fmt='o-', markersize=8, capsize=4, capthick=1.5,
            label=f'η = {lr}', color=colors[i], linewidth=2, alpha=0.8
        )

        # Plot theoretical fit if available
        if results['fits'][lr]['success']:
            k_fine = np.linspace(0, max(k_values), 100)
            gap_pred = exponential_model(
                k_fine,
                results['fits'][lr]['gap_max'],
                results['fits'][lr]['mu_eta']
            )
            ax1.plot(
                k_fine, gap_pred, '--',
                color=colors[i], linewidth=1.5, alpha=0.6
            )

    ax1.set_xlabel('Adaptation Steps (K)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Optimality Gap (Meta - Robust)', fontsize=14, fontweight='bold')
    ax1.set_title('Gap vs K for Different Learning Rates', fontsize=16, fontweight='bold')
    ax1.legend(fontsize=11, loc='lower right', title='Inner LR (η)')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=12)

    # Plot 2: Convergence rate (μη) vs learning rate
    lrs_successful = []
    mu_etas = []
    r2s = []

    for lr in learning_rates:
        if results['fits'][lr]['success']:
            lrs_successful.append(lr)
            mu_etas.append(results['fits'][lr]['mu_eta'])
            r2s.append(results['fits'][lr]['r2'])

    if len(lrs_successful) > 0:
        ax2_twin = ax2.twinx()

        # Plot μη
        line1 = ax2.plot(
            lrs_successful, mu_etas, 'o-',
            markersize=10, linewidth=2.5, color='steelblue',
            label='Convergence Rate (μη)'
        )
        ax2.set_xlabel('Inner Learning Rate (η)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('μη (Convergence Rate)', fontsize=14, fontweight='bold', color='steelblue')
        ax2.tick_params(axis='y', labelcolor='steelblue', labelsize=12)
        ax2.set_xscale('log')
        ax2.grid(True, alpha=0.3)

        # Plot R²
        line2 = ax2_twin.plot(
            lrs_successful, r2s, 's-',
            markersize=10, linewidth=2.5, color='darkred',
            label='Fit Quality (R²)'
        )
        ax2_twin.set_ylabel('R² (Fit Quality)', fontsize=14, fontweight='bold', color='darkred')
        ax2_twin.tick_params(axis='y', labelcolor='darkred', labelsize=12)
        ax2_twin.set_ylim([0, 1.05])

        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, fontsize=11, loc='upper left')

        ax2.set_title('Theory Validation: Convergence Rate vs LR', fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {output_path}")

    # Also save as PNG
    png_path = output_path.replace('.pdf', '.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {png_path}")

    plt.close()


def plot_summary_table(results: Dict, output_path: str = "results/gap_vs_k_lr/summary_table.txt"):
    """Generate a summary table of results"""
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("SUMMARY: Gap vs K for Different Learning Rates\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Number of test tasks: {results['n_test_tasks']}\n")
        f.write(f"K values tested: {results['k_values']}\n")
        f.write(f"Learning rates tested: {results['learning_rates']}\n\n")

        f.write(f"Robust baseline fidelity: {results['robust_baseline']['mean']:.4f} ± "
                f"{results['robust_baseline']['std']:.4f}\n\n")

        f.write("-" * 80 + "\n")
        f.write(f"{'LR':<10} {'Gap_max':<12} {'μη':<12} {'R²':<10} {'Status':<10}\n")
        f.write("-" * 80 + "\n")

        for lr in results['learning_rates']:
            fit = results['fits'][lr]
            if fit['success']:
                f.write(f"{lr:<10.4f} {fit['gap_max']:<12.4f} {fit['mu_eta']:<12.4f} "
                       f"{fit['r2']:<10.4f} Success\n")
            else:
                f.write(f"{lr:<10.4f} {'N/A':<12} {'N/A':<12} {'N/A':<10} Failed\n")

        f.write("-" * 80 + "\n\n")

        # Final gaps for each LR at max K
        f.write("Final gaps at K = {} (Meta fidelity advantage):\n".format(max(results['k_values'])))
        for lr in results['learning_rates']:
            final_gap = results['gaps'][lr]['mean'][-1]
            final_std = results['gaps'][lr]['std'][-1]
            f.write(f"  η = {lr:.4f}: {final_gap:.4f} ± {final_std:.4f}\n")

    print(f"Summary table saved to {output_path}")


if __name__ == "__main__":
    # Configuration
    config = {
        'num_qubits': 1,
        'n_controls': 2,
        'n_segments': 20,
        'horizon': 1.0,
        'target_gate': 'hadamard',
        'hidden_dim': 128,
        'n_hidden_layers': 2,
        'alpha_range': [0.5, 2.0],
        'A_range': [0.05, 0.3],
        'omega_c_range': [2.0, 8.0],
        'noise_frequencies': [1.0, 5.0, 10.0],
        'task_dist_type': 'uniform'
    }

    # Paths to trained models
    script_dir = Path(__file__).parent
    possible_meta_paths = [
        "checkpoints/maml_best.pt",
        "../checkpoints/maml_20251027_161519_best_policy.pt",
        script_dir.parent / "checkpoints" / "maml_best.pt",
        script_dir / "checkpoints" / "maml_best.pt",
    ]
    possible_robust_paths = [
        "checkpoints/robust_best.pt",
        "../checkpoints/robust_minimax_20251027_162238_best_policy.pt",
        script_dir.parent / "checkpoints" / "robust_best.pt",
        script_dir / "checkpoints" / "robust_best.pt",
    ]

    # Find existing paths
    meta_path = None
    for p in possible_meta_paths:
        if os.path.exists(p):
            meta_path = str(p)
            break

    robust_path = None
    for p in possible_robust_paths:
        if os.path.exists(p):
            robust_path = str(p)
            break

    # Check if models exist
    if meta_path is None:
        print(f"ERROR: Meta-policy not found. Searched in:")
        for p in possible_meta_paths:
            print(f"  - {p}")
        print("\nPlease train the meta-policy first using:")
        print("  python experiments/train_meta.py")
        sys.exit(1)

    if robust_path is None:
        print(f"ERROR: Robust policy not found. Searched in:")
        for p in possible_robust_paths:
            print(f"  - {p}")
        print("\nPlease train the robust policy first using:")
        print("  python experiments/train_robust.py")
        sys.exit(1)

    print(f"Using meta policy: {meta_path}")
    print(f"Using robust policy: {robust_path}")

    # Run experiment
    results = run_gap_vs_k_lr_experiment(
        meta_policy_path=meta_path,
        robust_policy_path=robust_path,
        config=config,
        k_values=[1, 2, 3, 5, 7, 10, 15, 20],
        learning_rates=[0.001, 0.005, 0.01, 0.05, 0.1],
        n_test_tasks=100,
        output_dir="experiments/paper_results/results/gap_vs_k_lr"
    )

    # Generate figures
    plot_gap_vs_k_lr(results, "experiments/paper_results/results/gap_vs_k_lr/figure.pdf")
    plot_summary_table(results, "experiments/paper_results/results/gap_vs_k_lr/summary_table.txt")

    print("\n✓ All done! Check experiments/paper_results/results/gap_vs_k_lr/ for results.")
