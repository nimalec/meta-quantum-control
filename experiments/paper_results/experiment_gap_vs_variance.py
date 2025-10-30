"""
Experiment: Optimality Gap vs Task Variance σ²_S

This script generates Figure 2 (Gap vs Variance) from the paper, validating:
    Gap(P, K) ∝ σ²_S

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

# Add src to path
#sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from metaqctrl.quantum.lindblad import LindbladSimulator
from metaqctrl.quantum.noise_models import TaskDistribution, NoiseParameters, PSDToLindblad
from metaqctrl.quantum.gates import state_fidelity
from metaqctrl.meta_rl.policy import PulsePolicy
from metaqctrl.meta_rl.maml import MAML
from metaqctrl.baselines.robust_control import RobustPolicy, GRAPEOptimizer
from metaqctrl.theory.quantum_environment import create_quantum_environment
from metaqctrl.theory.optimality_gap import OptimalityGapComputer, GapConstants
from metaqctrl.theory.physics_constants import compute_control_relevant_variance
from metaqctrl.utils.checkpoint_utils import load_policy_from_checkpoint


def linear_model(sigma2_S, slope):
    """Theoretical model: Gap ∝ σ²_S"""
    return slope * sigma2_S


def create_task_distributions_with_varying_variance(
    base_config: Dict,
    variance_levels: List[float] = [0.001, 0.002, 0.004, 0.008, 0.016]
) -> List[Tuple[TaskDistribution, float]]:
    """
    Create task distributions with different variances

    FIXED: Use direct range scaling without clipping to ensure monotonic variance

    Strategy: Scale ranges around center proportionally to variance_scale
    Small variance → narrow ranges (tasks are similar)
    Large variance → wide ranges (tasks are diverse)
    """
    distributions = []

    # Base configuration (full ranges from config)
    alpha_full_range = base_config.get('alpha_range', [0.5, 2.0])
    A_full_range = base_config.get('A_range', [0.05, 0.3])
    omega_c_full_range = base_config.get('omega_c_range', [2.0, 8.0])

    # Centers
    alpha_center = np.mean(alpha_full_range)
    A_center = np.mean(A_full_range)
    omega_c_center = np.mean(omega_c_full_range)

    # Maximum half-widths
    alpha_max_hw = (alpha_full_range[1] - alpha_full_range[0]) / 2
    A_max_hw = (A_full_range[1] - A_full_range[0]) / 2
    omega_c_max_hw = (omega_c_full_range[1] - omega_c_full_range[0]) / 2

    for var_scale in variance_levels:
        # Normalize var_scale to [0, 1] range
        # Assume max variance corresponds to full parameter ranges
        # Scale factor: sqrt for variance -> std -> range
        scale_factor = np.sqrt(var_scale / max(variance_levels))

        # Scale half-widths proportionally
        alpha_hw = alpha_max_hw * scale_factor
        A_hw = A_max_hw * scale_factor
        omega_c_hw = omega_c_max_hw * scale_factor

        # Create ranges (no clipping!)
        task_dist = TaskDistribution(
            dist_type='uniform',
            ranges={
                'alpha': (alpha_center - alpha_hw, alpha_center + alpha_hw),
                'A': (A_center - A_hw, A_center + A_hw),
                'omega_c': (omega_c_center - omega_c_hw, omega_c_center + omega_c_hw)
            }
        )

        # Compute expected parameter variance for this distribution
        expected_var = task_dist.compute_variance()

        distributions.append((task_dist, expected_var))

        print(f"  Variance level {var_scale:.5f}:")
        print(f"    Expected σ²_params = {expected_var:.6f}")
        print(f"    α: [{alpha_center - alpha_hw:.3f}, {alpha_center + alpha_hw:.3f}]")
        print(f"    A: [{A_center - A_hw:.3f}, {A_center + A_hw:.3f}]")
        print(f"    ω_c: [{omega_c_center - omega_c_hw:.3f}, {omega_c_center + omega_c_hw:.3f}]")

    return distributions


def run_gap_vs_variance_experiment(
    meta_policy_path: str,
    robust_policy_path: str,
    config: Dict,
    variance_levels: List[float] = [0.0001, 0.001, 0.004, 0.01, 0.025],
    K_fixed: int = 5,
    n_test_tasks: int = 100,
    output_dir: str = "results/gap_vs_variance",
    include_grape: bool = True,
    grape_iterations: int = 100
) -> Dict:
    """
    Main experiment: measure optimality gap as function of task variance σ²_S

    Args:
        include_grape: Whether to include GRAPE baseline comparison
        grape_iterations: Number of GRAPE optimization iterations per task

    Returns:
        results: Dictionary containing gaps, variances, and validation metrics
    """
    print("=" * 80)
    print("EXPERIMENT: Optimality Gap vs Task Variance σ²_S")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)

    # Load trained policies
    print("\n[1/6] Loading trained policies...")

    # Load meta policy with automatic architecture detection
    print("\nLoading meta policy...")
    meta_policy_template = load_policy_from_checkpoint(
        meta_policy_path, config, eval_mode=False, verbose=True
    )

    # Load robust policy with automatic architecture detection
    print("\nLoading robust policy...")
    robust_policy = load_policy_from_checkpoint(
        robust_policy_path, config, eval_mode=True, verbose=True
    )

    # Create quantum environment
    print("\n[2/6] Creating quantum environment...")
    # target_state will be created from config['target_gate']
    from metaqctrl.theory.quantum_environment import get_target_state_from_config
    target_state = get_target_state_from_config(config)
    env = create_quantum_environment(config, target_state)

    # Create task distributions with varying variance
    print(f"\n[3/6] Creating {len(variance_levels)} task distributions...")
    task_dists = create_task_distributions_with_varying_variance(
        config, variance_levels
    )

    # Measure gap for each variance level
    print(f"\n[4/7] Computing gaps for different variances (K={K_fixed})...")
    variances_computed = []
    gaps_mean = []
    gaps_std = []
    gaps_all = []
    grape_fidelities_all = [] if include_grape else None

    for idx, (task_dist, var_expected) in enumerate(task_dists):
        print(f"\n  Variance level {idx+1}/{len(task_dists)}:")
        print(f"    Expected σ²_params = {var_expected:.6f}")

        # Sample tasks from this distribution
        test_tasks = task_dist.sample(n_test_tasks)

        # Compute parameter variance (simple, direct measure of task diversity)
        params_array = np.array([[t.alpha, t.A, t.omega_c] for t in test_tasks])
        sigma2_params = np.var(params_array, axis=0).sum()  # Total variance across all params

        variances_computed.append(sigma2_params)
        print(f"    Computed σ²_params (empirical) = {sigma2_params:.6f}")
        print(f"    Ratio empirical/expected = {sigma2_params/var_expected:.3f}")

        # Diagnostic: show actual parameter ranges sampled
        print(f"    Sampled α: [{params_array[:, 0].min():.3f}, {params_array[:, 0].max():.3f}]")
        print(f"    Sampled A: [{params_array[:, 1].min():.3f}, {params_array[:, 1].max():.3f}]")
        print(f"    Sampled ω_c: [{params_array[:, 2].min():.3f}, {params_array[:, 2].max():.3f}]")

        fidelities_meta = []
        fidelities_robust = []
        fidelities_grape = []

        for i, task in enumerate(test_tasks):
            if i % 20 == 0:
                print(f"    Task {i}/{n_test_tasks}...", end='\r')

            task_features = torch.tensor(
                [task.alpha, task.A, task.omega_c],
                dtype=torch.float32
            )

            # Robust policy (no adaptation)
            with torch.no_grad():
                controls_robust = robust_policy(task_features).detach().numpy()
            fid_robust = env.compute_fidelity(controls_robust, task)
            fidelities_robust.append(fid_robust)

            # Meta policy with K_fixed adaptation steps
            # Clone policy for this task
            import copy
            adapted_policy = copy.deepcopy(meta_policy_template)
            adapted_policy.train()

            for k in range(K_fixed):
                # Compute loss with gradients
                # FIXED: Pass device parameter explicitly
                loss = env.compute_loss_differentiable(
                    adapted_policy, task, device=torch.device('cpu')
                )

                # Gradient step
                loss.backward()
                with torch.no_grad():
                    for param in adapted_policy.parameters():
                        if param.grad is not None:
                            param -= config['inner_lr'] * param.grad
                            param.grad.zero_()

            # Evaluate adapted policy
            adapted_policy.eval()
            with torch.no_grad():
                controls_meta = adapted_policy(task_features).detach().numpy()
            fid_meta = env.compute_fidelity(controls_meta, task)
            fidelities_meta.append(fid_meta)

            # GRAPE baseline (if enabled)
            if include_grape:
                grape = GRAPEOptimizer(
                    n_segments=config['n_segments'],
                    n_controls=config['n_controls'],
                    T=config.get('horizon', 1.0),
                    learning_rate=0.1,
                    method='adam',
                    device=torch.device('cpu')
                )

                def simulate_fn(controls_np, task_params):
                    return env.compute_fidelity(controls_np, task_params)

                optimal_controls = grape.optimize(
                    simulate_fn=simulate_fn,
                    task_params=task,
                    max_iterations=grape_iterations,
                    verbose=False
                )
                fid_grape = env.compute_fidelity(optimal_controls, task)
                fidelities_grape.append(fid_grape)

        # Compute gap for this variance level
        gap_tasks = np.array(fidelities_meta) - np.array(fidelities_robust)
        gap_mean = np.mean(gap_tasks)
        gap_std = np.std(gap_tasks) / np.sqrt(n_test_tasks)  # Standard error

        gaps_mean.append(gap_mean)
        gaps_std.append(gap_std)
        gaps_all.append(gap_tasks.tolist())

        if include_grape:
            grape_mean = np.mean(fidelities_grape)
            grape_std = np.std(fidelities_grape) / np.sqrt(n_test_tasks)
            grape_fidelities_all.append({
                'mean': float(grape_mean),
                'std': float(grape_std),
                'values': fidelities_grape
            })
            print(f"    Gap = {gap_mean:.4f} ± {gap_std:.4f}, GRAPE = {grape_mean:.4f} ± {grape_std:.4f}")
        else:
            print(f"    Gap = {gap_mean:.4f} ± {gap_std:.4f}")

    # Fit theoretical model
    print("\n[5/7] Fitting theoretical model...")
    variances_array = np.array(variances_computed)
    gaps_array = np.array(gaps_mean)

    # Fit: Gap = slope * σ²_S
    try:
        popt, pcov = curve_fit(
            linear_model,
            variances_array,
            gaps_array,
            p0=[gaps_array[-1] / variances_array[-1]],  # Initial guess
            sigma=gaps_std,
            absolute_sigma=True
        )
        slope_fit = popt[0]

        # Compute R²
        gaps_predicted = linear_model(variances_array, slope_fit)
        r2 = r2_score(gaps_array, gaps_predicted)

        print(f"\n  Fitted parameters:")
        print(f"    slope (Gap/σ²_S) = {slope_fit:.2f}")
        print(f"    R² = {r2:.4f}")

        fit_success = True
    except Exception as e:
        print(f"\n  Fitting failed: {e}")
        slope_fit = r2 = None
        fit_success = False

    # Save results
    print(f"\n[6/7] Saving results to {output_dir}...")
    results = {
        'variances': variances_computed,
        'gaps_mean': gaps_mean,
        'gaps_std': gaps_std,
        'gaps_all': gaps_all,
        'K_fixed': K_fixed,
        'fit': {
            'success': fit_success,
            'slope': float(slope_fit) if fit_success else None,
            'r2': float(r2) if fit_success else None
        },
        'grape': {
            'included': include_grape,
            'fidelities_per_variance': grape_fidelities_all if include_grape else [],
            'iterations': grape_iterations if include_grape else None
        },
        'config': config,
        'n_test_tasks': n_test_tasks
    }

    with open(f"{output_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)

    return results


def plot_gap_vs_variance(results: Dict, output_path: str = "results/gap_vs_variance/figure.pdf"):
    """Generate publication-quality figure with diagnostics"""
    sns.set_style("whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    variances = np.array(results['variances'])
    gaps_mean = np.array(results['gaps_mean'])
    gaps_std = np.array(results['gaps_std'])

    # Plot 1: Gap vs Variance with fit
    ax1.errorbar(
        variances, gaps_mean, yerr=gaps_std,
        fmt='o', markersize=10, capsize=5, capthick=2,
        label=f'MAML Gap (K={results["K_fixed"]})',
        color='steelblue', linewidth=2
    )

    # Plot theoretical fit
    if results['fit']['success']:
        var_fine = np.linspace(0, max(variances), 100)
        gap_pred = linear_model(var_fine, results['fit']['slope'])
        ax1.plot(
            var_fine, gap_pred, '--',
            label=f"Linear Fit: Gap = {results['fit']['slope']:.2f} × σ²\n" +
                  f"$R^2 = {results['fit']['r2']:.3f}$",
            color='darkred', linewidth=2
        )

    # Plot GRAPE baseline per variance level
    if results['grape']['included']:
        grape_means = [d['mean'] for d in results['grape']['fidelities_per_variance']]
        grape_stds = [d['std'] for d in results['grape']['fidelities_per_variance']]
        ax1.errorbar(
            variances, grape_means, yerr=grape_stds,
            fmt='s', markersize=8, capsize=4, capthick=2,
            label='GRAPE Baseline',
            color='green', linewidth=2, alpha=0.7
        )

    ax1.set_xlabel('Task Parameter Variance ($\\sigma^2_{params}$)',
                   fontsize=14, fontweight='bold')
    ax1.set_ylabel('Optimality Gap (Meta - Robust)', fontsize=14, fontweight='bold')
    ax1.set_title('Gap vs Task Diversity', fontsize=16, fontweight='bold')
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=12)

    # Plot 2: Diagnostic - verify variance is monotonic
    ax2.plot(range(len(variances)), variances, 'o-', markersize=10,
             linewidth=2, color='purple', label='Computed Variance')
    ax2.set_xlabel('Variance Level Index', fontsize=14, fontweight='bold')
    ax2.set_ylabel('$\\sigma^2_{params}$', fontsize=14, fontweight='bold')
    ax2.set_title('Variance Verification (Should be Monotonic)', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=12)

    # Add text showing if monotonic
    is_monotonic = all(variances[i] <= variances[i+1] for i in range(len(variances)-1))
    status_text = "✓ Monotonic" if is_monotonic else "✗ NOT Monotonic"
    status_color = 'green' if is_monotonic else 'red'
    ax2.text(0.05, 0.95, status_text, transform=ax2.transAxes,
             fontsize=14, fontweight='bold', color=status_color,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {output_path}")

    # Also save PNG
    png_path = output_path.replace('.pdf', '.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {png_path}")

    plt.close()


if __name__ == "__main__":
    # Configuration matching paper Section 5
    config = {
        'num_qubits': 1,
        'n_controls': 2,
        'n_segments': 20,
        'horizon': 1.0,
        'target_gate': 'hadamard',
        'hidden_dim': 128,
        'n_hidden_layers': 2,
        'inner_lr': 0.01,
        'alpha_range': [0.5, 2.0],
        'A_range': [0.05, 0.3],
        'omega_c_range': [2.0, 8.0],
        'noise_frequencies': [1.0, 5.0, 10.0]
    }

    # Paths to trained models
    meta_path = "../checkpoints/maml_20251027_161519_best_policy.pt"
    robust_path = "../checkpoints/robust_minimax_20251027_162238_best_policy.pt" 

    # Check if models exist
    if not os.path.exists(meta_path) or not os.path.exists(robust_path):
        print("ERROR: Trained models not found")
        print("Please train policies first using experiments/train_meta.py and train_robust.py")
        sys.exit(1)

    # Run experiment with FIXED parameters
    results = run_gap_vs_variance_experiment(
        meta_policy_path=meta_path,
        robust_policy_path=robust_path,
        config=config,
        variance_levels=[0.0001, 0.001, 0.004, 0.02, 0.25],  # FIXED: Better range
        K_fixed=1,
        n_test_tasks=10,  # FIXED: Increased from 2 to 100 for proper statistics
        output_dir="results/gap_vs_variance",
        include_grape=False
    )

    # Generate figure
    plot_gap_vs_variance(results)
