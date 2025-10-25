"""
Experiment: Optimality Gap vs Task Variance σ²_S

This script generates Figure 2 (Gap vs Variance) from the paper, validating:
    Gap(P, K) ∝ σ²_S

Expected result: R² ≈ 0.94 for linear fit
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
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from quantum.lindblad import LindbladSimulator
from quantum.noise_models import TaskDistribution, NoiseParameters, PSDToLindblad
from quantum.gates import state_fidelity
from meta_rl.policy import PulsePolicy
from meta_rl.maml import MAML
from baselines.robust_control import RobustPolicy
from theory.quantum_environment import create_quantum_environment
from theory.optimality_gap import OptimalityGapComputer, GapConstants
from theory.physics_constants import compute_control_relevant_variance


def linear_model(sigma2_S, slope):
    """Theoretical model: Gap ∝ σ²_S"""
    return slope * sigma2_S


def create_task_distributions_with_varying_variance(
    base_config: Dict,
    variance_levels: List[float] = [0.001, 0.002, 0.004, 0.008, 0.016]
) -> List[Tuple[TaskDistribution, float]]:
    """
    Create task distributions with different variances

    Strategy: vary the width of the parameter ranges to control σ²_S
    """
    distributions = []

    for var_scale in variance_levels:
        # Scale parameter ranges (wider range → higher variance)
        alpha_width = 1.5 * np.sqrt(var_scale / 0.004)  # Normalized to base
        A_width = 0.25 * np.sqrt(var_scale / 0.004)
        omega_c_width = 6.0 * np.sqrt(var_scale / 0.004)

        # Center values
        alpha_center = 1.25
        A_center = 0.175
        omega_c_center = 5.0

        task_dist = TaskDistribution(
            alpha_range=[
                max(0.5, alpha_center - alpha_width/2),
                min(2.0, alpha_center + alpha_width/2)
            ],
            A_range=[
                max(0.05, A_center - A_width/2),
                min(0.3, A_center + A_width/2)
            ],
            omega_c_range=[
                max(2.0, omega_c_center - omega_c_width/2),
                min(8.0, omega_c_center + omega_c_width/2)
            ]
        )

        distributions.append((task_dist, var_scale))

    return distributions


def run_gap_vs_variance_experiment(
    meta_policy_path: str,
    robust_policy_path: str,
    config: Dict,
    variance_levels: List[float] = [0.001, 0.002, 0.004, 0.008, 0.016],
    K_fixed: int = 5,
    n_test_tasks: int = 100,
    output_dir: str = "results/gap_vs_variance"
) -> Dict:
    """
    Main experiment: measure optimality gap as function of task variance σ²_S

    Returns:
        results: Dictionary containing gaps, variances, and validation metrics
    """
    print("=" * 80)
    print("EXPERIMENT: Optimality Gap vs Task Variance σ²_S")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)

    # Load trained policies
    print("\n[1/6] Loading trained policies...")
    meta_policy_template = PulsePolicy(
        input_dim=3,
        output_dim=config['num_segments'] * config['num_controls'],
        hidden_dims=config['policy_hidden_dims']
    )
    meta_policy_template.load_state_dict(torch.load(meta_policy_path))

    robust_policy = PulsePolicy(
        input_dim=3,
        output_dim=config['num_segments'] * config['num_controls'],
        hidden_dims=config['policy_hidden_dims']
    )
    robust_policy.load_state_dict(torch.load(robust_policy_path))
    robust_policy.eval()

    # Create quantum environment
    print("\n[2/6] Creating quantum environment...")
    env = create_quantum_environment(config)

    # Create task distributions with varying variance
    print(f"\n[3/6] Creating {len(variance_levels)} task distributions...")
    task_dists = create_task_distributions_with_varying_variance(
        config, variance_levels
    )

    # Measure gap for each variance level
    print(f"\n[4/6] Computing gaps for different variances (K={K_fixed})...")
    variances_computed = []
    gaps_mean = []
    gaps_std = []
    gaps_all = []

    for idx, (task_dist, var_target) in enumerate(task_dists):
        print(f"\n  Variance level {idx+1}/{len(task_dists)} (target: {var_target:.5f}):")

        # Sample tasks from this distribution
        test_tasks = [task_dist.sample() for _ in range(n_test_tasks)]

        # Compute actual variance
        sigma2_S = compute_control_relevant_variance(
            test_tasks,
            env.omega_control,
            env.control_susceptibility
        )
        variances_computed.append(sigma2_S)
        print(f"    Computed σ²_S = {sigma2_S:.5f}")

        fidelities_meta = []
        fidelities_robust = []

        for i, task in enumerate(test_tasks):
            if i % 20 == 0:
                print(f"    Task {i}/{n_test_tasks}...", end='\r')

            task_features = torch.tensor(
                [task.alpha, task.A, task.omega_c],
                dtype=torch.float32
            )

            # Robust policy (no adaptation)
            controls_robust = robust_policy(task_features).detach().numpy()
            controls_robust = controls_robust.reshape(
                config['num_segments'], config['num_controls']
            )
            fid_robust = env.compute_fidelity(controls_robust, task)
            fidelities_robust.append(fid_robust)

            # Meta policy with K_fixed adaptation steps
            # Clone policy for this task
            import copy
            adapted_policy = copy.deepcopy(meta_policy_template)
            adapted_policy.train()

            for k in range(K_fixed):
                # Compute loss with gradients
                loss = env.compute_loss_differentiable(
                    adapted_policy, task_features, task
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
            controls_meta = adapted_policy(task_features).detach().numpy()
            controls_meta = controls_meta.reshape(
                config['num_segments'], config['num_controls']
            )
            fid_meta = env.compute_fidelity(controls_meta, task)
            fidelities_meta.append(fid_meta)

        # Compute gap for this variance level
        gap_tasks = np.array(fidelities_meta) - np.array(fidelities_robust)
        gap_mean = np.mean(gap_tasks)
        gap_std = np.std(gap_tasks) / np.sqrt(n_test_tasks)  # Standard error

        gaps_mean.append(gap_mean)
        gaps_std.append(gap_std)
        gaps_all.append(gap_tasks.tolist())

        print(f"    Gap = {gap_mean:.4f} ± {gap_std:.4f}")

    # Fit theoretical model
    print("\n[5/6] Fitting theoretical model...")
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
    print(f"\n[6/6] Saving results to {output_dir}...")
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
    """Generate publication-quality figure"""
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))

    variances = np.array(results['variances'])
    gaps_mean = np.array(results['gaps_mean'])
    gaps_std = np.array(results['gaps_std'])

    # Plot empirical data
    ax.errorbar(
        variances, gaps_mean, yerr=gaps_std,
        fmt='o', markersize=10, capsize=5, capthick=2,
        label=f'Empirical Gap (K={results["K_fixed"]})',
        color='steelblue', linewidth=2
    )

    # Plot theoretical fit
    if results['fit']['success']:
        var_fine = np.linspace(0, max(variances), 100)
        gap_pred = linear_model(var_fine, results['fit']['slope'])
        ax.plot(
            var_fine, gap_pred, '--',
            label=f"Theory: $Gap \\propto \\sigma^2_S$\n" +
                  f"$R^2 = {results['fit']['r2']:.3f}$",
            color='darkred', linewidth=2
        )

    ax.set_xlabel('Control-Relevant Task Variance ($\\sigma^2_S$)',
                   fontsize=14, fontweight='bold')
    ax.set_ylabel('Optimality Gap (Fidelity)', fontsize=14, fontweight='bold')
    ax.set_title('Meta-Learning Gap vs Task Diversity',
                  fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {output_path}")
    plt.close()


if __name__ == "__main__":
    # Configuration matching paper Section 5
    config = {
        'num_qubits': 1,
        'num_controls': 2,
        'num_segments': 20,
        'evolution_time': 1.0,
        'target_gate': 'hadamard',
        'policy_hidden_dims': [128, 128, 128],
        'inner_lr': 0.01,
        'task_dist': {
            'alpha_range': [0.5, 2.0],
            'A_range': [0.05, 0.3],
            'omega_c_range': [2.0, 8.0]
        },
        'noise_frequencies': [1.0, 5.0, 10.0]
    }

    # Paths to trained models
    meta_path = "checkpoints/maml_best.pt"
    robust_path = "checkpoints/robust_best.pt"

    # Check if models exist
    if not os.path.exists(meta_path) or not os.path.exists(robust_path):
        print("ERROR: Trained models not found")
        print("Please train policies first using experiments/train_meta.py and train_robust.py")
        sys.exit(1)

    # Run experiment
    results = run_gap_vs_variance_experiment(
        meta_policy_path=meta_path,
        robust_policy_path=robust_path,
        config=config,
        variance_levels=[0.001, 0.002, 0.004, 0.008, 0.016],
        K_fixed=5,
        n_test_tasks=100,
        output_dir="results/gap_vs_variance"
    )

    # Generate figure
    plot_gap_vs_variance(results)
