"""
Experiment: Optimality Gap vs Adaptation Steps K

This script generates Figure 1 (Gap vs K) from the paper, validating:
    Gap(P, K) ∝ (1 - e^(-μηK))

Expected result: R² ≈ 0.96 for exponential fit
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
from metaqctrl.baselines.robust_control import RobustPolicy
from metaqctrl.theory.quantum_environment import create_quantum_environment
from metaqctrl.theory.optimality_gap import OptimalityGapComputer, GapConstants
from metaqctrl.theory.physics_constants import estimate_all_constants


def exponential_model(K, gap_max, mu_eta):
    """Theoretical model: Gap(K) = gap_max * (1 - exp(-μηK))"""
    return gap_max * (1 - np.exp(-mu_eta * K))


def run_gap_vs_k_experiment(
    meta_policy_path: str,
    robust_policy_path: str,
    config: Dict,
    k_values: List[int] = [1, 2, 3, 5, 7, 10, 15, 20],
    n_test_tasks: int = 100,
    output_dir: str = "results/gap_vs_k"
) -> Dict:
    """
    Main experiment: measure optimality gap as function of adaptation steps K

    Returns:
        results: Dictionary containing gaps, fits, and validation metrics
    """
    print("=" * 80)
    print("EXPERIMENT: Optimality Gap vs Adaptation Steps K")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)

    # Load trained policies
    print("\n[1/6] Loading trained policies...")
    meta_policy_template = PulsePolicy(
        task_feature_dim=3,
        hidden_dim=config['policy_hidden_dims'][0],
        n_hidden_layers=len(config['policy_hidden_dims']) - 1,
        n_segments=config['num_segments'],
        n_controls=config['num_controls']
    )
    meta_policy_template.load_state_dict(torch.load(meta_policy_path))
    meta_policy_template.eval()

    robust_policy = PulsePolicy(
        task_feature_dim=3,
        hidden_dim=config['policy_hidden_dims'][0],
        n_hidden_layers=len(config['policy_hidden_dims']) - 1,
        n_segments=config['num_segments'],
        n_controls=config['num_controls']
    )
    robust_policy.load_state_dict(torch.load(robust_policy_path))
    robust_policy.eval()

    # Create quantum environment
    print("\n[2/6] Creating quantum environment...")
    # target_state will be created from config['target_gate']
    from metaqctrl.theory.quantum_environment import get_target_state_from_config
    target_state = get_target_state_from_config(config)
    env = create_quantum_environment(config, target_state)

    # Sample test tasks
    print(f"\n[3/6] Sampling {n_test_tasks} test tasks...")
    task_dist = TaskDistribution(
        dist_type=config.get('task_dist_type', 'uniform'),
        ranges={
            'alpha': tuple(config['task_dist']['alpha_range']),
            'A': tuple(config['task_dist']['A_range']),
            'omega_c': tuple(config['task_dist']['omega_c_range'])
        }
    )
    test_tasks = task_dist.sample(n_test_tasks)

    # Measure gap for each K
    print(f"\n[4/6] Computing gaps for K = {k_values}...")
    gaps_mean = []
    gaps_std = []
    gaps_all = {k: [] for k in k_values}

    for K in k_values:
        print(f"\n  K = {K}:")
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
            with torch.no_grad():
                controls_robust = robust_policy(task_features).detach().numpy()
            fid_robust = env.compute_fidelity(controls_robust, task)
            fidelities_robust.append(fid_robust)

            # Meta policy with K adaptation steps
            # CRITICAL FIX: Clone policy for each task to avoid mutation
            import copy
            adapted_policy = copy.deepcopy(meta_policy_template)
            adapted_policy.train()

            for k in range(K):
                # Compute loss with gradients
                loss = env.compute_loss_differentiable(
                    adapted_policy, task
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

        # Compute gap for this K
        gap_tasks = np.array(fidelities_meta) - np.array(fidelities_robust)
        gap_mean = np.mean(gap_tasks)
        gap_std = np.std(gap_tasks) / np.sqrt(n_test_tasks)  # Standard error

        gaps_mean.append(gap_mean)
        gaps_std.append(gap_std)
        gaps_all[K] = gap_tasks.tolist()

        print(f"    Gap = {gap_mean:.4f} ± {gap_std:.4f}")

    # Fit theoretical model
    print("\n[5/6] Fitting theoretical model...")
    k_array = np.array(k_values)
    gaps_array = np.array(gaps_mean)

    # Fit: Gap(K) = gap_max * (1 - exp(-μηK))
    try:
        popt, pcov = curve_fit(
            exponential_model,
            k_array,
            gaps_array,
            p0=[gaps_array[-1], 0.1],  # Initial guess
            sigma=gaps_std,
            absolute_sigma=True,
            maxfev=10000
        )
        gap_max_fit, mu_eta_fit = popt

        # Compute R²
        gaps_predicted = exponential_model(k_array, *popt)
        r2 = r2_score(gaps_array, gaps_predicted)

        print(f"\n  Fitted parameters:")
        print(f"    gap_max = {gap_max_fit:.4f}")
        print(f"    μη = {mu_eta_fit:.4f}")
        print(f"    R² = {r2:.4f}")

        fit_success = True
    except Exception as e:
        print(f"\n  Fitting failed: {e}")
        gap_max_fit = mu_eta_fit = r2 = None
        fit_success = False

    # Save results
    print(f"\n[6/6] Saving results to {output_dir}...")
    results = {
        'k_values': k_values,
        'gaps_mean': gaps_mean,
        'gaps_std': gaps_std,
        'gaps_all': gaps_all,
        'fit': {
            'success': fit_success,
            'gap_max': float(gap_max_fit) if fit_success else None,
            'mu_eta': float(mu_eta_fit) if fit_success else None,
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


def plot_gap_vs_k(results: Dict, output_path: str = "results/gap_vs_k/figure.pdf"):
    """Generate publication-quality figure"""
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))

    k_values = np.array(results['k_values'])
    gaps_mean = np.array(results['gaps_mean'])
    gaps_std = np.array(results['gaps_std'])

    # Plot empirical data
    ax.errorbar(
        k_values, gaps_mean, yerr=gaps_std,
        fmt='o', markersize=10, capsize=5, capthick=2,
        label='Empirical Gap', color='steelblue', linewidth=2
    )

    # Plot theoretical fit
    if results['fit']['success']:
        k_fine = np.linspace(0, max(k_values), 100)
        gap_pred = exponential_model(
            k_fine,
            results['fit']['gap_max'],
            results['fit']['mu_eta']
        )
        ax.plot(
            k_fine, gap_pred, '--',
            label=f"Theory: $Gap_{{max}}(1 - e^{{-\\mu\\eta K}})$\n" +
                  f"$R^2 = {results['fit']['r2']:.3f}$",
            color='darkred', linewidth=2
        )

    ax.set_xlabel('Adaptation Steps (K)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Optimality Gap (Fidelity)', fontsize=14, fontweight='bold')
    ax.set_title('Meta-Learning Gap vs Adaptation Steps', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='lower right')
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

    # Paths to trained models (adjust as needed)
    meta_path = "checkpoints/maml_best.pt"
    robust_path = "checkpoints/robust_best.pt"

    # Check if models exist
    if not os.path.exists(meta_path):
        print(f"ERROR: Meta-policy not found at {meta_path}")
        print("Please train the meta-policy first using experiments/train_meta.py")
        sys.exit(1)

    if not os.path.exists(robust_path):
        print(f"ERROR: Robust policy not found at {robust_path}")
        print("Please train the robust policy first using experiments/train_robust.py")
        sys.exit(1)

    # Run experiment
    results = run_gap_vs_k_experiment(
        meta_policy_path=meta_path,
        robust_policy_path=robust_path,
        config=config,
        k_values=[1, 2, 3, 5, 7, 10, 15, 20],
        n_test_tasks=100,
        output_dir="results/gap_vs_k"
    )

    # Generate figure
    plot_gap_vs_k(results)
