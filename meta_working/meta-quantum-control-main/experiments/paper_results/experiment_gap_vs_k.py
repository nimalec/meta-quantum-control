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
from metaqctrl.baselines.robust_control import RobustPolicy, GRAPEOptimizer
from metaqctrl.theory.quantum_environment import create_quantum_environment
from metaqctrl.theory.optimality_gap import OptimalityGapComputer, GapConstants
from metaqctrl.theory.physics_constants import estimate_all_constants
from metaqctrl.utils.checkpoint_utils import load_policy_from_checkpoint


def exponential_model(K, gap_max, mu_eta):
    """Theoretical model: Gap(K) = gap_max * (1 - exp(-μηK))"""
    return gap_max * (1 - np.exp(-mu_eta * K))


def run_gap_vs_k_experiment(
    meta_policy_path: str,
    robust_policy_path: str,
    config: Dict,
    k_values: List[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,12, 13, 14,  15, 16, 17, 18, 19, 20],
    n_test_tasks: int = 30,
    output_dir: str = "results/gap_vs_k",
    include_grape: bool = False,
    grape_iterations: int = 100
) -> Dict:
    """
    Main experiment: measure optimality gap as function of adaptation steps K

    Args:
        include_grape: Whether to include GRAPE baseline comparison
        grape_iterations: Number of GRAPE optimization iterations per task

    Returns:
        results: Dictionary containing gaps, fits, and validation metrics
    """
    print("=" * 80)
    print("EXPERIMENT: Optimality Gap vs Adaptation Steps K")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)

    # Load trained policies
    print("\n[1/6] Loading trained policies...")

    # Load meta policy with automatic architecture detection
    print("Loading meta policy...")
    meta_policy_template = load_policy_from_checkpoint(
        meta_policy_path, config, eval_mode=False, verbose=True
    )

    # Load robust policy with automatic architecture detection
    print("Loading robust policy...")
    #robust_policy = load_policy_from_checkpoint(
    #    robust_policy_path, config, eval_mode=True, verbose=True
   #     )

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
            'alpha': tuple(config['alpha_range']),
            'A': tuple(config['A_range']),
            'omega_c': tuple(config['omega_c_range'])
        }
    )
    test_tasks = task_dist.sample(n_test_tasks)

    # Optionally compute GRAPE baseline (independent of K)
    fidelities_grape = []
    if include_grape:
        print(f"\n[4/7] Computing GRAPE baseline (once for all K)...")
        for i, task in enumerate(test_tasks):
            if i % 10 == 0:
                print(f"    Task {i}/{n_test_tasks}...", end='\r')

            # Create GRAPE optimizer for this task
            grape = GRAPEOptimizer(
                n_segments=config['n_segments'],
                n_controls=config['n_controls'],
                T=config.get('horizon', 1.0),
                learning_rate=0.1,
                method='adam',
                device=torch.device('cpu')
            )

            # Define simulation function for GRAPE
            def simulate_fn(controls_np, task_params):
                return env.compute_fidelity(controls_np, task_params)

            # Optimize with GRAPE
            optimal_controls = grape.optimize(
                simulate_fn=simulate_fn,
                task_params=task,
                max_iterations=grape_iterations,
                verbose=False
            )

            # Compute final fidelity
            fid_grape = env.compute_fidelity(optimal_controls, task)
            fidelities_grape.append(fid_grape)

        grape_mean = np.mean(fidelities_grape)
        grape_std = np.std(fidelities_grape) / np.sqrt(n_test_tasks)
        print(f"\n    GRAPE Fidelity = {grape_mean:.4f} ± {grape_std:.4f}")
        print()

    # Measure gap for each K
    print(f"\n[5/7] Computing gaps for K = {k_values}...")
    gaps_mean = []
    gaps_std = []
    gaps_all = {k: [] for k in k_values}

    for K in k_values:
        print(f"\n  K = {K}:")
        fidelities_meta = []
        fidelities_robust = []

        for i, task in enumerate(test_tasks):
            if i % 1 == 0:
                print(f"    Task {i}/{n_test_tasks}...", end='\r')

            task_features = torch.tensor(
                [task.alpha, task.A, task.omega_c],
                dtype=torch.float32
            )

            # Robust policy (no adaptation)
            #with torch.no_grad():
             #   controls_robust = robust_policy(task_features).detach().numpy()
            #fid_robust = env.compute_fidelity(controls_robust, task)
            fidelities_robust.append(0)

            # Meta policy with K adaptation steps
            # CRITICAL FIX: Clone policy for each task to avoid mutation
            import copy
            adapted_policy = copy.deepcopy(meta_policy_template)
            adapted_policy.train()

            for k in range(K):
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

        # Compute gap for this K
        #gap_tasks = np.array(fidelities_meta) - np.array(fidelities_robust)
        gap_tasks = np.array(fidelities_meta)
        gap_mean = np.mean(gap_tasks)
        gap_std = np.std(gap_tasks) / np.sqrt(n_test_tasks)  # Standard error

        gaps_mean.append(gap_mean)
        gaps_std.append(gap_std)
        gaps_all[K] = gap_tasks.tolist()

        print(f"    Gap = {gap_mean:.4f} ± {gap_std:.4f}")

    # Fit theoretical model
    print("\n[6/7] Fitting theoretical model...")
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

        fit_success = True
    except Exception as e:
        print(f"\n  Fitting failed: {e}")
        gap_max_fit = mu_eta_fit = r2 = None
        fit_success = False

    # Save results
    print(f"\n[7/7] Saving results to {output_dir}...")
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
        'grape': {
            'included': include_grape,
            'fidelities': fidelities_grape if include_grape else [],
            'mean': float(grape_mean) if include_grape else None,
            'std': float(grape_std) if include_grape else None,
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


def plot_gap_vs_k(results: Dict, output_path: str = "results/gap_vs_k/figure.pdf"):
    """Generate publication-quality figure with GRAPE baseline"""
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    k_values = np.array(results['k_values'])
    gaps_mean = np.array(results['gaps_mean'])
    gaps_std = np.array(results['gaps_std'])

    # Plot empirical gap data
    ax.errorbar(
        k_values, gaps_mean, yerr=gaps_std,
        fmt='o', markersize=10, capsize=5, capthick=2,
        label='MAML Gap (Meta - Robust)', color='steelblue', linewidth=2
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

    # Add GRAPE baseline as horizontal line
    if results['grape']['included']:
        grape_mean = results['grape']['mean']
        grape_std = results['grape']['std']
        ax.axhline(
            y=grape_mean,
            color='green',
            linestyle='-',
            linewidth=2,
            label=f'GRAPE Baseline (fid={grape_mean:.3f}±{grape_std:.3f})',
            alpha=0.7
        )
        # Add shaded region for GRAPE uncertainty
        ax.fill_between(
            [0, max(k_values)],
            grape_mean - grape_std,
            grape_mean + grape_std,
            color='green',
            alpha=0.2
        )

    ax.set_xlabel('Adaptation Steps (K)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Fidelity / Optimality Gap', fontsize=14, fontweight='bold')
    ax.set_title('Meta-Learning Gap vs Adaptation Steps (with GRAPE Baseline)', fontsize=16, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
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
        'n_controls': 2,
        'n_segments': 100,
        'horizon': 1.0,
        'target_gate': 'pauli_x',
        'hidden_dim': 256,
        'n_hidden_layers': 2,
        'inner_lr': 0.01,
        'alpha_range': [0.9,1.1],    # ±75%
         'A_range': [0.09, 0.11],
         'omega_c_range': [9e3, 11e3], 
        'noise_frequencies': np.linspace(1e3, 1e6,1000) 
    }

    # Paths to trained models (adjust as needed)
    # Try multiple possible locations
    meta_path = "../train_scripts/checkpoints/maml_best_pauli_x_best_policy.pt" 
    robust_path =  "../train_scripts/checkpoints/robust_minimax_best_policy.pt"

    print(f"Using meta policy: {meta_path}")
    print(f"Using robust policy: {robust_path}")

    # Run experiment
    results = run_gap_vs_k_experiment(
        meta_policy_path=meta_path,
        robust_policy_path=robust_path,
        config=config,
         k_values=[1, 2, 3, 4, 5, 6,  7,8, 9,  10,11, 12], 
       # k_values=[1, 2, 3, 4, 5, 6,  7,8, 9,  10,11, 12, 13, 14,  15, 16, 17, 18, 19, 20, 22, 25, 27,  30,35,  40,45,  50, 55, 60],
        n_test_tasks=60,
        output_dir="results/gap_vs_k_v2", 
        include_grape = False
    )

    # Generate figure
    plot_gap_vs_k(results)
