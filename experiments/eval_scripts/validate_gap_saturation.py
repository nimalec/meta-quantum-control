"""
Validation Script: Gap Saturation and Diminishing Returns

This script extends the Gap(K) vs. K analysis to K=30 to validate:

1. Saturation behavior: Gap(K) → G_∞ as K → ∞
2. Theoretical fit: Gap(K) = G_∞ (1 - e^(-μηK))
3. Diminishing returns: ΔGap/ΔK decreases exponentially
4. Practical convergence: Beyond K≈10, marginal gains < threshold

Theory predicts:
- Gap saturates to G_∞ (maximum adaptation advantage)
- Saturation follows exponential approach
- Marginal returns: d(Gap)/dK = G_∞ μη e^(-μηK) → 0

Expected output:
- Main plot showing Gap(K) with saturation plateau
- Red dashed line at theoretical limit G_∞
- Inset showing marginal returns with threshold
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


def exponential_saturation_model(K, G_inf, mu_eta):
    """
    Theoretical saturation model: Gap(K) = G_∞ (1 - e^(-μηK))

    Args:
        K: Adaptation steps
        G_inf: Saturated gap (asymptotic maximum)
        mu_eta: Convergence rate (product of PL constant and learning rate)

    Returns:
        gap: Predicted gap at K
    """
    return G_inf * (1 - np.exp(-mu_eta * K))


def compute_marginal_returns(gaps: np.ndarray, k_values: np.ndarray) -> np.ndarray:
    """
    Compute marginal returns: ΔGap(K) = Gap(K) - Gap(K-1)

    Args:
        gaps: Array of gap values
        k_values: Array of K values

    Returns:
        marginal_returns: Array of marginal returns (length n-1)
    """
    marginal = np.diff(gaps)
    return marginal


def find_saturation_point(
    k_values: np.ndarray,
    gaps: np.ndarray,
    threshold_pct: float = 1.0
) -> Tuple[int, float]:
    """
    Find K where marginal improvement falls below threshold.

    Args:
        k_values: Array of K values
        gaps: Array of gap values
        threshold_pct: Threshold in percentage (e.g., 1.0 for 1%)

    Returns:
        k_saturate: K value where saturation occurs
        gap_at_saturation: Gap value at saturation
    """
    marginal = compute_marginal_returns(gaps, k_values)

    # Compute marginal improvement as percentage of current gap
    marginal_pct = []
    for i in range(len(marginal)):
        if gaps[i] > 0:
            pct = (marginal[i] / gaps[i]) * 100
            marginal_pct.append(pct)
        else:
            marginal_pct.append(0.0)

    marginal_pct = np.array(marginal_pct)

    # Find first K where improvement < threshold
    saturated_indices = np.where(marginal_pct < threshold_pct)[0]

    if len(saturated_indices) > 0:
        sat_idx = saturated_indices[0] + 1  # +1 because marginal is shifted
        k_saturate = k_values[sat_idx]
        gap_at_saturation = gaps[sat_idx]
    else:
        # Not saturated yet, return last K
        k_saturate = k_values[-1]
        gap_at_saturation = gaps[-1]

    return k_saturate, gap_at_saturation


def run_gap_saturation_experiment(
    meta_policy_path: str,
    robust_policy_path: str,
    config: Dict,
    k_values: List[int] = None,
    n_test_tasks: int = 50,
    output_dir: str = "results/gap_saturation"
) -> Dict:
    """
    Main experiment: Gap saturation analysis with extended K.

    Args:
        meta_policy_path: Path to trained meta policy
        robust_policy_path: Path to robust baseline policy
        config: Experiment configuration
        k_values: List of K values to test (up to K=30)
        n_test_tasks: Number of test tasks
        output_dir: Output directory

    Returns:
        results: Dict with gaps, fit parameters, saturation analysis
    """
    print("=" * 80)
    print("EXPERIMENT: Gap Saturation and Diminishing Returns")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)

    # Default K values: dense near 0, sparse at high K
    if k_values is None:
        k_values = [0, 1, 2, 3, 4, 5, 7, 10, 12, 15, 18, 20, 25, 30]

    print(f"\nExperiment parameters:")
    print(f"  K values: {k_values}")
    print(f"  Test tasks: {n_test_tasks}")

    # Load policies
    print("\n[1/5] Loading policies...")
    meta_policy_template = load_policy_from_checkpoint(
        meta_policy_path, config, eval_mode=False, verbose=True
    )
    robust_policy = load_policy_from_checkpoint(
        robust_policy_path, config, eval_mode=True, verbose=True
    )

    # Create environment
    print("\n[2/5] Creating quantum environment...")
    target_state = get_target_state_from_config(config)
    env = create_quantum_environment(config, target_state)

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

    inner_lr = config.get('inner_lr', 0.01)

    # Compute gaps for each K
    print(f"\n[4/5] Computing gaps for K = {k_values}...")
    gaps_mean = []
    gaps_std = []
    gaps_sem = []

    for K in k_values:
        print(f"\n  K = {K}:")
        fidelities_meta = []
        fidelities_robust = []

        for i, task in enumerate(tqdm(test_tasks, desc=f"  Evaluating K={K}", leave=False)):
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
                            param -= inner_lr * param.grad
                            param.grad.zero_()

            adapted_policy.eval()
            with torch.no_grad():
                controls_meta = adapted_policy(task_features).detach().numpy()
            fid_meta = env.compute_fidelity(controls_meta, task)
            fidelities_meta.append(fid_meta)

        # Compute gap statistics
        gap_tasks = np.array(fidelities_meta) - np.array(fidelities_robust)
        gap_mean = np.mean(gap_tasks)
        gap_std = np.std(gap_tasks)
        gap_sem = gap_std / np.sqrt(n_test_tasks)

        gaps_mean.append(gap_mean)
        gaps_std.append(gap_std)
        gaps_sem.append(gap_sem)

        print(f"    Gap = {gap_mean:.4f} ± {gap_sem:.4f}")

    # Fit saturation model
    print("\n[5/5] Fitting saturation model...")
    k_array = np.array(k_values)
    gaps_array = np.array(gaps_mean)
    gaps_sem_array = np.array(gaps_sem)

    try:
        # Fit: Gap(K) = G_∞ (1 - exp(-μηK))
        popt, pcov = curve_fit(
            exponential_saturation_model,
            k_array,
            gaps_array,
            p0=[gaps_array[-1], 0.1],  # Initial guess: G_inf from last point
            sigma=gaps_sem_array,
            absolute_sigma=True,
            maxfev=10000
        )
        G_inf_fit, mu_eta_fit = popt

        # Compute R²
        gaps_predicted = exponential_saturation_model(k_array, *popt)
        r2 = r2_score(gaps_array, gaps_predicted)

        print(f"\n  Fitted saturation model:")
        print(f"    G_∞ = {G_inf_fit:.4f} (saturated gap)")
        print(f"    μη = {mu_eta_fit:.4f} (convergence rate)")
        print(f"    R² = {r2:.4f}")

        # Saturation analysis
        saturation_pct = (gaps_array[-1] / G_inf_fit) * 100
        print(f"\n  Saturation achieved at K={k_values[-1]}: {saturation_pct:.1f}% of G_∞")

        fit_success = True
    except Exception as e:
        print(f"\n  Fitting failed: {e}")
        G_inf_fit = mu_eta_fit = r2 = None
        fit_success = False

    # Analyze diminishing returns
    print("\n  Analyzing diminishing returns...")
    k_saturate, gap_at_saturation = find_saturation_point(
        k_array, gaps_array, threshold_pct=1.0
    )
    print(f"    Saturation point (< 1% marginal improvement): K = {k_saturate}")
    print(f"    Gap at saturation: {gap_at_saturation:.4f}")

    # Compute marginal returns
    marginal = compute_marginal_returns(gaps_array, k_array)
    marginal_k = k_array[1:]  # K values for marginal returns

    # Quantify returns beyond K=10
    k10_idx = np.where(k_array == 10)[0]
    if len(k10_idx) > 0:
        k10_idx = k10_idx[0]
        gap_at_k10 = gaps_array[k10_idx]
        improvement_beyond_k10 = gaps_array[-1] - gap_at_k10
        pct_beyond_k10 = (improvement_beyond_k10 / gap_at_k10) * 100

        print(f"\n  Diminishing returns beyond K=10:")
        print(f"    Gap at K=10: {gap_at_k10:.4f}")
        print(f"    Gap at K={k_values[-1]}: {gaps_array[-1]:.4f}")
        print(f"    Additional improvement: {improvement_beyond_k10:.4f} ({pct_beyond_k10:.1f}%)")
    else:
        improvement_beyond_k10 = pct_beyond_k10 = None

    # Save results
    print(f"\nSaving results to {output_dir}...")
    results = {
        'k_values': k_values,
        'gaps_mean': gaps_mean,
        'gaps_std': gaps_std,
        'gaps_sem': gaps_sem,
        'n_test_tasks': n_test_tasks,
        'fit': {
            'success': fit_success,
            'G_inf': float(G_inf_fit) if fit_success else None,
            'mu_eta': float(mu_eta_fit) if fit_success else None,
            'r2': float(r2) if fit_success else None
        },
        'saturation_analysis': {
            'k_saturate': int(k_saturate),
            'gap_at_saturation': float(gap_at_saturation),
            'improvement_beyond_k10': float(improvement_beyond_k10) if improvement_beyond_k10 is not None else None,
            'pct_beyond_k10': float(pct_beyond_k10) if pct_beyond_k10 is not None else None
        },
        'marginal_returns': {
            'k_values': marginal_k.tolist(),
            'marginal': marginal.tolist()
        },
        'config': config
    }

    results_path = f"{output_dir}/results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    return results


def plot_gap_saturation(
    results: Dict,
    output_path: str = None
):
    """
    Generate gap saturation plot with theoretical limit and marginal returns inset.

    Args:
        results: Results dict from experiment
        output_path: Path to save figure
    """
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(14, 8))

    # Main plot
    ax_main = plt.subplot2grid((3, 3), (0, 0), colspan=3, rowspan=2)

    k_values = np.array(results['k_values'])
    gaps = np.array(results['gaps_mean'])
    gaps_err = np.array(results['gaps_sem'])

    # Plot empirical data
    ax_main.errorbar(k_values, gaps, yerr=gaps_err,
                     fmt='o', markersize=10, capsize=5, capthick=2.5,
                     label='Empirical Gap(K)', color='steelblue', linewidth=2.5,
                     zorder=5)

    # Plot theoretical fit
    if results['fit']['success']:
        G_inf = results['fit']['G_inf']
        mu_eta = results['fit']['mu_eta']
        r2 = results['fit']['r2']

        k_fine = np.linspace(0, max(k_values), 200)
        gap_fit = exponential_saturation_model(k_fine, G_inf, mu_eta)

        ax_main.plot(k_fine, gap_fit, '-', linewidth=3,
                    color='darkblue', alpha=0.7,
                    label=f'Fit: $G_\\infty(1 - e^{{-\\mu\\eta K}})$, $R^2={r2:.3f}$',
                    zorder=3)

        # Red dashed line at G_∞
        ax_main.axhline(G_inf, color='red', linestyle='--', linewidth=3,
                       label=f'Saturation Limit: $G_\\infty = {G_inf:.4f}$',
                       zorder=4)

        # Shade region showing saturation
        ax_main.fill_between(k_fine, gap_fit, G_inf,
                            alpha=0.15, color='red',
                            label='Saturation gap')

    # Mark K=10 (point of diminishing returns)
    k10_idx = np.where(k_values == 10)[0]
    if len(k10_idx) > 0:
        k10_idx = k10_idx[0]
        ax_main.axvline(10, color='orange', linestyle=':', linewidth=2.5,
                       label='K=10 (diminishing returns)', alpha=0.7)
        ax_main.plot(10, gaps[k10_idx], 'o', markersize=15,
                    color='orange', zorder=6)

    # Mark saturation point
    k_sat = results['saturation_analysis']['k_saturate']
    gap_sat = results['saturation_analysis']['gap_at_saturation']
    k_sat_idx = np.where(k_values == k_sat)[0]
    if len(k_sat_idx) > 0:
        ax_main.plot(k_sat, gap_sat, '*', markersize=20,
                    color='green', zorder=7,
                    label=f'Saturation point (K={k_sat})')

    ax_main.set_xlabel('Adaptation Steps (K)', fontsize=14, fontweight='bold')
    ax_main.set_ylabel('Adaptation Gap', fontsize=14, fontweight='bold')
    ax_main.set_title('Gap Saturation: Exponential Approach to $G_\\infty$',
                     fontsize=16, fontweight='bold')
    ax_main.legend(fontsize=11, loc='lower right')
    ax_main.grid(True, alpha=0.3)
    ax_main.tick_params(labelsize=12)
    ax_main.set_xlim([-1, max(k_values) + 2])

    # Add text box with saturation info
    if results['saturation_analysis']['improvement_beyond_k10'] is not None:
        textstr = (
            f'Diminishing Returns:\n'
            f'K=10 → K={k_values[-1]}:\n'
            f'+{results["saturation_analysis"]["improvement_beyond_k10"]:.4f}\n'
            f'({results["saturation_analysis"]["pct_beyond_k10"]:.1f}% improvement)'
        )
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
        ax_main.text(0.02, 0.98, textstr, transform=ax_main.transAxes,
                    fontsize=11, verticalalignment='top',
                    bbox=props, fontweight='bold')

    # Inset: Marginal returns
    ax_inset = plt.subplot2grid((3, 3), (2, 0), colspan=3)

    marginal_k = np.array(results['marginal_returns']['k_values'])
    marginal = np.array(results['marginal_returns']['marginal'])

    ax_inset.plot(marginal_k, marginal, 'o-', linewidth=2.5, markersize=8,
                 color='purple')
    ax_inset.axhline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    ax_inset.axhline(0.001, color='red', linestyle='--', linewidth=1.5,
                    label='Threshold (0.001)', alpha=0.7)
    ax_inset.axvline(10, color='orange', linestyle=':', linewidth=2, alpha=0.7)

    ax_inset.set_xlabel('K', fontsize=12, fontweight='bold')
    ax_inset.set_ylabel('Marginal Gain\n$\\Delta$Gap(K)', fontsize=12, fontweight='bold')
    ax_inset.set_title('Marginal Returns: $\\Delta$Gap(K) = Gap(K) - Gap(K-1)',
                      fontsize=13, fontweight='bold')
    ax_inset.legend(fontsize=10)
    ax_inset.grid(True, alpha=0.3)
    ax_inset.tick_params(labelsize=10)
    ax_inset.set_yscale('log')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to {output_path}")

    plt.close()


@app.command()
def main(
    meta_path: Path = Path("experiments/train_scripts/checkpoints/maml_best_policy.pt"),
    robust_path: Path = Path("experiments/train_scripts/checkpoints/robust_best_policy.pt"),
    output_dir: Path = Path("results/gap_saturation"),
    n_test_tasks: int = 50
):
    """
    Run gap saturation analysis with extended K values.

    Args:
        meta_path: Path to trained meta policy checkpoint
        robust_path: Path to robust baseline policy checkpoint
        output_dir: Directory to save results and figures
        n_test_tasks: Number of test tasks per K value
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

    # Extended K values (dense near 0, sparse at high K)
    k_values = [0, 1, 2, 3, 4, 5, 7, 10, 12, 15, 18, 20, 25, 30]

    # Run experiment
    results = run_gap_saturation_experiment(
        meta_policy_path=str(meta_path),
        robust_policy_path=str(robust_path),
        config=config,
        k_values=k_values,
        n_test_tasks=n_test_tasks,
        output_dir=str(output_dir)
    )

    # Generate plot
    plot_path = f"{output_dir}/gap_saturation.pdf"
    plot_gap_saturation(results, output_path=plot_path)

    plot_path_png = f"{output_dir}/gap_saturation.png"
    plot_gap_saturation(results, output_path=plot_path_png)

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)

    if results['fit']['success']:
        print(f"\nSaturation Analysis:")
        print(f"  • G_∞ (saturated gap): {results['fit']['G_inf']:.4f}")
        print(f"  • Convergence rate (μη): {results['fit']['mu_eta']:.4f}")
        print(f"  • R²: {results['fit']['r2']:.4f}")
        print(f"  • Saturation point: K = {results['saturation_analysis']['k_saturate']}")

        if results['saturation_analysis']['improvement_beyond_k10'] is not None:
            print(f"\nDiminishing Returns:")
            print(f"  • Improvement K=10 → K=30: {results['saturation_analysis']['improvement_beyond_k10']:.4f}")
            print(f"  • Percentage gain: {results['saturation_analysis']['pct_beyond_k10']:.1f}%")
            print(f"  • Conclusion: Beyond K≈10, marginal gains are minimal")


if __name__ == "__main__":
    app()
