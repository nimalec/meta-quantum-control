"""
Validation Script A: Saturated Gap vs Control-Relevant Variance (σ²_S)

This script validates the theoretical prediction:
    Gap(∞) ∝ σ²_S

Where:
- Gap(∞) is the saturated adaptation gap (at large K)
- σ²_S is the control-relevant variance of the task distribution

Generates:
- Log-log plot showing power-law relationship (slope ≈ 1)
- Inset with linear-linear plot showing R² > 0.89
- Validates that gap scales linearly with task variance

Expected result: Linear fit in log-log space with slope ≈ 1.0, R² > 0.89
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
from metaqctrl.theory.physics_constants import compute_control_relevant_variance
from metaqctrl.utils.checkpoint_utils import load_policy_from_checkpoint

app = Typer()


def compute_saturated_gap(
    meta_policy_path: str,
    robust_policy_path: str,
    config: Dict,
    task_distribution: TaskDistribution,
    K_saturate: int = 30,
    n_test_tasks: int = 40
) -> Tuple[float, float, float]:
    """
    Compute saturated gap Gap(∞) ≈ Gap(K_saturate) for a task distribution.

    Args:
        meta_policy_path: Path to meta policy
        robust_policy_path: Path to robust policy
        config: Configuration dict
        task_distribution: TaskDistribution to test
        K_saturate: Large K to approximate saturation (default 30)
        n_test_tasks: Number of test tasks

    Returns:
        gap_mean: Mean gap across tasks
        gap_std: Standard error
        gap_sem: Standard error of mean
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

    inner_lr = config.get('inner_lr', 0.01)

    fidelities_meta = []
    fidelities_robust = []

    for task in tqdm(test_tasks, desc="Evaluating tasks", leave=False):
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

        # Meta policy with K_saturate steps (approximate K → ∞)
        adapted_policy = copy.deepcopy(meta_policy_template)
        adapted_policy.train()

        for k in range(K_saturate):
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

    return gap_mean, gap_std, gap_sem


def run_gap_vs_control_variance_experiment(
    meta_policy_path: str,
    robust_policy_path: str,
    config: Dict,
    variance_multipliers: List[float] = None,
    K_saturate: int = 30,
    n_test_tasks: int = 40,
    output_dir: str = "results/gap_vs_control_variance"
) -> Dict:
    """
    Main experiment: Saturated gap vs control-relevant variance.

    Args:
        meta_policy_path: Path to trained meta policy
        robust_policy_path: Path to robust baseline policy
        config: Experiment configuration
        variance_multipliers: Multipliers for base task variance
        K_saturate: Large K to approximate saturation
        n_test_tasks: Number of test tasks per variance level
        output_dir: Output directory

    Returns:
        results: Dict with gaps, variances, and fit parameters
    """
    print("=" * 80)
    print("EXPERIMENT: Saturated Gap vs Control-Relevant Variance (σ²_S)")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)

    # Default variance multipliers
    if variance_multipliers is None:
        # Wide range to see power law
        variance_multipliers = [0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

    print(f"\nExperiment parameters:")
    print(f"  Variance multipliers: {variance_multipliers}")
    print(f"  K_saturate: {K_saturate}")
    print(f"  Test tasks per variance: {n_test_tasks}")

    # Create environment for computing σ²_S
    print("\n[1/3] Setting up environment...")
    target_state = get_target_state_from_config(config)
    env = create_quantum_environment(config, target_state)

    # Base task distribution ranges
    base_alpha = config.get('alpha_range', [0.5, 2.0])
    base_A = config.get('A_range', [0.05, 0.3])
    base_omega = config.get('omega_c_range', [2.0, 8.0])

    alpha_center = np.mean(base_alpha)
    A_center = np.mean(base_A)
    omega_center = np.mean(base_omega)

    # Compute gaps and variances
    print("\n[2/3] Computing saturated gaps for each variance level...")

    control_variances = []
    gaps_mean = []
    gaps_std = []
    gaps_sem = []

    for i, var_mult in enumerate(variance_multipliers):
        print(f"\n[{i+1}/{len(variance_multipliers)}] Variance multiplier: {var_mult}")

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

        # Create task distribution
        task_dist = TaskDistribution(
            dist_type='uniform',
            ranges={
                'alpha': tuple(alpha_range),
                'A': tuple(A_range),
                'omega_c': tuple(omega_range)
            }
        )

        # Compute control-relevant variance σ²_S
        print("  Computing σ²_S...", end=' ')
        sample_tasks = task_dist.sample(100)  # Sample for variance computation
        sigma_S_sq = compute_control_relevant_variance(env, sample_tasks)
        control_variances.append(sigma_S_sq)
        print(f"σ²_S = {sigma_S_sq:.6f}")

        # Compute saturated gap
        print(f"  Computing Gap(K={K_saturate})...", end=' ')
        gap_mean, gap_std, gap_sem = compute_saturated_gap(
            meta_policy_path=meta_policy_path,
            robust_policy_path=robust_policy_path,
            config=config,
            task_distribution=task_dist,
            K_saturate=K_saturate,
            n_test_tasks=n_test_tasks
        )
        gaps_mean.append(gap_mean)
        gaps_std.append(gap_std)
        gaps_sem.append(gap_sem)
        print(f"Gap = {gap_mean:.4f} ± {gap_sem:.4f}")

    # Fit power law in log-log space
    print("\n[3/3] Fitting power law...")

    sigma_S_sq_array = np.array(control_variances)
    gap_array = np.array(gaps_mean)

    # Remove any zero or negative values (shouldn't happen, but be safe)
    valid_mask = (sigma_S_sq_array > 0) & (gap_array > 0)
    sigma_S_sq_valid = sigma_S_sq_array[valid_mask]
    gap_valid = gap_array[valid_mask]

    if len(sigma_S_sq_valid) < 2:
        print("ERROR: Not enough valid data points for fitting")
        return None

    # Log-log linear fit: log(Gap) = log(C) + β log(σ²_S)
    log_sigma = np.log(sigma_S_sq_valid)
    log_gap = np.log(gap_valid)

    # Linear fit in log space
    coeffs = np.polyfit(log_sigma, log_gap, 1)
    slope, intercept = coeffs
    C = np.exp(intercept)

    # Predicted values
    log_gap_pred = np.polyval(coeffs, log_sigma)
    r2_loglog = r2_score(log_gap, log_gap_pred)

    print(f"\n  Log-log fit: log(Gap) = {intercept:.3f} + {slope:.3f} log(σ²_S)")
    print(f"  Power law: Gap = {C:.4f} * (σ²_S)^{slope:.3f}")
    print(f"  R² (log-log): {r2_loglog:.4f}")
    print(f"  Expected slope: ≈ 1.0, Actual slope: {slope:.3f}")

    # Also compute R² in linear space for inset
    gap_pred_linear = C * (sigma_S_sq_valid ** slope)
    r2_linear = r2_score(gap_valid, gap_pred_linear)
    print(f"  R² (linear): {r2_linear:.4f}")

    # Save results
    print(f"\nSaving results to {output_dir}...")
    results = {
        'variance_multipliers': variance_multipliers,
        'control_variances': control_variances,
        'gaps_mean': gaps_mean,
        'gaps_std': gaps_std,
        'gaps_sem': gaps_sem,
        'K_saturate': K_saturate,
        'n_test_tasks': n_test_tasks,
        'fit_loglog': {
            'slope': float(slope),
            'intercept': float(intercept),
            'C': float(C),
            'r2': float(r2_loglog)
        },
        'fit_linear': {
            'r2': float(r2_linear)
        },
        'config': config
    }

    results_path = f"{output_dir}/results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    return results


def plot_gap_vs_control_variance(
    results: Dict,
    output_path: str = None
):
    """
    Generate log-log plot with linear inset.

    Args:
        results: Results dict from experiment
        output_path: Path to save figure
    """
    sns.set_style("whitegrid")

    # Main figure with inset
    fig, ax_main = plt.subplots(figsize=(10, 8))

    sigma_S_sq = np.array(results['control_variances'])
    gaps = np.array(results['gaps_mean'])
    gaps_err = np.array(results['gaps_sem'])

    # Remove any invalid values
    valid_mask = (sigma_S_sq > 0) & (gaps > 0)
    sigma_S_sq = sigma_S_sq[valid_mask]
    gaps = gaps[valid_mask]
    gaps_err = gaps_err[valid_mask]

    # Main plot: Log-log
    ax_main.errorbar(sigma_S_sq, gaps, yerr=gaps_err,
                     fmt='o', markersize=12, capsize=6, capthick=2.5,
                     label='Empirical Gap(∞)', color='steelblue', linewidth=2.5)

    # Fit line
    slope = results['fit_loglog']['slope']
    C = results['fit_loglog']['C']
    r2_loglog = results['fit_loglog']['r2']

    sigma_fit = np.logspace(np.log10(sigma_S_sq.min()), np.log10(sigma_S_sq.max()), 100)
    gap_fit = C * (sigma_fit ** slope)

    ax_main.plot(sigma_fit, gap_fit, '--',
                 linewidth=3, color='darkred',
                 label=f'Power law: Gap ∝ (σ²_S)^{slope:.2f}\n$R^2$ = {r2_loglog:.3f}')

    # Reference line with slope 1
    gap_ref = gaps[len(gaps)//2] * (sigma_fit / sigma_S_sq[len(sigma_S_sq)//2])
    ax_main.plot(sigma_fit, gap_ref, ':',
                linewidth=2, color='gray', alpha=0.5,
                label='Reference: slope = 1.0')

    ax_main.set_xlabel(r'Control-Relevant Variance ($\sigma^2_S$)',
                      fontsize=15, fontweight='bold')
    ax_main.set_ylabel(r'Saturated Gap Gap($\infty$)',
                      fontsize=15, fontweight='bold')
    ax_main.set_title(f'Power-Law Scaling: Gap(∞) vs σ²_S (K={results["K_saturate"]})',
                     fontsize=16, fontweight='bold')
    ax_main.legend(fontsize=13, loc='upper left')
    ax_main.set_xscale('log')
    ax_main.set_yscale('log')
    ax_main.grid(True, alpha=0.3, which='both')
    ax_main.tick_params(labelsize=13)

    # Add text box with slope
    textstr = f'Slope = {slope:.3f}\nExpected = 1.0'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax_main.text(0.95, 0.05, textstr, transform=ax_main.transAxes,
                fontsize=12, verticalalignment='bottom', horizontalalignment='right',
                bbox=props)

    # Inset: Linear-linear plot
    ax_inset = fig.add_axes([0.55, 0.2, 0.3, 0.25])  # [left, bottom, width, height]

    ax_inset.errorbar(sigma_S_sq, gaps, yerr=gaps_err,
                     fmt='o', markersize=6, capsize=3,
                     color='steelblue', linewidth=1.5)

    gap_pred_linear = C * (sigma_S_sq ** slope)
    ax_inset.plot(sigma_S_sq, gap_pred_linear, '--',
                 linewidth=2, color='darkred')

    r2_linear = results['fit_linear']['r2']
    ax_inset.set_xlabel(r'$\sigma^2_S$', fontsize=10, fontweight='bold')
    ax_inset.set_ylabel('Gap(∞)', fontsize=10, fontweight='bold')
    ax_inset.set_title(f'Linear Scale (R²={r2_linear:.3f})',
                      fontsize=10, fontweight='bold')
    ax_inset.grid(True, alpha=0.3)
    ax_inset.tick_params(labelsize=9)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to {output_path}")

    plt.close()


@app.command()
def main(
    meta_path: Path = Path("experiments/train_scripts/checkpoints/maml_best_policy.pt"),
    robust_path: Path = Path("experiments/train_scripts/checkpoints/robust_best_policy.pt"),
    output_dir: Path = Path("results/gap_vs_control_variance"),
    k_saturate: int = 30,
    n_test_tasks: int = 40
):
    """
    Run saturated gap vs control-relevant variance experiment.

    Args:
        meta_path: Path to trained meta policy checkpoint
        robust_path: Path to robust baseline policy checkpoint
        output_dir: Directory to save results and figures
        k_saturate: Large K to approximate saturation (default 30)
        n_test_tasks: Number of test tasks per variance level
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

    # Variance multipliers (wide range to see power law)
    variance_multipliers = [0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

    # Run experiment
    results = run_gap_vs_control_variance_experiment(
        meta_policy_path=str(meta_path),
        robust_policy_path=str(robust_path),
        config=config,
        variance_multipliers=variance_multipliers,
        K_saturate=k_saturate,
        n_test_tasks=n_test_tasks,
        output_dir=str(output_dir)
    )

    if results is not None:
        # Generate plot
        plot_path = f"{output_dir}/gap_vs_control_variance.pdf"
        plot_gap_vs_control_variance(results, output_path=plot_path)

        plot_path_png = f"{output_dir}/gap_vs_control_variance.png"
        plot_gap_vs_control_variance(results, output_path=plot_path_png)

        print("\n" + "=" * 80)
        print("VALIDATION COMPLETE")
        print(f"Expected: slope ≈ 1.0, R² > 0.89")
        print(f"Actual: slope = {results['fit_loglog']['slope']:.3f}, R² = {results['fit_loglog']['r2']:.3f}")
        print("=" * 80)


if __name__ == "__main__":
    app()
