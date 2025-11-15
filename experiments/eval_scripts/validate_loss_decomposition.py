"""
Validation Script B: Loss Decomposition vs Control-Relevant Variance (σ²_S)

This script decomposes the adaptation gap by showing how different policies
perform as task variance increases:

1. L(π_rob): Robust policy loss (increases with variance)
2. L(π₀): Meta-initialization loss (slight increase with variance)
3. L(π_K): Adapted policy loss (stays near optimal L*)

The gap emerges as the vertical distance between robust and adapted policies.

Theoretical prediction:
- Robust policy suffers as variance increases (must handle all tasks)
- Meta-initialization is reasonably good (learned good starting point)
- Adapted policy stays near optimal (adapts to each task)

Expected output: Three curves showing loss decomposition vs σ²_S
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
from metaqctrl.theory.physics_constants import compute_control_relevant_variance
from metaqctrl.utils.checkpoint_utils import load_policy_from_checkpoint

app = Typer()


def compute_loss_decomposition(
    meta_policy_path: str,
    robust_policy_path: str,
    config: Dict,
    task_distribution: TaskDistribution,
    K_adapt: int = 5,
    n_test_tasks: int = 40
) -> Dict:
    """
    Compute loss decomposition: L(π_rob), L(π₀), L(π_K).

    Args:
        meta_policy_path: Path to meta policy
        robust_policy_path: Path to robust policy
        config: Configuration dict
        task_distribution: TaskDistribution to test
        K_adapt: Number of adaptation steps
        n_test_tasks: Number of test tasks

    Returns:
        results: Dict with mean losses and standard errors
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

    losses_robust = []
    losses_meta_init = []
    losses_meta_adapted = []

    for task in tqdm(test_tasks, desc="Evaluating tasks", leave=False):
        task_features = torch.tensor(
            [task.alpha, task.A, task.omega_c],
            dtype=torch.float32
        )

        # 1. L(π_rob): Robust policy (no adaptation)
        robust_policy.eval()
        with torch.no_grad():
            controls_robust = robust_policy(task_features).detach().numpy()
        fid_robust = env.compute_fidelity(controls_robust, task)
        loss_robust = 1.0 - fid_robust  # Convert to loss (infidelity)
        losses_robust.append(loss_robust)

        # 2. L(π₀): Meta-initialization (K=0, no adaptation)
        meta_policy_template.eval()
        with torch.no_grad():
            controls_init = meta_policy_template(task_features).detach().numpy()
        fid_init = env.compute_fidelity(controls_init, task)
        loss_init = 1.0 - fid_init
        losses_meta_init.append(loss_init)

        # 3. L(π_K): Adapted policy (K adaptation steps)
        adapted_policy = copy.deepcopy(meta_policy_template)
        adapted_policy.train()

        for k in range(K_adapt):
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
            controls_adapted = adapted_policy(task_features).detach().numpy()
        fid_adapted = env.compute_fidelity(controls_adapted, task)
        loss_adapted = 1.0 - fid_adapted
        losses_meta_adapted.append(loss_adapted)

    # Compute statistics
    losses_robust = np.array(losses_robust)
    losses_meta_init = np.array(losses_meta_init)
    losses_meta_adapted = np.array(losses_meta_adapted)

    return {
        'L_robust_mean': np.mean(losses_robust),
        'L_robust_std': np.std(losses_robust),
        'L_robust_sem': np.std(losses_robust) / np.sqrt(n_test_tasks),
        'L_meta_init_mean': np.mean(losses_meta_init),
        'L_meta_init_std': np.std(losses_meta_init),
        'L_meta_init_sem': np.std(losses_meta_init) / np.sqrt(n_test_tasks),
        'L_meta_adapted_mean': np.mean(losses_meta_adapted),
        'L_meta_adapted_std': np.std(losses_meta_adapted),
        'L_meta_adapted_sem': np.std(losses_meta_adapted) / np.sqrt(n_test_tasks),
    }


def run_loss_decomposition_experiment(
    meta_policy_path: str,
    robust_policy_path: str,
    config: Dict,
    variance_multipliers: List[float] = None,
    K_adapt: int = 5,
    n_test_tasks: int = 40,
    output_dir: str = "results/loss_decomposition"
) -> Dict:
    """
    Main experiment: Loss decomposition vs control-relevant variance.

    Args:
        meta_policy_path: Path to trained meta policy
        robust_policy_path: Path to robust baseline policy
        config: Experiment configuration
        variance_multipliers: Multipliers for base task variance
        K_adapt: Number of adaptation steps for L(π_K)
        n_test_tasks: Number of test tasks per variance level
        output_dir: Output directory

    Returns:
        results: Dict with loss decomposition data
    """
    print("=" * 80)
    print("EXPERIMENT: Loss Decomposition vs Control-Relevant Variance (σ²_S)")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)

    # Default variance multipliers
    if variance_multipliers is None:
        variance_multipliers = [0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

    print(f"\nExperiment parameters:")
    print(f"  Variance multipliers: {variance_multipliers}")
    print(f"  K_adapt: {K_adapt}")
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

    # Compute losses for each variance level
    print("\n[2/3] Computing loss decomposition for each variance level...")

    control_variances = []
    all_losses = []

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
        sample_tasks = task_dist.sample(100)
        sigma_S_sq = compute_control_relevant_variance(env, sample_tasks)
        control_variances.append(sigma_S_sq)
        print(f"σ²_S = {sigma_S_sq:.6f}")

        # Compute loss decomposition
        print(f"  Computing loss decomposition...")
        losses = compute_loss_decomposition(
            meta_policy_path=meta_policy_path,
            robust_policy_path=robust_policy_path,
            config=config,
            task_distribution=task_dist,
            K_adapt=K_adapt,
            n_test_tasks=n_test_tasks
        )
        all_losses.append(losses)

        print(f"    L(π_rob) = {losses['L_robust_mean']:.4f} ± {losses['L_robust_sem']:.4f}")
        print(f"    L(π₀)    = {losses['L_meta_init_mean']:.4f} ± {losses['L_meta_init_sem']:.4f}")
        print(f"    L(π_K)   = {losses['L_meta_adapted_mean']:.4f} ± {losses['L_meta_adapted_sem']:.4f}")
        print(f"    Gap      = {losses['L_robust_mean'] - losses['L_meta_adapted_mean']:.4f}")

    # Organize results
    print("\n[3/3] Organizing results...")

    results = {
        'variance_multipliers': variance_multipliers,
        'control_variances': control_variances,
        'K_adapt': K_adapt,
        'n_test_tasks': n_test_tasks,
        'losses': all_losses,
        'config': config
    }

    # Save results
    print(f"\nSaving results to {output_dir}...")
    results_path = f"{output_dir}/results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    return results


def plot_loss_decomposition(
    results: Dict,
    output_path: str = None
):
    """
    Generate loss decomposition plot.

    Args:
        results: Results dict from experiment
        output_path: Path to save figure
    """
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 8))

    sigma_S_sq = np.array(results['control_variances'])
    losses = results['losses']

    # Extract loss arrays
    L_robust = np.array([l['L_robust_mean'] for l in losses])
    L_robust_err = np.array([l['L_robust_sem'] for l in losses])

    L_init = np.array([l['L_meta_init_mean'] for l in losses])
    L_init_err = np.array([l['L_meta_init_sem'] for l in losses])

    L_adapted = np.array([l['L_meta_adapted_mean'] for l in losses])
    L_adapted_err = np.array([l['L_meta_adapted_sem'] for l in losses])

    # Plot three curves
    ax.errorbar(sigma_S_sq, L_robust, yerr=L_robust_err,
                fmt='s-', markersize=10, capsize=5, capthick=2,
                label=r'$L(\pi_{\mathrm{rob}})$ (Robust baseline)',
                color='orangered', linewidth=2.5, alpha=0.9)

    ax.errorbar(sigma_S_sq, L_init, yerr=L_init_err,
                fmt='o-', markersize=10, capsize=5, capthick=2,
                label=r'$L(\pi_0)$ (Meta-init, K=0)',
                color='steelblue', linewidth=2.5, alpha=0.9)

    ax.errorbar(sigma_S_sq, L_adapted, yerr=L_adapted_err,
                fmt='^-', markersize=10, capsize=5, capthick=2,
                label=rf'$L(\pi_K)$ (Adapted, K={results["K_adapt"]})',
                color='green', linewidth=2.5, alpha=0.9)

    # Shade gap region between robust and adapted
    ax.fill_between(sigma_S_sq, L_robust, L_adapted,
                    alpha=0.2, color='purple',
                    label='Adaptation Gap')

    # Add arrows showing gap at a few points
    arrow_indices = [0, len(sigma_S_sq)//2, -1]
    for idx in arrow_indices:
        x = sigma_S_sq[idx]
        y_rob = L_robust[idx]
        y_adapt = L_adapted[idx]
        gap = y_rob - y_adapt

        # Only draw if gap is significant
        if gap > 0.01:
            ax.annotate('', xy=(x, y_adapt), xytext=(x, y_rob),
                       arrowprops=dict(arrowstyle='<->', color='purple',
                                     lw=2, shrinkA=0, shrinkB=0))
            ax.text(x * 1.1, (y_rob + y_adapt) / 2,
                   f'{gap:.3f}',
                   fontsize=10, color='purple', fontweight='bold')

    ax.set_xlabel(r'Control-Relevant Variance ($\sigma^2_S$)',
                 fontsize=15, fontweight='bold')
    ax.set_ylabel('Loss (Infidelity)',
                 fontsize=15, fontweight='bold')
    ax.set_title(f'Loss Decomposition vs Task Variance\n'
                 f'Gap emerges as vertical distance between robust and adapted',
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=13, loc='upper left', framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=13)

    # Set y-axis to start from 0
    ax.set_ylim(bottom=0)

    # Add text box with insights
    textstr = (
        'Observations:\n'
        '• Robust loss increases with variance\n'
        '• Meta-init stays moderate\n'
        '• Adapted loss stays near optimal\n'
        '• Gap grows with variance'
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.98, 0.98, textstr, transform=ax.transAxes,
           fontsize=11, verticalalignment='top', horizontalalignment='right',
           bbox=props)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to {output_path}")

    plt.close()


def plot_loss_decomposition_with_fidelity(
    results: Dict,
    output_path: str = None
):
    """
    Generate dual-axis plot showing both loss and fidelity.

    Args:
        results: Results dict from experiment
        output_path: Path to save figure
    """
    sns.set_style("whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    sigma_S_sq = np.array(results['control_variances'])
    losses = results['losses']

    # Extract arrays
    L_robust = np.array([l['L_robust_mean'] for l in losses])
    L_robust_err = np.array([l['L_robust_sem'] for l in losses])
    L_init = np.array([l['L_meta_init_mean'] for l in losses])
    L_init_err = np.array([l['L_meta_init_sem'] for l in losses])
    L_adapted = np.array([l['L_meta_adapted_mean'] for l in losses])
    L_adapted_err = np.array([l['L_meta_adapted_sem'] for l in losses])

    # Convert to fidelities
    F_robust = 1.0 - L_robust
    F_robust_err = L_robust_err  # Same error bars
    F_init = 1.0 - L_init
    F_init_err = L_init_err
    F_adapted = 1.0 - L_adapted
    F_adapted_err = L_adapted_err

    # Plot 1: Loss space
    ax1.errorbar(sigma_S_sq, L_robust, yerr=L_robust_err,
                fmt='s-', markersize=8, capsize=4,
                label=r'$L(\pi_{\mathrm{rob}})$',
                color='orangered', linewidth=2)
    ax1.errorbar(sigma_S_sq, L_init, yerr=L_init_err,
                fmt='o-', markersize=8, capsize=4,
                label=r'$L(\pi_0)$',
                color='steelblue', linewidth=2)
    ax1.errorbar(sigma_S_sq, L_adapted, yerr=L_adapted_err,
                fmt='^-', markersize=8, capsize=4,
                label=rf'$L(\pi_K)$ (K={results["K_adapt"]})',
                color='green', linewidth=2)
    ax1.fill_between(sigma_S_sq, L_robust, L_adapted,
                    alpha=0.2, color='purple')

    ax1.set_xlabel(r'$\sigma^2_S$', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Loss (Infidelity)', fontsize=13, fontweight='bold')
    ax1.set_title('(a) Loss Space', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # Plot 2: Fidelity space
    ax2.errorbar(sigma_S_sq, F_robust, yerr=F_robust_err,
                fmt='s-', markersize=8, capsize=4,
                label=r'$F(\pi_{\mathrm{rob}})$',
                color='orangered', linewidth=2)
    ax2.errorbar(sigma_S_sq, F_init, yerr=F_init_err,
                fmt='o-', markersize=8, capsize=4,
                label=r'$F(\pi_0)$',
                color='steelblue', linewidth=2)
    ax2.errorbar(sigma_S_sq, F_adapted, yerr=F_adapted_err,
                fmt='^-', markersize=8, capsize=4,
                label=rf'$F(\pi_K)$ (K={results["K_adapt"]})',
                color='green', linewidth=2)
    ax2.fill_between(sigma_S_sq, F_robust, F_adapted,
                    alpha=0.2, color='purple')

    ax2.set_xlabel(r'$\sigma^2_S$', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Fidelity', fontsize=13, fontweight='bold')
    ax2.set_title('(b) Fidelity Space', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])

    plt.suptitle('Loss Decomposition: Two Perspectives',
                fontsize=16, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nDual-axis figure saved to {output_path}")

    plt.close()


@app.command()
def main(
    meta_path: Path = Path("experiments/train_scripts/checkpoints/maml_best_policy.pt"),
    robust_path: Path = Path("experiments/train_scripts/checkpoints/robust_best_policy.pt"),
    output_dir: Path = Path("results/loss_decomposition"),
    k_adapt: int = 5,
    n_test_tasks: int = 40
):
    """
    Run loss decomposition vs control-relevant variance experiment.

    Args:
        meta_path: Path to trained meta policy checkpoint
        robust_path: Path to robust baseline policy checkpoint
        output_dir: Directory to save results and figures
        k_adapt: Number of adaptation steps for L(π_K)
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

    # Variance multipliers
    variance_multipliers = [0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

    # Run experiment
    results = run_loss_decomposition_experiment(
        meta_policy_path=str(meta_path),
        robust_policy_path=str(robust_path),
        config=config,
        variance_multipliers=variance_multipliers,
        K_adapt=k_adapt,
        n_test_tasks=n_test_tasks,
        output_dir=str(output_dir)
    )

    # Generate plots
    plot_path = f"{output_dir}/loss_decomposition.pdf"
    plot_loss_decomposition(results, output_path=plot_path)

    plot_path_png = f"{output_dir}/loss_decomposition.png"
    plot_loss_decomposition(results, output_path=plot_path_png)

    # Also generate dual-axis plot
    dual_path = f"{output_dir}/loss_decomposition_dual.pdf"
    plot_loss_decomposition_with_fidelity(results, output_path=dual_path)

    dual_path_png = f"{output_dir}/loss_decomposition_dual.png"
    plot_loss_decomposition_with_fidelity(results, output_path=dual_path_png)

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    app()
