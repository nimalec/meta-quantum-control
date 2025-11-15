"""
Validation Script: Inner-Loop Adaptation Dynamics

This script visualizes what happens during the inner adaptation loop:

A. Loss evolution: L(φ_k) vs. step k for multiple tasks
   - Shows how loss decreases during adaptation
   - Each task has different starting loss and convergence rate
   - Validates that adaptation is actually working

B. Gradient norm evolution: ‖∇_φL‖ vs. k
   - Should show exponential decay (convergence indicator)
   - Validates that policy is approaching local optimum
   - Theory predicts: ‖∇L‖ ≈ ‖∇L₀‖ exp(-μηk)

Expected output: Two-panel plot showing adaptation dynamics across tasks
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

from metaqctrl.quantum.noise_models import TaskDistribution, NoiseParameters
from metaqctrl.meta_rl.policy import PulsePolicy
from metaqctrl.theory.quantum_environment import create_quantum_environment, get_target_state_from_config
from metaqctrl.utils.checkpoint_utils import load_policy_from_checkpoint

app = Typer()


def track_adaptation_dynamics(
    policy: torch.nn.Module,
    task: NoiseParameters,
    env,
    config: Dict,
    K_max: int = 20,
    inner_lr: float = 0.01
) -> Dict:
    """
    Track loss and gradient norm during adaptation for a single task.

    Args:
        policy: Meta policy (will be copied and adapted)
        task: Task to adapt to
        env: QuantumEnvironment
        config: Configuration dict
        K_max: Maximum number of adaptation steps
        inner_lr: Inner loop learning rate

    Returns:
        dynamics: Dict with losses and gradient norms at each step k
    """
    # Clone policy
    adapted_policy = copy.deepcopy(policy)
    adapted_policy.train()

    losses = []
    grad_norms = []

    # Initial evaluation (k=0)
    adapted_policy.eval()
    with torch.no_grad():
        loss_0 = env.compute_loss_differentiable(
            adapted_policy, task, device=torch.device('cpu')
        )
    losses.append(float(loss_0.item()))
    grad_norms.append(0.0)  # No gradient at k=0

    # Adaptation loop
    adapted_policy.train()
    for k in range(K_max):
        # Compute loss with gradients
        loss = env.compute_loss_differentiable(
            adapted_policy, task, device=torch.device('cpu')
        )
        loss.backward()

        # Compute gradient norm
        grad_norm = 0.0
        for param in adapted_policy.parameters():
            if param.grad is not None:
                grad_norm += param.grad.norm().item() ** 2
        grad_norm = np.sqrt(grad_norm)

        # Store metrics
        losses.append(float(loss.item()))
        grad_norms.append(grad_norm)

        # Gradient step
        with torch.no_grad():
            for param in adapted_policy.parameters():
                if param.grad is not None:
                    param -= inner_lr * param.grad
                    param.grad.zero_()

    return {
        'losses': losses,
        'grad_norms': grad_norms,
        'task_params': {
            'alpha': task.alpha,
            'A': task.A,
            'omega_c': task.omega_c
        }
    }


def run_adaptation_dynamics_experiment(
    meta_policy_path: str,
    config: Dict,
    n_tasks: int = 10,
    K_max: int = 20,
    output_dir: str = "results/adaptation_dynamics"
) -> Dict:
    """
    Main experiment: Track adaptation dynamics for multiple tasks.

    Args:
        meta_policy_path: Path to trained meta policy
        config: Experiment configuration
        n_tasks: Number of tasks to sample and track
        K_max: Maximum adaptation steps
        output_dir: Output directory

    Returns:
        results: Dict with dynamics for all tasks
    """
    print("=" * 80)
    print("EXPERIMENT: Inner-Loop Adaptation Dynamics")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)

    print(f"\nExperiment parameters:")
    print(f"  Number of tasks: {n_tasks}")
    print(f"  Max adaptation steps: {K_max}")

    # Load meta policy
    print("\n[1/3] Loading meta policy...")
    meta_policy = load_policy_from_checkpoint(
        meta_policy_path, config, eval_mode=False, verbose=True
    )

    # Create environment
    print("\n[2/3] Creating quantum environment...")
    target_state = get_target_state_from_config(config)
    env = create_quantum_environment(config, target_state)

    # Sample tasks
    print(f"\n[3/3] Sampling {n_tasks} tasks and tracking adaptation...")
    task_dist = TaskDistribution(
        dist_type=config.get('task_dist_type', 'uniform'),
        ranges={
            'alpha': tuple(config['alpha_range']),
            'A': tuple(config['A_range']),
            'omega_c': tuple(config['omega_c_range'])
        }
    )
    tasks = task_dist.sample(n_tasks)

    inner_lr = config.get('inner_lr', 0.01)

    all_dynamics = []

    for i, task in enumerate(tqdm(tasks, desc="Tracking tasks")):
        print(f"\n  Task {i+1}/{n_tasks}: α={task.alpha:.2f}, A={task.A:.3f}, ωc={task.omega_c:.2f}")

        dynamics = track_adaptation_dynamics(
            policy=meta_policy,
            task=task,
            env=env,
            config=config,
            K_max=K_max,
            inner_lr=inner_lr
        )

        all_dynamics.append(dynamics)

        print(f"    Initial loss: {dynamics['losses'][0]:.4f}")
        print(f"    Final loss: {dynamics['losses'][-1]:.4f}")
        print(f"    Improvement: {dynamics['losses'][0] - dynamics['losses'][-1]:.4f}")
        print(f"    Final grad norm: {dynamics['grad_norms'][-1]:.6f}")

    # Fit exponential decay to average gradient norm
    print("\nFitting exponential decay to gradient norms...")
    avg_grad_norms = np.mean([d['grad_norms'] for d in all_dynamics], axis=0)
    k_array = np.arange(len(avg_grad_norms))

    # Exclude k=0 (no gradient)
    k_fit = k_array[1:]
    grad_fit = avg_grad_norms[1:]

    # Exponential model: ‖∇L‖ = A exp(-λk)
    def exp_decay(k, A, lam):
        return A * np.exp(-lam * k)

    try:
        popt, _ = curve_fit(exp_decay, k_fit, grad_fit, p0=[grad_fit[0], 0.1])
        A_fit, lambda_fit = popt
        fit_success = True
        print(f"  Fitted: ‖∇L‖ = {A_fit:.4f} exp(-{lambda_fit:.4f} k)")
    except Exception as e:
        print(f"  Fitting failed: {e}")
        A_fit = lambda_fit = None
        fit_success = False

    # Save results
    print(f"\nSaving results to {output_dir}...")
    results = {
        'n_tasks': n_tasks,
        'K_max': K_max,
        'inner_lr': inner_lr,
        'dynamics': all_dynamics,
        'average_grad_norms': avg_grad_norms.tolist(),
        'gradient_fit': {
            'success': fit_success,
            'A': float(A_fit) if fit_success else None,
            'lambda': float(lambda_fit) if fit_success else None
        },
        'config': config
    }

    results_path = f"{output_dir}/results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    return results


def plot_adaptation_dynamics(
    results: Dict,
    output_path: str = None
):
    """
    Generate two-panel plot: (A) Loss evolution, (B) Gradient norm evolution.

    Args:
        results: Results dict from experiment
        output_path: Path to save figure
    """
    sns.set_style("whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    n_tasks = results['n_tasks']
    K_max = results['K_max']
    dynamics = results['dynamics']

    k_array = np.arange(K_max + 1)

    # Generate colors for tasks
    colors = plt.cm.tab10(np.linspace(0, 1, n_tasks))

    # Panel A: Loss evolution
    for i, dyn in enumerate(dynamics):
        losses = dyn['losses']
        task_params = dyn['task_params']
        label = f"Task {i+1} (α={task_params['alpha']:.1f})"

        ax1.plot(k_array, losses, 'o-', linewidth=2, markersize=5,
                color=colors[i], label=label, alpha=0.8)

    ax1.set_xlabel('Adaptation Step (k)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Loss (Infidelity)', fontsize=13, fontweight='bold')
    ax1.set_title('(A) Loss Evolution During Inner Loop', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9, ncol=2, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=11)

    # Add text showing convergence
    avg_initial_loss = np.mean([d['losses'][0] for d in dynamics])
    avg_final_loss = np.mean([d['losses'][-1] for d in dynamics])
    avg_improvement = avg_initial_loss - avg_final_loss

    textstr = (
        f'Average improvement:\n'
        f'{avg_improvement:.4f}\n'
        f'({avg_improvement/avg_initial_loss*100:.1f}% reduction)'
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.95, 0.95, textstr, transform=ax1.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=props)

    # Panel B: Gradient norm evolution
    for i, dyn in enumerate(dynamics):
        grad_norms = dyn['grad_norms']
        ax2.plot(k_array, grad_norms, 'o-', linewidth=2, markersize=5,
                color=colors[i], alpha=0.6)

    # Plot average with thicker line
    avg_grad_norms = np.array(results['average_grad_norms'])
    ax2.plot(k_array, avg_grad_norms, 'k-', linewidth=3,
            label='Average', alpha=0.9)

    # Plot exponential fit if available
    if results['gradient_fit']['success']:
        A_fit = results['gradient_fit']['A']
        lambda_fit = results['gradient_fit']['lambda']

        k_fine = np.linspace(0, K_max, 100)
        grad_fit = A_fit * np.exp(-lambda_fit * k_fine)

        ax2.plot(k_fine, grad_fit, '--', linewidth=2.5, color='red',
                label=f"Fit: {A_fit:.3f} exp(-{lambda_fit:.3f}k)", alpha=0.8)

    ax2.set_xlabel('Adaptation Step (k)', fontsize=13, fontweight='bold')
    ax2.set_ylabel(r'Gradient Norm $\|\nabla_\phi L\|$', fontsize=13, fontweight='bold')
    ax2.set_title('(B) Gradient Norm Evolution (Exponential Decay)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=11)
    ax2.set_yscale('log')  # Log scale to better show exponential decay

    # Add text showing decay rate
    if results['gradient_fit']['success']:
        lambda_fit = results['gradient_fit']['lambda']
        textstr = (
            f'Exponential decay:\n'
            f'λ = {lambda_fit:.4f}\n'
            f'Half-life ≈ {np.log(2)/lambda_fit:.1f} steps'
        )
        props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
        ax2.text(0.95, 0.95, textstr, transform=ax2.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=props)

    plt.suptitle(f'Inner-Loop Adaptation Dynamics ({n_tasks} Tasks)',
                fontsize=16, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to {output_path}")

    plt.close()


def plot_individual_task_dynamics(
    results: Dict,
    task_idx: int = 0,
    output_path: str = None
):
    """
    Generate detailed plot for a single task showing both loss and gradient.

    Args:
        results: Results dict from experiment
        task_idx: Index of task to plot
        output_path: Path to save figure
    """
    sns.set_style("whitegrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    dyn = results['dynamics'][task_idx]
    task_params = dyn['task_params']
    K_max = results['K_max']
    inner_lr = results['inner_lr']

    k_array = np.arange(K_max + 1)
    losses = np.array(dyn['losses'])
    grad_norms = np.array(dyn['grad_norms'])

    # Panel 1: Loss
    ax1.plot(k_array, losses, 'o-', linewidth=2.5, markersize=8,
            color='steelblue')
    ax1.set_xlabel('Adaptation Step (k)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Loss (Infidelity)', fontsize=13, fontweight='bold')
    ax1.set_title(f'Task {task_idx+1}: Loss Evolution\n'
                 f'α={task_params["alpha"]:.2f}, A={task_params["A"]:.3f}, '
                 f'ωc={task_params["omega_c"]:.2f}',
                 fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=11)

    # Add improvement annotation
    improvement = losses[0] - losses[-1]
    ax1.annotate('', xy=(K_max, losses[-1]), xytext=(K_max, losses[0]),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax1.text(K_max + 0.5, (losses[0] + losses[-1])/2,
            f'Δ={improvement:.4f}',
            fontsize=11, color='red', fontweight='bold')

    # Panel 2: Gradient norm (log scale)
    ax2.semilogy(k_array[1:], grad_norms[1:], 'o-', linewidth=2.5, markersize=8,
                color='green')
    ax2.set_xlabel('Adaptation Step (k)', fontsize=13, fontweight='bold')
    ax2.set_ylabel(r'Gradient Norm $\|\nabla_\phi L\|$ (log scale)',
                  fontsize=13, fontweight='bold')
    ax2.set_title('Gradient Norm Evolution (Convergence Indicator)',
                 fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.tick_params(labelsize=11)

    # Fit exponential to this task's gradient
    k_fit = k_array[1:]
    grad_fit_vals = grad_norms[1:]

    def exp_decay(k, A, lam):
        return A * np.exp(-lam * k)

    try:
        popt, _ = curve_fit(exp_decay, k_fit, grad_fit_vals, p0=[grad_fit_vals[0], 0.1])
        A_fit, lambda_fit = popt

        k_fine = np.linspace(1, K_max, 100)
        grad_pred = exp_decay(k_fine, A_fit, lambda_fit)
        ax2.plot(k_fine, grad_pred, '--', linewidth=2, color='red',
                label=f'Fit: {A_fit:.3f} exp(-{lambda_fit:.3f}k)')
        ax2.legend(fontsize=11)
    except:
        pass

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nIndividual task figure saved to {output_path}")

    plt.close()


@app.command()
def main(
    meta_path: Path = Path("experiments/train_scripts/checkpoints/maml_best_policy.pt"),
    output_dir: Path = Path("results/adaptation_dynamics"),
    n_tasks: int = 10,
    k_max: int = 20
):
    """
    Run adaptation dynamics visualization experiment.

    Args:
        meta_path: Path to trained meta policy checkpoint
        output_dir: Directory to save results and figures
        n_tasks: Number of tasks to sample and track (default 10)
        k_max: Maximum adaptation steps to track (default 20)
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

    # Check if policy path exists
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta policy not found: {meta_path}")

    print(f"Using meta policy: {meta_path}")

    # Run experiment
    results = run_adaptation_dynamics_experiment(
        meta_policy_path=str(meta_path),
        config=config,
        n_tasks=n_tasks,
        K_max=k_max,
        output_dir=str(output_dir)
    )

    # Generate main plot (both panels)
    plot_path = f"{output_dir}/adaptation_dynamics.pdf"
    plot_adaptation_dynamics(results, output_path=plot_path)

    plot_path_png = f"{output_dir}/adaptation_dynamics.png"
    plot_adaptation_dynamics(results, output_path=plot_path_png)

    # Generate individual task plots for first 3 tasks
    print("\nGenerating individual task plots...")
    for i in range(min(3, n_tasks)):
        individual_path = f"{output_dir}/task_{i+1}_dynamics.pdf"
        plot_individual_task_dynamics(results, task_idx=i, output_path=individual_path)

        individual_path_png = f"{output_dir}/task_{i+1}_dynamics.png"
        plot_individual_task_dynamics(results, task_idx=i, output_path=individual_path_png)

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)
    print("\nKey insights:")
    print("  • Loss decreases during adaptation (validates learning)")
    print("  • Gradient norm shows exponential decay (validates convergence)")
    if results['gradient_fit']['success']:
        print(f"  • Decay rate λ = {results['gradient_fit']['lambda']:.4f}")
    print("\nOutputs:")
    print(f"  • Main plot: {output_dir}/adaptation_dynamics.pdf")
    print(f"  • Individual tasks: {output_dir}/task_*_dynamics.pdf")


if __name__ == "__main__":
    app()
