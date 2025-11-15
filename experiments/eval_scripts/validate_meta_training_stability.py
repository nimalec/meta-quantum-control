"""
Validation Script: Meta-Training Stability and Gradient Dynamics

This script monitors the meta-training process to validate numerical stability
and gradient behavior:

A. Meta-gradient norm: ‖∇_θ₀ L_meta‖ vs. meta-iteration
   - Tracks outer-loop gradient magnitudes
   - Should decrease over training (approaching local optimum)
   - Sudden spikes indicate instability

B. Gradient alignment: cos(∇_outer, ∇_inner)
   - Measures alignment between task-specific gradients and meta-objective
   - Should stay > 0 (positive alignment)
   - Values near 1 indicate strong alignment

C. NaN/Inf counter: Cumulative count of numerical errors
   - Should remain at zero (validates numerical stability)
   - Any non-zero count indicates gradient explosion or underflow

Expected output: Three-panel plot showing meta-training health metrics
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
from metaqctrl.utils.checkpoint_utils import load_policy_from_checkpoint

app = Typer()


def compute_gradient_alignment(
    grad_inner: List[torch.Tensor],
    grad_outer: List[torch.Tensor]
) -> float:
    """
    Compute cosine similarity between inner and outer gradients.

    cos(θ) = (g_inner · g_outer) / (‖g_inner‖ ‖g_outer‖)

    Args:
        grad_inner: List of parameter gradients from inner loop
        grad_outer: List of parameter gradients from outer loop

    Returns:
        alignment: Cosine similarity [-1, 1]
    """
    # Flatten gradients
    inner_flat = torch.cat([g.flatten() for g in grad_inner if g is not None])
    outer_flat = torch.cat([g.flatten() for g in grad_outer if g is not None])

    # Compute cosine similarity
    dot_product = torch.dot(inner_flat, outer_flat)
    norm_inner = torch.norm(inner_flat)
    norm_outer = torch.norm(outer_flat)

    if norm_inner == 0 or norm_outer == 0:
        return 0.0

    alignment = dot_product / (norm_inner * norm_outer)
    return float(alignment.item())


def check_for_nans_infs(tensors: List[torch.Tensor]) -> Tuple[bool, bool]:
    """
    Check if any tensors contain NaN or Inf.

    Args:
        tensors: List of tensors to check

    Returns:
        has_nan: True if any NaN found
        has_inf: True if any Inf found
    """
    has_nan = False
    has_inf = False

    for t in tensors:
        if t is not None:
            if torch.isnan(t).any():
                has_nan = True
            if torch.isinf(t).any():
                has_inf = True

    return has_nan, has_inf


def run_meta_training_iteration(
    meta_policy: torch.nn.Module,
    tasks: List[NoiseParameters],
    env,
    config: Dict,
    meta_optimizer: torch.optim.Optimizer
) -> Dict:
    """
    Run a single meta-training iteration (outer loop) and track metrics.

    Args:
        meta_policy: Meta policy network
        tasks: Batch of tasks
        env: QuantumEnvironment
        config: Configuration dict
        meta_optimizer: Meta optimizer (e.g., Adam)

    Returns:
        metrics: Dict with meta-loss, gradients, alignment, etc.
    """
    K_inner = config.get('inner_steps', 5)
    inner_lr = config.get('inner_lr', 0.01)

    meta_optimizer.zero_grad()

    task_losses = []
    inner_gradients_all = []
    outer_gradients_all = []

    # Meta-training loop (MAML-style)
    for task in tasks:
        # Clone policy for inner loop
        adapted_policy = copy.deepcopy(meta_policy)
        adapted_policy.train()

        # Inner loop adaptation
        for k in range(K_inner):
            loss_inner = env.compute_loss_differentiable(
                adapted_policy, task, device=torch.device('cpu')
            )
            loss_inner.backward()

            # Store inner gradients (from first inner step for alignment computation)
            if k == 0:
                inner_grads = [
                    p.grad.clone() if p.grad is not None else None
                    for p in adapted_policy.parameters()
                ]
                inner_gradients_all.append(inner_grads)

            # Manual SGD step
            with torch.no_grad():
                for param in adapted_policy.parameters():
                    if param.grad is not None:
                        param -= inner_lr * param.grad
                        param.grad.zero_()

        # Outer loop: compute meta-loss on adapted policy
        loss_outer = env.compute_loss_differentiable(
            adapted_policy, task, device=torch.device('cpu')
        )
        loss_outer.backward()

        task_losses.append(float(loss_outer.item()))

        # Store outer gradients (with respect to meta-policy parameters)
        outer_grads = [
            p.grad.clone() if p.grad is not None else None
            for p in adapted_policy.parameters()
        ]
        outer_gradients_all.append(outer_grads)

    # Compute meta-loss (average over tasks)
    meta_loss = np.mean(task_losses)

    # Aggregate gradients for meta-update
    # Note: In MAML, we'd accumulate gradients across tasks
    # For simplicity, we compute metrics on first task's gradients
    outer_grads = outer_gradients_all[0]
    inner_grads = inner_gradients_all[0]

    # Compute meta-gradient norm
    meta_grad_norm = 0.0
    for g in outer_grads:
        if g is not None:
            meta_grad_norm += g.norm().item() ** 2
    meta_grad_norm = np.sqrt(meta_grad_norm)

    # Compute gradient alignment
    alignment = compute_gradient_alignment(inner_grads, outer_grads)

    # Check for NaN/Inf
    has_nan, has_inf = check_for_nans_infs(outer_grads)

    # Meta-update (apply gradients)
    # Note: In real training, gradients would be accumulated across tasks
    meta_optimizer.step()

    return {
        'meta_loss': meta_loss,
        'meta_grad_norm': meta_grad_norm,
        'gradient_alignment': alignment,
        'has_nan': has_nan,
        'has_inf': has_inf
    }


def run_meta_training_stability_experiment(
    meta_policy_path: str,
    config: Dict,
    n_iterations: int = 50,
    batch_size: int = 5,
    output_dir: str = "results/meta_training_stability"
) -> Dict:
    """
    Run simulated meta-training to track stability metrics.

    Args:
        meta_policy_path: Path to trained meta policy (starting point)
        config: Experiment configuration
        n_iterations: Number of meta-iterations to simulate
        batch_size: Number of tasks per meta-iteration
        output_dir: Output directory

    Returns:
        results: Dict with metrics over iterations
    """
    print("=" * 80)
    print("EXPERIMENT: Meta-Training Stability and Gradient Dynamics")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)

    print(f"\nExperiment parameters:")
    print(f"  Meta-iterations: {n_iterations}")
    print(f"  Batch size: {batch_size}")
    print(f"  Inner steps: {config.get('inner_steps', 5)}")
    print(f"  Inner lr: {config.get('inner_lr', 0.01)}")

    # Load meta policy
    print("\n[1/3] Loading meta policy...")
    meta_policy = load_policy_from_checkpoint(
        meta_policy_path, config, eval_mode=False, verbose=True
    )

    # Create environment
    print("\n[2/3] Creating quantum environment...")
    target_state = get_target_state_from_config(config)
    env = create_quantum_environment(config, target_state)

    # Create task distribution
    task_dist = TaskDistribution(
        dist_type=config.get('task_dist_type', 'uniform'),
        ranges={
            'alpha': tuple(config['alpha_range']),
            'A': tuple(config['A_range']),
            'omega_c': tuple(config['omega_c_range'])
        }
    )

    # Create meta-optimizer
    meta_lr = config.get('meta_lr', 0.001)
    meta_optimizer = torch.optim.Adam(meta_policy.parameters(), lr=meta_lr)

    # Run meta-training iterations
    print(f"\n[3/3] Running {n_iterations} meta-training iterations...")

    meta_losses = []
    meta_grad_norms = []
    gradient_alignments = []
    nan_count = 0
    inf_count = 0
    nan_inf_cumulative = []

    for iteration in tqdm(range(n_iterations), desc="Meta-iterations"):
        # Sample tasks for this iteration
        tasks = task_dist.sample(batch_size)

        # Run meta-training iteration
        metrics = run_meta_training_iteration(
            meta_policy=meta_policy,
            tasks=tasks,
            env=env,
            config=config,
            meta_optimizer=meta_optimizer
        )

        # Store metrics
        meta_losses.append(metrics['meta_loss'])
        meta_grad_norms.append(metrics['meta_grad_norm'])
        gradient_alignments.append(metrics['gradient_alignment'])

        if metrics['has_nan']:
            nan_count += 1
        if metrics['has_inf']:
            inf_count += 1

        nan_inf_cumulative.append(nan_count + inf_count)

        # Print progress every 10 iterations
        if (iteration + 1) % 10 == 0:
            print(f"\n  Iteration {iteration+1}/{n_iterations}:")
            print(f"    Meta-loss: {metrics['meta_loss']:.4f}")
            print(f"    Meta-grad norm: {metrics['meta_grad_norm']:.6f}")
            print(f"    Alignment: {metrics['gradient_alignment']:.4f}")
            print(f"    NaN/Inf count: {nan_inf_cumulative[-1]}")

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Meta-loss: {meta_losses[0]:.4f} → {meta_losses[-1]:.4f}")
    print(f"Meta-grad norm: {meta_grad_norms[0]:.6f} → {meta_grad_norms[-1]:.6f}")
    print(f"Gradient alignment: mean={np.mean(gradient_alignments):.4f}, "
          f"std={np.std(gradient_alignments):.4f}")
    print(f"NaN occurrences: {nan_count}")
    print(f"Inf occurrences: {inf_count}")
    print(f"Total numerical errors: {nan_count + inf_count}")

    if nan_count + inf_count == 0:
        print("\n✓ VALIDATION PASSED: No numerical instabilities detected!")
    else:
        print("\n✗ WARNING: Numerical instabilities detected!")

    # Save results
    print(f"\nSaving results to {output_dir}...")
    results = {
        'n_iterations': n_iterations,
        'batch_size': batch_size,
        'meta_losses': meta_losses,
        'meta_grad_norms': meta_grad_norms,
        'gradient_alignments': gradient_alignments,
        'nan_inf_cumulative': nan_inf_cumulative,
        'nan_count': nan_count,
        'inf_count': inf_count,
        'config': config
    }

    results_path = f"{output_dir}/results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    return results


def plot_meta_training_stability(
    results: Dict,
    output_path: str = None
):
    """
    Generate three-panel plot showing meta-training stability metrics.

    Args:
        results: Results dict from experiment
        output_path: Path to save figure
    """
    sns.set_style("whitegrid")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))

    n_iterations = results['n_iterations']
    iterations = np.arange(n_iterations)

    meta_losses = np.array(results['meta_losses'])
    meta_grad_norms = np.array(results['meta_grad_norms'])
    gradient_alignments = np.array(results['gradient_alignments'])
    nan_inf_cumulative = np.array(results['nan_inf_cumulative'])

    # Panel A: Meta-gradient norm
    ax1.plot(iterations, meta_grad_norms, 'o-', linewidth=2, markersize=5,
            color='steelblue', alpha=0.8)
    ax1.set_xlabel('Meta-Iteration', fontsize=13, fontweight='bold')
    ax1.set_ylabel(r'Meta-Gradient Norm $\|\nabla_{\theta_0} \mathcal{L}_{meta}\|$',
                  fontsize=13, fontweight='bold')
    ax1.set_title('(A) Meta-Gradient Norm Evolution',
                 fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=11)

    # Add trend line
    if len(iterations) > 10:
        from scipy.signal import savgol_filter
        try:
            smoothed = savgol_filter(meta_grad_norms, window_length=11, polyorder=2)
            ax1.plot(iterations, smoothed, '--', linewidth=2.5, color='red',
                    label='Trend (smoothed)', alpha=0.7)
            ax1.legend(fontsize=11)
        except:
            pass

    # Add text box with stats
    textstr = (
        f'Initial: {meta_grad_norms[0]:.6f}\n'
        f'Final: {meta_grad_norms[-1]:.6f}\n'
        f'Mean: {np.mean(meta_grad_norms):.6f}\n'
        f'Std: {np.std(meta_grad_norms):.6f}'
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.98, 0.98, textstr, transform=ax1.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=props)

    # Panel B: Gradient alignment
    ax2.plot(iterations, gradient_alignments, 'o-', linewidth=2, markersize=5,
            color='green', alpha=0.8)
    ax2.axhline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.5,
               label='Zero alignment')
    ax2.axhline(1, color='gray', linestyle=':', linewidth=1.5, alpha=0.5,
               label='Perfect alignment')
    ax2.set_xlabel('Meta-Iteration', fontsize=13, fontweight='bold')
    ax2.set_ylabel(r'Gradient Alignment $\cos(\nabla_{outer}, \nabla_{inner})$',
                  fontsize=13, fontweight='bold')
    ax2.set_title('(B) Inner-Outer Gradient Alignment',
                 fontsize=14, fontweight='bold')
    ax2.set_ylim([-1.1, 1.1])
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=11)
    ax2.legend(fontsize=10, loc='lower right')

    # Add text box with stats
    textstr = (
        f'Mean: {np.mean(gradient_alignments):.4f}\n'
        f'Std: {np.std(gradient_alignments):.4f}\n'
        f'Min: {np.min(gradient_alignments):.4f}\n'
        f'Max: {np.max(gradient_alignments):.4f}'
    )
    props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.8)
    ax2.text(0.98, 0.02, textstr, transform=ax2.transAxes,
            fontsize=10, verticalalignment='bottom', horizontalalignment='right',
            bbox=props)

    # Panel C: NaN/Inf counter
    ax3.plot(iterations, nan_inf_cumulative, 'o-', linewidth=2.5, markersize=6,
            color='orangered', alpha=0.9)
    ax3.axhline(0, color='green', linestyle='--', linewidth=2, alpha=0.7,
               label='Stable (zero errors)')
    ax3.set_xlabel('Meta-Iteration', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Cumulative NaN/Inf Count', fontsize=13, fontweight='bold')
    ax3.set_title('(C) Numerical Stability Monitor',
                 fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(labelsize=11)
    ax3.legend(fontsize=11)

    # Add status text
    total_errors = results['nan_count'] + results['inf_count']
    if total_errors == 0:
        status_text = '✓ STABLE: No numerical errors'
        status_color = 'lightgreen'
    else:
        status_text = f'✗ UNSTABLE: {total_errors} errors\n(NaN: {results["nan_count"]}, Inf: {results["inf_count"]})'
        status_color = 'lightcoral'

    props = dict(boxstyle='round', facecolor=status_color, alpha=0.9)
    ax3.text(0.98, 0.98, status_text, transform=ax3.transAxes,
            fontsize=11, verticalalignment='top', horizontalalignment='right',
            bbox=props, fontweight='bold')

    plt.suptitle(f'Meta-Training Stability Analysis ({n_iterations} iterations)',
                fontsize=16, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to {output_path}")

    plt.close()


def plot_combined_meta_loss(
    results: Dict,
    output_path: str = None
):
    """
    Generate additional plot showing meta-loss evolution.

    Args:
        results: Results dict from experiment
        output_path: Path to save figure
    """
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    iterations = np.arange(results['n_iterations'])
    meta_losses = np.array(results['meta_losses'])

    ax.plot(iterations, meta_losses, 'o-', linewidth=2, markersize=6,
           color='purple', alpha=0.8, label='Meta-loss')

    # Add smoothed trend
    if len(iterations) > 10:
        from scipy.signal import savgol_filter
        try:
            smoothed = savgol_filter(meta_losses, window_length=11, polyorder=2)
            ax.plot(iterations, smoothed, '--', linewidth=3, color='darkred',
                   label='Trend (smoothed)', alpha=0.8)
        except:
            pass

    ax.set_xlabel('Meta-Iteration', fontsize=13, fontweight='bold')
    ax.set_ylabel(r'Meta-Loss $\mathcal{L}_{meta}$', fontsize=13, fontweight='bold')
    ax.set_title('Meta-Loss Evolution During Training',
                fontsize=15, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)

    # Add improvement annotation
    improvement = meta_losses[0] - meta_losses[-1]
    improvement_pct = (improvement / meta_losses[0]) * 100

    textstr = (
        f'Initial: {meta_losses[0]:.4f}\n'
        f'Final: {meta_losses[-1]:.4f}\n'
        f'Improvement: {improvement:.4f}\n'
        f'({improvement_pct:.1f}% reduction)'
    )
    props = dict(boxstyle='round', facecolor='lavender', alpha=0.8)
    ax.text(0.98, 0.98, textstr, transform=ax.transAxes,
           fontsize=11, verticalalignment='top', horizontalalignment='right',
           bbox=props)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Meta-loss figure saved to {output_path}")

    plt.close()


@app.command()
def main(
    meta_path: Path = Path("experiments/train_scripts/checkpoints/maml_best_policy.pt"),
    output_dir: Path = Path("results/meta_training_stability"),
    n_iterations: int = 50,
    batch_size: int = 5
):
    """
    Run meta-training stability validation experiment.

    Args:
        meta_path: Path to trained meta policy checkpoint
        output_dir: Directory to save results and figures
        n_iterations: Number of meta-iterations to simulate (default 50)
        batch_size: Number of tasks per meta-iteration (default 5)
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
        'inner_steps': 5,
        'meta_lr': 0.001,
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
    results = run_meta_training_stability_experiment(
        meta_policy_path=str(meta_path),
        config=config,
        n_iterations=n_iterations,
        batch_size=batch_size,
        output_dir=str(output_dir)
    )

    # Generate main three-panel plot
    plot_path = f"{output_dir}/meta_training_stability.pdf"
    plot_meta_training_stability(results, output_path=plot_path)

    plot_path_png = f"{output_dir}/meta_training_stability.png"
    plot_meta_training_stability(results, output_path=plot_path_png)

    # Generate meta-loss plot
    loss_path = f"{output_dir}/meta_loss_evolution.pdf"
    plot_combined_meta_loss(results, output_path=loss_path)

    loss_path_png = f"{output_dir}/meta_loss_evolution.png"
    plot_combined_meta_loss(results, output_path=loss_path_png)

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)
    print("\nKey insights:")
    print(f"  • Meta-gradient norm: {results['meta_grad_norms'][0]:.6f} → {results['meta_grad_norms'][-1]:.6f}")
    print(f"  • Gradient alignment: mean={np.mean(results['gradient_alignments']):.4f}")
    print(f"  • Numerical stability: {results['nan_count'] + results['inf_count']} errors")
    print("\nOutputs:")
    print(f"  • Stability metrics: {output_dir}/meta_training_stability.pdf")
    print(f"  • Meta-loss evolution: {output_dir}/meta_loss_evolution.pdf")


if __name__ == "__main__":
    app()
