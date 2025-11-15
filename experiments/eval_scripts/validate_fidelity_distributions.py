"""
Validation Script: Fidelity Distribution Analysis

This script analyzes how fidelity is distributed across the full task distribution
for three policies:
- π_rob: Robust baseline (trained on average task)
- π₀: Meta-initialization (K=0, no adaptation)
- π_K: Adapted meta-policy (K=5 adaptation steps)

Generates:
A. Violin plots: Show full fidelity distribution shape
   - Visualizes mean, median, quartiles, and outliers
   - Reveals multimodality and tail behavior
   - Shows spread/variance of performance

B. Cumulative Distribution Functions (CDFs): F(fidelity)
   - Shows probability that fidelity ≤ threshold
   - Useful for comparing reliability and robustness
   - Reveals worst-case and typical-case performance

Expected output: Two-panel figure showing distributions and CDFs
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


def evaluate_policy_on_tasks(
    policy: torch.nn.Module,
    tasks: List[NoiseParameters],
    env,
    config: Dict,
    adapt: bool = False,
    K: int = 0
) -> List[float]:
    """
    Evaluate policy on a list of tasks.

    Args:
        policy: Policy network
        tasks: List of tasks to evaluate
        env: QuantumEnvironment
        config: Configuration dict
        adapt: If True, adapt policy before evaluation
        K: Number of adaptation steps (if adapt=True)

    Returns:
        fidelities: List of fidelities achieved on each task
    """
    inner_lr = config.get('inner_lr', 0.01)
    fidelities = []

    for task in tqdm(tasks, desc=f"Evaluating ({'adapted' if adapt else 'direct'})", leave=False):
        task_features = torch.tensor(
            [task.alpha, task.A, task.omega_c],
            dtype=torch.float32
        )

        if adapt and K > 0:
            # Adapt policy
            adapted_policy = copy.deepcopy(policy)
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

            eval_policy = adapted_policy
        else:
            eval_policy = policy

        # Evaluate
        eval_policy.eval()
        with torch.no_grad():
            controls = eval_policy(task_features).detach().numpy()
        fidelity = env.compute_fidelity(controls, task)
        fidelities.append(fidelity)

    return fidelities


def run_fidelity_distribution_experiment(
    meta_policy_path: str,
    robust_policy_path: str,
    config: Dict,
    n_tasks: int = 200,
    K_adapt: int = 5,
    output_dir: str = "results/fidelity_distributions"
) -> Dict:
    """
    Main experiment: Evaluate fidelity distributions for three policies.

    Args:
        meta_policy_path: Path to trained meta policy
        robust_policy_path: Path to robust baseline policy
        config: Experiment configuration
        n_tasks: Number of tasks to sample from distribution
        K_adapt: Number of adaptation steps for π_K
        output_dir: Output directory

    Returns:
        results: Dict with fidelity distributions
    """
    print("=" * 80)
    print("EXPERIMENT: Fidelity Distribution Analysis")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)

    print(f"\nExperiment parameters:")
    print(f"  Number of tasks: {n_tasks}")
    print(f"  Adaptation steps (π_K): {K_adapt}")

    # Load policies
    print("\n[1/4] Loading policies...")
    meta_policy = load_policy_from_checkpoint(
        meta_policy_path, config, eval_mode=False, verbose=True
    )
    robust_policy = load_policy_from_checkpoint(
        robust_policy_path, config, eval_mode=True, verbose=True
    )

    # Create environment
    print("\n[2/4] Creating quantum environment...")
    target_state = get_target_state_from_config(config)
    env = create_quantum_environment(config, target_state)

    # Sample tasks from full distribution
    print(f"\n[3/4] Sampling {n_tasks} tasks from distribution...")
    task_dist = TaskDistribution(
        dist_type=config.get('task_dist_type', 'uniform'),
        ranges={
            'alpha': tuple(config['alpha_range']),
            'A': tuple(config['A_range']),
            'omega_c': tuple(config['omega_c_range'])
        }
    )
    tasks = task_dist.sample(n_tasks)

    # Evaluate all three policies
    print(f"\n[4/4] Evaluating policies on {n_tasks} tasks...")

    print("\n  π_rob (Robust baseline)...")
    fidelities_robust = evaluate_policy_on_tasks(
        policy=robust_policy,
        tasks=tasks,
        env=env,
        config=config,
        adapt=False,
        K=0
    )

    print("\n  π₀ (Meta-initialization, K=0)...")
    fidelities_meta_init = evaluate_policy_on_tasks(
        policy=meta_policy,
        tasks=tasks,
        env=env,
        config=config,
        adapt=False,
        K=0
    )

    print(f"\n  π_K (Adapted meta-policy, K={K_adapt})...")
    fidelities_meta_adapted = evaluate_policy_on_tasks(
        policy=meta_policy,
        tasks=tasks,
        env=env,
        config=config,
        adapt=True,
        K=K_adapt
    )

    # Compute statistics
    print("\n" + "=" * 60)
    print("DISTRIBUTION STATISTICS")
    print("=" * 60)

    stats = {}
    for name, fids in [
        ('π_rob', fidelities_robust),
        ('π₀', fidelities_meta_init),
        ('π_K', fidelities_meta_adapted)
    ]:
        fids_array = np.array(fids)
        stats[name] = {
            'mean': float(np.mean(fids_array)),
            'median': float(np.median(fids_array)),
            'std': float(np.std(fids_array)),
            'min': float(np.min(fids_array)),
            'max': float(np.max(fids_array)),
            'q25': float(np.percentile(fids_array, 25)),
            'q75': float(np.percentile(fids_array, 75)),
        }

        print(f"\n{name}:")
        print(f"  Mean: {stats[name]['mean']:.4f}")
        print(f"  Median: {stats[name]['median']:.4f}")
        print(f"  Std: {stats[name]['std']:.4f}")
        print(f"  Range: [{stats[name]['min']:.4f}, {stats[name]['max']:.4f}]")
        print(f"  IQR: [{stats[name]['q25']:.4f}, {stats[name]['q75']:.4f}]")

    # Save results
    print(f"\nSaving results to {output_dir}...")
    results = {
        'n_tasks': n_tasks,
        'K_adapt': K_adapt,
        'fidelities': {
            'robust': fidelities_robust,
            'meta_init': fidelities_meta_init,
            'meta_adapted': fidelities_meta_adapted
        },
        'statistics': stats,
        'config': config
    }

    results_path = f"{output_dir}/results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    return results


def plot_fidelity_distributions(
    results: Dict,
    output_path: str = None
):
    """
    Generate two-panel plot: (A) Violin plots, (B) CDFs.

    Args:
        results: Results dict from experiment
        output_path: Path to save figure
    """
    sns.set_style("whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Extract data
    fids_robust = results['fidelities']['robust']
    fids_meta_init = results['fidelities']['meta_init']
    fids_meta_adapted = results['fidelities']['meta_adapted']
    K_adapt = results['K_adapt']

    # Panel A: Violin plots
    data_violin = {
        r'$\pi_{\mathrm{rob}}$': fids_robust,
        r'$\pi_0$': fids_meta_init,
        rf'$\pi_K$ (K={K_adapt})': fids_meta_adapted
    }

    # Create violin plot data
    positions = [1, 2, 3]
    colors = ['orangered', 'steelblue', 'green']

    parts = ax1.violinplot(
        [fids_robust, fids_meta_init, fids_meta_adapted],
        positions=positions,
        widths=0.6,
        showmeans=True,
        showmedians=True,
        showextrema=True
    )

    # Color the violins
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
        pc.set_linewidth(1.5)

    # Style the other elements
    for partname in ['cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans']:
        if partname in parts:
            parts[partname].set_edgecolor('black')
            parts[partname].set_linewidth(2)

    # Add mean markers
    means = [np.mean(fids_robust), np.mean(fids_meta_init), np.mean(fids_meta_adapted)]
    ax1.scatter(positions, means, color='red', s=100, zorder=3,
                marker='D', label='Mean', edgecolor='black', linewidth=1.5)

    ax1.set_xticks(positions)
    ax1.set_xticklabels([r'$\pi_{\mathrm{rob}}$', r'$\pi_0$', rf'$\pi_K$ (K={K_adapt})'],
                        fontsize=13)
    ax1.set_ylabel('Gate Fidelity', fontsize=14, fontweight='bold')
    ax1.set_title('(A) Fidelity Distribution: Violin Plots',
                 fontsize=15, fontweight='bold')
    ax1.set_ylim([0.0, 1.05])
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(labelsize=12)
    ax1.legend(fontsize=11)

    # Add text box with statistics
    textstr = (
        f'Statistics (n={results["n_tasks"]}):\n'
        f'π_rob:  μ={results["statistics"]["π_rob"]["mean"]:.3f}, '
        f'σ={results["statistics"]["π_rob"]["std"]:.3f}\n'
        f'π₀:      μ={results["statistics"]["π₀"]["mean"]:.3f}, '
        f'σ={results["statistics"]["π₀"]["std"]:.3f}\n'
        f'π_K:     μ={results["statistics"]["π_K"]["mean"]:.3f}, '
        f'σ={results["statistics"]["π_K"]["std"]:.3f}'
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.02, 0.02, textstr, transform=ax1.transAxes,
            fontsize=9, verticalalignment='bottom',
            bbox=props, family='monospace')

    # Panel B: Cumulative Distribution Functions (CDFs)
    fids_all = [fids_robust, fids_meta_init, fids_meta_adapted]
    labels = [r'$\pi_{\mathrm{rob}}$ (Robust)',
              r'$\pi_0$ (Meta-init)',
              rf'$\pi_K$ (Adapted, K={K_adapt})']

    for i, (fids, label, color) in enumerate(zip(fids_all, labels, colors)):
        fids_sorted = np.sort(fids)
        cdf = np.arange(1, len(fids_sorted) + 1) / len(fids_sorted)

        ax2.plot(fids_sorted, cdf, linewidth=3, color=color,
                label=label, alpha=0.9)

    # Add reference lines
    ax2.axhline(0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.5,
               label='Median (50th percentile)')
    ax2.axvline(0.99, color='green', linestyle=':', linewidth=1.5, alpha=0.5,
               label='High fidelity (0.99)')

    ax2.set_xlabel('Gate Fidelity', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Cumulative Probability F(fidelity)', fontsize=14, fontweight='bold')
    ax2.set_title('(B) Cumulative Distribution Functions',
                 fontsize=15, fontweight='bold')
    ax2.set_xlim([0.0, 1.05])
    ax2.set_ylim([0.0, 1.05])
    ax2.legend(fontsize=11, loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=12)

    # Add annotations showing key percentiles
    for i, (fids, color) in enumerate(zip(fids_all, colors)):
        median = np.median(fids)
        p95 = np.percentile(fids, 95)

        # Mark median on CDF
        ax2.plot(median, 0.5, 'o', markersize=10, color=color,
                markeredgecolor='black', markeredgewidth=1.5, zorder=5)

    # Add text box with CDF insights
    p5_robust = np.percentile(fids_robust, 5)
    p5_adapted = np.percentile(fids_meta_adapted, 5)

    textstr = (
        f'Worst-case performance (5th percentile):\n'
        f'π_rob: {p5_robust:.3f}\n'
        f'π_K:   {p5_adapted:.3f}\n'
        f'Improvement: {(p5_adapted - p5_robust):.3f}'
    )
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=props, family='monospace')

    plt.suptitle(f'Fidelity Distribution Analysis ({results["n_tasks"]} tasks)',
                fontsize=17, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to {output_path}")

    plt.close()


def plot_detailed_comparison(
    results: Dict,
    output_path: str = None
):
    """
    Generate additional detailed comparison plots.

    Args:
        results: Results dict from experiment
        output_path: Path to save figure
    """
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    fids_robust = np.array(results['fidelities']['robust'])
    fids_meta_init = np.array(results['fidelities']['meta_init'])
    fids_meta_adapted = np.array(results['fidelities']['meta_adapted'])
    K_adapt = results['K_adapt']

    # Plot 1: Histograms with overlays
    ax1 = axes[0, 0]
    bins = np.linspace(0, 1, 30)

    ax1.hist(fids_robust, bins=bins, alpha=0.6, color='orangered',
            label=r'$\pi_{\mathrm{rob}}$', density=True)
    ax1.hist(fids_meta_init, bins=bins, alpha=0.6, color='steelblue',
            label=r'$\pi_0$', density=True)
    ax1.hist(fids_meta_adapted, bins=bins, alpha=0.6, color='green',
            label=rf'$\pi_K$ (K={K_adapt})', density=True)

    ax1.set_xlabel('Fidelity', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Overlapping Histograms', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Box plots
    ax2 = axes[0, 1]
    bp = ax2.boxplot([fids_robust, fids_meta_init, fids_meta_adapted],
                     labels=[r'$\pi_{\mathrm{rob}}$', r'$\pi_0$', rf'$\pi_K$'],
                     patch_artist=True,
                     widths=0.6)

    for patch, color in zip(bp['boxes'], ['orangered', 'steelblue', 'green']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax2.set_ylabel('Fidelity', fontsize=12, fontweight='bold')
    ax2.set_title('(b) Box Plots (Quartiles)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0, 1.05])

    # Plot 3: Q-Q plot (Robust vs Adapted)
    ax3 = axes[1, 0]

    # Sort for Q-Q plot
    n = min(len(fids_robust), len(fids_meta_adapted))
    robust_sorted = np.sort(fids_robust)[:n]
    adapted_sorted = np.sort(fids_meta_adapted)[:n]

    ax3.scatter(robust_sorted, adapted_sorted, alpha=0.5, s=20, color='purple')
    ax3.plot([0, 1], [0, 1], 'r--', linewidth=2, label='y=x (equal performance)')

    ax3.set_xlabel(r'$\pi_{\mathrm{rob}}$ Fidelity', fontsize=12, fontweight='bold')
    ax3.set_ylabel(r'$\pi_K$ Fidelity', fontsize=12, fontweight='bold')
    ax3.set_title(f'(c) Q-Q Plot: Robust vs Adapted', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1])
    ax3.set_aspect('equal')

    # Plot 4: Improvement distribution
    ax4 = axes[1, 1]

    improvement = fids_meta_adapted - fids_robust
    ax4.hist(improvement, bins=30, color='purple', alpha=0.7, edgecolor='black')
    ax4.axvline(0, color='red', linestyle='--', linewidth=2,
               label='No improvement')
    ax4.axvline(np.mean(improvement), color='green', linestyle='-', linewidth=2,
               label=f'Mean: {np.mean(improvement):.4f}')

    ax4.set_xlabel(r'Improvement ($\pi_K - \pi_{\mathrm{rob}}$)',
                  fontsize=12, fontweight='bold')
    ax4.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax4.set_title('(d) Adaptation Improvement Distribution', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')

    # Add statistics
    pct_improved = np.sum(improvement > 0) / len(improvement) * 100
    textstr = (
        f'Tasks improved: {pct_improved:.1f}%\n'
        f'Mean improvement: {np.mean(improvement):.4f}\n'
        f'Max improvement: {np.max(improvement):.4f}'
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax4.text(0.98, 0.98, textstr, transform=ax4.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=props)

    plt.suptitle('Detailed Fidelity Distribution Comparison',
                fontsize=16, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Detailed comparison saved to {output_path}")

    plt.close()


@app.command()
def main(
    meta_path: Path = Path("experiments/train_scripts/checkpoints/maml_best_policy.pt"),
    robust_path: Path = Path("experiments/train_scripts/checkpoints/robust_best_policy.pt"),
    output_dir: Path = Path("results/fidelity_distributions"),
    n_tasks: int = 200,
    k_adapt: int = 5
):
    """
    Run fidelity distribution analysis experiment.

    Args:
        meta_path: Path to trained meta policy checkpoint
        robust_path: Path to robust baseline policy checkpoint
        output_dir: Directory to save results and figures
        n_tasks: Number of tasks to sample from distribution (default 200)
        k_adapt: Number of adaptation steps for π_K (default 5)
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

    # Run experiment
    results = run_fidelity_distribution_experiment(
        meta_policy_path=str(meta_path),
        robust_policy_path=str(robust_path),
        config=config,
        n_tasks=n_tasks,
        K_adapt=k_adapt,
        output_dir=str(output_dir)
    )

    # Generate main plot (violin + CDF)
    plot_path = f"{output_dir}/fidelity_distributions.pdf"
    plot_fidelity_distributions(results, output_path=plot_path)

    plot_path_png = f"{output_dir}/fidelity_distributions.png"
    plot_fidelity_distributions(results, output_path=plot_path_png)

    # Generate detailed comparison
    detailed_path = f"{output_dir}/detailed_comparison.pdf"
    plot_detailed_comparison(results, output_path=detailed_path)

    detailed_path_png = f"{output_dir}/detailed_comparison.png"
    plot_detailed_comparison(results, output_path=detailed_path_png)

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)
    print("\nKey insights:")
    print(f"  • Robust baseline: μ={results['statistics']['π_rob']['mean']:.4f}, "
          f"σ={results['statistics']['π_rob']['std']:.4f}")
    print(f"  • Meta-init (K=0): μ={results['statistics']['π₀']['mean']:.4f}, "
          f"σ={results['statistics']['π₀']['std']:.4f}")
    print(f"  • Adapted (K={k_adapt}): μ={results['statistics']['π_K']['mean']:.4f}, "
          f"σ={results['statistics']['π_K']['std']:.4f}")

    improvement = results['statistics']['π_K']['mean'] - results['statistics']['π_rob']['mean']
    print(f"  • Mean improvement: {improvement:.4f}")

    print("\nOutputs:")
    print(f"  • Main plot: {output_dir}/fidelity_distributions.pdf")
    print(f"  • Detailed comparison: {output_dir}/detailed_comparison.pdf")


if __name__ == "__main__":
    app()
