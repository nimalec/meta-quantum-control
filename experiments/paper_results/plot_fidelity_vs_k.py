"""
Plot: Fidelity vs Adaptation Steps K

Generates plots showing:
1. Post-adaptation fidelity as function of K
2. Pre-adaptation vs post-adaptation comparison
3. Adaptation gain (fidelity improvement) vs K
4. Distribution of fidelities across tasks for different K values

Loads trained policy from checkpoint and evaluates on test tasks.
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
import yaml

from metaqctrl.quantum.noise_models import TaskDistribution, NoiseParameters
from metaqctrl.meta_rl.policy import PulsePolicy
from metaqctrl.theory.quantum_environment import create_quantum_environment
from metaqctrl.quantum.gates import TargetGates


def load_policy_from_checkpoint(
    checkpoint_path: str,
    input_dim: int,
    output_dim: int,
    hidden_dims: List[int],
    device: torch.device = torch.device('cpu')
) -> PulsePolicy:
    """
    Load trained policy from checkpoint.

    Args:
        checkpoint_path: Path to .pt checkpoint file
        input_dim: Policy input dimension
        output_dim: Policy output dimension
        hidden_dims: Hidden layer dimensions
        device: Torch device

    Returns:
        Loaded policy model
    """
    policy = PulsePolicy(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=hidden_dims
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        policy.load_state_dict(checkpoint['model_state_dict'])
    elif 'policy_state_dict' in checkpoint:
        policy.load_state_dict(checkpoint['policy_state_dict'])
    else:
        # Assume checkpoint is the state dict itself
        policy.load_state_dict(checkpoint)

    policy.eval()
    print(f"Loaded policy from {checkpoint_path}")

    return policy


def evaluate_fidelity_vs_k(
    policy: PulsePolicy,
    env,
    test_tasks: List[NoiseParameters],
    k_values: List[int],
    inner_lr: float = 0.01,
    device: torch.device = torch.device('cpu')
) -> Dict:
    """
    Evaluate fidelity for different adaptation steps K.

    Args:
        policy: Trained policy
        env: Quantum environment
        test_tasks: List of test tasks
        k_values: List of K values to test
        inner_lr: Inner loop learning rate
        device: Torch device

    Returns:
        Dictionary with fidelities for each K
    """
    results = {
        'k_values': k_values,
        'pre_adapt_fidelities': [],
        'post_adapt_fidelities': {},
        'adaptation_gains': {}
    }

    print(f"\nEvaluating on {len(test_tasks)} test tasks...")
    print(f"K values: {k_values}")

    # Initialize storage for each K
    for K in k_values:
        results['post_adapt_fidelities'][K] = []
        results['adaptation_gains'][K] = []

    # Evaluate each task
    for task_idx, task in enumerate(test_tasks):
        if task_idx % 10 == 0:
            print(f"  Task {task_idx}/{len(test_tasks)}...", end='\r')

        task_features = torch.tensor(
            [task.alpha, task.A, task.omega_c],
            dtype=torch.float32
        ).to(device)

        # Pre-adaptation fidelity
        with torch.no_grad():
            controls_pre = policy(task_features).cpu().numpy()
            controls_pre = controls_pre.reshape(env.n_seg, env.num_controls)

        fid_pre = env.compute_fidelity(controls_pre, task)
        results['pre_adapt_fidelities'].append(fid_pre)

        # Post-adaptation fidelities for each K
        for K in k_values:
            # Clone policy for adaptation
            import copy
            adapted_policy = copy.deepcopy(policy)
            adapted_policy.train()

            optimizer = torch.optim.SGD(adapted_policy.parameters(), lr=inner_lr)

            # K adaptation steps
            for _ in range(K):
                optimizer.zero_grad()
                loss = env.compute_loss_differentiable(
                    adapted_policy, task_features, task
                )
                loss.backward()
                optimizer.step()

            # Evaluate adapted policy
            adapted_policy.eval()
            with torch.no_grad():
                controls_post = adapted_policy(task_features).cpu().numpy()
                controls_post = controls_post.reshape(env.n_seg, env.num_controls)

            fid_post = env.compute_fidelity(controls_post, task)
            results['post_adapt_fidelities'][K].append(fid_post)
            results['adaptation_gains'][K].append(fid_post - fid_pre)

    print(f"\n  Evaluation complete!")

    # Compute statistics
    results['pre_adapt_mean'] = np.mean(results['pre_adapt_fidelities'])
    results['pre_adapt_std'] = np.std(results['pre_adapt_fidelities'])

    for K in k_values:
        fids = results['post_adapt_fidelities'][K]
        results[f'post_adapt_mean_K{K}'] = np.mean(fids)
        results[f'post_adapt_std_K{K}'] = np.std(fids)

        gains = results['adaptation_gains'][K]
        results[f'adaptation_gain_mean_K{K}'] = np.mean(gains)
        results[f'adaptation_gain_std_K{K}'] = np.std(gains)

    return results


def plot_fidelity_vs_k(
    results: Dict,
    save_path: str = "results/fidelity_vs_k.pdf",
    figsize: Tuple[float, float] = (8, 6),
    dpi: int = 300
):
    """Plot mean fidelity vs K."""
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=figsize)

    k_values = results['k_values']
    pre_mean = results['pre_adapt_mean']
    pre_std = results['pre_adapt_std']

    post_means = [results[f'post_adapt_mean_K{K}'] for K in k_values]
    post_stds = [results[f'post_adapt_std_K{K}'] / np.sqrt(len(results['pre_adapt_fidelities']))
                 for K in k_values]

    # Plot pre-adaptation baseline
    ax.axhline(
        y=pre_mean, color='gray', linestyle='--', linewidth=2,
        label=f'Pre-Adaptation: {pre_mean:.4f}'
    )
    ax.fill_between(
        [0, max(k_values)],
        pre_mean - pre_std,
        pre_mean + pre_std,
        color='gray', alpha=0.2
    )

    # Plot post-adaptation
    ax.errorbar(
        k_values, post_means, yerr=post_stds,
        fmt='o-', markersize=8, capsize=5, capthick=2,
        label='Post-Adaptation', color='steelblue', linewidth=2
    )

    ax.set_xlabel('Adaptation Steps (K)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Gate Fidelity', fontsize=14, fontweight='bold')
    ax.set_title('Post-Adaptation Fidelity vs Adaptation Steps',
                 fontsize=16, fontweight='bold')
    ax.set_ylim([0.5, 1.0])
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved fidelity vs K plot to {save_path}")
    plt.close()


def plot_adaptation_gain_vs_k(
    results: Dict,
    save_path: str = "results/adaptation_gain_vs_k.pdf",
    figsize: Tuple[float, float] = (8, 6),
    dpi: int = 300
):
    """Plot adaptation gain (improvement) vs K."""
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=figsize)

    k_values = results['k_values']
    gain_means = [results[f'adaptation_gain_mean_K{K}'] for K in k_values]
    gain_stds = [results[f'adaptation_gain_std_K{K}'] / np.sqrt(len(results['pre_adapt_fidelities']))
                 for K in k_values]

    ax.errorbar(
        k_values, gain_means, yerr=gain_stds,
        fmt='o-', markersize=8, capsize=5, capthick=2,
        color='darkgreen', linewidth=2
    )

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_xlabel('Adaptation Steps (K)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Adaptation Gain (Fidelity Improvement)', fontsize=14, fontweight='bold')
    ax.set_title('Meta-Learning Adaptation Gain vs K', fontsize=16, fontweight='bold')
    ax.legend([f'Mean Gain'], fontsize=12, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved adaptation gain vs K plot to {save_path}")
    plt.close()


def plot_fidelity_distributions(
    results: Dict,
    save_path: str = "results/fidelity_distributions.pdf",
    figsize: Tuple[float, float] = (12, 8),
    dpi: int = 300
):
    """Plot fidelity distributions for different K values."""
    sns.set_style("whitegrid")

    k_values = results['k_values']
    n_plots = len(k_values)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_plots > 1 else [axes]

    for idx, K in enumerate(k_values):
        ax = axes[idx]

        pre_fids = results['pre_adapt_fidelities']
        post_fids = results['post_adapt_fidelities'][K]

        # Plot histograms
        ax.hist(pre_fids, bins=20, alpha=0.5, label='Pre-Adaptation',
                color='gray', edgecolor='black')
        ax.hist(post_fids, bins=20, alpha=0.5, label='Post-Adaptation',
                color='steelblue', edgecolor='black')

        ax.set_xlabel('Fidelity', fontsize=10, fontweight='bold')
        ax.set_ylabel('Count', fontsize=10, fontweight='bold')
        ax.set_title(f'K = {K}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Fidelity Distributions: Pre vs Post Adaptation',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved fidelity distributions to {save_path}")
    plt.close()


def plot_combined_analysis(
    results: Dict,
    save_path: str = "results/combined_fidelity_analysis.pdf",
    figsize: Tuple[float, float] = (14, 10),
    dpi: int = 300
):
    """Create comprehensive 2x2 plot with all fidelity analyses."""
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    k_values = results['k_values']

    # Top left: Fidelity vs K
    ax = axes[0, 0]
    pre_mean = results['pre_adapt_mean']
    post_means = [results[f'post_adapt_mean_K{K}'] for K in k_values]
    post_stds = [results[f'post_adapt_std_K{K}'] / np.sqrt(len(results['pre_adapt_fidelities']))
                 for K in k_values]

    ax.axhline(y=pre_mean, color='gray', linestyle='--', linewidth=2,
               label='Pre-Adaptation')
    ax.errorbar(k_values, post_means, yerr=post_stds, fmt='o-', markersize=8,
                capsize=5, label='Post-Adaptation', color='steelblue', linewidth=2)
    ax.set_xlabel('Adaptation Steps (K)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Fidelity', fontsize=12, fontweight='bold')
    ax.set_title('(a) Fidelity vs K', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Top right: Adaptation gain vs K
    ax = axes[0, 1]
    gain_means = [results[f'adaptation_gain_mean_K{K}'] for K in k_values]
    gain_stds = [results[f'adaptation_gain_std_K{K}'] / np.sqrt(len(results['pre_adapt_fidelities']))
                 for K in k_values]

    ax.errorbar(k_values, gain_means, yerr=gain_stds, fmt='o-', markersize=8,
                capsize=5, color='darkgreen', linewidth=2)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Adaptation Steps (K)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Adaptation Gain', fontsize=12, fontweight='bold')
    ax.set_title('(b) Adaptation Gain vs K', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Bottom left: Pre vs Post scatter for K=5
    ax = axes[1, 0]
    K_mid = k_values[len(k_values)//2]  # Middle K value
    pre_fids = results['pre_adapt_fidelities']
    post_fids = results['post_adapt_fidelities'][K_mid]

    ax.scatter(pre_fids, post_fids, alpha=0.5, s=30, color='steelblue')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5, label='No improvement')
    ax.set_xlabel('Pre-Adaptation Fidelity', fontsize=12, fontweight='bold')
    ax.set_ylabel('Post-Adaptation Fidelity', fontsize=12, fontweight='bold')
    ax.set_title(f'(c) Pre vs Post (K={K_mid})', fontsize=14, fontweight='bold')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Bottom right: Box plot of fidelities for selected K values
    ax = axes[1, 1]
    K_subset = [k_values[0], k_values[len(k_values)//2], k_values[-1]]
    data_to_plot = []
    labels = []

    # Add pre-adaptation
    data_to_plot.append(results['pre_adapt_fidelities'])
    labels.append('Pre')

    # Add post-adaptation for selected K
    for K in K_subset:
        data_to_plot.append(results['post_adapt_fidelities'][K])
        labels.append(f'K={K}')

    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', alpha=0.7),
                    medianprops=dict(color='darkred', linewidth=2))

    ax.set_ylabel('Fidelity', fontsize=12, fontweight='bold')
    ax.set_title('(d) Fidelity Distributions', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Comprehensive Fidelity Analysis', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved combined analysis to {save_path}")
    plt.close()


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot fidelity vs adaptation steps K from checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained policy checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="../../configs/experiment_config.yaml",
        help="Path to experiment config"
    )
    parser.add_argument(
        "--k_values",
        type=int,
        nargs='+',
        default=[1, 2, 3, 5, 7, 10, 15, 20],
        help="K values to test"
    )
    parser.add_argument(
        "--n_tasks",
        type=int,
        default=100,
        help="Number of test tasks"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/fidelity_vs_k",
        help="Output directory"
    )
    parser.add_argument(
        "--inner_lr",
        type=float,
        default=0.01,
        help="Inner loop learning rate"
    )

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("PLOT: Fidelity vs Adaptation Steps K")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {args.config}")
    print(f"K values: {args.k_values}")
    print(f"Test tasks: {args.n_tasks}")
    print(f"Output: {args.output_dir}")
    print("=" * 80)

    # Load policy
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    policy = load_policy_from_checkpoint(
        checkpoint_path=args.checkpoint,
        input_dim=3,
        output_dim=config['num_segments'] * config['num_controls'],
        hidden_dims=config['policy_hidden_dims'],
        device=device
    )

    # Create environment
    print("\nCreating quantum environment...")
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    ket_0 = np.array([1, 0], dtype=complex)
    U_target = TargetGates.pauli_x()
    target_state = np.outer(U_target @ ket_0, (U_target @ ket_0).conj())

    env = create_quantum_environment(config, target_state)

    # Sample test tasks
    print(f"\nSampling {args.n_tasks} test tasks...")
    task_dist = TaskDistribution(
        dist_type=config.get('task_dist_type', 'uniform'),
        ranges={
            'alpha': tuple(config.get('alpha_range', [0.5, 2.0])),
            'A': tuple(config.get('A_range', [0.05, 0.3])),
            'omega_c': tuple(config.get('omega_c_range', [2.0, 8.0]))
        }
    )
    test_tasks = task_dist.sample(args.n_tasks)

    # Evaluate
    results = evaluate_fidelity_vs_k(
        policy=policy,
        env=env,
        test_tasks=test_tasks,
        k_values=args.k_values,
        inner_lr=args.inner_lr,
        device=device
    )

    # Save results
    results_to_save = {
        'k_values': results['k_values'],
        'pre_adapt_mean': float(results['pre_adapt_mean']),
        'pre_adapt_std': float(results['pre_adapt_std']),
    }

    for K in args.k_values:
        results_to_save[f'post_adapt_mean_K{K}'] = float(results[f'post_adapt_mean_K{K}'])
        results_to_save[f'post_adapt_std_K{K}'] = float(results[f'post_adapt_std_K{K}'])
        results_to_save[f'adaptation_gain_mean_K{K}'] = float(results[f'adaptation_gain_mean_K{K}'])
        results_to_save[f'adaptation_gain_std_K{K}'] = float(results[f'adaptation_gain_std_K{K}'])

    with open(f"{args.output_dir}/results.json", 'w') as f:
        json.dump(results_to_save, f, indent=2)

    # Generate plots
    print("\nGenerating plots...")
    plot_fidelity_vs_k(results, f"{args.output_dir}/fidelity_vs_k.pdf")
    plot_adaptation_gain_vs_k(results, f"{args.output_dir}/adaptation_gain_vs_k.pdf")
    plot_fidelity_distributions(results, f"{args.output_dir}/fidelity_distributions.pdf")
    plot_combined_analysis(results, f"{args.output_dir}/combined_analysis.pdf")

    print("\n" + "=" * 80)
    print("PLOTTING COMPLETE")
    print("=" * 80)
    print(f"\nGenerated plots:")
    print(f"  1. {args.output_dir}/fidelity_vs_k.pdf")
    print(f"  2. {args.output_dir}/adaptation_gain_vs_k.pdf")
    print(f"  3. {args.output_dir}/fidelity_distributions.pdf")
    print(f"  4. {args.output_dir}/combined_analysis.pdf")
    print(f"\nResults saved to {args.output_dir}/results.json")
    print()


if __name__ == "__main__":
    main()
