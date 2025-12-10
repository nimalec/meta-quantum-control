"""
Robust Baseline Variants Comparison
===================================

Compares three baseline approaches:
1. Fixed Average Task (standard robust baseline)
2. Domain Randomization (trained on full distribution)
3. Worst-Case Robust (minimax optimization)

Shows the adaptation gap of meta-learned policy vs each baseline.

Usage:
    python generate_baseline_comparison_figure.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from copy import deepcopy
import json
import argparse

from metaqctrl.meta_rl.policy import PulsePolicy
from metaqctrl.quantum.lindblad_torch import DifferentiableLindbladSimulator
from metaqctrl.utils.checkpoint_utils import load_policy_from_checkpoint

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'font.family': 'serif',
    'figure.dpi': 150,
})


def create_single_qubit_system(gamma_deph=0.05, gamma_relax=0.02, device='cpu'):
    """Create a single-qubit Lindblad simulator with given noise rates."""
    sigma_x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device=device)
    sigma_y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64, device=device)
    sigma_z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=device)

    H0 = 0.5 * sigma_z
    H_controls = [sigma_x, sigma_y]

    L_ops = []
    if gamma_deph > 0:
        L_ops.append(np.sqrt(gamma_deph) * sigma_z)
    if gamma_relax > 0:
        L_ops.append(np.sqrt(gamma_relax) * torch.tensor([[0, 1], [0, 0]], dtype=torch.complex64, device=device))

    sim = DifferentiableLindbladSimulator(
        H0=H0,
        H_controls=H_controls,
        L_operators=L_ops,
        dt=0.01,
        method='rk4',
        device=torch.device(device)
    )
    return sim


def compute_fidelity(rho, target_rho):
    """Compute state fidelity F(rho, target)."""
    fid = torch.trace(rho @ target_rho).real
    return torch.clamp(fid, 0, 1)


def compute_loss(policy, task_data):
    """Compute loss for a task: L = 1 - F(rho_final, rho_target)."""
    task_features = task_data['task_features']
    sim = task_data['simulator']
    rho0 = task_data['rho0']
    target_rho = task_data['target_rho']
    T = task_data['T']

    controls = policy(task_features)
    rho_final = sim(rho0, controls, T)
    fidelity = compute_fidelity(rho_final, target_rho)
    return 1.0 - fidelity


def sample_tasks(n_tasks, sigma_squared=0.001, device='cpu', seed=None):
    """Sample tasks with specified diversity."""
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    tasks = []
    mean_gamma_deph = 0.05
    mean_gamma_relax = 0.025
    std_gamma = np.sqrt(sigma_squared)

    rho0 = torch.tensor([[1, 0], [0, 0]], dtype=torch.complex64, device=device)
    target_rho = torch.tensor([[0, 0], [0, 1]], dtype=torch.complex64, device=device)

    for i in range(n_tasks):
        gamma_deph = np.clip(np.random.normal(mean_gamma_deph, std_gamma), 0.0001, 0.15)
        gamma_relax = np.clip(np.random.normal(mean_gamma_relax, std_gamma * 0.5), 0.0001, 0.08)

        sim = create_single_qubit_system(gamma_deph, gamma_relax, device)

        task_features = torch.tensor([
            gamma_deph / 0.1,
            gamma_relax / 0.05,
            (gamma_deph + gamma_relax) / 0.15
        ], dtype=torch.float32, device=device)

        tasks.append({
            'task_features': task_features,
            'simulator': sim,
            'rho0': rho0,
            'target_rho': target_rho,
            'T': 1.0,
            'gamma_deph': gamma_deph,
            'gamma_relax': gamma_relax,
        })

    return tasks


def get_default_config():
    """Get default policy configuration."""
    return {
        'task_feature_dim': 3,
        'hidden_dim': 128,
        'n_hidden_layers': 2,
        'n_segments': 60,
        'n_controls': 2,
        'output_scale': 1.0,
    }


def train_fixed_average_baseline(n_iterations=300, device='cpu'):
    """
    Baseline 1: Fixed Average Task
    Train on a single fixed task with average noise parameters.
    """
    print("  Training Fixed Average baseline...")
    config = get_default_config()

    policy = PulsePolicy(
        task_feature_dim=config['task_feature_dim'],
        hidden_dim=config['hidden_dim'],
        n_hidden_layers=config['n_hidden_layers'],
        n_segments=config['n_segments'],
        n_controls=config['n_controls'],
        output_scale=config['output_scale']
    ).to(device)

    optimizer = optim.Adam(policy.parameters(), lr=1e-3)

    # Fixed average noise level
    avg_gamma_deph = 0.05
    avg_gamma_relax = 0.025

    sim = create_single_qubit_system(avg_gamma_deph, avg_gamma_relax, device)
    rho0 = torch.tensor([[1, 0], [0, 0]], dtype=torch.complex64, device=device)
    target_rho = torch.tensor([[0, 0], [0, 1]], dtype=torch.complex64, device=device)

    fixed_task = {
        'task_features': torch.tensor([
            avg_gamma_deph / 0.1,
            avg_gamma_relax / 0.05,
            (avg_gamma_deph + avg_gamma_relax) / 0.15
        ], dtype=torch.float32, device=device),
        'simulator': sim,
        'rho0': rho0,
        'target_rho': target_rho,
        'T': 1.0,
    }

    for iteration in range(n_iterations):
        optimizer.zero_grad()
        loss = compute_loss(policy, fixed_task)
        loss.backward()
        optimizer.step()

    return policy


def train_domain_randomization_baseline(n_iterations=300, device='cpu'):
    """
    Baseline 2: Domain Randomization
    Train on randomly sampled tasks from the full distribution.
    """
    print("  Training Domain Randomization baseline...")
    config = get_default_config()

    policy = PulsePolicy(
        task_feature_dim=config['task_feature_dim'],
        hidden_dim=config['hidden_dim'],
        n_hidden_layers=config['n_hidden_layers'],
        n_segments=config['n_segments'],
        n_controls=config['n_controls'],
        output_scale=config['output_scale']
    ).to(device)

    optimizer = optim.Adam(policy.parameters(), lr=1e-3)

    rho0 = torch.tensor([[1, 0], [0, 0]], dtype=torch.complex64, device=device)
    target_rho = torch.tensor([[0, 0], [0, 1]], dtype=torch.complex64, device=device)

    for iteration in range(n_iterations):
        # Sample random task
        gamma_deph = np.clip(np.random.normal(0.05, 0.03), 0.001, 0.15)
        gamma_relax = np.clip(np.random.normal(0.025, 0.015), 0.001, 0.08)

        sim = create_single_qubit_system(gamma_deph, gamma_relax, device)

        task = {
            'task_features': torch.tensor([
                gamma_deph / 0.1,
                gamma_relax / 0.05,
                (gamma_deph + gamma_relax) / 0.15
            ], dtype=torch.float32, device=device),
            'simulator': sim,
            'rho0': rho0,
            'target_rho': target_rho,
            'T': 1.0,
        }

        optimizer.zero_grad()
        loss = compute_loss(policy, task)
        loss.backward()
        optimizer.step()

    return policy


def train_worst_case_baseline(n_iterations=300, device='cpu'):
    """
    Baseline 3: Worst-Case Robust (Minimax)
    Train on the worst-case task (highest noise).
    """
    print("  Training Worst-Case Robust baseline...")
    config = get_default_config()

    policy = PulsePolicy(
        task_feature_dim=config['task_feature_dim'],
        hidden_dim=config['hidden_dim'],
        n_hidden_layers=config['n_hidden_layers'],
        n_segments=config['n_segments'],
        n_controls=config['n_controls'],
        output_scale=config['output_scale']
    ).to(device)

    optimizer = optim.Adam(policy.parameters(), lr=1e-3)

    # Worst-case: high noise
    worst_gamma_deph = 0.12
    worst_gamma_relax = 0.06

    sim = create_single_qubit_system(worst_gamma_deph, worst_gamma_relax, device)
    rho0 = torch.tensor([[1, 0], [0, 0]], dtype=torch.complex64, device=device)
    target_rho = torch.tensor([[0, 0], [0, 1]], dtype=torch.complex64, device=device)

    worst_task = {
        'task_features': torch.tensor([
            worst_gamma_deph / 0.1,
            worst_gamma_relax / 0.05,
            (worst_gamma_deph + worst_gamma_relax) / 0.15
        ], dtype=torch.float32, device=device),
        'simulator': sim,
        'rho0': rho0,
        'target_rho': target_rho,
        'T': 1.0,
    }

    for iteration in range(n_iterations):
        optimizer.zero_grad()
        loss = compute_loss(policy, worst_task)
        loss.backward()
        optimizer.step()

    return policy


def evaluate_policy_on_tasks(policy, tasks, K_adapt=0, inner_lr=0.01):
    """Evaluate a policy on tasks, optionally with adaptation."""
    fidelities = []

    for task in tasks:
        if K_adapt > 0:
            # Adapt the policy
            adapted_policy = deepcopy(policy)
            inner_opt = optim.SGD(adapted_policy.parameters(), lr=inner_lr)

            for k in range(K_adapt):
                inner_opt.zero_grad()
                loss = compute_loss(adapted_policy, task)
                loss.backward()
                inner_opt.step()

            with torch.no_grad():
                loss = compute_loss(adapted_policy, task).item()
        else:
            with torch.no_grad():
                loss = compute_loss(policy, task).item()

        fidelities.append(1.0 - loss)

    return np.mean(fidelities), np.std(fidelities)


def main():
    parser = argparse.ArgumentParser(description='Baseline variants comparison')
    parser.add_argument('--n_tasks', type=int, default=50, help='Number of test tasks')
    parser.add_argument('--K_adapt', type=int, default=5, help='Adaptation steps for meta-policy')
    parser.add_argument('--inner_lr', type=float, default=0.01, help='Inner learning rate')
    parser.add_argument('--train_iters', type=int, default=300, help='Training iterations for baselines')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    config = get_default_config()

    # Load meta-learned policy
    checkpoint_path = project_root / "experiments" / "checkpoints" / "maml_best_pauli_x_best.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    print("Loading pre-trained meta-policy...")
    meta_policy = load_policy_from_checkpoint(
        str(checkpoint_path),
        config,
        device=torch.device(device),
        eval_mode=False,
        verbose=True
    )

    # Train baseline variants
    print("\n" + "=" * 60)
    print("Training baseline variants...")
    print("=" * 60)

    baselines = {
        'fixed_average': train_fixed_average_baseline(args.train_iters, device),
        'domain_randomization': train_domain_randomization_baseline(args.train_iters, device),
        'worst_case': train_worst_case_baseline(args.train_iters, device),
    }

    # Sample test tasks
    print("\n" + "=" * 60)
    print(f"Sampling {args.n_tasks} test tasks...")
    print("=" * 60)

    tasks = sample_tasks(args.n_tasks, sigma_squared=0.002, device=device, seed=42)

    # Evaluate all methods
    print("\n" + "=" * 60)
    print("Evaluating methods...")
    print("=" * 60)

    results = {}

    # Evaluate baselines (no adaptation)
    baseline_labels = {
        'fixed_average': 'Fixed Average',
        'domain_randomization': 'Domain Rand.',
        'worst_case': 'Worst-Case',
    }

    for name, policy in baselines.items():
        mean_fid, std_fid = evaluate_policy_on_tasks(policy, tasks, K_adapt=0)
        results[name] = {
            'mean_fidelity': float(mean_fid),
            'std_fidelity': float(std_fid),
            'adapted': False,
        }
        print(f"  {baseline_labels[name]}: F = {mean_fid:.4f} ± {std_fid:.4f}")

    # Evaluate meta-policy without adaptation
    mean_fid, std_fid = evaluate_policy_on_tasks(meta_policy, tasks, K_adapt=0)
    results['meta_no_adapt'] = {
        'mean_fidelity': float(mean_fid),
        'std_fidelity': float(std_fid),
        'adapted': False,
    }
    print(f"  Meta (K=0): F = {mean_fid:.4f} ± {std_fid:.4f}")

    # Evaluate meta-policy with adaptation
    mean_fid, std_fid = evaluate_policy_on_tasks(
        meta_policy, tasks, K_adapt=args.K_adapt, inner_lr=args.inner_lr
    )
    results['meta_adapted'] = {
        'mean_fidelity': float(mean_fid),
        'std_fidelity': float(std_fid),
        'adapted': True,
        'K': args.K_adapt,
    }
    print(f"  Meta (K={args.K_adapt}): F = {mean_fid:.4f} ± {std_fid:.4f}")

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel (a): Bar chart of fidelities
    methods = ['fixed_average', 'domain_randomization', 'worst_case', 'meta_no_adapt', 'meta_adapted']
    labels = ['Fixed\nAverage', 'Domain\nRand.', 'Worst\nCase', 'Meta\n(K=0)', f'Meta\n(K={args.K_adapt})']
    colors = ['#bdc3c7', '#95a5a6', '#7f8c8d', '#3498db', '#2ecc71']

    fidelities = [results[m]['mean_fidelity'] for m in methods]
    stds = [results[m]['std_fidelity'] for m in methods]

    bars = axes[0].bar(range(len(methods)), fidelities, yerr=stds, color=colors,
                       capsize=5, alpha=0.8)
    axes[0].set_xlabel('Method')
    axes[0].set_ylabel('Fidelity $F$')
    axes[0].set_title('(a) Fidelity Comparison Across Methods')
    axes[0].set_xticks(range(len(methods)))
    axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_ylim(0.5, 1.0)
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].axhline(y=0.99, color='gray', linestyle='--', alpha=0.5)

    # Panel (b): Adaptation gap vs each baseline
    baseline_names = ['fixed_average', 'domain_randomization', 'worst_case']
    baseline_labels_short = ['Fixed Avg', 'Domain Rand', 'Worst Case']

    meta_adapted_fid = results['meta_adapted']['mean_fidelity']
    gaps = [meta_adapted_fid - results[b]['mean_fidelity'] for b in baseline_names]

    colors_b = ['#e74c3c', '#f39c12', '#9b59b6']
    bars = axes[1].bar(range(len(baseline_names)), gaps, color=colors_b, alpha=0.8)

    axes[1].set_xlabel('Baseline Method')
    axes[1].set_ylabel(f'Adaptation Gap (Meta K={args.K_adapt} - Baseline)')
    axes[1].set_title('(b) Meta-Learning Advantage Over Baselines')
    axes[1].set_xticks(range(len(baseline_names)))
    axes[1].set_xticklabels(baseline_labels_short, fontsize=9)
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    # Add value annotations
    for i, (bar, gap) in enumerate(zip(bars, gaps)):
        axes[1].annotate(f'+{gap:.3f}',
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()

    # Save figure
    output_dir = Path(__file__).parent
    save_path = output_dir / "baseline_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(str(save_path).replace('.png', '.pdf'), bbox_inches='tight')
    print(f"\nFigure saved to: {save_path}")

    # Save data
    json_path = output_dir / "baseline_comparison_data.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Data saved to: {json_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("Summary: Meta-Learning Advantage")
    print("=" * 60)
    for b, label in zip(baseline_names, baseline_labels_short):
        gap = meta_adapted_fid - results[b]['mean_fidelity']
        print(f"  vs {label}: +{gap:.4f} fidelity improvement")

    plt.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
