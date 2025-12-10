"""
Meta-Training Task Distribution Shift Analysis
===============================================

Generates plots comparing adaptation performance when test tasks are
in-distribution vs out-of-distribution.

Shows how robust the meta-initialization is to distribution shift.

Usage:
    python generate_distribution_shift_figure.py
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
from scipy.optimize import curve_fit
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


def sample_tasks_from_distribution(n_tasks, dist_type, device='cpu', seed=None):
    """
    Sample tasks from different distributions.

    dist_type:
        'in_distribution': gamma_deph ~ N(0.05, 0.02), same as training
        'shifted_mean': gamma_deph ~ N(0.10, 0.02), shifted mean
        'high_variance': gamma_deph ~ N(0.05, 0.05), higher variance
        'out_of_distribution': gamma_deph ~ N(0.12, 0.03), both shifted
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    dist_params = {
        'in_distribution': {'mean_deph': 0.05, 'std_deph': 0.02, 'mean_relax': 0.025, 'std_relax': 0.01},
        'shifted_mean': {'mean_deph': 0.10, 'std_deph': 0.02, 'mean_relax': 0.05, 'std_relax': 0.01},
        'high_variance': {'mean_deph': 0.05, 'std_deph': 0.04, 'mean_relax': 0.025, 'std_relax': 0.02},
        'out_of_distribution': {'mean_deph': 0.12, 'std_deph': 0.03, 'mean_relax': 0.06, 'std_relax': 0.015},
    }

    params = dist_params[dist_type]
    tasks = []

    rho0 = torch.tensor([[1, 0], [0, 0]], dtype=torch.complex64, device=device)
    target_rho = torch.tensor([[0, 0], [0, 1]], dtype=torch.complex64, device=device)

    for i in range(n_tasks):
        gamma_deph = np.clip(
            np.random.normal(params['mean_deph'], params['std_deph']),
            0.001, 0.20
        )
        gamma_relax = np.clip(
            np.random.normal(params['mean_relax'], params['std_relax']),
            0.001, 0.10
        )

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


def compute_adaptation_gap_vs_K(meta_policy, tasks, max_K=20, inner_lr=0.01):
    """Compute adaptation gap G_K for each K."""
    n_tasks = len(tasks)
    all_fidelities = np.zeros((n_tasks, max_K + 1))

    for task_idx, task in enumerate(tasks):
        adapted_policy = deepcopy(meta_policy)
        inner_opt = optim.SGD(adapted_policy.parameters(), lr=inner_lr)

        # K=0
        with torch.no_grad():
            loss = compute_loss(adapted_policy, task).item()
            all_fidelities[task_idx, 0] = 1.0 - loss

        # K=1 to max_K
        for k in range(1, max_K + 1):
            inner_opt.zero_grad()
            loss = compute_loss(adapted_policy, task)
            loss.backward()
            inner_opt.step()

            with torch.no_grad():
                loss_val = compute_loss(adapted_policy, task).item()
                all_fidelities[task_idx, k] = 1.0 - loss_val

    K_values = np.arange(max_K + 1)
    mean_fidelities = np.mean(all_fidelities, axis=0)
    std_fidelities = np.std(all_fidelities, axis=0)

    return K_values, mean_fidelities, std_fidelities


def main():
    parser = argparse.ArgumentParser(description='Distribution shift analysis')
    parser.add_argument('--n_tasks', type=int, default=30, help='Number of tasks per distribution')
    parser.add_argument('--max_K', type=int, default=30, help='Maximum K')
    parser.add_argument('--inner_lr', type=float, default=0.01, help='Inner learning rate')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    config = {
        'task_feature_dim': 3,
        'hidden_dim': 128,
        'n_hidden_layers': 2,
        'n_segments': 60,
        'n_controls': 2,
        'output_scale': 1.0,
    }

    # Load checkpoint
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

    distributions = ['in_distribution', 'shifted_mean', 'high_variance', 'out_of_distribution']
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
    labels = ['In-Distribution', 'Shifted Mean', 'High Variance', 'Out-of-Distribution']

    results = {}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    print("\n" + "=" * 60)
    print("Computing adaptation performance for different distributions...")
    print("=" * 60)

    for dist, color, label in zip(distributions, colors, labels):
        print(f"\nProcessing {label}...")

        tasks = sample_tasks_from_distribution(
            args.n_tasks, dist, device=device, seed=42
        )

        K_values, mean_fids, std_fids = compute_adaptation_gap_vs_K(
            meta_policy, tasks, args.max_K, args.inner_lr
        )

        results[dist] = {
            'K_values': K_values.tolist(),
            'mean_fidelities': mean_fids.tolist(),
            'std_fidelities': std_fids.tolist(),
            'initial_fidelity': float(mean_fids[0]),
            'final_fidelity': float(mean_fids[-1]),
            'improvement': float(mean_fids[-1] - mean_fids[0]),
        }

        print(f"  Initial fidelity: {mean_fids[0]:.4f}")
        print(f"  Final fidelity: {mean_fids[-1]:.4f}")
        print(f"  Improvement: {mean_fids[-1] - mean_fids[0]:.4f}")

        # Plot on panel (a)
        axes[0].plot(K_values, mean_fids, '-', color=color, linewidth=2, label=label)
        axes[0].fill_between(K_values,
                             mean_fids - std_fids,
                             mean_fids + std_fids,
                             color=color, alpha=0.2)

    # Panel (a): Fidelity curves
    axes[0].set_xlabel('Inner-loop Steps $K$')
    axes[0].set_ylabel('Fidelity $F$')
    axes[0].set_title('(a) Adaptation Performance vs Distribution Shift')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, args.max_K)
    axes[0].set_ylim(0.4, 1.0)

    # Panel (b): Bar chart comparing initial vs final fidelity
    x = np.arange(len(distributions))
    width = 0.35

    initial_fids = [results[d]['initial_fidelity'] for d in distributions]
    final_fids = [results[d]['final_fidelity'] for d in distributions]

    bars1 = axes[1].bar(x - width/2, initial_fids, width, label='Initial (K=0)', color='#bdc3c7')
    bars2 = axes[1].bar(x + width/2, final_fids, width, label=f'Final (K={args.max_K})', color='#3498db')

    axes[1].set_xlabel('Test Distribution')
    axes[1].set_ylabel('Fidelity $F$')
    axes[1].set_title('(b) Initial vs Adapted Performance')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(['In-Dist', 'Shifted\nMean', 'High\nVar', 'OOD'], fontsize=9)
    axes[1].legend(loc='lower right')
    axes[1].set_ylim(0.4, 1.0)
    axes[1].grid(True, alpha=0.3, axis='y')

    # Add improvement annotations
    for i, (init, final) in enumerate(zip(initial_fids, final_fids)):
        improvement = final - init
        axes[1].annotate(f'+{improvement:.2f}',
                        xy=(i, final + 0.02),
                        ha='center', fontsize=8, color='#27ae60')

    plt.tight_layout()

    # Save figure
    output_dir = Path(__file__).parent
    save_path = output_dir / "distribution_shift_analysis.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(str(save_path).replace('.png', '.pdf'), bbox_inches='tight')
    print(f"\nFigure saved to: {save_path}")

    # Save data
    json_path = output_dir / "distribution_shift_data.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Data saved to: {json_path}")

    plt.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
