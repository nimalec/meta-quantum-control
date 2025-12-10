"""
Number of Adaptation Steps Analysis
===================================

Generates plots showing final fidelity vs K for different task difficulties.
Shows diminishing returns and provides practical guidance on when to stop adapting.

Usage:
    python generate_adaptation_steps_figure.py
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


def create_task_with_difficulty(difficulty_level, device='cpu'):
    """
    Create a task with specified difficulty level.

    difficulty_level: 'easy', 'medium', 'hard', 'very_hard'
    """
    difficulty_params = {
        'easy': {'gamma_deph': 0.01, 'gamma_relax': 0.005},
        'medium': {'gamma_deph': 0.05, 'gamma_relax': 0.025},
        'hard': {'gamma_deph': 0.10, 'gamma_relax': 0.05},
        'very_hard': {'gamma_deph': 0.15, 'gamma_relax': 0.08},
    }

    params = difficulty_params[difficulty_level]
    gamma_deph = params['gamma_deph']
    gamma_relax = params['gamma_relax']

    sim = create_single_qubit_system(gamma_deph, gamma_relax, device)

    rho0 = torch.tensor([[1, 0], [0, 0]], dtype=torch.complex64, device=device)
    target_rho = torch.tensor([[0, 0], [0, 1]], dtype=torch.complex64, device=device)

    task_features = torch.tensor([
        gamma_deph / 0.1,
        gamma_relax / 0.05,
        (gamma_deph + gamma_relax) / 0.15
    ], dtype=torch.float32, device=device)

    return {
        'task_features': task_features,
        'simulator': sim,
        'rho0': rho0,
        'target_rho': target_rho,
        'T': 1.0,
        'gamma_deph': gamma_deph,
        'gamma_relax': gamma_relax,
        'difficulty': difficulty_level,
    }


def compute_fidelity_vs_K(meta_policy, task, max_K=50, inner_lr=0.01):
    """Compute fidelity at each adaptation step K."""
    fidelities = []

    adapted_policy = deepcopy(meta_policy)
    inner_opt = optim.SGD(adapted_policy.parameters(), lr=inner_lr)

    # K=0: initial fidelity
    with torch.no_grad():
        loss = compute_loss(adapted_policy, task).item()
        fidelities.append(1.0 - loss)

    # K=1 to max_K
    for k in range(1, max_K + 1):
        inner_opt.zero_grad()
        loss = compute_loss(adapted_policy, task)
        loss.backward()
        inner_opt.step()

        with torch.no_grad():
            loss_val = compute_loss(adapted_policy, task).item()
            fidelities.append(1.0 - loss_val)

    return np.array(fidelities)


def main():
    parser = argparse.ArgumentParser(description='Adaptation steps analysis')
    parser.add_argument('--max_K', type=int, default=50, help='Maximum K')
    parser.add_argument('--inner_lr', type=float, default=0.01, help='Inner learning rate')
    parser.add_argument('--n_trials', type=int, default=10, help='Number of trials per difficulty')
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

    difficulties = ['easy', 'medium', 'hard', 'very_hard']
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
    labels = ['Easy (γ=0.01)', 'Medium (γ=0.05)', 'Hard (γ=0.10)', 'Very Hard (γ=0.15)']

    results = {}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    print("\n" + "=" * 60)
    print("Computing fidelity curves for different task difficulties...")
    print("=" * 60)

    K_values = np.arange(args.max_K + 1)

    for diff, color, label in zip(difficulties, colors, labels):
        print(f"\nProcessing {diff} tasks...")

        # Run multiple trials and average
        all_fidelities = []
        for trial in range(args.n_trials):
            task = create_task_with_difficulty(diff, device=device)
            fidelities = compute_fidelity_vs_K(meta_policy, task, args.max_K, args.inner_lr)
            all_fidelities.append(fidelities)

        mean_fidelities = np.mean(all_fidelities, axis=0)
        std_fidelities = np.std(all_fidelities, axis=0)

        results[diff] = {
            'K_values': K_values.tolist(),
            'mean_fidelities': mean_fidelities.tolist(),
            'std_fidelities': std_fidelities.tolist(),
            'final_fidelity': float(mean_fidelities[-1]),
        }

        print(f"  Initial fidelity: {mean_fidelities[0]:.4f}")
        print(f"  Final fidelity (K={args.max_K}): {mean_fidelities[-1]:.4f}")

        # Plot fidelity curve
        axes[0].plot(K_values, mean_fidelities, '-', color=color, linewidth=2, label=label)
        axes[0].fill_between(K_values,
                             mean_fidelities - std_fidelities,
                             mean_fidelities + std_fidelities,
                             color=color, alpha=0.2)

    # Panel (a): Fidelity vs K
    axes[0].set_xlabel('Inner-loop Steps $K$')
    axes[0].set_ylabel('Fidelity $F$')
    axes[0].set_title('(a) Fidelity vs Adaptation Steps')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, args.max_K)
    axes[0].set_ylim(0.5, 1.0)
    axes[0].axhline(y=0.99, color='gray', linestyle='--', alpha=0.5, label='F=0.99')

    # Panel (b): Marginal improvement (derivative)
    for diff, color, label in zip(difficulties, colors, labels):
        mean_fids = np.array(results[diff]['mean_fidelities'])
        # Compute marginal improvement
        marginal = np.diff(mean_fids)
        axes[1].plot(K_values[1:], marginal, '-', color=color, linewidth=2, label=label)

    axes[1].set_xlabel('Inner-loop Steps $K$')
    axes[1].set_ylabel('Marginal Fidelity Improvement $\\Delta F$')
    axes[1].set_title('(b) Diminishing Returns Analysis')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, args.max_K)
    axes[1].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    axes[1].axhline(y=0.001, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()

    # Save figure
    output_dir = Path(__file__).parent
    save_path = output_dir / "adaptation_steps_analysis.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(str(save_path).replace('.png', '.pdf'), bbox_inches='tight')
    print(f"\nFigure saved to: {save_path}")

    # Save data
    json_path = output_dir / "adaptation_steps_data.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Data saved to: {json_path}")

    # Print practical recommendations
    print("\n" + "=" * 60)
    print("Practical Recommendations:")
    print("=" * 60)
    for diff in difficulties:
        mean_fids = np.array(results[diff]['mean_fidelities'])
        # Find K where 95% of improvement is achieved
        total_improvement = mean_fids[-1] - mean_fids[0]
        if total_improvement > 0:
            threshold = mean_fids[0] + 0.95 * total_improvement
            K_95 = np.argmax(mean_fids >= threshold)
            print(f"  {diff}: 95% improvement at K={K_95}")

    plt.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
