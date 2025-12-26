"""
Number of Adaptation Steps Analysis - GAMMA-RATE Version
  
 
Usage:
    python generate_adaptation_steps_figure_gamma_backup.py
"""

import sys
from pathlib import Path

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


def create_single_qubit_system(gamma_deph=0.05, gamma_relax=0.025, device='cpu'):
    """Create simulator with direct gamma rates."""
    sigma_x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device=device)
    sigma_y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64, device=device)
    sigma_z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=device)
    sigma_p = torch.tensor([[0, 1], [0, 0]], dtype=torch.complex64, device=device)

    H0 = 0.1 * sigma_z
    H_controls = [sigma_x, sigma_y]

    L_operators = []
    if gamma_deph > 0:
        L_operators.append(np.sqrt(gamma_deph) * sigma_z)
    if gamma_relax > 0:
        L_operators.append(np.sqrt(gamma_relax) * sigma_p)

    if not L_operators:
        L_operators.append(torch.zeros(2, 2, dtype=torch.complex64, device=device))

    sim = DifferentiableLindbladSimulator(
        H0=H0,
        H_controls=H_controls,
        L_operators=L_operators,
        dt=0.01,
        method='rk4',
        device=device
    )
    return sim


def compute_loss_gamma(policy, gamma_deph, gamma_relax, device='cpu'):
    """Compute loss using gamma-rate task features."""
    sim = create_single_qubit_system(gamma_deph, gamma_relax, device=device)

    task_features = torch.tensor([
        gamma_deph / 0.1,
        gamma_relax / 0.05,
        (gamma_deph + gamma_relax) / 0.15
    ], dtype=torch.float32, device=device)

    controls = policy(task_features)

    rho0 = torch.zeros(2, 2, dtype=torch.complex64, device=device)
    rho0[0, 0] = 1.0

    rho_final = sim.forward(rho0, controls, T=1.0)

    target = torch.zeros(2, 2, dtype=rho_final.dtype, device=device)
    target[1, 1] = 1.0

    fidelity = torch.abs(torch.trace(target @ rho_final)).real
    return 1.0 - fidelity


def train_robust_policy_gamma(n_iterations=500, device='cpu'):
    """Train robust policy on fixed average gamma rates."""
    print("Training robust baseline on FIXED average gamma rates...")

    policy = PulsePolicy(
        task_feature_dim=3,
        hidden_dim=128,
        n_hidden_layers=2,
        n_segments=60,
        n_controls=2,
        output_scale=1.0
    ).to(device)

    optimizer = optim.Adam(policy.parameters(), lr=1e-3)

    avg_gamma_deph = 0.05
    avg_gamma_relax = 0.025

    for iteration in range(n_iterations):
        optimizer.zero_grad()
        loss = compute_loss_gamma(policy, avg_gamma_deph, avg_gamma_relax, device)
        loss.backward()
        optimizer.step()

        if iteration % 100 == 0:
            print(f"  Iter {iteration}: Loss = {loss.item():.4f}")

    print(f"Robust training complete.")
    return policy


def create_task_with_difficulty(difficulty_level):
    """Create gamma-rate tasks for different difficulty levels."""
    difficulty_params = {
        'easy': {'gamma_deph': 0.02, 'gamma_relax': 0.01},
        'medium': {'gamma_deph': 0.05, 'gamma_relax': 0.025},
        'hard': {'gamma_deph': 0.10, 'gamma_relax': 0.05},
        'very_hard': {'gamma_deph': 0.15, 'gamma_relax': 0.08},
    }
    return difficulty_params[difficulty_level]


def compute_fidelity_vs_K(robust_policy, gamma_deph, gamma_relax, device,
                          max_K=50, inner_lr=0.01):
    """Compute fidelity at each adaptation step K."""
    fidelities = []

    adapted_policy = deepcopy(robust_policy)
    inner_opt = optim.SGD(adapted_policy.parameters(), lr=inner_lr)

    # K=0
    with torch.no_grad():
        loss = compute_loss_gamma(adapted_policy, gamma_deph, gamma_relax, device).item()
        fidelities.append(1.0 - loss)

    # K=1 to max_K
    for k in range(1, max_K + 1):
        inner_opt.zero_grad()
        loss = compute_loss_gamma(adapted_policy, gamma_deph, gamma_relax, device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapted_policy.parameters(), max_norm=1.0)
        inner_opt.step()

        with torch.no_grad():
            loss_val = compute_loss_gamma(adapted_policy, gamma_deph, gamma_relax, device).item()
            fidelities.append(1.0 - loss_val)

    return np.array(fidelities)


def main():
    parser = argparse.ArgumentParser(description='Adaptation steps analysis (gamma version)')
    parser.add_argument('--max_K', type=int, default=50, help='Maximum K')
    parser.add_argument('--inner_lr', type=float, default=0.01, help='Inner learning rate')
    parser.add_argument('--n_trials', type=int, default=5, help='Number of trials per difficulty')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cpu')
    print(f"Using device: {device}")

    # Train fresh robust baseline
    robust_policy = train_robust_policy_gamma(n_iterations=500, device=device)

    difficulties = ['easy', 'medium', 'hard', 'very_hard']
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
    labels = ['Easy ($\\gamma_d$=0.02)', 'Medium ($\\gamma_d$=0.05)',
              'Hard ($\\gamma_d$=0.10)', 'Very Hard ($\\gamma_d$=0.15)']

    results = {}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    print("\n" + "=" * 60)
    print("Computing fidelity curves for different task difficulties...")
    print("=" * 60)

    K_values = np.arange(args.max_K + 1)

    for diff, color, label in zip(difficulties, colors, labels):
        print(f"\nProcessing {diff} tasks...")

        params = create_task_with_difficulty(diff)

        # Run multiple trials with slight noise variation
        all_fidelities = []
        rng = np.random.default_rng(args.seed)

        for trial in range(args.n_trials):
            # Add small variation to gamma rates
            gamma_deph = params['gamma_deph'] * (1 + 0.1 * rng.normal())
            gamma_relax = params['gamma_relax'] * (1 + 0.1 * rng.normal())
            gamma_deph = np.clip(gamma_deph, 0.001, 0.2)
            gamma_relax = np.clip(gamma_relax, 0.001, 0.1)

            fidelities = compute_fidelity_vs_K(
                robust_policy, gamma_deph, gamma_relax, device,
                args.max_K, args.inner_lr
            )
            all_fidelities.append(fidelities)

        mean_fidelities = np.mean(all_fidelities, axis=0)
        std_fidelities = np.std(all_fidelities, axis=0)

        results[diff] = {
            'K_values': K_values.tolist(),
            'mean_fidelities': mean_fidelities.tolist(),
            'std_fidelities': std_fidelities.tolist(),
            'final_fidelity': float(mean_fidelities[-1]),
            'gamma_deph': params['gamma_deph'],
            'gamma_relax': params['gamma_relax'],
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
    axes[0].axhline(y=0.99, color='gray', linestyle='--', alpha=0.5)

    # Panel (b): Marginal improvement
    for diff, color, label in zip(difficulties, colors, labels):
        mean_fids = np.array(results[diff]['mean_fidelities'])
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

    # Save
    output_dir = Path(__file__).parent
    save_path = output_dir / "adaptation_steps_gamma_backup.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(str(save_path).replace('.png', '.pdf'), bbox_inches='tight')
    print(f"\nFigure saved to: {save_path}")

    json_path = output_dir / "adaptation_steps_gamma_backup_data.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Data saved to: {json_path}")

    # Print recommendations
    print("\n" + "=" * 60)
    print("Practical Recommendations:")
    print("=" * 60)
    for diff in difficulties:
        mean_fids = np.array(results[diff]['mean_fidelities'])
        total_improvement = mean_fids[-1] - mean_fids[0]
        if total_improvement > 0.001:
            threshold = mean_fids[0] + 0.95 * total_improvement
            K_95 = np.argmax(mean_fids >= threshold)
            print(f"  {diff}: 95% improvement at K={K_95}")
        else:
            print(f"  {diff}: Already near optimal (improvement < 0.001)")

    plt.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
