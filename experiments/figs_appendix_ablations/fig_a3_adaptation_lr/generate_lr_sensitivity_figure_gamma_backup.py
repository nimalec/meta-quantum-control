"""
Inner Learning Rate Sensitivity Ablation - GAMMA-RATE Version
=============================================================

Uses direct gamma_deph/gamma_relax parameterization (Markovian limit).
Trains fresh robust baseline, avoiding checkpoint compatibility issues.

Usage:
    python generate_lr_sensitivity_figure_gamma_backup.py
"""

import sys
from pathlib import Path

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

    # Normalized gamma rates as task features
    task_features = torch.tensor([
        gamma_deph / 0.1,
        gamma_relax / 0.05,
        (gamma_deph + gamma_relax) / 0.15
    ], dtype=torch.float32, device=device)

    controls = policy(task_features)

    rho0 = torch.zeros(2, 2, dtype=torch.complex64, device=device)
    rho0[0, 0] = 1.0

    rho_final = sim.forward(rho0, controls, T=1.0)

    # Target: |1><1| (Pauli-X gate on |0>)
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


def sample_gamma_tasks(n_tasks, diversity_scale=1.0, rng=None):
    """Sample gamma-rate tasks with specified diversity."""
    if rng is None:
        rng = np.random.default_rng(42)

    gamma_deph_center = 0.05
    gamma_relax_center = 0.025
    gamma_deph_std = 0.03 * diversity_scale
    gamma_relax_std = 0.015 * diversity_scale

    tasks = []
    for _ in range(n_tasks):
        gamma_deph = np.clip(rng.normal(gamma_deph_center, gamma_deph_std), 0.001, 0.15)
        gamma_relax = np.clip(rng.normal(gamma_relax_center, gamma_relax_std), 0.001, 0.08)
        tasks.append((gamma_deph, gamma_relax))

    return tasks


def compute_adaptation_gap_vs_K(robust_policy, task_params_list,
                                 max_K=20, inner_lr=0.01, device='cpu'):
    """Compute adaptation gap G_K."""
    n_tasks = len(task_params_list)
    all_gaps = np.zeros((n_tasks, max_K + 1))

    for task_idx, (gamma_deph, gamma_relax) in enumerate(task_params_list):
        adapted_policy = deepcopy(robust_policy)
        inner_opt = optim.SGD(adapted_policy.parameters(), lr=inner_lr)

        with torch.no_grad():
            L_0 = compute_loss_gamma(adapted_policy, gamma_deph, gamma_relax, device).item()
            all_gaps[task_idx, 0] = 0.0

        for k in range(1, max_K + 1):
            inner_opt.zero_grad()
            loss = compute_loss_gamma(adapted_policy, gamma_deph, gamma_relax, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapted_policy.parameters(), max_norm=1.0)
            inner_opt.step()

            with torch.no_grad():
                L_K = compute_loss_gamma(adapted_policy, gamma_deph, gamma_relax, device).item()
                all_gaps[task_idx, k] = L_0 - L_K

    K_values = np.arange(max_K + 1)
    mean_gaps = np.mean(all_gaps, axis=0)
    std_gaps = np.std(all_gaps, axis=0)

    return K_values, mean_gaps, std_gaps


def exponential_saturation(K, A_inf, beta, G0=0):
    return G0 + A_inf * (1 - np.exp(-beta * K))


def fit_exponential(K_values, mean_gaps):
    """Fit exponential saturation model."""
    A_inf_init = max(0.01, mean_gaps[-1] * 1.1)
    beta_init = 0.3

    try:
        popt, _ = curve_fit(
            exponential_saturation,
            K_values[1:],
            mean_gaps[1:],
            p0=[A_inf_init, beta_init, 0],
            bounds=([0, 0.001, -0.1], [1, 10, 0.1]),
            maxfev=5000
        )
        A_inf, beta, G0 = popt

        fitted = exponential_saturation(K_values, A_inf, beta, G0)
        ss_res = np.sum((mean_gaps - fitted) ** 2)
        ss_tot = np.sum((mean_gaps - np.mean(mean_gaps)) ** 2)
        R_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return A_inf, beta, G0, R_squared, fitted
    except Exception as e:
        print(f"  Fitting failed: {e}")
        return None, None, None, None, None


def main():
    parser = argparse.ArgumentParser(description='Inner LR sensitivity (gamma version)')
    parser.add_argument('--n_tasks', type=int, default=20, help='Number of tasks')
    parser.add_argument('--max_K', type=int, default=25, help='Maximum K')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device = torch.device('cpu')
    print(f"Using device: {device}")

    # Train fresh robust baseline
    robust_policy = train_robust_policy_gamma(n_iterations=500, device=device)

    # Sample tasks
    print(f"\nSampling {args.n_tasks} gamma-rate tasks...")
    task_params_list = sample_gamma_tasks(args.n_tasks, diversity_scale=1.0, rng=rng)

    # Learning rates to test
    learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1]
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12']

    results = {}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    print("\n" + "=" * 60)
    print("Computing adaptation curves for different learning rates...")
    print("=" * 60)

    for lr, color in zip(learning_rates, colors):
        print(f"\nProcessing alpha_inner = {lr}...")

        K_values, mean_gaps, std_gaps = compute_adaptation_gap_vs_K(
            robust_policy, task_params_list,
            max_K=args.max_K, inner_lr=lr, device=device
        )

        A_inf, beta, G0, R_sq, fitted = fit_exponential(K_values, mean_gaps)

        results[str(lr)] = {
            'K_values': K_values.tolist(),
            'mean_gaps': mean_gaps.tolist(),
            'std_gaps': std_gaps.tolist(),
            'A_inf': float(A_inf) if A_inf else None,
            'beta': float(beta) if beta else None,
            'R_squared': float(R_sq) if R_sq else None,
        }

        if A_inf is not None:
            print(f"  Fit: A_inf={A_inf:.4f}, beta={beta:.4f}, R^2={R_sq:.4f}")

        # Plot
        axes[0].plot(K_values, mean_gaps, 'o-', color=color, markersize=4,
                     label=f'$\\alpha$={lr}', alpha=0.8)

    # Panel (a): Adaptation curves
    axes[0].set_xlabel('Inner-loop Steps $K$')
    axes[0].set_ylabel('Adaptation Gap $G_K$')
    axes[0].set_title('(a) Adaptation Gap vs K for Different Learning Rates')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(-0.5, args.max_K + 0.5)

    # Panel (b): Beta vs learning rate
    lrs_with_fit = [lr for lr in learning_rates if results[str(lr)]['beta'] is not None]
    betas = [results[str(lr)]['beta'] for lr in lrs_with_fit]

    axes[1].plot(lrs_with_fit, betas, 'o-', color='#e74c3c', markersize=10, linewidth=2)
    axes[1].set_xlabel('Inner Learning Rate $\\alpha_{inner}$')
    axes[1].set_ylabel('Rate Constant $\\beta$')
    axes[1].set_title('(b) Adaptation Rate vs Learning Rate')
    axes[1].set_xscale('log')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    output_dir = Path(__file__).parent
    save_path = output_dir / "inner_lr_sensitivity_gamma_backup.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(str(save_path).replace('.png', '.pdf'), bbox_inches='tight')
    print(f"\nFigure saved to: {save_path}")

    json_path = output_dir / "inner_lr_sensitivity_gamma_backup_data.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Data saved to: {json_path}")

    plt.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
