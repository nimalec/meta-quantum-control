"""
Inner Learning Rate Sensitivity Ablation
=========================================

Generates adaptation gap curves G_K for different inner learning rates:
α_inner ∈ {0.001, 0.005, 0.01, 0.05, 0.1}

Shows that the scaling law holds across learning rates, with β parameter
changing predictably with learning rate.

Usage:
    python generate_lr_sensitivity_figure.py
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


def compute_adaptation_gap_vs_K(meta_policy, tasks, max_K=20, inner_lr=0.01):
    """Compute adaptation gap G_K for each K."""
    n_tasks = len(tasks)
    all_gaps = np.zeros((n_tasks, max_K + 1))

    for task_idx, task in enumerate(tasks):
        # Clone policy for adaptation
        adapted_policy = deepcopy(meta_policy)
        inner_opt = optim.SGD(adapted_policy.parameters(), lr=inner_lr)

        # Get initial loss
        with torch.no_grad():
            L_0 = compute_loss(adapted_policy, task).item()
            all_gaps[task_idx, 0] = 0  # Gap at K=0 is 0 (relative to self)

        # Compute gap at each K (relative to L_0)
        for k in range(1, max_K + 1):
            inner_opt.zero_grad()
            loss = compute_loss(adapted_policy, task)
            loss.backward()
            inner_opt.step()

            with torch.no_grad():
                L_K = compute_loss(adapted_policy, task).item()
                all_gaps[task_idx, k] = L_0 - L_K  # Improvement from adaptation

    K_values = np.arange(max_K + 1)
    mean_gaps = np.mean(all_gaps, axis=0)
    std_gaps = np.std(all_gaps, axis=0)

    return K_values, mean_gaps, std_gaps


def exponential_saturation(K, A_inf, beta, G0=0):
    """Exponential saturation model: G_K = G0 + A_inf(1 - e^(-beta*K))"""
    return G0 + A_inf * (1 - np.exp(-beta * K))


def fit_exponential(K_values, mean_gaps):
    """Fit exponential saturation model."""
    A_inf_init = mean_gaps[-1] * 1.1 if mean_gaps[-1] > 0 else 0.05
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
    parser = argparse.ArgumentParser(description='Inner LR sensitivity ablation')
    parser.add_argument('--n_tasks', type=int, default=30, help='Number of tasks')
    parser.add_argument('--max_K', type=int, default=25, help='Maximum K')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Learning rates to test
    learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1]
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12']

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

    # Sample tasks (same for all LRs for fair comparison)
    print(f"\nSampling {args.n_tasks} tasks...")
    tasks = sample_tasks(args.n_tasks, sigma_squared=0.001, device=device, seed=42)

    # Compute adaptation curves for each learning rate
    results = {}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    print("\n" + "=" * 60)
    print("Computing adaptation curves for different learning rates...")
    print("=" * 60)

    for lr, color in zip(learning_rates, colors):
        print(f"\nProcessing α_inner = {lr}...")

        K_values, mean_gaps, std_gaps = compute_adaptation_gap_vs_K(
            meta_policy, tasks, max_K=args.max_K, inner_lr=lr
        )

        A_inf, beta, G0, R_sq, fitted = fit_exponential(K_values, mean_gaps)

        results[lr] = {
            'K_values': K_values.tolist(),
            'mean_gaps': mean_gaps.tolist(),
            'std_gaps': std_gaps.tolist(),
            'A_inf': float(A_inf) if A_inf else None,
            'beta': float(beta) if beta else None,
            'R_squared': float(R_sq) if R_sq else None,
        }

        if A_inf is not None:
            print(f"  Fit: A_inf={A_inf:.4f}, β={beta:.4f}, R²={R_sq:.4f}")

        # Plot on left panel
        axes[0].plot(K_values, mean_gaps, 'o-', color=color, markersize=4,
                     label=f'α={lr}', alpha=0.8)

    # Panel (a): Adaptation curves
    axes[0].set_xlabel('Inner-loop Steps $K$')
    axes[0].set_ylabel('Adaptation Gap $G_K$')
    axes[0].set_title('(a) Adaptation Gap vs K for Different Learning Rates')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(-0.5, args.max_K + 0.5)
    axes[0].set_ylim(0, None)

    # Panel (b): Beta vs learning rate
    lrs_with_fit = [lr for lr in learning_rates if results[lr]['beta'] is not None]
    betas = [results[lr]['beta'] for lr in lrs_with_fit]

    axes[1].plot(lrs_with_fit, betas, 'o-', color='#e74c3c', markersize=10, linewidth=2)
    axes[1].set_xlabel('Inner Learning Rate $\\alpha_{inner}$')
    axes[1].set_ylabel('Rate Constant $\\beta$')
    axes[1].set_title('(b) Adaptation Rate vs Learning Rate')
    axes[1].set_xscale('log')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_dir = Path(__file__).parent
    save_path = output_dir / "inner_lr_sensitivity.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(str(save_path).replace('.png', '.pdf'), bbox_inches='tight')
    print(f"\nFigure saved to: {save_path}")

    # Save data
    json_path = output_dir / "inner_lr_sensitivity_data.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Data saved to: {json_path}")

    plt.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
