"""
Generate adaptation gap figure for 2-qubit CZ gate. Uses the J=8.0, 20 segments and inner_lr=2e-4.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress
from copy import deepcopy
import json
import argparse
from dataclasses import dataclass
from typing import List, Tuple
  
from two_qubit_cz_maml_fast import (
    TwoQubitTaskParams,
    TwoQubitTaskDistribution,
    TwoQubitCZPolicy,
    compute_loss,
    CZ_IDEAL_GATE_TIME,
    J_COUPLING,
)

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})


def load_cz_policy(checkpoint_path: str, device: str = 'cpu') -> TwoQubitCZPolicy:
    """Load pretrained CZ policy from checkpoint."""
    print(f"Loading CZ policy from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get architecture from checkpoint or use defaults
    hidden_dim = checkpoint.get('hidden_dim', 256)
    n_hidden_layers = checkpoint.get('n_hidden_layers', 4)
    n_segments = checkpoint.get('n_segments', 20)
    n_controls = checkpoint.get('n_controls', 6)
    task_feature_dim = checkpoint.get('task_feature_dim', 4)
    
    policy = TwoQubitCZPolicy(
        task_feature_dim=task_feature_dim,
        hidden_dim=hidden_dim,
        n_hidden_layers=n_hidden_layers,
        n_segments=n_segments,
        n_controls=n_controls,
    ).to(device)

    if 'policy_state_dict' in checkpoint:
        policy.load_state_dict(checkpoint['policy_state_dict'])
        print(f"  Loaded policy state dict (iteration {checkpoint.get('iteration', 'unknown')})")
    else:
        policy.load_state_dict(checkpoint)
        print("  Loaded policy weights directly")

    policy.eval()
    return policy


def compute_adaptation_gap_vs_K(
    policy: TwoQubitCZPolicy,
    task_params_list: List[TwoQubitTaskParams],
    max_K: int = 30,
    inner_lr: float = 2e-4,  # Using 2e-4 as specified
    device: str = 'cpu',
):
    """Compute adaptation gap G_K for varying K."""
    n_tasks = len(task_params_list)
    all_gaps = np.zeros((n_tasks, max_K + 1))
    all_fidelities = np.zeros((n_tasks, max_K + 1))
    initial_losses = np.zeros(n_tasks)

    for task_idx, task_params in enumerate(task_params_list):

        # Clone policy for adaptation
        adapted_policy = deepcopy(policy)
        adapted_policy.train()
        inner_opt = optim.Adam(adapted_policy.parameters(), lr=inner_lr)

        # Initial loss (K=0)
        with torch.no_grad():
            L_0 = compute_loss(adapted_policy, task_params, device=device).item()
            initial_losses[task_idx] = L_0
            all_gaps[task_idx, 0] = 0.0
            all_fidelities[task_idx, 0] = 1 - L_0

        # Adaptation loop
        for k in range(1, max_K + 1):
            inner_opt.zero_grad()
            loss = compute_loss(adapted_policy, task_params, device=device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapted_policy.parameters(), max_norm=1.0)
            inner_opt.step()

            with torch.no_grad():
                L_K = compute_loss(adapted_policy, task_params, device=device).item()
                all_gaps[task_idx, k] = L_0 - L_K
                all_fidelities[task_idx, k] = 1 - L_K

        print(f"F_0={100*(1-L_0):.1f}%, F_{max_K}={100*all_fidelities[task_idx, max_K]:.1f}%, "
              f"Gap={100*all_gaps[task_idx, max_K]:.1f}%")

    K_values = np.arange(max_K + 1)
    mean_gaps = np.mean(all_gaps, axis=0)
    std_gaps = np.std(all_gaps, axis=0)
    mean_fidelities = np.mean(all_fidelities, axis=0)
    std_fidelities = np.std(all_fidelities, axis=0)
    mean_initial_loss = np.mean(initial_losses)

    return K_values, mean_gaps, std_gaps, mean_fidelities, std_fidelities, mean_initial_loss


def exponential_saturation(K, c, beta):
    """Exponential saturation: G_K = c*(1 - exp(-beta*K))"""
    return c * (1 - np.exp(-beta * K))


def fit_exponential_saturation(K_values, mean_gaps):
    """Fit G_K = c*(1 - exp(-beta*K))."""
    c_init = max(0.01, mean_gaps[-1] * 1.1)
    beta_init = 0.1

    try:
        popt, _ = curve_fit(
            exponential_saturation,
            K_values,
            mean_gaps,
            p0=[c_init, beta_init],
            bounds=([0, 0.001], [2, 10]),
            maxfev=5000
        )
        c, beta = popt

        fitted = exponential_saturation(K_values, c, beta)
        ss_res = np.sum((mean_gaps - fitted) ** 2)
        ss_tot = np.sum((mean_gaps - np.mean(mean_gaps)) ** 2)
        R_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return c, beta, R_squared, fitted
    except Exception as e:
        print(f"  Fitting failed: {e}")
        return None, None, None, None


def sample_cz_tasks(n_tasks: int, diversity_scale: float = 1.0) -> List[TwoQubitTaskParams]:
    """Sample CZ tasks with given diversity scale."""
    task_dist = TwoQubitTaskDistribution(
        gamma_deph_range=(0.001, 0.01),  
        gamma_relax_range=(0.0005, 0.005),
        correlated=True,
        diversity_scale=diversity_scale,
    )
    return task_dist.sample(n_tasks)


def create_figure(K_values, mean_gaps, std_gaps, mean_fidelities, std_fidelities,
                  c, beta, R_squared, inner_lr, save_path=None):
    """Create adaptation gap figure."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel (a): Adaptation Gap vs K
    ax = axes[0]
    ax.errorbar(K_values, mean_gaps * 100, yerr=std_gaps * 100,
                fmt='o', color='#3498db', markersize=5, capsize=3, label='Data')

    if c is not None:
        K_fine = np.linspace(0, K_values[-1], 100)
        fitted_fine = exponential_saturation(K_fine, c, beta) * 100
        ax.plot(K_fine, fitted_fine, '-', color='#e74c3c', linewidth=2,
                label=f'Fit: $G_K = c(1-e^{{-\\beta K}})$')
        ax.text(0.95, 0.05, f'$c = {c*100:.2f}\\%$\n$\\beta = {beta:.3f}$\n$R^2 = {R_squared:.3f}$',
                transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('Inner-loop Steps $K$')
    ax.set_ylabel('Adaptation Gap $G_K$ (%)')
    ax.set_title(f'(a) CZ Gate Adaptation Gap (lr={inner_lr:.0e})')
    ax.set_xlim(-0.5, K_values[-1] + 0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=9)

    ax = axes[1]
    ax.errorbar(K_values, mean_fidelities * 100, yerr=std_fidelities * 100,
                fmt='s', color='#2ecc71', markersize=5, capsize=3, label='Mean Fidelity')
    ax.axhline(y=mean_fidelities[0] * 100, color='gray', linestyle='--',
               label=f'Initial: {mean_fidelities[0]*100:.1f}%')

    ax.set_xlabel('Inner-loop Steps $K$')
    ax.set_ylabel('Gate Fidelity (%)')
    ax.set_title('(b) CZ Gate Fidelity vs Adaptation Steps')
    ax.set_xlim(-0.5, K_values[-1] + 0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_tasks', type=int, default=20)
    parser.add_argument('--max_K', type=int, default=30)
    parser.add_argument('--inner_lr', type=float, default=2e-4)  # Using 2e-4 as specified
    parser.add_argument('--checkpoint', type=str, default='checkpoints_cz_1000iter_fast/maml_cz_best.pt')
    parser.add_argument('--output', type=str, default='cz_adaptation_gap_fast')
    args = parser.parse_args()

    device = 'cpu'


    # Load pretrained CZ policy
    checkpoint_path = Path(__file__).parent / args.checkpoint
    if not checkpoint_path.exists():
        checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists(): 
        return

    policy = load_cz_policy(str(checkpoint_path), device=device)

    # Generate data
    task_params_list = sample_cz_tasks(args.n_tasks, diversity_scale=1.0)

    K_values, mean_gaps, std_gaps, mean_fidelities, std_fidelities, mean_initial_loss = \
        compute_adaptation_gap_vs_K(
            policy, task_params_list, max_K=args.max_K,
            inner_lr=args.inner_lr, device=device
        )

    c, beta, R_squared, fitted_curve = fit_exponential_saturation(K_values, mean_gaps)

    # Print summary 
    if c is not None:
        print(f"\nExponential Fit: G_K = c*(1-exp(-beta*K))")
        print(f"  c = {c:.4f} ({c*100:.2f}%)")
        print(f"  beta = {beta:.4f}")
        print(f"  R^2 = {R_squared:.4f}")

    # Create figure
    output_dir = Path(__file__).parent
    save_path = str(output_dir / f"{args.output}.png")
    create_figure(K_values, mean_gaps, std_gaps, mean_fidelities, std_fidelities,
                  c, beta, R_squared, args.inner_lr, save_path=save_path)

    # Save data
    data_path = str(output_dir / f"{args.output}_data.json")
    data_to_save = {
        'settings': {
            'inner_lr': args.inner_lr,
            'max_K': args.max_K,
            'n_tasks': args.n_tasks,
            'J_coupling': J_COUPLING,
            'gate_time': CZ_IDEAL_GATE_TIME,
        },
        'K_values': K_values.tolist(),
        'mean_gaps': mean_gaps.tolist(),
        'std_gaps': std_gaps.tolist(),
        'mean_fidelities': mean_fidelities.tolist(),
        'std_fidelities': std_fidelities.tolist(),
        'mean_initial_loss': float(mean_initial_loss),
        'fit': {
            'c': float(c) if c else None,
            'beta': float(beta) if beta else None,
            'R_squared': float(R_squared) if R_squared else None,
        }
    }
    with open(data_path, 'w') as f:
        json.dump(data_to_save, f, indent=2)


if __name__ == '__main__':
    main()
