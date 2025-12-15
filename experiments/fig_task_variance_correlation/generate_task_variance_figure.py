"""
Task Variance vs Loss Variance Correlation

Shows that with a FIXED pulse sequence, the variance in the loss function
is directly correlated with the variance in the task distribution.

This demonstrates that task diversity drives the need for adaptation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import stats
import json
import argparse

from metaqctrl.quantum.lindblad_torch import DifferentiableLindbladSimulator


def basis_state(n: int, dim: int):
    """Create a basis state |n> in dimension dim."""
    state = torch.zeros(dim, dtype=torch.complex64)
    state[n] = 1.0
    return state


def create_gamma_simulator(gamma_deph, gamma_relax, device='cpu'):
    """Create Lindblad simulator with gamma noise parameters."""
    # Pauli matrices for control
    sigma_x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device=device)
    sigma_y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64, device=device)

    # Lindblad operators for dephasing and relaxation
    sigma_z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=device)
    sigma_m = torch.tensor([[0, 0], [1, 0]], dtype=torch.complex64, device=device)  # Lowering operator

    L_deph = np.sqrt(gamma_deph) * sigma_z
    L_relax = np.sqrt(gamma_relax) * sigma_m

    L_operators = [L_relax, L_deph]
    H_controls = [sigma_x, sigma_y]
    H0 = torch.zeros((2, 2), dtype=torch.complex64, device=device)

    return DifferentiableLindbladSimulator(
        H0=H0,
        H_controls=H_controls,
        L_operators=L_operators,
        dt=0.01,
        device=device
    )


def compute_fidelity(rho, target_state, device='cpu'):
    """Compute state fidelity."""
    target_dm = torch.outer(target_state, target_state.conj())
    fidelity = torch.real(torch.trace(rho @ target_dm))
    return fidelity.item()


def evaluate_fixed_pulse_on_tasks(pulse_sequence, tasks, T=1.0, device='cpu'):
    """
    Evaluate a FIXED pulse sequence on multiple tasks.
    Returns the loss (1 - fidelity) for each task.
    """
    # Target: X gate on |0> -> |1>
    initial_state = basis_state(0, 2).to(device)
    target_state = basis_state(1, 2).to(device)
    rho0 = torch.outer(initial_state, initial_state.conj())

    losses = []
    for gamma_deph, gamma_relax in tasks:
        simulator = create_gamma_simulator(gamma_deph, gamma_relax, device)
        rho_final = simulator.forward(rho0, pulse_sequence, T)
        fidelity = compute_fidelity(rho_final, target_state, device)
        losses.append(1.0 - fidelity)

    return np.array(losses)


def sample_tasks_with_variance(n_tasks, center_deph, center_relax, spread, rng=None):
    """
    Sample tasks around a center point with specified spread (variance proxy).
    spread: multiplier for range (e.g., 0.1 means ±10% of center)
    """
    if rng is None:
        rng = np.random.default_rng()

    gamma_deph = rng.uniform(
        center_deph * (1 - spread),
        center_deph * (1 + spread),
        n_tasks
    )
    gamma_relax = rng.uniform(
        center_relax * (1 - spread),
        center_relax * (1 + spread),
        n_tasks
    )

    # Clip to valid ranges
    gamma_deph = np.clip(gamma_deph, 0.001, 2.0)
    gamma_relax = np.clip(gamma_relax, 0.001, 1.0)

    return list(zip(gamma_deph, gamma_relax))


def compute_task_variance(tasks):
    """Compute variance of task parameters."""
    gamma_deph = np.array([t[0] for t in tasks])
    gamma_relax = np.array([t[1] for t in tasks])

    # Combined variance (sum of individual variances, normalized)
    var_deph = np.var(gamma_deph)
    var_relax = np.var(gamma_relax)

    return var_deph + var_relax


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_tasks', type=int, default=50, help='Tasks per spread level')
    parser.add_argument('--n_spread_levels', type=int, default=15, help='Number of spread levels')
    parser.add_argument('--n_segments', type=int, default=20, help='Pulse segments')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = 'cpu'
    rng = np.random.default_rng(args.seed)

    # Center point for task distribution
    center_deph = 0.10
    center_relax = 0.05

    # Generate a FIXED pulse sequence (e.g., approximate X gate pulse)
    # Using a simple constant pulse that gives reasonable fidelity
    T = 1.0
    n_segments = args.n_segments

    # Fixed pulse: roughly π rotation around x-axis
    pulse_sequence = torch.zeros((n_segments, 2), dtype=torch.float32, device=device)
    pulse_sequence[:, 0] = np.pi  # u_x = π for X gate
    pulse_sequence[:, 1] = 0.0    # u_y = 0

    print(f"Using FIXED pulse sequence: constant u_x={np.pi:.3f}, u_y=0")
    print(f"Evaluating on {args.n_spread_levels} spread levels with {args.n_tasks} tasks each\n")

    # Spread levels from very narrow (0.01) to wide (0.8)
    spread_levels = np.linspace(0.01, 0.8, args.n_spread_levels)

    results = {
        'spread_levels': spread_levels.tolist(),
        'task_variances': [],
        'loss_means': [],
        'loss_variances': [],
        'loss_stds': []
    }

    print("Spread | Task Var | Loss Mean | Loss Var | Loss Std")
    print("-" * 55)

    for spread in spread_levels:
        # Sample tasks with this spread
        tasks = sample_tasks_with_variance(
            args.n_tasks, center_deph, center_relax, spread, rng
        )

        # Compute task variance
        task_var = compute_task_variance(tasks)

        # Evaluate fixed pulse on all tasks
        losses = evaluate_fixed_pulse_on_tasks(pulse_sequence, tasks, T, device)

        loss_mean = np.mean(losses)
        loss_var = np.var(losses)
        loss_std = np.std(losses)

        results['task_variances'].append(task_var)
        results['loss_means'].append(loss_mean)
        results['loss_variances'].append(loss_var)
        results['loss_stds'].append(loss_std)

        print(f"{spread:.2f}   | {task_var:.6f} | {loss_mean:.4f}    | {loss_var:.6f} | {loss_std:.4f}")

    # Compute correlations
    task_vars = np.array(results['task_variances'])
    loss_vars = np.array(results['loss_variances'])
    loss_stds = np.array(results['loss_stds'])
    loss_means = np.array(results['loss_means'])

    # Linear regression for task variance vs loss variance
    slope_var, intercept_var, r_var, p_var, se_var = stats.linregress(task_vars, loss_vars)

    # Linear regression for task variance vs loss std
    slope_std, intercept_std, r_std, p_std, se_std = stats.linregress(task_vars, loss_stds)

    print(f"\n{'='*55}")
    print("Correlation Analysis:")
    print(f"  Task Var vs Loss Var: R²={r_var**2:.4f}, slope={slope_var:.4f}")
    print(f"  Task Var vs Loss Std: R²={r_std**2:.4f}, slope={slope_std:.4f}")

    # Create figure - single panel
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    # Task Variance vs Loss Variance
    ax.scatter(task_vars, loss_vars, c='#3498db', alpha=0.7, s=60, edgecolors='white', linewidth=0.5)

    # Fit line
    x_fit = np.linspace(task_vars.min(), task_vars.max(), 100)
    y_fit = slope_var * x_fit + intercept_var
    ax.plot(x_fit, y_fit, 'r-', linewidth=2, label=f'Linear fit (R²={r_var**2:.3f})')

    ax.set_xlabel(r'$\sigma_\tau^2$', fontsize=14)
    ax.set_ylabel(r'$\sigma_L^2$', fontsize=14)
    ax.set_title('Task Variance vs Loss Variance', fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # Add R² annotation
    ax.text(0.95, 0.05, f'R² = {r_var**2:.4f}', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save figure
    output_dir = Path(__file__).parent
    save_path = str(output_dir / "task_variance_correlation.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"\nFigure saved to: {save_path}")

    # Save data
    results['correlation'] = {
        'task_var_vs_loss_var': {'R2': r_var**2, 'slope': slope_var, 'intercept': intercept_var},
        'task_var_vs_loss_std': {'R2': r_std**2, 'slope': slope_std, 'intercept': intercept_std}
    }

    data_path = str(output_dir / "task_variance_correlation_data.json")
    with open(data_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Data saved to: {data_path}")


if __name__ == "__main__":
    main()
