"""
Adaptation Gap Analysis Figure
==============================

Generates a 2-panel figure:
(a) Adaptation gap G_K vs inner-loop steps K with exponential saturation fit
    G_K = A_inf(1 - e^(-beta*K)) with R^2 reported
(b) Asymptotic gap G_inf vs task diversity sigma^2_S showing linear scaling
    (confirming theoretical prediction from Theorem 1)

Adaptation gap is defined as: G_K = L(theta_0) - L(theta_K)
where L is the loss (infidelity) before and after K adaptation steps.

Usage:
    python generate_adaptation_gap_figure.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress
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


# =============================================================================
# Quantum System Setup
# =============================================================================

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


# =============================================================================
# Task Sampling with Controllable Diversity
# =============================================================================

def train_robust_policy(n_iterations=300, device='cpu', config=None):
    """
    Train a robust policy on a FIXED average noise level (no adaptation).

    This represents a controller optimized for "typical" conditions that cannot
    adapt to different noise environments - showing the benefit of meta-learning.
    """
    print("Training robust baseline policy on FIXED average noise...")

    # Use config if provided, otherwise defaults matching experiment_config.yaml
    if config is None:
        config = {
            'task_feature_dim': 3,
            'hidden_dim': 128,
            'n_hidden_layers': 2,
            'n_segments': 60,
            'n_controls': 2,
            'output_scale': 1.0
        }

    policy = PulsePolicy(
        task_feature_dim=config['task_feature_dim'],
        hidden_dim=config['hidden_dim'],
        n_hidden_layers=config['n_hidden_layers'],
        n_segments=config['n_segments'],
        n_controls=config['n_controls'],
        output_scale=config['output_scale']
    ).to(device)

    optimizer = optim.Adam(policy.parameters(), lr=1e-3)

    # Fixed average noise level (center of distribution) - matches task sampling
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
        total_loss = compute_loss(policy, fixed_task)
        total_loss.backward()
        optimizer.step()

        if iteration % 100 == 0:
            print(f"  Iter {iteration}: Loss = {total_loss.item():.4f}")

    print(f"Robust training complete (fixed gamma_deph={avg_gamma_deph:.3f}, gamma_relax={avg_gamma_relax:.3f}).")
    return policy


def sample_tasks_with_diversity(n_tasks, sigma_squared, device='cpu', seed=None):
    """
    Sample tasks with specified diversity (variance in noise parameters).

    Args:
        n_tasks: Number of tasks to sample
        sigma_squared: Task diversity (variance of gamma_deph)
        device: torch device
        seed: Random seed

    Returns:
        List of task dictionaries
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    tasks = []

    # Center of distribution
    mean_gamma_deph = 0.05
    mean_gamma_relax = 0.025

    # Standard deviation from variance
    std_gamma = np.sqrt(sigma_squared)

    rho0 = torch.tensor([[1, 0], [0, 0]], dtype=torch.complex64, device=device)
    # Target: X gate applied to |0> gives |1>
    target_rho = torch.tensor([[0, 0], [0, 1]], dtype=torch.complex64, device=device)

    for i in range(n_tasks):
        # Sample with specified variance, clamp to valid range
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
            'task_id': i
        })

    return tasks


# =============================================================================
# Adaptation Gap Computation
# =============================================================================

def compute_adaptation_gap_vs_K(meta_policy, robust_policy, tasks, max_K=20, inner_lr=0.01):
    """
    Compute adaptation gap G_K for each K.

    G_K = L(robust) - L(theta_K) = improvement of adapted meta-policy over robust baseline
    (positive gap = meta-adapted outperforms robust)

    Returns:
        K_values: array of K values [0, 1, ..., max_K]
        mean_gaps: mean adaptation gap at each K
        std_gaps: standard deviation at each K
        mean_robust_loss: mean loss of robust policy across tasks
    """
    n_tasks = len(tasks)
    all_gaps = np.zeros((n_tasks, max_K + 1))
    robust_losses = np.zeros(n_tasks)

    for task_idx, task in enumerate(tasks):
        # Get robust baseline loss (no adaptation)
        with torch.no_grad():
            L_robust = compute_loss(robust_policy, task).item()
            robust_losses[task_idx] = L_robust

        # Clone policy for adaptation
        adapted_policy = deepcopy(meta_policy)
        inner_opt = optim.SGD(adapted_policy.parameters(), lr=inner_lr)

        # G_0 = L(robust) - L(meta_init)
        with torch.no_grad():
            L_0 = compute_loss(adapted_policy, task).item()
            all_gaps[task_idx, 0] = L_robust - L_0

        # Compute gap at each K
        for k in range(1, max_K + 1):
            inner_opt.zero_grad()
            loss = compute_loss(adapted_policy, task)
            loss.backward()
            inner_opt.step()

            with torch.no_grad():
                L_K = compute_loss(adapted_policy, task).item()
                all_gaps[task_idx, k] = L_robust - L_K  # Gap vs robust baseline

    K_values = np.arange(max_K + 1)
    mean_gaps = np.mean(all_gaps, axis=0)
    std_gaps = np.std(all_gaps, axis=0)
    mean_robust_loss = np.mean(robust_losses)

    return K_values, mean_gaps, std_gaps, mean_robust_loss


def exponential_saturation(K, A_inf, beta, G0=0):
    """Exponential saturation model: G_K = G0 + A_inf(1 - e^(-beta*K))"""
    return G0 + A_inf * (1 - np.exp(-beta * K))


def fit_exponential_saturation(K_values, mean_gaps):
    """
    Fit exponential saturation model to adaptation gap data.

    Model: G_K = G0 + A_inf(1 - e^(-beta*K))
    where G0 is the initial gap at K=0.

    Returns:
        A_inf: Asymptotic gap improvement from adaptation
        beta: Rate constant
        G0: Initial gap (meta-policy vs robust baseline before adaptation)
        R_squared: Coefficient of determination
        fitted_curve: Fitted values
    """
    # Initial guesses
    G0_init = mean_gaps[0]
    A_inf_init = (mean_gaps[-1] - mean_gaps[0]) * 1.1 if mean_gaps[-1] > mean_gaps[0] else 0.05
    beta_init = 0.3

    try:
        popt, _ = curve_fit(
            exponential_saturation,
            K_values,
            mean_gaps,
            p0=[A_inf_init, beta_init, G0_init],
            bounds=([0, 0.01, -0.5], [1, 10, 0.5]),
            maxfev=5000
        )
        A_inf, beta, G0 = popt

        # Compute R^2
        fitted = exponential_saturation(K_values, A_inf, beta, G0)
        ss_res = np.sum((mean_gaps - fitted) ** 2)
        ss_tot = np.sum((mean_gaps - np.mean(mean_gaps)) ** 2)
        R_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return A_inf, beta, G0, R_squared, fitted
    except Exception as e:
        print(f"  Fitting failed: {e}")
        return None, None, None, None, None


# =============================================================================
# Panel (a): Adaptation Gap vs K
# =============================================================================

def generate_panel_a_data(meta_policy, robust_policy, n_tasks=10, max_K=30, inner_lr=0.01, device='cpu'):
    """Generate data for Panel (a): G_K vs K with exponential fit."""
    print("Generating Panel (a) data: Adaptation gap vs K...")

    # Use moderate diversity for this panel
    sigma_squared = 0.001
    tasks = sample_tasks_with_diversity(n_tasks, sigma_squared, device=device, seed=42)

    K_values, mean_gaps, std_gaps, mean_robust_loss = compute_adaptation_gap_vs_K(
        meta_policy, robust_policy, tasks, max_K=max_K, inner_lr=inner_lr
    )

    # Fit exponential saturation
    A_inf, beta, G0, R_squared, fitted_curve = fit_exponential_saturation(K_values, mean_gaps)

    print(f"  Mean robust baseline loss: {mean_robust_loss:.4f}")
    print(f"  Max gap at K={max_K}: {mean_gaps[-1]:.4f}")
    if A_inf is not None:
        print(f"  Fit: G0={G0:.4f}, A_inf={A_inf:.4f}, beta={beta:.4f}, R^2={R_squared:.4f}")

    return {
        'K_values': K_values,
        'mean_gaps': mean_gaps,
        'std_gaps': std_gaps,
        'A_inf': A_inf,
        'beta': beta,
        'G0': G0,
        'R_squared': R_squared,
        'fitted_curve': fitted_curve,
        'mean_robust_loss': mean_robust_loss
    }


# =============================================================================
# Panel (b): Asymptotic Gap vs Task Diversity
# =============================================================================

def generate_panel_b_data(meta_policy, robust_policy, n_tasks_per_diversity=20, K_adapt=5,
                          inner_lr=0.01, device='cpu'):
    """
    Generate data for Panel (b): G_K vs sigma^2_S.

    Compute adaptation gap at K=K_adapt for different task diversities.
    """
    print(f"Generating Panel (b) data: Adaptation gap at K={K_adapt} vs task diversity...")

    # 20 points linearly spaced between 1e-5 and 1e-2
    sigma_squared_values = list(np.linspace(1e-5, 1e-2, 20))

    G_inf_means = []
    G_inf_stds = []

    for sigma_sq in sigma_squared_values:
        print(f"  Processing sigma^2 = {sigma_sq:.4f}...", end=" ")

        # Sample tasks with this diversity
        tasks = sample_tasks_with_diversity(
            n_tasks_per_diversity, sigma_sq, device=device, seed=int(sigma_sq * 100000)
        )

        # Compute adaptation gaps
        _, mean_gaps, std_gaps, _ = compute_adaptation_gap_vs_K(
            meta_policy, robust_policy, tasks, max_K=K_adapt, inner_lr=inner_lr
        )

        # Use the gap at K=K_adapt
        G_inf_means.append(mean_gaps[K_adapt])
        G_inf_stds.append(std_gaps[K_adapt])
        print(f"G_{K_adapt}={mean_gaps[K_adapt]:.4f}")

    # Linear regression
    slope, intercept, r_value, p_value, std_err = linregress(
        sigma_squared_values, G_inf_means
    )

    print(f"  Linear fit: slope={slope:.4f}, intercept={intercept:.4f}, R^2={r_value**2:.4f}")

    return {
        'sigma_squared_values': np.array(sigma_squared_values),
        'G_inf_means': np.array(G_inf_means),
        'G_inf_stds': np.array(G_inf_stds),
        'slope': slope,
        'intercept': intercept,
        'R_squared': r_value ** 2
    }


# =============================================================================
# Plotting
# =============================================================================

def create_figure(panel_a_data, panel_b_data, save_path=None):
    """Create the 2-panel figure."""

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # --- Panel (a): Adaptation Gap vs K ---
    ax = axes[0]

    K = panel_a_data['K_values']
    mean_gaps = panel_a_data['mean_gaps']
    std_gaps = panel_a_data['std_gaps']
    A_inf = panel_a_data['A_inf']
    beta = panel_a_data['beta']
    G0 = panel_a_data['G0']
    R_sq = panel_a_data['R_squared']

    # Plot data (no error bars)
    ax.plot(K, mean_gaps, 'o', color='#3498db', markersize=5, label='Data')

    # Plot fitted curve if fit succeeded
    if A_inf is not None and beta is not None and G0 is not None:
        K_fine = np.linspace(0, K[-1], 100)
        fitted_fine = exponential_saturation(K_fine, A_inf, beta, G0)
        ax.plot(K_fine, fitted_fine, '-', color='#e74c3c', linewidth=2,
                label=f'Fit: $G_K = G_0 + A_\\infty(1-e^{{-\\beta K}})$')

    ax.set_xlabel('Inner-loop Steps $K$')
    ax.set_ylabel('Adaptation Gap $G_K$')
    ax.set_title('(a) Adaptation Gap vs Steps')
    ax.set_xlim(-0.5, K[-1] + 0.5)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=9)

    # --- Panel (b): Asymptotic Gap vs Task Diversity (Linear Scale) ---
    ax = axes[1]

    sigma_sq = panel_b_data['sigma_squared_values']
    G_inf = panel_b_data['G_inf_means']
    G_inf_std = panel_b_data['G_inf_stds']
    slope = panel_b_data['slope']
    intercept = panel_b_data['intercept']
    R_sq_b = panel_b_data['R_squared']

    # Plot data - linear scale
    ax.plot(sigma_sq, G_inf, 's', color='#2ecc71', markersize=8, label='Data')

    # Plot linear fit
    sigma_sq_fine = np.linspace(sigma_sq.min(), sigma_sq.max(), 100)
    fitted_line = slope * sigma_sq_fine + intercept
    ax.plot(sigma_sq_fine, fitted_line, '-', color='#e74c3c', linewidth=2,
            label=f'Linear fit')

    # Add R^2 annotation
    ax.text(0.95, 0.05, f'$R^2 = {R_sq_b:.2f}$',
            transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
            horizontalalignment='right')

    ax.set_xlabel('Task Diversity $\\sigma^2_S$')
    ax.set_ylabel('Adaptation Gap $G_5$')
    ax.set_title('(b) Gap at $K=5$ vs Task Diversity')
    # Linear scale (no log)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        pdf_path = save_path.replace('.png', '.pdf')
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
        print(f"Figure saved to: {pdf_path}")

    plt.close()
    return fig


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate adaptation gap figure')
    parser.add_argument('--n_tasks', type=int, default=100, help='Number of tasks (batch size)')
    parser.add_argument('--max_K', type=int, default=30, help='Maximum K for panel (a)')
    parser.add_argument('--inner_lr', type=float, default=0.01, help='Inner loop learning rate')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Configuration matching the checkpoint
    config = {
        'task_feature_dim': 3,
        'hidden_dim': 128,
        'n_hidden_layers': 2,
        'n_segments': 60,
        'n_controls': 2,
        'output_scale': 1.0,
        'inner_lr': args.inner_lr,
    }

    # Load checkpoint (same as reference implementation)
    checkpoint_path = project_root / "experiments" / "checkpoints" / "maml_best_pauli_x_best.pt"

    print(f"Looking for checkpoint: {checkpoint_path}")

    if checkpoint_path.exists():
        print("Loading pre-trained meta-policy...")
        meta_policy = load_policy_from_checkpoint(
            str(checkpoint_path),
            config,
            device=torch.device(device),
            eval_mode=False,
            verbose=True
        )
    else:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    inner_lr = config['inner_lr']
    print(f"\nUsing inner_lr={inner_lr}")

    # Train robust baseline policy on fixed average noise
    print("\n" + "=" * 60)
    robust_policy = train_robust_policy(n_iterations=300, device=device, config=config)

    # Generate data for each panel
    print("\n" + "=" * 60)
    panel_a_data = generate_panel_a_data(
        meta_policy, robust_policy, n_tasks=args.n_tasks, max_K=args.max_K, inner_lr=inner_lr, device=device
    )

    print("\n" + "=" * 60)
    panel_b_data = generate_panel_b_data(
        meta_policy, robust_policy, n_tasks_per_diversity=args.n_tasks, K_adapt=5,
        inner_lr=inner_lr, device=device
    )

    # Create figure
    print("\n" + "=" * 60)
    print("Creating figure...")

    output_dir = Path(__file__).parent
    output_dir.mkdir(exist_ok=True)
    save_path = str(output_dir / f"adaptation_gap_figure_n{args.n_tasks}.png")

    create_figure(panel_a_data, panel_b_data, save_path=save_path)

    # Save data as JSON
    results = {
        'panel_a': {
            'K_values': panel_a_data['K_values'].tolist(),
            'mean_gaps': panel_a_data['mean_gaps'].tolist(),
            'std_gaps': panel_a_data['std_gaps'].tolist(),
            'G0': float(panel_a_data['G0']) if panel_a_data['G0'] else None,
            'A_inf': float(panel_a_data['A_inf']) if panel_a_data['A_inf'] else None,
            'beta': float(panel_a_data['beta']) if panel_a_data['beta'] else None,
            'R_squared': float(panel_a_data['R_squared']) if panel_a_data['R_squared'] else None,
            'mean_robust_loss': float(panel_a_data['mean_robust_loss']),
        },
        'panel_b': {
            'sigma_squared_values': panel_b_data['sigma_squared_values'].tolist(),
            'G_inf_means': panel_b_data['G_inf_means'].tolist(),
            'G_inf_stds': panel_b_data['G_inf_stds'].tolist(),
            'slope': float(panel_b_data['slope']),
            'intercept': float(panel_b_data['intercept']),
            'R_squared': float(panel_b_data['R_squared']),
        }
    }
    json_path = output_dir / f"adaptation_gap_data_n{args.n_tasks}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Data saved to: {json_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)

    print("\nPanel (a) - Exponential Saturation Fit:")
    if panel_a_data['A_inf'] is not None:
        print(f"  G0 (initial gap) = {panel_a_data['G0']:.4f}")
        print(f"  A_inf (adaptation improvement) = {panel_a_data['A_inf']:.4f}")
        print(f"  beta (rate constant) = {panel_a_data['beta']:.4f}")
        print(f"  R^2 = {panel_a_data['R_squared']:.4f}")
    else:
        print("  Fit failed")

    print("\nPanel (b) - Linear Scaling with Task Diversity:")
    print(f"  Slope = {panel_b_data['slope']:.4f}")
    print(f"  Intercept = {panel_b_data['intercept']:.4f}")
    print(f"  R^2 = {panel_b_data['R_squared']:.4f}")

    print("\n" + "=" * 60)
    print("Figure generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
