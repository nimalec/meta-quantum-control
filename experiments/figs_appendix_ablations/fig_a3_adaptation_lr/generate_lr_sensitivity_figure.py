"""
Inner Learning Rate Sensitivity Ablation
=========================================

Generates adaptation gap curves G_K for different inner learning rates:
alpha_inner in {0.001, 0.005, 0.01, 0.05, 0.1}

Shows that the scaling law holds across learning rates, with beta parameter
changing predictably with learning rate.

FIXED: Now uses PSD-based task representation (alpha, A, omega_c)
matching the meta-training setup.

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
import yaml

from metaqctrl.meta_rl.policy import PulsePolicy
from metaqctrl.quantum.noise_adapter import TaskDistribution, NoiseParameters
from metaqctrl.theory.quantum_environment import create_quantum_environment
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


def load_config():
    """Load experiment configuration."""
    config_path = project_root / "configs" / "experiment_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_task_distribution(config: dict):
    """Create task distribution matching meta-training."""
    return TaskDistribution(
        dist_type=config.get('task_dist_type', 'uniform'),
        ranges={
            'alpha': tuple(config.get('alpha_range', [0.1, 2.0])),
            'A': tuple(config.get('A_range', [0.01, 10.0])),
            'omega_c': tuple(config.get('omega_c_range', [1, 300]))
        },
        model_types=config.get('model_types', ['one_over_f']),
        model_probs=config.get('model_probs', [1.0])
    )


def get_target_state(config: dict):
    """Get target state from config."""
    from metaqctrl.quantum.gates import TargetGates

    target_gate_name = config.get('target_gate', 'pauli_x')
    if target_gate_name == 'hadamard':
        U_target = TargetGates.hadamard()
    elif target_gate_name == 'pauli_x':
        U_target = TargetGates.pauli_x()
    elif target_gate_name == 'pauli_y':
        U_target = TargetGates.pauli_y()
    else:
        raise ValueError(f"Unknown target gate: {target_gate_name}")

    ket_0 = np.array([1, 0], dtype=complex)
    target_state = np.outer(U_target @ ket_0, (U_target @ ket_0).conj())
    return target_state


def compute_loss(policy, task_params, env, device, config):
    """
    Compute loss using the quantum environment (matching meta-training).
    Loss = 1 - Fidelity
    """
    return env.compute_loss_differentiable(
        policy,
        task_params,
        device,
        use_rk4=config.get('use_rk4_training', True),
        dt=config.get('dt_training', 0.01)
    )


def compute_adaptation_gap_vs_K(meta_policy, task_params_list, env, config,
                                 max_K=20, inner_lr=0.01, device='cpu'):
    """Compute adaptation gap G_K for each K."""
    n_tasks = len(task_params_list)
    all_gaps = np.zeros((n_tasks, max_K + 1))

    for task_idx, task_params in enumerate(task_params_list):
        # Clone policy for adaptation
        adapted_policy = deepcopy(meta_policy)
        inner_opt = optim.SGD(adapted_policy.parameters(), lr=inner_lr)

        # Get initial loss
        with torch.no_grad():
            L_0 = compute_loss(adapted_policy, task_params, env, device, config).item()
            all_gaps[task_idx, 0] = 0  # Gap at K=0 is 0 (relative to self)

        # Compute gap at each K (relative to L_0)
        for k in range(1, max_K + 1):
            inner_opt.zero_grad()
            loss = compute_loss(adapted_policy, task_params, env, device, config)
            loss.backward()
            inner_opt.step()

            with torch.no_grad():
                L_K = compute_loss(adapted_policy, task_params, env, device, config).item()
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
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load config
    config = load_config()
    policy_config = {
        'task_feature_dim': 3,  # alpha, A, omega_c
        'hidden_dim': 128,
        'n_hidden_layers': 2,
        'n_segments': 60,
        'n_controls': 2,
        'output_scale': 1.0,
        'activation': 'tanh',
    }

    # Create target state and quantum environment
    print("\nSetting up quantum environment...")
    target_state = get_target_state(config)
    env = create_quantum_environment(config, target_state)
    print(f"  Environment created: {env.get_cache_stats()}")

    # Create task distribution
    task_dist = create_task_distribution(config)
    print(f"  Task distribution: alpha={config['alpha_range']}, A={config['A_range']}, omega_c={config['omega_c_range']}")

    # Learning rates to test
    learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1]
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12']

    # Load checkpoint
    checkpoint_path = project_root / "experiments" / "checkpoints" / "maml_best_pauli_x_best.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    print("\nLoading pre-trained meta-policy...")
    meta_policy = load_policy_from_checkpoint(
        str(checkpoint_path),
        policy_config,
        device=device,
        eval_mode=False,
        verbose=True
    )

    # Sample tasks (same for all LRs for fair comparison)
    print(f"\nSampling {args.n_tasks} tasks from PSD distribution...")
    task_params_list = task_dist.sample(args.n_tasks, rng)

    # Compute adaptation curves for each learning rate
    results = {}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    print("\n" + "=" * 60)
    print("Computing adaptation curves for different learning rates...")
    print("=" * 60)

    for lr, color in zip(learning_rates, colors):
        print(f"\nProcessing alpha_inner = {lr}...")

        K_values, mean_gaps, std_gaps = compute_adaptation_gap_vs_K(
            meta_policy, task_params_list, env, config,
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

        # Plot on left panel
        axes[0].plot(K_values, mean_gaps, 'o-', color=color, markersize=4,
                     label=f'alpha={lr}', alpha=0.8)

    # Panel (a): Adaptation curves
    axes[0].set_xlabel('Inner-loop Steps $K$')
    axes[0].set_ylabel('Adaptation Gap $G_K$')
    axes[0].set_title('(a) Adaptation Gap vs K for Different Learning Rates')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(-0.5, args.max_K + 0.5)
    axes[0].set_ylim(0, None)

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

    # Save figure
    output_dir = Path(__file__).parent
    save_path = output_dir / "inner_lr_sensitivity_v2.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(str(save_path).replace('.png', '.pdf'), bbox_inches='tight')
    print(f"\nFigure saved to: {save_path}")

    # Save data
    json_path = output_dir / "inner_lr_sensitivity_v2_data.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Data saved to: {json_path}")

    plt.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
