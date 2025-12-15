"""
Number of Adaptation Steps Analysis
===================================

Generates plots showing final fidelity vs K for different task difficulties.
Shows diminishing returns and provides practical guidance on when to stop adapting.

FIXED: Now uses PSD-based task representation (alpha, A, omega_c)
matching the meta-training setup.

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


def create_task_with_difficulty(difficulty_level, config):
    """
    Create PSD noise parameters for different difficulty levels.

    Difficulty is determined by noise amplitude A and spectral exponent alpha.
    Higher A = more noise = harder task.
    """
    difficulty_params = {
        'easy': {'alpha': 1.0, 'A': 0.05, 'omega_c': 100},
        'medium': {'alpha': 1.0, 'A': 0.5, 'omega_c': 100},
        'hard': {'alpha': 1.0, 'A': 2.0, 'omega_c': 100},
        'very_hard': {'alpha': 1.5, 'A': 5.0, 'omega_c': 50},
    }

    params = difficulty_params[difficulty_level]

    return NoiseParameters(
        alpha=params['alpha'],
        A=params['A'],
        omega_c=params['omega_c'],
        model_type='one_over_f'
    )


def compute_fidelity_vs_K(meta_policy, task_params, env, config, device,
                          max_K=50, inner_lr=0.01):
    """Compute fidelity at each adaptation step K."""
    fidelities = []

    adapted_policy = deepcopy(meta_policy)
    inner_opt = optim.SGD(adapted_policy.parameters(), lr=inner_lr)

    # K=0: initial fidelity
    with torch.no_grad():
        loss = compute_loss(adapted_policy, task_params, env, device, config).item()
        fidelities.append(1.0 - loss)

    # K=1 to max_K
    for k in range(1, max_K + 1):
        inner_opt.zero_grad()
        loss = compute_loss(adapted_policy, task_params, env, device, config)
        loss.backward()
        inner_opt.step()

        with torch.no_grad():
            loss_val = compute_loss(adapted_policy, task_params, env, device, config).item()
            fidelities.append(1.0 - loss_val)

    return np.array(fidelities)


def main():
    parser = argparse.ArgumentParser(description='Adaptation steps analysis')
    parser.add_argument('--max_K', type=int, default=50, help='Maximum K')
    parser.add_argument('--inner_lr', type=float, default=0.01, help='Inner learning rate')
    parser.add_argument('--n_trials', type=int, default=10, help='Number of trials per difficulty')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

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

    difficulties = ['easy', 'medium', 'hard', 'very_hard']
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
    labels = ['Easy (A=0.05)', 'Medium (A=0.5)', 'Hard (A=2.0)', 'Very Hard (A=5.0)']

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
            task_params = create_task_with_difficulty(diff, config)
            fidelities = compute_fidelity_vs_K(
                meta_policy, task_params, env, config, device,
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
    save_path = output_dir / "adaptation_steps_analysis_v2.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(str(save_path).replace('.png', '.pdf'), bbox_inches='tight')
    print(f"\nFigure saved to: {save_path}")

    # Save data
    json_path = output_dir / "adaptation_steps_v2_data.json"
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
