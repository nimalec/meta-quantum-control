"""
Train a policy on the average task (mean noise parameters).

This creates a baseline policy trained on a single task with mean parameters
from the task distribution. This baseline represents a non-adaptive approach
trained for the "average" environment.

Usage:
    python train_average_task.py --config ../../configs/experiment_config.yaml
"""

import torch
import numpy as np
import yaml
from pathlib import Path
import argparse
from datetime import datetime

from metaqctrl.quantum.noise_adapter import TaskDistribution, NoiseParameters
from metaqctrl.quantum.gates import TargetGates
from metaqctrl.meta_rl.policy import PulsePolicy
from metaqctrl.theory.quantum_environment import create_quantum_environment


def compute_mean_task(task_dist: TaskDistribution, n_samples: int = 10000) -> NoiseParameters:
    """
    Compute mean task parameters from the task distribution.

    Args:
        task_dist: Task distribution
        n_samples: Number of samples to use for computing mean

    Returns:
        NoiseParameters with mean values
    """
    print(f"\nComputing mean task from {n_samples} samples...")
    rng = np.random.default_rng(42)
    tasks = task_dist.sample(n_samples, rng)

    # Compute means
    alpha_mean = np.mean([t.alpha for t in tasks])
    A_mean = np.mean([t.A for t in tasks])
    omega_c_mean = np.mean([t.omega_c for t in tasks])

    # Get model type (assume single model for now)
    model_type = tasks[0].model_type

    print(f"  Mean α: {alpha_mean:.4f}")
    print(f"  Mean A: {A_mean:.4f}")
    print(f"  Mean ω_c: {omega_c_mean:.4f}")
    print(f"  Model: {model_type}")

    return NoiseParameters(
        alpha=alpha_mean,
        A=A_mean,
        omega_c=omega_c_mean,
        model_type=model_type
    )


def train_on_single_task(
    policy: torch.nn.Module,
    task_params: NoiseParameters,
    env,
    config: dict,
    device: torch.device,
    n_iterations: int = 5000,
    lr: float = 0.001,
    log_interval: int = 100
):
    """
    Train policy on a single task using gradient descent.

    Args:
        policy: Policy network to train
        task_params: Task parameters
        env: Quantum environment
        config: Configuration dictionary
        device: Device to train on
        n_iterations: Number of training iterations
        lr: Learning rate
        log_interval: How often to print progress
    """
    print("\nTraining policy on average task...")
    print(f"  Iterations: {n_iterations}")
    print(f"  Learning rate: {lr}")

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    # GPU-optimized settings
    dt = config.get('dt_training', 0.01)
    use_rk4 = config.get('use_rk4_training', True)

    best_loss = float('inf')
    best_fidelity = 0.0

    for iteration in range(n_iterations):
        optimizer.zero_grad()

        # Compute loss
        loss = env.compute_loss_differentiable(
            policy,
            task_params,
            device,
            use_rk4=use_rk4,
            dt=dt
        )

        # Backward pass
        loss.backward()
        optimizer.step()

        # Logging
        if (iteration + 1) % log_interval == 0:
            with torch.no_grad():
                task_features = torch.tensor(
                    task_params.to_array(),
                    dtype=torch.float32,
                    device=device
                )
                controls = policy(task_features).cpu().numpy()
                fidelity = env.compute_fidelity(controls, task_params)

            print(f"  Iter {iteration+1}/{n_iterations}: Loss = {loss.item():.6f}, Fidelity = {fidelity:.6f}")

            if fidelity > best_fidelity:
                best_fidelity = fidelity
                best_loss = loss.item()

    print(f"\n  Training complete!")
    print(f"  Best Loss: {best_loss:.6f}")
    print(f"  Best Fidelity: {best_fidelity:.6f}")

    return best_loss, best_fidelity


def main(config_path: str):
    """Main training loop for average task policy."""

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("=" * 70)
    print("Training Policy on Average Task")
    print("=" * 70)
    print(f"Config: {config_path}\n")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Target gate
    target_gate_name = config.get('target_gate', 'pauli_x')
    if target_gate_name == 'hadamard':
        U_target = TargetGates.hadamard()
    elif target_gate_name == 'pauli_x':
        U_target = TargetGates.pauli_x()
    else:
        raise ValueError(f"Unknown target gate: {target_gate_name}")

    ket_0 = np.array([1, 0], dtype=complex)
    target_state = np.outer(U_target @ ket_0, (U_target @ ket_0).conj())
    print(f"Target gate: {target_gate_name}")

    # Create quantum environment
    print("\nSetting up quantum environment...")
    from metaqctrl.theory.quantum_environment import create_quantum_environment
    env = create_quantum_environment(config, target_state)

    # Create task distribution
    print("\nCreating task distribution...")
    task_dist = TaskDistribution(
        dist_type=config.get('task_dist_type', 'uniform'),
        ranges={
            'alpha': tuple(config.get('alpha_range')),
            'A': tuple(config.get('A_range')),
            'omega_c': tuple(config.get('omega_c_range'))
        },
        model_types=config.get('model_types'),
        model_probs=config.get('model_probs')
    )

    # Compute mean task
    mean_task = compute_mean_task(task_dist, n_samples=10000)

    # Create policy
    print("\nCreating policy network...")
    policy = PulsePolicy(
        task_feature_dim=config.get('task_feature_dim', 3),
        hidden_dim=config.get('hidden_dim', 128),
        n_hidden_layers=config.get('n_hidden_layers', 2),
        n_segments=config.get('n_segments', 100),
        n_controls=config.get('n_controls', 2),
        output_scale=config.get('output_scale', 1.0),
        activation=config.get('activation', 'tanh')
    )
    policy = policy.to(device)
    print(f"  Parameters: {policy.count_parameters():,}")

    # Train on mean task
    best_loss, best_fidelity = train_on_single_task(
        policy=policy,
        task_params=mean_task,
        env=env,
        config=config,
        device=device,
        n_iterations=5000,
        lr=0.001,
        log_interval=500
    )

    # Save policy and mean task parameters
    save_dir = Path(config.get('save_dir', 'checkpoints')) / 'baseline_comparison'
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / 'average_task_policy.pt'

    checkpoint = {
        'policy_state_dict': policy.state_dict(),
        'mean_task': {
            'alpha': mean_task.alpha,
            'A': mean_task.A,
            'omega_c': mean_task.omega_c,
            'model_type': mean_task.model_type
        },
        'config': config,
        'best_loss': best_loss,
        'best_fidelity': best_fidelity,
        'timestamp': datetime.now().isoformat()
    }

    torch.save(checkpoint, save_path)

    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)
    print(f"\nPolicy saved to: {save_path}")
    print(f"Mean task parameters saved in checkpoint")
    print(f"\nFinal metrics:")
    print(f"  Loss: {best_loss:.6f}")
    print(f"  Fidelity: {best_fidelity:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train policy on average task for baseline comparison'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='../../configs/experiment_config.yaml',
        help='Path to config file'
    )

    args = parser.parse_args()
    main(args.config)
