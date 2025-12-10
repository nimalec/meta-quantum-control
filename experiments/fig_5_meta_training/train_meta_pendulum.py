"""
Train MAML on classical pendulum control task.

This provides a fast sanity check for the MAML implementation before
running expensive quantum simulations.

Usage:
    python train_meta_pendulum.py [--n_iterations 500] [--inner_steps 5]

Expected behavior:
    - Meta-loss should decrease over training
    - Validation fidelity should improve
    - Adaptation gain should be positive (post-adapt better than pre-adapt)

If this doesn't work, debug MAML before running quantum experiments!
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from metaqctrl.classical.pendulum import (
    PendulumEnvironment,
    create_pendulum_loss_fn
)
from metaqctrl.classical.task_distribution import PendulumTaskDistribution
from metaqctrl.meta_rl.maml import MAML, MAMLTrainer
from metaqctrl.meta_rl.policy import PulsePolicy


def create_pendulum_data_generator(env: PendulumEnvironment, n_support: int, n_query: int):
    """
    Create data generator function for MAMLTrainer.

    Args:
        env: PendulumEnvironment
        n_support: Support set size
        n_query: Query set size

    Returns:
        data_generator: Function(task_params, n_trajectories, split) -> data_dict
    """
    def data_generator(task_params, n_trajectories, split):
        """
        Generate support/query data for a task.

        Args:
            task_params: PendulumTask instance
            n_trajectories: Number of trajectories (ignored for pendulum)
            split: 'support' or 'query' (ignored for pendulum)

        Returns:
            data_dict: {'task_features': tensor}
        """
        task_features = task_params.to_array()

        # Return batch of task features
        return {
            'task_features': torch.tensor(
                np.tile(task_features, (n_trajectories, 1)),
                dtype=torch.float32
            )
        }

    return data_generator


def main(args):
    """Main training loop."""

    print("=" * 70)
    print("MAML Training: Classical Pendulum Control")
    print("=" * 70)
    print()

    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print()

    # ========================================
    # 1. Create environment
    # ========================================
    print("Creating pendulum environment...")
    env = PendulumEnvironment(
        n_segments=args.n_segments,
        horizon=args.horizon
    )
    print(f"  Segments: {env.n_segments}")
    print(f"  Horizon: {env.T:.2f} s")
    print(f"  dt: {env.dt:.3f} s/segment")
    print()

    # ========================================
    # 2. Create task distribution
    # ========================================
    print("Creating task distribution...")
    task_dist = PendulumTaskDistribution(
        dist_type='uniform',
        ranges={
            'mass': (args.mass_min, args.mass_max),
            'length': (args.length_min, args.length_max),
            'friction': (args.friction_min, args.friction_max)
        }
    )
    variance = task_dist.compute_variance()
    print(f"  Mass range: [{args.mass_min}, {args.mass_max}] kg")
    print(f"  Length range: [{args.length_min}, {args.length_max}] m")
    print(f"  Friction range: [{args.friction_min}, {args.friction_max}]")
    print(f"  Task variance σ²: {variance:.4f}")
    print()

    # ========================================
    # 3. Create policy
    # ========================================
    print("Creating policy network...")
    policy = PulsePolicy(
        task_feature_dim=3,  # (mass, length, friction)
        hidden_dim=args.hidden_dim,
        n_hidden_layers=args.n_hidden_layers,
        n_segments=args.n_segments,
        n_controls=1  # Single torque control
    )
    n_params = policy.count_parameters()
    print(f"  Architecture: {args.n_hidden_layers} hidden layers, {args.hidden_dim} units")
    print(f"  Total parameters: {n_params:,}")
    print()

    # ========================================
    # 4. Create MAML
    # ========================================
    print("Initializing MAML...")
    maml = MAML(
        policy=policy,
        inner_lr=args.inner_lr,
        inner_steps=args.inner_steps,
        meta_lr=args.meta_lr,
        first_order=args.first_order,
        device=device
    )
    print(f"  Inner LR: {args.inner_lr}")
    print(f"  Inner steps: {args.inner_steps}")
    print(f"  Meta LR: {args.meta_lr}")
    print(f"  Mode: {'First-order (FOMAML)' if args.first_order else 'Second-order'}")
    print()

    # ========================================
    # 5. Create loss function
    # ========================================
    loss_fn = create_pendulum_loss_fn(env)

    # ========================================
    # 6. Create trainer
    # ========================================
    print("Creating trainer...")

    def task_sampler(n_tasks, split='train'):
        """Sample tasks from distribution."""
        rng = np.random.default_rng()
        return task_dist.sample(n_tasks, rng)

    data_generator = create_pendulum_data_generator(env, args.n_support, args.n_query)

    trainer = MAMLTrainer(
        maml=maml,
        task_sampler=task_sampler,
        data_generator=data_generator,
        loss_fn=loss_fn,
        n_support=args.n_support,
        n_query=args.n_query,
        log_interval=args.log_interval,
        val_interval=args.val_interval
    )
    print(f"  Support set size: {args.n_support}")
    print(f"  Query set size: {args.n_query}")
    print(f"  Log interval: {args.log_interval}")
    print(f"  Validation interval: {args.val_interval}")
    print()

    # ========================================
    # 7. Train
    # ========================================
    print("Starting training...")
    print("=" * 70)
    print()

    # Create checkpoint directory
    checkpoint_dir = Path(__file__).parent.parent.parent / "checkpoints" / "pendulum"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_path = checkpoint_dir / f"maml_pendulum_iter{args.n_iterations}.pt"

    trainer.train(
        n_iterations=args.n_iterations,
        tasks_per_batch=args.tasks_per_batch,
        val_tasks=args.val_tasks,
        save_path=str(save_path)
    )

    print()
    print("=" * 70)
    print("Training complete!")
    print(f"Checkpoint saved to: {save_path}")
    print()

    # ========================================
    # 8. Final evaluation
    # ========================================
    print("Final Evaluation")
    print("-" * 70)

    # Sample some test tasks
    test_tasks = task_sampler(5, split='test')

    print(f"\nTesting on {len(test_tasks)} held-out tasks:")
    print(f"{'Task':<6} {'Mass':<8} {'Length':<8} {'Friction':<10} {'Pre-Loss':<10} {'Post-Loss':<10} {'Gain':<10}")
    print("-" * 70)

    for i, task in enumerate(test_tasks):
        # Generate test data
        test_data = data_generator(task, n_trajectories=1, split='test')

        # Pre-adaptation loss
        with torch.no_grad():
            pre_loss = loss_fn(maml.policy, test_data).item()

        # Adapt
        task_batch = {'support': test_data, 'query': test_data}
        adapted_policy, _ = maml.inner_loop(task_batch, loss_fn, num_steps=args.inner_steps)

        # Post-adaptation loss
        with torch.no_grad():
            post_loss = loss_fn(adapted_policy, test_data).item()

        gain = pre_loss - post_loss

        print(f"{i:<6} {task.mass:<8.2f} {task.length:<8.2f} {task.friction:<10.2f} "
              f"{pre_loss:<10.4f} {post_loss:<10.4f} {gain:<10.4f}")

    print()
    print("Notes:")
    print("  - Loss is final state error (lower is better)")
    print("  - Gain should be positive (adaptation helps)")
    print("  - If gain is negative, MAML may not be working correctly")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train MAML on pendulum control')

    # Environment
    parser.add_argument('--n_segments', type=int, default=20, help='Control segments')
    parser.add_argument('--horizon', type=float, default=2.0, help='Time horizon [s]')

    # Task distribution
    parser.add_argument('--mass_min', type=float, default=0.5, help='Min mass [kg]')
    parser.add_argument('--mass_max', type=float, default=2.0, help='Max mass [kg]')
    parser.add_argument('--length_min', type=float, default=0.5, help='Min length [m]')
    parser.add_argument('--length_max', type=float, default=1.5, help='Max length [m]')
    parser.add_argument('--friction_min', type=float, default=0.0, help='Min friction')
    parser.add_argument('--friction_max', type=float, default=0.3, help='Max friction')

    # Policy architecture
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden layer size')
    parser.add_argument('--n_hidden_layers', type=int, default=2, help='Number of hidden layers')

    # MAML hyperparameters
    parser.add_argument('--inner_lr', type=float, default=0.01, help='Inner loop learning rate')
    parser.add_argument('--inner_steps', type=int, default=5, help='Inner loop steps')
    parser.add_argument('--meta_lr', type=float, default=0.001, help='Meta learning rate')
    parser.add_argument('--first_order', action='store_true', help='Use first-order MAML')

    # Training
    parser.add_argument('--n_iterations', type=int, default=500, help='Training iterations')
    parser.add_argument('--tasks_per_batch', type=int, default=4, help='Tasks per meta-batch')
    parser.add_argument('--n_support', type=int, default=10, help='Support set size')
    parser.add_argument('--n_query', type=int, default=10, help='Query set size')
    parser.add_argument('--val_tasks', type=int, default=20, help='Validation tasks')

    # Logging
    parser.add_argument('--log_interval', type=int, default=10, help='Log every N iterations')
    parser.add_argument('--val_interval', type=int, default=50, help='Validate every N iterations')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    main(args)
