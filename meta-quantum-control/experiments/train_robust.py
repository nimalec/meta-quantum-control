"""
Train Robust Baseline Policy

Train a policy without adaptation capability for comparison with meta-learning.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import yaml
from pathlib import Path
from datetime import datetime
import argparse

from src.meta_rl.policy import PulsePolicy
from src.baselines.robust_control import RobustPolicy, RobustTrainer

# Import system creation from train_meta
from train_meta import (
    create_quantum_system, 
    create_task_distribution,
    task_sampler,
    data_generator,
    create_loss_function
)
from src.quantum.gates import TargetGates


def main(config_path: str):
    """Main training loop for robust baseline."""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 70)
    print("Robust Baseline Training")
    print("=" * 70)
    print(f"Config: {config_path}\n")
    
    # Set random seeds
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Create quantum system
    print("Setting up quantum system...")
    quantum_system = create_quantum_system(config)
    
    # Create task distribution
    print("Creating task distribution...")
    task_dist = create_task_distribution(config)
    variance = task_dist.compute_variance()
    print(f"  Task variance σ²_θ = {variance:.4f}\n")
    
    # Target gate
    target_gate_name = config.get('target_gate', 'hadamard')
    if target_gate_name == 'hadamard':
        U_target = TargetGates.hadamard()
    elif target_gate_name == 'pauli_x':
        U_target = TargetGates.pauli_x()
    else:
        raise ValueError(f"Unknown target gate: {target_gate_name}")
    
    ket_0 = np.array([1, 0], dtype=complex)
    target_state = np.outer(U_target @ ket_0, (U_target @ ket_0).conj())
    print(f"Target gate: {target_gate_name}")
    
    # Create policy
    print("\nCreating policy network...")
    policy = PulsePolicy(
        task_feature_dim=config.get('task_feature_dim', 3),
        hidden_dim=config.get('hidden_dim', 128),
        n_hidden_layers=config.get('n_hidden_layers', 2),
        n_segments=config.get('n_segments', 20),
        n_controls=config.get('n_controls', 2),
        output_scale=config.get('output_scale', 1.0),
        activation=config.get('activation', 'tanh')
    )
    policy = policy.to(device)
    print(f"  Parameters: {policy.count_parameters():,}")
    
    # Create robust policy wrapper
    print("\nInitializing robust policy...")
    robust_type = config.get('robust_type', 'minimax')
    robust_policy = RobustPolicy(
        policy=policy,
        learning_rate=config.get('meta_lr', 0.001),  # Use same LR as meta
        robust_type=robust_type,
        device=device
    )
    print(f"  Robust type: {robust_type}")
    
    # Create loss function
    loss_fn = create_loss_function(config, target_state)
    
    # Create trainer
    print("\nSetting up trainer...")
    n_samples = config.get('n_support', 10) + config.get('n_query', 10)
    trainer = RobustTrainer(
        robust_policy=robust_policy,
        task_sampler=lambda n, split: task_sampler(n, split, task_dist, rng),
        data_generator=lambda tp, n, split: data_generator(
            tp, n, split, quantum_system, config, device
        ),
        loss_fn=loss_fn,
        n_samples_per_task=n_samples,
        log_interval=config.get('log_interval', 10)
    )
    
    # Create save directory
    save_dir = Path(config.get('save_dir', 'checkpoints'))
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = save_dir / f"robust_{robust_type}_{timestamp}.pt"
    
    print(f"\nCheckpoints will be saved to: {save_path}")
    
    # Train
    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70 + "\n")
    
    n_iterations = config.get('robust_iterations', config.get('n_iterations', 2000))
    tasks_per_batch = config.get('robust_tasks_per_batch', 16)
    
    trainer.train(
        n_iterations=n_iterations,
        tasks_per_batch=tasks_per_batch,
        val_tasks=config.get('val_tasks', 50),
        val_interval=config.get('val_interval', 50),
        save_path=str(save_path)
    )
    
    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)
    print(f"\nFinal model saved to: {save_path}")
    print(f"Best model saved to: {str(save_path).replace('.pt', '_best.pt')}")
    
    # Compare training curves if meta model exists
    print("\n" + "=" * 70)
    print("Training Summary")
    print("=" * 70)
    print(f"Robust type: {robust_type}")
    print(f"Final training loss: {robust_policy.train_losses[-1]:.4f}")
    print(f"Best validation loss: (check logs)")
    print("\nNext steps:")
    print("1. Compare with meta-learned policy using eval_gap.py")
    print("2. Analyze learning curves in notebooks/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train robust baseline')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/experiment_config.yaml',
        help='Path to config file'
    )
    
    args = parser.parse_args()
    
    main(args.config)
