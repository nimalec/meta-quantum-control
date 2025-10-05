"""
Main Meta-Training Script

Train meta-learned initialization for quantum control.
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

from src.quantum.lindblad import LindbladSimulator
from src.quantum.noise_models import (
    TaskDistribution, NoisePSDModel, PSDToLindblad, NoiseParameters
)
from src.quantum.gates import GateFidelityComputer, TargetGates
from src.meta_rl.policy import PulsePolicy
from src.meta_rl.maml import MAML, MAMLTrainer


def create_quantum_system(config: dict):
    """Create quantum system simulator."""
    # Pauli matrices for 1-qubit
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    # System Hamiltonians
    H0 = 0.0 * sigma_z  # Drift (can be non-zero)
    H_controls = [sigma_x, sigma_y]  # Control Hamiltonians
    
    # PSD model
    psd_model = NoisePSDModel(model_type=config.get('psd_model', 'one_over_f'))
    
    # Sampling frequencies
    omega_sample = np.array([1.0, 5.0, 10.0])
    
    # PSD to Lindblad converter
    psd_to_lindblad = PSDToLindblad(
        basis_operators=[sigma_x, sigma_y, sigma_z],
        sampling_freqs=omega_sample,
        psd_model=psd_model
    )
    
    return {
        'H0': H0,
        'H_controls': H_controls,
        'psd_to_lindblad': psd_to_lindblad,
        'psd_model': psd_model
    }


def create_task_distribution(config: dict):
    """Create task distribution P."""
    return TaskDistribution(
        dist_type=config.get('task_dist_type', 'uniform'),
        ranges={
            'alpha': tuple(config.get('alpha_range', [0.5, 2.0])),
            'A': tuple(config.get('A_range', [0.05, 0.3])),
            'omega_c': tuple(config.get('omega_c_range', [2.0, 8.0]))
        }
    )


def task_sampler(n_tasks: int, split: str, task_dist: TaskDistribution, rng: np.random.Generator):
    """Sample tasks from distribution."""
    # Different random seeds for train/val/test
    if split == 'train':
        seed_offset = 0
    elif split == 'val':
        seed_offset = 100000
    else:  # test
        seed_offset = 200000
    
    local_rng = np.random.default_rng(rng.integers(0, 1000000) + seed_offset)
    return task_dist.sample(n_tasks, local_rng)


def data_generator(
    task_params: NoiseParameters,
    n_trajectories: int,
    split: str,
    quantum_system: dict,
    config: dict,
    device: torch.device
):
    """Generate data for a task."""
    # Just return task features - actual simulation happens in loss function
    task_features = torch.tensor(
        task_params.to_array(),
        dtype=torch.float32,
        device=device
    )
    
    # Repeat for batch
    task_features_batch = task_features.unsqueeze(0).repeat(n_trajectories, 1)
    
    return {
        'task_features': task_features_batch,
        'task_params': task_params,
        'quantum_system': quantum_system
    }


def create_loss_function(env, device):
    """Create loss function using QuantumEnvironment."""
    
    def loss_fn(policy: torch.nn.Module, data: dict):
        """
        Loss = 1 - Fidelity(ρ_final, ρ_target)
        
        Args:
            policy: Policy network
            data: Dictionary with task_features and task_params
            
        Returns:
            loss: Scalar tensor
        """
        task_params = data['task_params']
        
        # Use environment to compute loss
        loss = env.compute_loss(policy, task_params, device)
        
        return loss
    
    return loss_fn


def main(config_path: str):
    """Main training loop."""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 70)
    print("Meta-RL for Quantum Control - Training")
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
    
    # Target gate (e.g., Hadamard)
    target_gate_name = config.get('target_gate', 'hadamard')
    if target_gate_name == 'hadamard':
        U_target = TargetGates.hadamard()
    elif target_gate_name == 'pauli_x':
        U_target = TargetGates.pauli_x()
    else:
        raise ValueError(f"Unknown target gate: {target_gate_name}")
    
    # Target state: U|0⟩
    ket_0 = np.array([1, 0], dtype=complex)
    target_state = np.outer(U_target @ ket_0, (U_target @ ket_0).conj())
    print(f"Target gate: {target_gate_name}")
    
    # Create quantum environment (NEW!)
    print("\nSetting up quantum environment...")
    from src.theory.quantum_environment import create_quantum_environment
    env = create_quantum_environment(config, target_state)
    print(f"  Environment created: {env.get_cache_stats()}")
    
    # Create task distribution
    print("\nCreating task distribution...")
    task_dist = create_task_distribution(config)
    variance = task_dist.compute_variance()
    print(f"  Task variance σ²_θ = {variance:.4f}\n")
    
    # Create policy
    print("Creating policy network...")
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
    print(f"  Lipschitz constant: {policy.get_lipschitz_constant():.2f}")
    
    # Create MAML
    print("\nInitializing MAML...")
    maml = MAML(
        policy=policy,
        inner_lr=config.get('inner_lr', 0.01),
        inner_steps=config.get('inner_steps', 5),
        meta_lr=config.get('meta_lr', 0.001),
        first_order=config.get('first_order', False),
        device=device
    )
    print(f"  Inner: {maml.inner_steps} steps @ lr={maml.inner_lr}")
    print(f"  Meta lr: {maml.meta_lr}")
    print(f"  Second-order: {not maml.first_order}")
    
    # Create loss function (FIXED!)
    loss_fn = create_loss_function(env, device)
    
    # Modified data generator to work with environment
    def data_generator_env(task_params, n_trajectories, split):
        """Generate data compatible with environment."""
        task_features = torch.tensor(
            task_params.to_array(),
            dtype=torch.float32,
            device=device
        )
        
        # Repeat for batch
        task_features_batch = task_features.unsqueeze(0).repeat(n_trajectories, 1)
        
        return {
            'task_features': task_features_batch,
            'task_params': task_params  # Single task params
        }
    
    # Create trainer
    print("\nSetting up trainer...")
    trainer = MAMLTrainer(
        maml=maml,
        task_sampler=lambda n, split: task_sampler(n, split, task_dist, rng),
        data_generator=data_generator_env,
        loss_fn=loss_fn,
        n_support=config.get('n_support', 10),
        n_query=config.get('n_query', 10),
        log_interval=config.get('log_interval', 10),
        val_interval=config.get('val_interval', 50)
    )
    
    # Create save directory
    save_dir = Path(config.get('save_dir', 'checkpoints'))
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = save_dir / f"maml_{timestamp}.pt"
    
    print(f"\nCheckpoints will be saved to: {save_path}")
    
    # Train
    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70 + "\n")
    
    trainer.train(
        n_iterations=config.get('n_iterations', 1000),
        tasks_per_batch=config.get('tasks_per_batch', 4),
        val_tasks=config.get('val_tasks', 20),
        save_path=str(save_path)
    )
    
    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)
    print(f"\nFinal model saved to: {save_path}")
    print(f"Best model saved to: {str(save_path).replace('.pt', '_best.pt')}")
    print(f"\nCache stats: {env.get_cache_stats()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train meta-learned quantum controller')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/experiment_config.yaml',
        help='Path to config file'
    )
    
    args = parser.parse_args()
    
    main(args.config)
