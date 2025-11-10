"""
Main Meta-Training Script

Train meta-learned initialization for quantum control.
"""

import torch
import numpy as np
import yaml
from pathlib import Path
from datetime import datetime
import argparse

from metaqctrl.quantum.lindblad import LindbladSimulator
from metaqctrl.quantum.noise_adapter import (
    TaskDistribution, NoisePSDModel, PSDToLindblad2, NoiseParameters,
    estimate_qubit_frequency_from_hamiltonian  
)
from metaqctrl.quantum.gates import GateFidelityComputer, TargetGates
from metaqctrl.meta_rl.policy import PulsePolicy
from metaqctrl.meta_rl.maml import MAML, MAMLTrainer


def create_quantum_system(config: dict):
    ## create a 1 qubit simulator ==> 
    """Create quantum system simulator."""
    # Pauli matrices for 1-qubit
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    # System Hamiltonians ==> initialize Hamiltonian
    drift_strength = config.get('drift_strength')
    H0 = drift_strength * sigma_z  # Drift (small non-zero for better dynamics)
    H_controls = [sigma_x, sigma_y]  # Control Hamiltonians

    # PSD model ==> define a PSD noise model
    psd_model = NoisePSDModel(model_type=config.get('psd_model'))

    # Sampling frequencies ==> define sample frequencies across control bandwidth
    omega_sample = np.linspace(1, 1e5, 1000)

    # Physics parameters for v2 (with sensible defaults)
    T = config.get('horizon')  # Gate time

    # Estimate qubit frequency from Hamiltonian
    omega0 = config.get('omega0')
    if omega0 is None:
        omega0 = estimate_qubit_frequency_from_hamiltonian(H0)

    # Get coupling for noise type
    noise_type = config.get('noise_type')

    sequence = config.get('sequence')
    Gamma_h = 100    

    # PSD to Lindblad converter (uses v2 physics via adapter)
    psd_to_lindblad = PSDToLindblad2(
        basis_operators=[sigma_x, sigma_y, sigma_z],
        sampling_freqs=omega_sample,
        psd_model=psd_model,
        T=T,
        sequence=sequence,
        omega0=omega0,
        Gamma_h=Gamma_h
    )
    
    return {
        'H0': H0,
        'H_controls': H_controls,
        'psd_to_lindblad': psd_to_lindblad,
        'psd_model': psd_model
    }


def create_task_distribution(config: dict):
    ## Create a distribution of tasks , generating P
    """Create task distribution P (with optional mixed model support)."""
    # NEW: Support for mixed model sampling
    model_types = config.get('model_types')
    model_probs = config.get('model_probs')

    return TaskDistribution(
        dist_type=config.get('task_dist_type'),
        ranges={
            'alpha': tuple(config.get('alpha_range')),
            'A': tuple(config.get('A_range')),
            'omega_c': tuple(config.get('omega_c_range'))
        },
        model_types=model_types,  # NEW: List of model types or None for single model
        model_probs=model_probs   # NEW: Probabilities for each model type
    )


def task_sampler(n_tasks: int, split: str, task_dist: TaskDistribution, rng: np.random.Generator):
    ## Sample tasks 
    """Sample tasks from distribution."""  
    #local_rng = np.random.default_rng(rng.integers(0, 1000000) + seed_offset)
    local_rng = np.random.default_rng() 
    return task_dist.sample(n_tasks, local_rng)


def data_generator(
    task_params: NoiseParameters,
    n_trajectories: int,
    split: str,
    quantum_system: dict,
    config: dict,
    device: torch.device
):
    ## Generates data for a given task
    """Generate data for a task."""
    # Just return task features - actual simulation happens in loss function
    # NEW: Include model type in features (4D instead of 3D)
    task_features = torch.tensor(
        task_params.to_array(include_model=True),  # 4D: [alpha, A, omega_c, model_encoding]
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


def create_loss_function(env, device, config):
    #Make a loss function .
    """Create loss function using QuantumEnvironment."""

    # GPU-optimized settings from config
    dt = config.get('dt_training')  # Default to 0.01 if not specified
    use_rk4 = config.get('use_rk4_training')    # Default to RK4 if not specified

    def loss_fn(policy: torch.nn.Module, data: dict):
        ## define a loss function
        """
        Loss = 1 - Fidelity(ρ_final, ρ_target)

        Args:
            policy: Policy network
            data: Dictionary with task_features and task_params

        Returns:
            loss: Scalar tensor
        """
        task_params = data['task_params']

        # compute differentiable loss with GPU-optimized settings
        loss = env.compute_loss_differentiable(
            policy,
            task_params,
            device,
            use_rk4=use_rk4,
            dt=dt
        )

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
    rng = np.random.default_rng()
    
    # Device --> define a GPU device 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Target gate (e.g., Hadamard)
    ## Establish target operation  --> trained to optimize a pauli-x operator  
    target_gate_name = config.get('target_gate') 
  
    if target_gate_name == 'hadamard':
        U_target = TargetGates.hadamard()
    elif target_gate_name == 'pauli_x':
        U_target = TargetGates.pauli_x()
    else:
        raise ValueError(f"Unknown target gate: {target_gate_name}")
    
    # Target state: U|0⟩
    ## Establish target operation   
    ket_0 = np.array([1, 0], dtype=complex)
    target_state = np.outer(U_target @ ket_0, (U_target @ ket_0).conj())
    print(f"Target gate: {target_gate_name}")
    
    # Create quantum environment (NEW!)
    print("\nSetting up quantum environment...")
    ## Create a setting for a new enviroment
    from metaqctrl.theory.quantum_environment import create_quantum_environment
    ## Make a quantum enviroment
    ## Establish quantum enviroment  
    env = create_quantum_environment(config, target_state)
    print(f"  Environment created: {env.get_cache_stats()}")
    
    # Create task distribution --> generate a new task distribution 
    print("\nCreating task distribution...")
    task_dist = create_task_distribution(config)
   # variance = task_dist.compute_variance()
   # print(f"  Task variance σ²_θ = {variance:.4f}\n")

    
    # Create policy
    print("Creating policy network...")
    policy = PulsePolicy(
        task_feature_dim=config.get('task_feature_dim'),
        hidden_dim=config.get('hidden_dim'),
        n_hidden_layers=config.get('n_hidden_layers'),
        n_segments=config.get('n_segments'),
        n_controls=config.get('n_controls'),
        output_scale=config.get('output_scale'),
        activation=config.get('activation')
    )
    ## Make a policy  
    policy = policy.to(device)
    print(f"  Parameters: {policy.count_parameters():,}")
    print(f"  Lipschitz constant: {policy.get_lipschitz_constant():.2f}")
    
    # Create MAML
    print("\nInitializing MAML...")
    maml = MAML(
        policy=policy,
        inner_lr=config.get('inner_lr'),
        inner_steps=config.get('inner_steps'),
        meta_lr=config.get('meta_lr'),
        first_order=config.get('first_order'),
        device=device
    )
    print(f"  Inner: {maml.inner_steps} steps @ lr={maml.inner_lr}")
    print(f"  Meta lr: {maml.meta_lr}")
    print(f"  Second-order: {not maml.first_order}")
    
    # Create loss function with GPU-optimized settings
    loss_fn = create_loss_function(env, device, config)

    # Print integration settings
    print(f"\nIntegration settings:")
    print(f"  dt: {config.get('dt_training', 0.01)}")
    print(f"  method: {'RK4' if config.get('use_rk4_training', True) else 'Euler'}")
    
    # Modified data generator to work with environment
    def data_generator_env(task_params, n_trajectories, split):
        """Generate data compatible with environment."""
        # NEW: Include model type in features (4D instead of 3D)
        task_features = torch.tensor(
            task_params.to_array(include_model=True),  # 4D: [alpha, A, omega_c, model_encoding]
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
        n_support=config.get('n_support'),
        n_query=config.get('n_query'),
        log_interval=config.get('log_interval'),
        val_interval=config.get('val_interval')
    )
    
    # Create save directory
    save_dir = Path(config.get('save_dir', 'checkpoints'))
    save_dir.mkdir(parents=True, exist_ok=True) 
    save_path = save_dir / f"maml_best_pauli_x.pt"
    
    print(f"\nCheckpoints will be saved to: {save_path}")
    
    # Train
    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70 + "\n")
    
    trainer.train(
        n_iterations=config.get('n_iterations'),
        tasks_per_batch=config.get('tasks_per_batch'),
        val_tasks=config.get('val_tasks'),
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
        default='../../configs/experiment_config.yaml',
        help='Path to config file'
    )
    
    args = parser.parse_args()
    
    main(args.config)
