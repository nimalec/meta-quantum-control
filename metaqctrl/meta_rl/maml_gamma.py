"""
Model-Agnostic Meta-Learning (MAML) for Gamma-Parameterized Quantum Control

This module provides MAML training for policies using gamma-rate noise parameters.
The MAML algorithm itself is task-agnostic, so we re-export the original implementation
with gamma-specific documentation and helper classes.

Key differences from maml.py usage:
- Uses GammaNoiseParameters instead of NoiseParameters
- Uses GammaTaskDistribution instead of TaskDistribution
- Loss function uses DifferentiableLindbladSimulator with direct gamma rates

Usage:
    from metaqctrl.meta_rl.maml_gamma import MAML, GammaMAMLTrainer
    from metaqctrl.meta_rl.policy_gamma import GammaPulsePolicy
    from metaqctrl.quantum.noise_models_gamma import GammaTaskDistribution

    policy = GammaPulsePolicy(task_feature_dim=3, ...)
    maml = MAML(policy, inner_lr=0.01, inner_steps=5, ...)
    trainer = GammaMAMLTrainer(maml, task_dist, loss_fn, ...)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch import autograd
import numpy as np
from typing import List, Tuple, Dict, Callable, Optional
from copy import deepcopy

# Import core MAML classes (algorithm is task-agnostic)
from metaqctrl.meta_rl.maml import MAML, MAMLTrainer

# Import gamma-specific modules
from metaqctrl.quantum.noise_models_gamma import GammaNoiseParameters, GammaTaskDistribution
from metaqctrl.quantum.lindblad_torch import DifferentiableLindbladSimulator


def create_gamma_lindblad_simulator(
    gamma_deph: float,
    gamma_relax: float,
    device: str = 'cpu'
) -> DifferentiableLindbladSimulator:
    """
    Create a DifferentiableLindbladSimulator with gamma-rate Lindblad operators.

    This directly uses gamma rates - no PSD-to-Lindblad conversion needed.

    Args:
        gamma_deph: Pure dephasing rate [1/s]
        gamma_relax: Relaxation rate [1/s]
        device: 'cpu' or 'cuda'

    Returns:
        sim: DifferentiableLindbladSimulator ready for quantum evolution
    """
    # Pauli matrices
    sigma_x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device=device)
    sigma_y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64, device=device)
    sigma_z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=device)

    # Lindblad operators for gamma rates
    # L_relax = sqrt(gamma_relax) * |g><e| (relaxation)
    L_relax = torch.sqrt(torch.tensor(gamma_relax, dtype=torch.float32, device=device)) * \
              torch.tensor([[0, 1], [0, 0]], dtype=torch.complex64, device=device)

    # L_deph = sqrt(gamma_deph/2) * sigma_z (pure dephasing)
    L_deph = torch.sqrt(torch.tensor(gamma_deph / 2.0, dtype=torch.float32, device=device)) * \
             sigma_z.to(torch.complex64)

    L_operators = [L_relax, L_deph]

    # Create simulator
    H0 = torch.zeros((2, 2), dtype=torch.complex64, device=device)  # No drift
    H_controls = [sigma_x, sigma_y]

    sim = DifferentiableLindbladSimulator(
        H0=H0,
        H_controls=H_controls,
        L_operators=L_operators,
        dt=0.05,
        device=device
    )

    return sim


def compute_gamma_loss(
    policy: nn.Module,
    task_params: GammaNoiseParameters,
    target_state: torch.Tensor,
    device: str = 'cpu',
    T: float = 1.0
) -> torch.Tensor:
    """
    Compute loss for a gamma-parameterized task.

    Loss = 1 - Fidelity(ρ_final, ρ_target)

    Args:
        policy: GammaPulsePolicy network
        task_params: GammaNoiseParameters with gamma_deph, gamma_relax
        target_state: Target density matrix (2x2 complex tensor)
        device: 'cpu' or 'cuda'
        T: Total evolution time

    Returns:
        loss: Scalar tensor (differentiable)
    """
    # Create simulator for this task's gamma rates
    sim = create_gamma_lindblad_simulator(
        task_params.gamma_deph,
        task_params.gamma_relax,
        device=device
    )

    # Get task features for policy input
    task_features = torch.tensor(
        task_params.to_array(normalized=True),
        dtype=torch.float32,
        device=device
    )

    # Generate control sequence
    controls = policy(task_features)

    # Initial state: |0><0|
    rho0 = torch.tensor(
        [[1, 0], [0, 0]],
        dtype=torch.complex64,
        device=device
    )

    # Evolve quantum state
    rho_final = sim.forward(rho0, controls, T=T)

    # Compute fidelity: F = Tr(ρ_final * ρ_target)
    # For pure target states, this simplifies
    fidelity = torch.real(torch.trace(rho_final @ target_state.to(device)))

    # Loss = 1 - Fidelity
    loss = 1.0 - fidelity

    return loss


class GammaMAMLTrainer(MAMLTrainer):
    """
    High-level trainer for MAML with gamma-parameterized tasks.

    Extends MAMLTrainer with gamma-specific task sampling and data generation.

    Example:
        # Setup
        policy = GammaPulsePolicy(task_feature_dim=3, ...)
        maml = MAML(policy, inner_lr=0.01, inner_steps=5)

        task_dist = GammaTaskDistribution(
            gamma_deph_range=(0.02, 0.15),
            gamma_relax_range=(0.01, 0.08)
        )

        def gamma_loss_fn(policy, data):
            return compute_gamma_loss(policy, data['task_params'], target_state)

        trainer = GammaMAMLTrainer(
            maml=maml,
            task_sampler=lambda n, split: task_dist.sample(n),
            data_generator=gamma_data_generator,
            loss_fn=gamma_loss_fn
        )

        trainer.train(n_iterations=1000)
    """

    def __init__(
        self,
        maml: MAML,
        task_distribution: GammaTaskDistribution,
        target_state: torch.Tensor,
        device: torch.device = torch.device('cpu'),
        n_support: int = 1,
        n_query: int = 1,
        log_interval: int = 10,
        val_interval: int = 50
    ):
        """
        Args:
            maml: MAML instance with GammaPulsePolicy
            task_distribution: GammaTaskDistribution for sampling tasks
            target_state: Target quantum state (2x2 density matrix)
            device: torch device
            n_support: Number of support evaluations per task
            n_query: Number of query evaluations per task
            log_interval: Log every N iterations
            val_interval: Validate every N iterations
        """
        self.task_distribution = task_distribution
        self.target_state = target_state
        self.device = device

        # Create loss function
        def gamma_loss_fn(policy, data):
            task_params = data['task_params']
            return compute_gamma_loss(
                policy, task_params, self.target_state, str(self.device)
            )

        # Create task sampler
        def gamma_task_sampler(n_tasks, split='train'):
            return self.task_distribution.sample(n_tasks)

        # Create data generator
        def gamma_data_generator(task_params, n_trajectories, split):
            task_features = torch.tensor(
                task_params.to_array(normalized=True),
                dtype=torch.float32,
                device=self.device
            )
            return {
                'task_features': task_features.unsqueeze(0).repeat(n_trajectories, 1),
                'task_params': task_params
            }

        # Initialize parent
        super().__init__(
            maml=maml,
            task_sampler=gamma_task_sampler,
            data_generator=gamma_data_generator,
            loss_fn=gamma_loss_fn,
            n_support=n_support,
            n_query=n_query,
            log_interval=log_interval,
            val_interval=val_interval
        )


# Re-export for convenience
__all__ = [
    'MAML',
    'MAMLTrainer',
    'GammaMAMLTrainer',
    'GammaNoiseParameters',
    'GammaTaskDistribution',
    'create_gamma_lindblad_simulator',
    'compute_gamma_loss'
]


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("Gamma MAML - Example Usage")
    print("=" * 60)

    from metaqctrl.meta_rl.policy_gamma import GammaPulsePolicy

    # Create policy
    policy = GammaPulsePolicy(
        task_feature_dim=3,
        hidden_dim=64,
        n_hidden_layers=2,
        n_segments=20,
        n_controls=2
    )

    # Initialize MAML
    maml = MAML(
        policy=policy,
        inner_lr=0.01,
        inner_steps=5,
        meta_lr=0.001,
        first_order=True  # FOMAML for faster training
    )

    print(f"\nMAML initialized with gamma policy: {policy.count_parameters():,} parameters")
    print(f"Inner loop: {maml.inner_steps} steps @ lr={maml.inner_lr}")
    print(f"Meta-learning rate: {maml.meta_lr}")

    # Create task distribution
    task_dist = GammaTaskDistribution(
        gamma_deph_range=(0.02, 0.15),
        gamma_relax_range=(0.01, 0.08)
    )

    # Sample some tasks
    tasks = task_dist.sample(3)
    print(f"\nSampled {len(tasks)} tasks:")
    for i, t in enumerate(tasks):
        print(f"  Task {i}: {t}")

    # Test loss computation
    print("\nTesting loss computation...")
    target_state = torch.tensor(
        [[0.5, 0.5], [0.5, 0.5]],  # |+> state for X gate
        dtype=torch.complex64
    )

    loss = compute_gamma_loss(policy, tasks[0], target_state)
    print(f"Loss for task 0: {loss.item():.4f}")
    print(f"Fidelity: {1 - loss.item():.4f}")
