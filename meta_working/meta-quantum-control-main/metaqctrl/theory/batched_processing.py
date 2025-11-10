"""
Batched Task Processing for GPU Optimization

This module provides batched processing utilities for evaluating multiple
tasks in parallel on GPU. This significantly improves GPU utilization compared
to sequential task processing.

Key optimization: Process multiple tasks simultaneously using PyTorch's
vectorization capabilities.
"""

import torch
import numpy as np
from typing import List, Tuple
from metaqctrl.quantum.noise_models import NoiseParameters
from metaqctrl.theory.quantum_environment import QuantumEnvironment


class BatchedLossComputation:
    """
    Compute losses for multiple tasks in parallel on GPU.

    This is a significant optimization over sequential processing, especially
    for GPU training where parallelism is crucial.
    """

    def __init__(self, env: QuantumEnvironment, device: torch.device, dt: float = 0.01, use_rk4: bool = True):
        """
        Args:
            env: QuantumEnvironment instance
            device: torch device
            dt: Integration time step
            use_rk4: If True, use RK4 integration
        """
        self.env = env
        self.device = device
        self.dt = dt
        self.use_rk4 = use_rk4

    def compute_batch(
        self,
        policy: torch.nn.Module,
        task_params_list: List[NoiseParameters]
    ) -> torch.Tensor:
        """
        Compute losses for a batch of tasks in parallel.

        OPTIMIZATION: Instead of processing tasks sequentially:
        - for task in tasks: loss = compute_loss(task)

        We process them in parallel:
        - losses = compute_batch(all_tasks)

        This better utilizes GPU parallelism.

        Args:
            policy: Policy network
            task_params_list: List of task parameters

        Returns:
            losses: Tensor of losses (batch_size,)
        """
        batch_size = len(task_params_list)

        # Stack task features
        task_features_list = [
            torch.tensor(tp.to_array(), dtype=torch.float32, device=self.device)
            for tp in task_params_list
        ]
        task_features_batch = torch.stack(task_features_list)  # (batch_size, 3)

        # Generate controls for all tasks in one forward pass
        # This is much more efficient than calling policy separately for each task
        controls_batch = policy(task_features_batch)  # (batch_size, n_segments, n_controls)

        # Compute losses for each task
        # NOTE: We still need to process tasks sequentially here because
        # each task has different Lindblad operators (different noise)
        # A future optimization could batch tasks with similar noise parameters
        losses = []
        for i, task_params in enumerate(task_params_list):
            controls = controls_batch[i]  # (n_segments, n_controls)

            # Get cached simulator
            sim = self.env.get_torch_simulator(
                task_params,
                self.device,
                dt=self.dt,
                use_rk4=self.use_rk4
            )

            # Initial state |0⟩
            rho0 = torch.zeros((self.env.d, self.env.d), dtype=torch.complex64, device=self.device)
            rho0[0, 0] = 1.0

            # Evolve
            rho_final = sim(rho0, controls, self.env.T)

            # Compute fidelity
            from metaqctrl.quantum.lindblad_torch import numpy_to_torch_complex
            target_state_torch = numpy_to_torch_complex(self.env.target_state, self.device)
            fidelity = self.env._torch_state_fidelity(rho_final, target_state_torch)

            # Loss = infidelity
            loss = 1.0 - fidelity
            losses.append(loss)

        # Stack losses
        losses_tensor = torch.stack(losses)

        return losses_tensor

    def __call__(self, policy: torch.nn.Module, task_params_list: List[NoiseParameters]) -> torch.Tensor:
        """Callable interface for convenience."""
        return self.compute_batch(policy, task_params_list)


def create_batched_loss_function(env: QuantumEnvironment, device: torch.device, config: dict):
    """
    Create a batched loss function for improved GPU utilization.

    This function processes multiple tasks more efficiently by:
    1. Batching policy forward passes
    2. Reusing cached simulators
    3. Minimizing CPU/GPU transfers

    Args:
        env: QuantumEnvironment instance
        device: torch device
        config: Configuration dictionary

    Returns:
        loss_fn: Batched loss function
    """
    dt = config.get('dt_training', 0.01)
    use_rk4 = config.get('use_rk4_training', True)

    batch_computer = BatchedLossComputation(env, device, dt, use_rk4)

    def batched_loss_fn(policy: torch.nn.Module, data_batch: List[dict]) -> Tuple[torch.Tensor, List[float]]:
        """
        Compute losses for a batch of tasks.

        Args:
            policy: Policy network
            data_batch: List of data dictionaries, each with 'task_params'

        Returns:
            mean_loss: Mean loss across batch (scalar tensor)
            losses: List of individual losses (for logging)
        """
        # Extract task parameters
        task_params_list = [data['task_params'] for data in data_batch]

        # Compute losses in batch
        losses_tensor = batch_computer(policy, task_params_list)

        # Mean loss for backprop
        mean_loss = losses_tensor.mean()

        # Individual losses for logging
        losses_list = [loss.item() for loss in losses_tensor]

        return mean_loss, losses_list

    return batched_loss_fn


# Utility function for benchmarking
def benchmark_batching_speedup(
    env: QuantumEnvironment,
    policy: torch.nn.Module,
    task_params_list: List[NoiseParameters],
    device: torch.device,
    n_repeats: int = 3
):
    """
    Benchmark speedup from batched processing.

    Args:
        env: QuantumEnvironment instance
        policy: Policy network
        task_params_list: List of tasks to test
        device: torch device
        n_repeats: Number of timing runs

    Returns:
        speedup: Speedup factor (sequential_time / batched_time)
    """
    import time

    # Sequential processing
    sequential_times = []
    for _ in range(n_repeats):
        torch.cuda.synchronize() if device.type == 'cuda' else None
        t0 = time.time()

        for task_params in task_params_list:
            loss = env.compute_loss_differentiable(policy, task_params, device)
            loss.backward(retain_graph=True)

        torch.cuda.synchronize() if device.type == 'cuda' else None
        sequential_times.append(time.time() - t0)

    # Batched processing
    batch_computer = BatchedLossComputation(env, device)
    batched_times = []
    for _ in range(n_repeats):
        torch.cuda.synchronize() if device.type == 'cuda' else None
        t0 = time.time()

        losses = batch_computer(policy, task_params_list)
        losses.mean().backward()

        torch.cuda.synchronize() if device.type == 'cuda' else None
        batched_times.append(time.time() - t0)

    avg_sequential = np.mean(sequential_times)
    avg_batched = np.mean(batched_times)
    speedup = avg_sequential / avg_batched

    print(f"Benchmark results ({len(task_params_list)} tasks):")
    print(f"  Sequential: {avg_sequential:.3f}s")
    print(f"  Batched: {avg_batched:.3f}s")
    print(f"  Speedup: {speedup:.2f}x")

    return speedup
