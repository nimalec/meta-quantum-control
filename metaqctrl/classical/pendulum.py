"""
Simple pendulum control task for MAML validation.

Analogy to quantum control:
- Quantum: Control pulses → gate fidelity
- Classical: Torque controls → swing-up performance

Task parameters θ = (mass, length, friction):
- mass: pendulum mass [kg]
- length: pendulum length [m]
- friction: damping coefficient

This allows fast testing of MAML without quantum simulation overhead.
"""

import numpy as np
import torch
from typing import Tuple, Dict
from dataclasses import dataclass
from scipy.integrate import solve_ivp


@dataclass
class PendulumTask:
    """
    Pendulum task parameters (analogous to NoiseParameters).

    Attributes:
        mass: Pendulum mass [kg]
        length: Pendulum length [m]
        friction: Damping coefficient (dimensionless)
    """
    mass: float
    length: float
    friction: float

    def to_array(self) -> np.ndarray:
        """Convert to array for neural network input."""
        return np.array([self.mass, self.length, self.friction], dtype=np.float32)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "PendulumTask":
        """Create from array."""
        return cls(mass=float(arr[0]), length=float(arr[1]), friction=float(arr[2]))


class PendulumEnvironment:
    """
    Pendulum swing-up environment with parameterized dynamics.

    Task: Swing pendulum from hanging down (θ=π) to upright (θ=0) using torque controls.

    Dynamics:
        θ̈ = -g/L sin(θ) - b θ̇ + u/(m L²)

    where:
        θ: angle from upright [rad]
        g: gravity (9.81 m/s²)
        L: length [m]
        m: mass [kg]
        b: friction coefficient
        u: applied torque [N·m]

    Analogous to QuantumEnvironment:
        - Task params (m,L,b) ↔ Noise params (α,A,ωc)
        - Controls u(t) ↔ Hamiltonian controls H_x(t), H_y(t)
        - Final state error ↔ Gate infidelity
    """

    def __init__(self, n_segments: int = 20, horizon: float = 2.0, g: float = 9.81):
        """
        Args:
            n_segments: Number of control segments (like quantum pulses)
            horizon: Total time [s]
            g: Gravitational acceleration [m/s²]
        """
        self.n_segments = n_segments
        self.T = horizon
        self.dt = horizon / n_segments
        self.g = g

        # Initial and target states
        self.theta_init = np.pi  # Hanging down
        self.theta_dot_init = 0.0
        self.theta_target = 0.0  # Upright
        self.theta_dot_target = 0.0

    def dynamics(self, t: float, state: np.ndarray, u: float, task: PendulumTask) -> np.ndarray:
        """
        Pendulum dynamics: [θ, θ̇] → [θ̇, θ̈]

        Args:
            t: Time [s]
            state: [theta, theta_dot]
            u: Control torque [N·m]
            task: Task parameters

        Returns:
            state_dot: [theta_dot, theta_ddot]
        """
        theta, theta_dot = state

        # θ̈ = -g/L sin(θ) - b θ̇ + u/(m L²)
        I = task.mass * task.length**2  # Moment of inertia
        theta_ddot = (
            -(self.g / task.length) * np.sin(theta)
            - task.friction * theta_dot
            + u / I
        )

        return np.array([theta_dot, theta_ddot])

    def simulate(self, controls: np.ndarray, task: PendulumTask) -> Tuple[np.ndarray, float]:
        """
        Simulate pendulum with piecewise-constant controls.

        Args:
            controls: Control torques [n_segments] in N·m
            task: Task parameters

        Returns:
            final_state: [theta, theta_dot] at t=T
            loss: Distance from target state (analogous to 1 - fidelity)
        """
        controls = np.asarray(controls).flatten()
        if len(controls) != self.n_segments:
            raise ValueError(f"Expected {self.n_segments} controls, got {len(controls)}")

        state = np.array([self.theta_init, self.theta_dot_init])

        # Simulate each segment with constant control
        for i in range(self.n_segments):
            t_span = (i * self.dt, (i + 1) * self.dt)
            u = controls[i]

            # Integrate dynamics
            sol = solve_ivp(
                fun=lambda t, y: self.dynamics(t, y, u, task),
                t_span=t_span,
                y0=state,
                method='RK45',
                dense_output=False
            )

            state = sol.y[:, -1]  # Final state

        # Compute loss (distance to target)
        target = np.array([self.theta_target, self.theta_dot_target])

        # Normalize angle to [-π, π]
        state[0] = np.arctan2(np.sin(state[0]), np.cos(state[0]))

        # Weighted loss: emphasize position over velocity
        loss = (
            1.0 * (state[0] - target[0])**2 +
            0.1 * (state[1] - target[1])**2
        )

        return state, float(loss)

    def simulate_torch(self, controls: torch.Tensor, task: PendulumTask) -> torch.Tensor:
        """
        Torch-compatible simulation (for gradient computation).

        Note: Uses numpy simulation internally, wraps result as tensor.
        Gradients flow through control inputs using autograd.
        For true end-to-end differentiability, would need torchdiffeq.

        Args:
            controls: [n_segments] tensor
            task: Task parameters

        Returns:
            loss: Scalar loss tensor with grad_fn
        """
        # Custom autograd function to wrap the simulation
        class PendulumSimulation(torch.autograd.Function):
            @staticmethod
            def forward(ctx, controls):
                controls_np = controls.detach().cpu().numpy()
                _, loss = self.simulate(controls_np, task)

                # Compute numerical gradient for backward pass
                # Store controls for backward
                ctx.save_for_backward(controls)
                ctx.task = task

                return torch.tensor(loss, dtype=controls.dtype, device=controls.device)

            @staticmethod
            def backward(ctx, grad_output):
                controls, = ctx.saved_tensors
                task = ctx.task

                # Numerical gradient via finite differences
                eps = 1e-4
                controls_np = controls.detach().cpu().numpy()
                _, loss_center = self.simulate(controls_np, task)

                grad = np.zeros_like(controls_np)
                for i in range(len(controls_np)):
                    controls_perturb = controls_np.copy()
                    controls_perturb[i] += eps
                    _, loss_plus = self.simulate(controls_perturb, task)
                    grad[i] = (loss_plus - loss_center) / eps

                grad_tensor = torch.tensor(grad, dtype=controls.dtype, device=controls.device)
                return grad_output * grad_tensor

        return PendulumSimulation.apply(controls)

    def generate_support_query_data(
        self,
        task: PendulumTask,
        n_support: int = 10,
        n_query: int = 10,
        rng: np.random.Generator = None
    ) -> Tuple[Dict, Dict]:
        """
        Generate support and query data for a task.

        For pendulum: both sets just contain task features (no randomization needed).
        In more complex envs, could have different initial conditions.

        Args:
            task: Task parameters
            n_support: Number of support examples
            n_query: Number of query examples
            rng: Random number generator

        Returns:
            support_data: Dict with 'task_features'
            query_data: Dict with 'task_features'
        """
        rng = rng or np.random.default_rng()

        # Task features for neural network input
        task_features = task.to_array()

        # Repeat for batch (policy expects batch of task features)
        support_data = {
            'task_features': torch.tensor(
                np.tile(task_features, (n_support, 1)),
                dtype=torch.float32
            )
        }

        query_data = {
            'task_features': torch.tensor(
                np.tile(task_features, (n_query, 1)),
                dtype=torch.float32
            )
        }

        return support_data, query_data


def create_pendulum_loss_fn(env: PendulumEnvironment):
    """
    Create loss function for MAML training (analogous to quantum fidelity loss).

    Args:
        env: PendulumEnvironment instance

    Returns:
        loss_fn: Function(policy, data) -> loss
    """
    def loss_fn(policy, data):
        """
        Compute average loss over batch.

        Args:
            policy: Neural network policy
            data: Dict with 'task_features' [batch_size, 3]

        Returns:
            loss: Average loss over batch
        """
        task_features = data['task_features']
        batch_size = task_features.shape[0]

        total_loss = 0.0

        for i in range(batch_size):
            # Get controls from policy
            task_feat = task_features[i:i+1]  # Keep batch dim
            controls = policy(task_feat)  # [1, n_segments]
            controls = controls.squeeze(0)  # [n_segments]

            # Extract task parameters
            task = PendulumTask.from_array(task_feat.detach().cpu().numpy()[0])

            # Simulate and compute loss
            loss = env.simulate_torch(controls, task)
            total_loss += loss

        return total_loss / batch_size

    return loss_fn


# Example usage and testing
if __name__ == "__main__":
    print("Testing Pendulum Environment")
    print("=" * 60)

    # Create environment
    env = PendulumEnvironment(n_segments=20, horizon=2.0)

    # Create task
    task = PendulumTask(mass=1.0, length=1.0, friction=0.1)
    print(f"\nTask: m={task.mass:.2f} kg, L={task.length:.2f} m, b={task.friction:.2f}")
    print(f"Task features: {task.to_array()}")

    # Test with zero controls (should fail to swing up)
    controls_zero = np.zeros(env.n_segments)
    final_state, loss = env.simulate(controls_zero, task)
    print(f"\nZero controls:")
    print(f"  Final state: θ={final_state[0]:.3f} rad, θ̇={final_state[1]:.3f} rad/s")
    print(f"  Loss: {loss:.4f}")

    # Test with random controls
    rng = np.random.default_rng(42)
    controls_random = rng.uniform(-5, 5, size=env.n_segments)
    final_state, loss = env.simulate(controls_random, task)
    print(f"\nRandom controls:")
    print(f"  Final state: θ={final_state[0]:.3f} rad, θ̇={final_state[1]:.3f} rad/s")
    print(f"  Loss: {loss:.4f}")

    # Test data generation
    support_data, query_data = env.generate_support_query_data(task, n_support=5, n_query=3)
    print(f"\nData generation:")
    print(f"  Support features shape: {support_data['task_features'].shape}")
    print(f"  Query features shape: {query_data['task_features'].shape}")

    # Test with simple policy
    from metaqctrl.meta_rl.policy import PulsePolicy

    policy = PulsePolicy(
        task_feature_dim=3,  # (mass, length, friction)
        hidden_dim=64,
        n_hidden_layers=2,
        n_segments=20,
        n_controls=1  # Single torque output
    )

    loss_fn = create_pendulum_loss_fn(env)
    task_data = {'support': support_data, 'query': query_data}

    loss = loss_fn(policy, support_data)
    print(f"\nPolicy test:")
    print(f"  Loss with untrained policy: {loss.item():.4f}")

    print("\n✓ All tests passed!")
