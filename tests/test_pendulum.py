"""
Tests for classical pendulum environment.

These tests verify:
1. Pendulum dynamics are physically correct
2. Task distribution works properly
3. Integration with MAML works
4. Faster than quantum tests (for CI)
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from metaqctrl.classical.pendulum import (
    PendulumEnvironment,
    PendulumTask,
    create_pendulum_loss_fn
)
from metaqctrl.classical.task_distribution import PendulumTaskDistribution
from metaqctrl.meta_rl.policy import PulsePolicy
from metaqctrl.meta_rl.maml import MAML


class TestPendulumTask:
    """Test PendulumTask dataclass."""

    def test_creation(self):
        """Test creating a task."""
        task = PendulumTask(mass=1.0, length=1.0, friction=0.1)
        assert task.mass == 1.0
        assert task.length == 1.0
        assert task.friction == 0.1

    def test_to_array(self):
        """Test conversion to array."""
        task = PendulumTask(mass=1.5, length=0.8, friction=0.2)
        arr = task.to_array()
        assert arr.shape == (3,)
        assert np.allclose(arr, [1.5, 0.8, 0.2])

    def test_from_array(self):
        """Test creation from array."""
        arr = np.array([2.0, 1.2, 0.15])
        task = PendulumTask.from_array(arr)
        assert task.mass == 2.0
        assert task.length == 1.2
        assert task.friction == 0.15

    def test_roundtrip(self):
        """Test array conversion roundtrip."""
        task1 = PendulumTask(mass=1.3, length=0.9, friction=0.25)
        arr = task1.to_array()
        task2 = PendulumTask.from_array(arr)
        assert np.allclose(task1.to_array(), task2.to_array())


class TestPendulumEnvironment:
    """Test PendulumEnvironment."""

    def setup_method(self):
        """Set up test fixtures."""
        self.env = PendulumEnvironment(n_segments=20, horizon=2.0)
        self.task = PendulumTask(mass=1.0, length=1.0, friction=0.1)

    def test_initialization(self):
        """Test environment initialization."""
        assert self.env.n_segments == 20
        assert self.env.T == 2.0
        assert np.isclose(self.env.dt, 0.1)
        assert np.isclose(self.env.g, 9.81)

    def test_dynamics_shape(self):
        """Test dynamics function returns correct shape."""
        state = np.array([np.pi, 0.0])
        state_dot = self.env.dynamics(0.0, state, u=0.0, task=self.task)
        assert state_dot.shape == (2,)

    def test_zero_control_hangs_down(self):
        """Test that zero control leaves pendulum hanging."""
        controls = np.zeros(self.env.n_segments)
        final_state, loss = self.env.simulate(controls, self.task)

        # Should stay near hanging down position (θ ≈ π)
        # With friction, velocity should decay to zero
        assert abs(final_state[0] - np.pi) < 0.5, "Pendulum should stay near hanging position"
        assert abs(final_state[1]) < 0.5, "Velocity should be small with friction"

    def test_loss_positive(self):
        """Test that loss is always non-negative."""
        rng = np.random.default_rng(42)
        for _ in range(10):
            controls = rng.uniform(-10, 10, size=self.env.n_segments)
            _, loss = self.env.simulate(controls, self.task)
            assert loss >= 0, "Loss must be non-negative"

    def test_different_tasks_different_behavior(self):
        """Test that different task parameters produce different results."""
        controls = np.ones(self.env.n_segments) * 5.0

        task1 = PendulumTask(mass=0.5, length=0.5, friction=0.0)
        task2 = PendulumTask(mass=2.0, length=1.5, friction=0.3)

        state1, _ = self.env.simulate(controls, task1)
        state2, _ = self.env.simulate(controls, task2)

        # Different tasks should produce different final states
        assert not np.allclose(state1, state2), "Different tasks should behave differently"

    def test_torch_simulation(self):
        """Test torch-compatible simulation."""
        controls = torch.randn(self.env.n_segments) * 5.0
        loss = self.env.simulate_torch(controls, self.task)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert loss.item() >= 0

    def test_data_generation(self):
        """Test support/query data generation."""
        support_data, query_data = self.env.generate_support_query_data(
            self.task,
            n_support=5,
            n_query=3
        )

        assert 'task_features' in support_data
        assert 'task_features' in query_data
        assert support_data['task_features'].shape == (5, 3)
        assert query_data['task_features'].shape == (3, 3)


class TestPendulumTaskDistribution:
    """Test PendulumTaskDistribution."""

    def test_uniform_sampling(self):
        """Test uniform distribution."""
        dist = PendulumTaskDistribution(
            dist_type='uniform',
            ranges={
                'mass': (0.5, 2.0),
                'length': (0.5, 1.5),
                'friction': (0.0, 0.3)
            }
        )

        rng = np.random.default_rng(42)
        tasks = dist.sample(100, rng)

        assert len(tasks) == 100
        for task in tasks:
            assert 0.5 <= task.mass <= 2.0
            assert 0.5 <= task.length <= 1.5
            assert 0.0 <= task.friction <= 0.3

    def test_variance_computation(self):
        """Test variance computation."""
        dist = PendulumTaskDistribution(
            dist_type='uniform',
            ranges={
                'mass': (0.5, 2.0),
                'length': (0.5, 1.5),
                'friction': (0.0, 0.3)
            }
        )

        variance = dist.compute_variance()
        assert variance > 0

        # For uniform distribution: Var[U(a,b)] = (b-a)²/12
        expected = (
            ((2.0 - 0.5) ** 2) / 12.0 +
            ((1.5 - 0.5) ** 2) / 12.0 +
            ((0.3 - 0.0) ** 2) / 12.0
        )
        assert np.isclose(variance, expected)


class TestPendulumWithMAML:
    """Test integration with MAML."""

    def setup_method(self):
        """Set up test fixtures."""
        self.env = PendulumEnvironment(n_segments=20, horizon=2.0)
        self.policy = PulsePolicy(
            task_feature_dim=3,
            hidden_dim=32,
            n_hidden_layers=1,
            n_segments=20,
            n_controls=1
        )
        self.loss_fn = create_pendulum_loss_fn(self.env)

    def test_loss_function(self):
        """Test loss function works with policy."""
        task = PendulumTask(mass=1.0, length=1.0, friction=0.1)
        task_features = torch.tensor([task.to_array()], dtype=torch.float32)
        data = {'task_features': task_features}

        loss = self.loss_fn(self.policy, data)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_loss_function_batch(self):
        """Test loss function with batch."""
        tasks = [
            PendulumTask(mass=1.0, length=1.0, friction=0.1),
            PendulumTask(mass=1.5, length=1.2, friction=0.2),
            PendulumTask(mass=0.8, length=0.7, friction=0.05)
        ]
        task_features = torch.tensor([t.to_array() for t in tasks], dtype=torch.float32)
        data = {'task_features': task_features}

        loss = self.loss_fn(self.policy, data)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_maml_inner_loop(self):
        """Test that MAML inner loop runs without errors."""
        maml = MAML(
            policy=self.policy,
            inner_lr=0.01,
            inner_steps=3,
            meta_lr=0.001,
            first_order=True
        )

        task = PendulumTask(mass=1.0, length=1.0, friction=0.1)
        support_data, query_data = self.env.generate_support_query_data(
            task, n_support=5, n_query=3
        )

        task_data = {'support': support_data, 'query': query_data}

        # Run inner loop
        adapted_policy, losses = maml.inner_loop(task_data, self.loss_fn)

        assert len(losses) == 3
        assert all(l >= 0 for l in losses)

    def test_maml_meta_step(self):
        """Test that MAML meta step runs without errors."""
        maml = MAML(
            policy=self.policy,
            inner_lr=0.01,
            inner_steps=2,
            meta_lr=0.001,
            first_order=True  # Use FOMAML for speed
        )

        # Create task batch
        dist = PendulumTaskDistribution()
        tasks = dist.sample(2)

        task_batch = []
        for task in tasks:
            support_data, query_data = self.env.generate_support_query_data(
                task, n_support=5, n_query=3
            )
            task_batch.append({
                'support': support_data,
                'query': query_data
            })

        # Run meta step
        metrics = maml.meta_train_step(task_batch, self.loss_fn, use_higher=False)

        assert 'meta_loss' in metrics
        assert 'mean_task_loss' in metrics
        assert metrics['meta_loss'] >= 0


def run_tests():
    """Run all tests."""
    pytest.main([__file__, '-v', '--tb=short'])


if __name__ == "__main__":
    print("=" * 70)
    print("Pendulum Environment Test Suite")
    print("=" * 70)
    print()
    print("Testing classical pendulum for MAML validation...")
    print()

    run_tests()
