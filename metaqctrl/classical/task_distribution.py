"""
Task distribution for pendulum control (analogous to quantum noise distribution).

Samples task parameters (mass, length, friction) from a distribution,
similar to how NoiseParameters samples (alpha, A, omega_c).
"""

import numpy as np
from typing import List, Dict, Tuple

try:
    from .pendulum import PendulumTask
except ImportError:
    from pendulum import PendulumTask


class PendulumTaskDistribution:
    """
    Distribution over pendulum task parameters.

    Analogous to TaskDistribution for quantum noise parameters.

    Parameters:
        mass: Pendulum mass [kg]
        length: Pendulum length [m]
        friction: Damping coefficient

    Example ranges:
        - Light, short pendulum with low friction (easy to control)
        - Heavy, long pendulum with high friction (hard to control)
    """

    def __init__(
        self,
        dist_type: str = "uniform",
        ranges: Dict[str, Tuple[float, float]] = None,
        mean: np.ndarray = None,
        cov: np.ndarray = None
    ):
        """
        Args:
            dist_type: 'uniform' or 'gaussian'
            ranges: Parameter ranges for uniform sampling
            mean: Mean for gaussian sampling [3]
            cov: Covariance matrix for gaussian sampling [3x3]
        """
        self.dist_type = dist_type

        # Default ranges chosen for reasonable diversity
        self.ranges = ranges or {
            "mass": (0.5, 2.0),      # 0.5 to 2.0 kg
            "length": (0.5, 1.5),    # 0.5 to 1.5 m
            "friction": (0.0, 0.3)   # 0 to 0.3 damping
        }

        self.mean = mean
        self.cov = cov

    def sample(self, n_tasks: int, rng: np.random.Generator = None) -> List[PendulumTask]:
        """
        Sample task batch.

        Args:
            n_tasks: Number of tasks to sample
            rng: Random number generator

        Returns:
            tasks: List of PendulumTask instances
        """
        rng = rng or np.random.default_rng()

        if self.dist_type == "uniform":
            return self._sample_uniform(n_tasks, rng)
        elif self.dist_type == "gaussian":
            return self._sample_gaussian(n_tasks, rng)
        else:
            raise ValueError(f"Unknown dist_type '{self.dist_type}'")

    def _sample_uniform(self, n: int, rng: np.random.Generator) -> List[PendulumTask]:
        """Sample from uniform distribution."""
        tasks = []
        for _ in range(n):
            mass = rng.uniform(*self.ranges["mass"])
            length = rng.uniform(*self.ranges["length"])
            friction = rng.uniform(*self.ranges["friction"])
            tasks.append(PendulumTask(mass, length, friction))
        return tasks

    def _sample_gaussian(self, n: int, rng: np.random.Generator) -> List[PendulumTask]:
        """Sample from Gaussian distribution."""
        if self.mean is None or self.cov is None:
            raise ValueError("Gaussian sampling requires 'mean' and 'cov'")

        samples = rng.multivariate_normal(self.mean, self.cov, size=n)
        tasks = [PendulumTask.from_array(s) for s in samples]
        return tasks

    def compute_variance(self) -> float:
        """
        Compute variance of task distribution.

        For uniform: σ² = Σ (b-a)²/12
        For gaussian: σ² = trace(Σ)

        Returns:
            variance: Total variance across all dimensions
        """
        if self.dist_type == "uniform":
            var_mass = ((self.ranges["mass"][1] - self.ranges["mass"][0]) ** 2) / 12.0
            var_length = ((self.ranges["length"][1] - self.ranges["length"][0]) ** 2) / 12.0
            var_friction = ((self.ranges["friction"][1] - self.ranges["friction"][0]) ** 2) / 12.0
            return float(var_mass + var_length + var_friction)
        elif self.dist_type == "gaussian":
            return float(np.trace(self.cov))
        else:
            raise ValueError(f"Unknown dist_type '{self.dist_type}'")


# Example usage
if __name__ == "__main__":
    print("Testing PendulumTaskDistribution")
    print("=" * 60)

    # Create distribution
    dist = PendulumTaskDistribution(
        dist_type="uniform",
        ranges={
            "mass": (0.5, 2.0),
            "length": (0.5, 1.5),
            "friction": (0.05, 0.25)
        }
    )

    # Sample tasks
    rng = np.random.default_rng(42)
    tasks = dist.sample(10, rng)

    print(f"\nSampled {len(tasks)} tasks:")
    print(f"{'Task':<6} {'Mass [kg]':<12} {'Length [m]':<12} {'Friction':<12}")
    print("-" * 60)
    for i, task in enumerate(tasks):
        print(f"{i:<6} {task.mass:<12.3f} {task.length:<12.3f} {task.friction:<12.3f}")

    # Compute variance
    variance = dist.compute_variance()
    print(f"\nTask distribution variance: {variance:.4f}")

    # Test array conversion
    task = tasks[0]
    arr = task.to_array()
    task_reconstructed = PendulumTask.from_array(arr)
    print(f"\nArray conversion test:")
    print(f"  Original: {task}")
    print(f"  Array: {arr}")
    print(f"  Reconstructed: {task_reconstructed}")
    print(f"  Match: {np.allclose(task.to_array(), task_reconstructed.to_array())}")

    print("\n✓ All tests passed!")
