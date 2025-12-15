"""
Gamma-Rate Noise Models for Quantum Control
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass


# ---------- Gamma Noise Parameters ----------
@dataclass
class GammaNoiseParameters:
    """
    Markovian noise parameters for quantum systems.

    Attributes:
        gamma_deph: Pure dephasing rate [1/s] (γ_φ in Lindblad equation)
        gamma_relax: Relaxation rate [1/s] (Γ_↓ in Lindblad equation, T1 = 1/gamma_relax)

    Physical interpretation:
        - gamma_deph controls T2* decay (coherence loss)
        - gamma_relax controls T1 decay (energy relaxation)
        - Total dephasing: gamma_total = gamma_deph + gamma_relax/2

    Normalization ranges (for neural network input):
        - gamma_deph: [0.01, 0.2] → normalized by /0.1
        - gamma_relax: [0.005, 0.1] → normalized by /0.05
    """
    gamma_deph: float   # Pure dephasing rate [1/s]
    gamma_relax: float  # Relaxation rate [1/s]

    # Default normalization scales (typical experimental ranges)
    GAMMA_DEPH_SCALE = 0.1    # Normalize by this value
    GAMMA_RELAX_SCALE = 0.05  # Normalize by this value
    GAMMA_SUM_SCALE = 0.15    # For sum feature

    # Typical ranges
    GAMMA_DEPH_RANGE = (0.01, 0.2)
    GAMMA_RELAX_RANGE = (0.005, 0.1)

    def to_array(self, normalized: bool = True) -> np.ndarray:
        """
        Convert to array representation for neural network input.

        Args:
            normalized: If True (default), normalize features to ~[0,1] range.
                       This is the standard format for gamma-trained policies.

        Returns:
            arr: Array [gamma_deph/0.1, gamma_relax/0.05, sum/0.15]
                 3D features matching gamma-trained policy input dimension.
        """
        if normalized:
            return np.array([
                self.gamma_deph / self.GAMMA_DEPH_SCALE,
                self.gamma_relax / self.GAMMA_RELAX_SCALE,
                (self.gamma_deph + self.gamma_relax) / self.GAMMA_SUM_SCALE
            ], dtype=float)
        else:
            return np.array([
                self.gamma_deph,
                self.gamma_relax,
                self.gamma_deph + self.gamma_relax
            ], dtype=float)

    @classmethod
    def from_array(cls, arr: np.ndarray, normalized: bool = True) -> "GammaNoiseParameters":
        """
        Create from array representation.

        Args:
            arr: Array of parameters (at least 2 elements)
            normalized: If True, arr contains normalized values that need to be denormalized.

        Returns:
            params: GammaNoiseParameters instance
        """
        if normalized:
            gamma_deph = float(arr[0]) * cls.GAMMA_DEPH_SCALE
            gamma_relax = float(arr[1]) * cls.GAMMA_RELAX_SCALE
        else:
            gamma_deph = float(arr[0])
            gamma_relax = float(arr[1])

        return cls(gamma_deph=gamma_deph, gamma_relax=gamma_relax)

    def get_T1(self) -> float:
        """Get T1 relaxation time [s]."""
        return 1.0 / self.gamma_relax if self.gamma_relax > 0 else float('inf')

    def get_T2_star(self) -> float:
        """Get T2* dephasing time [s] (includes relaxation contribution)."""
        gamma_total = self.gamma_deph + self.gamma_relax / 2.0
        return 1.0 / gamma_total if gamma_total > 0 else float('inf')

    def get_T2(self) -> float:
        """Get pure T2 dephasing time [s] (without relaxation contribution)."""
        return 1.0 / self.gamma_deph if self.gamma_deph > 0 else float('inf')

    def __repr__(self) -> str:
        return f"GammaNoiseParameters(γ_deph={self.gamma_deph:.4f}, γ_relax={self.gamma_relax:.4f})"


# ---------- Gamma Task Distribution ----------
class GammaTaskDistribution:
    """
    Distribution P over gamma noise parameters.

    Samples (gamma_deph, gamma_relax) pairs for meta-learning experiments.

    Distribution types:
        - 'uniform': Uniform over specified ranges
        - 'log_uniform': Log-uniform (better for parameters spanning orders of magnitude)
        - 'gaussian': Gaussian with specified mean and covariance

    Example:
        dist = GammaTaskDistribution(
            dist_type='uniform',
            gamma_deph_range=(0.02, 0.15),
            gamma_relax_range=(0.01, 0.08)
        )
        tasks = dist.sample(n_tasks=10)
    """

    def __init__(
        self,
        dist_type: str = "uniform",
        gamma_deph_range: Tuple[float, float] = (0.02, 0.15),
        gamma_relax_range: Tuple[float, float] = (0.01, 0.08),
        center_deph: Optional[float] = None,
        center_relax: Optional[float] = None,
        diversity_scale: float = 1.0,
        mean: Optional[np.ndarray] = None,
        cov: Optional[np.ndarray] = None
    ):
        """
        Args:
            dist_type: 'uniform', 'log_uniform', or 'gaussian'
            gamma_deph_range: (min, max) for dephasing rate sampling
            gamma_relax_range: (min, max) for relaxation rate sampling
            center_deph: Center of distribution for dephasing (for diversity experiments)
            center_relax: Center of distribution for relaxation (for diversity experiments)
            diversity_scale: Scale factor for distribution width (1.0 = full range)
            mean: Mean for Gaussian sampling [gamma_deph, gamma_relax]
            cov: Covariance matrix for Gaussian sampling (2x2)
        """
        self.dist_type = dist_type
        self.gamma_deph_range = gamma_deph_range
        self.gamma_relax_range = gamma_relax_range
        self.diversity_scale = diversity_scale
        self.mean = mean
        self.cov = cov

        # Compute effective ranges based on diversity
        if center_deph is not None and center_relax is not None:
            # Use centered distribution with diversity scaling
            deph_width = (gamma_deph_range[1] - gamma_deph_range[0]) * diversity_scale / 2.0
            relax_width = (gamma_relax_range[1] - gamma_relax_range[0]) * diversity_scale / 2.0

            self.effective_deph_range = (
                max(gamma_deph_range[0], center_deph - deph_width),
                min(gamma_deph_range[1], center_deph + deph_width)
            )
            self.effective_relax_range = (
                max(gamma_relax_range[0], center_relax - relax_width),
                min(gamma_relax_range[1], center_relax + relax_width)
            )
        else:
            self.effective_deph_range = gamma_deph_range
            self.effective_relax_range = gamma_relax_range

    def sample(self, n_tasks: int, rng: Optional[np.random.Generator] = None) -> List[GammaNoiseParameters]:
        """
        Sample n_tasks gamma parameter sets.

        Args:
            n_tasks: Number of tasks to sample
            rng: Random number generator (optional)

        Returns:
            tasks: List of GammaNoiseParameters instances
        """
        rng = rng or np.random.default_rng()

        if self.dist_type == "uniform":
            return self._sample_uniform(n_tasks, rng)
        elif self.dist_type == "log_uniform":
            return self._sample_log_uniform(n_tasks, rng)
        elif self.dist_type == "gaussian":
            return self._sample_gaussian(n_tasks, rng)
        else:
            raise ValueError(f"Unknown distribution type '{self.dist_type}'")

    def _sample_uniform(self, n: int, rng: np.random.Generator) -> List[GammaNoiseParameters]:
        """Sample uniformly from ranges."""
        tasks = []
        for _ in range(n):
            gamma_deph = rng.uniform(*self.effective_deph_range)
            gamma_relax = rng.uniform(*self.effective_relax_range)
            tasks.append(GammaNoiseParameters(gamma_deph, gamma_relax))
        return tasks

    def _sample_log_uniform(self, n: int, rng: np.random.Generator) -> List[GammaNoiseParameters]:
        """Sample log-uniformly (better for rates spanning orders of magnitude)."""
        tasks = []
        for _ in range(n):
            log_deph = rng.uniform(
                np.log10(self.effective_deph_range[0]),
                np.log10(self.effective_deph_range[1])
            )
            log_relax = rng.uniform(
                np.log10(self.effective_relax_range[0]),
                np.log10(self.effective_relax_range[1])
            )
            tasks.append(GammaNoiseParameters(10**log_deph, 10**log_relax))
        return tasks

    def _sample_gaussian(self, n: int, rng: np.random.Generator) -> List[GammaNoiseParameters]:
        """Sample from Gaussian distribution."""
        if self.mean is None or self.cov is None:
            raise ValueError("Gaussian sampling requires 'mean' and 'cov'")

        samples = rng.multivariate_normal(self.mean, self.cov, size=n)

        tasks = []
        for sample in samples:
            # Clip to valid ranges
            gamma_deph = np.clip(sample[0], *self.effective_deph_range)
            gamma_relax = np.clip(sample[1], *self.effective_relax_range)
            tasks.append(GammaNoiseParameters(gamma_deph, gamma_relax))

        return tasks

    def compute_variance(self) -> float:
        """
        Compute variance of task distribution (sum of per-dimension variances).

        Returns:
            variance: Total variance (useful for theory bounds)
        """
        if self.dist_type in ["uniform", "log_uniform"]:
            # Uniform variance: (b-a)^2 / 12
            var_deph = ((self.effective_deph_range[1] - self.effective_deph_range[0]) ** 2) / 12.0
            var_relax = ((self.effective_relax_range[1] - self.effective_relax_range[0]) ** 2) / 12.0
            return float(var_deph + var_relax)
        elif self.dist_type == "gaussian":
            return float(np.trace(self.cov))
        else:
            return 0.0

    def get_center(self) -> Tuple[float, float]:
        """Get center of the distribution."""
        center_deph = (self.effective_deph_range[0] + self.effective_deph_range[1]) / 2.0
        center_relax = (self.effective_relax_range[0] + self.effective_relax_range[1]) / 2.0
        return center_deph, center_relax


# ---------- Gamma to Lindblad Operators ----------
def gamma_to_lindblad_operators(
    gamma_deph: float,
    gamma_relax: float
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Convert gamma rates directly to Lindblad jump operators.

    This is the Markovian limit - no PSD integration needed.

    Args:
        gamma_deph: Pure dephasing rate [1/s]
        gamma_relax: Relaxation rate [1/s]

    Returns:
        L_ops: List of Lindblad operators [√γ_relax σ-, √γ_deph σ_z/√2]
        rates: Array [gamma_relax, gamma_deph]

    Physical meaning:
        L_1 = √γ_relax |g⟩⟨e| (relaxation/decay)
        L_2 = √γ_deph σ_z/√2 (pure dephasing)

    Lindblad equation:
        dρ/dt = -i[H, ρ] + Σ_k (L_k ρ L_k† - 1/2 {L_k† L_k, ρ})
    """
    # Relaxation operator: |g⟩⟨e| = σ-
    L_relax = np.sqrt(gamma_relax) * np.array([[0, 1], [0, 0]], dtype=complex)

    # Dephasing operator: σ_z / √2 (normalized)
    L_deph = np.sqrt(gamma_deph / 2.0) * np.array([[1, 0], [0, -1]], dtype=complex)

    L_ops = [L_relax, L_deph]
    rates = np.array([gamma_relax, gamma_deph], dtype=float)

    return L_ops, rates


# ---------- Conversion utilities ----------
def psd_to_gamma_approximate(alpha: float, A: float, omega_c: float, T: float = 1.0) -> Tuple[float, float]:
    """
    Approximate conversion from PSD parameters to gamma rates.

    This is a rough approximation for reference - the actual relationship
    depends on the filter function and integration details.

    For 1/f^alpha noise:
        gamma_deph ~ A * T^(alpha-1) / omega_c^alpha (rough scaling)
        gamma_relax ~ A / omega_c^alpha (depends on omega_0)

    Args:
        alpha: PSD spectral exponent
        A: PSD amplitude
        omega_c: PSD cutoff frequency
        T: Gate time [s]

    Returns:
        gamma_deph: Approximate dephasing rate
        gamma_relax: Approximate relaxation rate

    Note: This is for reference only. For accurate results, use the
    full PSD-to-Lindblad conversion in noise_models_v2.py.
    """
    # Very rough approximation
    gamma_deph = A * 0.0078  # Empirical scaling factor
    gamma_relax = gamma_deph * 0.5  # Typical ratio
    return gamma_deph, gamma_relax


# ---------- Example usage ----------
if __name__ == "__main__":
    print("=" * 60)
    print("Gamma Noise Models - Example Usage")
    print("=" * 60)

    # 1) Create task distribution
    task_dist = GammaTaskDistribution(
        dist_type="uniform",
        gamma_deph_range=(0.02, 0.15),
        gamma_relax_range=(0.01, 0.08)
    )

    # 2) Sample tasks
    rng = np.random.default_rng(42)
    tasks = task_dist.sample(5, rng)

    print("\nSampled tasks:")
    for i, t in enumerate(tasks):
        arr = t.to_array(normalized=True)
        print(f"  Task {i}: γ_deph={t.gamma_deph:.4f}, γ_relax={t.gamma_relax:.4f}")
        print(f"           T1={t.get_T1():.2f}s, T2*={t.get_T2_star():.2f}s")
        print(f"           Features (normalized): {arr}")

    # 3) Get Lindblad operators
    print("\nLindblad operators for Task 0:")
    L_ops, rates = gamma_to_lindblad_operators(tasks[0].gamma_deph, tasks[0].gamma_relax)
    print(f"  Rates: γ_relax={rates[0]:.4f}, γ_deph={rates[1]:.4f}")
    print(f"  L_relax shape: {L_ops[0].shape}")
    print(f"  L_deph shape: {L_ops[1].shape}")

    # 4) Variance for theory bounds
    print(f"\nDistribution variance: {task_dist.compute_variance():.6f}")
