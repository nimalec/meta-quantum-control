"""
Noise Models and PSD Parameterization

Tasks are parameterized by power spectral density (PSD) of noise:
S(ω; θ) where θ = (α, A, ωc) controls spectral shape.

This induces Lindblad operators L_j,θ via filter/correlation functions.
"""

import numpy as np
from scipy.special import gamma as gamma_func
from typing import Tuple, List, Dict
from dataclasses import dataclass


@dataclass
class NoiseParameters:
    """Task parameters defining a noise environment."""
    alpha: float  # Spectral exponent (1/f^α noise)
    A: float      # Amplitude/strength
    omega_c: float  # Cutoff frequency
    
    def to_array(self) -> np.ndarray:
        return np.array([self.alpha, self.A, self.omega_c])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'NoiseParameters':
        return cls(alpha=arr[0], A=arr[1], omega_c=arr[2])


class NoisePSDModel:
    """
    Power spectral density models for colored noise.
    
    Models available:
    - 1/f^α noise (pink/brown noise)
    - Lorentzian (Ornstein-Uhlenbeck)
    - Double-exponential
    """
    
    def __init__(self, model_type: str = '1/f'):
        """
        Args:
            model_type: 'one_over_f', 'lorentzian', 'double_exp'
        """
        self.model_type = model_type
    
    def psd(self, omega: np.ndarray, theta: NoiseParameters) -> np.ndarray:
        """
        Compute S(ω; θ).
        
        Args:
            omega: Frequency array (rad/s)
            theta: Noise parameters
            
        Returns:
            S: PSD values at each frequency
        """
        if self.model_type == 'one_over_f':
            return self._one_over_f_psd(omega, theta)
        elif self.model_type == 'lorentzian':
            return self._lorentzian_psd(omega, theta)
        elif self.model_type == 'double_exp':
            return self._double_exp_psd(omega, theta)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _one_over_f_psd(self, omega: np.ndarray, theta: NoiseParameters) -> np.ndarray:
        """
        1/f^α noise with cutoff:
        S(ω) = A / (|ω|^α + ωc^α)
        """
        return theta.A / (np.abs(omega)**theta.alpha + theta.omega_c**theta.alpha)
    
    def _lorentzian_psd(self, omega: np.ndarray, theta: NoiseParameters) -> np.ndarray:
        """
        Lorentzian (Ornstein-Uhlenbeck):
        S(ω) = A / (ω² + ωc²)
        """
        return theta.A / (omega**2 + theta.omega_c**2)
    
    def _double_exp_psd(self, omega: np.ndarray, theta: NoiseParameters) -> np.ndarray:
        """
        Sum of two Lorentzians (multi-scale noise):
        S(ω) = A₁/(ω² + ωc₁²) + A₂/(ω² + ωc₂²)
        
        Here we use α to interpolate between scales.
        """
        omega_c1 = theta.omega_c
        omega_c2 = theta.omega_c * (1 + theta.alpha)
        A1 = theta.A * (1 - theta.alpha / 5)
        A2 = theta.A * (theta.alpha / 5)
        return A1 / (omega**2 + omega_c1**2) + A2 / (omega**2 + omega_c2**2)
    
    def correlation_function(self, tau: np.ndarray, theta: NoiseParameters) -> np.ndarray:
        """
        Compute correlation function C(τ) = ∫ S(ω) e^{iωτ} dω via inverse Fourier.
        
        For Lorentzian: C(τ) = (A/2ωc) exp(-ωc|τ|)
        For 1/f: numerical integration required
        """
        if self.model_type == 'lorentzian':
            return (theta.A / (2 * theta.omega_c)) * np.exp(-theta.omega_c * np.abs(tau))
        else:
            # Numerical inverse Fourier transform
            omega = np.linspace(-100, 100, 10000)
            S_omega = self.psd(omega, theta)
            C_tau = np.trapz(S_omega[:, None] * np.exp(1j * omega[:, None] * tau), omega, axis=0)
            return C_tau.real / (2 * np.pi)


class PSDToLindblad:
    """
    Convert PSD parameters to Lindblad operators.
    
    Approaches:
    1. Phenomenological: L_j,θ = sqrt(Γ_j(θ)) * σ_j where Γ_j ∝ S(ω_j)
    2. Spectral decomposition: Sample PSD at control-relevant frequencies
    3. Filter-based: Design filter with frequency response matching PSD
    """
    
    def __init__(
        self,
        basis_operators: List[np.ndarray],
        sampling_freqs: np.ndarray,
        psd_model: NoisePSDModel
    ):
        """
        Args:
            basis_operators: Pauli operators [σx, σy, σz] or other basis
            sampling_freqs: Frequencies at which to sample PSD
            psd_model: PSD model instance
        """
        self.basis_ops = basis_operators
        self.sampling_freqs = sampling_freqs
        self.psd_model = psd_model
    
 of Lindblad operators [L₁, L₂, ...]
        """
        # Evaluate PSD at sampling frequencies
        S_values = self.psd_model.psd(self.sampling_freqs, theta)
        
        # Map to dissipation rates (phenomenological)
        # Γ_j ∝ ∫ S(ω) |H_j(ω)|² dω where H_j is filter response
        # For simplicity: Γ_j = S(ω_j) at characteristic frequency
        
        L_ops = []
        for j, sigma in enumerate(self.basis_ops):
            # Use PSD value at corresponding frequency
            freq_idx = min(j, len(self.sampling_freqs) - 1)
            gamma_j = S_values[freq_idx]
            
            # Lindblad operator: L_j = sqrt(Γ_j) * σ_j
            L_ops.append(np.sqrt(gamma_j) * sigma)
        
        return L_ops
    
    def get_effective_rates(self, theta: NoiseParameters) -> np.ndarray:
        """Get effective decay rates for each channel."""
        S_values = self.psd_model.psd(self.sampling_freqs, theta)
        return S_values


class TaskDistribution:
    """
    Distribution P over task parameters Θ.
    
    Supports:
    - Uniform over box
    - Gaussian
    - Mixture of Gaussians (multi-modal)
    """
    
    def __init__(
        self,
        dist_type: str = 'uniform',
        ranges: Dict[str, Tuple[float, float]] = None,
        mean: np.ndarray = None,
        cov: np.ndarray = None
    ):
        """
        Args:
            dist_type: 'uniform', 'gaussian', 'mixture'
            ranges: For uniform: {'alpha': (min, max), ...}
            mean: For Gaussian: mean parameter vector
            cov: For Gaussian: covariance matrix
        """
        self.dist_type = dist_type
        self.ranges = ranges or {
            'alpha': (0.5, 2.0),
            'A': (0.01, 0.5),
            'omega_c': (1.0, 10.0)
        }
        self.mean = mean
        self.cov = cov
    
    def sample(self, n_tasks: int, rng: np.random.Generator = None) -> List[NoiseParameters]:
        """
        Sample n tasks from P.
        
        Returns:
            tasks: List of NoiseParameters
        """
        if rng is None:
            rng = np.random.default_rng()
        
        if self.dist_type == 'uniform':
            return self._sample_uniform(n_tasks, rng)
        elif self.dist_type == 'gaussian':
            return self._sample_gaussian(n_tasks, rng)
        else:
            raise ValueError(f"Unknown distribution: {self.dist_type}")
    
    def _sample_uniform(self, n: int, rng: np.random.Generator) -> List[NoiseParameters]:
        """Sample uniformly from box."""
        tasks = []
        for _ in range(n):
            alpha = rng.uniform(*self.ranges['alpha'])
            A = rng.uniform(*self.ranges['A'])
            omega_c = rng.uniform(*self.ranges['omega_c'])
            tasks.append(NoiseParameters(alpha, A, omega_c))
        return tasks
    
    def _sample_gaussian(self, n: int, rng: np.random.Generator) -> List[NoiseParameters]:
        """Sample from Gaussian."""
        samples = rng.multivariate_normal(self.mean, self.cov, size=n)
        tasks = [NoiseParameters.from_array(s) for s in samples]
        return tasks
    
    def compute_variance(self) -> float:
        """Compute σ²_θ for theoretical bounds."""
        if self.dist_type == 'uniform':
            # Variance of uniform: (b-a)²/12 for each dimension
            var_alpha = ((self.ranges['alpha'][1] - self.ranges['alpha'][0])**2) / 12
            var_A = ((self.ranges['A'][1] - self.ranges['A'][0])**2) / 12
            var_omega = ((self.ranges['omega_c'][1] - self.ranges['omega_c'][0])**2) / 12
            return var_alpha + var_A + var_omega
        elif self.dist_type == 'gaussian':
            return np.trace(self.cov)
        else:
            return 0.0


def psd_distance(theta1: NoiseParameters, theta2: NoiseParameters, omega_grid: np.ndarray) -> float:
    """
    Compute distance d_Θ(θ, θ') = sup_ω |S(ω; θ) - S(ω; θ')|.
    
    Args:
        theta1, theta2: Noise parameters
        omega_grid: Frequency grid for supremum
        
    Returns:
        dist: Supremum distance
    """
    psd_model = NoisePSDModel()
    S1 = psd_model.psd(omega_grid, theta1)
    S2 = psd_model.psd(omega_grid, theta2)
    return np.max(np.abs(S1 - S2))


# Example usage
if __name__ == "__main__":
    # Define task distribution
    task_dist = TaskDistribution(
        dist_type='uniform',
        ranges={
            'alpha': (0.5, 2.0),
            'A': (0.05, 0.3),
            'omega_c': (2.0, 8.0)
        }
    )
    
    # Sample tasks
    rng = np.random.default_rng(42)
    tasks = task_dist.sample(5, rng)
    
    print("Sampled tasks:")
    for i, task in enumerate(tasks):
        print(f"Task {i}: α={task.alpha:.2f}, A={task.A:.3f}, ωc={task.omega_c:.2f}")
    
    # Visualize PSDs
    psd_model = NoisePSDModel(model_type='one_over_f')
    omega = np.logspace(-1, 2, 1000)
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    for i, task in enumerate(tasks):
        S = psd_model.psd(omega, task)
        plt.loglog(omega, S, label=f'Task {i}')
    plt.xlabel('Frequency ω (rad/s)')
    plt.ylabel('PSD S(ω)')
    plt.title('Power Spectral Densities of Sampled Tasks')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('psd_samples.png', dpi=150, bbox_inches='tight')
    print("\nPSD plot saved to psd_samples.png")
    
    # Compute pairwise distances
    print("\nPairwise PSD distances:")
    omega_grid = np.logspace(-1, 2, 500)
    for i in range(len(tasks)):
        for j in range(i+1, len(tasks)):
            dist = psd_distance(tasks[i], tasks[j], omega_grid)
            print(f"  d(task{i}, task{j}) = {dist:.4f}")
    
    # Task distribution variance
    print(f"\nTask distribution variance σ²_θ = {task_dist.compute_variance():.4f}")