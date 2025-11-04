"""
Noise Models and PSD Parameterization

Tasks are parameterized by power spectral density (PSD) of noise:
S(ω; θ) where θ = (α, A, ωc) controls spectral shape.

This induces Lindblad operators L_j,θ via filter/correlation functions.

IMPORTANT IMPLEMENTATION NOTES:
-------------------------------

Physically Correct PSD to Lindblad Conversion:
The decay rate for each noise channel must be computed by integrating the PSD
over the control-relevant frequency band:

    Γ_j = ∫ S(ω; θ) |χ_j(ω)|² dω

where:
- S(ω; θ) is the noise power spectral density
- χ_j(ω) is the filter/susceptibility function for channel j
- The integral is taken over the control bandwidth [0, ω_max]

This replaces the previous simplified approach of evaluating S at a single point.

The filter function χ_j(ω) represents how sensitive the quantum system is to
noise at frequency ω on channel j. For typical quantum control:
- Use 'uniform' filter: boxcar over control bandwidth (default, most physical)
- Use 'lorentzian' filter: for systems with resonant coupling
- Use 'gaussian' filter: for smooth frequency response

Units and Normalization:
- PSD S(ω): power per unit frequency [dimensionless or power/Hz]
- Integrated rate Γ_j: decay rate per unit time [Hz or rad/s]
- Lindblad operators L_j = √Γ_j * σ_j: [√Hz] * [dimensionless]
- Properly normalized for Lindblad equation: dρ/dt = Σ_j (L_j ρ L_j† - ½{L_j†L_j, ρ})

For backward compatibility, set integration_method='point' to use old single-point
evaluation (not recommended for physical accuracy).
"""

import numpy as np
from scipy.special import gamma as gamma_func
from typing import Tuple, List, Dict
from dataclasses import dataclass


@dataclass
class NoiseParameters:
    """Good. 
    Task parameters defining a noise environment."""
    alpha: float  # Spectral exponent (1/f^α noise)
    A: float      # Amplitude/strength
    omega_c: float  # Cutoff frequency
    
    def to_array(self) -> np.ndarray:
        return np.array([self.alpha, self.A, self.omega_c])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'NoiseParameters':
        return cls(alpha=arr[0], A=arr[1], omega_c=arr[2])


class NoisePSDModel:
    """Good. 
    Power spectral density models for colored noise.
    
    Models available:
    - 1/f^α noise (pink/brown noise)
    - Lorentzian (Ornstein-Uhlenbeck)
    - Double-exponential
    """
    
    def __init__(self, model_type: str = 'one_over_f'):
        """
        Args:
            model_type: 'one_over_f', 'lorentzian', 'double_exp'
        """
        self.model_type = model_type
    
    def psd(self, omega: np.ndarray, theta: NoiseParameters) -> np.ndarray:
        """Good. 
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
        """Good. 
        1/f^α noise with cutoff:
        S(ω) = A / (|ω|^α + ωc^α)
        """
        # Add small epsilon to prevent division by zero when omega=0 and omega_c=0
        epsilon = 1e-12
        omega_term = np.abs(omega)**theta.alpha if theta.alpha > 0 else np.ones_like(omega)
        cutoff_term = theta.omega_c**theta.alpha if theta.alpha > 0 else 1.0
        return theta.A / (omega_term + cutoff_term + epsilon)
    
    def _lorentzian_psd(self, omega: np.ndarray, theta: NoiseParameters) -> np.ndarray:
        """Good. 
        Lorentzian (Ornstein-Uhlenbeck):
        S(ω) = A / (ω² + ωc²)
        """
        # Add small epsilon to prevent numerical issues
        epsilon = 1e-12
        return theta.A / (omega**2 + theta.omega_c**2 + epsilon)
    
    def _double_exp_psd(self, omega: np.ndarray, theta: NoiseParameters) -> np.ndarray:
        """Good. 
        Sum of two Lorentzians (multi-scale noise):
        S(ω) = A₁/(ω² + ωc₁²) + A₂/(ω² + ωc₂²)

        Here we use α to interpolate between scales.
        """
        epsilon = 1e-12
        omega_c1 = theta.omega_c
        omega_c2 = theta.omega_c * (1 + theta.alpha)
        A1 = theta.A * (1 - theta.alpha / 5)
        A2 = theta.A * (theta.alpha / 5)
        return A1 / (omega**2 + omega_c1**2 + epsilon) + A2 / (omega**2 + omega_c2**2 + epsilon)
    
    def correlation_function(self, tau: np.ndarray, theta: NoiseParameters) -> np.ndarray:
        """Good. 
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
    """Good.
    Convert PSD parameters to Lindblad operators.

    Approaches:
    1. Phenomenological: L_j,θ = sqrt(Γ_j(θ)) * σ_j where Γ_j ∝ S(ω_j)
    2. Spectral decomposition: Sample PSD at control-relevant frequencies
    3. Filter-based: Design filter with frequency response matching PSD

    Physically correct conversion integrates PSD over control bandwidth:
    Γ_j = ∫ S(ω) |χ_j(ω)|² dω
    where χ_j(ω) is the susceptibility/filter function for channel j.
    """

    def __init__(
        self,
        basis_operators: List[np.ndarray],
        sampling_freqs: np.ndarray,
        psd_model: NoisePSDModel,
        integration_method: str = 'trapz',
        omega_integration_range: Tuple[float, float] = None,
        n_integration_points: int = 500
        ):
        """
        Args:
            basis_operators: Pauli operators [σx, σy, σz] or other basis
            sampling_freqs: Frequencies defining control bandwidth (for filter function)
            psd_model: PSD model instance
            integration_method: 'trapz' (trapezoidal), 'simpson', or 'point' (old behavior)
            omega_integration_range: (omega_min, omega_max) for integration.
                                     If None, uses (0, max(sampling_freqs))
            n_integration_points: Number of frequency points for numerical integration
        """
        self.basis_ops = basis_operators
        self.sampling_freqs = sampling_freqs
        self.psd_model = psd_model
        self.integration_method = integration_method
        self.n_integration_points = n_integration_points

        # Set integration range based on control bandwidth
        if omega_integration_range is None:
            omega_max = np.max(sampling_freqs) if len(sampling_freqs) > 0 else 20.0
            # Extend slightly beyond Nyquist to capture tail
            self.omega_integration_range = (0.0, 2.0 * omega_max)
        else:
            self.omega_integration_range = omega_integration_range
    
    def _filter_function(self, omega: np.ndarray, channel_idx: int) -> np.ndarray:
        """
        Compute filter function |χ_j(ω)|² for channel j.

        Options:
        1. 'uniform': Boxcar filter over control bandwidth (default, physically motivated)
        2. 'lorentzian': Lorentzian response (for specific resonant coupling)
        3. 'custom': User-provided filter function

        For most quantum control applications, 'uniform' is appropriate as it represents
        uniform susceptibility to noise across the control bandwidth.

        Args:
            omega: Frequency array (rad/s)
            channel_idx: Index of noise channel

        Returns:
            chi_squared: |χ_j(ω)|² filter function values
        """
        filter_type = getattr(self, 'filter_type', 'uniform')
        omega_max = np.max(self.sampling_freqs) if len(self.sampling_freqs) > 0 else 20.0

        if filter_type == 'uniform':
            # Boxcar filter: uniform susceptibility in control band
            chi_squared = np.where(
                (omega >= 0) & (omega <= omega_max),
                1.0,  # Constant susceptibility in band
                0.0   # No response outside band
            )

        elif filter_type == 'lorentzian':
            # Lorentzian filter centered at control bandwidth
            # Useful for resonant noise coupling
            omega_center = omega_max / 2
            width = omega_max / 4
            chi_squared = width**2 / ((omega - omega_center)**2 + width**2)

        elif filter_type == 'gaussian':
            # Gaussian filter centered at control bandwidth
            omega_center = omega_max / 2
            sigma = omega_max / 4
            chi_squared = np.exp(-((omega - omega_center)**2) / (2 * sigma**2))

        else:
            # Default to uniform
            chi_squared = np.where(
                (omega >= 0) & (omega <= omega_max),
                1.0,
                0.0
            )

        return chi_squared

    def get_lindblad_operators(self, theta: NoiseParameters):
        """Good.
        Get Lindblad operators for given task parameters.

        Physically correct mapping to dissipation rates:
        Γ_j = ∫ S(ω; θ) |χ_j(ω)|² dω
        where χ_j(ω) is the filter/susceptibility function for channel j.

        For backward compatibility, set integration_method='point' to use old behavior.

        Args:
            theta: NoiseParameters with (alpha, A, omega_c)

        Returns:
            L_ops: List of Lindblad operators [L_1, L_2, ...]
        """
        if self.integration_method == 'point':
            # Old behavior: single-point evaluation (for backward compatibility)
            S_values = self.psd_model.psd(self.sampling_freqs, theta)
            L_ops = []
            for j, sigma in enumerate(self.basis_ops):
                freq_idx = min(j, len(self.sampling_freqs) - 1)
                gamma_j = S_values[freq_idx]
                L_ops.append(np.sqrt(gamma_j) * sigma)
            return L_ops

        # New behavior: proper frequency integration
        omega_min, omega_max = self.omega_integration_range
        omega_grid = np.linspace(omega_min, omega_max, self.n_integration_points)

        # Evaluate PSD over full frequency range
        S_omega = self.psd_model.psd(omega_grid, theta)

        L_ops = []
        for j, sigma in enumerate(self.basis_ops):
            # Compute filter function for this channel
            chi_squared = self._filter_function(omega_grid, j)

            # Integrate: Γ_j = ∫ S(ω) |χ_j(ω)|² dω
            integrand = S_omega * chi_squared

            if self.integration_method == 'trapz':
                gamma_j = np.trapz(integrand, omega_grid)
            elif self.integration_method == 'simpson':
                from scipy.integrate import simpson
                gamma_j = simpson(integrand, x=omega_grid)
            else:
                raise ValueError(f"Unknown integration method: {self.integration_method}")

            # Ensure non-negative rate (numerical errors can cause small negative values)
            gamma_j = max(gamma_j, 0.0)

            # Lindblad operator: L_j = sqrt(Γ_j / Δω) * σ_j
            # Note: normalization by Δω = omega_max - omega_min converts
            # integrated PSD to effective rate per unit time
            bandwidth = omega_max - omega_min
            if bandwidth > 0:
                gamma_j_normalized = gamma_j / bandwidth
            else:
                gamma_j_normalized = gamma_j

            L_ops.append(np.sqrt(gamma_j_normalized) * sigma)

        return L_ops
    
    def get_effective_rates(self, theta: NoiseParameters) -> np.ndarray:
        """
        Get effective decay rates for each channel.

        Returns:
            rates: Array of Γ_j values for each channel
        """
        if self.integration_method == 'point':
            # Old behavior: point evaluation
            S_values = self.psd_model.psd(self.sampling_freqs, theta)
            return S_values

        # New behavior: integrated rates
        omega_min, omega_max = self.omega_integration_range
        omega_grid = np.linspace(omega_min, omega_max, self.n_integration_points)
        S_omega = self.psd_model.psd(omega_grid, theta)

        rates = []
        for j in range(len(self.basis_ops)):
            chi_squared = self._filter_function(omega_grid, j)
            integrand = S_omega * chi_squared

            if self.integration_method == 'trapz':
                gamma_j = np.trapz(integrand, omega_grid)
            elif self.integration_method == 'simpson':
                from scipy.integrate import simpson
                gamma_j = simpson(integrand, x=omega_grid)
            else:
                gamma_j = 0.0

            bandwidth = omega_max - omega_min 
            gamma_j_normalized = gamma_j / bandwidth if bandwidth > 0 else gamma_j
            gamma_j_normalized  = gamma_j
            rates.append(max(gamma_j_normalized, 0.0))

        return np.array(rates)


class TaskDistribution:
    """good. 
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
        """Good. 
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
        """Good. 
        Compute σ²_θ for theoretical bounds."""
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
    """Good. 
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
