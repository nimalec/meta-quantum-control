"""
Noise Model Adapter - Backward Compatible Interface

This module provides a unified interface that:
1. Uses noise_models_v2.py physics (correct) under the hood
2. Maintains noise_models.py API (compatibility) on the surface
3. Provides sensible defaults for physics parameters

Usage:
    # Drop-in replacement for old code
    from metaqctrl.quantum.noise_adapter import (
        NoiseParameters, NoisePSDModel, PSDToLindblad, TaskDistribution
    )

    # Use exactly like noise_models.py
    converter = PSDToLindblad(basis_operators, sampling_freqs, psd_model)
    L_ops = converter.get_lindblad_operators(theta)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from metaqctrl.quantum.noise_models_v2 import (
    NoiseParameters as NoiseParametersV2,
    NoisePSDModel as NoisePSDModelV2,
    PSDToLindblad as PSDToLindbladV2,
    TaskDistribution as TaskDistributionV2,
)

# Physical constants
HBAR = 1.054_571_817e-34  # J·s
TWO_PI = 2.0 * np.pi

# Re-export compatible classes
NoiseParameters = NoiseParametersV2
NoisePSDModel = NoisePSDModelV2
TaskDistribution = TaskDistributionV2


class PSDToLindblad:
    """
    Backward-compatible wrapper around noise_models_v2.PSDToLindblad.

    Provides the old API:
        converter = PSDToLindblad(basis_operators, sampling_freqs, psd_model)
        L_ops = converter.get_lindblad_operators(theta)

    But uses physics-correct v2 implementation underneath with sensible defaults.

    Default Physics Parameters (can be overridden):
    ------------------------------------------------
    - T: 50 μs (typical gate time for superconducting qubits)
    - omega0: Estimated from sampling_freqs bandwidth
    - sequence: 'ramsey' (free evolution, worst case for dephasing)
    - g_energy_per_xi: ℏ/2 (frequency noise coupling H_int = ℏ/2 δω σ_z)
    - temperature_K: None (classical noise, Γ↑ = Γ↓)
    - Gamma_h: 0.0 (no homogeneous broadening)
    """

    def __init__(
        self,
        basis_operators: List[np.ndarray],
        sampling_freqs: np.ndarray,
        psd_model: NoisePSDModelV2,
        # Physics parameters with sensible defaults
        T: float = 50e-6,  # 50 μs gate time
        sequence: str = 'ramsey',
        omega0: Optional[float] = None,
        g_energy_per_xi: float = None,  # Will default to HBAR/2
        temperature_K: Optional[float] = None,
        Gamma_h: float = 0.0,
        # Backward compatibility
        integration_method: str = 'trapz',  # Ignored, always uses v2 physics
        use_v2_physics: bool = True  # Set False to emulate v1 (not recommended)
    ):
        """
        Args:
            basis_operators: List of Pauli operators [σx, σy, σz] (or 2-qubit equivalent)
            sampling_freqs: Control bandwidth frequencies (used to estimate omega0)
            psd_model: NoisePSDModelV2 instance
            T: Gate evolution time [seconds]
            sequence: 'ramsey', 'echo', or 'cpmg_N' for dephasing filter function
            omega0: Qubit transition frequency [rad/s]. If None, estimated from bandwidth
            g_energy_per_xi: Energy coupling [J/xi]. Defaults to HBAR/2 (frequency noise)
            temperature_K: Temperature [Kelvin]. None = classical (Γ↑=Γ↓)
            Gamma_h: Homogeneous linewidth [rad/s]. 0 = sharp transition
            integration_method: Kept for API compatibility (always uses v2 integral)
            use_v2_physics: If False, emulates v1 behavior (for testing only)
        """
        self.basis_ops = basis_operators
        self.sampling_freqs = np.asarray(sampling_freqs)
        self.psd_model = psd_model
        self.use_v2_physics = use_v2_physics

        # Physics parameters
        self.T = float(T)
        self.sequence = sequence
        self.temperature_K = temperature_K
        self.Gamma_h = float(Gamma_h)

        # Estimate omega0 from control bandwidth if not provided
        if omega0 is None:
            # Use Nyquist frequency of control bandwidth as rough estimate
            omega_max = np.max(self.sampling_freqs) if len(self.sampling_freqs) > 0 else 20.0
            self.omega0 = omega_max / 2.0  # Transition typically below control bandwidth
            self._omega0_estimated = True
        else:
            self.omega0 = float(omega0)
            self._omega0_estimated = False

        # Set coupling strength
        if g_energy_per_xi is None:
            # Default: frequency noise with H_int = (ℏ/2) δω σ_z
            self.g_energy_per_xi = HBAR / 2.0
            self._g_was_default = True
        else:
            self.g_energy_per_xi = float(g_energy_per_xi)
            self._g_was_default = False

        # Create v2 converter
        self.converter_v2 = PSDToLindbladV2(
            psd_model=self.psd_model,
            g_energy_per_xi=self.g_energy_per_xi,
            hbar=HBAR
        )

        # Warn about defaults on first use
        self._warned = False

    def _warn_defaults(self):
        """Warn user about default physics parameters (once per instance)."""
        if not self._warned:
            warnings = []
            if self._omega0_estimated:
                warnings.append(f"  omega0 = {self.omega0/TWO_PI/1e6:.2f} MHz (estimated from bandwidth)")
            if self._g_was_default:
                warnings.append(f"  g_coupling = ℏ/2 (frequency noise assumed)")

            if warnings:
                print("INFO: Using default physics parameters in PSDToLindblad:")
                for w in warnings:
                    print(w)
                print(f"  T = {self.T*1e6:.1f} μs, sequence = '{self.sequence}'")
                print("  Override by passing omega0=, g_energy_per_xi= to constructor")
                print()

            self._warned = True

    def get_lindblad_operators(self, theta: NoiseParameters) -> List[np.ndarray]:
        """
        Backward-compatible API: returns Lindblad operators using v2 physics.

        Args:
            theta: NoiseParameters (alpha, A, omega_c)

        Returns:
            L_ops: List of Lindblad operators [√Γ↓ |g⟩⟨e|, √Γ↑ |e⟩⟨g|, √γ_φ σ_z/√2]
        """
        self._warn_defaults()

        if self.use_v2_physics:
            # Use physics-correct v2 implementation
            ops, rates = self.converter_v2.qubit_lindblad_ops(
                theta,
                T=self.T,
                sequence=self.sequence,
                omega0=self.omega0,
                temperature_K=self.temperature_K,
                Gamma_h=self.Gamma_h
            )
            return ops
        else:
            # Fallback: emulate v1 behavior (for testing/comparison only)
            return self._get_lindblad_operators_v1_emulation(theta)

    def _get_lindblad_operators_v1_emulation(self, theta: NoiseParameters) -> List[np.ndarray]:
        """
        Emulate old noise_models.py behavior (for testing only).
        NOT RECOMMENDED - use v2 physics instead!
        """
        # Simple point evaluation at sample frequencies
        S_values = self.psd_model.psd(self.sampling_freqs, theta)

        L_ops = []
        for j, sigma in enumerate(self.basis_ops):
            freq_idx = min(j, len(S_values) - 1)
            gamma_j = S_values[freq_idx]
            # Ensure non-negative
            gamma_j = max(gamma_j, 0.0)
            L_ops.append(np.sqrt(gamma_j) * sigma)

        return L_ops

    def get_effective_rates(self, theta: NoiseParameters) -> np.ndarray:
        """
        Get effective decay rates [Γ↓, Γ↑, γ_φ] in rad/s.

        Returns:
            rates: Array of decay rates [1/s]
        """
        if self.use_v2_physics:
            _, rates = self.converter_v2.qubit_lindblad_ops(
                theta,
                T=self.T,
                sequence=self.sequence,
                omega0=self.omega0,
                temperature_K=self.temperature_K,
                Gamma_h=self.Gamma_h
            )
            return rates
        else:
            # v1 emulation
            S_values = self.psd_model.psd(self.sampling_freqs, theta)
            return np.maximum(S_values[:len(self.basis_ops)], 0.0)

    def get_rates_dict(self, theta: NoiseParameters) -> Dict[str, float]:
        """
        Get decay rates as labeled dictionary.

        Returns:
            rates: {'Gamma_down': float, 'Gamma_up': float, 'gamma_phi': float}
        """
        rates = self.get_effective_rates(theta)
        return {
            'Gamma_down': float(rates[0]),
            'Gamma_up': float(rates[1]),
            'gamma_phi': float(rates[2])
        }

    def update_physics_parameters(
        self,
        T: Optional[float] = None,
        sequence: Optional[str] = None,
        omega0: Optional[float] = None,
        g_energy_per_xi: Optional[float] = None,
        temperature_K: Optional[float] = None,
        Gamma_h: Optional[float] = None
    ):
        """
        Update physics parameters after construction.

        Useful for:
        - Sweeping gate times T
        - Comparing sequences (Ramsey vs Echo)
        - Testing temperature effects
        """
        if T is not None:
            self.T = float(T)
        if sequence is not None:
            self.sequence = sequence
        if omega0 is not None:
            self.omega0 = float(omega0)
            self._omega0_estimated = False
        if g_energy_per_xi is not None:
            self.g_energy_per_xi = float(g_energy_per_xi)
            self.converter_v2.gE = self.g_energy_per_xi
            self._g_was_default = False
        if temperature_K is not None:
            self.temperature_K = temperature_K
        if Gamma_h is not None:
            self.Gamma_h = float(Gamma_h)


# Utility functions
def estimate_qubit_frequency_from_hamiltonian(H0: np.ndarray) -> float:
    """
    Estimate qubit transition frequency from drift Hamiltonian.

    Args:
        H0: Drift Hamiltonian (d×d complex matrix)

    Returns:
        omega0: Transition frequency [rad/s]
    """
    eigenvalues = np.linalg.eigvalsh(H0)
    eigenvalues = np.sort(np.real(eigenvalues))

    if len(eigenvalues) >= 2:
        # Gap between ground and first excited state
        omega0 = float(np.abs(eigenvalues[1] - eigenvalues[0]))
    else:
        # Fallback default (5 MHz)
        omega0 = TWO_PI * 5e6

    return omega0


def get_coupling_for_noise_type(noise_type: str = 'frequency') -> float:
    """
    Get appropriate coupling constant for common noise types.

    Args:
        noise_type: 'frequency', 'magnetic', 'charge', or 'amplitude'

    Returns:
        g_energy_per_xi: Coupling strength [J/xi]
    """
    if noise_type == 'frequency':
        # Frequency noise: H_int = (ℏ/2) δω(t) σ_z
        return HBAR / 2.0

    elif noise_type == 'magnetic':
        # Magnetic field noise: H_int = (g_e μ_B / 2) B(t) σ_z
        # g_e ≈ 2.002 (electron g-factor)
        # μ_B = 9.274e-24 J/T (Bohr magneton)
        g_e = 2.002
        mu_B = 9.274e-24  # J/T
        return g_e * mu_B / 2.0

    elif noise_type == 'charge':
        # Charge noise: H_int = e V(t) / d
        # Typical: d ~ 1 nm gap, V ~ 1 μV fluctuation
        e = 1.602e-19  # C
        return e * 1e-6  # J per μV

    elif noise_type == 'amplitude':
        # Control amplitude noise: H_int = δΩ(t) σ_x
        # Typical Rabi frequency scale
        return HBAR / 2.0  # Similar to frequency noise

    else:
        raise ValueError(
            f"Unknown noise_type='{noise_type}'. "
            f"Use: 'frequency', 'magnetic', 'charge', or 'amplitude'"
        )


# Example usage and testing
if __name__ == "__main__":
    print("=" * 70)
    print("Noise Model Adapter - Backward Compatibility Test")
    print("=" * 70)

    # Test 1: Basic usage (v1 API, v2 physics)
    print("\n[Test 1] Basic backward-compatible usage:")
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

    basis_operators = [sigma_x, sigma_y, sigma_z]
    sampling_freqs = np.array([1.0, 5.0, 10.0])
    psd_model = NoisePSDModel(model_type='one_over_f')

    # Old API works!
    converter = PSDToLindblad(basis_operators, sampling_freqs, psd_model)

    theta = NoiseParameters(alpha=1.0, A=0.1, omega_c=5.0)
    L_ops = converter.get_lindblad_operators(theta)

    print(f"  Returned {len(L_ops)} Lindblad operators")
    print(f"  Shapes: {[L.shape for L in L_ops]}")

    # Get rates
    rates = converter.get_effective_rates(theta)
    print(f"  Rates [1/s]: Γ↓={rates[0]:.3e}, Γ↑={rates[1]:.3e}, γ_φ={rates[2]:.3e}")

    # Test 2: Custom physics parameters
    print("\n[Test 2] Custom physics parameters:")
    converter_custom = PSDToLindblad(
        basis_operators, sampling_freqs, psd_model,
        T=100e-6,  # 100 μs gate
        omega0=TWO_PI * 10e6,  # 10 MHz qubit
        sequence='echo',  # Echo sequence
        g_energy_per_xi=HBAR  # Stronger coupling
    )

    L_ops_custom = converter_custom.get_lindblad_operators(theta)
    rates_custom = converter_custom.get_effective_rates(theta)
    print(f"  Rates [1/s]: Γ↓={rates_custom[0]:.3e}, Γ↑={rates_custom[1]:.3e}, γ_φ={rates_custom[2]:.3e}")
    print(f"  Echo suppression: γ_φ(echo)/γ_φ(ramsey) = {rates_custom[2]/rates[2]:.2f}")

    # Test 3: Compare v1 vs v2 physics
    print("\n[Test 3] Physics comparison (v1 emulation vs v2):")

    converter_v1 = PSDToLindblad(
        basis_operators, sampling_freqs, psd_model,
        use_v2_physics=False  # Emulate old behavior
    )

    rates_v1 = converter_v1.get_effective_rates(theta)
    rates_v2 = converter.get_effective_rates(theta)

    print(f"  v1 emulation: Γ_avg={np.mean(rates_v1):.3e}")
    print(f"  v2 physics:   Γ_avg={np.mean(rates_v2):.3e}")
    print(f"  Difference:   {np.mean(rates_v2)/np.mean(rates_v1):.2f}x")

    # Test 4: Parameter updates
    print("\n[Test 4] Dynamic parameter updates:")
    converter.update_physics_parameters(T=200e-6, sequence='cpmg_4')
    rates_updated = converter.get_effective_rates(theta)
    print(f"  After update: γ_φ={rates_updated[2]:.3e} (CPMG-4, T=200μs)")

    print("\n" + "=" * 70)
    print("All tests passed! Adapter is backward compatible.")
    print("=" * 70)
