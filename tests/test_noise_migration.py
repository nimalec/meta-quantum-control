"""
Comprehensive test for noise_models migration to v2.

This test verifies:
1. Backward compatibility of API
2. Correct physics implementation
3. Integration with existing code
4. No breaking changes
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from metaqctrl.quantum.noise_adapter import (
    NoiseParameters, NoisePSDModel, PSDToLindblad, TaskDistribution,
    estimate_qubit_frequency_from_hamiltonian, get_coupling_for_noise_type
)
from metaqctrl.theory.quantum_environment import create_quantum_environment


class TestNoiseParametersCompatibility:
    """Test NoiseParameters backward compatibility."""

    def test_creation(self):
        """Test creating NoiseParameters."""
        theta = NoiseParameters(alpha=1.0, A=0.1, omega_c=5.0)
        assert theta.alpha == 1.0
        assert theta.A == 0.1
        assert theta.omega_c == 5.0

    def test_to_array(self):
        """Test conversion to array."""
        theta = NoiseParameters(alpha=1.0, A=0.1, omega_c=5.0)
        arr = theta.to_array()
        assert arr.shape == (3,)
        assert np.allclose(arr, [1.0, 0.1, 5.0])

    def test_from_array(self):
        """Test creation from array."""
        arr = np.array([1.5, 0.2, 6.0])
        theta = NoiseParameters.from_array(arr)
        assert theta.alpha == 1.5
        assert theta.A == 0.2
        assert theta.omega_c == 6.0


class TestPSDModelCompatibility:
    """Test PSD model backward compatibility."""

    def test_one_over_f(self):
        """Test 1/f noise model."""
        psd_model = NoisePSDModel(model_type='one_over_f')
        theta = NoiseParameters(alpha=1.0, A=0.1, omega_c=5.0)
        omega = np.array([1.0, 5.0, 10.0])
        S = psd_model.psd(omega, theta)

        assert S.shape == omega.shape
        assert np.all(S > 0), "PSD must be positive"
        assert S[0] > S[-1], "1/f noise should decrease with frequency"

    def test_lorentzian(self):
        """Test Lorentzian noise model."""
        psd_model = NoisePSDModel(model_type='lorentzian')
        theta = NoiseParameters(alpha=1.0, A=0.1, omega_c=5.0)
        omega = np.array([1.0, 5.0, 10.0])
        S = psd_model.psd(omega, theta)

        assert S.shape == omega.shape
        assert np.all(S > 0)


class TestPSDToLindbladAdapter:
    """Test PSD to Lindblad adapter backward compatibility."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        self.sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self.sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        self.basis_operators = [self.sigma_x, self.sigma_y, self.sigma_z]
        self.sampling_freqs = np.array([1.0, 5.0, 10.0])
        self.psd_model = NoisePSDModel(model_type='one_over_f')

    def test_old_api_works(self):
        """Test that old API still works."""
        # Old API (v1 style)
        converter = PSDToLindblad(
            self.basis_operators,
            self.sampling_freqs,
            self.psd_model
        )

        theta = NoiseParameters(alpha=1.0, A=0.1, omega_c=5.0)
        L_ops = converter.get_lindblad_operators(theta)

        assert len(L_ops) == 3, "Should have 3 Lindblad operators"
        for L in L_ops:
            assert L.shape == (2, 2), "Operators should be 2x2"
            assert L.dtype == complex, "Operators should be complex"

    def test_physics_parameters(self):
        """Test new physics parameters."""
        converter = PSDToLindblad(
            self.basis_operators,
            self.sampling_freqs,
            self.psd_model,
            T=100e-6,  # 100 μs
            omega0=2*np.pi*5e6,  # 5 MHz
            sequence='echo'
        )

        theta = NoiseParameters(alpha=1.0, A=0.1, omega_c=5.0)
        L_ops = converter.get_lindblad_operators(theta)

        assert len(L_ops) == 3

    def test_effective_rates(self):
        """Test getting effective decay rates."""
        converter = PSDToLindblad(
            self.basis_operators,
            self.sampling_freqs,
            self.psd_model
        )

        theta = NoiseParameters(alpha=1.0, A=0.1, omega_c=5.0)
        rates = converter.get_effective_rates(theta)

        assert rates.shape == (3,), "Should have 3 rates"
        assert np.all(rates >= 0), "Rates must be non-negative"

    def test_rates_dict(self):
        """Test getting rates as dictionary."""
        converter = PSDToLindblad(
            self.basis_operators,
            self.sampling_freqs,
            self.psd_model
        )

        theta = NoiseParameters(alpha=1.0, A=0.1, omega_c=5.0)
        rates_dict = converter.get_rates_dict(theta)

        assert 'Gamma_down' in rates_dict
        assert 'Gamma_up' in rates_dict
        assert 'gamma_phi' in rates_dict
        assert all(v >= 0 for v in rates_dict.values())

    def test_parameter_updates(self):
        """Test dynamic parameter updates."""
        converter = PSDToLindblad(
            self.basis_operators,
            self.sampling_freqs,
            self.psd_model,
            T=50e-6
        )

        theta = NoiseParameters(alpha=1.0, A=0.1, omega_c=5.0)
        rates_before = converter.get_effective_rates(theta)

        # Update to longer gate time
        converter.update_physics_parameters(T=200e-6)
        rates_after = converter.get_effective_rates(theta)

        # Rates should change with different T
        assert not np.allclose(rates_before, rates_after)


class TestTaskDistributionCompatibility:
    """Test TaskDistribution backward compatibility."""

    def test_uniform_sampling(self):
        """Test uniform distribution sampling."""
        dist = TaskDistribution(
            dist_type='uniform',
            ranges={
                'alpha': (0.5, 2.0),
                'A': (0.05, 0.3),
                'omega_c': (2.0, 8.0)
            }
        )

        rng = np.random.default_rng(42)
        tasks = dist.sample(10, rng)

        assert len(tasks) == 10
        for task in tasks:
            assert isinstance(task, NoiseParameters)
            assert 0.5 <= task.alpha <= 2.0
            assert 0.05 <= task.A <= 0.3
            assert 2.0 <= task.omega_c <= 8.0

    def test_variance_computation(self):
        """Test variance computation."""
        dist = TaskDistribution(
            dist_type='uniform',
            ranges={
                'alpha': (0.5, 2.0),
                'A': (0.05, 0.3),
                'omega_c': (2.0, 8.0)
            }
        )

        variance = dist.compute_variance()
        assert variance > 0, "Variance must be positive"


class TestUtilityFunctions:
    """Test utility functions."""

    def test_estimate_qubit_frequency(self):
        """Test qubit frequency estimation."""
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        H0 = 0.5 * sigma_z  # 0.5 rad/s splitting

        omega0 = estimate_qubit_frequency_from_hamiltonian(H0)
        assert omega0 > 0
        assert np.isclose(omega0, 1.0, rtol=0.1)  # Should be ~1.0 rad/s

    def test_coupling_constants(self):
        """Test coupling constant retrieval."""
        g_freq = get_coupling_for_noise_type('frequency')
        g_mag = get_coupling_for_noise_type('magnetic')

        assert g_freq > 0
        assert g_mag > 0
        # Frequency noise uses ℏ/2, magnetic is different
        assert g_freq != g_mag


class TestQuantumEnvironmentIntegration:
    """Test integration with QuantumEnvironment."""

    def test_create_environment_minimal_config(self):
        """Test creating environment with minimal config."""
        config = {
            'num_qubits': 1,
            'target_gate': 'pauli_x',
            'n_segments': 20,
            'horizon': 1.0
        }

        env = create_quantum_environment(config)

        assert env is not None
        assert env.d == 2  # 1-qubit system
        assert env.n_controls == 2  # X and Y controls
        assert env.T == 1.0

    def test_create_environment_with_physics(self):
        """Test creating environment with physics parameters."""
        config = {
            'num_qubits': 1,
            'target_gate': 'hadamard',
            'n_segments': 20,
            'horizon': 50e-6,  # 50 μs
            'omega0': 2*np.pi*10e6,  # 10 MHz
            'sequence': 'echo',
            'noise_type': 'frequency',
            'drift_strength': 0.1
        }

        env = create_quantum_environment(config)

        assert env is not None
        assert env.T == 50e-6

    def test_lindblad_operators_cached(self):
        """Test that Lindblad operators are cached."""
        config = {
            'num_qubits': 1,
            'target_gate': 'pauli_x',
            'n_segments': 20,
            'horizon': 1.0
        }

        env = create_quantum_environment(config)

        theta = NoiseParameters(alpha=1.0, A=0.1, omega_c=5.0)

        # First call - should cache
        L_ops1 = env.get_lindblad_operators(theta)

        # Second call - should use cache
        L_ops2 = env.get_lindblad_operators(theta)

        # Should be same objects (cached)
        assert L_ops1 is L_ops2

        # Cache should have 1 entry
        stats = env.get_cache_stats()
        assert stats['n_cached_operators'] == 1


class TestPhysicsCorrectness:
    """Test that v2 physics is being used correctly."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        self.sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self.sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        self.basis_operators = [self.sigma_x, self.sigma_y, self.sigma_z]
        self.sampling_freqs = np.array([1.0, 5.0, 10.0])
        self.psd_model = NoisePSDModel(model_type='one_over_f')

    def test_echo_suppresses_dephasing(self):
        """Test that echo sequence suppresses dephasing vs Ramsey."""
        theta = NoiseParameters(alpha=1.0, A=0.1, omega_c=5.0)

        # Ramsey
        converter_ramsey = PSDToLindblad(
            self.basis_operators,
            self.sampling_freqs,
            self.psd_model,
            T=50e-6,
            sequence='ramsey'
        )
        rates_ramsey = converter_ramsey.get_rates_dict(theta)

        # Echo
        converter_echo = PSDToLindblad(
            self.basis_operators,
            self.sampling_freqs,
            self.psd_model,
            T=50e-6,
            sequence='echo'
        )
        rates_echo = converter_echo.get_rates_dict(theta)

        # Echo should suppress low-frequency noise (dephasing)
        assert rates_echo['gamma_phi'] < rates_ramsey['gamma_phi'], \
            "Echo should suppress dephasing compared to Ramsey"

    def test_relaxation_at_transition_frequency(self):
        """Test that relaxation uses transition frequency."""
        theta = NoiseParameters(alpha=1.0, A=0.1, omega_c=5.0)

        # Different transition frequencies
        omega0_low = 2*np.pi*1e6  # 1 MHz
        omega0_high = 2*np.pi*10e6  # 10 MHz

        converter_low = PSDToLindblad(
            self.basis_operators,
            self.sampling_freqs,
            self.psd_model,
            omega0=omega0_low
        )

        converter_high = PSDToLindblad(
            self.basis_operators,
            self.sampling_freqs,
            self.psd_model,
            omega0=omega0_high
        )

        rates_low = converter_low.get_rates_dict(theta)
        rates_high = converter_high.get_rates_dict(theta)

        # For 1/f noise, rates should be different at different frequencies
        assert rates_low['Gamma_down'] != rates_high['Gamma_down'], \
            "Relaxation should depend on transition frequency"


def run_tests():
    """Run all tests."""
    pytest.main([__file__, '-v', '--tb=short'])


if __name__ == "__main__":
    print("=" * 70)
    print("Noise Model Migration Test Suite")
    print("=" * 70)
    print()
    print("Testing backward compatibility and physics correctness...")
    print()

    run_tests()
