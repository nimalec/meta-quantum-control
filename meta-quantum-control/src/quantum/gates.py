"""
Gate Fidelity Computation

Implements various fidelity measures:
- State fidelity: F(ρ, σ) = tr(√(√ρ σ √ρ))²
- Gate/process fidelity: F(Φ, Ψ) = tr(Φ†Ψ) / d²
- Average gate fidelity
"""

import numpy as np
from scipy.linalg import sqrtm
from typing import Tuple, Optional
import jax.numpy as jnp
from jax import jit


def state_fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    """
    Compute quantum state fidelity F(ρ, σ) = [tr(√(√ρ σ √ρ))]².
    
    For pure states: F = |⟨ψ|φ⟩|²
    
    Args:
        rho: Density matrix 1
        sigma: Density matrix 2
        
    Returns:
        fidelity: Value in [0, 1]
    """
    sqrt_rho = sqrtm(rho)
    M = sqrt_rho @ sigma @ sqrt_rho
    sqrt_M = sqrtm(M)
    return np.real(np.trace(sqrt_M)) ** 2


def process_fidelity_choi(Phi_choi: np.ndarray, Psi_choi: np.ndarray) -> float:
    """
    Process fidelity via Choi matrices:
    F(Φ, Ψ) = tr(Φ† Ψ) / d²
    
    Args:
        Phi_choi: Choi matrix of channel Φ
        Psi_choi: Choi matrix of channel Ψ
        
    Returns:
        fidelity: Value in [0, 1]
    """
    d2 = Phi_choi.shape[0]
    d = int(np.sqrt(d2))
    return np.real(np.trace(Phi_choi.conj().T @ Psi_choi)) / (d ** 2)


def average_gate_fidelity(rho_final: np.ndarray, rho_target: np.ndarray) -> float:
    """
    Average gate fidelity (averaged over pure input states on Bloch sphere).
    
    For single qubit: F_avg = (2F + 1) / 3 where F is state fidelity
    
    Args:
        rho_final: Achieved final state
        rho_target: Target state
        
    Returns:
        avg_fidelity: Value in [0, 1]
    """
    F_state = state_fidelity(rho_final, rho_target)
    d = rho_final.shape[0]
    # Average gate fidelity formula
    F_avg = (d * F_state + 1) / (d + 1)
    return F_avg


def gate_infidelity(rho_final: np.ndarray, rho_target: np.ndarray) -> float:
    """
    Gate infidelity: 1 - F(ρ_final, ρ_target).
    This is what we minimize in training.
    """
    return 1.0 - state_fidelity(rho_final, rho_target)


def diamond_norm_distance(Phi_choi: np.ndarray, Psi_choi: np.ndarray) -> float:
    """
    Diamond norm distance ||Φ - Ψ||◇ (approximate via Choi).
    
    For channels, ||Φ - Ψ||◇ ≈ λ_max(|Φ_choi - Ψ_choi|)
    
    This is expensive to compute exactly; we use spectral norm approximation.
    """
    diff = Phi_choi - Psi_choi
    eigenvalues = np.linalg.eigvalsh(diff.conj().T @ diff)
    return np.sqrt(np.max(eigenvalues))


class GateFidelityComputer:
    """Unified interface for computing various fidelity measures."""
    
    def __init__(
        self,
        target_gate: np.ndarray,
        fidelity_type: str = 'state',
        d: int = 2
    ):
        """
        Args:
            target_gate: Target unitary or target state
            fidelity_type: 'state', 'process', 'average'
            d: Hilbert space dimension
        """
        self.target_gate = target_gate
        self.fidelity_type = fidelity_type
        self.d = d
        
        # Precompute target state if needed
        if fidelity_type == 'state':
            self.rho_target = target_gate  # Assume already a density matrix
        elif fidelity_type == 'process':
            # Store target Choi matrix
            self.target_choi = self._unitary_to_choi(target_gate)
    
    def compute(self, rho_final: np.ndarray) -> float:
        """
        Compute fidelity between achieved state and target.
        
        Args:
            rho_final: Final density matrix achieved
            
        Returns:
            fidelity: Value in [0, 1]
        """
        if self.fidelity_type == 'state':
            return state_fidelity(rho_final, self.rho_target)
        elif self.fidelity_type == 'average':
            return average_gate_fidelity(rho_final, self.rho_target)
        else:
            raise NotImplementedError(f"Fidelity type {self.fidelity_type} not yet supported")
    
    def compute_from_unitary(self, U_achieved: np.ndarray, rho_init: np.ndarray) -> float:
        """
        Compute fidelity given achieved unitary and initial state.
        
        Args:
            U_achieved: Unitary approximation
            rho_init: Initial state
            
        Returns:
            fidelity
        """
        rho_final = U_achieved @ rho_init @ U_achieved.conj().T
        return self.compute(rho_final)
    
    def _unitary_to_choi(self, U: np.ndarray) -> np.ndarray:
        """Convert unitary to Choi matrix."""
        d = U.shape[0]
        # Choi matrix: sum_ij |i⟩⟨j| ⊗ U|i⟩⟨j|U†
        choi = np.zeros((d**2, d**2), dtype=complex)
        for i in range(d):
            for j in range(d):
                ket_i = np.zeros(d)
                ket_i[i] = 1
                ket_j = np.zeros(d)
                ket_j[j] = 1
                
                input_op = np.outer(ket_i, ket_j.conj())
                output_op = U @ input_op @ U.conj().T
                
                choi += np.kron(input_op, output_op)
        
        return choi / d


class TargetGates:
    """Standard quantum gates for benchmarking."""
    
    @staticmethod
    def pauli_x() -> np.ndarray:
        """Pauli X gate (NOT gate)."""
        return np.array([[0, 1], [1, 0]], dtype=complex)
    
    @staticmethod
    def pauli_y() -> np.ndarray:
        """Pauli Y gate."""
        return np.array([[0, -1j], [1j, 0]], dtype=complex)
    
    @staticmethod
    def pauli_z() -> np.ndarray:
        """Pauli Z gate."""
        return np.array([[1, 0], [0, -1]], dtype=complex)
    
    @staticmethod
    def hadamard() -> np.ndarray:
        """Hadamard gate."""
        return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    
    @staticmethod
    def phase(phi: float) -> np.ndarray:
        """Phase gate R_φ."""
        return np.array([[1, 0], [0, np.exp(1j * phi)]], dtype=complex)
    
    @staticmethod
    def rotation_x(theta: float) -> np.ndarray:
        """Rotation around X-axis."""
        return np.array([
            [np.cos(theta/2), -1j * np.sin(theta/2)],
            [-1j * np.sin(theta/2), np.cos(theta/2)]
        ], dtype=complex)
    
    @staticmethod
    def rotation_y(theta: float) -> np.ndarray:
        """Rotation around Y-axis."""
        return np.array([
            [np.cos(theta/2), -np.sin(theta/2)],
            [np.sin(theta/2), np.cos(theta/2)]
        ], dtype=complex)
    
    @staticmethod
    def rotation_z(theta: float) -> np.ndarray:
        """Rotation around Z-axis."""
        return np.array([
            [np.exp(-1j * theta/2), 0],
            [0, np.exp(1j * theta/2)]
        ], dtype=complex)
    
    @staticmethod
    def arbitrary_unitary(alpha: float, beta: float, gamma: float) -> np.ndarray:
        """
        Arbitrary single-qubit unitary via Euler angles.
        U = Rz(γ) Ry(β) Rz(α)
        """
        Rz_alpha = TargetGates.rotation_z(alpha)
        Ry_beta = TargetGates.rotation_y(beta)
        Rz_gamma = TargetGates.rotation_z(gamma)
        return Rz_gamma @ Ry_beta @ Rz_alpha


# JAX versions for fast autodiff
@jit
def state_fidelity_jax(rho: jnp.ndarray, sigma: jnp.ndarray) -> float:
    """JAX implementation of state fidelity."""
    # Simplified for pure states or use iterative sqrtm
    # For now, assume pure states: F = |tr(ρσ)|
    return jnp.abs(jnp.trace(rho @ sigma)) ** 2


@jit
def gate_infidelity_jax(rho_final: jnp.ndarray, rho_target: jnp.ndarray) -> float:
    """JAX gate infidelity for gradient computation."""
    fidelity = state_fidelity_jax(rho_final, rho_target)
    return 1.0 - fidelity


# Example usage
if __name__ == "__main__":
    # Test fidelity computation
    print("Testing fidelity measures...")
    
    # Pure states
    ket_0 = np.array([1, 0], dtype=complex)
    ket_1 = np.array([0, 1], dtype=complex)
    ket_plus = (ket_0 + ket_1) / np.sqrt(2)
    
    rho_0 = np.outer(ket_0, ket_0.conj())
    rho_1 = np.outer(ket_1, ket_1.conj())
    rho_plus = np.outer(ket_plus, ket_plus.conj())
    
    print(f"F(|0⟩, |0⟩) = {state_fidelity(rho_0, rho_0):.6f}")
    print(f"F(|0⟩, |1⟩) = {state_fidelity(rho_0, rho_1):.6f}")
    print(f"F(|0⟩, |+⟩) = {state_fidelity(rho_0, rho_plus):.6f}")
    
    # Target gates
    print("\nTarget gates:")
    gates = TargetGates()
    
    X = gates.pauli_x()
    print(f"X gate:\n{X}")
    
    H = gates.hadamard()
    rho_H = H @ rho_0 @ H.conj().T
    print(f"\nH|0⟩ fidelity with |+⟩: {state_fidelity(rho_H, rho_plus):.6f}")
    
    # Rotation gates
    Rx_pi = gates.rotation_x(np.pi)
    print(f"\nRx(π) ≈ X? {np.allclose(Rx_pi, 1j * X)}")
    
    # Fidelity computer
    fid_computer = GateFidelityComputer(rho_plus, fidelity_type='state')
    achieved_fidelity = fid_computer.compute(rho_H)
    print(f"\nFidelity computer: F = {achieved_fidelity:.6f}")
