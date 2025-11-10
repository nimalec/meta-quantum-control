"""
Two-Qubit Gate Definitions and Operators

Extends the quantum gate module to support 2-qubit systems (d=4).
Includes CNOT, CZ, SWAP, and other essential 2-qubit gates.
"""

import numpy as np
from typing import Tuple, List


def pauli_matrices():
    """Standard Pauli matrices"""
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return I, X, Y, Z


def tensor_product(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Kronecker product for tensor product of operators"""
    return np.kron(A, B)


def single_qubit_on_two_qubit(
    pauli: str,
    qubit: int = 0
) -> np.ndarray:
    """
    Apply single-qubit Pauli on specified qubit of 2-qubit system

    Args:
        pauli: 'I', 'X', 'Y', or 'Z'
        qubit: 0 or 1

    Returns:
        4x4 operator
    """
    I, X, Y, Z = pauli_matrices()

    pauli_map = {'I': I, 'X': X, 'Y': Y, 'Z': Z}
    P = pauli_map[pauli.upper()]

    if qubit == 0:
        return tensor_product(P, I)
    elif qubit == 1:
        return tensor_product(I, P)
    else:
        raise ValueError(f"Qubit must be 0 or 1, got {qubit}")


def get_two_qubit_control_hamiltonians() -> List[np.ndarray]:
    """
    Standard control Hamiltonians for 2-qubit system

    Returns:
        List of 4x4 control Hamiltonians:
        - H1: X on qubit 0 (XI)
        - H2: X on qubit 1 (IX)
        - H3: Y on qubit 0 (YI)
        - H4: Z⊗Z interaction (entangling)
    """
    I, X, Y, Z = pauli_matrices()

    return [
        tensor_product(X, I),  # XI
        tensor_product(I, X),  # IX
        tensor_product(Y, I),  # YI
        tensor_product(Z, Z),  # ZZ (entangling)
    ]


def get_two_qubit_noise_operators(qubit: int = None) -> List[np.ndarray]:
    """
    Standard noise operators for 2-qubit system

    Args:
        qubit: If specified, only return noise on that qubit
               If None, return noise on both qubits

    Returns:
        List of Lindblad operators (before scaling by sqrt(Gamma))
    """
    I, X, Y, Z = pauli_matrices()

    # Dephasing and relaxation operators
    sigma_z_0 = tensor_product(Z, I)
    sigma_z_1 = tensor_product(I, Z)

    # Lowering operators (relaxation)
    sigma_minus = np.array([[0, 0], [1, 0]], dtype=complex)
    sigma_m_0 = tensor_product(sigma_minus, I)
    sigma_m_1 = tensor_product(I, sigma_minus)

    if qubit is None:
        # Both qubits
        return [sigma_z_0, sigma_z_1, sigma_m_0, sigma_m_1]
    elif qubit == 0:
        return [sigma_z_0, sigma_m_0]
    elif qubit == 1:
        return [sigma_z_1, sigma_m_1]
    else:
        raise ValueError(f"Qubit must be 0, 1, or None, got {qubit}")


# ============================================================================
# Standard 2-Qubit Gates
# ============================================================================

def cnot_gate() -> np.ndarray:
    """
    CNOT gate (controlled-X)
    Control: qubit 0, Target: qubit 1

    CNOT = |0⟩⟨0| ⊗ I + |1⟩⟨1| ⊗ X
    """
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=complex)


def cz_gate() -> np.ndarray:
    """
    CZ gate (controlled-Z)
    Control: qubit 0, Target: qubit 1

    CZ = |0⟩⟨0| ⊗ I + |1⟩⟨1| ⊗ Z
    """
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, -1]
    ], dtype=complex)


def swap_gate() -> np.ndarray:
    """
    SWAP gate
    Exchanges states of qubits 0 and 1
    """
    return np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ], dtype=complex)


def iswap_gate() -> np.ndarray:
    """
    iSWAP gate
    Swaps qubits with phase
    """
    return np.array([
        [1, 0, 0, 0],
        [0, 0, 1j, 0],
        [0, 1j, 0, 0],
        [0, 0, 0, 1]
    ], dtype=complex)


# ============================================================================
# State Preparation
# ============================================================================

def computational_basis_state(state_str: str) -> np.ndarray:
    """
    Create computational basis state from string

    Args:
        state_str: '00', '01', '10', or '11'

    Returns:
        State vector (4,)
    """
    state_map = {
        '00': np.array([1, 0, 0, 0], dtype=complex),
        '01': np.array([0, 1, 0, 0], dtype=complex),
        '10': np.array([0, 0, 1, 0], dtype=complex),
        '11': np.array([0, 0, 0, 1], dtype=complex),
    }

    if state_str not in state_map:
        raise ValueError(f"Invalid state string: {state_str}. Use '00', '01', '10', or '11'")

    return state_map[state_str]


def bell_state(which: str = 'phi_plus') -> np.ndarray:
    """
    Create Bell state (maximally entangled 2-qubit state)

    Args:
        which: 'phi_plus', 'phi_minus', 'psi_plus', or 'psi_minus'

    Returns:
        State vector (4,)
    """
    sqrt2 = np.sqrt(2)

    bell_states = {
        'phi_plus':  np.array([1, 0, 0, 1], dtype=complex) / sqrt2,   # (|00⟩ + |11⟩)/√2
        'phi_minus': np.array([1, 0, 0, -1], dtype=complex) / sqrt2,  # (|00⟩ - |11⟩)/√2
        'psi_plus':  np.array([0, 1, 1, 0], dtype=complex) / sqrt2,   # (|01⟩ + |10⟩)/√2
        'psi_minus': np.array([0, 1, -1, 0], dtype=complex) / sqrt2,  # (|01⟩ - |10⟩)/√2
    }

    if which not in bell_states:
        raise ValueError(f"Invalid Bell state: {which}")

    return bell_states[which]


# ============================================================================
# Fidelity Measures for 2-Qubit Systems
# ============================================================================

def process_fidelity(rho: np.ndarray, U_target: np.ndarray) -> float:
    """
    Process fidelity between density matrix and target unitary

    F = |Tr(U†_target ρ)|² / d

    Args:
        rho: Final density matrix (4x4)
        U_target: Target unitary (4x4)

    Returns:
        Fidelity in [0, 1]
    """
    d = rho.shape[0]
    trace = np.trace(U_target.conj().T @ rho)
    return float(np.abs(trace)**2 / d)


def state_fidelity_two_qubit(rho: np.ndarray, psi_target: np.ndarray) -> float:
    """
    Fidelity between density matrix and pure target state

    F = ⟨ψ|ρ|ψ⟩

    Args:
        rho: Density matrix (4x4)
        psi_target: Target state vector (4,)

    Returns:
        Fidelity in [0, 1]
    """
    psi = psi_target.reshape(-1, 1)
    fidelity = psi.conj().T @ rho @ psi
    return float(np.real(fidelity[0, 0]))


def entanglement_fidelity(rho: np.ndarray, U_target: np.ndarray) -> float:
    """
    Entanglement fidelity (alternative fidelity measure)

    F_e = Tr(U†_target ρ U_target) / d

    Args:
        rho: Final density matrix (4x4)
        U_target: Target unitary (4x4)

    Returns:
        Entanglement fidelity in [0, 1]
    """
    d = rho.shape[0]
    rotated_rho = U_target.conj().T @ rho @ U_target
    return float(np.real(np.trace(rotated_rho)) / d)


# ============================================================================
# Helper Functions
# ============================================================================

def density_matrix(psi: np.ndarray) -> np.ndarray:
    """Convert state vector to density matrix"""
    psi = psi.reshape(-1, 1)
    return psi @ psi.conj().T


def partial_trace(rho: np.ndarray, keep: int = 0) -> np.ndarray:
    """
    Partial trace over 2-qubit system

    Args:
        rho: 4x4 density matrix
        keep: Which qubit to keep (0 or 1)

    Returns:
        2x2 reduced density matrix
    """
    rho_reshaped = rho.reshape(2, 2, 2, 2)

    if keep == 0:
        # Trace out qubit 1
        return np.trace(rho_reshaped, axis1=1, axis2=3)
    elif keep == 1:
        # Trace out qubit 0
        return np.trace(rho_reshaped, axis1=0, axis2=2)
    else:
        raise ValueError(f"keep must be 0 or 1, got {keep}")


def get_target_gate(gate_name: str) -> np.ndarray:
    """
    Get target gate by name

    Args:
        gate_name: 'cnot', 'cz', 'swap', 'iswap'

    Returns:
        4x4 unitary matrix
    """
    gate_map = {
        'cnot': cnot_gate(),
        'cz': cz_gate(),
        'swap': swap_gate(),
        'iswap': iswap_gate(),
    }

    gate_name_lower = gate_name.lower()
    if gate_name_lower not in gate_map:
        raise ValueError(f"Unknown gate: {gate_name}. Options: {list(gate_map.keys())}")

    return gate_map[gate_name_lower]


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("2-Qubit Gate Library")
    print("=" * 60)

    # Test CNOT gate
    cnot = cnot_gate()
    print("\nCNOT gate:")
    print(cnot)

    # Test state preparation
    psi_00 = computational_basis_state('00')
    print("\n|00⟩ state:")
    print(psi_00)

    # Test Bell state
    bell = bell_state('phi_plus')
    print("\n|Φ+⟩ Bell state:")
    print(bell)

    # Test control Hamiltonians
    H_controls = get_two_qubit_control_hamiltonians()
    print(f"\nNumber of control Hamiltonians: {len(H_controls)}")
    print(f"Shape of each: {H_controls[0].shape}")

    # Test fidelity
    rho_bell = density_matrix(bell)
    fid = process_fidelity(rho_bell, cnot)
    print(f"\nFidelity of |Φ+⟩ with CNOT: {fid:.4f}")
