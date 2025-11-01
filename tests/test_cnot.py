"""
Proper CNOT gate testing - using multiple initial states

The issue: CNOT|00⟩ = |00⟩, so state fidelity on |00⟩ alone doesn't test the gate!

We need to test on computational basis states:
- CNOT|00⟩ = |00⟩  ✓
- CNOT|01⟩ = |01⟩  ✓
- CNOT|10⟩ = |11⟩  ← This is the real test!
- CNOT|11⟩ = |10⟩  ← This is the real test!
"""

import numpy as np
from metaqctrl.theory.quantum_environment import create_quantum_environment
from metaqctrl.quantum.two_qubit_gates import (
    cnot_gate,
    computational_basis_state,
    density_matrix,
    state_fidelity_two_qubit
)
from metaqctrl.quantum.noise_models import NoiseParameters

def test_cnot_all_inputs():
    """Test CNOT gate on all 4 computational basis states"""

    print("=" * 70)
    print("Proper CNOT Gate Testing - All Input States")
    print("=" * 70)

    # Configuration
    config = {
        'num_qubits': 2,
        'target_gate': 'cnot',
        'psd_model': 'one_over_f',
        'n_segments': 30,  # More segments for 2-qubit
        'horizon': 2.0,     # Longer time for 2-qubit gates
        'drift_strength': 1.0,  # Stronger drift
        'integration_method': 'RK45'
    }

    env = create_quantum_environment(config)
    task_params = NoiseParameters(alpha=1.0, A=0.05, omega_c=10.0)  # Lower noise
    sim = env.get_simulator(task_params)

    # Get CNOT gate
    U_cnot = cnot_gate()

    # Test on all computational basis states
    basis_states = ['00', '01', '10', '11']
    expected_outputs = {
        '00': '00',  # CNOT doesn't flip
        '01': '01',  # CNOT doesn't flip
        '10': '11',  # CNOT flips second qubit
        '11': '10',  # CNOT flips second qubit
    }

    print(f"\nCNOT gate truth table:")
    print(f"  |00⟩ → |00⟩")
    print(f"  |01⟩ → |01⟩")
    print(f"  |10⟩ → |11⟩  ← Tests the flip!")
    print(f"  |11⟩ → |10⟩  ← Tests the flip!")

    # Design better control pulses (optimized by hand for demonstration)
    t = np.linspace(0, config['horizon'], config['n_segments'])

    # CNOT requires: single-qubit rotations + ZZ entangling gate
    # Rough pulse sequence: X1 rotation, ZZ interaction, X1 rotation
    controls = np.zeros((config['n_segments'], env.n_controls))

    # Segment the evolution
    n = len(t)
    n1, n2, n3 = n // 3, 2 * n // 3, n

    # Stage 1: Hadamard-like on qubit 0 (control qubit)
    controls[0:n1, 0] = 5.0  # XI (strong)
    controls[0:n1, 2] = 2.0  # YI

    # Stage 2: ZZ entangling interaction
    controls[n1:n2, 3] = 8.0 * np.sin(np.pi * t[n1:n2] / config['horizon'])  # ZZ

    # Stage 3: Rotate back
    controls[n2:n3, 0] = -5.0  # XI
    controls[n2:n3, 1] = 3.0   # IX

    print(f"\n" + "=" * 70)
    print("Testing with hand-designed pulse sequence:")
    print("=" * 70)

    fidelities = []

    for input_state_str in basis_states:
        # Get input and expected output states
        ket_in = computational_basis_state(input_state_str)
        ket_out_expected = U_cnot @ ket_in

        rho_in = density_matrix(ket_in)
        rho_out_expected = density_matrix(ket_out_expected)

        # Simulate with controls
        rho_final, _ = sim.evolve(rho_in, controls, config['horizon'])

        # Compute fidelity
        fidelity = state_fidelity_two_qubit(rho_final, ket_out_expected)
        fidelities.append(fidelity)

        expected_str = expected_outputs[input_state_str]
        print(f"|{input_state_str}⟩ → |{expected_str}⟩: F = {fidelity:.4f}")

    avg_fidelity = np.mean(fidelities)
    print(f"\n{'=' * 70}")
    print(f"Average gate fidelity: {avg_fidelity:.4f}")
    print(f"{'=' * 70}")

    if avg_fidelity < 0.5:
        print("\n⚠️  DIAGNOSIS:")
        print("   Gate fidelity is still low. This is EXPECTED without optimization!")
        print("   Hand-designed pulses rarely work for complex gates like CNOT.")
        print("\n   Solutions:")
        print("   1. Use GRAPE or other gradient-based optimization")
        print("   2. Use meta-RL (MAML) to learn optimal pulses")
        print("   3. Increase evolution time (try T=3-5)")
        print("   4. Reduce noise (decrease A parameter)")
        print("   5. Tune drift_strength to match your system")
        print(f"\n   For reference:")
        print(f"   - Random gate: F ≈ 0.20-0.25")
        print(f"   - Your result: F = {avg_fidelity:.4f}")
        print(f"   - Good gate:   F > 0.90")
        print(f"   - Excellent:   F > 0.99")

    return avg_fidelity


def compute_average_gate_fidelity_from_state(config_override=None):
    """
    Alternative: Use average gate fidelity formula

    F_avg = (d F_ent + 1) / (d + 1)

    where F_ent is entanglement fidelity
    """
    print("\n" + "=" * 70)
    print("Alternative: Entanglement Fidelity Method")
    print("=" * 70)

    config = {
        'num_qubits': 2,
        'target_gate': 'cnot',
        'psd_model': 'one_over_f',
        'n_segments': 30,
        'horizon': 2.0,
        'drift_strength': 1.0,
        'integration_method': 'RK45'
    }

    if config_override:
        config.update(config_override)

    env = create_quantum_environment(config)
    task_params = NoiseParameters(alpha=1.0, A=0.05, omega_c=10.0)
    sim = env.get_simulator(task_params)

    # Zero controls
    controls = np.zeros((config['n_segments'], env.n_controls))

    U_cnot = cnot_gate()
    rho_init = env.rho0

    rho_final, _ = sim.evolve(rho_init, controls, config['horizon'])

    # Entanglement fidelity: F_e = Tr(U† ρ U) / d
    from metaqctrl.quantum.two_qubit_gates import entanglement_fidelity
    F_ent = entanglement_fidelity(rho_final, U_cnot)

    # Average gate fidelity
    d = 4
    F_avg = (d * F_ent + 1) / (d + 1)

    print(f"Entanglement fidelity: {F_ent:.4f}")
    print(f"Average gate fidelity: {F_avg:.4f}")

    return F_avg


if __name__ == "__main__":
    # Test on all input states
    avg_fid = test_cnot_all_inputs()

    # Alternative method
    # avg_fid_ent = compute_average_gate_fidelity_from_state()

    print("\n" + "=" * 70)
    print("KEY INSIGHT:")
    print("=" * 70)
    print("Your original 50% fidelity was from testing CNOT|00⟩ = |00⟩")
    print("This doesn't actually test the CNOT gate functionality!")
    print("\nThe real test is: CNOT|10⟩ = |11⟩ (flip when control is |1⟩)")
    print("\nYou need to either:")
    print("  1. Average fidelity over all 4 input states (shown above)")
    print("  2. Use process fidelity / Choi matrix representation")
    print("  3. Use entanglement fidelity formula")
    print("=" * 70)
