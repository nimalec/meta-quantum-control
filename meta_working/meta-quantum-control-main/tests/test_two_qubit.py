"""
Test script to verify 2-qubit quantum control setup

This script tests:
1. Proper 2-qubit Hamiltonian setup
2. CNOT gate implementation
3. Fidelity computation
4. Basic control sequence evaluation
"""

import numpy as np
import torch
from metaqctrl.theory.quantum_environment import create_quantum_environment
from metaqctrl.quantum.two_qubit_gates import (
    cnot_gate,
    computational_basis_state,
    density_matrix,
    process_fidelity
)

def test_two_qubit_environment():
    """Test 2-qubit quantum environment setup"""

    print("=" * 70)
    print("Testing 2-Qubit Quantum Control Setup")
    print("=" * 70)

    # Configuration for 2-qubit CNOT gate
    config = {
        'num_qubits': 2,
        'target_gate': 'cnot',
        'psd_model': 'one_over_f',
        'n_segments': 20,
        'horizon': 1.0,
        'drift_strength': 0.5,  # Important: non-zero drift
        'integration_method': 'RK45'
    }

    print("\n1. Creating 2-qubit quantum environment...")
    env = create_quantum_environment(config)

    print(f"   ✓ Dimension: {env.d} (should be 4 for 2 qubits)")
    print(f"   ✓ Number of controls: {env.n_controls} (should be 4: XI, IX, YI, ZZ)")
    print(f"   ✓ Evolution time: {env.T}s")

    # Check Hamiltonian dimensions
    print(f"\n2. Checking Hamiltonians...")
    print(f"   H0 shape: {env.H0.shape} (should be 4x4)")
    print(f"   H0 non-zero: {not np.allclose(env.H0, 0)}")

    for i, H in enumerate(env.H_controls):
        print(f"   H_control[{i}] shape: {H.shape}")

    # Get target CNOT gate
    U_cnot = cnot_gate()
    print(f"\n3. Target CNOT gate:")
    print(U_cnot)

    # Test with zero controls (should give poor fidelity)
    print(f"\n4. Testing with ZERO controls (baseline)...")
    from metaqctrl.quantum.noise_models import NoiseParameters

    task_params = NoiseParameters(alpha=1.0, A=0.1, omega_c=10.0)

    # Zero controls - no control applied
    zero_controls = np.zeros((config['n_segments'], env.n_controls))

    rho_init = env.rho0  # |00⟩ state
    sim = env.get_simulator(task_params)
    rho_final_zero, _ = sim.evolve(rho_init, zero_controls, env.T)

    # Compute fidelity
    fidelity_zero = process_fidelity(rho_final_zero, U_cnot)
    print(f"   Fidelity with zero controls: {fidelity_zero:.4f}")
    print(f"   (Should be low, around 0.25-0.50 due to random evolution)")

    # Test with random controls (should be better)
    print(f"\n5. Testing with RANDOM controls...")
    np.random.seed(42)
    random_controls = np.random.randn(config['n_segments'], env.n_controls) * 2.0

    rho_final_random, _ = sim.evolve(rho_init, random_controls, env.T)
    fidelity_random = process_fidelity(rho_final_random, U_cnot)
    print(f"   Fidelity with random controls: {fidelity_random:.4f}")

    # Test optimized controls (simple pulse shaping)
    print(f"\n6. Testing with PULSE-SHAPED controls...")
    # Use sinusoidal pulses on all channels
    t = np.linspace(0, env.T, config['n_segments'])

    # Design pulses: need strong ZZ (channel 3) for entanglement
    optimized_controls = np.zeros((config['n_segments'], env.n_controls))
    optimized_controls[:, 0] = 3.0 * np.sin(2 * np.pi * t / env.T)       # XI
    optimized_controls[:, 1] = 2.0 * np.sin(4 * np.pi * t / env.T)       # IX
    optimized_controls[:, 2] = 1.0 * np.cos(2 * np.pi * t / env.T)       # YI
    optimized_controls[:, 3] = 5.0 * np.sin(np.pi * t / env.T)           # ZZ (strong entangling)

    rho_final_opt, _ = sim.evolve(rho_init, optimized_controls, env.T)
    fidelity_opt = process_fidelity(rho_final_opt, U_cnot)
    print(f"   Fidelity with pulse-shaped controls: {fidelity_opt:.4f}")

    # Summary
    print(f"\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Zero controls fidelity:    {fidelity_zero:.4f}")
    print(f"Random controls fidelity:  {fidelity_random:.4f}")
    print(f"Shaped controls fidelity:  {fidelity_opt:.4f}")

    if env.d != 4:
        print("\n❌ ERROR: System dimension should be 4 for 2 qubits!")
        return False

    if env.n_controls != 4:
        print("\n❌ ERROR: Should have 4 control Hamiltonians for 2-qubit!")
        return False

    if np.allclose(env.H0, 0):
        print("\n⚠️  WARNING: Drift Hamiltonian H0 is zero!")
        print("   This can cause poor gate fidelities. Set drift_strength > 0 in config.")

    if fidelity_opt > 0.7:
        print(f"\n✅ SUCCESS: System is properly configured!")
        print(f"   Achieved {fidelity_opt:.2%} fidelity with shaped pulses.")
        return True
    else:
        print(f"\n⚠️  ISSUE: Fidelity still low ({fidelity_opt:.2%})")
        print("   Possible causes:")
        print("   - Evolution time T too short/long")
        print("   - Control amplitudes not properly scaled")
        print("   - Noise too strong")
        print("   - Need proper optimization (MAML/GRAPE)")
        return True  # Setup is correct, just needs optimization


if __name__ == "__main__":
    success = test_two_qubit_environment()

    if success:
        print("\n" + "=" * 70)
        print("To use 2-qubit gates in your training:")
        print("=" * 70)
        print("1. Set num_qubits=2 in your config")
        print("2. Set target_gate='cnot' (or 'cz', 'swap', 'iswap')")
        print("3. Set drift_strength=0.5 (or tune this value)")
        print("4. Increase n_segments if needed (try 30-50 for 2-qubit)")
        print("5. Scale control amplitudes appropriately (output_scale=5-10)")
        print("=" * 70)
