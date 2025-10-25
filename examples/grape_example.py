"""
Example: Using GRAPE for Quantum Control Optimization

This script demonstrates how to use the GRAPE (Gradient Ascent Pulse Engineering)
baseline for optimizing quantum control pulses.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

from metaqctrl.baselines.robust_control import GRAPEOptimizer
from metaqctrl.quantum.lindblad import LindbladSimulator
from metaqctrl.quantum.noise_models import NoiseParameters, NoisePSDModel, PSDToLindblad
from metaqctrl.quantum.gates import state_fidelity, TargetGates


def create_quantum_system():
    """Create a simple 1-qubit quantum system."""
    # Pauli matrices
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

    # System Hamiltonians
    H0 = 0.0 * sigma_z  # No drift
    H_controls = [sigma_x, sigma_y]

    # PSD model for noise
    psd_model = NoisePSDModel(model_type='one_over_f')
    omega_sample = np.array([1.0, 5.0, 10.0])

    psd_to_lindblad = PSDToLindblad(
        basis_operators=[sigma_x, sigma_y, sigma_z],
        sampling_freqs=omega_sample,
        psd_model=psd_model
    )

    return H0, H_controls, psd_to_lindblad


def main():
    print("="*70)
    print("GRAPE Optimization for Quantum Control")
    print("="*70)

    # Setup quantum system
    H0, H_controls, psd_to_lindblad = create_quantum_system()

    # Target state: |+⟩ = (|0⟩ + |1⟩)/√2
    ket_0 = np.array([1, 0], dtype=complex)
    ket_1 = np.array([0, 1], dtype=complex)
    ket_plus = (ket_0 + ket_1) / np.sqrt(2)
    target_state = np.outer(ket_plus, ket_plus.conj())

    # Initial state: |0⟩
    rho0 = np.outer(ket_0, ket_0.conj())

    # Task parameters (noise environment)
    task_params = NoiseParameters(alpha=1.0, A=0.1, omega_c=5.0)

    print("\nQuantum System Setup:")
    print(f"  Target: |+⟩ state")
    print(f"  Initial: |0⟩ state")
    print(f"  Noise: 1/f noise (α={task_params.alpha}, A={task_params.A}, ωc={task_params.omega_c})")

    # Define simulation function for GRAPE
    def simulate_fidelity(controls, task_params):
        """Simulate quantum evolution and return fidelity."""
        # Get Lindblad operators for this task
        L_ops = psd_to_lindblad.get_lindblad_operators(task_params)

        # Create simulator
        sim = LindbladSimulator(
            H0=H0,
            H_controls=H_controls,
            L_operators=L_ops,
            method='RK45'
        )

        # Evolve
        T = 1.0
        rho_final, _ = sim.evolve(rho0, controls, T)

        # Compute fidelity
        fidelity = state_fidelity(rho_final, target_state)

        return fidelity

    # Initialize GRAPE optimizer
    print("\nInitializing GRAPE optimizer...")
    grape = GRAPEOptimizer(
        n_segments=20,
        n_controls=2,
        T=1.0,
        control_bounds=(-3.0, 3.0),
        learning_rate=0.1,
        method='adam'
    )

    print(f"  Optimization method: Adam")
    print(f"  Number of segments: 20")
    print(f"  Number of controls: 2")
    print(f"  Control bounds: [-3.0, 3.0]")

    # Run GRAPE optimization
    print("\n" + "-"*70)
    print("Running GRAPE optimization (single task)...")
    print("-"*70)

    optimal_controls = grape.optimize(
        simulate_fn=simulate_fidelity,
        task_params=task_params,
        max_iterations=50,
        tolerance=1e-6,
        verbose=True
    )

    # Evaluate final fidelity
    final_fidelity = simulate_fidelity(optimal_controls, task_params)

    print("\n" + "="*70)
    print("OPTIMIZATION RESULTS")
    print("="*70)
    print(f"Final Fidelity: {final_fidelity:.6f}")
    print(f"Control shape: {optimal_controls.shape}")
    print(f"Control range: [{optimal_controls.min():.3f}, {optimal_controls.max():.3f}]")

    # Plot results
    print("\nGenerating plots...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Plot 1: Fidelity history
    ax = axes[0, 0]
    ax.plot(grape.fidelity_history, 'b-', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Fidelity')
    ax.set_title('GRAPE Convergence')
    ax.grid(True, alpha=0.3)

    # Plot 2: Optimal control pulses
    ax = axes[0, 1]
    time_points = np.linspace(0, 1.0, grape.n_segments)
    ax.plot(time_points, optimal_controls[:, 0], 'r-', label='Control X', linewidth=2)
    ax.plot(time_points, optimal_controls[:, 1], 'b-', label='Control Y', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Control Amplitude')
    ax.set_title('Optimized Control Pulses')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Gradient norms
    if grape.gradient_norms:
        ax = axes[1, 0]
        ax.semilogy(grape.gradient_norms, 'g-', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Gradient Norm')
        ax.set_title('Gradient Magnitude')
        ax.grid(True, alpha=0.3)

    # Plot 4: Control amplitude distribution
    ax = axes[1, 1]
    ax.hist(optimal_controls.flatten(), bins=20, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Control Amplitude')
    ax.set_ylabel('Frequency')
    ax.set_title('Control Amplitude Distribution')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('grape_optimization_results.png', dpi=150, bbox_inches='tight')
    print("Plot saved to: grape_optimization_results.png")

    # Test robust GRAPE
    print("\n" + "="*70)
    print("Testing Robust GRAPE (multiple tasks)...")
    print("="*70)

    # Create task distribution
    task_distribution = [
        NoiseParameters(alpha=0.8, A=0.05, omega_c=4.0),
        NoiseParameters(alpha=1.0, A=0.10, omega_c=5.0),
        NoiseParameters(alpha=1.2, A=0.15, omega_c=6.0),
    ]

    print(f"\nOptimizing over {len(task_distribution)} tasks...")

    # Reset GRAPE
    grape.reset()

    robust_controls = grape.optimize_robust(
        simulate_fn=simulate_fidelity,
        task_distribution=task_distribution,
        max_iterations=30,
        robust_type='average',
        verbose=True
    )

    print("\n" + "="*70)
    print("GRAPE Example Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
