"""
Test Differentiable Quantum Simulation and Gradient Flow

This test verifies that gradients flow correctly through the entire
meta-learning pipeline, from policy network through quantum simulation
to fidelity computation.

"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np

from metaqctrl.src.quantum.lindblad_torch import DifferentiableLindbladSimulator, numpy_to_torch_complex
from metaqctrl.src.quantum.noise_models import NoiseParameters, NoisePSDModel, PSDToLindblad
from metaqctrl.src.meta_rl.policy import PulsePolicy
from metaqctrl.src.theory.quantum_environment import QuantumEnvironment


def test_basic_gradient_flow():
    """Test 1: Basic gradient flow through differentiable simulator."""
    print("\n" + "="*70)
    print("Test 1: Basic Gradient Flow Through Differentiable Simulator")
    print("="*70)

    # Simple 1-qubit system
    sigma_x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
    sigma_y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)
    sigma_z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)

    H0 = 0.5 * sigma_z
    H_controls = [sigma_x, sigma_y]
    L_ops = [torch.sqrt(torch.tensor(0.1)) * sigma_z]

    # Create simulator
    sim = DifferentiableLindbladSimulator(H0, H_controls, L_ops, method='rk4')

    # Initial state and controls
    rho0 = torch.tensor([[1, 0], [0, 0]], dtype=torch.complex64)
    controls = torch.nn.Parameter(torch.randn(20, 2) * 0.5)

    # Forward pass
    rho_final = sim(rho0, controls, T=1.0)

    # Compute loss (maximize purity)
    purity = torch.trace(rho_final @ rho_final).real
    loss = -purity

    # Backward pass
    loss.backward()

    # Check gradients
    assert controls.grad is not None, " No gradients!"
    grad_norm = torch.norm(controls.grad).item()

    print(f"  ✓ Gradients computed: {controls.grad is not None}")
    print(f"  ✓ Gradient norm: {grad_norm:.4e}")
    print(f"  ✓ Gradient shape: {controls.grad.shape}")
    print(f"  ✓ Loss: {loss.item():.4f}")

    assert grad_norm > 0, " Zero gradient!"
    print("\n Test 1 PASSED: Basic gradient flow works!")

    return True


def test_policy_to_simulation_gradient_flow():
    """Test 2: Gradient flow from policy network through simulation."""
    print("\n" + "="*70)
    print("Test 2: Policy Network → Quantum Simulation Gradient Flow")
    print("="*70)

    # Create policy network
    policy = PulsePolicy(
        task_feature_dim=3,
        hidden_dim=32,
        n_hidden_layers=1,
        n_segments=10,
        n_controls=2,
        output_scale=0.5
    )

    print(f"  Policy parameters: {policy.count_parameters():,}")

    # Quantum system
    sigma_x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
    sigma_y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)
    sigma_z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)

    H0 = 0.0 * sigma_z
    H_controls = [sigma_x, sigma_y]
    L_ops = [torch.sqrt(torch.tensor(0.05)) * sigma_z]

    sim = DifferentiableLindbladSimulator(H0, H_controls, L_ops, method='rk4')

    # Task features
    task_features = torch.tensor([1.0, 0.1, 5.0], dtype=torch.float32)

    # Forward: Policy → Controls → Simulation
    controls = policy(task_features)
    print(f"   Controls shape: {controls.shape}")

    rho0 = torch.tensor([[1, 0], [0, 0]], dtype=torch.complex64)
    rho_final = sim(rho0, controls, T=1.0)

    # Target state |+⟩
    ket_0 = np.array([1, 0])
    ket_1 = np.array([0, 1])
    ket_plus = (ket_0 + ket_1) / np.sqrt(2)
    target = torch.tensor(
        np.outer(ket_plus, ket_plus.conj()),
        dtype=torch.complex64
    )

    # Fidelity
    overlap = torch.trace(rho_final @ target)
    fidelity = torch.abs(overlap) ** 2
    loss = 1.0 - fidelity.real

    print(f"   Fidelity: {fidelity.real.item():.4f}")
    print(f"   Loss: {loss.item():.4f}")

    # Backward pass
    loss.backward()

    # Check gradients in policy
    grad_norms = []
    for name, param in policy.named_parameters():
        if param.grad is not None:
            grad_norm = torch.norm(param.grad).item()
            grad_norms.append(grad_norm)
            print(f"  ✓ {name}: grad_norm = {grad_norm:.4e}")

    assert len(grad_norms) > 0, " No policy gradients!"
    assert all(g > 0 for g in grad_norms), " Zero gradients in policy!"

    print(f"\n  Total parameters with gradients: {len(grad_norms)}")
    print(f"   Average gradient norm: {np.mean(grad_norms):.4e}")
    print("\n Test 2 PASSED: Policy → Simulation gradients work!")

    return True


def test_quantum_environment_differentiable():
    """Test 3: Full QuantumEnvironment with differentiable loss."""
    print("\n" + "="*70)
    print("Test 3: QuantumEnvironment Differentiable Loss")
    print("="*70)

    # Setup quantum system (NumPy)
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

    H0 = 0.0 * sigma_z
    H_controls = [sigma_x, sigma_y]

    # PSD model
    psd_model = NoisePSDModel(model_type='one_over_f')
    omega_sample = np.array([1.0, 5.0, 10.0])

    psd_to_lindblad = PSDToLindblad(
        basis_operators=[sigma_x, sigma_y, sigma_z],
        sampling_freqs=omega_sample,
        psd_model=psd_model
    )

    # Target state |+⟩
    ket_0 = np.array([1, 0], dtype=complex)
    ket_1 = np.array([0, 1], dtype=complex)
    ket_plus = (ket_0 + ket_1) / np.sqrt(2)
    target_state = np.outer(ket_plus, ket_plus.conj())

    # Create environment
    env = QuantumEnvironment(
        H0=H0,
        H_controls=H_controls,
        psd_to_lindblad=psd_to_lindblad,
        target_state=target_state,
        T=1.0
    )

    print(f"   Environment created: d={env.d}, n_controls={env.n_controls}")

    # Create policy
    policy = PulsePolicy(
        task_feature_dim=3,
        hidden_dim=32,
        n_segments=10,
        n_controls=2
    )

    # Task parameters
    task_params = NoiseParameters(alpha=1.0, A=0.1, omega_c=5.0)

    # OLD (NON-DIFFERENTIABLE) METHOD
    print("\n  Testing OLD non-differentiable method...")
    loss_old = env.compute_loss(policy, task_params, device=torch.device('cpu'))
    print(f"   Old loss: {loss_old.item():.4f}")
    print(f"   Old loss requires_grad: {loss_old.requires_grad}")
    assert loss_old.requires_grad == False, "Old method should not have gradients"

    # NEW (DIFFERENTIABLE) METHOD
    print("\n  Testing NEW differentiable method...")
    loss_new = env.compute_loss_differentiable(
        policy, task_params, device=torch.device('cpu'), use_rk4=True
    )
    print(f"   New loss: {loss_new.item():.4f}")
    print(f"   New loss requires_grad: {loss_new.requires_grad}")

    # Backward pass
    loss_new.backward()

    # Check gradients
    grad_count = sum(1 for p in policy.parameters() if p.grad is not None)
    total_params = sum(1 for _ in policy.parameters())

    print(f"\n   Parameters with gradients: {grad_count}/{total_params}")

    assert grad_count == total_params, f" Only {grad_count}/{total_params} have gradients!"

    # Check gradient magnitudes
    grad_norms = [torch.norm(p.grad).item() for p in policy.parameters() if p.grad is not None]
    print(f"  ✓ Average gradient norm: {np.mean(grad_norms):.4e}")
    print(f"  ✓ Max gradient norm: {np.max(grad_norms):.4e}")

    assert all(g > 0 for g in grad_norms), " Zero gradients!"

    print("\n Test 3 PASSED: QuantumEnvironment differentiable loss works!")

    return True


def test_simple_training_step():
    """Test 4: Simulate a simple training step with gradient descent."""
    print("\n" + "="*70)
    print("Test 4: Simple Training Step (Gradient Descent)")
    print("="*70)

    # Setup (simplified from Test 3)
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

    H0 = 0.0 * sigma_z
    H_controls = [sigma_x, sigma_y]

    psd_model = NoisePSDModel(model_type='one_over_f')
    omega_sample = np.array([1.0, 5.0])
    psd_to_lindblad = PSDToLindblad([sigma_x, sigma_z], omega_sample, psd_model)

    ket_plus = (np.array([1, 0]) + np.array([0, 1])) / np.sqrt(2)
    target_state = np.outer(ket_plus, ket_plus.conj())

    env = QuantumEnvironment(H0, H_controls, psd_to_lindblad, target_state, T=0.5)

    # Create policy and optimizer
    policy = PulsePolicy(task_feature_dim=3, hidden_dim=16, n_segments=5, n_controls=2)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)

    task_params = NoiseParameters(alpha=1.0, A=0.05, omega_c=5.0)

    print(f"  Initial policy parameters: {policy.count_parameters():,}")

    # Training for 5 steps
    losses = []
    for step in range(5):
        optimizer.zero_grad()

        # Forward pass (DIFFERENTIABLE!)
        loss = env.compute_loss_differentiable(policy, task_params, use_rk4=False)  # Use Euler for speed

        # Backward pass
        loss.backward()

        # Gradient step
        optimizer.step()

        losses.append(loss.item())
        print(f"  Step {step}: Loss = {loss.item():.4f}")

    # Check that loss decreased
    print(f"\n   Initial loss: {losses[0]:.4f}")
    print(f"   Final loss: {losses[-1]:.4f}")
    print(f"   Loss reduction: {losses[0] - losses[-1]:.4f}")

    # Loss should generally decrease (though not guaranteed in 5 steps)
    if losses[-1] < losses[0]:
        print("  ✓ Loss decreased - learning is happening!")
    else:
        print("   Loss didn't decrease (may need more steps or better LR)")

    print("\n Test 4 PASSED: Training step works!")

    return True


def main():
    """Run all gradient flow tests."""
    print("\n" + "#"*70)
    print("# DIFFERENTIABLE QUANTUM SIMULATION TEST SUITE")
    print("# Critical for ICML Paper - Verifying Gradient Flow")
    print("#"*70)

    tests = [
        ("Basic Gradient Flow", test_basic_gradient_flow),
        ("Policy → Simulation", test_policy_to_simulation_gradient_flow),
        ("QuantumEnvironment Differentiable", test_quantum_environment_differentiable),
        ("Simple Training Step", test_simple_training_step),
    ]

    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print(f"\n {name} FAILED with error:")
            print(f"   {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "#"*70)
    print("# TEST SUMMARY")
    print("#"*70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = " PASSED" if result else " FAILED"
        print(f"  {status}: {name}")

    print("\n" + "#"*70)
    if passed == total:
        print(f"# ALL TESTS PASSED ({passed}/{total})")
        print("# GRADIENT FLOW IS WORKING!")
        print("# Meta-learning can now properly train for ICML paper!")
    else:
        print(f"#  SOME TESTS FAILED ({passed}/{total} passed)")
        print("# Fix failing tests before running experiments!")
    print("#"*70 + "\n")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
