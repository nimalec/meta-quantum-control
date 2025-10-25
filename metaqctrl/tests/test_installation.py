"""
Test basic installation and imports
"""

import pytest
import numpy as np
import torch


def test_imports():
    """Test that all modules can be imported."""
    from src.quantum.lindblad import LindbladSimulator
    from src.quantum.noise_models import NoisePSDModel, TaskDistribution
    from src.quantum.gates import GateFidelityComputer, TargetGates
    from src.meta_rl.policy import PulsePolicy
    from src.meta_rl.maml import MAML
    from src.baselines.robust_control import RobustPolicy
    from src.theory.optimality_gap import OptimalityGapComputer, GapConstants
    
    assert True, "All imports successful"


def test_lindblad_simulator():
    """Test basic Lindblad simulation."""
    from src.quantum.lindblad import LindbladSimulator
    
    # Pauli matrices
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    H0 = 0.0 * sigma_z
    H_controls = [sigma_x, sigma_y]
    L_ops = [0.1 * sigma_z]  # Dephasing
    
    sim = LindbladSimulator(H0, H_controls, L_ops, method='RK45')
    
    # Initial state |0⟩
    rho0 = np.array([[1, 0], [0, 0]], dtype=complex)
    
    # Random controls
    controls = np.random.randn(10, 2) * 0.1
    
    rho_final, _ = sim.evolve(rho0, controls, T=1.0)
    
    # Check properties
    assert rho_final.shape == (2, 2), "Wrong shape"
    assert np.abs(np.trace(rho_final) - 1.0) < 1e-6, "Trace not preserved"
    
    eigenvals = np.linalg.eigvalsh(rho_final)
    assert np.all(eigenvals >= -1e-10), "Not positive semidefinite"


def test_noise_psd():
    """Test PSD model."""
    from src.quantum.noise_models import NoisePSDModel, NoiseParameters
    
    psd_model = NoisePSDModel(model_type='one_over_f')
    
    theta = NoiseParameters(alpha=1.0, A=0.1, omega_c=5.0)
    omega = np.logspace(-1, 2, 100)
    
    S = psd_model.psd(omega, theta)
    
    assert S.shape == omega.shape, "Wrong shape"
    assert np.all(S > 0), "PSD must be positive"
    assert np.all(np.isfinite(S)), "PSD has invalid values"


def test_task_distribution():
    """Test task sampling."""
    from src.quantum.noise_models import TaskDistribution
    
    task_dist = TaskDistribution(
        dist_type='uniform',
        ranges={
            'alpha': (0.5, 2.0),
            'A': (0.05, 0.3),
            'omega_c': (2.0, 8.0)
        }
    )
    
    rng = np.random.default_rng(42)
    tasks = task_dist.sample(10, rng)
    
    assert len(tasks) == 10, "Wrong number of tasks"
    
    for task in tasks:
        assert 0.5 <= task.alpha <= 2.0, "Alpha out of range"
        assert 0.05 <= task.A <= 0.3, "A out of range"
        assert 2.0 <= task.omega_c <= 8.0, "omega_c out of range"


def test_fidelity_computation():
    """Test fidelity measures."""
    from src.quantum.gates import state_fidelity, TargetGates
    
    # Test with pure states
    ket_0 = np.array([1, 0], dtype=complex)
    ket_1 = np.array([0, 1], dtype=complex)
    
    rho_0 = np.outer(ket_0, ket_0.conj())
    rho_1 = np.outer(ket_1, ket_1.conj())
    
    F_same = state_fidelity(rho_0, rho_0)
    F_orthog = state_fidelity(rho_0, rho_1)
    
    assert np.abs(F_same - 1.0) < 1e-10, "Same state fidelity should be 1"
    assert np.abs(F_orthog) < 1e-10, "Orthogonal state fidelity should be 0"
    
    # Test Hadamard gate
    H = TargetGates.hadamard()
    assert H.shape == (2, 2), "Wrong shape"
    assert np.allclose(H @ H.conj().T, np.eye(2)), "Not unitary"


def test_policy_network():
    """Test policy network forward pass."""
    from src.meta_rl.policy import PulsePolicy
    
    policy = PulsePolicy(
        task_feature_dim=3,
        hidden_dim=32,
        n_hidden_layers=2,
        n_segments=10,
        n_controls=2
    )
    
    # Single input
    task_features = torch.randn(3)
    controls = policy(task_features)
    
    assert controls.shape == (10, 2), "Wrong output shape"
    assert torch.all(torch.isfinite(controls)), "Invalid controls"
    
    # Batch input
    batch_features = torch.randn(16, 3)
    batch_controls = policy(batch_features)
    
    assert batch_controls.shape == (16, 10, 2), "Wrong batch shape"


def test_maml_initialization():
    """Test MAML can be initialized."""
    from src.meta_rl.policy import PulsePolicy
    from src.meta_rl.maml import MAML
    
    policy = PulsePolicy(
        task_feature_dim=3,
        hidden_dim=32,
        n_segments=5,
        n_controls=2
    )
    
    maml = MAML(
        policy=policy,
        inner_lr=0.01,
        inner_steps=3,
        meta_lr=0.001,
        first_order=False
    )
    
    assert maml.inner_steps == 3, "Wrong inner steps"
    assert maml.inner_lr == 0.01, "Wrong inner LR"


def test_gap_constants():
    """Test gap bound computation."""
    from src.theory.optimality_gap import GapConstants
    
    constants = GapConstants(
        C_sep=0.5,
        mu=0.1,
        L=1.0,
        L_F=0.5,
        C_K=1.0
    )
    
    gap_bound = constants.gap_lower_bound(sigma_sq=0.1, K=5, eta=0.01)
    
    assert gap_bound >= 0, "Gap bound must be non-negative"
    assert np.isfinite(gap_bound), "Gap bound must be finite"
    
    # Gap should increase with K
    gap_K1 = constants.gap_lower_bound(sigma_sq=0.1, K=1, eta=0.01)
    gap_K10 = constants.gap_lower_bound(sigma_sq=0.1, K=10, eta=0.01)
    
    assert gap_K10 > gap_K1, "Gap should increase with K"


def test_end_to_end_mini():
    """Minimal end-to-end test."""
    from src.quantum.lindblad import LindbladSimulator
    from src.quantum.noise_models import NoiseParameters, NoisePSDModel, PSDToLindblad
    from src.quantum.gates import state_fidelity
    from src.meta_rl.policy import PulsePolicy
    
    # Setup system
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    H0 = 0.0 * sigma_z
    H_controls = [sigma_x, sigma_y]
    
    psd_model = NoisePSDModel(model_type='one_over_f')
    psd_to_lindblad = PSDToLindblad(
        basis_operators=[sigma_x, sigma_y, sigma_z],
        sampling_freqs=np.array([1.0, 5.0, 10.0]),
        psd_model=psd_model
    )
    
    # Create task
    task_params = NoiseParameters(alpha=1.0, A=0.1, omega_c=5.0)
    L_ops = psd_to_lindblad.get_lindblad_operators(task_params)
    
    # Create simulator
    sim = LindbladSimulator(H0, H_controls, L_ops, method='RK45')
    
    # Create policy
    policy = PulsePolicy(
        task_feature_dim=3,
        hidden_dim=16,
        n_segments=5,
        n_controls=2
    )
    
    # Generate controls
    task_features = torch.tensor([1.0, 0.1, 5.0])
    controls = policy(task_features)
    controls_np = controls.detach().numpy()
    
    # Simulate
    rho0 = np.array([[1, 0], [0, 0]], dtype=complex)
    rho_final, _ = sim.evolve(rho0, controls_np, T=0.5)
    
    # Compute fidelity
    target = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)  # |+⟩⟨+|
    fidelity = state_fidelity(rho_final, target)
    
    assert 0.0 <= fidelity <= 1.0, "Fidelity out of range"
    assert np.isfinite(fidelity), "Invalid fidelity"
    
    print(f"End-to-end test passed! Fidelity: {fidelity:.4f}")


if __name__ == "__main__":
    # Run all tests
    print("Running installation tests...\n")
    
    print("1. Testing imports...")
    test_imports()
    print("   ✓ Passed\n")
    
    print("2. Testing Lindblad simulator...")
    test_lindblad_simulator()
    print("   ✓ Passed\n")
    
    print("3. Testing noise PSD...")
    test_noise_psd()
    print("   ✓ Passed\n")
    
    print("4. Testing task distribution...")
    test_task_distribution()
    print("   ✓ Passed\n")
    
    print("5. Testing fidelity computation...")
    test_fidelity_computation()
    print("   ✓ Passed\n")
    
    print("6. Testing policy network...")
    test_policy_network()
    print("   ✓ Passed\n")
    
    print("7. Testing MAML initialization...")
    test_maml_initialization()
    print("   ✓ Passed\n")
    
    print("8. Testing gap constants...")
    test_gap_constants()
    print("   ✓ Passed\n")
    
    print("9. Running end-to-end test...")
    test_end_to_end_mini()
    print("   ✓ Passed\n")
    
    print("=" * 50)
    print("All tests passed! ✓")
    print("=" * 50)
    print("\nInstallation verified successfully.")
    print("\nNext steps:")
    print("1. Run: python experiments/train_meta.py --config configs/experiment_config.yaml")
    print("2. Run: python experiments/train_robust.py --config configs/experiment_config.yaml")
    print("3. Run: python experiments/eval_gap.py --meta_path <path> --robust_path <path>")
