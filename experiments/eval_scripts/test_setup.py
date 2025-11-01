"""
Quick Setup Test

Run this script to verify that all dependencies are installed and
the codebase is properly set up for running the paper experiments.

This should complete in ~10 seconds.
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")

    required_modules = [
        ('numpy', 'NumPy'),
        ('torch', 'PyTorch'),
        ('matplotlib', 'Matplotlib'),
        ('scipy', 'SciPy'),
    ]

    failed = []

    for module_name, display_name in required_modules:
        try:
            __import__(module_name)
            print(f"  ✓ {display_name}")
        except ImportError:
            print(f"  ✗ {display_name} - NOT INSTALLED")
            failed.append(display_name)

    # Test project imports
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    project_modules = [
        ('metaqctrl.quantum.noise_models', 'Noise Models'),
        ('metaqctrl.theory.quantum_environment', 'Quantum Environment'),
        ('metaqctrl.theory.optimality_gap', 'Optimality Gap'),
        ('metaqctrl.meta_rl.maml', 'MAML'),
        ('metaqctrl.meta_rl.policy', 'Policy'),
    ]

    for module_name, display_name in project_modules:
        try:
            __import__(module_name)
            print(f"  ✓ {display_name}")
        except ImportError as e:
            print(f"  ✗ {display_name} - IMPORT ERROR")
            print(f"    {e}")
            failed.append(display_name)

    return len(failed) == 0, failed


def test_noise_models():
    """Test noise model functionality."""
    print("\nTesting noise models...")

    try:
        from metaqctrl.quantum.noise_models import (
            NoiseParameters, NoisePSDModel, PSDToLindblad
        )
        import numpy as np

        # Test PSD computation
        theta = NoiseParameters(alpha=1.0, A=0.1, omega_c=5.0)
        psd_model = NoisePSDModel(model_type='one_over_f')
        omega = np.array([1.0, 5.0, 10.0])
        S = psd_model.psd(omega, theta)

        assert not np.isnan(S).any(), "PSD contains NaN"
        assert not np.isinf(S).any(), "PSD contains Inf"

        # Test integration
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        psd_to_lindblad = PSDToLindblad(
            basis_operators=[sigma_x],
            sampling_freqs=np.linspace(0, 20, 10),
            psd_model=psd_model,
            integration_method='trapz'
        )
        L_ops = psd_to_lindblad.get_lindblad_operators(theta)

        assert len(L_ops) == 1, "Wrong number of Lindblad operators"

        print("  ✓ Noise models working correctly")
        return True

    except Exception as e:
        print(f"  ✗ Noise models test failed: {e}")
        return False


def test_maml():
    """Test MAML functionality."""
    print("\nTesting MAML...")

    try:
        import torch
        from metaqctrl.meta_rl.maml import MAML
        from metaqctrl.meta_rl.policy import QuantumControlPolicy

        policy = QuantumControlPolicy(
            task_dim=3,
            hidden_dim=16,
            n_segments=10,
            n_controls=2
        )

        maml = MAML(
            policy=policy,
            inner_lr=0.01,
            inner_steps=2,
            meta_lr=0.001,
            first_order=True
        )

        # Test that meta_train_step can be called
        # (won't actually train without data)

        print("  ✓ MAML initialized correctly")
        return True

    except Exception as e:
        print(f"  ✗ MAML test failed: {e}")
        return False


def test_gap_computer():
    """Test optimality gap computation."""
    print("\nTesting gap computer...")

    try:
        from metaqctrl.theory.quantum_environment import QuantumEnvironment
        from metaqctrl.theory.optimality_gap import OptimalityGapComputer

        config = {
            'n_segments': 10,
            'horizon': 1.0,
            'target_gate': 'X',
            'psd_model': 'one_over_f',
            'method': 'rk4'
        }

        env = QuantumEnvironment(config)

        gap_computer = OptimalityGapComputer(
            env=env,
            C_sep=0.1,
            L=1.0,
            L_F=1.0,
            mu=0.1
        )

        print("  ✓ Gap computer initialized correctly")
        return True

    except Exception as e:
        print(f"  ✗ Gap computer test failed: {e}")
        return False


def check_results_directory():
    """Check that results directory can be created."""
    print("\nChecking results directory...")

    try:
        results_dir = Path(__file__).parent.parent / 'paper_results'
        results_dir.mkdir(parents=True, exist_ok=True)

        test_file = results_dir / 'test.txt'
        test_file.write_text('test')
        test_file.unlink()

        print(f"  ✓ Results directory accessible: {results_dir}")
        return True

    except Exception as e:
        print(f"  ✗ Cannot create results directory: {e}")
        return False


def main():
    print("="*70)
    print("SETUP VERIFICATION")
    print("="*70)
    print()

    results = {}

    results['Imports'], failed_imports = test_imports()
    results['Noise Models'] = test_noise_models()
    results['MAML'] = test_maml()
    results['Gap Computer'] = test_gap_computer()
    results['Results Directory'] = check_results_directory()

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "="*70)

    if all_passed:
        print("✓ SETUP COMPLETE")
        print()
        print("You can now run the paper experiments:")
        print("  python run_all_paper_experiments.py")
        print()
        print("Or run individual experiments:")
        print("  python validate_psd_integration.py")
        print("  python validate_maml_fixes.py")
        print("  python validate_gap_scaling.py")
    else:
        print("✗ SETUP INCOMPLETE")
        print()
        print("Please fix the failed tests above before running experiments.")

        if not results['Imports']:
            print()
            print("Missing dependencies. Install with:")
            print("  pip install numpy torch matplotlib scipy")

    print("="*70)


if __name__ == "__main__":
    main()
