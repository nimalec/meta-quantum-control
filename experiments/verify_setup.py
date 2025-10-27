#!/usr/bin/env python
"""
Verification Script: Check Codebase Setup

Run this script to verify all fixes have been applied correctly.
"""

import os
import sys
from pathlib import Path
import importlib.util

def check_mark(condition, message):
    """Print check mark or X based on condition."""
    symbol = "✓" if condition else "✗"
    status = "PASS" if condition else "FAIL"
    print(f"  [{symbol}] {message:60s} {status}")
    return condition

def main():
    print("=" * 80)
    print("Meta Quantum Control - Setup Verification")
    print("=" * 80)

    all_passed = True

    # Check 1: Directory Structure
    print("\n[1] Directory Structure:")
    checkpoints_exists = os.path.isdir("checkpoints")
    all_passed &= check_mark(checkpoints_exists, "checkpoints/ directory exists")

    paper_results_exists = os.path.isdir("paper_results")
    all_passed &= check_mark(paper_results_exists, "paper_results/ directory exists")

    # Check 2: Required Scripts
    print("\n[2] Required Scripts:")
    scripts = [
        "train_meta.py",
        "train_robust.py",
        "train_grape.py",
        "paper_results/experiment_gap_vs_k.py",
        "paper_results/experiment_gap_vs_variance.py",
        "paper_results/generate_all_results.py"
    ]

    for script in scripts:
        exists = os.path.isfile(script)
        all_passed &= check_mark(exists, f"{script} exists")

    # Check 3: Documentation
    print("\n[3] Documentation:")
    docs = [
        "GRAPE_USAGE.md",
        "ISSUES_FIXED.md",
        "paper_results/README.md"
    ]

    for doc in docs:
        exists = os.path.isfile(doc)
        check_mark(exists, f"{doc} exists")

    # Check 4: Import Tests
    print("\n[4] Python Imports:")

    # Add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    try:
        from metaqctrl.baselines.robust_control import GRAPEOptimizer, RobustPolicy
        all_passed &= check_mark(True, "Import GRAPEOptimizer")
    except ImportError as e:
        all_passed &= check_mark(False, f"Import GRAPEOptimizer (Error: {e})")

    try:
        from metaqctrl.meta_rl.policy import PulsePolicy
        all_passed &= check_mark(True, "Import PulsePolicy")
    except ImportError as e:
        all_passed &= check_mark(False, f"Import PulsePolicy (Error: {e})")

    try:
        from metaqctrl.meta_rl.maml import MAML
        all_passed &= check_mark(True, "Import MAML")
    except ImportError as e:
        all_passed &= check_mark(False, f"Import MAML (Error: {e})")

    try:
        from metaqctrl.quantum.noise_models import TaskDistribution
        all_passed &= check_mark(True, "Import TaskDistribution")
    except ImportError as e:
        all_passed &= check_mark(False, f"Import TaskDistribution (Error: {e})")

    # Check 5: Function Signatures
    print("\n[5] Function Signature Checks:")

    try:
        # Load train_meta module
        spec = importlib.util.spec_from_file_location("train_meta", "train_meta.py")
        train_meta = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(train_meta)

        # Check data_generator signature
        import inspect
        sig = inspect.signature(train_meta.data_generator)
        params = list(sig.parameters.keys())

        has_n_trajectories = 'n_trajectories' in params
        all_passed &= check_mark(has_n_trajectories, "data_generator has n_trajectories parameter")

        correct_order = params == ['task_params', 'n_trajectories', 'split', 'quantum_system', 'config', 'device']
        all_passed &= check_mark(correct_order, "data_generator parameters in correct order")

    except Exception as e:
        all_passed &= check_mark(False, f"Check data_generator signature (Error: {e})")

    # Check 6: Config Consistency
    print("\n[6] Config Key Consistency:")

    try:
        # Check generate_all_results.py
        with open("paper_results/generate_all_results.py", 'r') as f:
            content = f.read()

        # Check for correct config keys in the config dict (not comments)
        import re
        config_match = re.search(r"config = \{([^}]+)\}", content, re.DOTALL)
        if config_match:
            config_str = config_match.group(1)
            has_n_segments = "'n_segments': 20" in config_str
            has_n_controls = "'n_controls': 2" in config_str
            has_horizon = "'horizon':" in config_str
        else:
            has_n_segments = has_n_controls = has_horizon = False

        all_passed &= check_mark(has_n_segments, "generate_all_results.py config uses 'n_segments': 20")
        all_passed &= check_mark(has_n_controls, "generate_all_results.py config uses 'n_controls': 2")
        all_passed &= check_mark(has_horizon, "generate_all_results.py config uses 'horizon'")

    except Exception as e:
        all_passed &= check_mark(False, f"Check config consistency (Error: {e})")

    # Check 7: GRAPE Implementation
    print("\n[7] GRAPE Implementation:")

    try:
        from metaqctrl.baselines.robust_control import GRAPEOptimizer

        # Check GRAPE has required methods
        has_optimize = hasattr(GRAPEOptimizer, 'optimize')
        all_passed &= check_mark(has_optimize, "GRAPEOptimizer.optimize() method exists")

        has_optimize_robust = hasattr(GRAPEOptimizer, 'optimize_robust')
        check_mark(has_optimize_robust, "GRAPEOptimizer.optimize_robust() method exists")

        # Check initialization parameters
        import inspect
        sig = inspect.signature(GRAPEOptimizer.__init__)
        params = list(sig.parameters.keys())

        has_n_segments = 'n_segments' in params
        all_passed &= check_mark(has_n_segments, "GRAPEOptimizer.__init__ has n_segments parameter")

        has_n_controls = 'n_controls' in params
        all_passed &= check_mark(has_n_controls, "GRAPEOptimizer.__init__ has n_controls parameter")

    except Exception as e:
        all_passed &= check_mark(False, f"Check GRAPE implementation (Error: {e})")

    # Summary
    print("\n" + "=" * 80)
    if all_passed:
        print("✓✓✓ ALL CHECKS PASSED ✓✓✓")
        print("\nYour setup is ready! Next steps:")
        print("  1. Train policies: python train_meta.py && python train_robust.py")
        print("  2. Run experiments: cd paper_results && python generate_all_results.py")
    else:
        print("✗✗✗ SOME CHECKS FAILED ✗✗✗")
        print("\nPlease review the failed checks above and:")
        print("  1. Ensure you're in the experiments/ directory")
        print("  2. Check that all dependencies are installed")
        print("  3. Verify the codebase is up to date")
    print("=" * 80)

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
