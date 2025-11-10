"""
Run All Paper Experiments

Master script that runs all validation experiments and generates figures/tables for the paper.

This script:
1. Validates PSD to Lindblad integration fix
2. Validates MAML bug fixes
3. Validates adaptation gap vs variance scaling
4. Generates all paper figures and tables

Run this script to reproduce all results.
"""

import sys
import subprocess
from pathlib import Path
import time

def run_script(script_name, description):
    """Run a validation script and report results."""
    print("\n" + "="*80)
    print(f"Running: {description}")
    print("="*80)

    script_path = Path(__file__).parent / script_name

    if not script_path.exists():
        print(f"✗ Script not found: {script_path}")
        return False

    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        elapsed = time.time() - start_time

        print(result.stdout)

        if result.returncode == 0:
            print(f"✓ Completed in {elapsed:.1f}s")
            return True
        else:
            print(f"✗ Failed with return code {result.returncode}")
            print("Error output:")
            print(result.stderr)
            return False

    except subprocess.TimeoutExpired:
        print(f"✗ Timed out after 10 minutes")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    print("="*80)
    print("RUNNING ALL PAPER EXPERIMENTS")
    print("="*80)
    print()
    print("This will run all validation scripts and generate figures/tables.")
    print("Expected runtime: ~10-20 minutes")
    print()

    results = {}

    # Experiment 1: PSD Integration
    results['PSD Integration'] = run_script(
        'validate_psd_integration.py',
        'Experiment 1: PSD to Lindblad Integration Fix'
    )

    # Experiment 2: MAML Fixes
    results['MAML Validation'] = run_script(
        'validate_maml_fixes.py',
        'Experiment 2: MAML Bug Fixes Validation'
    )

    # Experiment 3: Gap Scaling
    results['Gap Scaling'] = run_script(
        'validate_gap_scaling.py',
        'Experiment 3: Adaptation Gap vs Variance Scaling'
    )

    # Summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)

    all_passed = True
    for experiment, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {experiment}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "="*80)

    if all_passed:
        print("✓ ALL EXPERIMENTS COMPLETED SUCCESSFULLY")
        print()
        print("Results are saved in:")
        print("  experiments/paper_results/psd_integration/")
        print("  experiments/paper_results/maml_validation/")
        print("  experiments/paper_results/gap_scaling/")
        print()
        print("Figures generated:")
        print("  - psd_integration_comparison.png")
        print("  - bandwidth_dependence.png")
        print("  - training_curves.png")
        print("  - gap_vs_param_variance.png")
        print("  - gap_vs_control_variance.png")
        print("  - gap_vs_K.png")
        print("  - combined_gap_scaling.png")
        print()
        print("LaTeX tables printed in the output above.")
    else:
        print("⚠ SOME EXPERIMENTS FAILED")
        print("Review the output above for details.")

    print("="*80)


if __name__ == "__main__":
    main()
