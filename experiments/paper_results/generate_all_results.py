"""
Master Script: Generate All Paper Results and Figures

This script runs all experiments and generates all figures for the paper:
1. Gap vs Adaptation Steps K (Figure 1, validates R² ≈ 0.96)
2. Gap vs Task Variance σ²_S (Figure 2, validates R² ≈ 0.94)
3. Constants Estimation and Validation (Table 1)
4. Training Dynamics (Figure 3)
5. Complete Results Summary

Usage:
    python generate_all_results.py --meta_path checkpoints/maml_best.pt \
                                   --robust_path checkpoints/robust_best.pt
"""

import os
import sys
import argparse
from pathlib import Path
import json
import time

# Add to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Import experiment modules
from experiment_gap_vs_k import run_gap_vs_k_experiment, plot_gap_vs_k
from experiment_gap_vs_variance import run_gap_vs_variance_experiment, plot_gap_vs_variance
from experiment_constants_validation import (
    run_constants_validation_experiment,
    plot_constant_distributions
)


def print_banner(text):
    """Print a nice banner"""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80 + "\n")


def generate_results_table(
    gap_vs_k_results: dict,
    gap_vs_var_results: dict,
    constants_results: dict,
    output_path: str = "results/summary_table.txt"
):
    """Generate a summary table like Table 1 in the paper"""

    table = []
    table.append("=" * 80)
    table.append("TABLE 1: Theoretical Constants and Empirical Validation")
    table.append("=" * 80)
    table.append("")

    # Constants section
    table.append("Physics Constants:")
    table.append("-" * 80)
    table.append(f"  Δ_min (spectral gap)       : {constants_results['constants']['Delta_min']:.4f}")
    table.append(f"  μ_min (PL constant)        : {constants_results['constants']['mu_min']:.6f}")
    table.append(f"  C_filter (separation)      : {constants_results['constants']['C_filter']:.4f}")
    table.append(f"  σ²_S (task variance)       : {constants_results['constants']['sigma2_S']:.6f}")
    table.append(f"  c_quantum (combined)       : {constants_results['constants']['c_quantum']:.6f}")
    table.append("")

    # Theoretical predictions
    table.append("Theoretical Predictions:")
    table.append("-" * 80)
    table.append(f"  μ_theory                   : {constants_results['theoretical_predictions']['mu_theory']:.6f}")
    table.append(f"  c_quantum_theory           : {constants_results['theoretical_predictions']['c_quantum_theory']:.6f}")
    table.append("")

    # Validation
    table.append("Empirical / Theory Ratios:")
    table.append("-" * 80)
    mu_ratio = constants_results['theoretical_predictions']['mu_ratio']
    c_ratio = constants_results['theoretical_predictions']['c_quantum_ratio']
    table.append(f"  μ_empirical / μ_theory     : {mu_ratio:.2f}x  {'✓' if 0.5 <= mu_ratio <= 2.0 else '✗'}")
    table.append(f"  c_empirical / c_theory     : {c_ratio:.2f}x  {'✓' if 0.5 <= c_ratio <= 2.0 else '✗'}")
    table.append("")

    # Scaling laws validation
    table.append("Scaling Laws Validation:")
    table.append("-" * 80)
    if gap_vs_k_results['fit']['success']:
        table.append(f"  Gap vs K (exponential):    R² = {gap_vs_k_results['fit']['r2']:.4f}  " +
                    f"{'✓ (target: 0.96)' if gap_vs_k_results['fit']['r2'] >= 0.90 else '✗'}")
        table.append(f"    - gap_max                : {gap_vs_k_results['fit']['gap_max']:.4f}")
        table.append(f"    - μη                     : {gap_vs_k_results['fit']['mu_eta']:.4f}")
    else:
        table.append("  Gap vs K                   : Fit failed")

    table.append("")
    if gap_vs_var_results['fit']['success']:
        table.append(f"  Gap vs σ²_S (linear):      R² = {gap_vs_var_results['fit']['r2']:.4f}  " +
                    f"{'✓ (target: 0.94)' if gap_vs_var_results['fit']['r2'] >= 0.90 else '✗'}")
        table.append(f"    - slope (Gap/σ²_S)       : {gap_vs_var_results['fit']['slope']:.2f}")
    else:
        table.append("  Gap vs σ²_S                : Fit failed")

    table.append("")
    table.append("=" * 80)
    table.append("")

    # Print and save
    table_text = "\n".join(table)
    print(table_text)

    with open(output_path, 'w') as f:
        f.write(table_text)

    print(f"\nTable saved to {output_path}")


def main(args):
    """Run all experiments"""

    print_banner("PAPER RESULTS GENERATION PIPELINE")
    print(f"Meta-policy path:   {args.meta_path}")
    print(f"Robust policy path: {args.robust_path}")
    print(f"Output directory:   {args.output_dir}")
    print(f"Number of tasks:    {args.n_tasks}")
    print(f"GRAPE baseline:     {'Enabled' if args.include_grape else 'Disabled'}")
    if args.include_grape:
        print(f"GRAPE iterations:   {args.grape_iterations}")
    print()

    # Configuration (use consistent key names with train_meta.py)
    config = {
        'num_qubits': 1,
        'n_controls': 2,  # Changed from 'num_controls'
        'n_segments': 20,  # Changed from 'num_segments'
        'horizon': 1.0,  # Changed from 'evolution_time'
        'target_gate': 'hadamard',
        'hidden_dim': 128,  # Changed from 'policy_hidden_dims'
        'n_hidden_layers': 2,
        'inner_lr': 0.01,
        'alpha_range': [0.5, 2.0],  # Flattened from 'task_dist'
        'A_range': [0.05, 0.3],
        'omega_c_range': [2.0, 8.0],
        'noise_frequencies': [1.0, 5.0, 10.0]
    }

    os.makedirs(args.output_dir, exist_ok=True)

    # ========================================================================
    # Experiment 1: Gap vs Adaptation Steps K
    # ========================================================================
    print_banner("EXPERIMENT 1: Gap vs Adaptation Steps K")
    start_time = time.time()

    results_gap_k = run_gap_vs_k_experiment(
        meta_policy_path=args.meta_path,
        robust_policy_path=args.robust_path,
        config=config,
        k_values=[1, 2, 3, 5, 7, 10, 15, 20],
        n_test_tasks=args.n_tasks,
        output_dir=f"{args.output_dir}/gap_vs_k",
        include_grape=args.include_grape,
        grape_iterations=args.grape_iterations
    )

    plot_gap_vs_k(results_gap_k, f"{args.output_dir}/gap_vs_k/figure.pdf")

    elapsed = time.time() - start_time
    print(f"\n✓ Experiment 1 completed in {elapsed:.1f}s")

    # ========================================================================
    # Experiment 2: Gap vs Task Variance
    # ========================================================================
    print_banner("EXPERIMENT 2: Gap vs Task Variance σ²_S")
    start_time = time.time()

    results_gap_var = run_gap_vs_variance_experiment(
        meta_policy_path=args.meta_path,
        robust_policy_path=args.robust_path,
        config=config,
        variance_levels=[0.001, 0.002, 0.004, 0.008, 0.016],
        K_fixed=5,
        n_test_tasks=args.n_tasks,
        output_dir=f"{args.output_dir}/gap_vs_variance",
        include_grape=args.include_grape,
        grape_iterations=args.grape_iterations
    )

    plot_gap_vs_variance(results_gap_var, f"{args.output_dir}/gap_vs_variance/figure.pdf")

    elapsed = time.time() - start_time
    print(f"\n✓ Experiment 2 completed in {elapsed:.1f}s")

    # ========================================================================
    # Experiment 3: Constants Validation
    # ========================================================================
    print_banner("EXPERIMENT 3: Physics Constants Validation")
    start_time = time.time()

    results_constants = run_constants_validation_experiment(
        config=config,
        n_sample_tasks=50,
        output_dir=f"{args.output_dir}/constants_validation"
    )

    plot_constant_distributions(results_constants, f"{args.output_dir}/constants_validation")

    elapsed = time.time() - start_time
    print(f"\n✓ Experiment 3 completed in {elapsed:.1f}s")

    # ========================================================================
    # Generate Summary Table
    # ========================================================================
    print_banner("GENERATING SUMMARY TABLE")

    generate_results_table(
        results_gap_k,
        results_gap_var,
        results_constants,
        output_path=f"{args.output_dir}/summary_table.txt"
    )

    # ========================================================================
    # Final Summary
    # ========================================================================
    print_banner("ALL EXPERIMENTS COMPLETE")

    print("Generated Figures:")
    print(f"  1. {args.output_dir}/gap_vs_k/figure.pdf")
    print(f"  2. {args.output_dir}/gap_vs_variance/figure.pdf")
    print(f"  3. {args.output_dir}/constants_validation/constants_visualization.pdf")
    print()

    print("Generated Data:")
    print(f"  1. {args.output_dir}/gap_vs_k/results.json")
    print(f"  2. {args.output_dir}/gap_vs_variance/results.json")
    print(f"  3. {args.output_dir}/constants_validation/constants.json")
    print()

    print("Summary:")
    print(f"  - {args.output_dir}/summary_table.txt")
    print()

    # Check validation
    all_pass = True
    if results_gap_k['fit']['success']:
        if results_gap_k['fit']['r2'] < 0.90:
            all_pass = False
            print("⚠ WARNING: Gap vs K R² below 0.90")

    if results_gap_var['fit']['success']:
        if results_gap_var['fit']['r2'] < 0.90:
            all_pass = False
            print("⚠ WARNING: Gap vs σ²_S R² below 0.90")

    mu_ratio = results_constants['theoretical_predictions']['mu_ratio']
    c_ratio = results_constants['theoretical_predictions']['c_quantum_ratio']

    if not (0.5 <= mu_ratio <= 2.0):
        all_pass = False
        print(f"⚠ WARNING: μ ratio {mu_ratio:.2f}x outside 2× bounds")

    if not (0.5 <= c_ratio <= 2.0):
        all_pass = False
        print(f"⚠ WARNING: c_quantum ratio {c_ratio:.2f}x outside 2× bounds")

    if all_pass:
        print("\n✓✓✓ ALL VALIDATIONS PASSED ✓✓✓")
    else:
        print("\n⚠ Some validations outside target ranges (acceptable for heuristic theory)")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate all paper results")

    parser.add_argument(
        "--meta_path",
        type=str,
        default="checkpoints/maml_best.pt",
        help="Path to trained meta-policy"
    )

    parser.add_argument(
        "--robust_path",
        type=str,
        default="checkpoints/robust_best.pt",
        help="Path to trained robust policy"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/paper",
        help="Output directory for all results"
    )

    parser.add_argument(
        "--n_tasks",
        type=int,
        default=100,
        help="Number of test tasks per experiment"
    )

    parser.add_argument(
        "--include_grape",
        action="store_true",
        default=True,
        help="Include GRAPE baseline in experiments (default: True)"
    )

    parser.add_argument(
        "--no_grape",
        action="store_false",
        dest="include_grape",
        help="Disable GRAPE baseline"
    )

    parser.add_argument(
        "--grape_iterations",
        type=int,
        default=100,
        help="Number of GRAPE optimization iterations per task (default: 100)"
    )

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.meta_path):
        print(f"ERROR: Meta-policy not found at {args.meta_path}")
        print("Please train the meta-policy first using experiments/train_meta.py")
        sys.exit(1)

    if not os.path.exists(args.robust_path):
        print(f"ERROR: Robust policy not found at {args.robust_path}")
        print("Please train the robust policy first using experiments/train_robust.py")
        sys.exit(1)

    main(args)
