"""
Run Full Baseline Comparison Pipeline

This script automates the entire baseline comparison workflow:
1. Train average task policy (if not exists)
2. Evaluate all baselines
3. Generate plots

Usage:
    python run_full_comparison.py --meta_policy <path> [options]

Example:
    python run_full_comparison.py \
        --meta_policy ../train_scripts/checkpoints/maml_best_pauli_x_best_policy.pt \
        --config ../../configs/experiment_config.yaml \
        --skip_training  # Skip training if policy exists
"""

import argparse
import subprocess
from pathlib import Path
import sys


def run_command(cmd, description):
    """Run a command and handle errors."""
    print("\n" + "=" * 70)
    print(f"{description}")
    print("=" * 70)
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print(f"\n❌ ERROR: {description} failed!")
        sys.exit(1)
    else:
        print(f"\n✓ {description} completed successfully")


def main(args):
    """Run the full baseline comparison pipeline."""

    print("=" * 70)
    print("BASELINE COMPARISON PIPELINE")
    print("=" * 70)
    print(f"\nMeta policy: {args.meta_policy}")
    print(f"Config: {args.config}")
    print(f"Output directory: {args.output_dir}")

    # Check meta policy exists
    meta_policy_path = Path(args.meta_policy)
    if not meta_policy_path.exists():
        print(f"\n❌ ERROR: Meta policy not found at {args.meta_policy}")
        print("Please train the meta policy first using:")
        print("  cd experiments/train_scripts")
        print("  python train_meta.py --config ../../configs/experiment_config.yaml")
        sys.exit(1)

    # Step 1: Train average task policy (if needed)
    average_policy_path = Path(args.average_policy)

    if average_policy_path.exists() and args.skip_training:
        print(f"\n✓ Average task policy already exists at {args.average_policy}")
        print("  Skipping training (use --no-skip-training to retrain)")
    else:
        if average_policy_path.exists() and not args.skip_training:
            print(f"\n⚠️  Average task policy exists but will be retrained")

        cmd = [
            "python", "train_average_task.py",
            "--config", args.config
        ]
        run_command(cmd, "Step 1: Training Average Task Policy")

    # Verify average policy was created
    if not average_policy_path.exists():
        print(f"\n❌ ERROR: Average policy not found at {args.average_policy}")
        sys.exit(1)

    # Step 2: Evaluate baselines
    cmd = [
        "python", "evaluate_baselines.py",
        "--meta_policy", args.meta_policy,
        "--average_policy", args.average_policy,
        "--config", args.config,
        "--output_dir", args.output_dir,
        "--n_test_tasks", str(args.n_test_tasks),
        "--seed", str(args.seed)
    ]

    # Add K values
    cmd.extend(["--k_values"] + [str(k) for k in args.k_values])

    # Add variance scales
    cmd.extend(["--variance_scales"] + [str(v) for v in args.variance_scales])

    # Add other parameters
    cmd.extend([
        "--k_fixed", str(args.k_fixed),
        "--n_tasks_per_variance", str(args.n_tasks_per_variance)
    ])

    run_command(cmd, "Step 2: Evaluating Baselines")

    # Step 3: Plot results
    cmd = [
        "python", "plot_gap_analysis.py",
        "--results_dir", args.output_dir
    ]
    run_command(cmd, "Step 3: Generating Plots")

    # Print summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE!")
    print("=" * 70)

    output_dir = Path(args.output_dir)
    print(f"\nResults saved to: {output_dir}")
    print("\nGenerated files:")
    print(f"  • {output_dir / 'gap_vs_k_results.json'}")
    print(f"  • {output_dir / 'gap_vs_variance_results.json'}")
    print(f"  • {output_dir / 'gap_vs_k.pdf'}")
    print(f"  • {output_dir / 'gap_vs_variance.pdf'}")
    print(f"  • {output_dir / 'combined_summary.pdf'}")

    print("\n✓ All steps completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run full baseline comparison pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Full pipeline with default settings
  python run_full_comparison.py \\
      --meta_policy ../train_scripts/checkpoints/maml_best_pauli_x_best_policy.pt

  # Custom evaluation parameters
  python run_full_comparison.py \\
      --meta_policy ../train_scripts/checkpoints/maml_best_pauli_x_best_policy.pt \\
      --n_test_tasks 200 \\
      --k_values 0 1 2 3 5 7 10 15 20 \\
      --variance_scales 0.1 0.2 0.3 0.5 0.7 1.0

  # Skip training if average policy already exists
  python run_full_comparison.py \\
      --meta_policy ../train_scripts/checkpoints/maml_best_pauli_x_best_policy.pt \\
      --skip_training
        """
    )

    # Required arguments
    parser.add_argument(
        '--meta_policy',
        type=str,
        default='../train_scripts/checkpoints/maml_best_pauli_x_best_policy.pt',
        help='Path to trained meta-policy checkpoint'
    )

    # Optional paths
    parser.add_argument(
        '--average_policy',
        type=str,
        default='checkpoints/baseline_comparison/average_task_policy.pt',
        help='Path to save/load average task policy'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='../../configs/experiment_config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/baseline_comparison',
        help='Output directory for results'
    )

    # Training control
    parser.add_argument(
        '--skip_training',
        action='store_true',
        default=True,
        help='Skip training if average policy exists (default: True)'
    )
    parser.add_argument(
        '--no-skip-training',
        dest='skip_training',
        action='store_false',
        help='Always retrain average policy'
    )

    # Evaluation parameters
    parser.add_argument(
        '--n_test_tasks',
        type=int,
        default=100,
        help='Number of test tasks for gap vs K evaluation'
    )
    parser.add_argument(
        '--k_values',
        type=int,
        nargs='+',
        default=[0, 1, 2, 3, 5, 7, 10],
        help='K values for gap vs K evaluation'
    )
    parser.add_argument(
        '--k_fixed',
        type=int,
        default=5,
        help='Fixed K for gap vs variance evaluation'
    )
    parser.add_argument(
        '--variance_scales',
        type=float,
        nargs='+',
        default=[0.1, 0.3, 0.5, 0.7, 1.0],
        help='Variance scale factors (0-1)'
    )
    parser.add_argument(
        '--n_tasks_per_variance',
        type=int,
        default=50,
        help='Number of tasks per variance level'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )

    args = parser.parse_args()
    main(args)
