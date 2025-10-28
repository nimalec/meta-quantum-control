"""
Master Script: Generate All ICML Figures

This script runs all figure generation scripts in sequence.
"""

import os
import sys
from pathlib import Path
import argparse
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def find_checkpoints():
    """Find trained model checkpoints"""
    script_dir = Path(__file__).parent
    checkpoint_dir = script_dir.parent.parent / "checkpoints"

    possible_meta_paths = [
        checkpoint_dir / "maml_best.pt",
        checkpoint_dir / "maml_20251027_161519_best_policy.pt",
    ]
    possible_robust_paths = [
        checkpoint_dir / "robust_best.pt",
        checkpoint_dir / "robust_minimax_20251027_162238_best_policy.pt",
    ]

    meta_path = None
    for p in possible_meta_paths:
        if p.exists():
            meta_path = str(p)
            break

    robust_path = None
    for p in possible_robust_paths:
        if p.exists():
            robust_path = str(p)
            break

    return meta_path, robust_path


def run_figure2(meta_path, robust_path, config, output_dir, run_experiments):
    """Generate Figure 2"""
    print("\n" + "=" * 80)
    print("FIGURE 2: Fast Adaptation Scaling")
    print("=" * 80)

    from figure2_adaptation_scaling import generate_figure2

    try:
        start_time = time.time()
        generate_figure2(
            meta_policy_path=meta_path,
            robust_policy_path=robust_path,
            config=config,
            output_dir=output_dir,
            run_experiments=run_experiments
        )
        elapsed = time.time() - start_time
        print(f"\nFigure 2 completed in {elapsed/60:.1f} minutes")
        return True
    except Exception as e:
        print(f"\nERROR generating Figure 2: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_figure3(log_path, output_dir):
    """Generate Figure 3"""
    print("\n" + "=" * 80)
    print("FIGURE 3: Training Stability")
    print("=" * 80)

    from figure3_training_stability import generate_figure3

    try:
        start_time = time.time()
        generate_figure3(
            log_path=log_path,
            output_dir=output_dir
        )
        elapsed = time.time() - start_time
        print(f"\nFigure 3 completed in {elapsed:.1f} seconds")
        return True
    except Exception as e:
        print(f"\nERROR generating Figure 3: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_figure4(meta_path, robust_path, config, output_dir, run_experiments):
    """Generate Figure 4"""
    print("\n" + "=" * 80)
    print("FIGURE 4: Robustness and OOD")
    print("=" * 80)

    from figure4_robustness_ood import generate_figure4

    try:
        start_time = time.time()
        generate_figure4(
            meta_policy_path=meta_path,
            robust_policy_path=robust_path,
            config=config,
            output_dir=output_dir,
            run_experiments=run_experiments
        )
        elapsed = time.time() - start_time
        print(f"\nFigure 4 completed in {elapsed/60:.1f} minutes")
        return True
    except Exception as e:
        print(f"\nERROR generating Figure 4: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_appendix(meta_path, robust_path, config, output_dir, run_experiments):
    """Generate Appendix figures"""
    print("\n" + "=" * 80)
    print("APPENDIX FIGURES")
    print("=" * 80)

    from appendix_figures import generate_appendix_figures

    try:
        start_time = time.time()
        generate_appendix_figures(
            meta_policy_path=meta_path,
            robust_policy_path=robust_path,
            config=config,
            output_dir=output_dir,
            run_experiments=run_experiments
        )
        elapsed = time.time() - start_time
        print(f"\nAppendix figures completed in {elapsed/60:.1f} minutes")
        return True
    except Exception as e:
        print(f"\nERROR generating Appendix figures: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Generate all ICML figures')
    parser.add_argument('--output-dir', type=str, default='results/icml_figures',
                       help='Output directory for figures')
    parser.add_argument('--skip-experiments', action='store_true',
                       help='Skip running experiments, use cached data')
    parser.add_argument('--figures', type=str, nargs='+',
                       choices=['2', '3', '4', 'appendix', 'all'],
                       default=['all'],
                       help='Which figures to generate')
    parser.add_argument('--training-log', type=str, default=None,
                       help='Path to training log for Figure 3')
    parser.add_argument('--meta-policy', type=str, default=None,
                       help='Path to meta policy checkpoint')
    parser.add_argument('--robust-policy', type=str, default=None,
                       help='Path to robust policy checkpoint')

    args = parser.parse_args()

    print("=" * 80)
    print("ICML FIGURE GENERATION MASTER SCRIPT")
    print("=" * 80)
    print(f"\nOutput directory: {args.output_dir}")
    print(f"Run experiments: {not args.skip_experiments}")
    print(f"Figures to generate: {args.figures}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Configuration
    config = {
        'num_qubits': 1,
        'n_controls': 2,
        'n_segments': 20,
        'horizon': 1.0,
        'target_gate': 'hadamard',
        'hidden_dim': 128,
        'n_hidden_layers': 2,
        'inner_lr': 0.01,
        'noise_frequencies': [1.0, 5.0, 10.0]
    }

    # Find checkpoints
    if args.meta_policy and args.robust_policy:
        meta_path = args.meta_policy
        robust_path = args.robust_policy
    else:
        print("\nSearching for trained model checkpoints...")
        meta_path, robust_path = find_checkpoints()

    if meta_path is None or robust_path is None:
        print("\nERROR: Could not find trained models")
        print("Please specify paths with --meta-policy and --robust-policy")
        print("Or train models first using:")
        print("  python experiments/train_meta.py")
        print("  python experiments/train_robust.py")
        sys.exit(1)

    print(f"\nUsing meta policy: {meta_path}")
    print(f"Using robust policy: {robust_path}")

    # Determine which figures to generate
    figures_to_gen = args.figures
    if 'all' in figures_to_gen:
        figures_to_gen = ['2', '3', '4', 'appendix']

    # Track results
    results = {}
    total_start = time.time()

    # Generate figures
    if '2' in figures_to_gen:
        results['figure2'] = run_figure2(
            meta_path, robust_path, config, args.output_dir,
            run_experiments=not args.skip_experiments
        )

    if '3' in figures_to_gen:
        results['figure3'] = run_figure3(
            log_path=args.training_log,
            output_dir=args.output_dir
        )

    if '4' in figures_to_gen:
        results['figure4'] = run_figure4(
            meta_path, robust_path, config, args.output_dir,
            run_experiments=not args.skip_experiments
        )

    if 'appendix' in figures_to_gen:
        results['appendix'] = run_appendix(
            meta_path, robust_path, config, args.output_dir,
            run_experiments=not args.skip_experiments
        )

    total_elapsed = time.time() - total_start

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nTotal time: {total_elapsed/60:.1f} minutes\n")

    for fig_name, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {fig_name}: {status}")

    if all(results.values()):
        print("\n✓ All figures generated successfully!")
        print(f"\nResults saved to: {args.output_dir}")
        print("\nGenerated files:")
        print("  - figure2_complete.pdf/png")
        print("  - figure3_training_stability.pdf/png")
        print("  - figure4_robustness.pdf/png")
        print("  - appendix_s1_ablation.pdf/png")
        print("  - appendix_s2_flat_variance.pdf/png")
        print("  - appendix_s3_classical.pdf/png")
    else:
        print("\n✗ Some figures failed to generate")
        print("Check error messages above for details")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
