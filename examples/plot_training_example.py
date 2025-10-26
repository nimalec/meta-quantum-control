"""
Example: How to generate plots from MAML training checkpoints

This script demonstrates various ways to visualize your meta-RL training results.
"""

from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from metaqctrl.utils.plot_training import (
    plot_training_loss,
    plot_fidelity,
    plot_validation_losses,
    plot_combined_metrics,
    plot_loss_comparison,
    print_checkpoint_summary
)


def example_single_checkpoint():
    """
    Generate all standard plots from a single checkpoint.
    """
    # Path to your checkpoint
    checkpoint_path = "checkpoints/maml_20241025_120000.pt"

    # Output directory
    output_dir = "results/figures/training"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("Generating plots from single checkpoint...")

    # 1. Print summary of checkpoint contents
    print_checkpoint_summary(checkpoint_path)

    # 2. Plot training loss
    plot_training_loss(
        checkpoint_path,
        save_path=f"{output_dir}/training_loss.png",
        smooth_window=10,
        show_smoothed=True
    )

    # 3. Plot fidelity (1 - loss)
    plot_fidelity(
        checkpoint_path,
        save_path=f"{output_dir}/fidelity.png",
        smooth_window=10,
        show_smoothed=True
    )

    # 4. Plot validation losses
    plot_validation_losses(
        checkpoint_path,
        save_path=f"{output_dir}/validation_loss.png"
    )

    # 5. Combined metrics (2x2 grid)
    plot_combined_metrics(
        checkpoint_path,
        save_path=f"{output_dir}/combined_metrics.png",
        smooth_window=10
    )

    print(f"\nAll plots saved to {output_dir}/")


def example_compare_multiple_runs():
    """
    Compare training curves from multiple runs.

    Useful for hyperparameter tuning or comparing different configurations.
    """
    # Paths to multiple checkpoints
    checkpoints = {
        'Inner LR=0.01': 'checkpoints/maml_lr001.pt',
        'Inner LR=0.001': 'checkpoints/maml_lr0001.pt',
        'Inner Steps=1': 'checkpoints/maml_steps1.pt',
        'Inner Steps=5': 'checkpoints/maml_steps5.pt',
    }

    output_dir = "results/figures/comparison"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("Comparing multiple runs...")

    # Compare losses
    plot_loss_comparison(
        checkpoints,
        save_path=f"{output_dir}/loss_comparison.png",
        smooth_window=10,
        metric='loss'
    )

    # Compare fidelities
    plot_loss_comparison(
        checkpoints,
        save_path=f"{output_dir}/fidelity_comparison.png",
        smooth_window=10,
        metric='fidelity'
    )

    print(f"\nComparison plots saved to {output_dir}/")


def example_custom_plot():
    """
    Create a custom plot using checkpoint data.

    Shows how to access and manipulate the raw data.
    """
    import torch
    import matplotlib.pyplot as plt
    import numpy as np

    checkpoint_path = "checkpoints/maml_20241025_120000.pt"
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract data
    train_losses = checkpoint['meta_train_losses']
    val_losses = checkpoint.get('meta_val_losses', [])

    # Create custom visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Loss distribution (histogram)
    ax1.hist(train_losses, bins=50, alpha=0.7, color='tab:blue', edgecolor='black')
    ax1.axvline(np.mean(train_losses), color='red', linestyle='--',
                linewidth=2, label=f'Mean: {np.mean(train_losses):.4f}')
    ax1.set_xlabel('Loss Value')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Training Losses')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Improvement rate (derivative of loss)
    iterations = np.arange(len(train_losses))
    loss_gradient = np.gradient(train_losses)

    ax2.plot(iterations, loss_gradient, color='tab:purple', linewidth=1.5)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss Gradient (dL/diter)')
    ax2.set_title('Learning Rate (instantaneous improvement)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/figures/custom_analysis.png', dpi=300, bbox_inches='tight')
    print("Custom plot saved to results/figures/custom_analysis.png")


def example_publication_ready():
    """
    Generate publication-ready figures for a meta-RL paper.

    Creates high-resolution plots with proper formatting.
    """
    checkpoint_path = "checkpoints/maml_best.pt"
    output_dir = "results/figures/publication"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("Generating publication-ready figures...")

    # Figure 1: Training and validation curves
    plot_combined_metrics(
        checkpoint_path,
        save_path=f"{output_dir}/figure1_training_curves.png",
        smooth_window=20,  # More smoothing for cleaner look
        figsize=(12, 8),   # Larger for clarity
        dpi=600            # High resolution for publication
    )

    # Figure 2: Fidelity over iterations
    plot_fidelity(
        checkpoint_path,
        save_path=f"{output_dir}/figure2_fidelity.png",
        smooth_window=20,
        figsize=(8, 6),
        dpi=600,
        show_smoothed=True
    )

    print(f"\nPublication figures saved to {output_dir}/")
    print("Recommended settings:")
    print("  - DPI: 600 (journal quality)")
    print("  - Format: PNG or PDF (convert with: convert file.png file.pdf)")
    print("  - Font: Times New Roman (serif)")


if __name__ == '__main__':
    print("=" * 70)
    print("MAML Training Visualization Examples")
    print("=" * 70)

    # Note: Update checkpoint paths to match your actual checkpoints

    print("\n[Example 1] Single checkpoint analysis")
    print("-" * 70)
    print("Uncomment the line below and update the checkpoint path:")
    print("# example_single_checkpoint()")

    print("\n[Example 2] Compare multiple runs")
    print("-" * 70)
    print("Uncomment the line below and update checkpoint paths:")
    print("# example_compare_multiple_runs()")

    print("\n[Example 3] Custom analysis")
    print("-" * 70)
    print("Uncomment the line below and update the checkpoint path:")
    print("# example_custom_plot()")

    print("\n[Example 4] Publication-ready figures")
    print("-" * 70)
    print("Uncomment the line below and update the checkpoint path:")
    print("# example_publication_ready()")

    print("\n" + "=" * 70)
    print("To use these examples:")
    print("  1. Train your MAML model to generate checkpoints")
    print("  2. Update the checkpoint paths in this script")
    print("  3. Uncomment the example you want to run")
    print("  4. Run: python examples/plot_training_example.py")
    print("=" * 70)
