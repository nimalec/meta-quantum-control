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
    checkpoint_path = "../checkpoints/maml_1.pt"

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



if __name__ == '__main__':
    print("=" * 70)
    print("MAML Training Visualization Examples")
    print("=" * 70)

    # Note: Update checkpoint paths to match your actual checkpoints

    print("\n[Example 1] Single checkpoint analysis")
    print("-" * 70)
    print("Uncomment the line below and update the checkpoint path:")
    print("# example_single_checkpoint()")
    example_single_checkpoint() 
