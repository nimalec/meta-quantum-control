"""
Example script for plotting training results from training_history.json

This demonstrates how to use the plotting utilities to generate
publication-quality figures from your MAML training runs.
"""

from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "metaqctrl"))

from utils.plot_training import (
    plot_from_history,
    plot_validation_only,
    plot_complete_summary
)


def main():
    """
    Generate all plots from a training_history.json file.
    
    Usage:
        python plot_training_example.py
    """
    
    # Path to your training history
    # Modify this to point to your actual training history file
    history_path = "../checkpoints/training_history.json"
    
    # Output directory for figures
    output_dir = "../results/figures"
    
    print("=" * 70)
    print("Generating Training Plots")
    print("=" * 70)
    print(f"Input: {history_path}")
    print(f"Output: {output_dir}\n")
    
    # 1. Training and validation curves (side-by-side)
    print("[1/3] Creating training and validation curves...")
    plot_from_history(
        history_path=history_path,
        save_dir=output_dir,
        smooth_window=10,
        figsize=(12, 5),
        dpi=300
    )
    
    # 2. Validation fidelity only (with error bars)
    print("[2/3] Creating validation fidelity plot...")
    plot_validation_only(
        history_path=history_path,
        save_dir=output_dir,
        figsize=(7, 5),
        dpi=300
    )
    
    # 3. Complete summary (2x2 grid)
    print("[3/3] Creating complete summary...")
    plot_complete_summary(
        history_path=history_path,
        save_dir=output_dir,
        smooth_window=10,
        figsize=(14, 10),
        dpi=300
    )
    
    print("\n" + "=" * 70)
    print("Done! All plots saved to:")
    print("  - training_validation_curves.png")
    print("  - validation_fidelity.png")
    print("  - complete_training_summary.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
