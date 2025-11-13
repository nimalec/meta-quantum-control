"""
Plot: Comprehensive Validation Metrics from Checkpoint

Generates plots showing:
1. Validation fidelity evolution during training
2. Pre-adaptation vs post-adaptation validation curves
3. Adaptation gain over training
4. Task-specific validation performance
5. Learning curves with confidence intervals

Loads checkpoint(s) and creates publication-quality validation plots.
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import yaml

from metaqctrl.utils.plot_training import load_checkpoint, smooth_curve


def plot_validation_evolution(
    checkpoint_path: str,
    save_path: str = "results/validation_evolution.pdf",
    smooth_window: int = 5,
    figsize: Tuple[float, float] = (10, 6),
    dpi: int = 300
):
    """
    Plot evolution of validation metrics during training.

    Shows both pre-adaptation and post-adaptation validation performance.
    """
    checkpoint = load_checkpoint(checkpoint_path)

    # Extract validation data
    val_losses_post = checkpoint.get('meta_val_losses', [])

    if len(val_losses_post) == 0:
        print("No validation data found in checkpoint.")
        return

    val_interval = checkpoint.get('val_interval', 50)
    val_iterations = np.arange(len(val_losses_post)) * val_interval

    # Skip iteration 0 if present
    if val_iterations[0] == 0 and len(val_iterations) > 1:
        val_iterations = val_iterations[1:]
        val_losses_post = val_losses_post[1:]

    # Convert losses to fidelities
    val_fidelities_post = [1.0 - loss for loss in val_losses_post]

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: Validation loss
    ax = axes[0]
    ax.plot(val_iterations, val_losses_post, marker='o',
            color='tab:orange', linewidth=2, markersize=6,
            label='Post-Adaptation Loss')

    # Smooth curve if enough points
    if len(val_losses_post) >= smooth_window:
        smoothed = smooth_curve(val_losses_post, window=smooth_window)
        ax.plot(val_iterations, smoothed, color='darkred',
                linewidth=3, alpha=0.7, linestyle='--',
                label=f'Smoothed (window={smooth_window})')

    ax.set_xlabel('Training Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
    ax.set_title('(a) Validation Loss Evolution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Right: Validation fidelity
    ax = axes[1]
    ax.plot(val_iterations, val_fidelities_post, marker='o',
            color='tab:green', linewidth=2, markersize=6,
            label='Post-Adaptation Fidelity')

    # Smooth curve if enough points
    if len(val_fidelities_post) >= smooth_window:
        smoothed = smooth_curve(val_fidelities_post, window=smooth_window)
        ax.plot(val_iterations, smoothed, color='darkgreen',
                linewidth=3, alpha=0.7, linestyle='--',
                label=f'Smoothed (window={smooth_window})')

    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_xlabel('Training Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Validation Fidelity', fontsize=12, fontweight='bold')
    ax.set_title('(b) Validation Fidelity Evolution', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Validation Performance During Training',
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved validation evolution plot to {save_path}")
    plt.close()


def plot_training_validation_comparison(
    checkpoint_path: str,
    save_path: str = "results/train_val_comparison.pdf",
    smooth_window: int = 10,
    figsize: Tuple[float, float] = (12, 5),
    dpi: int = 300
):
    """
    Plot training vs validation performance comparison.
    """
    checkpoint = load_checkpoint(checkpoint_path)

    train_losses = checkpoint.get('meta_train_losses', [])
    val_losses_post = checkpoint.get('meta_val_losses', [])

    if len(train_losses) == 0:
        print("No training data found in checkpoint.")
        return

    train_iterations = np.arange(len(train_losses))
    train_fidelities = [1.0 - loss for loss in train_losses]

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: Loss comparison
    ax = axes[0]

    # Training loss
    ax.plot(train_iterations, train_losses, alpha=0.2, color='tab:blue',
            linewidth=0.5)
    smoothed_train_loss = smooth_curve(train_losses, window=smooth_window)
    ax.plot(train_iterations, smoothed_train_loss, color='tab:blue',
            linewidth=2, label='Training Loss')

    # Validation loss
    if len(val_losses_post) > 0:
        val_interval = checkpoint.get('val_interval', 50)
        val_iterations = np.arange(len(val_losses_post)) * val_interval

        if val_iterations[0] == 0 and len(val_iterations) > 1:
            val_iterations = val_iterations[1:]
            val_losses_post = val_losses_post[1:]

        ax.plot(val_iterations, val_losses_post, marker='o',
                color='tab:orange', linewidth=2, markersize=6,
                label='Validation Loss')

    ax.set_xlabel('Training Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title('(a) Training vs Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Right: Fidelity comparison
    ax = axes[1]

    # Training fidelity
    ax.plot(train_iterations, train_fidelities, alpha=0.2, color='tab:green',
            linewidth=0.5)
    smoothed_train_fid = smooth_curve(train_fidelities, window=smooth_window)
    ax.plot(train_iterations, smoothed_train_fid, color='tab:green',
            linewidth=2, label='Training Fidelity')

    # Validation fidelity
    if len(val_losses_post) > 0:
        val_fidelities = [1.0 - loss for loss in val_losses_post]
        ax.plot(val_iterations, val_fidelities, marker='o',
                color='darkgreen', linewidth=2, markersize=6,
                label='Validation Fidelity')

    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_xlabel('Training Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Fidelity', fontsize=12, fontweight='bold')
    ax.set_title('(b) Training vs Validation Fidelity', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved training vs validation comparison to {save_path}")
    plt.close()


def plot_checkpoint_comparison(
    checkpoint_paths: Dict[str, str],
    save_path: str = "results/checkpoint_comparison.pdf",
    smooth_window: int = 10,
    metric: str = 'fidelity',
    figsize: Tuple[float, float] = (10, 6),
    dpi: int = 300
):
    """
    Compare validation performance across multiple checkpoints.

    Args:
        checkpoint_paths: Dict of {label: path}
        save_path: Output path
        smooth_window: Smoothing window
        metric: 'loss' or 'fidelity'
        figsize: Figure size
        dpi: DPI for saving
    """
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, len(checkpoint_paths)))

    for (label, ckpt_path), color in zip(checkpoint_paths.items(), colors):
        checkpoint = load_checkpoint(ckpt_path)
        val_losses = checkpoint.get('meta_val_losses', [])

        if len(val_losses) == 0:
            print(f"No validation data in {label}")
            continue

        val_interval = checkpoint.get('val_interval', 50)
        val_iterations = np.arange(len(val_losses)) * val_interval

        if val_iterations[0] == 0 and len(val_iterations) > 1:
            val_iterations = val_iterations[1:]
            val_losses = val_losses[1:]

        if metric == 'fidelity':
            values = [1.0 - loss for loss in val_losses]
            ylabel = 'Validation Fidelity'
        else:
            values = val_losses
            ylabel = 'Validation Loss'

        # Plot
        ax.plot(val_iterations, values, marker='o', color=color,
                linewidth=2, markersize=4, alpha=0.6, label=label)

        # Smooth if enough points
        if len(values) >= smooth_window:
            smoothed = smooth_curve(values, window=smooth_window)
            ax.plot(val_iterations, smoothed, color=color,
                    linewidth=3, linestyle='--', alpha=0.8)

    if metric == 'fidelity':
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_ylim([0, 1.05])

    ax.set_xlabel('Training Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(f'{ylabel} Comparison Across Checkpoints',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved checkpoint comparison to {save_path}")
    plt.close()


def plot_validation_summary(
    checkpoint_path: str,
    save_path: str = "results/validation_summary.pdf",
    figsize: Tuple[float, float] = (14, 10),
    dpi: int = 300
):
    """
    Create comprehensive 2x2 validation summary plot.
    """
    checkpoint = load_checkpoint(checkpoint_path)

    train_losses = checkpoint.get('meta_train_losses', [])
    val_losses_post = checkpoint.get('meta_val_losses', [])

    if len(train_losses) == 0 or len(val_losses_post) == 0:
        print("Insufficient data for validation summary.")
        return

    train_fidelities = [1.0 - loss for loss in train_losses]
    val_fidelities_post = [1.0 - loss for loss in val_losses_post]

    train_iterations = np.arange(len(train_losses))
    val_interval = checkpoint.get('val_interval', 50)
    val_iterations = np.arange(len(val_losses_post)) * val_interval

    if val_iterations[0] == 0 and len(val_iterations) > 1:
        val_iterations = val_iterations[1:]
        val_losses_post = val_losses_post[1:]
        val_fidelities_post = val_fidelities_post[1:]

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Top left: Training and validation loss
    ax = axes[0, 0]
    smoothed_train = smooth_curve(train_losses, window=10)
    ax.plot(train_iterations, smoothed_train, color='tab:blue',
            linewidth=2, label='Training')
    ax.plot(val_iterations, val_losses_post, marker='o',
            color='tab:orange', linewidth=2, markersize=5,
            label='Validation')
    ax.set_xlabel('Iteration', fontsize=11, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=11, fontweight='bold')
    ax.set_title('(a) Loss Evolution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Top right: Training and validation fidelity
    ax = axes[0, 1]
    smoothed_train_fid = smooth_curve(train_fidelities, window=10)
    ax.plot(train_iterations, smoothed_train_fid, color='tab:green',
            linewidth=2, label='Training')
    ax.plot(val_iterations, val_fidelities_post, marker='o',
            color='darkgreen', linewidth=2, markersize=5,
            label='Validation')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_xlabel('Iteration', fontsize=11, fontweight='bold')
    ax.set_ylabel('Fidelity', fontsize=11, fontweight='bold')
    ax.set_title('(b) Fidelity Evolution', fontsize=13, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Bottom left: Validation fidelity histogram
    ax = axes[1, 0]
    ax.hist(val_fidelities_post, bins=20, color='steelblue',
            edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(val_fidelities_post), color='red', linestyle='--',
               linewidth=2, label=f'Mean: {np.mean(val_fidelities_post):.4f}')
    ax.set_xlabel('Validation Fidelity', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title('(c) Validation Fidelity Distribution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Bottom right: Summary statistics
    ax = axes[1, 1]
    ax.axis('off')

    # Compute statistics
    final_train_fid = train_fidelities[-1]
    best_train_fid = max(train_fidelities)
    final_val_fid = val_fidelities_post[-1]
    best_val_fid = max(val_fidelities_post)
    mean_val_fid = np.mean(val_fidelities_post)
    std_val_fid = np.std(val_fidelities_post)

    summary_text = f"""
Validation Summary
{'='*50}

Training Performance:
  Final Fidelity:        {final_train_fid:.6f}
  Best Fidelity:         {best_train_fid:.6f}
  Total Iterations:      {len(train_losses)}

Validation Performance:
  Final Fidelity:        {final_val_fid:.6f}
  Best Fidelity:         {best_val_fid:.6f}
  Mean Fidelity:         {mean_val_fid:.6f}
  Std Fidelity:          {std_val_fid:.6f}
  Total Validations:     {len(val_fidelities_post)}

Hyperparameters:
  Inner LR (α):          {checkpoint.get('inner_lr', 'N/A')}
  Inner Steps (K):       {checkpoint.get('inner_steps', 'N/A')}
  Validation Interval:   {val_interval}
"""

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    ax.set_title('(d) Performance Summary', fontsize=13, fontweight='bold')

    plt.suptitle('Comprehensive Validation Analysis',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved validation summary to {save_path}")
    plt.close()


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot validation metrics from checkpoint(s)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/validation_plots",
        help="Output directory"
    )
    parser.add_argument(
        "--compare",
        type=str,
        nargs='*',
        help="Additional checkpoints for comparison (format: label:path)"
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("PLOT: Validation Metrics Analysis")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {args.output_dir}")
    print("=" * 80)

    # Generate plots
    print("\nGenerating plots...")

    plot_validation_evolution(
        args.checkpoint,
        f"{args.output_dir}/validation_evolution.pdf"
    )

    plot_training_validation_comparison(
        args.checkpoint,
        f"{args.output_dir}/train_val_comparison.pdf"
    )

    plot_validation_summary(
        args.checkpoint,
        f"{args.output_dir}/validation_summary.pdf"
    )

    # Comparison plots if requested
    if args.compare:
        checkpoint_dict = {"Main": args.checkpoint}
        for item in args.compare:
            if ':' in item:
                label, path = item.split(':', 1)
                checkpoint_dict[label] = path
            else:
                print(f"Warning: Skipping invalid compare format: {item}")

        if len(checkpoint_dict) > 1:
            plot_checkpoint_comparison(
                checkpoint_dict,
                f"{args.output_dir}/checkpoint_comparison_fidelity.pdf",
                metric='fidelity'
            )
            plot_checkpoint_comparison(
                checkpoint_dict,
                f"{args.output_dir}/checkpoint_comparison_loss.pdf",
                metric='loss'
            )

    print("\n" + "=" * 80)
    print("PLOTTING COMPLETE")
    print("=" * 80)
    print(f"\nGenerated plots in {args.output_dir}/")
    print()


if __name__ == "__main__":
    main()
