"""
Plotting utilities for MAML training metrics.

Creates publication-quality plots for meta-RL papers including:
- Training loss over iterations
- Validation loss over iterations
- Adaptation gain over iterations
- Pre-adaptation vs post-adaptation comparison
- Fidelity metrics (1 - loss) over iterations
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import seaborn as sns

# Publication-quality plot settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14


def load_checkpoint(checkpoint_path: str) -> Dict:
    """
    Load a MAML checkpoint file.

    Args:
        checkpoint_path: Path to .pt checkpoint file

    Returns:
        Dictionary containing checkpoint data
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    return checkpoint


def smooth_curve(values: List[float], window: int = 10) -> np.ndarray:
    """
    Apply moving average smoothing to a curve.

    Args:
        values: List of values to smooth
        window: Window size for moving average

    Returns:
        Smoothed values as numpy array
    """
    if len(values) < window:
        return np.array(values)

    weights = np.ones(window) / window
    smoothed = np.convolve(values, weights, mode='valid')

    # Pad beginning to match original length
    pad_length = len(values) - len(smoothed)
    pad_values = np.full(pad_length, smoothed[0])

    return np.concatenate([pad_values, smoothed])


def plot_training_loss(
    checkpoint_path: str,
    save_path: Optional[str] = None,
    smooth_window: int = 10,
    figsize: Tuple[float, float] = (7, 5),
    dpi: int = 300,
    show_smoothed: bool = True
):
    """
    Plot meta-training loss over iterations.

    Args:
        checkpoint_path: Path to MAML checkpoint
        save_path: Where to save the plot (if None, displays instead)
        smooth_window: Window size for smoothing
        figsize: Figure size in inches
        dpi: Resolution for saving
        show_smoothed: Whether to show smoothed curve
    """
    checkpoint = load_checkpoint(checkpoint_path)
    train_losses = checkpoint['meta_train_losses']

    iterations = np.arange(len(train_losses))

    fig, ax = plt.subplots(figsize=figsize)

    # Plot raw data
    if show_smoothed:
        ax.plot(iterations, train_losses, alpha=0.3, color='tab:blue',
                linewidth=0.5, label='Raw')
        # Plot smoothed curve
        smoothed = smooth_curve(train_losses, window=smooth_window)
        ax.plot(iterations, smoothed, color='tab:blue', linewidth=2,
                label=f'Smoothed (window={smooth_window})')
    else:
        ax.plot(iterations, train_losses, color='tab:blue', linewidth=1.5)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Meta-Training Loss')
    ax.set_title('Meta-Training Loss over Iterations')
    ax.grid(True, alpha=0.3)

    if show_smoothed:
        ax.legend(framealpha=0.9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved training loss plot to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_validation_losses(
    checkpoint_path: str,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (7, 5),
    dpi: int = 300
):
    """
    Plot pre-adaptation and post-adaptation validation losses.

    Args:
        checkpoint_path: Path to MAML checkpoint
        save_path: Where to save the plot
        figsize: Figure size in inches
        dpi: Resolution for saving
    """
    checkpoint = load_checkpoint(checkpoint_path)

    # Extract validation metrics (stored at validation intervals)
    val_losses_post = checkpoint.get('meta_val_losses', [])

    # Try to extract pre-adapt losses if saved separately
    # (These may be in kwargs saved with checkpoints)
    val_losses_pre = []
    adaptation_gains = []

    # If we have validation data
    if len(val_losses_post) > 0:
        # Validation happens every val_interval iterations
        # We need to infer the iteration numbers
        val_interval = checkpoint.get('val_interval', 50)  # Default from config
        val_iterations = np.arange(len(val_losses_post)) * val_interval

        # Skip iteration 0 if present
        if val_iterations[0] == 0 and len(val_iterations) > 1:
            val_iterations = val_iterations[1:]
            val_losses_post = val_losses_post[1:]

        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(val_iterations, val_losses_post, marker='o',
                color='tab:orange', linewidth=2, markersize=6,
                label='Post-Adaptation Loss')

        ax.set_xlabel('Iteration')
        ax.set_ylabel('Validation Loss')
        ax.set_title('Validation Loss over Iterations')
        ax.grid(True, alpha=0.3)
        ax.legend(framealpha=0.9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Saved validation loss plot to {save_path}")
        else:
            plt.show()

        plt.close()
    else:
        print("No validation losses found in checkpoint.")


def plot_adaptation_gain(
    checkpoint_path: str,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (7, 5),
    dpi: int = 300
):
    """
    Plot adaptation gain (pre-adapt loss - post-adapt loss) over iterations.

    Requires checkpoints saved with val_loss_pre_adapt and val_loss_post_adapt.

    Args:
        checkpoint_path: Path to MAML checkpoint
        save_path: Where to save the plot
        figsize: Figure size in inches
        dpi: Resolution for saving
    """
    checkpoint = load_checkpoint(checkpoint_path)

    # Extract pre and post adaptation losses if available
    pre_adapt = checkpoint.get('val_loss_pre_adapt', None)
    post_adapt = checkpoint.get('val_loss_post_adapt', None)
    adaptation_gain = checkpoint.get('adaptation_gain', None)

    if adaptation_gain is not None:
        # Single value from final checkpoint
        print(f"Final Adaptation Gain: {adaptation_gain:.4f}")
        print(f"Pre-adapt loss: {pre_adapt:.4f}")
        print(f"Post-adapt loss: {post_adapt:.4f}")
    else:
        print("Adaptation gain not found in checkpoint.")
        print("Note: Adaptation gain is typically saved only in validation checkpoints.")


def plot_fidelity(
    checkpoint_path: str,
    save_path: Optional[str] = None,
    smooth_window: int = 10,
    figsize: Tuple[float, float] = (7, 5),
    dpi: int = 300,
    show_smoothed: bool = True
):
    """
    Plot fidelity (1 - loss) over iterations.

    In quantum control, loss = 1 - fidelity, so fidelity = 1 - loss.
    Higher fidelity is better (closer to 1.0 = perfect gate).

    Args:
        checkpoint_path: Path to MAML checkpoint
        save_path: Where to save the plot
        smooth_window: Window size for smoothing
        figsize: Figure size in inches
        dpi: Resolution for saving
        show_smoothed: Whether to show smoothed curve
    """
    checkpoint = load_checkpoint(checkpoint_path)
    train_losses = checkpoint['meta_train_losses']

    # Convert losses to fidelities
    train_fidelities = [1.0 - loss for loss in train_losses]

    iterations = np.arange(len(train_fidelities))

    fig, ax = plt.subplots(figsize=figsize)

    # Plot raw data
    if show_smoothed:
        ax.plot(iterations, train_fidelities, alpha=0.3, color='tab:green',
                linewidth=0.5, label='Raw')
        # Plot smoothed curve
        smoothed = smooth_curve(train_fidelities, window=smooth_window)
        ax.plot(iterations, smoothed, color='tab:green', linewidth=2,
                label=f'Smoothed (window={smooth_window})')
    else:
        ax.plot(iterations, train_fidelities, color='tab:green', linewidth=1.5)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Fidelity')
    ax.set_title('Meta-Training Fidelity over Iterations')
    ax.set_ylim([0, 1.05])  # Fidelity is between 0 and 1
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.grid(True, alpha=0.3)

    if show_smoothed:
        ax.legend(framealpha=0.9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved fidelity plot to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_combined_metrics(
    checkpoint_path: str,
    save_path: Optional[str] = None,
    smooth_window: int = 10,
    figsize: Tuple[float, float] = (14, 10),
    dpi: int = 300
):
    """
    Create a combined figure with multiple subplots showing all training metrics.

    Creates a 2x2 grid with:
    - Top left: Training loss
    - Top right: Fidelity
    - Bottom left: Validation loss (if available)
    - Bottom right: Training statistics summary

    Args:
        checkpoint_path: Path to MAML checkpoint
        save_path: Where to save the plot
        smooth_window: Window size for smoothing
        figsize: Figure size in inches
        dpi: Resolution for saving
    """
    checkpoint = load_checkpoint(checkpoint_path)
    train_losses = checkpoint['meta_train_losses']
    val_losses_post = checkpoint.get('meta_val_losses', [])

    # Convert losses to fidelities
    train_fidelities = [1.0 - loss for loss in train_losses]

    iterations = np.arange(len(train_losses))

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Top left: Training loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(iterations, train_losses, alpha=0.3, color='tab:blue', linewidth=0.5)
    smoothed_loss = smooth_curve(train_losses, window=smooth_window)
    ax1.plot(iterations, smoothed_loss, color='tab:blue', linewidth=2,
             label=f'Smoothed (window={smooth_window})')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Meta-Training Loss')
    ax1.set_title('(a) Training Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend(framealpha=0.9)

    # Top right: Fidelity
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(iterations, train_fidelities, alpha=0.3, color='tab:green', linewidth=0.5)
    smoothed_fidelity = smooth_curve(train_fidelities, window=smooth_window)
    ax2.plot(iterations, smoothed_fidelity, color='tab:green', linewidth=2,
             label=f'Smoothed (window={smooth_window})')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Fidelity')
    ax2.set_title('(b) Training Fidelity')
    ax2.set_ylim([0, 1.05])
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax2.grid(True, alpha=0.3)
    ax2.legend(framealpha=0.9)

    # Bottom left: Validation loss (if available)
    ax3 = fig.add_subplot(gs[1, 0])
    if len(val_losses_post) > 0:
        val_interval = checkpoint.get('val_interval', 50)
        val_iterations = np.arange(len(val_losses_post)) * val_interval

        if val_iterations[0] == 0 and len(val_iterations) > 1:
            val_iterations = val_iterations[1:]
            val_losses_post = val_losses_post[1:]

        ax3.plot(val_iterations, val_losses_post, marker='o',
                color='tab:orange', linewidth=2, markersize=6)
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Validation Loss (Post-Adapt)')
        ax3.set_title('(c) Validation Loss')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No validation data available',
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('(c) Validation Loss')

    # Bottom right: Summary statistics
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    # Compute statistics
    final_loss = train_losses[-1]
    final_fidelity = train_fidelities[-1]
    min_loss = min(train_losses)
    max_loss = max(train_losses)
    mean_loss = np.mean(train_losses)

    # Get checkpoint metadata
    epoch = checkpoint.get('epoch', 'N/A')
    inner_lr = checkpoint.get('inner_lr', 'N/A')
    inner_steps = checkpoint.get('inner_steps', 'N/A')

    # Create summary text
    summary_text = f"""
Training Summary
{'='*40}

Total Iterations: {len(train_losses)}
Final Epoch: {epoch}

Loss Statistics:
  Final Loss: {final_loss:.6f}
  Min Loss: {min_loss:.6f}
  Max Loss: {max_loss:.6f}
  Mean Loss: {mean_loss:.6f}

Fidelity Statistics:
  Final Fidelity: {final_fidelity:.6f}
  Best Fidelity: {1.0 - min_loss:.6f}

MAML Hyperparameters:
  Inner LR (α): {inner_lr}
  Inner Steps (K): {inner_steps}
"""

    if len(val_losses_post) > 0:
        final_val_loss = val_losses_post[-1]
        min_val_loss = min(val_losses_post)
        summary_text += f"""
Validation:
  Final Val Loss: {final_val_loss:.6f}
  Best Val Loss: {min_val_loss:.6f}
  Best Val Fidelity: {1.0 - min_val_loss:.6f}
"""

    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    ax4.set_title('(d) Training Statistics')

    plt.suptitle('MAML Training Metrics', fontsize=16, y=0.98)

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved combined metrics plot to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_loss_comparison(
    checkpoint_paths: Dict[str, str],
    save_path: Optional[str] = None,
    smooth_window: int = 10,
    figsize: Tuple[float, float] = (10, 6),
    dpi: int = 300,
    metric: str = 'loss'
):
    """
    Compare training curves from multiple checkpoints.

    Useful for comparing different hyperparameters or runs.

    Args:
        checkpoint_paths: Dictionary of {label: checkpoint_path}
        save_path: Where to save the plot
        smooth_window: Window size for smoothing
        figsize: Figure size in inches
        dpi: Resolution for saving
        metric: 'loss' or 'fidelity'
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, len(checkpoint_paths)))

    for (label, checkpoint_path), color in zip(checkpoint_paths.items(), colors):
        checkpoint = load_checkpoint(checkpoint_path)
        train_losses = checkpoint['meta_train_losses']

        if metric == 'fidelity':
            values = [1.0 - loss for loss in train_losses]
            ylabel = 'Fidelity'
        else:
            values = train_losses
            ylabel = 'Loss'

        iterations = np.arange(len(values))

        # Plot raw with low alpha
        ax.plot(iterations, values, alpha=0.2, color=color, linewidth=0.5)

        # Plot smoothed
        smoothed = smooth_curve(values, window=smooth_window)
        ax.plot(iterations, smoothed, color=color, linewidth=2, label=label)

    ax.set_xlabel('Iteration')
    ax.set_ylabel(ylabel)
    ax.set_title(f'{ylabel.capitalize()} Comparison Across Runs')
    ax.grid(True, alpha=0.3)
    ax.legend(framealpha=0.9)

    if metric == 'fidelity':
        ax.set_ylim([0, 1.05])
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")
    else:
        plt.show()

    plt.close()


def print_checkpoint_summary(checkpoint_path: str):
    """
    Print a summary of checkpoint contents.

    Useful for quickly inspecting what data is available.

    Args:
        checkpoint_path: Path to MAML checkpoint
    """
    checkpoint = load_checkpoint(checkpoint_path)

    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print("=" * 60)

    print("\nAvailable Keys:")
    for key in checkpoint.keys():
        value = checkpoint[key]
        if isinstance(value, list):
            print(f"  {key}: list of length {len(value)}")
        elif isinstance(value, (int, float)):
            print(f"  {key}: {value}")
        elif isinstance(value, dict):
            print(f"  {key}: dict with {len(value)} items")
        else:
            print(f"  {key}: {type(value).__name__}")

    print("\nTraining Metrics:")
    train_losses = checkpoint.get('meta_train_losses', [])
    if train_losses:
        print(f"  Total iterations: {len(train_losses)}")
        print(f"  Final training loss: {train_losses[-1]:.6f}")
        print(f"  Min training loss: {min(train_losses):.6f}")
        print(f"  Max training loss: {max(train_losses):.6f}")

    val_losses = checkpoint.get('meta_val_losses', [])
    if val_losses:
        print(f"\nValidation Metrics:")
        print(f"  Validation points: {len(val_losses)}")
        print(f"  Final validation loss: {val_losses[-1]:.6f}")
        print(f"  Best validation loss: {min(val_losses):.6f}")

    print("\nMAML Configuration:")
    print(f"  Inner LR: {checkpoint.get('inner_lr', 'N/A')}")
    print(f"  Inner Steps: {checkpoint.get('inner_steps', 'N/A')}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")

    print("=" * 60)


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python plot_training.py <checkpoint_path> [output_dir]")
        print("\nExample:")
        print("  python plot_training.py checkpoints/maml_20241025_120000.pt results/figures/")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'results/figures'

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Print summary
    print_checkpoint_summary(checkpoint_path)

    # Generate all plots
    print("\nGenerating plots...")

    plot_training_loss(
        checkpoint_path,
        save_path=f"{output_dir}/training_loss.png"
    )

    plot_fidelity(
        checkpoint_path,
        save_path=f"{output_dir}/training_fidelity.png"
    )

    plot_validation_losses(
        checkpoint_path,
        save_path=f"{output_dir}/validation_loss.png"
    )

    plot_combined_metrics(
        checkpoint_path,
        save_path=f"{output_dir}/combined_metrics.png"
    )

    print(f"\nAll plots saved to {output_dir}/")
