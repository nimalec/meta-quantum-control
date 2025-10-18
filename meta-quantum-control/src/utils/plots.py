"""
Publication-quality plotting utilities for meta-quantum-control.

This module provides CLI tools for generating figures using consistent
formatting suitable for scientific publications.
"""

import typer
from pathlib import Path
from typing import Optional, List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from src.quantum.noise_models import (
    NoisePSDModel,
    TaskDistribution,
    psd_distance,
    NoiseParameters,
    PSDToLindblad
)

# Initialize Typer app
app = typer.Typer(help="Generate publication-quality figures for meta-quantum-control")

# ============================================================================
# Publication-quality plotting configuration
# ============================================================================

def configure_publication_style():
    """
    Configure matplotlib for publication-quality plots.

    Settings optimized for:
    - Two-column journal format (3.5" width)
    - High DPI (300+) for print quality
    - Professional font rendering
    - Consistent color scheme
    """
    # Use a clean style as base
    plt.style.use('seaborn-v0_8-paper')

    # Font settings
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    mpl.rcParams['font.size'] = 10
    mpl.rcParams['axes.labelsize'] = 11
    mpl.rcParams['axes.titlesize'] = 12
    mpl.rcParams['xtick.labelsize'] = 9
    mpl.rcParams['ytick.labelsize'] = 9
    mpl.rcParams['legend.fontsize'] = 9

    # Figure settings
    mpl.rcParams['figure.figsize'] = (7, 5)  # Two-column width
    mpl.rcParams['figure.dpi'] = 100  # Display DPI
    mpl.rcParams['savefig.dpi'] = 300  # Save DPI
    mpl.rcParams['savefig.bbox'] = 'tight'
    mpl.rcParams['savefig.pad_inches'] = 0.05

    # Line and marker settings
    mpl.rcParams['lines.linewidth'] = 1.5
    mpl.rcParams['lines.markersize'] = 6
    mpl.rcParams['axes.linewidth'] = 1.0

    # Grid settings
    mpl.rcParams['grid.alpha'] = 0.3
    mpl.rcParams['grid.linewidth'] = 0.5

    # Legend settings
    mpl.rcParams['legend.frameon'] = True
    mpl.rcParams['legend.framealpha'] = 0.9
    mpl.rcParams['legend.fancybox'] = False
    mpl.rcParams['legend.edgecolor'] = '0.8'

    # Color cycle - professional color palette
    colors = [
        '#1f77b4',  # blue
        '#ff7f0e',  # orange
        '#2ca02c',  # green
        '#d62728',  # red
        '#9467bd',  # purple
        '#8c564b',  # brown
        '#e377c2',  # pink
        '#7f7f7f',  # gray
    ]
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors)


def save_figure(fig: plt.Figure, output_path: Path, dpi: int = 300):
    """
    Save figure with consistent settings.

    Args:
        fig: Matplotlib figure object
        output_path: Path where to save the figure
        dpi: Resolution in dots per inch
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0.05)
    typer.echo(f"Figure saved to: {output_path}")


# ============================================================================
# Plotting functions
# ============================================================================

def plot_noise_psd(
    output_path: Path,
    model_type: str = 'one_over_f',
    n_tasks: int = 5,
    seed: int = 42,
    alpha_range: tuple = (0.5, 2.0),
    A_range: tuple = (0.05, 0.3),
    omega_c_range: tuple = (2.0, 8.0),
    omega_min: float = 0.1,
    omega_max: float = 100.0,
    n_omega: int = 1000,
    show_distances: bool = False
):
    """
    Plot power spectral densities of sampled tasks.

    Args:
        output_path: Where to save the figure
        model_type: PSD model type ('one_over_f', 'lorentzian', 'double_exp')
        n_tasks: Number of tasks to sample and plot
        seed: Random seed for reproducibility
        alpha_range: Range for spectral exponent α
        A_range: Range for amplitude A
        omega_c_range: Range for cutoff frequency ωc
        omega_min: Minimum frequency for plot
        omega_max: Maximum frequency for plot
        n_omega: Number of frequency points
        show_distances: Whether to print pairwise distances
    """
    configure_publication_style()

    # Define task distribution
    task_dist = TaskDistribution(
        dist_type='uniform',
        ranges={
            'alpha': alpha_range,
            'A': A_range,
            'omega_c': omega_c_range
        }
    )

    # Sample tasks
    rng = np.random.default_rng(seed)
    tasks = task_dist.sample(n_tasks, rng)

    # Setup PSD model and frequency grid
    psd_model = NoisePSDModel(model_type=model_type)
    omega = np.logspace(np.log10(omega_min), np.log10(omega_max), n_omega)

    # Create figure
    fig, ax = plt.subplots(figsize=(7, 5))

    # Plot each task
    for i, task in enumerate(tasks):
        S = psd_model.psd(omega, task)
        label = f'Task {i}: α={task.alpha:.2f}, A={task.A:.2f}, ωc={task.omega_c:.1f}'
        ax.loglog(omega, S, label=label, linewidth=1.5)

    # Formatting
    ax.set_xlabel('Frequency ω (rad/s)')
    ax.set_ylabel('PSD S(ω)')
    ax.set_title(f'Power Spectral Densities ({model_type} model)')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    # Save figure
    save_figure(fig, output_path)

    # Optionally compute and print distances
    if show_distances:
        typer.echo("\nPairwise PSD distances:")
        omega_grid = np.logspace(np.log10(omega_min), np.log10(omega_max), 500)
        for i in range(len(tasks)):
            for j in range(i + 1, len(tasks)):
                dist = psd_distance(tasks[i], tasks[j], omega_grid)
                typer.echo(f"  d(task{i}, task{j}) = {dist:.4f}")

        variance = task_dist.compute_variance()
        typer.echo(f"\nTask distribution variance σ²_θ = {variance:.4f}")

    plt.close(fig)


def plot_psd_comparison(
    output_path: Path,
    alpha: float = 1.0,
    A: float = 0.1,
    omega_c: float = 5.0,
    omega_min: float = 0.1,
    omega_max: float = 100.0,
    n_omega: int = 1000
):
    """
    Compare different PSD models for the same parameters.

    Args:
        output_path: Where to save the figure
        alpha: Spectral exponent
        A: Amplitude
        omega_c: Cutoff frequency
        omega_min: Minimum frequency for plot
        omega_max: Maximum frequency for plot
        n_omega: Number of frequency points
    """
    configure_publication_style()

    # Setup parameters
    theta = NoiseParameters(alpha=alpha, A=A, omega_c=omega_c)
    omega = np.logspace(np.log10(omega_min), np.log10(omega_max), n_omega)

    # Create figure
    fig, ax = plt.subplots(figsize=(7, 5))

    # Plot each model type
    models = ['one_over_f', 'lorentzian', 'double_exp']
    model_names = ['1/f^α', 'Lorentzian', 'Double Exponential']

    for model_type, model_name in zip(models, model_names):
        psd_model = NoisePSDModel(model_type=model_type)
        S = psd_model.psd(omega, theta)
        ax.loglog(omega, S, label=model_name, linewidth=1.5)

    # Formatting
    ax.set_xlabel('Frequency ω (rad/s)')
    ax.set_ylabel('PSD S(ω)')
    ax.set_title(f'PSD Model Comparison (α={alpha}, A={A}, ωc={omega_c})')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, which='both')

    # Save figure
    save_figure(fig, output_path)
    plt.close(fig)


def plot_effective_rates(
    output_path: Path,
    n_tasks: int = 5,
    seed: int = 42,
    alpha_range: tuple = (0.5, 2.0),
    A_range: tuple = (0.05, 0.3),
    omega_c_range: tuple = (2.0, 8.0),
    n_freqs: int = 10,
    omega_min: float = 0.1,
    omega_max: float = 100.0
):
    """
    Plot effective decay rates for different noise tasks.

    This plots the get_effective_rates() output from PSDToLindblad,
    showing how different noise parameters affect dissipation rates
    across frequency channels.

    Args:
        output_path: Where to save the figure
        n_tasks: Number of tasks to sample and plot
        seed: Random seed for reproducibility
        alpha_range: Range for spectral exponent α
        A_range: Range for amplitude A
        omega_c_range: Range for cutoff frequency ωc
        n_freqs: Number of sampling frequencies
        omega_min: Minimum sampling frequency
        omega_max: Maximum sampling frequency
    """
    configure_publication_style()

    # Define task distribution
    task_dist = TaskDistribution(
        dist_type='uniform',
        ranges={
            'alpha': alpha_range,
            'A': A_range,
            'omega_c': omega_c_range
        }
    )

    # Sample tasks
    rng = np.random.default_rng(seed)
    tasks = task_dist.sample(n_tasks, rng)

    # Setup PSD model and sampling frequencies
    psd_model = NoisePSDModel(model_type='one_over_f')
    sampling_freqs = np.logspace(np.log10(omega_min), np.log10(omega_max), n_freqs)

    # Create PSDToLindblad converter (basis_operators not needed for get_effective_rates)
    psd_to_lindblad = PSDToLindblad(
        basis_operators=[],  # Not used for get_effective_rates
        sampling_freqs=sampling_freqs,
        psd_model=psd_model
    )

    # Create figure
    fig, ax = plt.subplots(figsize=(7, 5))

    # Plot effective rates for each task
    for i, task in enumerate(tasks):
        rates = psd_to_lindblad.get_effective_rates(task)
        label = f'Task {i}: α={task.alpha:.2f}, A={task.A:.2f}, ωc={task.omega_c:.1f}'
        ax.loglog(sampling_freqs, rates, marker='o', label=label, linewidth=1.5, markersize=5)

    # Formatting
    ax.set_xlabel('Sampling Frequency ω (rad/s)')
    ax.set_ylabel('Effective Decay Rate Γ(ω)')
    ax.set_title('Effective Decay Rates from PSD (1/f model)')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    # Save figure
    save_figure(fig, output_path)
    plt.close(fig)


# ============================================================================
# CLI Commands
# ============================================================================

@app.command()
def noise(
    output: Path = typer.Argument(
        ...,
        help="Output file path (e.g., results/figures/noise.png)"
    ),
    model: str = typer.Option(
        'one_over_f',
        '--model', '-m',
        help="PSD model type: one_over_f, lorentzian, double_exp"
    ),
    n_tasks: int = typer.Option(
        5,
        '--n-tasks', '-n',
        help="Number of tasks to sample"
    ),
    seed: int = typer.Option(
        42,
        '--seed', '-s',
        help="Random seed for reproducibility"
    ),
    show_distances: bool = typer.Option(
        False,
        '--distances', '-d',
        help="Print pairwise PSD distances"
    )
):
    """
    Generate noise PSD plot showing sampled tasks.

    Example:
        uv run src/utils/plots.py noise results/figures/noise.png --n-tasks 5
    """
    plot_noise_psd(
        output_path=output,
        model_type=model,
        n_tasks=n_tasks,
        seed=seed,
        show_distances=show_distances
    )


@app.command()
def psd_comparison(
    output: Path = typer.Argument(
        ...,
        help="Output file path (e.g., results/figures/psd_comparison.png)"
    ),
    alpha: float = typer.Option(
        1.0,
        '--alpha', '-a',
        help="Spectral exponent"
    ),
    A: float = typer.Option(
        0.1,
        '--amplitude', '-A',
        help="Amplitude"
    ),
    omega_c: float = typer.Option(
        5.0,
        '--cutoff', '-w',
        help="Cutoff frequency"
    )
):
    """
    Generate PSD comparison plot for different models.

    Example:
        uv run src/utils/plots.py psd-comparison results/figures/psd_comp.png
    """
    plot_psd_comparison(
        output_path=output,
        alpha=alpha,
        A=A,
        omega_c=omega_c
    )


@app.command()
def effective_rates(
    output: Path = typer.Argument(
        ...,
        help="Output file path (e.g., results/figures/effective_rates.png)"
    ),
    n_tasks: int = typer.Option(
        5,
        '--n-tasks', '-n',
        help="Number of tasks to sample"
    ),
    seed: int = typer.Option(
        42,
        '--seed', '-s',
        help="Random seed for reproducibility"
    ),
    n_freqs: int = typer.Option(
        10,
        '--n-freqs', '-f',
        help="Number of sampling frequencies"
    )
):
    """
    Generate effective decay rates plot from PSD model.

    Example:
        uv run src/utils/plots.py effective-rates results/figures/effective_rates.png
    """
    plot_effective_rates(
        output_path=output,
        n_tasks=n_tasks,
        seed=seed,
        n_freqs=n_freqs
    )


if __name__ == "__main__":
    app()
