"""
Validation Script: Spectral Gap Analysis

This script analyzes the Lindblad superoperator eigenvalue spectrum:

A. Eigenvalue Spectrum: For 3 sampled tasks, plot all eigenvalues of L_θ
   - Lindblad superoperator L maps density matrices to density matrices
   - L has one zero eigenvalue (stationary state)
   - Spectral gap Δ = gap between 0 and next eigenvalue
   - Larger Δ → faster convergence to steady state

B. Spectral Gap vs Task Parameters: Δ(θ) across parameter space
   - Shows how Δ varies with noise parameters (α, A, ω_c)
   - Reveals which tasks have fast/slow dynamics
   - Connects to adaptation speed (larger Δ → easier control)

Theory:
- Lindblad superoperator: dρ/dt = L[ρ] = -i[H,ρ] + Σ_j(L_j ρ L_j† - {L_j†L_j, ρ}/2)
- Spectral gap Δ determines relaxation rate to steady state
- Larger Δ means noise dominates, faster decoherence
- Smaller Δ means coherent dynamics persist longer

Expected output: Two-panel plot showing eigenvalue spectra and gap variation
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Tuple
from typer import Typer
from tqdm import tqdm

from metaqctrl.quantum.noise_models import TaskDistribution, NoiseParameters
from metaqctrl.theory.quantum_environment import create_quantum_environment, get_target_state_from_config
from metaqctrl.theory.physics_constants import compute_spectral_gap

app = Typer()


def compute_lindblad_eigenvalues(env, task: NoiseParameters) -> Tuple[np.ndarray, float]:
    """
    Compute full eigenvalue spectrum of Lindblad superoperator.

    Args:
        env: QuantumEnvironment
        task: NoiseParameters

    Returns:
        eigenvalues: Array of eigenvalues (sorted by real part, descending)
        spectral_gap: Gap between largest two eigenvalues
    """
    L_ops = env.get_lindblad_operators(task)
    H0 = env.H0
    d = H0.shape[0]

    # Build Lindblad superoperator (d² × d² matrix)
    d2 = d * d
    L_super = np.zeros((d2, d2), dtype=complex)

    I = np.eye(d)

    # Hamiltonian part: -i[H, ρ]
    L_super += -1j * (np.kron(I, H0) - np.kron(H0.T, I))

    # Dissipation part
    for L in L_ops:
        # L_j ρ L_j†
        L_super += np.kron(L.conj(), L)

        # -½{L_j†L_j, ρ}
        anti_comm = L.conj().T @ L
        L_super -= 0.5 * np.kron(I, anti_comm)
        L_super -= 0.5 * np.kron(anti_comm.T, I)

    # Compute eigenvalues
    eigenvals = np.linalg.eigvals(L_super)
    eigenvals_real = np.real(eigenvals)

    # Sort in descending order
    eigenvals_sorted = np.sort(eigenvals_real)[::-1]

    # Spectral gap
    spectral_gap = eigenvals_sorted[0] - eigenvals_sorted[1]
    spectral_gap = abs(spectral_gap)

    return eigenvals_sorted, spectral_gap


def run_spectral_analysis_experiment(
    config: Dict,
    n_sample_tasks: int = 3,
    sweep_param: str = 'A',
    n_sweep_points: int = 30,
    output_dir: str = "results/spectral_gap_analysis"
) -> Dict:
    """
    Main experiment: Analyze Lindblad eigenvalue spectrum and spectral gap.

    Args:
        config: Experiment configuration
        n_sample_tasks: Number of tasks to sample for eigenvalue plots
        sweep_param: Parameter to sweep for Δ(θ) plot ('alpha', 'A', 'omega_c')
        n_sweep_points: Number of points in parameter sweep
        output_dir: Output directory

    Returns:
        results: Dict with eigenvalue data and spectral gaps
    """
    print("=" * 80)
    print("EXPERIMENT: Spectral Gap Analysis")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)

    print(f"\nExperiment parameters:")
    print(f"  Sample tasks for eigenvalue plots: {n_sample_tasks}")
    print(f"  Sweep parameter: {sweep_param}")
    print(f"  Sweep points: {n_sweep_points}")

    # Create environment
    print("\n[1/3] Creating quantum environment...")
    target_state = get_target_state_from_config(config)
    env = create_quantum_environment(config, target_state)

    # Sample tasks for eigenvalue spectrum plots
    print(f"\n[2/3] Sampling {n_sample_tasks} tasks for eigenvalue analysis...")
    task_dist = TaskDistribution(
        dist_type=config.get('task_dist_type', 'uniform'),
        ranges={
            'alpha': tuple(config['alpha_range']),
            'A': tuple(config['A_range']),
            'omega_c': tuple(config['omega_c_range'])
        }
    )
    sample_tasks = task_dist.sample(n_sample_tasks)

    # Compute eigenvalue spectra for sampled tasks
    eigenvalue_data = []

    for i, task in enumerate(sample_tasks):
        print(f"\n  Task {i+1}: α={task.alpha:.2f}, A={task.A:.3f}, ωc={task.omega_c:.2f}")

        eigenvals, gap = compute_lindblad_eigenvalues(env, task)

        print(f"    Spectral gap Δ = {gap:.6f}")
        print(f"    Largest eigenvalue: {eigenvals[0]:.6f}")
        print(f"    Second eigenvalue: {eigenvals[1]:.6f}")

        eigenvalue_data.append({
            'task_params': {
                'alpha': task.alpha,
                'A': task.A,
                'omega_c': task.omega_c
            },
            'eigenvalues': eigenvals.tolist(),
            'spectral_gap': float(gap)
        })

    # Sweep parameter to analyze Δ(θ)
    print(f"\n[3/3] Sweeping {sweep_param} to analyze spectral gap variation...")

    # Baseline task parameters
    baseline_params = {
        'alpha': np.mean(config.get('alpha_range', [0.5, 2.0])),
        'A': np.mean(config.get('A_range', [0.05, 0.3])),
        'omega_c': np.mean(config.get('omega_c_range', [2.0, 8.0]))
    }

    # Sweep range
    if sweep_param == 'alpha':
        sweep_range = config.get('alpha_range', [0.5, 2.0])
    elif sweep_param == 'A':
        sweep_range = config.get('A_range', [0.05, 0.3])
    elif sweep_param == 'omega_c':
        sweep_range = config.get('omega_c_range', [2.0, 8.0])
    else:
        raise ValueError(f"Unknown sweep parameter: {sweep_param}")

    # Extend range slightly
    sweep_min, sweep_max = sweep_range
    sweep_margin = (sweep_max - sweep_min) * 0.1
    sweep_values = np.linspace(sweep_min - sweep_margin, sweep_max + sweep_margin, n_sweep_points)

    spectral_gaps_sweep = []

    for sweep_val in tqdm(sweep_values, desc=f"Sweeping {sweep_param}"):
        # Create task with this parameter value
        task_params = baseline_params.copy()
        task_params[sweep_param] = sweep_val

        task = NoiseParameters(
            alpha=task_params['alpha'],
            A=task_params['A'],
            omega_c=task_params['omega_c']
        )

        # Compute spectral gap
        gap = compute_spectral_gap(env, task)
        spectral_gaps_sweep.append(gap)

    print(f"\n  Spectral gap range: [{min(spectral_gaps_sweep):.6f}, {max(spectral_gaps_sweep):.6f}]")
    print(f"  Mean: {np.mean(spectral_gaps_sweep):.6f}")
    print(f"  Std: {np.std(spectral_gaps_sweep):.6f}")

    # Save results
    print(f"\nSaving results to {output_dir}...")
    results = {
        'n_sample_tasks': n_sample_tasks,
        'eigenvalue_data': eigenvalue_data,
        'sweep_param': sweep_param,
        'sweep_values': sweep_values.tolist(),
        'spectral_gaps_sweep': spectral_gaps_sweep,
        'baseline_params': baseline_params,
        'config': config
    }

    results_path = f"{output_dir}/results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    return results


def plot_spectral_analysis(
    results: Dict,
    output_path: str = None
):
    """
    Generate two-panel plot: (A) Eigenvalue spectra, (B) Spectral gap variation.

    Args:
        results: Results dict from experiment
        output_path: Path to save figure
    """
    sns.set_style("whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Panel A: Eigenvalue spectra for sampled tasks
    eigenvalue_data = results['eigenvalue_data']
    colors = ['steelblue', 'orangered', 'green']

    for i, data in enumerate(eigenvalue_data):
        eigenvals = np.array(data['eigenvalues'])
        gap = data['spectral_gap']
        task_params = data['task_params']

        # Plot eigenvalue spectrum
        indices = np.arange(len(eigenvals))
        label = (f"Task {i+1}: α={task_params['alpha']:.2f}, "
                f"A={task_params['A']:.2f}, Δ={gap:.4f}")

        ax1.scatter(indices, eigenvals, s=50, alpha=0.7, color=colors[i % len(colors)],
                   label=label, edgecolors='black', linewidths=0.5)

        # Annotate spectral gap
        if i == 0:  # Only annotate first task for clarity
            # Draw arrow showing spectral gap
            ax1.annotate('', xy=(0, eigenvals[1]), xytext=(0, eigenvals[0]),
                        arrowprops=dict(arrowstyle='<->', color='red', lw=2.5))
            ax1.text(0.5, (eigenvals[0] + eigenvals[1]) / 2,
                    f'Δ = {gap:.4f}',
                    fontsize=11, color='red', fontweight='bold',
                    verticalalignment='center')

    # Reference line at zero
    ax1.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.5,
               label='Zero eigenvalue (stationary)')

    ax1.set_xlabel('Eigenvalue Index (sorted)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Eigenvalue (Real Part)', fontsize=13, fontweight='bold')
    ax1.set_title('(A) Lindblad Superoperator Eigenvalue Spectrum',
                 fontsize=15, fontweight='bold')
    ax1.legend(fontsize=10, loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=11)

    # Add text box with explanation
    textstr = (
        'Lindblad Spectrum:\n'
        '• Zero eigenvalue = stationary state\n'
        '• Spectral gap Δ = convergence rate\n'
        '• Larger Δ → faster decoherence'
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.98, 0.98, textstr, transform=ax1.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=props)

    # Panel B: Spectral gap vs task parameter
    sweep_param = results['sweep_param']
    sweep_values = np.array(results['sweep_values'])
    spectral_gaps = np.array(results['spectral_gaps_sweep'])

    ax2.plot(sweep_values, spectral_gaps, 'o-', linewidth=2.5, markersize=7,
            color='purple', alpha=0.8)

    # Mark baseline parameter value
    baseline_val = results['baseline_params'][sweep_param]
    baseline_idx = np.argmin(np.abs(sweep_values - baseline_val))
    baseline_gap = spectral_gaps[baseline_idx]

    ax2.axvline(baseline_val, color='gray', linestyle='--', linewidth=2, alpha=0.5,
               label=f'Baseline ({baseline_val:.2f})')
    ax2.plot(baseline_val, baseline_gap, '*', markersize=20, color='gold',
            edgecolor='black', linewidth=1.5, zorder=5,
            label=f'Δ = {baseline_gap:.4f}')

    # Parameter labels
    param_labels = {
        'alpha': r'Spectral Exponent ($\alpha$)',
        'A': r'Noise Amplitude ($A$)',
        'omega_c': r'Cutoff Frequency ($\omega_c$)'
    }
    param_label = param_labels.get(sweep_param, sweep_param)

    ax2.set_xlabel(param_label, fontsize=13, fontweight='bold')
    ax2.set_ylabel(r'Spectral Gap $\Delta(\theta)$', fontsize=13, fontweight='bold')
    ax2.set_title(f'(B) Spectral Gap Variation vs {param_label}',
                 fontsize=15, fontweight='bold')
    ax2.legend(fontsize=11, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=11)

    # Add shaded regions for interpretation
    gap_mean = np.mean(spectral_gaps)
    gap_std = np.std(spectral_gaps)

    ax2.fill_between(sweep_values,
                     gap_mean - gap_std,
                     gap_mean + gap_std,
                     alpha=0.2, color='purple',
                     label=f'Mean ± σ')

    # Add text box with statistics
    textstr = (
        f'Statistics:\n'
        f'Mean: {gap_mean:.6f}\n'
        f'Std: {gap_std:.6f}\n'
        f'Range: [{np.min(spectral_gaps):.6f},\n'
        f'        {np.max(spectral_gaps):.6f}]'
    )
    props = dict(boxstyle='round', facecolor='lavender', alpha=0.8)
    ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=props, family='monospace')

    plt.suptitle('Lindblad Superoperator Spectral Analysis',
                fontsize=17, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to {output_path}")

    plt.close()


def plot_spectral_gap_heatmap(
    config: Dict,
    output_dir: str,
    n_points: int = 30
):
    """
    Generate 2D heatmap of spectral gap across two parameters.

    Args:
        config: Configuration dict
        output_dir: Output directory
        n_points: Points per dimension
    """
    print("\n" + "=" * 60)
    print("Generating 2D spectral gap heatmap...")
    print("=" * 60)

    # Create environment
    target_state = get_target_state_from_config(config)
    env = create_quantum_environment(config, target_state)

    # Parameter ranges
    alpha_range = config.get('alpha_range', [0.5, 2.0])
    A_range = config.get('A_range', [0.05, 0.3])
    omega_c_baseline = np.mean(config.get('omega_c_range', [2.0, 8.0]))

    # Create grid
    alpha_values = np.linspace(alpha_range[0], alpha_range[1], n_points)
    A_values = np.linspace(A_range[0], A_range[1], n_points)

    spectral_gap_grid = np.zeros((n_points, n_points))

    print(f"Computing spectral gaps on {n_points}x{n_points} grid...")
    for i, alpha in enumerate(tqdm(alpha_values, desc="Alpha sweep")):
        for j, A in enumerate(A_values):
            task = NoiseParameters(
                alpha=alpha,
                A=A,
                omega_c=omega_c_baseline
            )

            gap = compute_spectral_gap(env, task)
            spectral_gap_grid[i, j] = gap

    # Plot heatmap
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(spectral_gap_grid.T, origin='lower', aspect='auto',
                   cmap='viridis', interpolation='bilinear',
                   extent=[alpha_range[0], alpha_range[1], A_range[0], A_range[1]])

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r'Spectral Gap $\Delta(\alpha, A)$', fontsize=13, fontweight='bold')

    ax.set_xlabel(r'Spectral Exponent ($\alpha$)', fontsize=13, fontweight='bold')
    ax.set_ylabel(r'Noise Amplitude ($A$)', fontsize=13, fontweight='bold')
    ax.set_title(f'Spectral Gap Heatmap (ωc = {omega_c_baseline:.2f})',
                fontsize=15, fontweight='bold')

    # Add contour lines
    contours = ax.contour(alpha_values, A_values, spectral_gap_grid.T,
                         levels=5, colors='white', linewidths=1, alpha=0.5)
    ax.clabel(contours, inline=True, fontsize=9, fmt='%.4f')

    plt.tight_layout()

    heatmap_path = f"{output_dir}/spectral_gap_heatmap.pdf"
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to {heatmap_path}")

    heatmap_path_png = f"{output_dir}/spectral_gap_heatmap.png"
    plt.savefig(heatmap_path_png, dpi=300, bbox_inches='tight')

    plt.close()


@app.command()
def main(
    output_dir: Path = Path("results/spectral_gap_analysis"),
    n_sample_tasks: int = 3,
    sweep_param: str = 'A',
    n_sweep_points: int = 30,
    generate_heatmap: bool = True
):
    """
    Run spectral gap analysis experiment.

    Args:
        output_dir: Directory to save results and figures
        n_sample_tasks: Number of tasks to sample for eigenvalue plots (default 3)
        sweep_param: Parameter to sweep ('alpha', 'A', 'omega_c') (default 'A')
        n_sweep_points: Number of points in parameter sweep (default 30)
        generate_heatmap: Generate 2D heatmap of spectral gap (default True)
    """
    # Configuration matching training setup
    config = {
        'num_qubits': 1,
        'n_controls': 2,
        'n_segments': 100,
        'horizon': 1.0,
        'target_gate': 'pauli_x',
        'hidden_dim': 256,
        'n_hidden_layers': 2,
        'inner_lr': 0.01,
        'alpha_range': [0.5, 2.0],
        'A_range': [0.05, 0.3],
        'omega_c_range': [2.0, 8.0],
        'noise_frequencies': [1.0, 5.0, 10.0]
    }

    print("Configuration loaded")

    # Run experiment
    results = run_spectral_analysis_experiment(
        config=config,
        n_sample_tasks=n_sample_tasks,
        sweep_param=sweep_param,
        n_sweep_points=n_sweep_points,
        output_dir=str(output_dir)
    )

    # Generate main plot (eigenvalue spectra + gap variation)
    plot_path = f"{output_dir}/spectral_gap_analysis.pdf"
    plot_spectral_analysis(results, output_path=plot_path)

    plot_path_png = f"{output_dir}/spectral_gap_analysis.png"
    plot_spectral_analysis(results, output_path=plot_path_png)

    # Generate heatmap
    if generate_heatmap:
        plot_spectral_gap_heatmap(
            config=config,
            output_dir=str(output_dir),
            n_points=30
        )

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)
    print("\nKey insights:")

    for i, data in enumerate(results['eigenvalue_data']):
        task_params = data['task_params']
        gap = data['spectral_gap']
        print(f"  • Task {i+1} (α={task_params['alpha']:.2f}, "
              f"A={task_params['A']:.2f}): Δ = {gap:.6f}")

    gaps = results['spectral_gaps_sweep']
    print(f"\n  • Spectral gap range: [{min(gaps):.6f}, {max(gaps):.6f}]")
    print(f"  • Mean spectral gap: {np.mean(gaps):.6f}")

    print("\nOutputs:")
    print(f"  • Main plot: {output_dir}/spectral_gap_analysis.pdf")
    if generate_heatmap:
        print(f"  • Heatmap: {output_dir}/spectral_gap_heatmap.pdf")


if __name__ == "__main__":
    app()
