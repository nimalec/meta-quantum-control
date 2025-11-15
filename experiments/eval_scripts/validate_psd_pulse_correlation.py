"""
Validation Script: PSD-Pulse Distance Correlation and Filter Analysis

This script analyzes the relationship between task similarity in noise space
and control pulse similarity:

A. Pulse Distance vs PSD Distance Scatter Plot:
   - For 100 task pairs, compute:
     * Pulse distance: ‖u_θ - u_θ'‖_L² (how different are the controls?)
     * PSD distance: ‖S_θ - S_θ'‖_L² (how different are the noise spectra?)
   - Linear fit: ‖Δu‖ ≈ C_filter · ‖ΔS‖
   - C_filter extracts filter constant (control susceptibility to noise)

B. Noise PSD Overlay Plot:
   - S(ω;θ) for different task parameters
   - Control susceptibility χ(ω) showing frequency response
   - Control bandwidth Ω_ctrl shaded region
   - Shows which noise frequencies affect control

Theory:
- Similar noise → similar optimal controls (if C_filter is well-defined)
- C_filter relates noise differences to pulse corrections
- Control bandwidth limits which noise frequencies matter

Expected output: Two-panel plot showing correlation and PSD analysis
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
import copy
from tqdm import tqdm
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

from metaqctrl.quantum.noise_models import TaskDistribution, NoiseParameters
from metaqctrl.quantum.noise_models_v2 import NoisePSDModel
from metaqctrl.meta_rl.policy import PulsePolicy
from metaqctrl.theory.quantum_environment import create_quantum_environment, get_target_state_from_config
from metaqctrl.utils.checkpoint_utils import load_policy_from_checkpoint

app = Typer()


def compute_pulse_for_task(
    policy: torch.nn.Module,
    task: NoiseParameters,
    config: Dict,
    env
) -> np.ndarray:
    """
    Compute optimal pulse for a task.

    Args:
        policy: Policy network
        task: Task parameters
        config: Configuration dict
        env: QuantumEnvironment

    Returns:
        controls: Control sequence (n_segments, n_controls)
    """
    task_features = torch.tensor(
        [task.alpha, task.A, task.omega_c],
        dtype=torch.float32
    )

    # Adapt policy to task
    adapted_policy = copy.deepcopy(policy)
    adapted_policy.train()

    inner_lr = config.get('inner_lr', 0.01)
    K_adapt = config.get('inner_steps', 5)

    for k in range(K_adapt):
        loss = env.compute_loss_differentiable(
            adapted_policy, task, device=torch.device('cpu')
        )
        loss.backward()
        with torch.no_grad():
            for param in adapted_policy.parameters():
                if param.grad is not None:
                    param -= inner_lr * param.grad
                    param.grad.zero_()

    # Generate controls
    adapted_policy.eval()
    with torch.no_grad():
        controls = adapted_policy(task_features).detach().numpy()

    return controls


def compute_pulse_distance(
    u1: np.ndarray,
    u2: np.ndarray,
    dt: float = 0.01
) -> float:
    """
    Compute L² norm of pulse difference.

    ‖u_θ - u_θ'‖_L² = sqrt(∫ |u_θ(t) - u_θ'(t)|² dt)

    Args:
        u1: First pulse sequence (n_segments, n_controls)
        u2: Second pulse sequence
        dt: Time step

    Returns:
        distance: L² distance
    """
    diff = u1 - u2
    # Sum over controls and integrate over time
    distance_sq = np.sum(diff**2) * dt
    return np.sqrt(distance_sq)


def compute_psd_distance(
    task1: NoiseParameters,
    task2: NoiseParameters,
    omega_max: float = 100.0,
    n_omega: int = 1000
) -> float:
    """
    Compute L² norm of PSD difference.

    ‖S_θ - S_θ'‖_L² = sqrt(∫ |S(ω;θ) - S(ω;θ')|² dω)

    Args:
        task1: First task parameters
        task2: Second task parameters
        omega_max: Maximum frequency for integration
        n_omega: Number of frequency points

    Returns:
        distance: L² distance in PSD space
    """
    psd_model = NoisePSDModel(model_type='one_over_f')

    omega = np.linspace(0, omega_max, n_omega)
    dw = omega[1] - omega[0]

    S1 = psd_model.psd(omega, task1)
    S2 = psd_model.psd(omega, task2)

    diff = S1 - S2
    distance_sq = np.sum(diff**2) * dw
    return np.sqrt(distance_sq)


def run_psd_pulse_correlation_experiment(
    meta_policy_path: str,
    config: Dict,
    n_pairs: int = 100,
    output_dir: str = "results/psd_pulse_correlation"
) -> Dict:
    """
    Main experiment: Analyze correlation between PSD distance and pulse distance.

    Args:
        meta_policy_path: Path to trained meta policy
        config: Experiment configuration
        n_pairs: Number of task pairs to analyze
        output_dir: Output directory

    Returns:
        results: Dict with distances and filter constant
    """
    print("=" * 80)
    print("EXPERIMENT: PSD-Pulse Distance Correlation")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)

    print(f"\nExperiment parameters:")
    print(f"  Number of task pairs: {n_pairs}")

    # Load meta policy
    print("\n[1/4] Loading meta policy...")
    meta_policy = load_policy_from_checkpoint(
        meta_policy_path, config, eval_mode=False, verbose=True
    )

    # Create environment
    print("\n[2/4] Creating quantum environment...")
    target_state = get_target_state_from_config(config)
    env = create_quantum_environment(config, target_state)

    # Sample tasks
    print(f"\n[3/4] Sampling {n_pairs * 2} tasks...")
    task_dist = TaskDistribution(
        dist_type=config.get('task_dist_type', 'uniform'),
        ranges={
            'alpha': tuple(config['alpha_range']),
            'A': tuple(config['A_range']),
            'omega_c': tuple(config['omega_c_range'])
        }
    )
    all_tasks = task_dist.sample(n_pairs * 2)

    # Create task pairs
    task_pairs = [(all_tasks[i], all_tasks[i + n_pairs]) for i in range(n_pairs)]

    # Compute distances
    print(f"\n[4/4] Computing pulse and PSD distances for {n_pairs} pairs...")

    pulse_distances = []
    psd_distances = []

    dt = config.get('horizon', 1.0) / config.get('n_segments', 100)

    for i, (task1, task2) in enumerate(tqdm(task_pairs, desc="Processing pairs")):
        # Compute pulses
        u1 = compute_pulse_for_task(meta_policy, task1, config, env)
        u2 = compute_pulse_for_task(meta_policy, task2, config, env)

        # Compute distances
        pulse_dist = compute_pulse_distance(u1, u2, dt)
        psd_dist = compute_psd_distance(task1, task2)

        pulse_distances.append(pulse_dist)
        psd_distances.append(psd_dist)

    pulse_distances = np.array(pulse_distances)
    psd_distances = np.array(psd_distances)

    # Linear fit: ‖Δu‖ = C_filter · ‖ΔS‖
    print("\nFitting linear relationship...")

    def linear_model(x, C_filter):
        return C_filter * x

    try:
        popt, pcov = curve_fit(linear_model, psd_distances, pulse_distances)
        C_filter = popt[0]

        pulse_pred = linear_model(psd_distances, C_filter)
        r2 = r2_score(pulse_distances, pulse_pred)

        print(f"  C_filter = {C_filter:.6f}")
        print(f"  R² = {r2:.4f}")

        fit_success = True
    except Exception as e:
        print(f"  Fitting failed: {e}")
        C_filter = r2 = None
        fit_success = False

    # Save results
    print(f"\nSaving results to {output_dir}...")
    results = {
        'n_pairs': n_pairs,
        'pulse_distances': pulse_distances.tolist(),
        'psd_distances': psd_distances.tolist(),
        'fit': {
            'success': fit_success,
            'C_filter': float(C_filter) if fit_success else None,
            'r2': float(r2) if fit_success else None
        },
        'config': config
    }

    results_path = f"{output_dir}/results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    return results


def plot_psd_pulse_correlation(
    results: Dict,
    output_path: str = None
):
    """
    Generate scatter plot of pulse distance vs PSD distance.

    Args:
        results: Results dict from experiment
        output_path: Path to save figure
    """
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 8))

    psd_dist = np.array(results['psd_distances'])
    pulse_dist = np.array(results['pulse_distances'])

    # Scatter plot
    ax.scatter(psd_dist, pulse_dist, alpha=0.6, s=50, color='steelblue',
              edgecolors='black', linewidths=0.5, label='Task pairs')

    # Linear fit
    if results['fit']['success']:
        C_filter = results['fit']['C_filter']
        r2 = results['fit']['r2']

        psd_fine = np.linspace(0, np.max(psd_dist) * 1.1, 100)
        pulse_fit = C_filter * psd_fine

        ax.plot(psd_fine, pulse_fit, 'r--', linewidth=3,
               label=f'Linear fit: $\\|\\Delta u\\| = C_{{filter}} \\|\\Delta S\\|$\n'
                     f'$C_{{filter}} = {C_filter:.6f}$, $R^2 = {r2:.4f}$')

    ax.set_xlabel(r'PSD Distance $\|\Delta S\|_{L^2}$ [a.u.]',
                 fontsize=14, fontweight='bold')
    ax.set_ylabel(r'Pulse Distance $\|\Delta u\|_{L^2}$ [a.u.]',
                 fontsize=14, fontweight='bold')
    ax.set_title('Pulse-PSD Distance Correlation: Extracting Filter Constant',
                fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)

    # Add text box with interpretation
    textstr = (
        'Interpretation:\n'
        '• C_filter: Control susceptibility to noise\n'
        '• Linear relationship validates theory\n'
        '• Similar noise → similar controls'
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.98, 0.02, textstr, transform=ax.transAxes,
           fontsize=11, verticalalignment='bottom', horizontalalignment='right',
           bbox=props)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nCorrelation plot saved to {output_path}")

    plt.close()


def plot_noise_psd_overlay(
    config: Dict,
    output_path: str = None,
    n_tasks: int = 5
):
    """
    Generate plot showing noise PSDs with control bandwidth and susceptibility.

    Args:
        config: Configuration dict
        output_path: Path to save figure
        n_tasks: Number of example tasks to plot
    """
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 8))

    # Sample diverse tasks
    task_dist = TaskDistribution(
        dist_type='uniform',
        ranges={
            'alpha': tuple(config['alpha_range']),
            'A': tuple(config['A_range']),
            'omega_c': tuple(config['omega_c_range'])
        }
    )
    tasks = task_dist.sample(n_tasks)

    # Frequency range
    omega = np.logspace(-2, 2, 1000)  # Log scale from 0.01 to 100

    # PSD model
    psd_model = NoisePSDModel(model_type='one_over_f')

    # Plot PSDs for different tasks
    colors = plt.cm.viridis(np.linspace(0, 1, n_tasks))

    for i, task in enumerate(tasks):
        S_omega = psd_model.psd(omega, task)

        label = f'Task {i+1}: α={task.alpha:.2f}, A={task.A:.2f}, ωc={task.omega_c:.1f}'
        ax.loglog(omega, S_omega, linewidth=2.5, color=colors[i],
                 label=label, alpha=0.8)

    # Control bandwidth (estimate from discretization)
    T = config.get('horizon', 1.0)
    n_segments = config.get('n_segments', 100)
    omega_ctrl = n_segments / T  # Nyquist-like estimate

    # Shade control bandwidth region
    ax.axvspan(0, omega_ctrl, alpha=0.15, color='red',
              label=f'Control bandwidth $\\Omega_{{ctrl}} \\approx {omega_ctrl:.1f}$')

    # Control susceptibility (simplified model: 1 in band, decay outside)
    chi_omega = np.ones_like(omega)
    chi_omega[omega > omega_ctrl] = (omega_ctrl / omega[omega > omega_ctrl])**2

    # Plot susceptibility on secondary y-axis
    ax2 = ax.twinx()
    ax2.plot(omega, chi_omega, 'k--', linewidth=3, alpha=0.7,
            label=r'Control susceptibility $\chi(\omega)$')
    ax2.set_ylabel(r'Control Susceptibility $\chi(\omega)$ [a.u.]',
                  fontsize=13, fontweight='bold')
    ax2.set_ylim([0, 1.5])
    ax2.tick_params(labelsize=11)

    # Mark cutoff
    ax.axvline(omega_ctrl, color='red', linestyle=':', linewidth=2.5,
              alpha=0.7)

    ax.set_xlabel(r'Frequency $\omega$ [rad/s]', fontsize=14, fontweight='bold')
    ax.set_ylabel(r'Noise PSD $S(\omega; \theta)$ [a.u.]', fontsize=14, fontweight='bold')
    ax.set_title('Noise Power Spectral Density with Control Bandwidth',
                fontsize=16, fontweight='bold')

    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='upper right')

    ax.grid(True, alpha=0.3, which='both')
    ax.tick_params(labelsize=11)

    # Add text box with explanation
    textstr = (
        'Key:\n'
        '• S(ω;θ): Noise spectrum (varies with task)\n'
        '• χ(ω): Control frequency response\n'
        f'• Ω_ctrl ≈ {omega_ctrl:.1f}: Control bandwidth\n'
        '• Noise beyond Ω_ctrl has less impact'
    )
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.9)
    ax.text(0.02, 0.02, textstr, transform=ax.transAxes,
           fontsize=11, verticalalignment='bottom',
           bbox=props)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"PSD overlay plot saved to {output_path}")

    plt.close()


def plot_combined_analysis(
    results: Dict,
    config: Dict,
    output_path: str = None
):
    """
    Generate combined two-panel plot: correlation + PSD overlay.

    Args:
        results: Results dict from correlation experiment
        config: Configuration dict
        output_path: Path to save figure
    """
    sns.set_style("whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Panel A: Pulse-PSD correlation
    psd_dist = np.array(results['psd_distances'])
    pulse_dist = np.array(results['pulse_distances'])

    ax1.scatter(psd_dist, pulse_dist, alpha=0.6, s=50, color='steelblue',
               edgecolors='black', linewidths=0.5, label='Task pairs')

    if results['fit']['success']:
        C_filter = results['fit']['C_filter']
        r2 = results['fit']['r2']

        psd_fine = np.linspace(0, np.max(psd_dist) * 1.1, 100)
        pulse_fit = C_filter * psd_fine

        ax1.plot(psd_fine, pulse_fit, 'r--', linewidth=3,
                label=f'$C_{{filter}} = {C_filter:.6f}$, $R^2 = {r2:.3f}$')

    ax1.set_xlabel(r'PSD Distance $\|\Delta S\|_{L^2}$', fontsize=13, fontweight='bold')
    ax1.set_ylabel(r'Pulse Distance $\|\Delta u\|_{L^2}$', fontsize=13, fontweight='bold')
    ax1.set_title('(A) Pulse-PSD Correlation', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Panel B: Noise PSD overlay
    task_dist = TaskDistribution(
        dist_type='uniform',
        ranges={
            'alpha': tuple(config['alpha_range']),
            'A': tuple(config['A_range']),
            'omega_c': tuple(config['omega_c_range'])
        }
    )
    tasks = task_dist.sample(4)

    omega = np.logspace(-2, 2, 1000)
    psd_model = NoisePSDModel(model_type='one_over_f')
    colors = plt.cm.viridis(np.linspace(0, 1, 4))

    for i, task in enumerate(tasks):
        S_omega = psd_model.psd(omega, task)
        ax2.loglog(omega, S_omega, linewidth=2.5, color=colors[i],
                  label=f'α={task.alpha:.1f}, A={task.A:.2f}', alpha=0.8)

    T = config.get('horizon', 1.0)
    n_segments = config.get('n_segments', 100)
    omega_ctrl = n_segments / T

    ax2.axvspan(0, omega_ctrl, alpha=0.15, color='red',
               label=f'$\\Omega_{{ctrl}} \\approx {omega_ctrl:.1f}$')
    ax2.axvline(omega_ctrl, color='red', linestyle=':', linewidth=2.5, alpha=0.7)

    ax2.set_xlabel(r'Frequency $\omega$ [rad/s]', fontsize=13, fontweight='bold')
    ax2.set_ylabel(r'Noise PSD $S(\omega; \theta)$', fontsize=13, fontweight='bold')
    ax2.set_title('(B) Noise Spectra & Control Bandwidth', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(True, alpha=0.3, which='both')

    plt.suptitle('Filter Constant Analysis: PSD-Pulse Correlation',
                fontsize=17, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nCombined analysis saved to {output_path}")

    plt.close()


@app.command()
def main(
    meta_path: Path = Path("experiments/train_scripts/checkpoints/maml_best_policy.pt"),
    output_dir: Path = Path("results/psd_pulse_correlation"),
    n_pairs: int = 100
):
    """
    Run PSD-pulse correlation experiment.

    Args:
        meta_path: Path to trained meta policy checkpoint
        output_dir: Directory to save results and figures
        n_pairs: Number of task pairs to analyze (default 100)
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
        'inner_steps': 5,
        'alpha_range': [0.5, 2.0],
        'A_range': [0.05, 0.3],
        'omega_c_range': [2.0, 8.0],
        'noise_frequencies': [1.0, 5.0, 10.0]
    }

    # Check if policy path exists
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta policy not found: {meta_path}")

    print(f"Using meta policy: {meta_path}")

    # Run correlation experiment
    results = run_psd_pulse_correlation_experiment(
        meta_policy_path=str(meta_path),
        config=config,
        n_pairs=n_pairs,
        output_dir=str(output_dir)
    )

    # Generate individual plots
    # 1. Correlation plot
    correlation_path = f"{output_dir}/pulse_psd_correlation.pdf"
    plot_psd_pulse_correlation(results, output_path=correlation_path)

    correlation_path_png = f"{output_dir}/pulse_psd_correlation.png"
    plot_psd_pulse_correlation(results, output_path=correlation_path_png)

    # 2. PSD overlay plot
    psd_path = f"{output_dir}/noise_psd_overlay.pdf"
    plot_noise_psd_overlay(config, output_path=psd_path, n_tasks=5)

    psd_path_png = f"{output_dir}/noise_psd_overlay.png"
    plot_noise_psd_overlay(config, output_path=psd_path_png, n_tasks=5)

    # 3. Combined plot
    combined_path = f"{output_dir}/combined_analysis.pdf"
    plot_combined_analysis(results, config, output_path=combined_path)

    combined_path_png = f"{output_dir}/combined_analysis.png"
    plot_combined_analysis(results, config, output_path=combined_path_png)

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)

    if results['fit']['success']:
        print(f"\nFilter Constant Analysis:")
        print(f"  • C_filter = {results['fit']['C_filter']:.6f}")
        print(f"  • R² = {results['fit']['r2']:.4f}")
        print(f"  • Interpretation: Pulse distance scales linearly with PSD distance")

    print("\nOutputs:")
    print(f"  • Correlation plot: {output_dir}/pulse_psd_correlation.pdf")
    print(f"  • PSD overlay: {output_dir}/noise_psd_overlay.pdf")
    print(f"  • Combined analysis: {output_dir}/combined_analysis.pdf")


if __name__ == "__main__":
    app()
