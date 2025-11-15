"""
Validation Script 1: Fidelity Landscape Across Task Parameters

This script generates fidelity landscapes by sweeping each noise parameter
(alpha, A, omega_c) individually while holding others constant.

Generates two curves for comparison:
  (a) Meta-adapted policy (with K adaptation steps)
  (b) Robust baseline policy (trained on average noise parameters, no adaptation)

Expected output: 3 plots showing how fidelity varies with each noise parameter
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

from metaqctrl.quantum.noise_models import TaskDistribution, NoiseParameters
from metaqctrl.meta_rl.policy import PulsePolicy
from metaqctrl.theory.quantum_environment import create_quantum_environment, get_target_state_from_config
from metaqctrl.utils.checkpoint_utils import load_policy_from_checkpoint

app = Typer()


def generate_fidelity_landscape(
    meta_policy_path: str,
    robust_policy_path: str,
    config: Dict,
    sweep_param: str = 'alpha',  # 'alpha', 'A', or 'omega_c'
    sweep_range: Tuple[float, float] = None,
    n_points: int = 20,
    K_adapt: int = 5,
    baseline_params: Dict = None
) -> Dict:
    """
    Generate fidelity landscape by sweeping one noise parameter.

    Args:
        meta_policy_path: Path to trained meta policy
        robust_policy_path: Path to robust baseline policy
        config: Experiment configuration
        sweep_param: Which parameter to sweep ('alpha', 'A', 'omega_c')
        sweep_range: (min, max) range for sweep
        n_points: Number of points in sweep
        K_adapt: Number of adaptation steps for meta policy
        baseline_params: Baseline task parameters for non-swept params

    Returns:
        results: Dict with sweep values and fidelities
    """
    print(f"\n{'='*80}")
    print(f"FIDELITY LANDSCAPE: Sweeping {sweep_param}")
    print(f"{'='*80}")

    # Load policies
    print("\n[1/5] Loading policies...")
    meta_policy_template = load_policy_from_checkpoint(
        meta_policy_path, config, eval_mode=False, verbose=True
    )
    robust_policy = load_policy_from_checkpoint(
        robust_policy_path, config, eval_mode=True, verbose=True
    )

    # Create quantum environment
    print("\n[2/5] Creating quantum environment...")
    target_state = get_target_state_from_config(config)
    env = create_quantum_environment(config, target_state)

    # Set up baseline parameters (average of training distribution)
    if baseline_params is None:
        baseline_params = {
            'alpha': np.mean(config.get('alpha_range', [0.5, 2.0])),
            'A': np.mean(config.get('A_range', [0.05, 0.3])),
            'omega_c': np.mean(config.get('omega_c_range', [2.0, 8.0]))
        }
    print(f"\nBaseline parameters: {baseline_params}")

    # Set up sweep range
    if sweep_range is None:
        if sweep_param == 'alpha':
            sweep_range = config.get('alpha_range', [0.5, 2.0])
        elif sweep_param == 'A':
            sweep_range = config.get('A_range', [0.05, 0.3])
        elif sweep_param == 'omega_c':
            sweep_range = config.get('omega_c_range', [2.0, 8.0])
        else:
            raise ValueError(f"Unknown sweep parameter: {sweep_param}")

    # Extend range slightly for better visualization
    sweep_min, sweep_max = sweep_range
    sweep_margin = (sweep_max - sweep_min) * 0.1
    sweep_values = np.linspace(sweep_min - sweep_margin, sweep_max + sweep_margin, n_points)

    print(f"\n[3/5] Sweeping {sweep_param} from {sweep_values[0]:.3f} to {sweep_values[-1]:.3f}")
    print(f"  Number of points: {n_points}")
    print(f"  Adaptation steps (meta): K = {K_adapt}")

    # Sweep and evaluate
    print(f"\n[4/5] Evaluating fidelities...")
    fidelities_meta = []
    fidelities_robust = []

    for i, sweep_val in enumerate(sweep_values):
        print(f"  Point {i+1}/{n_points}: {sweep_param} = {sweep_val:.4f}...", end=' ')

        # Create task with this parameter value
        task_params = baseline_params.copy()
        task_params[sweep_param] = sweep_val

        task = NoiseParameters(
            alpha=task_params['alpha'],
            A=task_params['A'],
            omega_c=task_params['omega_c']
        )

        task_features = torch.tensor(
            [task.alpha, task.A, task.omega_c],
            dtype=torch.float32
        )

        # Evaluate robust baseline (no adaptation)
        robust_policy.eval()
        with torch.no_grad():
            controls_robust = robust_policy(task_features).detach().numpy()
        fid_robust = env.compute_fidelity(controls_robust, task)
        fidelities_robust.append(fid_robust)

        # Evaluate meta policy (with K adaptation steps)
        adapted_policy = copy.deepcopy(meta_policy_template)
        adapted_policy.train()

        for k in range(K_adapt):
            loss = env.compute_loss_differentiable(
                adapted_policy, task, device=torch.device('cpu')
            )
            loss.backward()
            with torch.no_grad():
                for param in adapted_policy.parameters():
                    if param.grad is not None:
                        param -= config['inner_lr'] * param.grad
                        param.grad.zero_()

        adapted_policy.eval()
        with torch.no_grad():
            controls_meta = adapted_policy(task_features).detach().numpy()
        fid_meta = env.compute_fidelity(controls_meta, task)
        fidelities_meta.append(fid_meta)

        print(f"Meta: {fid_meta:.4f}, Robust: {fid_robust:.4f}, Gap: {fid_meta - fid_robust:.4f}")

    print(f"\n[5/5] Landscape complete!")

    results = {
        'sweep_param': sweep_param,
        'sweep_values': sweep_values.tolist(),
        'fidelities_meta': fidelities_meta,
        'fidelities_robust': fidelities_robust,
        'baseline_params': baseline_params,
        'K_adapt': K_adapt,
        'config': config
    }

    return results


def plot_fidelity_landscape(
    results: Dict,
    output_path: str = None,
    show_gap: bool = True
):
    """
    Plot fidelity landscape with meta and robust curves.

    Args:
        results: Results dict from generate_fidelity_landscape
        output_path: Path to save figure
        show_gap: If True, add gap subplot
    """
    sns.set_style("whitegrid")

    if show_gap:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

    sweep_param = results['sweep_param']
    sweep_values = np.array(results['sweep_values'])
    fid_meta = np.array(results['fidelities_meta'])
    fid_robust = np.array(results['fidelities_robust'])

    # Parameter labels
    param_labels = {
        'alpha': r'Spectral Exponent ($\alpha$)',
        'A': r'Noise Amplitude ($A$)',
        'omega_c': r'Cutoff Frequency ($\omega_c$) [a.u.]'
    }
    param_label = param_labels.get(sweep_param, sweep_param)

    # Plot fidelities
    ax1.plot(sweep_values, fid_meta, 'o-', linewidth=2.5, markersize=8,
             label=f'Meta-Adapted (K={results["K_adapt"]})', color='steelblue')
    ax1.plot(sweep_values, fid_robust, 's-', linewidth=2.5, markersize=8,
             label='Robust Baseline', color='orangered', alpha=0.8)

    ax1.set_xlabel(param_label, fontsize=14, fontweight='bold')
    ax1.set_ylabel('Gate Fidelity', fontsize=14, fontweight='bold')
    ax1.set_title(f'Fidelity Landscape vs {param_label}', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=12, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=12)
    ax1.set_ylim([0.0, 1.05])

    # Add horizontal line at baseline parameter value
    baseline_val = results['baseline_params'][sweep_param]
    ax1.axvline(baseline_val, color='gray', linestyle='--', linewidth=2, alpha=0.5,
                label=f'Training Center ({baseline_val:.2f})')
    ax1.legend(fontsize=12, loc='best')

    if show_gap:
        # Plot adaptation gap
        gap = fid_meta - fid_robust
        ax2.plot(sweep_values, gap, 'o-', linewidth=2.5, markersize=8,
                 color='green')
        ax2.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)
        ax2.axvline(baseline_val, color='gray', linestyle='--', linewidth=2, alpha=0.5)

        ax2.set_xlabel(param_label, fontsize=14, fontweight='bold')
        ax2.set_ylabel('Adaptation Gap\n(Meta - Robust)', fontsize=14, fontweight='bold')
        ax2.set_title('Adaptation Advantage Across Parameter Space', fontsize=15, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(labelsize=12)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to {output_path}")

    plt.close()


def generate_all_landscapes(
    meta_policy_path: str,
    robust_policy_path: str,
    config: Dict,
    output_dir: str = "results/fidelity_landscapes",
    K_adapt: int = 5,
    n_points: int = 25
):
    """
    Generate fidelity landscapes for all three noise parameters.

    Args:
        meta_policy_path: Path to trained meta policy
        robust_policy_path: Path to robust baseline policy
        config: Experiment configuration
        output_dir: Directory to save results
        K_adapt: Number of adaptation steps
        n_points: Points per sweep
    """
    print("=" * 80)
    print("FIDELITY LANDSCAPE VALIDATION")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)

    # Baseline parameters (center of training distribution)
    baseline_params = {
        'alpha': np.mean(config.get('alpha_range', [0.5, 2.0])),
        'A': np.mean(config.get('A_range', [0.05, 0.3])),
        'omega_c': np.mean(config.get('omega_c_range', [2.0, 8.0]))
    }

    all_results = {}

    # Sweep each parameter
    for param in ['alpha', 'A', 'omega_c']:
        print(f"\n{'='*80}")
        print(f"Processing parameter: {param}")
        print(f"{'='*80}")

        results = generate_fidelity_landscape(
            meta_policy_path=meta_policy_path,
            robust_policy_path=robust_policy_path,
            config=config,
            sweep_param=param,
            sweep_range=None,  # Use config ranges
            n_points=n_points,
            K_adapt=K_adapt,
            baseline_params=baseline_params
        )

        all_results[param] = results

        # Save individual results
        result_path = f"{output_dir}/landscape_{param}.json"
        with open(result_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {result_path}")

        # Generate plot
        plot_path = f"{output_dir}/landscape_{param}.pdf"
        plot_fidelity_landscape(results, output_path=plot_path, show_gap=True)

    # Generate combined plot
    print(f"\n{'='*80}")
    print("Generating combined figure...")
    print(f"{'='*80}")

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    param_labels = {
        'alpha': r'Spectral Exponent ($\alpha$)',
        'A': r'Noise Amplitude ($A$)',
        'omega_c': r'Cutoff Frequency ($\omega_c$)'
    }

    for i, param in enumerate(['alpha', 'A', 'omega_c']):
        ax = axes[i]
        res = all_results[param]

        sweep_vals = np.array(res['sweep_values'])
        fid_meta = np.array(res['fidelities_meta'])
        fid_robust = np.array(res['fidelities_robust'])

        ax.plot(sweep_vals, fid_meta, 'o-', linewidth=2, markersize=6,
                label=f'Meta (K={K_adapt})', color='steelblue')
        ax.plot(sweep_vals, fid_robust, 's-', linewidth=2, markersize=6,
                label='Robust', color='orangered', alpha=0.8)

        baseline_val = baseline_params[param]
        ax.axvline(baseline_val, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)

        ax.set_xlabel(param_labels[param], fontsize=12, fontweight='bold')
        if i == 0:
            ax.set_ylabel('Gate Fidelity', fontsize=12, fontweight='bold')
        ax.set_title(f'({chr(97+i)}) {param.upper()} Sweep', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.0, 1.05])

    plt.suptitle('Fidelity Landscapes Across Noise Parameters',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    combined_path = f"{output_dir}/landscapes_combined.pdf"
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    print(f"\nCombined figure saved to {combined_path}")

    combined_path_png = f"{output_dir}/landscapes_combined.png"
    plt.savefig(combined_path_png, dpi=300, bbox_inches='tight')
    print(f"Combined figure saved to {combined_path_png}")

    plt.close()

    print("\n" + "=" * 80)
    print("FIDELITY LANDSCAPE VALIDATION COMPLETE")
    print("=" * 80)

    return all_results


@app.command()
def main(
    meta_path: Path = Path("experiments/train_scripts/checkpoints/maml_best_policy.pt"),
    robust_path: Path = Path("experiments/train_scripts/checkpoints/robust_best_policy.pt"),
    output_dir: Path = Path("results/fidelity_landscapes"),
    k_adapt: int = 5,
    n_points: int = 25
):
    """
    Generate fidelity landscapes for all noise parameters.

    Args:
        meta_path: Path to trained meta policy checkpoint
        robust_path: Path to robust baseline policy checkpoint
        output_dir: Directory to save results and figures
        k_adapt: Number of adaptation steps for meta policy
        n_points: Number of points in each sweep
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

    # Check if policy paths exist
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta policy not found: {meta_path}")
    if not robust_path.exists():
        raise FileNotFoundError(f"Robust policy not found: {robust_path}")

    print(f"Using meta policy: {meta_path}")
    print(f"Using robust policy: {robust_path}")

    # Generate landscapes
    results = generate_all_landscapes(
        meta_policy_path=str(meta_path),
        robust_policy_path=str(robust_path),
        config=config,
        output_dir=str(output_dir),
        K_adapt=k_adapt,
        n_points=n_points
    )

    print("\nValidation complete!")


if __name__ == "__main__":
    app()
