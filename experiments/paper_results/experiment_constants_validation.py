"""
Experiment: Physics Constants Estimation and Validation

This script estimates all theoretical constants from the paper:
- Spectral gap Δ(θ)
- PL constant μ(θ)
- Filter constant C_filter
- Control-relevant variance σ²_S
- Combined constant c_quantum

Validates that empirical constants are within 2× of theoretical predictions.
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
from scipy import linalg


from metaqctrl.quantum.lindblad import LindbladSimulator
from metaqctrl.quantum.noise_models import TaskDistribution, NoiseParameters, PSDToLindblad
from metaqctrl.quantum.gates import state_fidelity
from metaqctrl.theory.quantum_environment import create_quantum_environment

from metaqctrl.quantum.gates import GateFidelityComputer, TargetGates 
import yaml 

from metaqctrl.theory.physics_constants import (
    compute_spectral_gap,
    estimate_filter_constant, estimate_pl_constant,  
    estimate_PL_constant_from_convergence,
    compute_control_relevant_variance,
    estimate_all_constants
)


def run_constants_validation_experiment(
    config: Dict,
    n_sample_tasks: int = 50,
    output_dir: str = "results/constants_validation"
) -> Dict:
    """
    Estimate all theoretical constants and compare with predictions

    Returns:
        results: Dictionary with all estimated constants and comparisons
    """
    print("=" * 80)
    print("EXPERIMENT: Physics Constants Estimation and Validation")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)

    # Create quantum environment
    print("\n[1/5] Creating quantum environment...")
    ## Add target state here. ..  
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex) 
    ket_0 = np.array([1, 0], dtype=complex) 
    U_target = TargetGates.pauli_x() 
    target_state = np.outer(U_target @ ket_0, (U_target @ ket_0).conj()) 
    
    env = create_quantum_environment(config,target_state)

    # Sample tasks
    print(f"\n[2/5] Sampling {n_sample_tasks} tasks...")
    task_dist = TaskDistribution(
        dist_type=config.get('task_dist_type', 'uniform'),
        ranges={
            'alpha': tuple(config.get('alpha_range', [0.5, 2.0])),
            'A': tuple(config.get('A_range', [0.05, 0.3])),
            'omega_c': tuple(config.get('omega_c_range', [2.0, 8.0]))
        }
    )  
     
   # tasks = [task_dist.sample() for _ in range(n_sample_tasks)]
     
    tasks = task_dist.sample(n_sample_tasks)  
    print(len(tasks))
    # Estimate all constants
    print("\n[3/5] Estimating theoretical constants...")
    constants = estimate_all_constants(
        env=env,
        task_distribution=task_dist,
        n_tasks=n_sample_tasks
    )

    print("\n  Estimated Constants:")
    print(f"    Δ_min (spectral gap) = {constants['Delta_min']:.4f}")
    print(f"    μ_min (PL constant) = {constants['mu_min']:.6f}")
    print(f"    C_filter = {constants['C_filter']:.4f}")
    print(f"    σ²_S (task variance) = {constants['sigma2_S']:.6f}")
    print(f"    c_quantum (combined) = {constants['c_quantum']:.6f}")

    # Compute theoretical predictions
    print("\n[4/5] Computing theoretical predictions...")

    d = 2  # Qubit dimension
    M = np.max([np.linalg.norm(env.H_controls[k]) for k in range(env.num_controls)])
    T = env.evolution_time

    # Theoretical PL constant from Lemma 4.5 (heuristic formula)
    mu_theory = constants['Delta_min'] / (d**2 * M**2 * T)
    print(f"    μ_theory (from formula) = {mu_theory:.6f}")
    print(f"    Ratio μ_empirical/μ_theory = {constants['mu_min']/mu_theory:.2f}x")

    # Theoretical combined constant
    c_quantum_theory = (constants['Delta_min'] * constants['C_filter']**2 *
                        constants['sigma2_S']) / (d**2 * M**2 * T**3)
    print(f"    c_quantum_theory = {c_quantum_theory:.6f}")
    print(f"    Ratio c_empirical/c_theory = {constants['c_quantum']/c_quantum_theory:.2f}x")

    # Distribution of constants across tasks
    print("\n[5/5] Analyzing constant distributions...")

    spectral_gaps = []
    pl_constants = []

    for task in tasks[:20]:  # Sample subset for speed
        # Spectral gap for this task
        Delta = compute_spectral_gap(env, task)
        spectral_gaps.append(Delta)

        # PL constant estimate (using theoretical formula)
        mu = estimate_pl_constant(env, task, num_samples=5)
        pl_constants.append(mu)

    spectral_gaps = np.array(spectral_gaps)
    pl_constants = np.array(pl_constants)

    print(f"    Δ: mean={np.mean(spectral_gaps):.4f}, std={np.std(spectral_gaps):.4f}")
    print(f"    μ: mean={np.mean(pl_constants):.6f}, std={np.std(pl_constants):.6f}")

    # Save results
    results = {
        'constants': {
            'Delta_min': float(constants['Delta_min']),
            'Delta_mean': float(np.mean(spectral_gaps)),
            'Delta_std': float(np.std(spectral_gaps)),
            'mu_min': float(constants['mu_min']),
            'mu_mean': float(np.mean(pl_constants)),
            'mu_std': float(np.std(pl_constants)),
            'C_filter': float(constants['C_filter']),
            'sigma2_S': float(constants['sigma2_S']),
            'c_quantum': float(constants['c_quantum'])
        },
        'theoretical_predictions': {
            'mu_theory': float(mu_theory),
            'mu_ratio': float(constants['mu_min'] / mu_theory),
            'c_quantum_theory': float(c_quantum_theory),
            'c_quantum_ratio': float(constants['c_quantum'] / c_quantum_theory)
        },
        'system_parameters': {
            'd': d,
            'M': float(M),
            'T': T,
            'm': env.num_controls
        },
        'config': config,
        'n_sample_tasks': n_sample_tasks
    }

    with open(f"{output_dir}/constants.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}/constants.json")

    print("\n" + "=" * 80)
    print("VALIDATION: Constants within 2× of theory?")
    print("=" * 80)

    mu_ok = 0.5 <= constants['mu_min'] / mu_theory <= 2.0
    c_ok = 0.5 <= constants['c_quantum'] / c_quantum_theory <= 2.0

    print(f"  μ constant: {'✓ PASS' if mu_ok else '✗ FAIL'}")
    print(f"  c_quantum: {'✓ PASS' if c_ok else '✗ FAIL'}")

    if mu_ok and c_ok:
        print("\n  ✓✓✓ ALL CONSTANTS VALIDATED ✓✓✓")
    else:
        print("\n  ⚠ Some constants outside 2× range (acceptable for heuristic theory)")

    return results


def plot_constant_distributions(results: Dict, output_dir: str = "results/constants_validation"):
    """Generate visualization of constant distributions"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Spectral Gap Distribution
    ax = axes[0, 0]
    Delta_min = results['constants']['Delta_min']
    Delta_mean = results['constants']['Delta_mean']
    Delta_std = results['constants']['Delta_std']

    x = np.linspace(Delta_mean - 3*Delta_std, Delta_mean + 3*Delta_std, 100)
    y = (1 / (Delta_std * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((x - Delta_mean) / Delta_std)**2)
    ax.plot(x, y, 'steelblue', linewidth=2)
    ax.axvline(Delta_min, color='darkred', linestyle='--', linewidth=2, label=f'Δ_min = {Delta_min:.4f}')
    ax.set_xlabel('Spectral Gap Δ(θ)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax.set_title('Spectral Gap Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # PL Constant Distribution
    ax = axes[0, 1]
    mu_min = results['constants']['mu_min']
    mu_mean = results['constants']['mu_mean']
    mu_std = results['constants']['mu_std']
    mu_theory = results['theoretical_predictions']['mu_theory']

    x = np.linspace(mu_mean - 3*mu_std, mu_mean + 3*mu_std, 100)
    y = (1 / (mu_std * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((x - mu_mean) / mu_std)**2)
    ax.plot(x, y, 'steelblue', linewidth=2, label='Empirical')
    ax.axvline(mu_theory, color='orange', linestyle=':', linewidth=2, label=f'μ_theory = {mu_theory:.6f}')
    ax.axvline(mu_min, color='darkred', linestyle='--', linewidth=2, label=f'μ_min = {mu_min:.6f}')
    ax.set_xlabel('PL Constant μ(θ)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax.set_title('PL Constant Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Constant Comparison Bar Chart
    ax = axes[1, 0]
    constant_names = ['μ', 'c_quantum']
    empirical = [
        results['constants']['mu_min'],
        results['constants']['c_quantum']
    ]
    theoretical = [
        results['theoretical_predictions']['mu_theory'],
        results['theoretical_predictions']['c_quantum_theory']
    ]

    x = np.arange(len(constant_names))
    width = 0.35

    ax.bar(x - width/2, empirical, width, label='Empirical', color='steelblue')
    ax.bar(x + width/2, theoretical, width, label='Theory', color='orange')

    ax.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax.set_title('Empirical vs Theoretical Constants', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(constant_names, fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Ratio Visualization
    ax = axes[1, 1]
    ratios = [
        results['theoretical_predictions']['mu_ratio'],
        results['theoretical_predictions']['c_quantum_ratio']
    ]

    colors = ['green' if 0.5 <= r <= 2.0 else 'red' for r in ratios]
    bars = ax.barh(constant_names, ratios, color=colors, alpha=0.7)

    ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5, label='2× bounds')
    ax.axvline(2.0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(1.0, color='black', linestyle='-', linewidth=2, label='Exact match')

    ax.set_xlabel('Empirical / Theory Ratio', fontsize=12, fontweight='bold')
    ax.set_title('Validation: Within 2× of Theory?', fontsize=14, fontweight='bold')
    ax.set_xlim([0, 3])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/constants_visualization.pdf", dpi=300, bbox_inches='tight')
    print(f"Figure saved to {output_dir}/constants_visualization.pdf")
    plt.close()


if __name__ == "__main__":
    # Configuration matching paper
    config_path='../../configs/experiment_config.yaml' 
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f) 
 

  #  Run experiment
    results = run_constants_validation_experiment(
        config=config,
        n_sample_tasks=50,
        output_dir="results/constants_validation"
    )

    # # Generate visualizations
    plot_constant_distributions(results)
