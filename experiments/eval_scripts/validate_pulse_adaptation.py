"""
Validation Script: Pulse Adaptation Analysis

This script provides detailed visualization of how control pulses change during adaptation:

A. Complete Pulse Sequences: u_X(t) and u_Y(t) for:
   - Robust baseline policy (trained on average task)
   - 3 task-adapted examples (meta-policy after K=5 steps)
   Shows full time-domain control sequences

B. Pulse Differences: Δu(t) = u_adapted(t) - u_robust(t)
   - Highlights what changed during adaptation
   - Reveals which time windows were modified
   - Shows magnitude and sign of corrections

Physical interpretation:
- u_X(t): Control field along X-axis (typically σ_x rotations)
- u_Y(t): Control field along Y-axis (typically σ_y rotations)
- Δu(t): Adaptation corrections to handle task-specific noise

Expected output: Multi-panel plots showing pulse sequences and adaptations
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


def get_pulse_sequences(
    policy: torch.nn.Module,
    task: NoiseParameters,
    config: Dict,
    env,
    adapt: bool = False,
    K: int = 0
) -> Tuple[np.ndarray, float]:
    """
    Get control pulse sequences from policy.

    Args:
        policy: Policy network
        task: Task parameters
        config: Configuration dict
        env: QuantumEnvironment
        adapt: If True, adapt policy before generating pulses
        K: Number of adaptation steps

    Returns:
        controls: Control sequences (n_segments, n_controls)
        fidelity: Achieved fidelity
    """
    task_features = torch.tensor(
        [task.alpha, task.A, task.omega_c],
        dtype=torch.float32
    )

    if adapt and K > 0:
        # Adapt policy
        adapted_policy = copy.deepcopy(policy)
        adapted_policy.train()

        inner_lr = config.get('inner_lr', 0.01)

        for k in range(K):
            loss = env.compute_loss_differentiable(
                adapted_policy, task, device=torch.device('cpu')
            )
            loss.backward()
            with torch.no_grad():
                for param in adapted_policy.parameters():
                    if param.grad is not None:
                        param -= inner_lr * param.grad
                        param.grad.zero_()

        eval_policy = adapted_policy
    else:
        eval_policy = policy

    # Generate controls
    eval_policy.eval()
    with torch.no_grad():
        controls = eval_policy(task_features).detach().numpy()

    # Compute fidelity
    fidelity = env.compute_fidelity(controls, task)

    return controls, fidelity


def run_pulse_adaptation_experiment(
    meta_policy_path: str,
    robust_policy_path: str,
    config: Dict,
    n_examples: int = 3,
    K_adapt: int = 5,
    output_dir: str = "results/pulse_adaptation"
) -> Dict:
    """
    Main experiment: Analyze pulse sequences and adaptations.

    Args:
        meta_policy_path: Path to trained meta policy
        robust_policy_path: Path to robust baseline policy
        config: Experiment configuration
        n_examples: Number of task examples to analyze
        K_adapt: Number of adaptation steps
        output_dir: Output directory

    Returns:
        results: Dict with pulse sequences and differences
    """
    print("=" * 80)
    print("EXPERIMENT: Pulse Adaptation Analysis")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)

    print(f"\nExperiment parameters:")
    print(f"  Number of examples: {n_examples}")
    print(f"  Adaptation steps: {K_adapt}")

    # Load policies
    print("\n[1/4] Loading policies...")
    meta_policy = load_policy_from_checkpoint(
        meta_policy_path, config, eval_mode=False, verbose=True
    )
    robust_policy = load_policy_from_checkpoint(
        robust_policy_path, config, eval_mode=True, verbose=True
    )

    # Create environment
    print("\n[2/4] Creating quantum environment...")
    target_state = get_target_state_from_config(config)
    env = create_quantum_environment(config, target_state)

    # Sample diverse tasks
    print(f"\n[3/4] Sampling {n_examples} diverse tasks...")
    task_dist = TaskDistribution(
        dist_type=config.get('task_dist_type', 'uniform'),
        ranges={
            'alpha': tuple(config['alpha_range']),
            'A': tuple(config['A_range']),
            'omega_c': tuple(config['omega_c_range'])
        }
    )

    # Sample tasks from different regions of parameter space
    # Strategy: sample uniformly but ensure diversity
    all_tasks = task_dist.sample(n_examples * 3)

    # Sort by noise amplitude to get diversity
    all_tasks.sort(key=lambda t: t.A)

    # Pick evenly spaced tasks
    indices = np.linspace(0, len(all_tasks) - 1, n_examples, dtype=int)
    sample_tasks = [all_tasks[i] for i in indices]

    # Generate pulse sequences
    print(f"\n[4/4] Generating pulse sequences...")

    time_array = np.linspace(0, config.get('horizon', 1.0), config.get('n_segments', 100))

    pulse_data = []

    for i, task in enumerate(sample_tasks):
        print(f"\n  Task {i+1}: α={task.alpha:.2f}, A={task.A:.3f}, ωc={task.omega_c:.2f}")

        # Robust policy
        controls_robust, fid_robust = get_pulse_sequences(
            policy=robust_policy,
            task=task,
            config=config,
            env=env,
            adapt=False,
            K=0
        )

        # Adapted policy
        controls_adapted, fid_adapted = get_pulse_sequences(
            policy=meta_policy,
            task=task,
            config=config,
            env=env,
            adapt=True,
            K=K_adapt
        )

        # Compute differences
        pulse_diff = controls_adapted - controls_robust

        print(f"    Robust fidelity: {fid_robust:.4f}")
        print(f"    Adapted fidelity: {fid_adapted:.4f}")
        print(f"    Improvement: {fid_adapted - fid_robust:.4f}")
        print(f"    Max pulse change: {np.max(np.abs(pulse_diff)):.4f}")
        print(f"    RMS pulse change: {np.sqrt(np.mean(pulse_diff**2)):.4f}")

        pulse_data.append({
            'task_params': {
                'alpha': task.alpha,
                'A': task.A,
                'omega_c': task.omega_c
            },
            'controls_robust': controls_robust.tolist(),
            'controls_adapted': controls_adapted.tolist(),
            'pulse_diff': pulse_diff.tolist(),
            'fidelity_robust': float(fid_robust),
            'fidelity_adapted': float(fid_adapted),
            'improvement': float(fid_adapted - fid_robust)
        })

    # Save results
    print(f"\nSaving results to {output_dir}...")
    results = {
        'n_examples': n_examples,
        'K_adapt': K_adapt,
        'time_array': time_array.tolist(),
        'pulse_data': pulse_data,
        'config': config
    }

    results_path = f"{output_dir}/results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    return results


def plot_pulse_sequences(
    results: Dict,
    output_path: str = None
):
    """
    Generate plot showing complete pulse sequences for robust and adapted policies.

    Args:
        results: Results dict from experiment
        output_path: Path to save figure
    """
    sns.set_style("whitegrid")

    n_examples = results['n_examples']
    time_array = np.array(results['time_array'])
    pulse_data = results['pulse_data']
    K_adapt = results['K_adapt']

    # Create figure: 2 columns (X and Y controls) x n_examples rows
    fig, axes = plt.subplots(n_examples, 2, figsize=(16, 5 * n_examples))

    if n_examples == 1:
        axes = axes.reshape(1, -1)

    for i, data in enumerate(pulse_data):
        controls_robust = np.array(data['controls_robust'])
        controls_adapted = np.array(data['controls_adapted'])
        task_params = data['task_params']
        fid_robust = data['fidelity_robust']
        fid_adapted = data['fidelity_adapted']
        improvement = data['improvement']

        # X control
        ax_x = axes[i, 0]
        ax_x.plot(time_array, controls_robust[:, 0], linewidth=2.5,
                 color='orangered', label='Robust baseline', alpha=0.8, linestyle='--')
        ax_x.plot(time_array, controls_adapted[:, 0], linewidth=2.5,
                 color='steelblue', label=f'Adapted (K={K_adapt})', alpha=0.9)

        ax_x.set_xlabel('Time (a.u.)', fontsize=12, fontweight='bold')
        ax_x.set_ylabel('$u_X(t)$', fontsize=13, fontweight='bold')
        ax_x.set_title(f'Task {i+1} - X Control\n'
                      f'α={task_params["alpha"]:.2f}, A={task_params["A"]:.3f}, '
                      f'ωc={task_params["omega_c"]:.2f}\n'
                      f'Fid: {fid_robust:.4f} → {fid_adapted:.4f} (+{improvement:.4f})',
                      fontsize=11, fontweight='bold')
        ax_x.legend(fontsize=10, loc='best')
        ax_x.grid(True, alpha=0.3)
        ax_x.tick_params(labelsize=10)

        # Y control
        ax_y = axes[i, 1]
        ax_y.plot(time_array, controls_robust[:, 1], linewidth=2.5,
                 color='orangered', label='Robust baseline', alpha=0.8, linestyle='--')
        ax_y.plot(time_array, controls_adapted[:, 1], linewidth=2.5,
                 color='green', label=f'Adapted (K={K_adapt})', alpha=0.9)

        ax_y.set_xlabel('Time (a.u.)', fontsize=12, fontweight='bold')
        ax_y.set_ylabel('$u_Y(t)$', fontsize=13, fontweight='bold')
        ax_y.set_title(f'Task {i+1} - Y Control\n'
                      f'α={task_params["alpha"]:.2f}, A={task_params["A"]:.3f}, '
                      f'ωc={task_params["omega_c"]:.2f}\n'
                      f'Fid: {fid_robust:.4f} → {fid_adapted:.4f} (+{improvement:.4f})',
                      fontsize=11, fontweight='bold')
        ax_y.legend(fontsize=10, loc='best')
        ax_y.grid(True, alpha=0.3)
        ax_y.tick_params(labelsize=10)

    plt.suptitle('Complete Pulse Sequences: Robust vs Adapted',
                fontsize=16, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nPulse sequences figure saved to {output_path}")

    plt.close()


def plot_pulse_differences(
    results: Dict,
    output_path: str = None
):
    """
    Generate plot showing pulse differences Δu(t) = u_adapted(t) - u_robust(t).

    Args:
        results: Results dict from experiment
        output_path: Path to save figure
    """
    sns.set_style("whitegrid")

    n_examples = results['n_examples']
    time_array = np.array(results['time_array'])
    pulse_data = results['pulse_data']

    # Create figure: 2 columns (X and Y) x n_examples rows
    fig, axes = plt.subplots(n_examples, 2, figsize=(16, 5 * n_examples))

    if n_examples == 1:
        axes = axes.reshape(1, -1)

    for i, data in enumerate(pulse_data):
        pulse_diff = np.array(data['pulse_diff'])
        task_params = data['task_params']
        improvement = data['improvement']

        # Compute RMS of difference
        rms_x = np.sqrt(np.mean(pulse_diff[:, 0]**2))
        rms_y = np.sqrt(np.mean(pulse_diff[:, 1]**2))

        # X control difference
        ax_x = axes[i, 0]
        ax_x.plot(time_array, pulse_diff[:, 0], linewidth=2.5,
                 color='purple', alpha=0.9)
        ax_x.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
        ax_x.fill_between(time_array, 0, pulse_diff[:, 0],
                         alpha=0.3, color='purple')

        ax_x.set_xlabel('Time (a.u.)', fontsize=12, fontweight='bold')
        ax_x.set_ylabel('$\\Delta u_X(t)$', fontsize=13, fontweight='bold')
        ax_x.set_title(f'Task {i+1} - X Control Difference\n'
                      f'α={task_params["alpha"]:.2f}, A={task_params["A"]:.3f}, '
                      f'ωc={task_params["omega_c"]:.2f}\n'
                      f'RMS = {rms_x:.4f}, Improvement = {improvement:.4f}',
                      fontsize=11, fontweight='bold')
        ax_x.grid(True, alpha=0.3)
        ax_x.tick_params(labelsize=10)

        # Add statistics text box
        textstr = (
            f'Max: {np.max(pulse_diff[:, 0]):.4f}\n'
            f'Min: {np.min(pulse_diff[:, 0]):.4f}\n'
            f'RMS: {rms_x:.4f}'
        )
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax_x.text(0.02, 0.98, textstr, transform=ax_x.transAxes,
                 fontsize=9, verticalalignment='top', bbox=props,
                 family='monospace')

        # Y control difference
        ax_y = axes[i, 1]
        ax_y.plot(time_array, pulse_diff[:, 1], linewidth=2.5,
                 color='darkgreen', alpha=0.9)
        ax_y.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
        ax_y.fill_between(time_array, 0, pulse_diff[:, 1],
                         alpha=0.3, color='darkgreen')

        ax_y.set_xlabel('Time (a.u.)', fontsize=12, fontweight='bold')
        ax_y.set_ylabel('$\\Delta u_Y(t)$', fontsize=13, fontweight='bold')
        ax_y.set_title(f'Task {i+1} - Y Control Difference\n'
                      f'α={task_params["alpha"]:.2f}, A={task_params["A"]:.3f}, '
                      f'ωc={task_params["omega_c"]:.2f}\n'
                      f'RMS = {rms_y:.4f}, Improvement = {improvement:.4f}',
                      fontsize=11, fontweight='bold')
        ax_y.grid(True, alpha=0.3)
        ax_y.tick_params(labelsize=10)

        # Add statistics text box
        textstr = (
            f'Max: {np.max(pulse_diff[:, 1]):.4f}\n'
            f'Min: {np.min(pulse_diff[:, 1]):.4f}\n'
            f'RMS: {rms_y:.4f}'
        )
        props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.8)
        ax_y.text(0.02, 0.98, textstr, transform=ax_y.transAxes,
                 fontsize=9, verticalalignment='top', bbox=props,
                 family='monospace')

    plt.suptitle('Pulse Adaptation Differences: $\\Delta u(t) = u_{adapted}(t) - u_{robust}(t)$',
                fontsize=16, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Pulse differences figure saved to {output_path}")

    plt.close()


def plot_combined_analysis(
    results: Dict,
    task_idx: int = 0,
    output_path: str = None
):
    """
    Generate combined plot for a single task showing sequences and differences.

    Args:
        results: Results dict from experiment
        task_idx: Index of task to plot
        output_path: Path to save figure
    """
    sns.set_style("whitegrid")

    time_array = np.array(results['time_array'])
    data = results['pulse_data'][task_idx]

    controls_robust = np.array(data['controls_robust'])
    controls_adapted = np.array(data['controls_adapted'])
    pulse_diff = np.array(data['pulse_diff'])
    task_params = data['task_params']
    fid_robust = data['fidelity_robust']
    fid_adapted = data['fidelity_adapted']
    improvement = data['improvement']

    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Top row: Pulse sequences
    # (a) X control sequences
    ax = axes[0, 0]
    ax.plot(time_array, controls_robust[:, 0], linewidth=3,
           color='orangered', label='Robust', alpha=0.7, linestyle='--')
    ax.plot(time_array, controls_adapted[:, 0], linewidth=3,
           color='steelblue', label='Adapted', alpha=0.9)
    ax.set_xlabel('Time (a.u.)', fontsize=12, fontweight='bold')
    ax.set_ylabel('$u_X(t)$', fontsize=13, fontweight='bold')
    ax.set_title('(a) X Control Sequences', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # (b) Y control sequences
    ax = axes[0, 1]
    ax.plot(time_array, controls_robust[:, 1], linewidth=3,
           color='orangered', label='Robust', alpha=0.7, linestyle='--')
    ax.plot(time_array, controls_adapted[:, 1], linewidth=3,
           color='green', label='Adapted', alpha=0.9)
    ax.set_xlabel('Time (a.u.)', fontsize=12, fontweight='bold')
    ax.set_ylabel('$u_Y(t)$', fontsize=13, fontweight='bold')
    ax.set_title('(b) Y Control Sequences', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Bottom row: Differences
    # (c) X control difference
    ax = axes[1, 0]
    ax.plot(time_array, pulse_diff[:, 0], linewidth=3, color='purple', alpha=0.9)
    ax.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.fill_between(time_array, 0, pulse_diff[:, 0], alpha=0.3, color='purple')
    ax.set_xlabel('Time (a.u.)', fontsize=12, fontweight='bold')
    ax.set_ylabel('$\\Delta u_X(t)$', fontsize=13, fontweight='bold')
    ax.set_title('(c) X Control Adaptation', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # (d) Y control difference
    ax = axes[1, 1]
    ax.plot(time_array, pulse_diff[:, 1], linewidth=3, color='darkgreen', alpha=0.9)
    ax.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.fill_between(time_array, 0, pulse_diff[:, 1], alpha=0.3, color='darkgreen')
    ax.set_xlabel('Time (a.u.)', fontsize=12, fontweight='bold')
    ax.set_ylabel('$\\Delta u_Y(t)$', fontsize=13, fontweight='bold')
    ax.set_title('(d) Y Control Adaptation', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Pulse Adaptation Analysis\n'
                f'Task: α={task_params["alpha"]:.2f}, A={task_params["A"]:.3f}, '
                f'ωc={task_params["omega_c"]:.2f} | '
                f'Fidelity: {fid_robust:.4f} → {fid_adapted:.4f} (+{improvement:.4f})',
                fontsize=15, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Combined analysis saved to {output_path}")

    plt.close()


@app.command()
def main(
    meta_path: Path = Path("experiments/train_scripts/checkpoints/maml_best_policy.pt"),
    robust_path: Path = Path("experiments/train_scripts/checkpoints/robust_best_policy.pt"),
    output_dir: Path = Path("results/pulse_adaptation"),
    n_examples: int = 3,
    k_adapt: int = 5
):
    """
    Run pulse adaptation analysis experiment.

    Args:
        meta_path: Path to trained meta policy checkpoint
        robust_path: Path to robust baseline policy checkpoint
        output_dir: Directory to save results and figures
        n_examples: Number of task examples to analyze (default 3)
        k_adapt: Number of adaptation steps (default 5)
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

    # Run experiment
    results = run_pulse_adaptation_experiment(
        meta_policy_path=str(meta_path),
        robust_policy_path=str(robust_path),
        config=config,
        n_examples=n_examples,
        K_adapt=k_adapt,
        output_dir=str(output_dir)
    )

    # Generate plots
    # 1. Complete pulse sequences
    sequences_path = f"{output_dir}/pulse_sequences.pdf"
    plot_pulse_sequences(results, output_path=sequences_path)

    sequences_path_png = f"{output_dir}/pulse_sequences.png"
    plot_pulse_sequences(results, output_path=sequences_path_png)

    # 2. Pulse differences
    differences_path = f"{output_dir}/pulse_differences.pdf"
    plot_pulse_differences(results, output_path=differences_path)

    differences_path_png = f"{output_dir}/pulse_differences.png"
    plot_pulse_differences(results, output_path=differences_path_png)

    # 3. Combined analysis for each task
    for i in range(n_examples):
        combined_path = f"{output_dir}/combined_task_{i+1}.pdf"
        plot_combined_analysis(results, task_idx=i, output_path=combined_path)

        combined_path_png = f"{output_dir}/combined_task_{i+1}.png"
        plot_combined_analysis(results, task_idx=i, output_path=combined_path_png)

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)
    print("\nKey insights:")

    for i, data in enumerate(results['pulse_data']):
        task_params = data['task_params']
        improvement = data['improvement']
        pulse_diff = np.array(data['pulse_diff'])
        rms = np.sqrt(np.mean(pulse_diff**2))

        print(f"\n  Task {i+1} (α={task_params['alpha']:.2f}, "
              f"A={task_params['A']:.3f}):")
        print(f"    • Fidelity improvement: {improvement:.4f}")
        print(f"    • RMS pulse change: {rms:.4f}")
        print(f"    • Max pulse change: {np.max(np.abs(pulse_diff)):.4f}")

    print("\nOutputs:")
    print(f"  • Pulse sequences: {output_dir}/pulse_sequences.pdf")
    print(f"  • Pulse differences: {output_dir}/pulse_differences.pdf")
    print(f"  • Combined analysis: {output_dir}/combined_task_*.pdf")


if __name__ == "__main__":
    app()
