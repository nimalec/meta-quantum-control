"""
Figure 4 — Robustness to Distribution Shift / Device Personalization

This script generates all panels for Figure 4:
  (a) Performance under increasing detuning mismatch (OOD wrt training)
  (b) Per-task heatmap in the hardest regime
  (c) Baseline comparison (bars): GRAPE vs Domain-Random vs Meta+Adapt
  (d) Cross-family transfer (optional)
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


from metaqctrl.quantum.noise_models import TaskDistribution, NoiseParameters
from metaqctrl.meta_rl.policy import PulsePolicy
from metaqctrl.baselines.robust_control import GRAPEOptimizer
from metaqctrl.theory.quantum_environment import create_quantum_environment, get_target_state_from_config
from metaqctrl.utils.checkpoint_utils import load_policy_from_checkpoint


def run_ood_detuning_experiment(
    meta_policy_path: str,
    robust_policy_path: str,
    config: Dict,
    ood_detuning_levels: List[float] = None,
    K_fixed: int = 5,
    n_test_tasks: int = 5
) -> Dict:
    """
    Panel 4(a): Performance under OOD detuning mismatch

    Train on nominal detuning ~0, test on large |Δ|
    """
    print("Running OOD detuning experiment...")

    if ood_detuning_levels is None:
        # Detuning levels outside training range
        # Assume training was on omega_c in [2, 8], so test outside
        ood_detuning_levels = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]

    # Load policies
    meta_policy_template = load_policy_from_checkpoint(
        meta_policy_path, config, eval_mode=False, verbose=False
    )
    robust_policy = load_policy_from_checkpoint(
        robust_policy_path, config, eval_mode=True, verbose=False
    )

    # Create environment
    target_state = get_target_state_from_config(config)
    env = create_quantum_environment(config, target_state)

    results = {
        'ood_levels': ood_detuning_levels,
        'fidelity_robust': [],
        'fidelity_meta_init': [],
        'fidelity_meta_adapted': [],
        'std_robust': [],
        'std_meta_init': [],
        'std_meta_adapted': []
    }

    for ood_level in ood_detuning_levels:
        print(f"  OOD detuning level={ood_level}...", end=' ')

        # Create OOD task distribution
        # Go outside the nominal range
        omega_c_ood = 8.0 + ood_level  # Beyond training max
        task_ranges = {
            'alpha': (1.0, 1.5),
            'A': (0.1, 0.2),
            'omega_c': (omega_c_ood - 0.5, omega_c_ood + 0.5)
        }

        task_dist = TaskDistribution(dist_type='uniform', ranges=task_ranges)
        test_tasks = task_dist.sample(n_test_tasks)

        fids_robust = []
        fids_meta_init = []
        fids_meta_adapted = []

        for task in test_tasks:
            task_features = torch.tensor([task.alpha, task.A, task.omega_c], dtype=torch.float32)

            # Robust policy
            with torch.no_grad():
                controls_robust = robust_policy(task_features).detach().numpy()
            fid_robust = env.compute_fidelity(controls_robust, task)
            fids_robust.append(fid_robust)

            # Meta init (K=0)
            with torch.no_grad():
                controls_meta_init = meta_policy_template(task_features).detach().numpy()
            fid_meta_init = env.compute_fidelity(controls_meta_init, task)
            fids_meta_init.append(fid_meta_init)

            # Meta after adaptation
            import copy
            adapted_policy = copy.deepcopy(meta_policy_template)
            adapted_policy.train()

            for k in range(K_fixed):
                loss = env.compute_loss_differentiable(adapted_policy, task, device=torch.device('cpu'))
                loss.backward()
                with torch.no_grad():
                    for param in adapted_policy.parameters():
                        if param.grad is not None:
                            param -= config['inner_lr'] * param.grad
                            param.grad.zero_()

            adapted_policy.eval()
            with torch.no_grad():
                controls_meta_adapted = adapted_policy(task_features).detach().numpy()
            fid_meta_adapted = env.compute_fidelity(controls_meta_adapted, task)
            fids_meta_adapted.append(fid_meta_adapted)

        # Store results
        results['fidelity_robust'].append(np.mean(fids_robust))
        results['fidelity_meta_init'].append(np.mean(fids_meta_init))
        results['fidelity_meta_adapted'].append(np.mean(fids_meta_adapted))
        results['std_robust'].append(np.std(fids_robust) / np.sqrt(n_test_tasks))
        results['std_meta_init'].append(np.std(fids_meta_init) / np.sqrt(n_test_tasks))
        results['std_meta_adapted'].append(np.std(fids_meta_adapted) / np.sqrt(n_test_tasks))

        print(f"Robust={results['fidelity_robust'][-1]:.3f}, " +
              f"Meta-init={results['fidelity_meta_init'][-1]:.3f}, " +
              f"Meta-adapt={results['fidelity_meta_adapted'][-1]:.3f}")

    return results


def run_per_task_heatmap_experiment(
    meta_policy_path: str,
    robust_policy_path: str,
    config: Dict,
    n_hard_tasks: int = 5,
    K_fixed: int = 5
) -> Dict:
    """
    Panel 4(b): Per-task heatmap in hardest regime
    """
    print("Running per-task heatmap experiment...")

    # Load policies
    meta_policy_template = load_policy_from_checkpoint(
        meta_policy_path, config, eval_mode=False, verbose=False
    )
    robust_policy = load_policy_from_checkpoint(
        robust_policy_path, config, eval_mode=True, verbose=False
    )

    # Create environment
    target_state = get_target_state_from_config(config)
    env = create_quantum_environment(config, target_state)

    # Create hardest regime: large detuning + high decoherence
    task_ranges = {
        'alpha': (0.5, 2.0),  # Full range
        'A': (0.2, 0.3),  # High decoherence
        'omega_c': (9.0, 12.0)  # OOD detuning
    }
    task_dist = TaskDistribution(dist_type='uniform', ranges=task_ranges)
    hard_tasks = task_dist.sample(n_hard_tasks)

    fidelities = {
        'robust': [],
        'meta_init': [],
        'meta_adapted': []
    }

    for i, task in enumerate(hard_tasks):
        print(f"  Task {i+1}/{n_hard_tasks}...", end='\r')
        task_features = torch.tensor([task.alpha, task.A, task.omega_c], dtype=torch.float32)

        # Robust
        with torch.no_grad():
            controls_robust = robust_policy(task_features).detach().numpy()
        fid_robust = env.compute_fidelity(controls_robust, task)
        fidelities['robust'].append(fid_robust)

        # Meta init
        with torch.no_grad():
            controls_meta_init = meta_policy_template(task_features).detach().numpy()
        fid_meta_init = env.compute_fidelity(controls_meta_init, task)
        fidelities['meta_init'].append(fid_meta_init)

        # Meta adapted
        import copy
        adapted_policy = copy.deepcopy(meta_policy_template)
        adapted_policy.train()

        for k in range(K_fixed):
            loss = env.compute_loss_differentiable(adapted_policy, task, device=torch.device('cpu'))
            loss.backward()
            with torch.no_grad():
                for param in adapted_policy.parameters():
                    if param.grad is not None:
                        param -= config['inner_lr'] * param.grad
                        param.grad.zero_()

        adapted_policy.eval()
        with torch.no_grad():
            controls_meta_adapted = adapted_policy(task_features).detach().numpy()
        fid_meta_adapted = env.compute_fidelity(controls_meta_adapted, task)
        fidelities['meta_adapted'].append(fid_meta_adapted)

    print()
    return fidelities


def run_baseline_comparison_experiment(
    meta_policy_path: str,
    robust_policy_path: str,
    config: Dict,
    n_test_tasks: int = 5,
    K_fixed: int = 5,
    include_grape: bool = False,
    grape_iterations: int = 50
) -> Dict:
    """
    Panel 4(c): Bar plot comparison

    Compare:
      - GRAPE (per-task optimization)
      - Robust controller
      - Domain randomization (if available)
      - Meta + adaptation
    """
    print("Running baseline comparison experiment...")

    # Load policies
    meta_policy_template = load_policy_from_checkpoint(
        meta_policy_path, config, eval_mode=False, verbose=False
    )
    robust_policy = load_policy_from_checkpoint(
        robust_policy_path, config, eval_mode=True, verbose=False
    )

    # Create environment
    target_state = get_target_state_from_config(config)
    env = create_quantum_environment(config, target_state)

    # Hard task distribution
    task_ranges = {
        'alpha': (0.5, 2.0),
        'A': (0.2, 0.3),
        'omega_c': (9.0, 12.0)
    }
    task_dist = TaskDistribution(dist_type='uniform', ranges=task_ranges)
    test_tasks = task_dist.sample(n_test_tasks)

    fidelities = {
        'grape': [],
        'robust': [],
        'meta_adapted': []
    }

    for i, task in enumerate(test_tasks):
        if i % 2 == 0:
            print(f"  Task {i}/{n_test_tasks}...")

        task_features = torch.tensor([task.alpha, task.A, task.omega_c], dtype=torch.float32)

        # Robust
        with torch.no_grad():
            controls_robust = robust_policy(task_features).detach().numpy()
        fid_robust = env.compute_fidelity(controls_robust, task)
        fidelities['robust'].append(fid_robust)

        # Meta adapted
        import copy
        adapted_policy = copy.deepcopy(meta_policy_template)
        adapted_policy.train()

        for k in range(K_fixed):
            loss = env.compute_loss_differentiable(adapted_policy, task, device=torch.device('cpu'))
            loss.backward()
            with torch.no_grad():
                for param in adapted_policy.parameters():
                    if param.grad is not None:
                        param -= config['inner_lr'] * param.grad
                        param.grad.zero_()

        adapted_policy.eval()
        with torch.no_grad():
            controls_meta_adapted = adapted_policy(task_features).detach().numpy()
        fid_meta_adapted = env.compute_fidelity(controls_meta_adapted, task)
        fidelities['meta_adapted'].append(fid_meta_adapted)

        # GRAPE (optional - slow)
        if include_grape and i < 20:  # Only first 20 tasks for speed
            grape = GRAPEOptimizer(
                n_segments=config['n_segments'],
                n_controls=config['n_controls'],
                T=config.get('horizon', 1.0),
                learning_rate=0.1,
                method='adam',
                device=torch.device('cpu')
            )

            def simulate_fn(controls_np, task_params):
                return env.compute_fidelity(controls_np, task_params)

            optimal_controls = grape.optimize(
                simulate_fn=simulate_fn,
                task_params=task,
                max_iterations=grape_iterations,
                verbose=False
            )
            fid_grape = env.compute_fidelity(optimal_controls, task)
            fidelities['grape'].append(fid_grape)

    # Compute statistics
    results = {
        'methods': ['Robust', 'Meta+Adapt'],
        'mean_fidelity': [
            np.mean(fidelities['robust']),
            np.mean(fidelities['meta_adapted'])
        ],
        'std_fidelity': [
            np.std(fidelities['robust']) / np.sqrt(len(fidelities['robust'])),
            np.std(fidelities['meta_adapted']) / np.sqrt(len(fidelities['meta_adapted']))
        ]
    }

    if include_grape and len(fidelities['grape']) > 0:
        results['methods'].insert(0, 'GRAPE')
        results['mean_fidelity'].insert(0, np.mean(fidelities['grape']))
        results['std_fidelity'].insert(0, np.std(fidelities['grape']) / np.sqrt(len(fidelities['grape'])))

    return results


def plot_panel_a_ood_performance(ax, results: Dict):
    """Panel 4(a): Performance vs OOD detuning"""
    ood_levels = results['ood_levels']

    ax.errorbar(ood_levels, results['fidelity_robust'],
               yerr=results['std_robust'],
               fmt='o-', markersize=8, capsize=4, linewidth=2,
               color='orange', label='Robust Baseline')

    ax.errorbar(ood_levels, results['fidelity_meta_init'],
               yerr=results['std_meta_init'],
               fmt='s-', markersize=8, capsize=4, linewidth=2,
               color='gray', label='Meta Init (K=0)', alpha=0.7)

    ax.errorbar(ood_levels, results['fidelity_meta_adapted'],
               yerr=results['std_meta_adapted'],
               fmt='o-', markersize=8, capsize=4, linewidth=2,
               color='steelblue', label='Meta + Adapt (K=5)')

    ax.set_xlabel('OOD Detuning Level', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Fidelity', fontsize=12, fontweight='bold')
    ax.set_title('(a) Robustness to Calibration Drift', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])


def plot_panel_b_heatmap(ax, fidelities: Dict):
    """Panel 4(b): Per-task heatmap"""
    # Create matrix: rows = tasks, columns = methods
    n_tasks = len(fidelities['robust'])
    matrix = np.array([
        fidelities['robust'],
        fidelities['meta_init'],
        fidelities['meta_adapted']
    ]).T

    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Robust', 'Meta-Init', 'Meta-Adapt'], fontsize=10)
    ax.set_ylabel('Test Task Index', fontsize=12, fontweight='bold')
    ax.set_title('(b) Per-Task Performance (Hard Regime)', fontsize=13, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Fidelity', fontsize=11)


def plot_panel_c_baseline_comparison(ax, results: Dict):
    """Panel 4(c): Bar plot comparison"""
    methods = results['methods']
    means = results['mean_fidelity']
    stds = results['std_fidelity']

    colors = {'GRAPE': 'green', 'Robust': 'orange', 'Meta+Adapt': 'steelblue'}
    bar_colors = [colors.get(m, 'gray') for m in methods]

    x = np.arange(len(methods))
    ax.bar(x, means, yerr=stds, capsize=5, color=bar_colors, alpha=0.8, edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylabel('Mean Fidelity', fontsize=12, fontweight='bold')
    ax.set_title('(c) Method Comparison (Hard Regime)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.05])

    # Add text labels on bars
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.02, f'{m:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')


def plot_panel_d_placeholder(ax):
    """Panel 4(d): Optional cross-family transfer"""
    ax.text(0.5, 0.5, 'Panel 4(d):\nCross-Family Transfer\n(Optional)',
           ha='center', va='center', fontsize=14, transform=ax.transAxes)
    ax.axis('off')


def generate_figure4(
    meta_policy_path: str,
    robust_policy_path: str,
    config: Dict,
    output_dir: str = "results/icml_figures",
    run_experiments: bool = True
):
    """Generate complete Figure 4 with all panels"""
    print("=" * 80)
    print("GENERATING FIGURE 4: Robustness and OOD Performance")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)

    if run_experiments:
        # Panel (a)
        print("\n[1/3] Running OOD detuning experiment...")
        results_ood = run_ood_detuning_experiment(
            meta_policy_path, robust_policy_path, config,
            ood_detuning_levels=[1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
            K_fixed=5,
            n_test_tasks=5
        )

        # Panel (b)
        print("\n[2/3] Running per-task heatmap experiment...")
        results_heatmap = run_per_task_heatmap_experiment(
            meta_policy_path, robust_policy_path, config,
            n_hard_tasks=5,
            K_fixed=5
        )

        # Panel (c)
        print("\n[3/3] Running baseline comparison experiment...")
        results_comparison = run_baseline_comparison_experiment(
            meta_policy_path, robust_policy_path, config,
            n_test_tasks=5,
            K_fixed=5,
            include_grape=False  # Set to True if you want GRAPE (slow)
        )

        # Save results
        results = {
            'ood_detuning': results_ood,
            'heatmap': results_heatmap,
            'comparison': results_comparison
        }

        with open(f"{output_dir}/figure4_data.json", 'w') as f:
            def convert(obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                if isinstance(obj, dict):
                    return {k: convert(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [convert(item) for item in obj]
                return obj
            json.dump(convert(results), f, indent=2)
    else:
        # Load existing results
        with open(f"{output_dir}/figure4_data.json", 'r') as f:
            results = json.load(f)
        results_ood = results['ood_detuning']
        results_heatmap = results['heatmap']
        results_comparison = results['comparison']

    # Create figure
    print("\nGenerating figure...")
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    ax_a = axes[0, 0]
    ax_b = axes[0, 1]
    ax_c = axes[1, 0]
    ax_d = axes[1, 1]

    # Generate panels
    plot_panel_a_ood_performance(ax_a, results_ood)
    plot_panel_b_heatmap(ax_b, results_heatmap)
    plot_panel_c_baseline_comparison(ax_c, results_comparison)
    plot_panel_d_placeholder(ax_d)

    plt.suptitle('Figure 4: Robustness to Distribution Shift / Device Personalization',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    # Save
    output_path = f"{output_dir}/figure4_robustness.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to {output_path}")

    output_path_png = f"{output_dir}/figure4_robustness.png"
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {output_path_png}")

    plt.close()

    print("\n" + "=" * 80)
    print("FIGURE 4 COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    # Configuration
    config = {
        'num_qubits': 1,
        'n_controls': 2,
        'n_segments': 20,
        'horizon': 1.0,
        'target_gate': 'paulix',
        'hidden_dim': 128,
        'n_hidden_layers': 2,
        'inner_lr': 0.01,
        'noise_frequencies': [1.0, 5.0, 10.0]
    }


    # Find checkpoints  
    meta_path = "../checkpoints/maml_best.pt"  
    robust_path = "../checkpoints/robust_best.pt"   
    
    if meta_path is None or robust_path is None:
        print("ERROR: Trained models not found")
        print("Please train policies first")
        sys.exit(1)

    print(f"Using meta policy: {meta_path}")
    print(f"Using robust policy: {robust_path}")

    # Generate figure
    generate_figure4(
        meta_policy_path=meta_path,
        robust_policy_path=robust_path,
        config=config,
        output_dir="results/icml_figures",
        run_experiments=True
    )
