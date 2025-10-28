"""
Figure 2 — Fast Adaptation Scaling: Theory Meets Experiment

This script generates all panels for Figure 2:
  (a) Schematic of adaptation gain vs K
  (b) Empirical gap vs K (high variance)
  (c) Empirical gap vs K (low variance)
  (d) Gap vs detuning spread
  (e) Gap vs decoherence/actuator spread (optional)
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
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from metaqctrl.quantum.noise_models import TaskDistribution, NoiseParameters
from metaqctrl.meta_rl.policy import PulsePolicy
from metaqctrl.theory.quantum_environment import create_quantum_environment, get_target_state_from_config
from metaqctrl.utils.checkpoint_utils import load_policy_from_checkpoint


def exponential_model(K, gap_max, mu_eta):
    """Theoretical model: Gap(K) = gap_max * (1 - exp(-μηK))"""
    return gap_max * (1 - np.exp(-mu_eta * K))


def plot_panel_a_schematic(ax):
    """Panel 2(a): Clean schematic curve"""
    K = np.linspace(0, 20, 100)
    gap = 0.15 * (1 - np.exp(-0.3 * K))

    ax.plot(K, gap, 'b-', linewidth=3)

    # Annotations
    ax.annotate('Rapid improvement\n(1-2 steps)', xy=(2, 0.06), xytext=(5, 0.10),
                arrowprops=dict(arrowstyle='->', lw=2), fontsize=11, ha='left')
    ax.annotate('Diminishing\nreturns', xy=(8, 0.13), xytext=(12, 0.11),
                arrowprops=dict(arrowstyle='->', lw=2), fontsize=11, ha='left')
    ax.annotate('Saturation', xy=(18, 0.145), xytext=(15, 0.125),
                arrowprops=dict(arrowstyle='->', lw=2), fontsize=11, ha='right')

    ax.set_xlabel('Adaptation Steps (K)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Adaptation Gain', fontsize=12, fontweight='bold')
    ax.set_title('(a) Theoretical Behavior', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 0.16])


def run_gap_vs_k_experiment(
    meta_policy_path: str,
    robust_policy_path: str,
    config: Dict,
    task_dist_config: Dict,  # For creating distributions with different variances
    k_values: List[int] = [1, 2, 3, 5, 7, 10, 15, 20],
    n_test_tasks: int = 100
) -> Dict:
    """
    Run gap vs K experiment for a given task distribution

    Args:
        task_dist_config: Dict with 'alpha_range', 'A_range', 'omega_c_range'
    """
    print(f"Running gap vs K experiment...")
    print(f"  Task ranges: α={task_dist_config['alpha_range']}, "
          f"A={task_dist_config['A_range']}, ω_c={task_dist_config['omega_c_range']}")

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

    # Sample test tasks
    task_dist = TaskDistribution(
        dist_type='uniform',
        ranges=task_dist_config
    )
    test_tasks = task_dist.sample(n_test_tasks)

    # Measure gap for each K
    gaps_mean = []
    gaps_std = []

    for K in k_values:
        print(f"  K={K}...", end=' ')
        fidelities_meta = []
        fidelities_robust = []

        for task in test_tasks:
            task_features = torch.tensor(
                [task.alpha, task.A, task.omega_c],
                dtype=torch.float32
            )

            # Robust policy
            with torch.no_grad():
                controls_robust = robust_policy(task_features).detach().numpy()
            fid_robust = env.compute_fidelity(controls_robust, task)
            fidelities_robust.append(fid_robust)

            # Meta policy with K adaptation steps
            import copy
            adapted_policy = copy.deepcopy(meta_policy_template)
            adapted_policy.train()

            for k in range(K):
                loss = env.compute_loss_differentiable(
                    adapted_policy, task, device=torch.device('cpu')
                )
                loss.backward()
                with torch.no_grad():
                    for param in adapted_policy.parameters():
                        if param.grad is not None:
                            param -= config['inner_lr'] * param.grad
                            param.grad.zero_()

            # Evaluate
            adapted_policy.eval()
            with torch.no_grad():
                controls_meta = adapted_policy(task_features).detach().numpy()
            fid_meta = env.compute_fidelity(controls_meta, task)
            fidelities_meta.append(fid_meta)

        gap_tasks = np.array(fidelities_meta) - np.array(fidelities_robust)
        gaps_mean.append(np.mean(gap_tasks))
        gaps_std.append(np.std(gap_tasks) / np.sqrt(n_test_tasks))
        print(f"Gap={gaps_mean[-1]:.4f}±{gaps_std[-1]:.4f}")

    # Fit theoretical model
    k_array = np.array(k_values)
    gaps_array = np.array(gaps_mean)

    try:
        popt, pcov = curve_fit(
            exponential_model,
            k_array,
            gaps_array,
            p0=[gaps_array[-1], 0.1],
            sigma=gaps_std,
            absolute_sigma=True,
            maxfev=10000
        )
        gap_max_fit, mu_eta_fit = popt
        gaps_predicted = exponential_model(k_array, *popt)
        r2 = r2_score(gaps_array, gaps_predicted)
        fit_success = True
        print(f"  Fit: gap_max={gap_max_fit:.4f}, μη={mu_eta_fit:.4f}, R²={r2:.4f}")
    except Exception as e:
        print(f"  Fitting failed: {e}")
        gap_max_fit = mu_eta_fit = r2 = None
        fit_success = False

    return {
        'k_values': k_values,
        'gaps_mean': gaps_mean,
        'gaps_std': gaps_std,
        'fit': {
            'success': fit_success,
            'gap_max': gap_max_fit,
            'mu_eta': mu_eta_fit,
            'r2': r2
        },
        'task_dist_config': task_dist_config
    }


def plot_panel_bc_empirical(ax_b, ax_c, results_high, results_low):
    """Panels 2(b) and 2(c): Empirical gap vs K"""

    # Panel (b): High variance
    k_vals = np.array(results_high['k_values'])
    gaps = np.array(results_high['gaps_mean'])
    gaps_err = np.array(results_high['gaps_std'])

    ax_b.errorbar(k_vals, gaps, yerr=gaps_err, fmt='o', markersize=8,
                  capsize=4, label='Empirical', color='steelblue', linewidth=2)

    if results_high['fit']['success']:
        k_fine = np.linspace(0, max(k_vals), 100)
        gap_pred = exponential_model(k_fine, results_high['fit']['gap_max'],
                                     results_high['fit']['mu_eta'])
        r2 = results_high['fit']['r2']
        ax_b.plot(k_fine, gap_pred, '--', label=f'Fit: $R^2={r2:.3f}$',
                 color='darkred', linewidth=2)

    ax_b.set_xlabel('Adaptation Steps (K)', fontsize=12, fontweight='bold')
    ax_b.set_ylabel('Adaptation Gap', fontsize=12, fontweight='bold')
    ax_b.set_title('(b) High Variance Regime', fontsize=13, fontweight='bold')
    ax_b.legend(fontsize=10)
    ax_b.grid(True, alpha=0.3)

    # Panel (c): Low variance
    k_vals = np.array(results_low['k_values'])
    gaps = np.array(results_low['gaps_mean'])
    gaps_err = np.array(results_low['gaps_std'])

    ax_c.errorbar(k_vals, gaps, yerr=gaps_err, fmt='o', markersize=8,
                  capsize=4, label='Empirical', color='steelblue', linewidth=2)

    if results_low['fit']['success']:
        k_fine = np.linspace(0, max(k_vals), 100)
        gap_pred = exponential_model(k_fine, results_low['fit']['gap_max'],
                                     results_low['fit']['mu_eta'])
        r2 = results_low['fit']['r2']
        ax_c.plot(k_fine, gap_pred, '--', label=f'Fit: $R^2={r2:.3f}$',
                 color='darkred', linewidth=2)

    ax_c.set_xlabel('Adaptation Steps (K)', fontsize=12, fontweight='bold')
    ax_c.set_ylabel('Adaptation Gap', fontsize=12, fontweight='bold')
    ax_c.set_title('(c) Low Variance Regime', fontsize=13, fontweight='bold')
    ax_c.legend(fontsize=10)
    ax_c.grid(True, alpha=0.3)


def run_gap_vs_diversity_experiment(
    meta_policy_path: str,
    robust_policy_path: str,
    config: Dict,
    diversity_type: str = 'detuning',  # 'detuning', 'decoherence', or 'actuator'
    diversity_levels: List[float] = None,
    K_fixed: int = 5,
    n_test_tasks: int = 100
) -> Dict:
    """
    Panel 2(d)/(e): Gap vs physically meaningful diversity

    Args:
        diversity_type: What to vary
        diversity_levels: Range levels (e.g., detuning spreads)
    """
    print(f"\nRunning gap vs {diversity_type} diversity experiment...")

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

    if diversity_levels is None:
        if diversity_type == 'detuning':
            # Vary detuning spread (in terms of omega_c range)
            diversity_levels = [1.0, 2.0, 3.0, 4.0, 6.0]
        elif diversity_type == 'decoherence':
            # Vary decoherence amplitude range
            diversity_levels = [0.05, 0.10, 0.15, 0.20, 0.30]
        else:  # actuator
            # Vary amplitude range
            diversity_levels = [0.05, 0.10, 0.15, 0.25, 0.40]

    gaps_mean = []
    gaps_std = []

    for div_level in diversity_levels:
        print(f"  Diversity level={div_level}...", end=' ')

        # Create task distribution for this diversity level
        if diversity_type == 'detuning':
            omega_c_center = 5.0
            task_ranges = {
                'alpha': (1.0, 1.5),  # Fixed
                'A': (0.1, 0.2),  # Fixed
                'omega_c': (omega_c_center - div_level, omega_c_center + div_level)
            }
        elif diversity_type == 'decoherence':
            task_ranges = {
                'alpha': (1.0, 1.5),  # Fixed
                'A': (0.15 - div_level/2, 0.15 + div_level/2),
                'omega_c': (4.0, 6.0)  # Fixed
            }
        else:  # actuator limits (affects control amplitude, simulated via A)
            task_ranges = {
                'alpha': (1.0, 1.5),
                'A': (0.05, 0.05 + div_level),
                'omega_c': (4.0, 6.0)
            }

        task_dist = TaskDistribution(dist_type='uniform', ranges=task_ranges)
        test_tasks = task_dist.sample(n_test_tasks)

        fidelities_meta = []
        fidelities_robust = []

        for task in test_tasks:
            task_features = torch.tensor([task.alpha, task.A, task.omega_c], dtype=torch.float32)

            # Robust
            with torch.no_grad():
                controls_robust = robust_policy(task_features).detach().numpy()
            fid_robust = env.compute_fidelity(controls_robust, task)
            fidelities_robust.append(fid_robust)

            # Meta with K steps
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
                controls_meta = adapted_policy(task_features).detach().numpy()
            fid_meta = env.compute_fidelity(controls_meta, task)
            fidelities_meta.append(fid_meta)

        gap_tasks = np.array(fidelities_meta) - np.array(fidelities_robust)
        gaps_mean.append(np.mean(gap_tasks))
        gaps_std.append(np.std(gap_tasks) / np.sqrt(n_test_tasks))
        print(f"Gap={gaps_mean[-1]:.4f}±{gaps_std[-1]:.4f}")

    return {
        'diversity_type': diversity_type,
        'diversity_levels': diversity_levels,
        'gaps_mean': gaps_mean,
        'gaps_std': gaps_std,
        'K_fixed': K_fixed
    }


def plot_panel_de_diversity(ax_d, ax_e, results_det, results_dec=None):
    """Panels 2(d) and 2(e): Gap vs diversity"""

    # Panel (d): Detuning
    divs = results_det['diversity_levels']
    gaps = results_det['gaps_mean']
    gaps_err = results_det['gaps_std']

    ax_d.errorbar(divs, gaps, yerr=gaps_err, fmt='o-', markersize=8,
                  capsize=4, color='steelblue', linewidth=2)
    ax_d.set_xlabel('Detuning Spread (a.u.)', fontsize=12, fontweight='bold')
    ax_d.set_ylabel('Adaptation Gap', fontsize=12, fontweight='bold')
    ax_d.set_title('(d) Gap vs Calibration Mismatch', fontsize=13, fontweight='bold')
    ax_d.grid(True, alpha=0.3)

    # Panel (e): Decoherence/actuator (optional)
    if results_dec is not None:
        divs = results_dec['diversity_levels']
        gaps = results_dec['gaps_mean']
        gaps_err = results_dec['gaps_std']

        ax_e.errorbar(divs, gaps, yerr=gaps_err, fmt='o-', markersize=8,
                      capsize=4, color='green', linewidth=2)
        div_label = 'Decoherence Spread' if results_dec['diversity_type'] == 'decoherence' else 'Actuator Limit Spread'
        ax_e.set_xlabel(div_label, fontsize=12, fontweight='bold')
        ax_e.set_ylabel('Adaptation Gap', fontsize=12, fontweight='bold')
        ax_e.set_title('(e) Gap vs Hardware Constraints', fontsize=13, fontweight='bold')
        ax_e.grid(True, alpha=0.3)
    else:
        ax_e.text(0.5, 0.5, 'Optional Panel\n(not computed)',
                 ha='center', va='center', fontsize=14, transform=ax_e.transAxes)
        ax_e.axis('off')


def generate_figure2(
    meta_policy_path: str,
    robust_policy_path: str,
    config: Dict,
    output_dir: str = "results/icml_figures",
    run_experiments: bool = True
):
    """
    Generate complete Figure 2 with all panels
    """
    print("=" * 80)
    print("GENERATING FIGURE 2: Fast Adaptation Scaling")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)

    if run_experiments:
        # Panel (b): High variance
        print("\n[1/4] Running high variance experiment...")
        results_high = run_gap_vs_k_experiment(
            meta_policy_path, robust_policy_path, config,
            task_dist_config={
                'alpha': (0.5, 2.0),
                'A': (0.05, 0.3),
                'omega_c': (2.0, 8.0)
            },
            k_values=[1, 2, 3, 5, 7, 10, 15, 20],
            n_test_tasks=100
        )

        # Panel (c): Low variance
        print("\n[2/4] Running low variance experiment...")
        results_low = run_gap_vs_k_experiment(
            meta_policy_path, robust_policy_path, config,
            task_dist_config={
                'alpha': (1.0, 1.5),
                'A': (0.15, 0.20),
                'omega_c': (4.0, 6.0)
            },
            k_values=[1, 2, 3, 5, 7, 10, 15, 20],
            n_test_tasks=100
        )

        # Panel (d): Detuning diversity
        print("\n[3/4] Running detuning diversity experiment...")
        results_detuning = run_gap_vs_diversity_experiment(
            meta_policy_path, robust_policy_path, config,
            diversity_type='detuning',
            diversity_levels=[1.0, 2.0, 3.0, 4.0, 6.0],
            K_fixed=5,
            n_test_tasks=100
        )

        # Panel (e): Decoherence diversity (optional)
        print("\n[4/4] Running decoherence diversity experiment...")
        try:
            results_decoherence = run_gap_vs_diversity_experiment(
                meta_policy_path, robust_policy_path, config,
                diversity_type='decoherence',
                diversity_levels=[0.05, 0.10, 0.15, 0.20, 0.30],
                K_fixed=5,
                n_test_tasks=100
            )
        except Exception as e:
            print(f"  Decoherence experiment failed: {e}")
            results_decoherence = None

        # Save results
        results = {
            'high_variance': results_high,
            'low_variance': results_low,
            'detuning_diversity': results_detuning,
            'decoherence_diversity': results_decoherence
        }

        with open(f"{output_dir}/figure2_data.json", 'w') as f:
            # Convert None to serializable
            def convert(obj):
                if obj is None:
                    return None
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
        with open(f"{output_dir}/figure2_data.json", 'r') as f:
            results = json.load(f)
        results_high = results['high_variance']
        results_low = results['low_variance']
        results_detuning = results['detuning_diversity']
        results_decoherence = results.get('decoherence_diversity')

    # Create figure
    print("\nGenerating figure...")
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(18, 10))

    # Layout: 2 rows x 3 columns
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])

    # Generate panels
    plot_panel_a_schematic(ax_a)
    plot_panel_bc_empirical(ax_b, ax_c, results_high, results_low)
    plot_panel_de_diversity(ax_d, ax_e, results_detuning, results_decoherence)

    plt.suptitle('Figure 2: Fast Adaptation Scaling — Theory Meets Experiment',
                 fontsize=16, fontweight='bold', y=0.98)

    # Save
    output_path = f"{output_dir}/figure2_complete.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {output_path}")

    output_path_png = f"{output_dir}/figure2_complete.png"
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {output_path_png}")

    plt.close()

    print("\n" + "=" * 80)
    print("FIGURE 2 COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    # Configuration
    config = {
        'num_qubits': 1,
        'n_controls': 2,
        'n_segments': 20,
        'horizon': 1.0,
        'target_gate': 'hadamard',
        'hidden_dim': 128,
        'n_hidden_layers': 2,
        'inner_lr': 0.01,
        'noise_frequencies': [1.0, 5.0, 10.0]
    }

    # Find checkpoints
    script_dir = Path(__file__).parent
    checkpoint_dir = script_dir.parent.parent / "checkpoints"

    possible_meta_paths = [
        checkpoint_dir / "maml_best.pt",
        checkpoint_dir / "maml_20251027_161519_best_policy.pt",
    ]
    possible_robust_paths = [
        checkpoint_dir / "robust_best.pt",
        checkpoint_dir / "robust_minimax_20251027_162238_best_policy.pt",
    ]

    meta_path = None
    for p in possible_meta_paths:
        if p.exists():
            meta_path = str(p)
            break

    robust_path = None
    for p in possible_robust_paths:
        if p.exists():
            robust_path = str(p)
            break

    if meta_path is None or robust_path is None:
        print("ERROR: Trained models not found")
        print("Please train policies first using experiments/train_meta.py and train_robust.py")
        sys.exit(1)

    print(f"Using meta policy: {meta_path}")
    print(f"Using robust policy: {robust_path}")

    # Generate figure
    generate_figure2(
        meta_policy_path=meta_path,
        robust_policy_path=robust_path,
        config=config,
        output_dir="results/icml_figures",
        run_experiments=True
    )
