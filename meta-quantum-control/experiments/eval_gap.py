"""
Evaluate Optimality Gap

Compare meta-learned policy vs robust baseline.
Generate plots for Section 7.3 (Optimality Gap Validation).
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from tqdm import tqdm

from src.quantum.lindblad import LindbladSimulator
from src.quantum.noise_models import TaskDistribution, PSDToLindblad, NoisePSDModel
from src.quantum.gates import state_fidelity, TargetGates
from src.meta_rl.policy import PulsePolicy
from src.meta_rl.maml import MAML
from src.baselines.robust_control import RobustPolicy
from src.theory.optimality_gap import OptimalityGapComputer, GapConstants

# Import system creation from train_meta
from train_meta import create_quantum_system, create_task_distribution


def load_models(meta_path: str, robust_path: str, config: dict, device: torch.device):
    """Load trained meta and robust policies."""
    
    # Create policy architecture
    policy_config = {
        'task_feature_dim': config.get('task_feature_dim', 3),
        'hidden_dim': config.get('hidden_dim', 128),
        'n_hidden_layers': config.get('n_hidden_layers', 2),
        'n_segments': config.get('n_segments', 20),
        'n_controls': config.get('n_controls', 2),
        'output_scale': config.get('output_scale', 0.5),
        'activation': config.get('activation', 'tanh')
    }
    
    # Load meta-learned policy
    meta_policy = PulsePolicy(**policy_config).to(device)
    meta_checkpoint = torch.load(meta_path, map_location=device)
    meta_policy.load_state_dict(meta_checkpoint['policy_state_dict'])
    print(f"Loaded meta policy from {meta_path}")
    
    # Load robust policy
    robust_policy = PulsePolicy(**policy_config).to(device)
    robust_checkpoint = torch.load(robust_path, map_location=device)
    robust_policy.load_state_dict(robust_checkpoint['policy_state_dict'])
    print(f"Loaded robust policy from {robust_path}")
    
    return meta_policy, robust_policy


def evaluate_fidelity(
    policy: torch.nn.Module,
    task_params,
    quantum_system: dict,
    target_state: np.ndarray,
    T: float,
    device: torch.device,
    adapt: bool = False,
    K: int = 5,
    inner_lr: float = 0.01
) -> float:
    """
    Evaluate fidelity of policy on a task.
    
    Args:
        policy: Policy network
        task_params: NoiseParameters
        quantum_system: System dict
        target_state: Target density matrix
        T: Evolution time
        device: torch device
        adapt: If True, perform K gradient steps before evaluation
        K: Number of adaptation steps
        inner_lr: Adaptation learning rate
        
    Returns:
        fidelity: Achieved fidelity [0, 1]
    """
    from copy import deepcopy
    
    if adapt:
        # Clone and adapt
        adapted_policy = deepcopy(policy)
        adapted_policy.train()
        optimizer = torch.optim.SGD(adapted_policy.parameters(), lr=inner_lr)
        
        task_features = torch.tensor(
            task_params.to_array(), dtype=torch.float32, device=device
        )
        
        for _ in range(K):
            optimizer.zero_grad()
            controls = adapted_policy(task_features)
            
            # Simulate and compute loss
            controls_np = controls.detach().cpu().numpy()
            L_ops = quantum_system['psd_to_lindblad'].get_lindblad_operators(task_params)
            
            sim = LindbladSimulator(
                H0=quantum_system['H0'],
                H_controls=quantum_system['H_controls'],
                L_operators=L_ops,
                method='RK45'
            )
            
            rho0 = np.array([[1, 0], [0, 0]], dtype=complex)
            rho_final, _ = sim.evolve(rho0, controls_np, T)
            
            fidelity = state_fidelity(rho_final, target_state)
            loss = torch.tensor(1.0 - fidelity, dtype=torch.float32, device=device)
            
            loss.backward()
            optimizer.step()
        
        eval_policy = adapted_policy
    else:
        eval_policy = policy
    
    # Final evaluation
    eval_policy.eval()
    with torch.no_grad():
        task_features = torch.tensor(
            task_params.to_array(), dtype=torch.float32, device=device
        )
        controls = eval_policy(task_features)
    
    controls_np = controls.cpu().numpy()
    L_ops = quantum_system['psd_to_lindblad'].get_lindblad_operators(task_params)
    
    sim = LindbladSimulator(
        H0=quantum_system['H0'],
        H_controls=quantum_system['H_controls'],
        L_operators=L_ops,
        method='RK45'
    )
    
    rho0 = np.array([[1, 0], [0, 0]], dtype=complex)
    rho_final, _ = sim.evolve(rho0, controls_np, T)
    
    fidelity = state_fidelity(rho_final, target_state)
    
    return fidelity


def compute_gap_vs_K(
    meta_policy: torch.nn.Module,
    robust_policy: torch.nn.Module,
    test_tasks: list,
    K_values: list,
    quantum_system: dict,
    target_state: np.ndarray,
    config: dict,
    device: torch.device
) -> dict:
    """Compute gap as function of adaptation steps K."""
    
    print("\nComputing gap vs K...")
    T = config.get('horizon', 1.0)
    inner_lr = config.get('inner_lr', 0.01)
    
    results = {'K': [], 'gap': [], 'meta_fid': [], 'robust_fid': []}
    
    for K in tqdm(K_values, desc="K values"):
        meta_fidelities = []
        robust_fidelities = []
        
        for task in test_tasks:
            # Meta with adaptation
            F_meta = evaluate_fidelity(
                meta_policy, task, quantum_system, target_state, T, device,
                adapt=True, K=K, inner_lr=inner_lr
            )
            meta_fidelities.append(F_meta)
            
            # Robust without adaptation
            F_robust = evaluate_fidelity(
                robust_policy, task, quantum_system, target_state, T, device,
                adapt=False
            )
            robust_fidelities.append(F_robust)
        
        mean_meta = np.mean(meta_fidelities)
        mean_robust = np.mean(robust_fidelities)
        gap = mean_meta - mean_robust
        
        results['K'].append(K)
        results['gap'].append(gap)
        results['meta_fid'].append(mean_meta)
        results['robust_fid'].append(mean_robust)
        
        print(f"  K={K:2d}: Gap = {gap:.4f}, Meta = {mean_meta:.4f}, Robust = {mean_robust:.4f}")
    
    return results


def compute_gap_vs_variance(
    meta_policy: torch.nn.Module,
    robust_policy: torch.nn.Module,
    base_config: dict,
    variance_multipliers: list,
    quantum_system: dict,
    target_state: np.ndarray,
    device: torch.device,
    n_tasks_per_variance: int = 50
) -> dict:
    """Compute gap as function of task distribution variance."""
    
    print("\nComputing gap vs variance...")
    K = base_config.get('inner_steps', 5)
    T = base_config.get('horizon', 1.0)
    inner_lr = base_config.get('inner_lr', 0.01)
    
    results = {'variance': [], 'gap': [], 'meta_fid': [], 'robust_fid': []}
    
    # Base ranges
    base_alpha = base_config.get('alpha_range', [0.5, 2.0])
    base_A = base_config.get('A_range', [0.05, 0.3])
    base_omega = base_config.get('omega_c_range', [2.0, 8.0])
    
    alpha_center = np.mean(base_alpha)
    A_center = np.mean(base_A)
    omega_center = np.mean(base_omega)
    
    for mult in tqdm(variance_multipliers, desc="Variance levels"):
        # Scale ranges around center
        alpha_range = [
            alpha_center - (alpha_center - base_alpha[0]) * mult,
            alpha_center + (base_alpha[1] - alpha_center) * mult
        ]
        A_range = [
            A_center - (A_center - base_A[0]) * mult,
            A_center + (base_A[1] - A_center) * mult
        ]
        omega_range = [
            omega_center - (omega_center - base_omega[0]) * mult,
            omega_center + (base_omega[1] - omega_center) * mult
        ]
        
        # Create task distribution with this variance
        task_dist = TaskDistribution(
            dist_type='uniform',
            ranges={
                'alpha': tuple(alpha_range),
                'A': tuple(A_range),
                'omega_c': tuple(omega_range)
            }
        )
        
        variance = task_dist.compute_variance()
        
        # Sample tasks
        rng = np.random.default_rng(42)
        tasks = task_dist.sample(n_tasks_per_variance, rng)
        
        # Evaluate
        meta_fidelities = []
        robust_fidelities = []
        
        for task in tasks:
            F_meta = evaluate_fidelity(
                meta_policy, task, quantum_system, target_state, T, device,
                adapt=True, K=K, inner_lr=inner_lr
            )
            meta_fidelities.append(F_meta)
            
            F_robust = evaluate_fidelity(
                robust_policy, task, quantum_system, target_state, T, device,
                adapt=False
            )
            robust_fidelities.append(F_robust)
        
        mean_meta = np.mean(meta_fidelities)
        mean_robust = np.mean(robust_fidelities)
        gap = mean_meta - mean_robust
        
        results['variance'].append(variance)
        results['gap'].append(gap)
        results['meta_fid'].append(mean_meta)
        results['robust_fid'].append(mean_robust)
        
        print(f"  σ²={variance:.4f}: Gap = {gap:.4f}")
    
    return results


def plot_results(gap_vs_K: dict, gap_vs_var: dict, constants: GapConstants, save_dir: Path):
    """Generate plots for paper."""
    
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Gap vs K
    ax = axes[0]
    K_vals = np.array(gap_vs_K['K'])
    gaps = np.array(gap_vs_K['gap'])
    
    ax.plot(K_vals, gaps, 'o-', linewidth=2, markersize=8, label='Empirical Gap')
    
    # Theoretical prediction (if constants available)
    if constants is not None:
        # Example: Gap ∝ (1 - e^(-μηK))
        eta = 0.01
        sigma_sq = 0.1  # Use actual variance from experiment
        K_theory = np.linspace(0, max(K_vals), 100)
        gap_theory = constants.gap_lower_bound(sigma_sq, K_theory, eta)
        ax.plot(K_theory, gap_theory, '--', linewidth=2, label='Theory Lower Bound', alpha=0.7)
    
    ax.set_xlabel('Adaptation Steps K', fontsize=12)
    ax.set_ylabel('Optimality Gap', fontsize=12)
    ax.set_title('Gap vs Adaptation Steps', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Gap vs Variance
    ax = axes[1]
    variances = np.array(gap_vs_var['variance'])
    gaps_var = np.array(gap_vs_var['gap'])
    
    ax.plot(variances, gaps_var, 's-', linewidth=2, markersize=8, label='Empirical Gap')
    
    # Linear fit (theory predicts Gap ∝ σ²)
    if len(variances) > 1:
        coeffs = np.polyfit(variances, gaps_var, 1)
        var_fit = np.linspace(min(variances), max(variances), 100)
        gap_fit = np.polyval(coeffs, var_fit)
        ax.plot(var_fit, gap_fit, '--', linewidth=2, 
                label=f'Linear Fit (slope={coeffs[0]:.3f})', alpha=0.7)
    
    ax.set_xlabel('Task Variance σ²_θ', fontsize=12)
    ax.set_ylabel('Optimality Gap', fontsize=12)
    ax.set_title('Gap vs Task Variance', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    save_path = save_dir / 'optimality_gap_analysis.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {save_path}")
    
    plt.show()


def main(args):
    """Main evaluation script."""
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 70)
    print("Optimality Gap Evaluation")
    print("=" * 70)
    print(f"Config: {args.config}\n")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Create quantum system
    print("Setting up quantum system...")
    quantum_system = create_quantum_system(config)
    
    # Target state
    target_gate_name = config.get('target_gate', 'hadamard')
    if target_gate_name == 'hadamard':
        U_target = TargetGates.hadamard()
    else:
        U_target = TargetGates.pauli_x()
    
    ket_0 = np.array([1, 0], dtype=complex)
    target_state = np.outer(U_target @ ket_0, (U_target @ ket_0).conj())
    
    # Load models
    print("\nLoading trained models...")
    meta_policy, robust_policy = load_models(
        args.meta_path, args.robust_path, config, device
    )
    
    # Sample test tasks
    print("\nSampling test tasks...")
    task_dist = create_task_distribution(config)
    rng = np.random.default_rng(config.get('seed', 42) + 200000)
    test_tasks = task_dist.sample(config.get('gap_n_samples', 100), rng)
    print(f"  Sampled {len(test_tasks)} test tasks")
    
    # Estimate theoretical constants
    print("\nEstimating theoretical constants...")
    # This is expensive - can be cached
    # constants = gap_computer.estimate_constants(meta_policy, test_tasks[:20], n_samples=20)
    constants = None  # Skip for now, or load from cache
    
    # Compute gap vs K
    K_values = config.get('gap_K_values', [1, 3, 5, 10, 20])
    gap_vs_K_results = compute_gap_vs_K(
        meta_policy, robust_policy, test_tasks[:50], K_values,
        quantum_system, target_state, config, device
    )
    
    # Compute gap vs variance
    variance_multipliers = [0.5, 0.75, 1.0, 1.25, 1.5]
    gap_vs_var_results = compute_gap_vs_variance(
        meta_policy, robust_policy, config, variance_multipliers,
        quantum_system, target_state, device, n_tasks_per_variance=30
    )
    
    # Plot
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    plot_results(gap_vs_K_results, gap_vs_var_results, constants, save_dir)
    
    # Save results
    import json
    results = {
        'gap_vs_K': gap_vs_K_results,
        'gap_vs_variance': gap_vs_var_results
    }
    
    results_path = save_dir / 'gap_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")
    
    print("\n" + "=" * 70)
    print("Evaluation complete!")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate optimality gap')
    parser.add_argument('--config', type=str, required=True, help='Config file')
    parser.add_argument('--meta_path', type=str, required=True, help='Path to meta model')
    parser.add_argument('--robust_path', type=str, required=True, help='Path to robust model')
    parser.add_argument('--save_dir', type=str, default='results/gap_eval', help='Save directory')
    
    args = parser.parse_args()
    main(args)
