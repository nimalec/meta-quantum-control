"""
Evaluate Optimality Gap

Compare meta-learned policy vs robust baseline.
Generate plots for Section 7.3 (Optimality Gap Validation).
"""

import torch
import numpy as np
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from tqdm import tqdm

from metaqctrl.quantum.lindblad import LindbladSimulator
from metaqctrl.quantum.noise_models import TaskDistribution, PSDToLindblad, NoisePSDModel
from metaqctrl.quantum.noise_adapter import PSDToLindblad2 
from metaqctrl.quantum.gates import state_fidelity, TargetGates
from metaqctrl.meta_rl.policy import PulsePolicy
from metaqctrl.meta_rl.maml import MAML  
from metaqctrl.theory.optimality_gap import OptimalityGapComputer, GapConstants
from metaqctrl.utils.checkpoint_utils import load_policy_from_checkpoint

# Import system creation from train_meta
#from ..metatrain_meta import  create_quantum_system, create_task_distribution

def create_quantum_system():
    """Create a simple 1-qubit quantum system."""
    # Pauli matrices
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    sigma_p = np.array([[0, 1], [0, 0]], dtype=complex)

    # System Hamiltonians
    H0 = 0.1 * sigma_z  # No drift
    H_controls = [sigma_x, sigma_y]

    # PSD model for noise
    psd_model = NoisePSDModel(model_type='one_over_f')
    omega_sample = np.linspace(1,1000,1000)

    psd_to_lindblad = PSDToLindblad2(
        basis_operators=[sigma_p, sigma_z],
        sampling_freqs=omega_sample,
        psd_model=psd_model
    )

    return H0, H_controls, psd_to_lindblad

def create_task_distribution(config: dict):
    ## Create a distribution of tasks , generating P
    """Create task distribution P (with optional mixed model support)."""
    # NEW: Support for mixed model sampling
    model_types = config.get('model_types')
    model_probs = config.get('model_probs')

    return TaskDistribution(
        dist_type=config.get('task_dist_type'),
        ranges={
            'alpha': tuple(config.get('alpha_range')),
            'A': tuple(config.get('A_range')),
            'omega_c': tuple(config.get('omega_c_range'))
        },
        model_types=model_types,  # NEW: List of model types or None for single model
        model_probs=model_probs   # NEW: Probabilities for each model type
    )

def load_models(meta_path: str, config: dict, device: torch.device):
    """Load trained meta and robust policies with automatic architecture detection."""

    # Load meta-learned policy with auto architecture detection
    meta_policy = load_policy_from_checkpoint(
        meta_path, config, device=device, eval_mode=True, verbose=True
    )
    print(f"Loaded meta policy from {meta_path}")
 
    return meta_policy 


def evaluate_fidelity(
    policy: torch.nn.Module,
    task_params,
    quantum_system: dict,
    target_state: np.ndarray,
    T: float,
    device: torch.device,
    adapt: bool = False,
    K: int = 5,
    inner_lr: float = 0.01,
    env=None  # NEW: Pass QuantumEnvironment for differentiable simulation
) -> float:
    """
    Evaluate fidelity of policy on a task.

    Args:
        policy: Policy network
        task_params: NoiseParameters
        quantum_system: System dict (DEPRECATED - use env instead)
        target_state: Target density matrix
        T: Evolution time
        device: torch device
        adapt: If True, perform K gradient steps before evaluation
        K: Number of adaptation steps
        inner_lr: Adaptation learning rate
        env: QuantumEnvironment instance (NEW - enables differentiable adaptation)

    Returns:
        fidelity: Achieved fidelity [0, 1]
    """
    from copy import deepcopy

    if adapt:
        # Clone and adapt
        adapted_policy = deepcopy(policy)
        adapted_policy.train()
        optimizer = torch.optim.SGD(adapted_policy.parameters(), lr=inner_lr)

        # K gradient steps with PROPER gradient flow
        for _ in range(K):
            optimizer.zero_grad()

            # FIXED: Use differentiable loss computation
            if env is not None:
                # NEW: Fully differentiable path through quantum simulation
                loss = env.compute_loss_differentiable(adapted_policy, task_params, device)
            else:
                # FALLBACK: Old non-differentiable path (for backwards compatibility)
                task_features = torch.tensor(
                    task_params.to_array(), dtype=torch.float32, device=device
                )
                controls = adapted_policy(task_features)
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
                fidelity_np = state_fidelity(rho_final, target_state)
                loss = torch.tensor(1.0 - fidelity_np, dtype=torch.float32, device=device)

            # Backprop (now with actual gradients!)
            loss.backward()
            optimizer.step()

        eval_policy = adapted_policy
    else:
        eval_policy = policy

    # Final evaluation
    eval_policy.eval()
    with torch.no_grad():
        if env is not None:
            # Use environment for evaluation
            fidelity = env.evaluate_policy(eval_policy, task_params, device)
        else:
            # Fallback to old method
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
    test_tasks: list,
    K_values: list,
    quantum_system: dict,
    target_state: np.ndarray,
    config: dict,
    device: torch.device,
    env=None  # NEW: Pass environment for differentiable adaptation
) -> dict:
    """Compute gap as function of adaptation steps K."""

    print("\nComputing gap vs K...")
    T = config.get('horizon', 1.0)
    inner_lr = config.get('inner_lr', 0.01)

    results = {'K': [], 'gap': [], 'meta_fid': []}

    for K in tqdm(K_values, desc="K values"):
        meta_fidelities = [] 

        for task in test_tasks:
            # Meta with adaptation (NOW WITH PROPER GRADIENTS!)
            F_meta = evaluate_fidelity(
                meta_policy, task, quantum_system, target_state, T, device,
                adapt=True, K=K, inner_lr=inner_lr, env=env
            )
            meta_fidelities.append(F_meta)


        mean_meta = np.mean(meta_fidelities)  
        gap = mean_meta  

        results['K'].append(K)
        results['gap'].append(gap)
        results['meta_fid'].append(mean_meta)


      #  print(f"  K={K:2d}: Gap = {gap:.4f}, Meta = {mean_meta:.4f}, Robust = {mean_robust:.4f}")

    return results


def compute_gap_vs_variance(
    meta_policy: torch.nn.Module, 
    base_config: dict,
    variance_multipliers: list,
    quantum_system: dict,
    target_state: np.ndarray,
    device: torch.device,
    n_tasks_per_variance: int = 50,
    env=None  # NEW: Pass environment for differentiable adaptation
) -> dict:
    """Compute gap as function of task distribution variance."""

    print("\nComputing gap vs variance...")
    K = base_config.get('inner_steps', 5)
    T = base_config.get('horizon', 1.0)
    inner_lr = base_config.get('inner_lr', 0.01)

    results = {'variance': [], 'gap': [], 'meta_fid': []}

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
            # Now with proper gradient flow!
            F_meta = evaluate_fidelity(
                meta_policy, task, quantum_system, target_state, T, device,
                adapt=True, K=K, inner_lr=inner_lr, env=env
            )
            meta_fidelities.append(F_meta)

            F_robust = evaluate_fidelity(
                robust_policy, task, quantum_system, target_state, T, device,
                adapt=False, env=env
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


def plot_results(gap_vs_K: dict, constants: GapConstants, save_dir: Path):
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
        
    ##Number of stpes with optimality gap --> check for agreement 
    ax.set_xlabel('Adaptation Steps K', fontsize=12)
    ax.set_ylabel('Optimality Gap', fontsize=12)
    ax.set_title('Gap vs Adaptation Steps', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # # Plot 2: Gap vs Variance
    # ax = axes[1]
    # variances = np.array(gap_vs_var['variance'])
    # gaps_var = np.array(gap_vs_var['gap'])
    # #calculate gap over variance 
    # ax.plot(variances, gaps_var, 's-', linewidth=2, markersize=8, label='Empirical Gap')
    
    # # Linear fit (theory predicts Gap ∝ σ²)
    # if len(variances) > 1:
    #     coeffs = np.polyfit(variances, gaps_var, 1)
    #     var_fit = np.linspace(min(variances), max(variances), 100)
    #     gap_fit = np.polyval(coeffs, var_fit)
    #     ax.plot(var_fit, gap_fit, '--', linewidth=2, 
    #             label=f'Linear Fit (slope={coeffs[0]:.3f})', alpha=0.7)
    
    # ax.set_xlabel('Task Variance σ²_θ', fontsize=12)
    # ax.set_ylabel('Optimality Gap', fontsize=12)
    # ax.set_title('Gap vs Task Variance', fontsize=14, fontweight='bold')
    # ax.legend(fontsize=10)
    # ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    save_path = save_dir / 'optimality_gap_analysis.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {save_path}")
    
    plt.show()


#def main(args):
def main():
    """Main evaluation script."""

    # Load config
    config_path = '../../configs/experiment_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("=" * 70)
    print("Optimality Gap Evaluation (WITH DIFFERENTIABLE ADAPTATION!)")
    print("=" * 70)
   # print(f"Config: {args.config}\n")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Create quantum system
    print("Setting up quantum system...")
    #quantum_system = create_quantum_system(config)
    quantum_system = create_quantum_system()  
    

    # Target state
    target_gate_name = config.get('target_gate', 'pauli_x')
    if target_gate_name == 'hadamard':
        U_target = TargetGates.hadamard()
    else:
        U_target = TargetGates.pauli_x()

    ket_0 = np.array([1, 0], dtype=complex)
    target_state = np.outer(U_target @ ket_0, (U_target @ ket_0).conj())

    # NEW: Create QuantumEnvironment for differentiable adaptation
    print("\nCreating quantum environment for differentiable adaptation...")
    from metaqctrl.theory.quantum_environment import create_quantum_environment
    env = create_quantum_environment(config, target_state)
    print(f"  Environment created: {env.get_cache_stats()}")
    print("   Gradient flow through quantum simulation enabled!")

    # Load models
    print("\nLoading trained models...")
    meta_path = "../train_scripts/checkpoints/maml_best_pauli_x_best_policy.pt"
    meta_policy= load_models(
       meta_path, config, device)

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

    # Compute gap vs K (NOW WITH PROPER GRADIENTS!)
    K_values = config.get('gap_K_values', [1, 2, 3,4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18 , 19, 20])
    gap_vs_K_results = compute_gap_vs_K(
        meta_policy, test_tasks[:50], K_values,
        quantum_system, target_state, config, device, env=env
    )

    # # Compute gap vs variance (NOW WITH PROPER GRADIENTS!)
    # variance_multipliers = [0.5, 0.75, 1.0, 1.25, 1.5]
    # gap_vs_var_results = compute_gap_vs_variance(
    #     meta_policy, robust_policy, config, variance_multipliers,
    #     quantum_system, target_state, device, n_tasks_per_variance=30, env=env
    # )

    # Plot
    save_dir = Path('results') 
    #save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    plot_results(gap_vs_K_results, constants, save_dir)

    # Save results
    import json
    results = {
        'gap_vs_K': gap_vs_K_results} 
   #     'gap_vs_variance': gap_vs_var_results
  #  }

    results_path = save_dir / 'gap_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    print(f"\nFinal cache stats: {env.get_cache_stats()}")
    print("\n" + "=" * 70)
    print("Evaluation complete!")
    print("=" * 70)


if __name__ == "__main__":  
    main()
