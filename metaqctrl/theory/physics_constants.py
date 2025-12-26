"""
Physics Constants Computation

"""

import numpy as np
from scipy.optimize import curve_fit
from typing import List, Dict
import torch
from tqdm import tqdm


def estimate_all_constants(
    env,
    policy_or_task_dist = None,
    tasks_or_n_tasks = None,
    device: torch.device = torch.device('cpu'),
    n_samples_gap: int = 10,
    n_samples_mu: int = 3,
    task_distribution = None,
    n_tasks: int = None
) -> Dict:
    """
    Estimate all theoretical constants.

    Supports two call signatures for backward compatibility:
    1. estimate_all_constants(env, policy, tasks, device, ...)
    2. estimate_all_constants(env, task_distribution=dist, n_tasks=N)

    Args:
        env: QuantumEnvironment
        policy_or_task_dist: Policy network or task distribution
        tasks_or_n_tasks: List of tasks or number of tasks
        device: torch device
        n_samples_gap: Number of tasks for gap estimation
        n_samples_mu: Number of tasks for μ estimation
        task_distribution: TaskDistribution (for backward compat)
        n_tasks: Number of tasks to sample (for backward compat)

    Returns:
        constants: Dictionary with all estimated constants
    """
    # Handle backward compatibility signatures
    if task_distribution is not None and n_tasks is not None:
        # Old signature: estimate_all_constants(env, task_distribution=dist, n_tasks=N)
        tasks = task_distribution.sample(n_tasks)
        policy = None
    elif policy_or_task_dist is not None and hasattr(policy_or_task_dist, 'sample'):
        # task_distribution passed as first arg
        tasks = policy_or_task_dist.sample(tasks_or_n_tasks if tasks_or_n_tasks else 50)
        policy = None
    else:
        # New signature: estimate_all_constants(env, policy, tasks, ...)
        policy = policy_or_task_dist
        tasks = tasks_or_n_tasks if tasks_or_n_tasks is not None else []
    print("="*60)
    print("ESTIMATING ALL THEORETICAL CONSTANTS")
    print("="*60)
    
    # 1. Spectral gap
    print("\n1. Spectral Gap Δ...")
    gap_stats = estimate_spectral_gap_distribution(env, tasks[:n_samples_gap])
    Delta_min = gap_stats['Delta_min']
    print(f"   Δ_min = {Delta_min:.6f}")
    print(f"   Δ_mean = {gap_stats['Delta_mean']:.6f}")
    print(f"   Δ_std = {gap_stats['Delta_std']:.6f}")
    
    # 2. Filter constant
    print("\n2. Filter Constant C_filter...")
    C_filter = estimate_filter_constant(env)
    print(f"   C_filter = {C_filter:.6f}")
    
    # 3. PL constant μ
    print("\n3. PL Constant μ...")
    mu_results = []
    if policy is not None:
        # Use convergence-based estimation with policy
        for task in tasks[:n_samples_mu]:
            result = estimate_PL_constant_from_convergence(env, policy, task, device=device)
            mu_results.append(result['mu'])
            print(f"   Task: μ = {result['mu']:.6f}, R² = {result['r_squared']:.3f}")
    else:
        # Use theoretical formula
        for task in tasks[:n_samples_mu]:
            mu = estimate_pl_constant(env, task)
            mu_results.append(mu)
            print(f"   Task: μ = {mu:.6f}")

    mu_mean = np.mean(mu_results)
    mu_min = np.min(mu_results)
    print(f"   μ_mean = {mu_mean:.6f}")
    print(f"   μ_min = {mu_min:.6f}")

    # 4. Control-relevant variance
    print("\n4. Control-Relevant Variance σ²_S...")
    sigma_S_sq = compute_control_relevant_variance(env, tasks)
    print(f"   σ²_S = {sigma_S_sq:.8f}")
    
    # 5. System parameters
    M = max(np.linalg.norm(H, ord=2) for H in env.H_controls)
    T = env.T
    
    print("\n5. System Parameters...")
    print(f"   M (control bound) = {M:.6f}")
    print(f"   T (horizon) = {T:.6f}")
    
    # 6. Derived constants
    c_quantum = C_filter / (M**2 * T**2)
    mu_theory = Delta_min / (4 * M**2 * T**2)  # From Lemma 3
    
    print("\n6. Derived Constants...")
    print(f"   c_quantum = C_filter/(M²T²) = {c_quantum:.8f}")
    print(f"   μ_theory = Δ/(4M²T²) = {mu_theory:.8f}")
    print(f"   μ_empirical = {mu_mean:.8f}")
    print(f"   Ratio μ_emp/μ_theory = {mu_mean/mu_theory:.3f}")
    
    print("\n" + "="*60)

    return {
        'Delta_min': Delta_min,
        'Delta_mean': gap_stats['Delta_mean'],
        'Delta_std': gap_stats['Delta_std'],
        'C_filter': C_filter,
        'mu_min': mu_min,
        'mu_mean': mu_mean,
        'mu_empirical': mu_mean,
        'mu_theory': mu_theory,
        'mu_results': mu_results,
        'sigma2_S': sigma_S_sq,  # Backward compat key
        'sigma_S_sq': sigma_S_sq,
        'M': M,
        'T': T,
        'c_quantum': c_quantum,
        'gap_stats': gap_stats
    }
