"""
Physics Constants Computation

Implements actual computation of all theoretical constants:
- Spectral gap Δ
- Filter constant C_filter
- PL constant μ
- Control-relevant variance σ²_S
"""

import numpy as np
from scipy.optimize import curve_fit
from typing import List, Dict
import torch
from tqdm import tqdm


def compute_spectral_gap(env, task_params) -> float:
    """ 
    Good. 
    Compute Lindblad spectral gap Δ(θ).
    
    Δ = gap between eigenvalue 0 and next eigenvalue of L_θ.
    
    Args:
        env: QuantumEnvironment
        task_params: NoiseParameters
        
    Returns:
        gap: Spectral gap (positive float)
    """
    L_ops = env.get_lindblad_operators(task_params)
    H0 = env.H0
    d = H0.shape[0]
    
    # Build Lindblad superoperator (d² × d² matrix)
    d2 = d * d
    L_super = np.zeros((d2, d2), dtype=complex)
    
    I = np.eye(d)
    
    # Hamiltonian part: -i[H, ρ]
    # vec(-i[H,ρ]) = -i(I⊗H - H^T⊗I)vec(ρ)
    # Constructs Lindblad superoperator 
    L_super += -1j * (np.kron(I, H0) - np.kron(H0.T, I))
    
    # Dissipation part: Σⱼ(Lⱼρ L†ⱼ - ½{L†ⱼLⱼ, ρ})
    ##Construct total superoperator 
    for L in L_ops:
        # Lⱼ ρ L†ⱼ → (L̄ⱼ ⊗ Lⱼ) vec(ρ)
        L_super += np.kron(L.conj(), L)
        
        # -½ L†ⱼLⱼ ρ → -½(I ⊗ L†ⱼLⱼ) vec(ρ)
        anti_comm = L.conj().T @ L
        L_super -= 0.5 * np.kron(I, anti_comm)
        
        # -½ ρ L†ⱼLⱼ → -½(L†ⱼLⱼ^T ⊗ I) vec(ρ)
        L_super -= 0.5 * np.kron(anti_comm.T, I)
    
    # Compute eigenvalues
    eigenvals = np.linalg.eigvals(L_super)
    eigenvals_real = np.real(eigenvals)
    
    # Sort in descending order (largest first)
    eigenvals_sorted = np.sort(eigenvals_real)[::-1]
    
    # Gap = λ₀ - λ₁ (should be 0 - (negative value) = positive)
    gap = eigenvals_sorted[0] - eigenvals_sorted[1]
    
    # Ensure positive (numerical errors may give small negative)
    gap = abs(gap)
    
    return gap


def estimate_spectral_gap_distribution(env, tasks: List) -> Dict:
    """
    Good. 
    Estimate spectral gap statistics over task distribution.
    
    Args:
        env: QuantumEnvironment
        tasks: List of NoiseParameters
        
    Returns:
        stats: Dictionary with gap statistics
    """
    gaps = []
    
    print("Computing spectral gaps...")
    for task in tqdm(tasks):
        gap = compute_spectral_gap(env, task)
        gaps.append(gap)
    
    gaps = np.array(gaps)
    
    return {
        'gaps': gaps,
        'Delta_min': np.min(gaps),
        'Delta_max': np.max(gaps),
        'Delta_mean': np.mean(gaps),
        'Delta_std': np.std(gaps)
    }


def compute_control_response_operator(env, omega_samples: np.ndarray = None) -> np.ndarray:
    """ Might be worth checking. 
    Compute control response operator M(ω).
    
    M(ω) ≈ G(ω)^{-1} N(ω) where:
    - G(ω) is Gram controllability matrix
    - N(ω) is noise-control coupling
    
    For now, use simplified version based on control Hamiltonian norms.
    
    Args:
        env: QuantumEnvironment
        omega_samples: Frequency samples (optional)
        
    Returns:
        M: Control response operator (n_controls × 1)
    """
    if omega_samples is None:
        # Default: sample up to control bandwidth
        n_seg = 20  # From typical config
        omega_max = n_seg / env.T
        omega_samples = np.linspace(0, omega_max, 50)
    
    H_controls = env.H_controls
    n_controls = len(H_controls)
    
    M_omega_list = []
    
    for omega in omega_samples:
        # Simplified Gram matrix: G_kl = <H_k, H_l>
        G = np.zeros((n_controls, n_controls), dtype=complex)
        for k in range(n_controls):
            for l in range(n_controls):
                # Inner product: tr(H†_k H_l)
                G[k, l] = np.trace(H_controls[k].conj().T @ H_controls[l])
        
        # Make Hermitian
        G = (G + G.conj().T) / 2
        
        # Noise coupling (simplified: uniform)
        N = np.ones(n_controls, dtype=complex)
        
        # Response: M = G^{-1} N
        # Add regularization for stability
        G_reg = G + 1e-6 * np.eye(n_controls)
        M_omega = np.linalg.solve(G_reg, N)
        
        M_omega_list.append(M_omega)
    
    # Average over frequencies
    M_avg = np.mean(M_omega_list, axis=0)
    
    return M_avg.reshape(-1, 1)


def estimate_filter_constant(env) -> float:
    """ Worth checking again. 
    Estimate filter separation constant C_filter.
    
    C_filter = σ²_min(M) / (2π)
    
    Args:
        env: QuantumEnvironment
        
    Returns:
        C_filter: Filter constant
    """
    M = compute_control_response_operator(env)
    
    # Singular values
    U, sigma, Vt = np.linalg.svd(M)
    sigma_min = sigma.min()
    
    C_filter = sigma_min**2 / (2 * np.pi)
    
    return C_filter


def estimate_pl_constant(env, task_params, num_samples: int = 5) -> float:
    """
    Estimate PL constant μ for a task using random control perturbations.

    Uses the theoretical formula: μ ≈ Δ / (d² M² T)
    where Δ is the spectral gap, d is the dimension, M is control strength, T is time.

    Args:
        env: QuantumEnvironment
        task_params: Task parameters (NoiseParameters)
        num_samples: Number of random samples for estimation

    Returns:
        mu: Estimated PL constant
    """
    # Compute spectral gap
    Delta = compute_spectral_gap(env, task_params)

    # System parameters
    d = env.d
    T = env.T

    # Estimate control strength M
    M = np.max([np.linalg.norm(H) for H in env.H_controls])

    # Theoretical PL constant (this is a heuristic approximation)
    mu = Delta / (d**2 * M**2 * T)

    return mu


def estimate_PL_constant_from_convergence(
    env,
    policy: torch.nn.Module,
    task_params,
    K_values: List[int] = [1, 3, 5, 10, 15],
    eta: float = 0.01,
    device: torch.device = torch.device('cpu')
) -> Dict:
    """ Good. 
    Estimate μ (PL constant) by measuring convergence rate empirically.
    
    Fits exponential: L(K) = L* + (L₀ - L*) exp(-μηK)
    
    Args:
        env: QuantumEnvironment
        policy: Policy network
        task_params: Single task to test on
        K_values: List of adaptation steps to try
        eta: Learning rate
        device: torch device
        
    Returns:
        results: Dictionary with μ estimate and fit quality
    """
    from copy import deepcopy
    
    losses = []

    print(f"Estimating μ for task (α={task_params.alpha:.2f}, A={task_params.A:.3f})...")
    
    ##Iterates over set of adaptation steps 
    for K in tqdm(K_values):
        # Clone policy
        ## Existing policy 
        adapted_policy = deepcopy(policy)
        adapted_policy.train() #Train policy 
        
        #Optimzer --> define SGD optimzer with learning rate 
        optimizer = torch.optim.SGD(adapted_policy.parameters(), lr=eta)
        
        # K gradient steps 
        for _ in range(K):
            ##Zero gradient initialization 
            optimizer.zero_grad()
            #Define loss function 
            loss = env.compute_loss(adapted_policy, task_params, device)
            #Perform a backward step 
            loss.backward()
            optimizer.step()
        
        # Final loss
        with torch.no_grad():
            #Recover final loss value 
            final_loss = env.compute_loss(adapted_policy, task_params, device).item()
        #Add loss function to list 
        losses.append(final_loss)
    #Concvert loss to array 
    losses = np.array(losses)
    
    # Fit exponential: L(K) = L_star + (L_0 - L_star) * exp(-rate * K)
    #Fitting procedure 
    def exp_decay(K, L_star, L_0, rate):
        return L_star + (L_0 - L_star) * np.exp(-rate * K)
    
    try:
        # Initial guess
        L_star_init = losses[-1]  # Assume converges
        L_0_init = losses[0]
        rate_init = 0.1
        #Fit curve 
        popt, pcov = curve_fit(
            exp_decay,
            K_values,
            losses,
            p0=[L_star_init, L_0_init, rate_init],
            maxfev=10000
        )
        ##Fit parameters 
        L_star_fit, L_0_fit, rate_fit = popt
        
        # μ = rate / η
        #Empirical mu , comparing learning rate to fitted rate 
        mu_empirical = rate_fit / eta
        
        # Quality metrics
        ## Compare loss functions 
        residuals = losses - exp_decay(np.array(K_values), *popt)
        ss_res = np.sum(residuals**2) 
        ss_tot = np.sum((losses - np.mean(losses))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'mu': mu_empirical,
            'rate': rate_fit,
            'L_star': L_star_fit,
            'L_0': L_0_fit,
            'r_squared': r_squared,
            'losses': losses,
            'K_values': K_values,
            'fit_params': popt
        }
    ##Run time erorr 
    except RuntimeError as e:
        print(f"Curve fitting failed: {e}")
        # Fallback: estimate from first and last losses
        if len(losses) > 1 and losses[0] > losses[-1]:
            # Assume exponential decay: loss(K) ≈ L_star + (L_0 - L_star) * exp(-rate * K)
            # Using first and last points
            L_min = min(losses)
            delta_loss = losses[0] - losses[-1]
            if delta_loss > 1e-10:
                # Rate from exponential fit between two points
                rate_approx = np.log((losses[0] - L_min + 1e-10) / (losses[-1] - L_min + 1e-10)) / K_values[-1]
                mu_approx = max(rate_approx / eta, 0.001)  # Ensure positive
            else:
                mu_approx = 0.01  # Default fallback
        else:
            mu_approx = 0.01  # Default fallback

        return {
            'mu': mu_approx,
            'rate': mu_approx * eta,
            'r_squared': 0.0,
            'losses': losses,
            'K_values': K_values,
            'fit_failed': True
        }


def compute_control_relevant_variance(
    tasks_or_env,
    omega_control_or_tasks = None,
    control_susceptibility = None
) -> float:
    """
    Compute control-relevant variance σ²_S.

    σ²_S = Var[∫_{Ω_control} S(ω; θ) χ(ω) dω]

    Supports two call signatures for backward compatibility:
    1. compute_control_relevant_variance(env, tasks, omega_control_band=None)
    2. compute_control_relevant_variance(tasks, omega_control, control_susceptibility)

    Args:
        tasks_or_env: Either QuantumEnvironment or list of NoiseParameters
        omega_control_or_tasks: Either list of tasks or omega control frequencies
        control_susceptibility: Control susceptibility matrix (optional)

    Returns:
        sigma2_S: Control-relevant variance (float)
    """
    # Detect call signature
    if hasattr(tasks_or_env, 'psd_to_lindblad'):
        # New signature: (env, tasks, omega_control_band)
        env = tasks_or_env
        tasks = omega_control_or_tasks
        omega_control_band = control_susceptibility
        return _compute_control_relevant_variance_with_env(env, tasks, omega_control_band)
    else:
        # Old signature: (tasks, omega_control, control_susceptibility)
        tasks = tasks_or_env
        omega_control = omega_control_or_tasks
        return _compute_control_relevant_variance_simple(tasks, omega_control, control_susceptibility)


def _compute_control_relevant_variance_simple(
    tasks: List,
    omega_control: np.ndarray,
    control_susceptibility: np.ndarray
) -> float:
    """
    Simple variance computation for backward compatibility.

    Args:
        tasks: List of NoiseParameters
        omega_control: Control frequency array
        control_susceptibility: Susceptibility matrix

    Returns:
        sigma2_S: Variance
    """
    from metaqctrl.quantum.noise_models import NoisePSDModel

    psd_model = NoisePSDModel(model_type='one_over_f')

    # Compute noise power for each task
    noise_powers = []
    for task in tasks:
        # Simple average over omega_control band
        omega_grid = np.linspace(0, np.max(omega_control), 50)
        S_omega = psd_model.psd(omega_grid, task)
        N_control = np.mean(S_omega)  # Simplified integration
        noise_powers.append(N_control)

    noise_powers = np.array(noise_powers)
    sigma2_S = np.var(noise_powers)

    return sigma2_S


def _compute_control_relevant_variance_with_env(
    env,
    tasks: List,
    omega_control_band: tuple = None
) -> float:
    """ Worth checking, but looks good.
    Compute control-relevant variance σ²_S with full environment context.

    σ²_S = Var[∫_{Ω_control} S(ω; θ) χ(ω) dω]

    Args:
        env: QuantumEnvironment
        tasks: List of NoiseParameters
        omega_control_band: (omega_min, omega_max) or None for auto

    Returns:
        sigma2_S: Control-relevant variance
    """
    # Determine control bandwidth
    if omega_control_band is None:
        n_seg = 20  # From config
        omega_max = n_seg / env.T
        omega_control_band = (0, omega_max)
    
    omega_min, omega_max = omega_control_band
    omega_grid = np.linspace(omega_min, omega_max, 100)
    
    # Susceptibility (simplified: uniform in band)
    chi = np.ones_like(omega_grid)
    
    # Compute weighted noise power for each task
    noise_powers = []
    psd_model = env.psd_to_lindblad.psd_model
    
    print("Computing control-relevant variance...")
    for task in tqdm(tasks):
        S_omega = psd_model.psd(omega_grid, task)
        
        # Weighted integral
        N_control = np.trapz(S_omega * chi, omega_grid)
        noise_powers.append(N_control)
    
    noise_powers = np.array(noise_powers)
    
    # Variance
    sigma_S_sq = np.var(noise_powers)
    
    # Also compute out-of-band for comparison
    omega_out_grid = np.linspace(omega_max, 10 * omega_max, 100)
    noise_powers_out = []
    
    for task in tasks:
        S_omega_out = psd_model.psd(omega_out_grid, task)
        N_out = np.trapz(S_omega_out, omega_out_grid)
        noise_powers_out.append(N_out)
    
    noise_powers_out = np.array(noise_powers_out)
    sigma_out_sq = np.var(noise_powers_out)

    # Return just the variance for backward compatibility
    return sigma_S_sq


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
