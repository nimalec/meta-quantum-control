"""
Optimality Gap Theory and Computation

Implements theoretical bounds from Section 4 (Phase 4):
- Gap(P, K) = E[F(π_meta, θ)] - E[F(π_rob, θ)]
- Constants: C_sep, μ, L
- Empirical verification
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Callable, Optional
from scipy.optimize import minimize
from dataclasses import dataclass


@dataclass
class GapConstants:
    """Constants in optimality gap bound."""
    C_sep: float  # Task-optimal policy separation
    mu: float     # Strong convexity / PL constant
    L: float      # Lipschitz constant (fidelity vs task)
    L_F: float    # Lipschitz constant (fidelity vs policy)
    C_K: float    # Inner loop Lipschitz constant
    
    def gap_lower_bound(self, sigma_sq: float, K: int, eta: float) -> float:
        """ Good. 
        Compute theoretical lower bound on gap:
        Gap(P, K) ≥ c_gap · σ²_θ · (1 - e^(-μηK))
        """
        c_gap = self.C_sep * self.L_F * self.L ** 2
        return c_gap * sigma_sq * (1 - np.exp(-self.mu * eta * K))


class OptimalityGapComputer:
    """ 
    Compute and analyze optimality gaps between meta-learning and robust control.
    """
    
    def __init__(
        self,
        quantum_system: Callable,
        fidelity_fn: Callable,
        device: torch.device = torch.device('cpu')
    ):
        """ Good. 
        Args:
            quantum_system: Function that simulates quantum dynamics
            fidelity_fn: Function that computes fidelity
            device: torch device
        """
        self.quantum_system = quantum_system
        self.fidelity_fn = fidelity_fn
        self.device = device
    
    def compute_gap(
        self,
        meta_policy: torch.nn.Module,
        robust_policy: torch.nn.Module,
        task_distribution: List,
        n_samples: int = 100,
        K: int = 5,
        inner_lr: float = 0.01
    ) -> Dict[str, float]:
        """ Good. 
        Compute empirical optimality gap.
        Compares fidelity for robust policy against adapted policy. 
        
        Gap = E_θ[F(AdaptK(π_meta; θ), θ)] - E_θ[F(π_rob, θ)]
        
        Args:
            meta_policy: Meta-learned initialization
            robust_policy: Robust baseline policy
            task_distribution: List of tasks to sample from
            n_samples: Number of tasks to sample
            K: Number of inner adaptation steps
            inner_lr: Inner loop learning rate
            
        Returns:
            results: Dictionary with gap and related metrics
        """
        meta_fidelities = []
        robust_fidelities = []
        
        for task_params in task_distribution[:n_samples]:
            # Meta-policy: adapt then evaluate
            ## Picks up adapted policy after K steps 
            adapted_policy = self._adapt_policy(
                meta_policy, task_params, K=K, lr=inner_lr
            )
            ## Fidelity for this policy 
            F_meta = self._evaluate_policy(adapted_policy, task_params)
            meta_fidelities.append(F_meta)
            
            # Robust policy: evaluate directly (no adaptation)
            ## Do the same for the robust policy 
            F_robust = self._evaluate_policy(robust_policy, task_params)
            robust_fidelities.append(F_robust)
        
        # Compute gap
        ## gets the gap between the two 
        mean_F_meta = np.mean(meta_fidelities)
        mean_F_robust = np.mean(robust_fidelities)
        gap = mean_F_meta - mean_F_robust
        
        results = {
            'gap': gap,
            'meta_fidelity_mean': mean_F_meta,
            'meta_fidelity_std': np.std(meta_fidelities),
            'robust_fidelity_mean': mean_F_robust,
            'robust_fidelity_std': np.std(robust_fidelities),
            'meta_fidelities': meta_fidelities,
            'robust_fidelities': robust_fidelities
        }
        
        return results
    
    def _adapt_policy(
        self,
        policy: torch.nn.Module,
        task_params: Dict,
        K: int,
        lr: float
    ) -> torch.nn.Module:
        """FIXED: Now uses differentiable simulation for proper gradient flow.
        Perform K-step gradient adaptation on task."""
        from copy import deepcopy

        adapted_policy = deepcopy(policy)
        adapted_policy.train()
        optimizer = torch.optim.SGD(adapted_policy.parameters(), lr=lr)

        # Check if quantum_system is actually a QuantumEnvironment
        if hasattr(self.quantum_system, 'compute_loss_differentiable'):
            # NEW PATH: Use differentiable environment
            for _ in range(K):
                optimizer.zero_grad()

                # FIXED: Fully differentiable loss through quantum simulation
                loss = self.quantum_system.compute_loss_differentiable(
                    adapted_policy, task_params, self.device
                )

                loss.backward()
                optimizer.step()
        else:
            # OLD PATH: Fallback to non-differentiable (for backwards compatibility)
            # WARNING: This won't work properly for gradient-based adaptation!
            task_features = self._task_to_features(task_params)

            for _ in range(K):
                optimizer.zero_grad()

                controls = adapted_policy(task_features)
                fidelity = self._evaluate_controls(controls, task_params)
                loss = 1.0 - fidelity

                # This won't backprop through simulation properly
                loss.backward()
                optimizer.step()

        return adapted_policy
    
    def _evaluate_policy(
        self,
        policy: torch.nn.Module,
        task_params: Dict
    ) -> float:
        """Good. Calculates fidelity for new policy. 
        Evaluate policy on task and return fidelity."""
        policy.eval()
        
        with torch.no_grad():
            task_features = self._task_to_features(task_params)
            controls = policy(task_features)
            fidelity = self._evaluate_controls(controls, task_params)
        
        return fidelity.item()
    
    def _evaluate_controls(
        self,
        controls: torch.Tensor,
        task_params: Dict
    ) -> torch.Tensor:
        """Simulate quantum system and compute fidelity."""
        # Convert to numpy for simulation
        controls_np = controls.detach().cpu().numpy()
        
        # Simulate dynamics
        rho_final = self.quantum_system(controls_np, task_params)
        
        # Compute fidelity
        fidelity = self.fidelity_fn(rho_final)
        
        return torch.tensor(fidelity, device=self.device)
    
    def _task_to_features(self, task_params: Dict) -> torch.Tensor:
        """Good. 
        Convert task parameters to feature tensor."""
        # Assuming task_params has 'alpha', 'A', 'omega_c'
        features = torch.tensor([
            task_params.alpha,
            task_params.A,
            task_params.omega_c
        ], dtype=torch.float32, device=self.device)
        return features
    
    def estimate_constants(
        self,
        policy: torch.nn.Module,
        task_distribution: List,
        n_samples: int = 50
    ) -> GapConstants:
        """Good, except for inner loop Lipschitz.  
        Estimate theoretical constants from data.
        
        Returns:
            constants: GapConstants object with estimates
        """
        print("Estimating theoretical constants...")
        
        # C_sep: Task-optimal policy separation
        C_sep = self._estimate_c_sep(policy, task_distribution, n_samples)
        print(f"  C_sep = {C_sep:.4f}")
        
        # μ: Curvature constant
        mu = self._estimate_mu(policy, task_distribution, n_samples)
        print(f"  μ = {mu:.4f}")
        
        # L: Lipschitz constant (fidelity vs task)
        L = self._estimate_lipschitz_task(policy, task_distribution, n_samples)
        print(f"  L = {L:.4f}")
        
        # L_F: Lipschitz constant (fidelity vs policy params)
        L_F = self._estimate_lipschitz_policy(policy, task_distribution[0])
        print(f"  L_F = {L_F:.4f}")
        
        # C_K: Inner loop Lipschitz
        C_K = 1.0  # Placeholder - depends on inner loop algorithm
        print(f"  C_K = {C_K:.4f}")
        
        return GapConstants(C_sep, mu, L, L_F, C_K)
    
    def _estimate_c_sep(
        self,
        policy: torch.nn.Module,
        tasks: List,
        n_samples: int
    ) -> float:
        """ Optimal policieses (u*) between two tasks. 
        Estimate C_sep: average separation of task-optimal policies.
        
        C_sep = E[||π*_θ - π*_θ'||²]^(1/2)
        """
        # Sample task pairs
        task_pairs = np.random.choice(len(tasks), size=(n_samples, 2), replace=True)
        
        separations = []
        for i, j in task_pairs:
            if i == j:
                continue
            
            # Find task-optimal policies (approximate via many gradient steps)
            pi_star_i = self._adapt_policy(policy, tasks[i], K=50, lr=0.01)
            pi_star_j = self._adapt_policy(policy, tasks[j], K=50, lr=0.01)
            
            # Compute parameter distance
            dist = 0.0
            for p1, p2 in zip(pi_star_i.parameters(), pi_star_j.parameters()):
                dist += torch.sum((p1 - p2) ** 2).item()
            
            separations.append(np.sqrt(dist))
        
        return np.mean(separations)
    
    def _estimate_mu(
        self,
        policy: torch.nn.Module,
        tasks: List,
        n_samples: int
    ) -> float:
        """
        Estimate μ: strong convexity / PL constant.
        
        Via PL condition: ||∇L||² ≥ 2μ(L - L*)
        Estimate from gradient norms near optima.
        """
        mu_estimates = []
        
        for task in tasks[:n_samples]:
            # Adapt to near-optimal 
            adapted = self._adapt_policy(policy, task, K=20, lr=0.01)
            
            # Compute gradient norm and loss 
            task_features = self._task_to_features(task)
            ##Generate new controls for an adapted poliicy. 
            controls = adapted(task_features)
            #Evaluate conttrol. 
            fidelity = self._evaluate_controls(controls, task)
            loss = 1.0 - fidelity
            
            # Compute gradient norm
            loss.backward()
            ##Recover the gradient itself ==> calculate the norm squared. 
            grad_norm_sq = sum(
                torch.sum(p.grad ** 2).item()
                for p in adapted.parameters() if p.grad is not None
            )
            
            # Estimate μ from PL condition (assume L* ≈ 0)
            ## Compute loss. Divide norm squared by loss itself. 
            if loss.item() > 1e-6:
                mu_est = grad_norm_sq / (2 * loss.item())
                mu_estimates.append(mu_est)
        
        return np.median(mu_estimates) if mu_estimates else 0.1
    
    def _estimate_lipschitz_task(
        self,
        policy: torch.nn.Module,
        tasks: List,
        n_samples: int
    ) -> float:
        """FIXED: Now normalizes task parameters before computing distance.
        Estimate L: Lipschitz constant of fidelity w.r.t. task parameters.

        L ≈ max |F(π, θ) - F(π, θ')| / ||θ_normalized - θ'_normalized||

        The normalization accounts for different scales of α, A, and ω_c.
        """
        task_pairs = np.random.choice(len(tasks), size=(n_samples, 2), replace=False)

        # Compute normalization constants from task distribution
        all_alphas = [t.alpha for t in tasks]
        all_As = [t.A for t in tasks]
        all_omegas = [t.omega_c for t in tasks]

        # Use ranges for normalization (robust to outliers)
        alpha_range = max(all_alphas) - min(all_alphas) + 1e-6
        A_range = max(all_As) - min(all_As) + 1e-6
        omega_range = max(all_omegas) - min(all_omegas) + 1e-6

        lipschitz_ratios = []
        for i, j in task_pairs:
            task_i, task_j = tasks[i], tasks[j]

            # Evaluate fidelity on both tasks
            F_i = self._evaluate_policy(policy, task_i)
            F_j = self._evaluate_policy(policy, task_j)

            # FIXED: Normalize task parameters before computing distance
            theta_i_norm = np.array([
                task_i.alpha / alpha_range,
                task_i.A / A_range,
                task_i.omega_c / omega_range
            ])
            theta_j_norm = np.array([
                task_j.alpha / alpha_range,
                task_j.A / A_range,
                task_j.omega_c / omega_range
            ])

            # Compute normalized distance
            task_dist = np.linalg.norm(theta_i_norm - theta_j_norm)

            if task_dist > 1e-6:
                ratio = abs(F_i - F_j) / task_dist
                lipschitz_ratios.append(ratio)

        return np.max(lipschitz_ratios) if lipschitz_ratios else 1.0
    
    def _estimate_lipschitz_policy(
        self,
        policy: torch.nn.Module,
        task: Dict
    ) -> float:
        """ Good. 
        Estimate L_F: Lipschitz constant of fidelity w.r.t. policy parameters. (wrt u) 
        
        Via gradient: L_F ≈ ||∇_π F||
        """
        policy.train()
        
        task_features = self._task_to_features(task)
        controls = policy(task_features)
        fidelity = self._evaluate_controls(controls, task)
        
        # Compute gradient w.r.t. policy parameters
        fidelity.backward()
        
        grad_norm = np.sqrt(sum(
            torch.sum(p.grad ** 2).item()
            for p in policy.parameters() if p.grad is not None
        ))
        
        policy.zero_grad()
        
        return grad_norm


def plot_gap_vs_control_relevant_variance(
    gap_computer: OptimalityGapComputer,
    meta_policy: torch.nn.Module,
    robust_policy: torch.nn.Module,
    env,
    variance_range: np.ndarray,
    K: int = 5,
    n_samples: int = 50,
    base_mean: np.ndarray = None,
    eta: float = 0.01,
    save_path: Optional[str] = None
):
    """ NEW: Plot gap vs CONTROL-RELEVANT variance σ²_S (not parameter variance σ²_θ).

    This uses the physically correct variance metric that accounts for how noise
    affects the quantum dynamics through the control bandwidth.

    Args:
        gap_computer: OptimalityGapComputer instance
        meta_policy: Meta-learned policy
        robust_policy: Robust baseline policy
        env: QuantumEnvironment instance (needed for PSD integration)
        variance_range: Array of variance values to test
        K: Number of adaptation steps
        n_samples: Number of tasks to sample per variance level
        base_mean: Mean task parameters (default: [1.0, 0.1, 5.0])
        eta: Learning rate for adaptation
        save_path: Path to save plot
    """
    import matplotlib.pyplot as plt
    from metaqctrl.quantum.noise_adapter import TaskDistribution, NoiseParameters
    from metaqctrl.theory.physics_constants import compute_control_relevant_variance

    if base_mean is None:
        base_mean = np.array([1.0, 0.1, 5.0])  # (alpha, A, omega_c)

    gaps = []
    param_variances = []  # σ²_θ
    control_variances = []  # σ²_S

    for sigma_sq in variance_range:
        width = np.sqrt(12 * sigma_sq / 3)

        task_dist = TaskDistribution(
            dist_type='uniform',
            ranges={
                'alpha': (max(0.1, base_mean[0] - width/2), base_mean[0] + width/2),
                'A': (max(0.001, base_mean[1] - width/2), base_mean[1] + width/2),
                'omega_c': (max(0.1, base_mean[2] - width/2), base_mean[2] + width/2)
            }
        )

        rng = np.random.default_rng()
        tasks = task_dist.sample(n_samples, rng)

        # Compute BOTH variances
        param_var = task_dist.compute_variance()
        control_var = compute_control_relevant_variance(env, tasks)

        param_variances.append(param_var)
        control_variances.append(control_var)

        # Compute gap
        gap_result = gap_computer.compute_gap(
            meta_policy, robust_policy, task_distribution=tasks,
            n_samples=n_samples, K=K
        )
        gaps.append(gap_result['gap'])

        print(f"σ²_θ = {param_var:.4f}, σ²_S = {control_var:.4f}, Gap = {gap_result['gap']:.6f}")

    gaps = np.array(gaps)
    param_variances = np.array(param_variances)
    control_variances = np.array(control_variances)

    # Create two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Gap vs Parameter Variance σ²_θ
    ax1.plot(param_variances, gaps, 'o-', linewidth=2, markersize=8, label='Empirical Gap')
    from scipy.stats import linregress
    slope1, intercept1, r_value1, _, _ = linregress(param_variances, gaps)
    fit_line1 = slope1 * param_variances + intercept1
    ax1.plot(param_variances, fit_line1, '--', linewidth=2, alpha=0.7,
             label=f'Linear Fit (R² = {r_value1**2:.3f})')
    ax1.set_xlabel('Parameter Variance σ²_θ', fontsize=12)
    ax1.set_ylabel('Optimality Gap', fontsize=12)
    ax1.set_title('Gap vs Parameter Variance', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Gap vs Control-Relevant Variance σ²_S
    ax2.plot(control_variances, gaps, 'o-', linewidth=2, markersize=8, color='orange', label='Empirical Gap')
    slope2, intercept2, r_value2, _, _ = linregress(control_variances, gaps)
    fit_line2 = slope2 * control_variances + intercept2
    ax2.plot(control_variances, fit_line2, '--', linewidth=2, alpha=0.7, color='red',
             label=f'Linear Fit (R² = {r_value2**2:.3f})')
    ax2.set_xlabel('Control-Relevant Variance σ²_S', fontsize=12)
    ax2.set_ylabel('Optimality Gap', fontsize=12)
    ax2.set_title('Gap vs Control-Relevant Variance', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f'Adaptation Gap Scaling (K={K} steps)', fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to {save_path}")

    plt.show()

    print(f"\n{'='*60}")
    print(f"VARIANCE SCALING ANALYSIS:")
    print(f"{'='*60}")
    print(f"Gap vs σ²_θ (parameter variance):  R² = {r_value1**2:.4f}, slope = {slope1:.6f}")
    print(f"Gap vs σ²_S (control variance):    R² = {r_value2**2:.4f}, slope = {slope2:.6f}")
    print(f"\nBetter fit: {'σ²_S (control-relevant)' if r_value2**2 > r_value1**2 else 'σ²_θ (parameter)'}")
    print(f"{'='*60}")

    return {
        'param_variances': param_variances,
        'control_variances': control_variances,
        'gaps': gaps,
        'param_r_squared': r_value1**2,
        'control_r_squared': r_value2**2,
        'param_slope': slope1,
        'control_slope': slope2
    }


def plot_gap_vs_variance(
    gap_computer: OptimalityGapComputer,
    meta_policy: torch.nn.Module,
    robust_policy: torch.nn.Module,
    variance_range: np.ndarray,
    K: int = 5,
    n_samples: int = 50,
    base_mean: np.ndarray = None,
    eta: float = 0.01,
    save_path: Optional[str] = None
):
    """ FIXED: Properly creates task distributions with varying variance.
    Plot empirical gap vs task variance and compare to theory.

    Theory predicts: Gap ∝ σ²_θ

    Args:
        gap_computer: OptimalityGapComputer instance
        meta_policy: Meta-learned policy
        robust_policy: Robust baseline policy
        variance_range: Array of variance values to test
        K: Number of adaptation steps
        n_samples: Number of tasks to sample per variance level
        base_mean: Mean task parameters (default: [1.0, 0.1, 5.0])
        eta: Learning rate for adaptation
        save_path: Path to save plot
    """
    import matplotlib.pyplot as plt
    from metaqctrl.quantum.noise_adapter import TaskDistribution, NoiseParameters

    if base_mean is None:
        base_mean = np.array([1.0, 0.1, 5.0])  # (alpha, A, omega_c)

    gaps = []
    empirical_variances = []

    for sigma_sq in variance_range:
        # FIXED: Create task distribution with specified variance
        # For uniform distribution, if we want variance σ²,
        # and uniform variance is (b-a)²/12, then (b-a) = √(12σ²)
        # We create symmetric ranges around the mean

        width = np.sqrt(12 * sigma_sq / 3)  # Divide by 3 because we have 3 dimensions

        # Create ranges symmetric around mean
        task_dist = TaskDistribution(
            dist_type='uniform',
            ranges={
                'alpha': (max(0.1, base_mean[0] - width/2), base_mean[0] + width/2),
                'A': (max(0.001, base_mean[1] - width/2), base_mean[1] + width/2),
                'omega_c': (max(0.1, base_mean[2] - width/2), base_mean[2] + width/2)
            }
        )

        # Sample tasks from this distribution
        rng = np.random.default_rng()
        tasks = task_dist.sample(n_samples, rng)

        # Verify actual variance
        actual_variance = task_dist.compute_variance()
        empirical_variances.append(actual_variance)

        # Compute gap for this variance level
        gap_result = gap_computer.compute_gap(
            meta_policy, robust_policy, task_distribution=tasks,
            n_samples=n_samples, K=K
        )
        gaps.append(gap_result['gap'])

        print(f"σ² = {sigma_sq:.4f} (actual: {actual_variance:.4f}), Gap = {gap_result['gap']:.6f}")

    gaps = np.array(gaps)
    empirical_variances = np.array(empirical_variances)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(empirical_variances, gaps, 'o-', linewidth=2, markersize=8, label='Empirical Gap')

    # Fit linear relationship
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(empirical_variances, gaps)
    fit_line = slope * empirical_variances + intercept
    plt.plot(empirical_variances, fit_line, '--', linewidth=2, alpha=0.7,
             label=f'Linear Fit: Gap = {slope:.4f}·σ² + {intercept:.4f}\n(R² = {r_value**2:.3f})')

    # Theoretical prediction (if constants are available)
    if hasattr(gap_computer, 'C_sep') and hasattr(gap_computer, 'L_F'):
        c_gap = gap_computer.C_sep * gap_computer.L_F * gap_computer.L ** 2
        mu = gap_computer.mu
        gaps_theory = c_gap * empirical_variances * (1 - np.exp(-mu * eta * K))
        plt.plot(empirical_variances, gaps_theory, ':', linewidth=2,
                 label=f'Theory: Gap = {c_gap:.4f}·σ²·(1-e^(-μηK))')

    plt.xlabel('Task Variance σ²_θ', fontsize=12)
    plt.ylabel('Optimality Gap (1 - Fidelity)', fontsize=12)
    plt.title(f'Gap vs Task Variance (K={K} adaptation steps)', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Add text box with statistics
    textstr = f'Slope: {slope:.6f}\nIntercept: {intercept:.6f}\nR²: {r_value**2:.4f}'
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to {save_path}")

    plt.show()

    return {
        'variances': empirical_variances,
        'gaps': gaps,
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value**2
    }


# Example usage
if __name__ == "__main__":
    ## Good 
    print("Optimality Gap Theory Module")
    print("=" * 50)
    
    # Example constants
    constants = GapConstants(
        C_sep=0.5,
        mu=0.1,
        L=1.0,
        L_F=0.5,
        C_K=1.0
    )
    
    # Test gap bound
    sigma_sq = 0.1
    K = 5
    eta = 0.01
    
    gap_bound = constants.gap_lower_bound(sigma_sq, K, eta)
    print(f"\nTheoretical gap lower bound:")
    print(f"  σ²_θ = {sigma_sq}, K = {K}, η = {eta}")
    print(f"  Gap ≥ {gap_bound:.4f}")
    
    # Vary K
    print("\nGap vs adaptation steps K:")
    for K in [1, 3, 5, 10, 20]:
        gap = constants.gap_lower_bound(sigma_sq, K, eta)
        print(f"  K = {K:2d}: Gap ≥ {gap:.4f}")
    
    # Vary variance
    print("\nGap vs task variance:")
    for sigma_sq in [0.01, 0.05, 0.1, 0.2, 0.5]:
        gap = constants.gap_lower_bound(sigma_sq, K=5, eta=eta)
        print(f"  σ² = {sigma_sq:.2f}: Gap ≥ {gap:.4f}")
