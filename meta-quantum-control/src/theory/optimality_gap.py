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
        """
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
        """
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
        """
        Compute empirical optimality gap.
        
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
            adapted_policy = self._adapt_policy(
                meta_policy, task_params, K=K, lr=inner_lr
            )
            F_meta = self._evaluate_policy(adapted_policy, task_params)
            meta_fidelities.append(F_meta)
            
            # Robust policy: evaluate directly (no adaptation)
            F_robust = self._evaluate_policy(robust_policy, task_params)
            robust_fidelities.append(F_robust)
        
        # Compute gap
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
        """Perform K-step gradient adaptation on task."""
        from copy import deepcopy
        
        adapted_policy = deepcopy(policy)
        adapted_policy.train()
        
        optimizer = torch.optim.SGD(adapted_policy.parameters(), lr=lr)
        
        # Get task features
        task_features = self._task_to_features(task_params)
        
        for _ in range(K):
            optimizer.zero_grad()
            
            # Generate controls
            controls = adapted_policy(task_features)
            
            # Simulate and compute loss
            fidelity = self._evaluate_controls(controls, task_params)
            loss = 1.0 - fidelity  # Minimize infidelity
            
            loss.backward()
            optimizer.step()
        
        return adapted_policy
    
    def _evaluate_policy(
        self,
        policy: torch.nn.Module,
        task_params: Dict
    ) -> float:
        """Evaluate policy on task and return fidelity."""
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
        """Convert task parameters to feature tensor."""
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
        """
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
        """
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
            controls = adapted(task_features)
            
            fidelity = self._evaluate_controls(controls, task)
            loss = 1.0 - fidelity
            
            # Compute gradient norm
            loss.backward()
            grad_norm_sq = sum(
                torch.sum(p.grad ** 2).item()
                for p in adapted.parameters() if p.grad is not None
            )
            
            # Estimate μ from PL condition (assume L* ≈ 0)
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
        """
        Estimate L: Lipschitz constant of fidelity w.r.t. task parameters.
        
        L ≈ max |F(π, θ) - F(π, θ')| / ||θ - θ'||
        """
        task_pairs = np.random.choice(len(tasks), size=(n_samples, 2), replace=False)
        
        lipschitz_ratios = []
        for i, j in task_pairs:
            task_i, task_j = tasks[i], tasks[j]
            
            # Evaluate fidelity on both tasks
            F_i = self._evaluate_policy(policy, task_i)
            F_j = self._evaluate_policy(policy, task_j)
            
            # Compute task distance
            theta_i = np.array([task_i.alpha, task_i.A, task_i.omega_c])
            theta_j = np.array([task_j.alpha, task_j.A, task_j.omega_c])
            task_dist = np.linalg.norm(theta_i - theta_j)
            
            if task_dist > 1e-6:
                ratio = abs(F_i - F_j) / task_dist
                lipschitz_ratios.append(ratio)
        
        return np.max(lipschitz_ratios) if lipschitz_ratios else 1.0
    
    def _estimate_lipschitz_policy(
        self,
        policy: torch.nn.Module,
        task: Dict
    ) -> float:
        """
        Estimate L_F: Lipschitz constant of fidelity w.r.t. policy parameters.
        
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


def plot_gap_vs_variance(
    gap_computer: OptimalityGapComputer,
    meta_policy: torch.nn.Module,
    robust_policy: torch.nn.Module,
    variance_range: np.ndarray,
    K: int = 5,
    save_path: Optional[str] = None
):
    """
    Plot empirical gap vs task variance and compare to theory.
    
    Theory predicts: Gap ∝ σ²_θ
    """
    import matplotlib.pyplot as plt
    
    gaps = []
    for sigma_sq in variance_range:
        # Generate task distribution with this variance
        # (Implementation depends on your task sampler)
        # For now, placeholder
        gap_result = gap_computer.compute_gap(
            meta_policy, robust_policy, task_distribution=[], n_samples=50, K=K
        )
        gaps.append(gap_result['gap'])
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(variance_range, gaps, 'o-', label='Empirical Gap')
    
    # Theoretical prediction (if constants known)
    # gaps_theory = constants.gap_lower_bound(variance_range, K, eta)
    # plt.plot(variance_range, gaps_theory, '--', label='Theory Lower Bound')
    
    plt.xlabel('Task Variance σ²_θ')
    plt.ylabel('Optimality Gap')
    plt.title(f'Gap vs Task Variance (K={K})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# Example usage
if __name__ == "__main__":
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
