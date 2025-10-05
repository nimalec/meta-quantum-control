"""
Robust Control Baselines

Implements:
1. Minimax robust: π_rob = argmin_π max_θ L(π, θ)
2. Average robust: π_rob = argmin_π E_θ[L(π, θ)] (no adaptation)
3. Nominal + robustification
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Callable, Optional
import cvxpy as cp
from scipy.optimize import minimize


class RobustPolicy:
    """
    Train a policy to be robust across task distribution.
    No adaptation allowed at test time.
    """
    
    def __init__(
        self,
        policy: nn.Module,
        learning_rate: float = 0.001,
        robust_type: str = 'average',  # 'average', 'minimax', 'cvar'
        device: torch.device = torch.device('cpu')
    ):
        """
        Args:
            policy: Policy network
            learning_rate: Learning rate for optimization
            robust_type: Type of robustness criterion
            device: torch device
        """
        self.policy = policy.to(device)
        self.learning_rate = learning_rate
        self.robust_type = robust_type
        self.device = device
        
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        self.train_losses = []
    
    def train_step(
        self,
        task_batch: List[Dict],
        loss_fn: Callable
    ) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            task_batch: Batch of tasks with data
            loss_fn: Loss function
            
        Returns:
            metrics: Training metrics
        """
        self.optimizer.zero_grad()
        
        if self.robust_type == 'average':
            total_loss, metrics = self._average_robust_loss(task_batch, loss_fn)
        elif self.robust_type == 'minimax':
            total_loss, metrics = self._minimax_robust_loss(task_batch, loss_fn)
        elif self.robust_type == 'cvar':
            total_loss, metrics = self._cvar_robust_loss(task_batch, loss_fn)
        else:
            raise ValueError(f"Unknown robust_type: {self.robust_type}")
        
        # Backprop and update
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self.train_losses.append(total_loss.item())
        
        return metrics
    
    def _average_robust_loss(
        self,
        task_batch: List[Dict],
        loss_fn: Callable
    ) -> tuple:
        """
        Average robust: min E_θ[L(π, θ)]
        """
        losses = []
        
        for task_data in task_batch:
            loss = loss_fn(self.policy, task_data['support'])
            losses.append(loss)
        
        total_loss = torch.stack(losses).mean()
        
        metrics = {
            'loss': total_loss.item(),
            'mean_task_loss': total_loss.item(),
            'max_task_loss': max(l.item() for l in losses),
            'min_task_loss': min(l.item() for l in losses)
        }
        
        return total_loss, metrics
    
    def _minimax_robust_loss(
        self,
        task_batch: List[Dict],
        loss_fn: Callable
    ) -> tuple:
        """
        Minimax robust: min max_θ L(π, θ)
        Implemented via max + smooth approximation.
        """
        losses = []
        
        for task_data in task_batch:
            loss = loss_fn(self.policy, task_data['support'])
            losses.append(loss)
        
        loss_tensor = torch.stack(losses)
        
        # Smooth max via LogSumExp
        # max(x) ≈ (1/β) log(Σ exp(β*x))
        beta = 10.0  # Temperature parameter
        smooth_max = torch.logsumexp(beta * loss_tensor, dim=0) / beta
        
        metrics = {
            'loss': smooth_max.item(),
            'mean_task_loss': loss_tensor.mean().item(),
            'max_task_loss': loss_tensor.max().item(),
            'worst_case_approx': smooth_max.item()
        }
        
        return smooth_max, metrics
    
    def _cvar_robust_loss(
        self,
        task_batch: List[Dict],
        loss_fn: Callable,
        alpha: float = 0.1
    ) -> tuple:
        """
        CVaR robust: min CVaR_α[L(π, θ)]
        Minimizes conditional value at risk (average of worst α fraction).
        """
        losses = []
        
        for task_data in task_batch:
            loss = loss_fn(self.policy, task_data['support'])
            losses.append(loss)
        
        loss_tensor = torch.stack(losses)
        
        # Sort losses and take worst α fraction
        k = max(1, int(alpha * len(losses)))
        worst_losses, _ = torch.topk(loss_tensor, k)
        cvar_loss = worst_losses.mean()
        
        metrics = {
            'loss': cvar_loss.item(),
            'mean_task_loss': loss_tensor.mean().item(),
            'cvar_loss': cvar_loss.item(),
            'alpha': alpha
        }
        
        return cvar_loss, metrics
    
    def evaluate(
        self,
        test_tasks: List[Dict],
        loss_fn: Callable
    ) -> Dict[str, float]:
        """Evaluate robust policy on test tasks."""
        self.policy.eval()
        
        test_losses = []
        with torch.no_grad():
            for task_data in test_tasks:
                loss = loss_fn(self.policy, task_data['support'])
                test_losses.append(loss.item())
        
        self.policy.train()
        
        metrics = {
            'test_loss_mean': np.mean(test_losses),
            'test_loss_std': np.std(test_losses),
            'test_loss_max': np.max(test_losses),
            'test_loss_min': np.min(test_losses)
        }
        
        return metrics
    
    def save(self, path: str):
        """Save robust policy."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'robust_type': self.robust_type,
            'train_losses': self.train_losses
        }, path)
        print(f"Robust policy saved to {path}")
    
    def load(self, path: str):
        """Load robust policy."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.robust_type = checkpoint['robust_type']
        self.train_losses = checkpoint.get('train_losses', [])
        print(f"Robust policy loaded from {path}")


class RobustTrainer:
    """High-level trainer for robust policies."""
    
    def __init__(
        self,
        robust_policy: RobustPolicy,
        task_sampler: Callable,
        data_generator: Callable,
        loss_fn: Callable,
        n_samples_per_task: int = 20,
        log_interval: int = 10
    ):
        self.robust_policy = robust_policy
        self.task_sampler = task_sampler
        self.data_generator = data_generator
        self.loss_fn = loss_fn
        self.n_samples_per_task = n_samples_per_task
        self.log_interval = log_interval
    
    def train(
        self,
        n_iterations: int,
        tasks_per_batch: int = 16,
        val_tasks: int = 50,
        val_interval: int = 50,
        save_path: Optional[str] = None
    ):
        """
        Train robust policy.
        
        Args:
            n_iterations: Number of training iterations
            tasks_per_batch: Tasks per batch
            val_tasks: Number of validation tasks
            val_interval: Validate every N iterations
            save_path: Path to save model
        """
        print(f"Training robust policy ({self.robust_policy.robust_type})...")
        print(f"Iterations: {n_iterations}, Tasks/batch: {tasks_per_batch}\n")
        
        best_val_loss = float('inf')
        
        for iteration in range(n_iterations):
            # Sample tasks
            tasks = self.task_sampler(tasks_per_batch, split='train')
            
            # Generate data for each task
            task_batch = []
            for task_params in tasks:
                data = self.data_generator(
                    task_params,
                    n_trajectories=self.n_samples_per_task,
                    split='train'
                )
                task_batch.append({
                    'task_params': task_params,
                    'support': data
                })
            
            # Training step
            metrics = self.robust_policy.train_step(task_batch, self.loss_fn)
            
            # Logging
            if iteration % self.log_interval == 0:
                print(f"Iter {iteration}/{n_iterations} | "
                      f"Loss: {metrics['loss']:.4f} | "
                      f"Max: {metrics['max_task_loss']:.4f}")
            
            # Validation
            if iteration % val_interval == 0 and iteration > 0:
                val_tasks_list = self.task_sampler(val_tasks, split='val')
                val_task_batch = []
                for task_params in val_tasks_list:
                    data = self.data_generator(
                        task_params,
                        n_trajectories=self.n_samples_per_task,
                        split='val'
                    )
                    val_task_batch.append({
                        'task_params': task_params,
                        'support': data
                    })
                
                val_metrics = self.robust_policy.evaluate(val_task_batch, self.loss_fn)
                
                print(f"\n[Validation] Iter {iteration}")
                print(f"  Mean loss: {val_metrics['test_loss_mean']:.4f} ± "
                      f"{val_metrics['test_loss_std']:.4f}")
                print(f"  Max loss:  {val_metrics['test_loss_max']:.4f}\n")
                
                # Save best
                if save_path and val_metrics['test_loss_mean'] < best_val_loss:
                    best_val_loss = val_metrics['test_loss_mean']
                    best_path = save_path.replace('.pt', '_best.pt')
                    self.robust_policy.save(best_path)
        
        # Save final
        if save_path:
            self.robust_policy.save(save_path)
        
        print("\nRobust training complete!")


class H2RobustControl:
    """
    H∞ / H2 optimal control baseline.
    
    Finds control that minimizes worst-case or average performance
    under bounded disturbances.
    """
    
    def __init__(
        self,
        system_matrices: Dict,
        disturbance_bound: float = 1.0
    ):
        """
        Args:
            system_matrices: Dict with A, B, C, D matrices
            disturbance_bound: Bound on disturbance magnitude
        """
        self.A = system_matrices['A']
        self.B = system_matrices['B']
        self.C = system_matrices.get('C', np.eye(self.A.shape[0]))
        self.D = system_matrices.get('D', np.zeros((self.C.shape[0], self.B.shape[1])))
        self.disturbance_bound = disturbance_bound
    
    def solve_h_infinity(self, gamma: float = 1.0) -> np.ndarray:
        """
        Solve H∞ optimal control problem.
        
        Find K such that ||T_zw||_∞ ≤ γ where T_zw is closed-loop transfer function.
        
        Returns:
            K: Optimal feedback gain
        """
        n = self.A.shape[0]
        m = self.B.shape[1]
        
        # Use CVX to solve Riccati equation
        # This is a simplified implementation
        # Full H∞ requires solving Riccati equations
        
        # For now, use LQR as approximation
        K = self._solve_lqr()
        
        return K
    
    def _solve_lqr(self) -> np.ndarray:
        """Solve LQR as fallback."""
        from scipy.linalg import solve_continuous_are
        
        n = self.A.shape[0]
        Q = np.eye(n)
        R = np.eye(self.B.shape[1])
        
        # Solve ARE: A^T P + P A - P B R^{-1} B^T P + Q = 0
        P = solve_continuous_are(self.A, self.B, Q, R)
        
        # Optimal gain: K = R^{-1} B^T P
        K = np.linalg.solve(R, self.B.T @ P)
        
        return K


class DomainRandomization:
    """
    Domain randomization baseline.
    
    Train on randomized task parameters to encourage robustness.
    Similar to robust average but with explicit augmentation.
    """
    
    def __init__(
        self,
        policy: nn.Module,
        randomization_strength: float = 0.1,
        learning_rate: float = 0.001
    ):
        self.policy = policy
        self.randomization_strength = randomization_strength
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    
    def randomize_task(self, task_params: Dict) -> Dict:
        """Add random perturbation to task parameters."""
        randomized = {}
        for key, value in task_params.items():
            noise = np.random.randn() * self.randomization_strength
            randomized[key] = value * (1 + noise)
        return randomized
    
    def train_step(self, task_batch: List[Dict], loss_fn: Callable) -> float:
        """Training step with domain randomization."""
        self.optimizer.zero_grad()
        
        total_loss = 0.0
        for task_data in task_batch:
            # Apply randomization
            randomized_task = self.randomize_task(task_data['task_params'])
            task_data_rand = {**task_data, 'task_params': randomized_task}
            
            # Compute loss
            loss = loss_fn(self.policy, task_data_rand['support'])
            total_loss += loss
        
        total_loss = total_loss / len(task_batch)
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()


# Example usage
if __name__ == "__main__":
    from src.meta_rl.policy import PulsePolicy
    
    # Create policy
    policy = PulsePolicy(
        task_feature_dim=3,
        hidden_dim=64,
        n_segments=20,
        n_controls=2
    )
    
    # Initialize robust policy
    robust_policy = RobustPolicy(
        policy=policy,
        learning_rate=0.001,
        robust_type='minimax'
    )
    
    print(f"Robust policy initialized: {robust_policy.robust_type}")
    print(f"Policy parameters: {policy.count_parameters():,}")
    
    # Dummy loss function
    def dummy_loss(policy, data):
        task_features = data['task_features']
        controls = policy(task_features)
        return torch.mean(controls ** 2)
    
    # Dummy task batch
    dummy_tasks = [
        {
            'support': {'task_features': torch.randn(10, 3)}
        }
        for _ in range(8)
    ]
    
    # Test training step
    print("\nTesting training step...")
    metrics = robust_policy.train_step(dummy_tasks, dummy_loss)
    print(f"Metrics: {metrics}")
    
    # Compare robust types
    print("\nComparing robust types:")
    for robust_type in ['average', 'minimax', 'cvar']:
        rp = RobustPolicy(
            policy=PulsePolicy(task_feature_dim=3, hidden_dim=32, n_segments=10, n_controls=2),
            robust_type=robust_type
        )
        m = rp.train_step(dummy_tasks, dummy_loss)
        print(f"  {robust_type:8s}: loss = {m['loss']:.4f}")
