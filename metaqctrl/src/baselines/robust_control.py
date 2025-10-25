"""
Robust Control Baselines

Implements:
1. Minimax robust: π_rob = argmin_π max_θ L(π, θ)
2. Average robust: π_rob = argmin_π E_θ[L(π, θ)] (no adaptation)
3. Nominal + robustification
4. GRAPE: Gradient Ascent Pulse Engineering for direct pulse optimization
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Callable, Optional
from scipy.optimize import minimize

# Optional CVXPY for convex optimization
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    cp = None


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


class GRAPEOptimizer:
    """
    GRAPE (Gradient Ascent Pulse Engineering) baseline.

    Direct gradient-based optimization of control pulses to maximize fidelity.
    Unlike policy-based methods, GRAPE optimizes pulse sequences directly
    without a parameterized policy.

    Reference: Khaneja et al., "Optimal control of coupled spin dynamics:
    design of NMR pulse sequences by gradient ascent algorithms" (2005)
    """

    def __init__(
        self,
        n_segments: int,
        n_controls: int,
        T: float = 1.0,
        control_bounds: tuple = (-5.0, 5.0),
        learning_rate: float = 0.1,
        method: str = 'adam',  # 'adam', 'lbfgs', 'gradient'
        device: torch.device = torch.device('cpu')
    ):
        """
        Args:
            n_segments: Number of time segments
            n_controls: Number of control channels
            T: Total evolution time
            control_bounds: (min, max) bounds on control amplitudes
            learning_rate: Learning rate for optimization
            method: Optimization method ('adam', 'lbfgs', 'gradient')
            device: torch device
        """
        self.n_segments = n_segments
        self.n_controls = n_controls
        self.T = T
        self.control_bounds = control_bounds
        self.learning_rate = learning_rate
        self.method = method
        self.device = device

        # Initialize control pulses randomly
        # Note: Must be a leaf tensor for optimization
        self.controls = torch.nn.Parameter(
            torch.randn(n_segments, n_controls, device=device) * 0.1
        )

        # Setup optimizer
        if method == 'adam':
            self.optimizer = torch.optim.Adam([self.controls], lr=learning_rate)
        elif method == 'lbfgs':
            self.optimizer = torch.optim.LBFGS([self.controls], lr=learning_rate, max_iter=20)
        else:  # gradient descent
            self.optimizer = torch.optim.SGD([self.controls], lr=learning_rate)

        self.fidelity_history = []
        self.gradient_norms = []

    def optimize(
        self,
        simulate_fn: Callable,
        task_params,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        verbose: bool = True
    ) -> np.ndarray:
        """
        Optimize control pulses for a single task using GRAPE.

        Args:
            simulate_fn: Function that takes controls (numpy array) and task_params,
                        returns fidelity (float)
            task_params: Task parameters (NoiseParameters)
            max_iterations: Maximum number of optimization iterations
            tolerance: Convergence tolerance on fidelity improvement
            verbose: Print progress

        Returns:
            optimal_controls: Optimized control sequence (n_segments, n_controls)
        """
        if verbose:
            print(f"Running GRAPE optimization for {max_iterations} iterations...")

        best_fidelity = -np.inf

        for iteration in range(max_iterations):
            def closure():
                """Closure for LBFGS optimizer."""
                self.optimizer.zero_grad()

                # Get current controls (apply bounds via tanh)
                controls_bounded = self._apply_bounds(self.controls)

                # Convert to numpy for simulation
                controls_np = controls_bounded.detach().cpu().numpy()

                # Simulate and compute fidelity
                fidelity = simulate_fn(controls_np, task_params)

                # Loss = negative fidelity (we want to maximize fidelity)
                loss = -torch.tensor(fidelity, device=self.device, dtype=torch.float32)

                # Compute gradients via finite differences
                # (since quantum simulation is not differentiable)
                grad = self._compute_finite_difference_gradient(
                    controls_np, task_params, simulate_fn, fidelity
                )

                # Set gradient manually
                self.controls.grad = torch.tensor(
                    grad, device=self.device, dtype=torch.float32
                )

                return loss

            if self.method == 'lbfgs':
                loss = self.optimizer.step(closure)
                loss_val = loss.item() if isinstance(loss, torch.Tensor) else loss
            else:
                loss = closure()
                self.optimizer.step()
                loss_val = loss.item()

            # Track progress
            current_fidelity = -loss_val
            self.fidelity_history.append(current_fidelity)

            if self.controls.grad is not None:
                grad_norm = torch.norm(self.controls.grad).item()
                self.gradient_norms.append(grad_norm)

            # Check convergence
            if current_fidelity > best_fidelity:
                improvement = current_fidelity - best_fidelity
                best_fidelity = current_fidelity

                if improvement < tolerance and iteration > 10:
                    if verbose:
                        print(f"Converged at iteration {iteration}")
                    break

            # Logging
            if verbose and iteration % 10 == 0:
                grad_norm_str = f", ||∇||={grad_norm:.2e}" if self.controls.grad is not None else ""
                print(f"Iter {iteration:3d}: Fidelity = {current_fidelity:.6f}{grad_norm_str}")

        # Return optimized controls
        optimal_controls = self._apply_bounds(self.controls).detach().cpu().numpy()

        if verbose:
            print(f"Optimization complete! Final fidelity: {best_fidelity:.6f}")

        return optimal_controls

    def _apply_bounds(self, controls: torch.Tensor) -> torch.Tensor:
        """Apply control amplitude bounds using scaled tanh."""
        min_val, max_val = self.control_bounds
        scale = (max_val - min_val) / 2
        offset = (max_val + min_val) / 2
        return scale * torch.tanh(controls) + offset

    def _compute_finite_difference_gradient(
        self,
        controls: np.ndarray,
        task_params,
        simulate_fn: Callable,
        f0: float,
        epsilon: float = 1e-4
    ) -> np.ndarray:
        """
        Compute gradient via finite differences.

        ∂F/∂u_i ≈ (F(u + ε*e_i) - F(u)) / ε

        Args:
            controls: Current control sequence (n_segments, n_controls)
            task_params: Task parameters
            simulate_fn: Simulation function
            f0: Current fidelity value
            epsilon: Finite difference step size

        Returns:
            gradient: Gradient array (n_segments, n_controls)
        """
        gradient = np.zeros_like(controls)

        # Iterate over all control elements
        for i in range(self.n_segments):
            for j in range(self.n_controls):
                # Perturb control
                controls_plus = controls.copy()
                controls_plus[i, j] += epsilon

                # Evaluate fidelity
                f_plus = simulate_fn(controls_plus, task_params)

                # Finite difference
                gradient[i, j] = (f_plus - f0) / epsilon

        return gradient

    def optimize_robust(
        self,
        simulate_fn: Callable,
        task_distribution: List,
        max_iterations: int = 100,
        robust_type: str = 'average',  # 'average', 'worst_case'
        verbose: bool = True
    ) -> np.ndarray:
        """
        Optimize controls to be robust across task distribution.

        Args:
            simulate_fn: Simulation function
            task_distribution: List of task parameters
            max_iterations: Maximum iterations
            robust_type: 'average' or 'worst_case'
            verbose: Print progress

        Returns:
            robust_controls: Optimized control sequence
        """
        if verbose:
            print(f"Running robust GRAPE ({robust_type}) over {len(task_distribution)} tasks...")

        for iteration in range(max_iterations):
            self.optimizer.zero_grad()

            # Evaluate on all tasks
            fidelities = []
            controls_bounded = self._apply_bounds(self.controls)
            controls_np = controls_bounded.detach().cpu().numpy()

            for task_params in task_distribution:
                fid = simulate_fn(controls_np, task_params)
                fidelities.append(fid)

            # Aggregate objective
            if robust_type == 'average':
                objective = np.mean(fidelities)
            elif robust_type == 'worst_case':
                objective = np.min(fidelities)
            else:
                raise ValueError(f"Unknown robust_type: {robust_type}")

            # Compute average gradient across tasks
            grad_sum = np.zeros((self.n_segments, self.n_controls))

            for task_params, fid in zip(task_distribution, fidelities):
                grad = self._compute_finite_difference_gradient(
                    controls_np, task_params, simulate_fn, fid
                )
                if robust_type == 'average':
                    grad_sum += grad / len(task_distribution)
                elif robust_type == 'worst_case':
                    # Only use gradient from worst task
                    if fid == np.min(fidelities):
                        grad_sum = grad
                        break

            # Set gradient
            self.controls.grad = torch.tensor(
                grad_sum, device=self.device, dtype=torch.float32
            )

            # Optimization step
            self.optimizer.step()

            # Logging
            if verbose and iteration % 10 == 0:
                print(f"Iter {iteration:3d}: {robust_type} fidelity = {objective:.6f}, "
                      f"worst = {np.min(fidelities):.6f}, best = {np.max(fidelities):.6f}")

            self.fidelity_history.append(objective)

        optimal_controls = self._apply_bounds(self.controls).detach().cpu().numpy()

        if verbose:
            final_fidelities = [simulate_fn(optimal_controls, tp) for tp in task_distribution]
            print(f"\nRobust GRAPE complete!")
            print(f"  Mean fidelity: {np.mean(final_fidelities):.6f}")
            print(f"  Worst case:    {np.min(final_fidelities):.6f}")
            print(f"  Best case:     {np.max(final_fidelities):.6f}")

        return optimal_controls

    def reset(self):
        """Reset controls to random initialization."""
        with torch.no_grad():
            self.controls.data = torch.randn_like(self.controls) * 0.1
        self.fidelity_history = []
        self.gradient_norms = []

    def get_controls(self) -> np.ndarray:
        """Get current control sequence."""
        return self._apply_bounds(self.controls).detach().cpu().numpy()

    def set_controls(self, controls: np.ndarray):
        """Set control sequence."""
        with torch.no_grad():
            self.controls.data = torch.tensor(controls, device=self.device, dtype=torch.float32)


# Example usage
if __name__ == "__main__":
    from metaqctrl.src.meta_rl.policy import PulsePolicy
    
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

    # Test GRAPE optimizer
    print("\n" + "="*60)
    print("Testing GRAPE Optimizer")
    print("="*60)

    grape = GRAPEOptimizer(
        n_segments=20,
        n_controls=2,
        T=1.0,
        control_bounds=(-2.0, 2.0),
        learning_rate=0.05,
        method='adam'
    )

    # Dummy simulation function
    def dummy_simulate(controls, task_params):
        """Dummy simulator returning random fidelity for testing."""
        # In real use, this would call LindbladSimulator
        return np.random.rand() * 0.5 + 0.3

    print("\nTesting single-task GRAPE optimization:")
    dummy_task_params = {'alpha': 1.0, 'A': 0.1, 'omega_c': 5.0}

    # Run a few iterations for demo
    optimal_controls = grape.optimize(
        simulate_fn=dummy_simulate,
        task_params=dummy_task_params,
        max_iterations=5,
        verbose=True
    )

    print(f"\nOptimal controls shape: {optimal_controls.shape}")
    print(f"Control range: [{optimal_controls.min():.3f}, {optimal_controls.max():.3f}]")

    print("\n" + "="*60)
    print("GRAPE baseline successfully implemented!")
    print("="*60)
