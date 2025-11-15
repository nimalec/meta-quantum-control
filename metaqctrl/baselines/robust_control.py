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

                # Finite difference: gradient of loss = -gradient of fidelity
                gradient[i, j] = -(f_plus - f0) / epsilon

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
    from metaqctrl.meta_rl.policy import PulsePolicy
    
    # Create policy
    policy = PulsePolicy(
        task_feature_dim=3,
        hidden_dim=64,
        n_segments=20,
        n_controls=2
    )
    
    
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
