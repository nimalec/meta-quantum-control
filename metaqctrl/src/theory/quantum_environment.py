"""
Quantum Environment Bridge

Unified interface between theory and experiments.
Handles simulation, caching, and fidelity computation efficiently.
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional
from functools import lru_cache
from metaqctrl.src.quantum.lindblad import LindbladSimulator
from metaqctrl.src.quantum.lindblad_torch import DifferentiableLindbladSimulator, numpy_to_torch_complex
from metaqctrl.src.quantum.noise_models import NoiseParameters
from metaqctrl.src.quantum.gates import state_fidelity


class QuantumEnvironment:
    """
    Unified environment for quantum control.
    
    Features:
    - Caching of Lindblad operators by task
    - Efficient simulation reuse
    - Unified interface for theory and experiments
    """
    
    def __init__(
        self,
        H0: np.ndarray,
        H_controls: list,
        psd_to_lindblad,
        target_state: np.ndarray,
        T: float = 1.0,
        method: str = 'RK45'
    ):
        """
        Args:
            H0: Drift Hamiltonian
            H_controls: List of control Hamiltonians
            psd_to_lindblad: PSDToLindblad instance
            target_state: Target density matrix
            T: Evolution time
            method: Integration method
        """
        self.H0 = H0
        self.H_controls = H_controls
        self.psd_to_lindblad = psd_to_lindblad
        self.target_state = target_state
        self.T = T
        self.method = method
        
        # System dimensions
        self.d = H0.shape[0]
        self.n_controls = len(H_controls)
        
        # Cache for Lindblad operators (keyed by task hash)
        self._L_cache = {}
        
        # Cache for simulators (keyed by task hash)
        self._sim_cache = {}
        
        # Initial state (|0⟩ for single qubit)
        self.rho0 = np.zeros((self.d, self.d), dtype=complex)
        self.rho0[0, 0] = 1.0
        
        print(f"QuantumEnvironment initialized: d={self.d}, n_controls={self.n_controls}, T={T}")
    
    def _task_hash(self, task_params: NoiseParameters) -> tuple:
        """Create hashable key for task."""
        return (
            round(task_params.alpha, 6),
            round(task_params.A, 6),
            round(task_params.omega_c, 6)
        )
    
    def get_lindblad_operators(self, task_params: NoiseParameters) -> list:
        """ Good.  
        Get Lindblad operators for task with caching.
        
        Args:
            task_params: Task noise parameters
            
        Returns:
            L_ops: List of Lindblad operators
        """
        key = self._task_hash(task_params)
        
        if key not in self._L_cache:
            L_ops = self.psd_to_lindblad.get_lindblad_operators(task_params)
            self._L_cache[key] = L_ops
        
        return self._L_cache[key]
    
    def get_simulator(self, task_params: NoiseParameters) -> LindbladSimulator:
        """ Good. 
        Get simulator for task with caching.
        
        Args:
            task_params: Task noise parameters
            
        Returns:
            sim: LindbladSimulator instance
        """
        key = self._task_hash(task_params)
        
        if key not in self._sim_cache:
            L_ops = self.get_lindblad_operators(task_params)
            
            sim = LindbladSimulator(
                H0=self.H0,
                H_controls=self.H_controls,
                L_operators=L_ops,
                method=self.method
            )
            
            self._sim_cache[key] = sim
        
        return self._sim_cache[key]
    
    def evaluate_controls(
        self,
        controls: np.ndarray,
        task_params: NoiseParameters,
        return_trajectory: bool = False
    ) -> float:
        """ Good. 
        Simulate and compute fidelity.
        
        Args:
            controls: Control sequence (n_segments, n_controls)
            task_params: Task parameters
            return_trajectory: If True, return (fidelity, trajectory)
            
        Returns:
            fidelity: Achieved fidelity (float)
            or (fidelity, trajectory) if return_trajectory=True
        """
        # Get cached simulator
        sim = self.get_simulator(task_params)
        
        # Simulate
        rho_final, trajectory = sim.evolve(self.rho0, controls, self.T)
        
        # Compute fidelity
        fidelity = state_fidelity(rho_final, self.target_state)
        
        if return_trajectory:
            return fidelity, trajectory
        return fidelity
    
    def evaluate_policy(
        self,
        policy: torch.nn.Module,
        task_params: NoiseParameters,
        device: torch.device = torch.device('cpu')
    ) -> float:
        """ Good. 
        Evaluate policy on task.
        
        Args:
            policy: Policy network
            task_params: Task parameters
            device: torch device
            
        Returns:
            fidelity: Achieved fidelity
        """
        policy.eval()
        
        with torch.no_grad():
            # Task features
            task_features = torch.tensor(
                task_params.to_array(),
                dtype=torch.float32,
                device=device
            )
            
            # Generate controls
            controls = policy(task_features)
            controls_np = controls.cpu().numpy()
        
        # Evaluate
        fidelity = self.evaluate_controls(controls_np, task_params)
        
        return fidelity
    
    def compute_loss(
        self,
        policy: torch.nn.Module,
        task_params: NoiseParameters,
        device: torch.device = torch.device('cpu')
    ) -> torch.Tensor:
        """ Take a look ==> which function is replacing it. 
        Compute loss (infidelity) with gradient support.

        WARNING: This implementation is NOT fully differentiable because the quantum
        simulation happens in numpy. For proper gradient flow through the quantum
        dynamics, you need:
        1. JAX-based differentiable quantum simulator, OR
        2. PyTorch-based quantum simulator, OR
        3. Analytical gradients via adjoint method

        Current behavior: Gradients flow through the policy network but NOT through
        the quantum simulation. This works for zeroth-order optimization and
        finite-difference based MAML, but not for full second-order MAML.

        Args:
            policy: Policy network
            task_params: Task parameters
            device: torch device

        Returns:
            loss: Loss tensor (gradients only partial)
        """
        # Task features
        task_features = torch.tensor(
            task_params.to_array(),
            dtype=torch.float32,
            device=device
        )

        # Generate controls
        controls = policy(task_features)

        # NOTE: detach() breaks gradient flow through quantum simulation
        # For full differentiability, need JAX or PyTorch quantum simulator
        controls_np = controls.detach().cpu().numpy()

        # Evaluate fidelity (non-differentiable step)
        fidelity = self.evaluate_controls(controls_np, task_params)

        # Create loss tensor
        # Setting requires_grad=False is honest about gradient limitations
        loss = torch.tensor(
            1.0 - fidelity,
            dtype=torch.float32,
            device=device,
            requires_grad=False
        )

        return loss

    def compute_loss_differentiable(
        self,
        policy: torch.nn.Module,
        task_params: NoiseParameters,
        device: torch.device = torch.device('cpu'),
        use_rk4: bool = True
    ) -> torch.Tensor:
        """ Good: OK... this is the differentiable version that propogates gradients. 
        Compute loss (infidelity) with FULL gradient support through quantum simulation.

        This is the NEW differentiable version that allows gradients to flow through
        the entire quantum dynamics. Use this for proper meta-learning!

        Args:
            policy: Policy network
            task_params: Task parameters
            device: torch device
            use_rk4: If True, use RK4 integration (more accurate but slower)

        Returns:
            loss: FULLY DIFFERENTIABLE loss tensor with gradients!
        """
        # Task features
        task_features = torch.tensor(
            task_params.to_array(),
            dtype=torch.float32,
            device=device
        )

        # Generate controls (this is differentiable)
        controls = policy(task_features)  # (n_segments, n_controls)

        # Get Lindblad operators for this task
        L_ops_numpy = self.psd_to_lindblad.get_lindblad_operators(task_params)

        # Convert system to PyTorch tensors
        H0_torch = numpy_to_torch_complex(self.H0, device)
        H_controls_torch = [numpy_to_torch_complex(H, device) for H in self.H_controls]
        L_ops_torch = [numpy_to_torch_complex(L, device) for L in L_ops_numpy]

        # Create differentiable simulator
        sim = DifferentiableLindbladSimulator(
            H0=H0_torch,
            H_controls=H_controls_torch,
            L_operators=L_ops_torch,
            dt=0.05,
            method='rk4' if use_rk4 else 'euler',
            device=device
        )

        # Initial state |0⟩
        rho0 = torch.zeros((self.d, self.d), dtype=torch.complex64, device=device)
        rho0[0, 0] = 1.0

        # DIFFERENTIABLE quantum evolution!
        rho_final = sim(rho0, controls, self.T)

        # Target state (convert to torch)
        target_state_torch = numpy_to_torch_complex(self.target_state, device)

        # Compute fidelity (differentiable)
        fidelity = self._torch_state_fidelity(rho_final, target_state_torch)

        # Loss = infidelity (differentiable!)
        loss = 1.0 - fidelity

        return loss

    def _torch_state_fidelity(
        self,
        rho: torch.Tensor,
        sigma: torch.Tensor
    ) -> torch.Tensor:
        """ Good ---> differential form of fidelity . can be implemented with autodiff. 
        Compute quantum state fidelity in PyTorch (differentiable).

        F(ρ, σ) = tr(√(√ρ σ √ρ))²

        For pure states or near-pure states, we use a simpler formula:
        F ≈ |tr(ρσ)|² (works well for high-fidelity control)

        Args:
            rho: Density matrix (d x d) complex tensor
            sigma: Density matrix (d x d) complex tensor

        Returns:
            fidelity: Real-valued fidelity in [0, 1]
        """
        # Simplified fidelity for differentiability
        # Full formula requires matrix square roots which are tricky to differentiate
        # This approximation works well for high-fidelity states

        # Method 1: Simple overlap (works for pure or near-pure states)
        overlap = torch.trace(rho @ sigma)
        fidelity = torch.abs(overlap) ** 2

        # Clamp to [0, 1] for numerical stability
        fidelity = torch.clamp(fidelity.real, 0.0, 1.0)

        return fidelity

    def clear_cache(self):
        """Clear all caches."""
        self._L_cache.clear()
        self._sim_cache.clear()
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            'n_cached_operators': len(self._L_cache),
            'n_cached_simulators': len(self._sim_cache),
            'cache_size_mb': (
                len(str(self._L_cache)) + len(str(self._sim_cache))
            ) / 1e6
        }


class BatchedQuantumEnvironment(QuantumEnvironment):
    """
    Batched version for parallel task evaluation.
    Uses JAX for vectorization.
    """
    
    def __init__(self, *args, use_jax: bool = True, **kwargs):
        ## This uses Jax 
        super().__init__(*args, **kwargs)
        self.use_jax = use_jax
        
        if use_jax:
            try:
                from metaqctrl.src.quantum.lindblad import LindbladJAX
                self.jax_sim = LindbladJAX(
                    self.H0,
                    self.H_controls,
                    n_segments=20,  # From config
                    T=self.T
                )
                print("JAX batching enabled")
            except ImportError:
                print("JAX not available, falling back to serial")
                self.use_jax = False
    
    def evaluate_controls_batch(
        self,
        controls_batch: np.ndarray,
        task_params_batch: list
    ) -> np.ndarray:
        """
        Evaluate multiple control sequences in parallel.
        
        Args:
            controls_batch: (batch_size, n_segments, n_controls)
            task_params_batch: List of NoiseParameters
            
        Returns:
            fidelities: (batch_size,) array of fidelities
        """
        if self.use_jax:
            # TODO: Implement JAX batching
            # For now, fall back to serial
            pass
        
        # Serial fallback
        fidelities = []
        for controls, task_params in zip(controls_batch, task_params_batch):
            fid = self.evaluate_controls(controls, task_params)
            fidelities.append(fid)
        
        return np.array(fidelities)


# Factory function
def create_quantum_environment(config: dict, target_state: np.ndarray) -> QuantumEnvironment:
    """
    Create quantum environment from config.
    
    Args:
        config: Configuration dictionary
        target_state: Target density matrix
        
    Returns:
        env: QuantumEnvironment instance
    """
    from metaqctrl.src.quantum.noise_models import NoisePSDModel, PSDToLindblad
    
    # Pauli matrices for 1-qubit
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    # System Hamiltonians
    H0 = 0.0 * sigma_z
    H_controls = [sigma_x, sigma_y]
    
    # PSD model
    psd_model = NoisePSDModel(model_type=config.get('psd_model', 'one_over_f'))
    
    # Sampling frequencies (control bandwidth)
    n_segments = config.get('n_segments', 20)
    T = config.get('horizon', 1.0)
    omega_max = n_segments / T
    omega_sample = np.linspace(0, omega_max, 10)
    
    # PSD to Lindblad converter
    psd_to_lindblad = PSDToLindblad(
        basis_operators=[sigma_x, sigma_y, sigma_z],
        sampling_freqs=omega_sample,
        psd_model=psd_model
    )
    
    # Create environment
    env = QuantumEnvironment(
        H0=H0,
        H_controls=H_controls,
        psd_to_lindblad=psd_to_lindblad,
        target_state=target_state,
        T=T,
        method=config.get('integration_method', 'RK45')
    )
    
    return env
