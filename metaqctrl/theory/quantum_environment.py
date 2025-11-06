"""
Quantum Environment Bridge

Unified interface between theory and experiments.
Handles simulation, caching, and fidelity computation efficiently.
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional
from functools import lru_cache
from metaqctrl.quantum.lindblad import LindbladSimulator
from metaqctrl.quantum.lindblad_torch import DifferentiableLindbladSimulator, numpy_to_torch_complex
# Use adapter for backward-compatible v2 physics
from metaqctrl.quantum.noise_adapter import NoiseParameters
from metaqctrl.quantum.gates import state_fidelity


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
        method: str = 'RK45',
        target_unitary: np.ndarray = None
    ):
        """
        Args:
            H0: Drift Hamiltonian
            H_controls: List of control Hamiltonians
            psd_to_lindblad: PSDToLindblad instance
            target_state: Target density matrix
            T: Evolution time
            method: Integration method
            target_unitary: Target unitary gate (optional, for process fidelity)
        """
        self.H0 = H0
        self.H_controls = H_controls
        self.psd_to_lindblad = psd_to_lindblad
        self.target_state = target_state
        self.target_unitary = target_unitary  # Store for process fidelity
        self.T = T
        self.method = method
        self.sequence = None  
        self.omega0 = None 

        # System dimensions
        self.d = H0.shape[0]
        self.n_controls = len(H_controls)

        # Cache for Lindblad operators (keyed by task hash)
        self._L_cache = {}

        # Cache for simulators (keyed by task hash)
        self._sim_cache = {}

        # Cache for differentiable PyTorch simulators (keyed by task hash)
        self._torch_sim_cache = {}

        # Initial state (|0⟩ for single qubit, |00⟩ for two qubit)
        self.rho0 = np.zeros((self.d, self.d), dtype=complex)
        self.rho0[0, 0] = 1.0

        print(f"QuantumEnvironment initialized: d={self.d}, n_controls={self.n_controls}, T={T}")

    # API compatibility properties
    @property
    def num_controls(self) -> int:
        """Alias for n_controls for backward compatibility."""
        return self.n_controls

    @property
    def evolution_time(self) -> float:
        """Alias for T for backward compatibility."""
        return self.T

    @property
    def omega_control(self) -> np.ndarray:
        """Control frequencies - placeholder for compatibility."""
        # Return typical control frequencies based on system
        return np.array([1.0, 5.0, 10.0])

    @property
    def control_susceptibility(self) -> np.ndarray:
        """Control susceptibility matrix - placeholder for compatibility."""
        # Return identity-like matrix as placeholder
        return np.eye(len(self.H_controls))

    def compute_fidelity(self, controls: np.ndarray, task_params: NoiseParameters) -> float:
        """Alias for evaluate_controls for backward compatibility."""
        return self.evaluate_controls(controls, task_params, return_trajectory=False)

    def _task_hash(self, task_params: NoiseParameters) -> tuple:
        """Create hashable key for task (including model type)."""
        return (
            round(task_params.alpha, 6),
            round(task_params.A, 6),
            round(task_params.omega_c, 6),
            task_params.model_type  # NEW: Include model type in hash
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
            # Use adapter's backward-compatible API (handles v2 physics internally)
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

    def get_torch_simulator(
        self,
        task_params: NoiseParameters,
        device: torch.device,
        dt: float = 0.01,
        use_rk4: bool = True
    ) -> DifferentiableLindbladSimulator:
        """
        Get cached differentiable PyTorch simulator for task.

        CRITICAL OPTIMIZATION: Caches simulators to avoid recreating them
        for every loss call, which was the main performance bottleneck.

        Args:
            task_params: Task noise parameters
            device: torch device
            dt: Integration time step
            use_rk4: If True, use RK4 integration

        Returns:
            sim: Cached DifferentiableLindbladSimulator instance
        """
        # Cache key includes device and integration settings
        key = (self._task_hash(task_params), str(device), dt, use_rk4)

        if key not in self._torch_sim_cache:
            # Get Lindblad operators
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
                dt=dt,
                method='rk4' if use_rk4 else 'euler',
                device=device
            )

            self._torch_sim_cache[key] = sim

        return self._torch_sim_cache[key]
    
    def evaluate_controls(
        self,
        controls: np.ndarray,
        task_params: NoiseParameters,
        return_trajectory: bool = False,
        use_process_fidelity: bool = False
    ) -> float:
        """ Good.
        Simulate and compute fidelity.

        Args:
            controls: Control sequence (n_segments, n_controls)
            task_params: Task parameters
            return_trajectory: If True, return (fidelity, trajectory)
            use_process_fidelity: If True, use average gate fidelity over all input states
                                  (important for multi-qubit gates like CNOT!)

        Returns:
            fidelity: Achieved fidelity (float)
            or (fidelity, trajectory) if return_trajectory=True
        """
        # Get cached simulator
        sim = self.get_simulator(task_params)

        if use_process_fidelity and self.d > 2:
            # For multi-qubit gates, compute average fidelity over all computational basis states
            fidelity = self._compute_average_gate_fidelity(sim, controls)

            if return_trajectory:
                # For trajectory, just use initial state |0...0⟩
                rho_final, trajectory = sim.evolve(self.rho0, controls, self.T)
                return fidelity, trajectory
            return fidelity
        else:
            # Standard single-state fidelity (works for 1-qubit)
            rho_final, trajectory = sim.evolve(self.rho0, controls, self.T)
            fidelity = state_fidelity(rho_final, self.target_state)

            if return_trajectory:
                return fidelity, trajectory
            return fidelity

    def _compute_average_gate_fidelity(
        self,
        sim: LindbladSimulator,
        controls: np.ndarray
    ) -> float:
        """
        Compute average gate fidelity over all computational basis states.

        This is the proper fidelity measure for multi-qubit gates!

        For 2-qubits: Average over |00⟩, |01⟩, |10⟩, |11⟩

        Args:
            sim: LindbladSimulator instance
            controls: Control sequence

        Returns:
            avg_fidelity: Average fidelity over all basis states
        """
        from metaqctrl.quantum.gates import state_fidelity

        if self.target_unitary is None:
            # Fallback: Use entanglement fidelity formula
            # This is less accurate but doesn't require storing target unitary
            print("WARNING: target_unitary not provided. Using approximate fidelity.")
            print("         Set target_unitary in QuantumEnvironment for accurate process fidelity.")

            # Just return single-state fidelity as fallback
            rho_final, _ = sim.evolve(self.rho0, controls, self.T)
            return state_fidelity(rho_final, self.target_state)

        # Proper average gate fidelity: Test on all computational basis states
        fidelities = []

        for i in range(self.d):
            # Create initial state |i⟩
            ket_i = np.zeros(self.d, dtype=complex)
            ket_i[i] = 1.0
            rho_i = np.outer(ket_i, ket_i.conj())

            # Evolve under controls
            rho_final, _ = sim.evolve(rho_i, controls, self.T)

            # Target output: U_target |i⟩
            ket_target = self.target_unitary @ ket_i
            rho_target_i = np.outer(ket_target, ket_target.conj())

            # Compute fidelity
            fid = state_fidelity(rho_final, rho_target_i)
            fidelities.append(fid)

        # Average over all input states
        return float(np.mean(fidelities))
    
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
        use_rk4: bool = True,
        dt: float = 0.01
    ) -> torch.Tensor:
        """ Good: OK... this is the differentiable version that propogates gradients.
        Compute loss (infidelity) with FULL gradient support through quantum simulation.

        This is the NEW differentiable version that allows gradients to flow through
        the entire quantum dynamics. Use this for proper meta-learning!

        PERFORMANCE OPTIMIZED: Now uses cached simulators to avoid recreating them
        on every call. This is crucial for GPU performance!

        Args:
            policy: Policy network
            task_params: Task parameters
            device: torch device
            use_rk4: If True, use RK4 integration (more accurate but slower)
            dt: Integration time step (larger = faster but less accurate)

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

        # CRITICAL OPTIMIZATION: Get cached simulator instead of creating new one!
        sim = self.get_torch_simulator(task_params, device, dt=dt, use_rk4=use_rk4)

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
        """FIXED: Proper quantum fidelity for density matrices (differentiable).

        F(ρ, σ) = Tr(ρσ) for pure states, approximated for mixed states

        This implementation uses a simplified fidelity that avoids eigendecomposition
        issues with complex matrices. For nearly pure states (as in gate optimization),
        this is a good approximation.

        GRADIENT FIX: Use torch.abs() instead of .real/.imag to maintain gradient flow!

        Args:
            rho: Density matrix (d x d) complex tensor
            sigma: Density matrix (d x d) complex tensor

        Returns:
            fidelity: Real-valued fidelity in [0, 1]
        """
        # Simplified fidelity: F ≈ Tr(ρσ)² as approximation
        # For pure states, F = |⟨ψ|φ⟩|² = |Tr(ρσ)|² exactly
        # For mixed states, this is an approximation but avoids eigh

        trace_prod = torch.trace(rho @ sigma)

        # CRITICAL FIX: Use torch.abs() which maintains gradients!
        # torch.abs() for complex tensors computes |z|² = real² + imag²
        # and properly backpropagates through both components
        fidelity = torch.abs(trace_prod) ** 2

        # Clamp to valid range [0, 1]
        fidelity = torch.clamp(fidelity, 0.0, 1.0)

        return fidelity

    def clear_cache(self):
        """Clear all caches."""
        self._L_cache.clear()
        self._sim_cache.clear()
        self._torch_sim_cache.clear()

    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            'n_cached_operators': len(self._L_cache),
            'n_cached_simulators': len(self._sim_cache),
            'n_cached_torch_simulators': len(self._torch_sim_cache),
            'cache_size_mb': (
                len(str(self._L_cache)) + len(str(self._sim_cache)) + len(str(self._torch_sim_cache))
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
                from metaqctrl.quantum.lindblad import LindbladJAX
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


# Helper functions
def get_target_state_from_config(config: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get target density matrix and unitary from config.

    Args:
        config: Configuration dictionary with 'target_gate' and 'num_qubits' keys

    Returns:
        target_state: Target density matrix (d x d)
        target_unitary: Target unitary gate (d x d)
    """
    from metaqctrl.quantum.gates import TargetGates

    target_gate_name = config.get('target_gate', 'pauli_x')
    num_qubits = config.get('num_qubits', 1)

    # Get target unitary
    if target_gate_name == 'hadamard':
        U_target = TargetGates.hadamard()
    elif target_gate_name == 'pauli_x':
        U_target = TargetGates.pauli_x()
    elif target_gate_name == 'pauli_y':
        U_target = TargetGates.pauli_y()
    elif target_gate_name == 'pauli_z':
        U_target = TargetGates.pauli_z()
    elif target_gate_name == 'cnot':
        U_target = TargetGates.cnot()
    else:
        raise ValueError(f"Unknown target gate: {target_gate_name}")

    # Initial state (|0...0⟩)
    d = 2 ** num_qubits
    ket_0 = np.zeros(d, dtype=complex)
    ket_0[0] = 1.0

    # Target state: U|0...0⟩
    target_ket = U_target @ ket_0
    target_state = np.outer(target_ket, target_ket.conj())

    return target_state, U_target


# Factory function
def create_quantum_environment(config: dict, target_state: np.ndarray = None, target_unitary: np.ndarray = None) -> QuantumEnvironment:
    """
    Create quantum environment from config.

    Args:
        config: Configuration dictionary
        target_state: Target density matrix. If None, will be created from config['target_gate']
        target_unitary: Target unitary gate. If None, will be created from config['target_gate']

    Returns:
        env: QuantumEnvironment instance
    """
    from metaqctrl.quantum.noise_adapter import NoisePSDModel, PSDToLindblad, estimate_qubit_frequency_from_hamiltonian, get_coupling_for_noise_type

    # Get number of qubits from config
    num_qubits = config.get('num_qubits', 1)

    # Get target state and unitary if not provided
    if target_state is None or target_unitary is None:
        target_state, target_unitary = get_target_state_from_config(config)

    if num_qubits == 1:
        # 1-qubit system (original code)
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

        # System Hamiltonians
        # FIXED: Use small non-zero drift for better dynamics
        drift_strength = config.get('drift_strength', 0.1)
        H0 = drift_strength * sigma_z
        H_controls = [sigma_x, sigma_y]

        # Noise basis operators
        basis_operators = [sigma_x, sigma_y, sigma_z]

    elif num_qubits == 2:
        # 2-qubit system (NEW)
        from metaqctrl.quantum.two_qubit_gates import (
            get_two_qubit_control_hamiltonians,
            get_two_qubit_noise_operators,
            single_qubit_on_two_qubit
        )

        # CRITICAL FIX: Use proper 2-qubit control Hamiltonians
        # These include entangling ZZ interaction needed for CNOT
        H_controls = get_two_qubit_control_hamiltonians()  # [XI, IX, YI, ZZ]

        # CRITICAL FIX: Non-zero drift Hamiltonian for 2-qubit
        # Use ZZ interaction as drift (common in superconducting qubits)
        drift_strength = config.get('drift_strength', 0.5)  # Stronger for 2-qubit
        I = np.eye(2, dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        ZZ = np.kron(Z, Z)  # ZZ interaction
        H0 = drift_strength * ZZ

        # Get noise operators for both qubits
        basis_operators = get_two_qubit_noise_operators(qubit=None)  # Both qubits

    else:
        raise ValueError(f"num_qubits={num_qubits} not supported. Use 1 or 2.")

    # PSD model - NEW: Support for mixed models
    # If config specifies multiple model_types, use dynamic model selection (psd_model=None)
    # Otherwise, create fixed model for backward compatibility
    model_types = config.get('model_types', None)
    if model_types is None:
        # Single model mode (backward compatibility)
        psd_model = NoisePSDModel(model_type=config.get('psd_model', 'one_over_f'))
    else:
        # Mixed model mode - use dynamic selection
        psd_model = None
        print(f"INFO: Mixed model mode enabled with models: {model_types}")

    # Sampling frequencies (control bandwidth)
    n_segments = config.get('n_segments', 20)
    T = config.get('horizon', 1.0)
    omega_max = n_segments / T
    omega_sample = np.linspace(0, omega_max, 10)

    # Physics parameters for v2 (with sensible defaults)
    # Estimate qubit frequency from Hamiltonian if not provided
    omega0 = config.get('omega0', None)
    if omega0 is None:
        omega0 = estimate_qubit_frequency_from_hamiltonian(H0)

    # Get coupling constant for noise type
    noise_type = config.get('noise_type', 'frequency')  # 'frequency', 'magnetic', 'charge', 'amplitude'
    g_energy_per_xi = config.get('g_energy_per_xi', None)
    if g_energy_per_xi is None:
        g_energy_per_xi = get_coupling_for_noise_type(noise_type)

    # Pulse sequence for dephasing filter function
    sequence = config.get('sequence', 'ramsey')  # 'ramsey', 'echo', 'cpmg_n'

    # Temperature (None = classical noise, Γ↑=Γ↓)
    temperature_K = config.get('temperature_K', None)

    # Homogeneous broadening (rad/s)
    Gamma_h = config.get('Gamma_h', 0.0)

    # PSD to Lindblad converter (uses v2 physics via adapter)
    psd_to_lindblad = PSDToLindblad(
        basis_operators=basis_operators,
        sampling_freqs=omega_sample,
        psd_model=psd_model,  # Can be None for dynamic model selection
        T=T,
        sequence=sequence,
        omega0=omega0,
        g_energy_per_xi=g_energy_per_xi,
        temperature_K=temperature_K,
        Gamma_h=Gamma_h
    )


    # Create environment
    env = QuantumEnvironment(
        H0=H0,
        H_controls=H_controls,
        psd_to_lindblad=psd_to_lindblad,
        target_state=target_state,
        T=T,
        method=config.get('integration_method', 'RK45'),
        target_unitary=target_unitary
    )

    return env
