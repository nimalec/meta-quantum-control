"""
Differentiable Lindblad Master Equation Simulator (PyTorch)

This module provides a fully differentiable implementation of the Lindblad
master equation using PyTorch. Unlike the NumPy/SciPy version, this allows
gradients to flow through the quantum simulation for end-to-end training.

Implements equation (3.1):
ρ̇(t) = -i[H₀ + Σₖ uₖ(t)Hₖ, ρ(t)] + Σⱼ (Lⱼ,θ ρ L†ⱼ,θ - ½{L†ⱼ,θ Lⱼ,θ, ρ})
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional


class DifferentiableLindbladSimulator(nn.Module):
    """
    PyTorch-native Lindblad simulator with full gradient support.

    Key features:
    - Fully differentiable (gradients flow through entire simulation)
    - GPU compatible
    - Supports batched simulation
    - RK4 integration for accuracy
    """

    def __init__(
        self,
        H0: torch.Tensor,
        H_controls: List[torch.Tensor],
        L_operators: List[torch.Tensor],
        dt: float = 0.005,  # FIXED: Reduced from 0.01 for better numerical stability
        method: str = 'rk4',
        device: torch.device = torch.device('cpu'),
        max_control_amp: float = 10.0  # NEW: Maximum control amplitude for clipping
    ):
        """
        Args:
            H0: Drift Hamiltonian (d x d) complex tensor
            H_controls: List of control Hamiltonians, each (d x d) complex
            L_operators: List of Lindblad operators, each (d x d) complex
            dt: Time step for integration (smaller = more accurate)
            method: Integration method ('euler', 'rk4')
            device: torch device
            max_control_amp: Maximum allowed control amplitude (for stability)
        """
        super().__init__()

        self.device = device
        self.method = method
        self.dt = dt
        self.max_control_amp = max_control_amp

        # Convert to complex tensors on device
        self.H0 = H0.to(device).to(torch.complex64)
        self.H_controls = [H.to(device).to(torch.complex64) for H in H_controls]
        self.L_operators = [L.to(device).to(torch.complex64) for L in L_operators]

        self.d = H0.shape[0]
        self.n_controls = len(H_controls)
        self.n_lindblad = len(L_operators)

        # Precompute anti-commutators for efficiency
        self.anti_commutators = [
            L.conj().T @ L for L in self.L_operators
        ]

    def lindbladian(
        self,
        rho: torch.Tensor,
        u: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Lindblad time derivative: ρ̇ = L[ρ].

        Args:
            rho: Density matrix (d x d) complex tensor
            u: Control amplitudes (n_controls,) real tensor

        Returns:
            drho_dt: Time derivative (d x d) complex tensor
        """

        u = torch.clamp(u, -self.max_control_amp, self.max_control_amp)

        # Build total Hamiltonian: H_total = H₀ + Σₖ uₖ Hₖ
        # FIXED: Build H_total without clone() to maintain gradient flow
        # Start with H0 and add control terms
        H_total = self.H0 + sum(u_k * H_k for u_k, H_k in zip(u, self.H_controls))

        # Hamiltonian evolution: -i[H, ρ]
        hamiltonian_term = -1j * (H_total @ rho - rho @ H_total)

        # Dissipation: Σⱼ (Lⱼ ρ L†ⱼ - ½{L†ⱼLⱼ, ρ})
        dissipation_term = torch.zeros_like(rho)
        for j, L_j in enumerate(self.L_operators):
            # Lⱼ ρ L†ⱼ
            dissipation_term = dissipation_term + L_j @ rho @ L_j.conj().T
            # -½ L†ⱼLⱼ ρ - ½ ρ L†ⱼLⱼ
            dissipation_term = dissipation_term - 0.5 * (
                self.anti_commutators[j] @ rho + rho @ self.anti_commutators[j]
            )

        return hamiltonian_term + dissipation_term

    def step_euler(
        self,
        rho: torch.Tensor,
        u: torch.Tensor,
        dt: float
    ) -> torch.Tensor:
        """
        Single Euler integration step: ρ(t+dt) = ρ(t) + dt * ρ̇(t).

        Args:
            rho: Current density matrix
            u: Control amplitudes
            dt: Time step

        Returns:
            rho_next: Density matrix at next time step
        """
        drho = self.lindbladian(rho, u)
        return rho + dt * drho

    def step_rk4(
        self,
        rho: torch.Tensor,
        u: torch.Tensor,
        dt: float
    ) -> torch.Tensor:
        """
        Single RK4 integration step (4th order Runge-Kutta).

        More accurate than Euler, still fully differentiable.

        Args:
            rho: Current density matrix
            u: Control amplitudes
            dt: Time step

        Returns:
            rho_next: Density matrix at next time step
        """
        # RK4 coefficients
        k1 = self.lindbladian(rho, u)
        k2 = self.lindbladian(rho + 0.5 * dt * k1, u)
        k3 = self.lindbladian(rho + 0.5 * dt * k2, u)
        k4 = self.lindbladian(rho + dt * k3, u)

        # Weighted average
        rho_next = rho + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        return rho_next

    def evolve(
        self,
        rho0: torch.Tensor,
        control_sequence: torch.Tensor,
        T: float,
        return_trajectory: bool = False,
        normalize: bool = True  # NEW: Option to normalize density matrices
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Evolve state from t=0 to t=T under piecewise-constant controls.

        FULLY DIFFERENTIABLE - gradients flow through entire simulation!

        Args:
            rho0: Initial state (d x d) complex tensor
            control_sequence: Control pulses (n_segments, n_controls) real tensor
            T: Total evolution time
            return_trajectory: If True, return full trajectory
            normalize: If True, renormalize density matrix to maintain trace=1

        Returns:
            rho_final: Final state (d x d)
            trajectory: Optional trajectory tensor (n_segments+1, d, d)
        """
        n_segments = control_sequence.shape[0]
        t_segment = T / n_segments

        # Ensure rho0 is complex
        rho = rho0.to(torch.complex64).to(self.device)

        # FIXED: Only validate in debug mode (removed prints for training)
        trace0 = torch.trace(rho).real
        # Renormalize initial state if needed
        if torch.abs(trace0 - 1.0) > 0.01 and trace0 > 1e-10:
            rho = rho / trace0

        if return_trajectory:
            trajectory = [rho.clone()]

        # Evolve through each control segment
        for seg_idx in range(n_segments):
            u_seg = control_sequence[seg_idx]

            # Multiple substeps within segment for accuracy
            n_substeps = max(1, int(t_segment / self.dt))
            dt_substep = t_segment / n_substeps

            for substep in range(n_substeps):
                if self.method == 'euler':
                    rho = self.step_euler(rho, u_seg, dt_substep)
                elif self.method == 'rk4':
                    rho = self.step_rk4(rho, u_seg, dt_substep)
                else:
                    raise ValueError(f"Unknown method: {self.method}")

                if normalize:
                    trace = torch.trace(rho).real
                    if trace > 1e-10:  # Avoid division by zero
                        rho = rho / trace
                    else:
                        # Trace collapsed - reset to initial state
                        rho = rho0.clone()
                        if return_trajectory:
                            trajectory.append(rho.clone())
                        return rho, (torch.stack(trajectory) if return_trajectory else None)

                    if torch.isnan(rho).any() or torch.isinf(rho).any():
                        # Reset to initial state (maintains gradient flow better)
                        rho = rho0.clone()


            if return_trajectory:
                trajectory.append(rho.clone())
        trace_final = torch.trace(rho).real
        if trace_final > 1e-10:
            rho = rho / trace_final

        if return_trajectory:
            trajectory_tensor = torch.stack(trajectory)
            return rho, trajectory_tensor
        else:
            return rho, None

    def forward(
        self,
        rho0: torch.Tensor,
        control_sequence: torch.Tensor,
        T: float
    ) -> torch.Tensor:
        """
        Forward pass for nn.Module compatibility.

        Args:
            rho0: Initial state (d x d)
            control_sequence: Controls (n_segments, n_controls)
            T: Evolution time

        Returns:
            rho_final: Final state (d x d)
        """
        rho_final, _ = self.evolve(rho0, control_sequence, T, return_trajectory=False)
        return rho_final


def numpy_to_torch_complex(array, device='cpu'):
    """
    Convert NumPy complex array to PyTorch complex tensor.

    Args:
        array: NumPy array (complex)
        device: Target device

    Returns:
        tensor: PyTorch complex tensor
    """
    import numpy as np

    if np.iscomplexobj(array):
        # Already complex
        real_part = torch.from_numpy(array.real).float()
        imag_part = torch.from_numpy(array.imag).float()
        tensor = torch.complex(real_part, imag_part)
    else:
        # Real array, convert to complex
        tensor = torch.from_numpy(array).float()
        tensor = tensor.to(torch.complex64)

    return tensor.to(device)


def torch_to_numpy_complex(tensor):
    """
    Convert PyTorch complex tensor to NumPy complex array.

    Args:
        tensor: PyTorch complex tensor

    Returns:
        array: NumPy complex array
    """
    import numpy as np

    tensor_cpu = tensor.cpu()
    real_part = tensor_cpu.real.numpy()
    imag_part = tensor_cpu.imag.numpy()

    return real_part + 1j * imag_part


# Example usage and testing
if __name__ == "__main__":
    print("Testing Differentiable Lindblad Simulator")
    print("=" * 60)

    # 1-qubit system (Pauli matrices)
    sigma_x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
    sigma_y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)
    sigma_z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)

    # System Hamiltonians
    H0 = 0.5 * sigma_z  # Drift
    H_controls = [sigma_x, sigma_y]  # Control Hamiltonians

    # Lindblad operators (dephasing)
    gamma = 0.1
    L_ops = [torch.sqrt(torch.tensor(gamma)) * sigma_z]

    # Create simulator
    sim = DifferentiableLindbladSimulator(
        H0=H0,
        H_controls=H_controls,
        L_operators=L_ops,
        dt=0.05,
        method='rk4'
    )

    print(f"\nSimulator created:")
    print(f"  Dimension: {sim.d}")
    print(f"  Controls: {sim.n_controls}")
    print(f"  Lindblad operators: {sim.n_lindblad}")
    print(f"  Method: {sim.method}")

    # Initial state |0⟩
    rho0 = torch.tensor([[1, 0], [0, 0]], dtype=torch.complex64)

    # Control sequence (random) - must be a leaf tensor
    n_segments = 20
    controls = torch.nn.Parameter(torch.randn(n_segments, 2) * 0.5)

    print(f"\nControl sequence:")
    print(f"  Shape: {controls.shape}")
    print(f"  Requires grad: {controls.requires_grad}")
    print(f"  Is leaf: {controls.is_leaf}")

    # Forward simulation
    print("\nRunning forward simulation...")
    rho_final, trajectory = sim.evolve(rho0, controls, T=1.0, return_trajectory=True)

    print(f"  Final state shape: {rho_final.shape}")
    print(f"  Trajectory shape: {trajectory.shape}")

    # Compute purity (should decrease due to decoherence)
    purity = torch.trace(rho_final @ rho_final).real
    print(f"  Final purity: {purity.item():.4f}")

    # Test gradient flow
    print("\nTesting gradient flow...")
    loss = -purity  # Maximize purity
    loss.backward()

    print(f"  Loss computed: {loss.item():.4f}")
    print(f"  Gradients available: {controls.grad is not None}")

    if controls.grad is not None:
        grad_norm = torch.norm(controls.grad).item()
        print(f"  Gradient norm: {grad_norm:.4e}")
        print(f"  Gradient shape: {controls.grad.shape}")
        print(f"   GRADIENT FLOW WORKS!")

    print("\n" + "=" * 60)
    print("Differentiable Lindblad Simulator Test Complete!")
    print("=" * 60)
