"""
Test script to verify the fidelity computation fix.

This script tests both the old (incorrect) and new (correct) fidelity formulas
to demonstrate the bug and its fix.
"""

import torch
import numpy as np

print("=" * 70)
print("FIDELITY COMPUTATION TEST")
print("=" * 70)

def fidelity_old(rho: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """OLD (INCORRECT) formula: |tr(ρσ)|²"""
    overlap = torch.trace(rho @ sigma)
    fidelity = torch.abs(overlap) ** 2
    return torch.clamp(fidelity.real, 0.0, 1.0)

def fidelity_new(rho: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """NEW (CORRECT) formula: [tr(√(√ρ σ √ρ))]²"""
    # Compute sqrt(rho)
    eigvals_rho, eigvecs_rho = torch.linalg.eigh(rho)
    eigvals_rho = torch.clamp(eigvals_rho.real, min=1e-12)
    sqrt_eigvals_rho = torch.sqrt(eigvals_rho)
    sqrt_rho = eigvecs_rho @ torch.diag(sqrt_eigvals_rho.to(torch.complex64)) @ eigvecs_rho.conj().T

    # Compute M = sqrt(rho) @ sigma @ sqrt(rho)
    M = sqrt_rho @ sigma @ sqrt_rho

    # Compute eigenvalues of M
    eigvals_M, _ = torch.linalg.eigh(M)
    eigvals_M = torch.clamp(eigvals_M.real, min=0.0)

    # F = [sum(sqrt(λ_i))]²
    fidelity = torch.sum(torch.sqrt(eigvals_M)) ** 2
    return torch.clamp(fidelity.real, 0.0, 1.0)

# Test 1: Pure states (identical)
print("\nTest 1: Identical pure states |0⟩⟨0|")
print("-" * 70)
rho_pure = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.complex64)
sigma_pure = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.complex64)

fid_old = fidelity_old(rho_pure, sigma_pure)
fid_new = fidelity_new(rho_pure, sigma_pure)

print(f"Old formula: F = {fid_old.item():.6f}")
print(f"New formula: F = {fid_new.item():.6f}")
print(f"Expected:    F = 1.000000")
print(f"✓ Both correct for pure states" if abs(fid_old - 1.0) < 1e-6 and abs(fid_new - 1.0) < 1e-6 else "✗ Error!")

# Test 2: Mixed states (identical)
print("\nTest 2: Identical mixed states (maximally mixed)")
print("-" * 70)
rho_mixed = torch.tensor([[0.5, 0.0], [0.0, 0.5]], dtype=torch.complex64)
sigma_mixed = torch.tensor([[0.5, 0.0], [0.0, 0.5]], dtype=torch.complex64)

fid_old = fidelity_old(rho_mixed, sigma_mixed)
fid_new = fidelity_new(rho_mixed, sigma_mixed)

print(f"Old formula: F = {fid_old.item():.6f}")
print(f"New formula: F = {fid_new.item():.6f}")
print(f"Expected:    F = 1.000000")
print(f"✓ New formula correct!" if abs(fid_new - 1.0) < 1e-6 else "✗ New formula error!")
print(f"✗ OLD FORMULA BUG: Should be 1.0, got {fid_old.item():.6f}" if abs(fid_old - 1.0) > 1e-3 else "")

# Test 3: Orthogonal states
print("\nTest 3: Orthogonal pure states |0⟩⟨0| vs |1⟩⟨1|")
print("-" * 70)
rho_0 = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.complex64)
rho_1 = torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=torch.complex64)

fid_old = fidelity_old(rho_0, rho_1)
fid_new = fidelity_new(rho_0, rho_1)

print(f"Old formula: F = {fid_old.item():.6f}")
print(f"New formula: F = {fid_new.item():.6f}")
print(f"Expected:    F = 0.000000")
print(f"✓ Both correct for orthogonal states" if abs(fid_old) < 1e-6 and abs(fid_new) < 1e-6 else "✗ Error!")

# Test 4: Partially mixed state
print("\nTest 4: Partially mixed state (80% |0⟩, 20% |1⟩) vs pure |0⟩")
print("-" * 70)
rho_partial = torch.tensor([[0.8, 0.0], [0.0, 0.2]], dtype=torch.complex64)
rho_0 = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.complex64)

fid_old = fidelity_old(rho_partial, rho_0)
fid_new = fidelity_new(rho_partial, rho_0)

# True fidelity for this case can be computed analytically
# F = (√0.8 + √0.0)² = 0.8
expected = 0.8

print(f"Old formula: F = {fid_old.item():.6f}")
print(f"New formula: F = {fid_new.item():.6f}")
print(f"Expected:    F ≈ {expected:.6f}")
print(f"✓ New formula correct!" if abs(fid_new - expected) < 1e-5 else f"✗ New formula error! (diff: {abs(fid_new - expected):.6f})")
print(f"✗ OLD FORMULA BUG: Off by {abs(fid_old - expected):.6f}" if abs(fid_old - expected) > 1e-3 else "")

# Test 5: Random mixed states
print("\nTest 5: Random mixed states")
print("-" * 70)

# Generate random density matrix
np.random.seed(42)
M = np.random.randn(2, 2) + 1j * np.random.randn(2, 2)
rho_rand = M @ M.conj().T
rho_rand = rho_rand / np.trace(rho_rand)
rho_rand_torch = torch.from_numpy(rho_rand).to(torch.complex64)

N = np.random.randn(2, 2) + 1j * np.random.randn(2, 2)
sigma_rand = N @ N.conj().T
sigma_rand = sigma_rand / np.trace(sigma_rand)
sigma_rand_torch = torch.from_numpy(sigma_rand).to(torch.complex64)

fid_old = fidelity_old(rho_rand_torch, sigma_rand_torch)
fid_new = fidelity_new(rho_rand_torch, sigma_rand_torch)

print(f"Old formula: F = {fid_old.item():.6f}")
print(f"New formula: F = {fid_new.item():.6f}")
print(f"Fidelity should be in [0, 1]: {0 <= fid_new <= 1}")
print(f"Note: Old and new formulas differ by {abs(fid_old - fid_new).item():.6f}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("The OLD formula |tr(ρσ)|² is ONLY correct for pure states.")
print("For mixed states (which occur with noise), it gives WRONG gradients.")
print("The NEW formula [tr(√(√ρ σ √ρ))]² is the correct quantum fidelity.")
print("\nThis bug was causing incorrect optimization in your meta-RL training!")
print("=" * 70)
