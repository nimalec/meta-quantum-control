# Noise Models Migration Guide: v1 → v2

## Overview

The noise modeling system has been upgraded from `noise_models.py` (v1) to `noise_models_v2.py` (v2) with physics-correct implementations. A **backward-compatible adapter** (`noise_adapter.py`) ensures all existing code continues to work while benefiting from improved physics.

### Key Physics Corrections

1. **Dephasing now uses filter functions**:
   - `F_Ramsey = 4 sin²(ωT/2)`
   - `F_Echo = 8 sin⁴(ωT/4)` (suppresses low-frequency noise)
   - `F_CPMG_n` (higher-order suppression)

2. **Relaxation at proper frequency**:
   - Evaluated at qubit transition ω₀, not arbitrary sampling frequencies

3. **Physical coupling constants**:
   - Frequency noise: `g = ℏ/2` for H_int = (ℏ/2) δω σ_z
   - Magnetic field: `g = g_e μ_B / 2` for H_int = (g_e μ_B / 2) B σ_z

## Migration Status: ✅ COMPLETE


## For Users: No Code Changes Required! 🎉

Your existing code will continue to work **without any modifications**:

```python
# This still works exactly as before!
from metaqctrl.quantum.noise_adapter import (
    NoiseParameters, NoisePSDModel, PSDToLindblad, TaskDistribution
)

converter = PSDToLindblad(basis_operators, sampling_freqs, psd_model)
L_ops = converter.get_lindblad_operators(theta)
```

The adapter automatically uses v2 physics underneath with sensible defaults.

## Using New Physics Features (Optional)

### 1. Specify Physics Parameters

Add to your config file:

```yaml
# experiment_config.yaml
horizon: 50e-6  # 50 μs gate time

# Physics parameters (optional, have sensible defaults)
omega0: 31.4e6  # 5 MHz qubit (rad/s)
sequence: 'ramsey'  # or 'echo', 'cpmg_4', etc.
noise_type: 'frequency'  # or 'magnetic', 'charge', 'amplitude'
drift_strength: 0.1  # Small drift for better dynamics

# Advanced (usually not needed)
temperature_K: null  # None = classical (Γ↑=Γ↓)
Gamma_h: 0.0  # Homogeneous broadening (rad/s)
g_energy_per_xi: null  # Auto-determined from noise_type
```

### 2. Programmatic Control

```python
from metaqctrl.quantum.noise_adapter import PSDToLindblad

converter = PSDToLindblad(
    basis_operators,
    sampling_freqs,
    psd_model,
    # New physics parameters
    T=100e-6,  # 100 μs gate time
    omega0=2*np.pi*10e6,  # 10 MHz qubit
    sequence='echo',  # Use echo sequence
    g_energy_per_xi=HBAR/2,  # Frequency noise coupling
    temperature_K=None,  # Classical noise
    Gamma_h=0.0  # No homogeneous broadening
)
```

### 3. Dynamic Updates

```python
# Change parameters after creation
converter.update_physics_parameters(
    T=200e-6,  # Longer gate time
    sequence='cpmg_4'  # Try CPMG-4 sequence
)
```

### 4. Get Detailed Rates

```python
rates = converter.get_rates_dict(theta)
print(f"Relaxation down: {rates['Gamma_down']:.3e} rad/s")
print(f"Relaxation up:   {rates['Gamma_up']:.3e} rad/s")
print(f"Dephasing:       {rates['gamma_phi']:.3e} rad/s")
```

## Default Physics Parameters

When not specified, the adapter uses these sensible defaults:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `T` | 50 μs | Typical superconducting qubit gate time |
| `omega0` | Estimated from H₀ | Qubit transition frequency |
| `sequence` | 'ramsey' | Free evolution (worst-case dephasing) |
| `g_energy_per_xi` | ℏ/2 | Frequency noise coupling |
| `temperature_K` | None | Classical noise (Γ↑=Γ↓) |
| `Gamma_h` | 0.0 | No homogeneous broadening |

These defaults work for most use cases!

## Testing

Run the comprehensive test suite:

```bash
cd /Users/nimalec/Documents/metarl_project/meta-quantum-control
python tests/test_noise_migration.py
```

Or with pytest:

```bash
pytest tests/test_noise_migration.py -v
```

Tests verify:
- Backward compatibility (old API works)
- Physics correctness (echo suppresses dephasing, etc.)
- Integration (QuantumEnvironment, caching, etc.)
- No breaking changes

## Example: Training with New Physics

Your existing training script needs **no changes**, but you can optionally add physics parameters:

```python
# experiments/train_scripts/train_meta.py
# Already updated to use adapter automatically!

# Optional: Add to your config for explicit control
config = {
    'horizon': 50e-6,  # 50 μs gates
    'omega0': 2*np.pi*5e6,  # 5 MHz qubit
    'sequence': 'ramsey',  # Can try 'echo' for robustness
    'noise_type': 'frequency',
    # ... rest of config
}
```

## Comparing v1 vs v2 Physics

The adapter includes a debug mode to compare old vs new:

```python
# Emulate v1 behavior (for comparison only)
converter_v1 = PSDToLindblad(
    basis_operators, sampling_freqs, psd_model,
    use_v2_physics=False  # Debug: use old physics
)

# Normal v2 behavior
converter_v2 = PSDToLindblad(
    basis_operators, sampling_freqs, psd_model,
    use_v2_physics=True  # Default
)

# Compare
rates_v1 = converter_v1.get_effective_rates(theta)
rates_v2 = converter_v2.get_effective_rates(theta)
print(f"Physics difference: {np.mean(rates_v2)/np.mean(rates_v1):.2f}x")
```

**Expected:** v2 rates can differ by 2-10× depending on parameters, especially for dephasing.

## Utility Functions

### Estimate Qubit Frequency

```python
from metaqctrl.quantum.noise_adapter import estimate_qubit_frequency_from_hamiltonian

H0 = 0.5 * sigma_z  # Drift Hamiltonian
omega0 = estimate_qubit_frequency_from_hamiltonian(H0)
print(f"Qubit frequency: {omega0/2/np.pi/1e6:.2f} MHz")
```

### Get Coupling for Noise Type

```python
from metaqctrl.quantum.noise_adapter import get_coupling_for_noise_type

g_freq = get_coupling_for_noise_type('frequency')  # ℏ/2
g_mag = get_coupling_for_noise_type('magnetic')    # g_e μ_B / 2
g_charge = get_coupling_for_noise_type('charge')   # e * 1μV
g_amp = get_coupling_for_noise_type('amplitude')   # ℏ/2
```

rter.omega0/2/np.pi/1e6:.2f} MHz")
```


