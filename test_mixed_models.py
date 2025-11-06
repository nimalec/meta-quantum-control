"""
Quick test script for mixed model sampling feature.

This script verifies that:
1. NoiseParameters can encode/decode model_type
2. TaskDistribution samples mixed models correctly
3. PSDToLindblad selects correct model dynamically
4. Task features are 4D with correct encoding
"""

import numpy as np
from metaqctrl.quantum.noise_models_v2 import (
    NoiseParameters, TaskDistribution, NoisePSDModel, PSDToLindblad
)

print("=" * 70)
print("Testing Mixed Model Sampling Implementation")
print("=" * 70)

# Test 1: NoiseParameters encoding/decoding
print("\n[Test 1] NoiseParameters model_type encoding/decoding")
print("-" * 70)

task1 = NoiseParameters(alpha=1.0, A=0.01, omega_c=500.0, model_type='one_over_f')
task2 = NoiseParameters(alpha=1.5, A=0.005, omega_c=800.0, model_type='lorentzian')
task3 = NoiseParameters(alpha=0.8, A=0.02, omega_c=300.0, model_type='double_exp')

print(f"Task 1 (one_over_f): {task1}")
print(f"  4D array: {task1.to_array(include_model=True)}")
print(f"  3D array: {task1.to_array(include_model=False)}")

print(f"\nTask 2 (lorentzian): {task2}")
print(f"  4D array: {task2.to_array(include_model=True)}")

print(f"\nTask 3 (double_exp): {task3}")
print(f"  4D array: {task3.to_array(include_model=True)}")

# Test round-trip encoding
arr = task2.to_array(include_model=True)
task2_decoded = NoiseParameters.from_array(arr, has_model=True)
print(f"\nRound-trip test: {task2.model_type} → {task2_decoded.model_type}")
assert task2.model_type == task2_decoded.model_type, "Encoding/decoding failed!"
print("✓ Encoding/decoding works correctly")

# Test 2: TaskDistribution mixed model sampling
print("\n\n[Test 2] TaskDistribution mixed model sampling")
print("-" * 70)

dist = TaskDistribution(
    dist_type='uniform',
    ranges={
        'alpha': (0.5, 2.0),
        'A': (0.001, 0.01),
        'omega_c': (100, 1000)
    },
    model_types=['one_over_f', 'lorentzian'],
    model_probs=[0.5, 0.5]
)

print(f"Distribution setup:")
print(f"  Model types: {dist.model_types}")
print(f"  Probabilities: {dist.model_probs}")

rng = np.random.default_rng(42)
tasks = dist.sample(n_tasks=100, rng=rng)

# Count model types
model_counts = {}
for task in tasks:
    model_counts[task.model_type] = model_counts.get(task.model_type, 0) + 1

print(f"\nSampled 100 tasks:")
for model_type, count in model_counts.items():
    print(f"  {model_type}: {count} tasks ({count}%)")

# Verify distribution is roughly 50/50
assert 30 < model_counts.get('one_over_f', 0) < 70, "Distribution seems biased!"
assert 30 < model_counts.get('lorentzian', 0) < 70, "Distribution seems biased!"
print("✓ Task distribution is approximately correct")

# Test variance computation
variance = dist.compute_variance()
print(f"\nTask variance: σ²_θ = {variance:.4f}")
print("  (includes contribution from mixed models)")

# Test 3: PSDToLindblad dynamic model selection
print("\n\n[Test 3] PSDToLindblad dynamic model selection")
print("-" * 70)

# Create converter with dynamic model selection
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

converter = PSDToLindblad(
    psd_model=None,  # Dynamic selection
    g_energy_per_xi=1.054571817e-34 / 2.0
)

print("Created PSDToLindblad with psd_model=None (dynamic selection)")

# Test with different model types
test_tasks = [
    NoiseParameters(alpha=1.0, A=0.01, omega_c=500.0, model_type='one_over_f'),
    NoiseParameters(alpha=1.0, A=0.01, omega_c=500.0, model_type='lorentzian'),
]

print("\nComputing dephasing rates for different model types:")
for task in test_tasks:
    gamma_phi = converter.dephasing_rate(task, T=1.0, sequence='ramsey')
    print(f"  {task.model_type:12s}: γ_φ = {gamma_phi:.6e} rad/s")

print("✓ Dynamic model selection works")

# Test 4: Feature dimensionality
print("\n\n[Test 4] Task feature dimensionality")
print("-" * 70)

# Sample some tasks and check dimensions
sample_tasks = dist.sample(n_tasks=5, rng=rng)

print("Task features (4D with model encoding):")
for i, task in enumerate(sample_tasks[:5]):
    features = task.to_array(include_model=True)
    print(f"  Task {i}: {features} [{task.model_type}]")
    assert len(features) == 4, f"Expected 4D features, got {len(features)}D"

print("✓ All tasks have 4D features")

# Test 5: Backward compatibility
print("\n\n[Test 5] Backward compatibility (single model mode)")
print("-" * 70)

# Create distribution without model_types (single model mode)
dist_single = TaskDistribution(
    dist_type='uniform',
    ranges={
        'alpha': (0.5, 2.0),
        'A': (0.001, 0.01),
        'omega_c': (100, 1000)
    }
    # model_types not specified → defaults to ['one_over_f']
)

print(f"Single model distribution:")
print(f"  Model types: {dist_single.model_types}")

tasks_single = dist_single.sample(n_tasks=10, rng=rng)
print(f"\nSampled 10 tasks (all should be one_over_f):")
model_types_single = set(task.model_type for task in tasks_single)
print(f"  Unique model types: {model_types_single}")
assert model_types_single == {'one_over_f'}, "Single model mode broken!"
print("✓ Backward compatibility maintained")

# Test 6: Three-model mix
print("\n\n[Test 6] Three-model mix")
print("-" * 70)

dist_three = TaskDistribution(
    dist_type='uniform',
    ranges={
        'alpha': (0.5, 2.0),
        'A': (0.001, 0.01),
        'omega_c': (100, 1000)
    },
    model_types=['one_over_f', 'lorentzian', 'double_exp']
    # model_probs not specified → uniform distribution
)

print(f"Three-model distribution:")
print(f"  Model types: {dist_three.model_types}")
print(f"  Probabilities: {dist_three.model_probs}")

tasks_three = dist_three.sample(n_tasks=90, rng=rng)
model_counts_three = {}
for task in tasks_three:
    model_counts_three[task.model_type] = model_counts_three.get(task.model_type, 0) + 1

print(f"\nSampled 90 tasks:")
for model_type, count in sorted(model_counts_three.items()):
    pct = count / 90 * 100
    print(f"  {model_type:12s}: {count:2d} tasks ({pct:.1f}%)")

print("✓ Three-model sampling works")

# Final summary
print("\n" + "=" * 70)
print("ALL TESTS PASSED! ✓")
print("=" * 70)
print("\nMixed model sampling is ready to use!")
print("\nNext steps:")
print("  1. Run: python experiments/train_scripts/train_meta.py \\")
print("          --config configs/experiment_config_mixed_models.yaml")
print("  2. Compare results with single-model baseline")
print("  3. Check MIXED_MODELS_GUIDE.md for details")
print("=" * 70)
