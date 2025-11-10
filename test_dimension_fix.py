"""
Verify that the dimension mismatch is fixed.
"""

import torch
import numpy as np
import yaml
from pathlib import Path

from metaqctrl.meta_rl.policy import PulsePolicy
from metaqctrl.quantum.noise_adapter import NoiseParameters

def test_dimension_fix():
    """Test that policy input dimensions match task features."""

    print("=" * 70)
    print("Testing Dimension Fix")
    print("=" * 70)

    # Load config
    config_path = Path(__file__).parent / "configs" / "experiment_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"\nConfig task_feature_dim: {config.get('task_feature_dim')}")

    # Create policy
    policy = PulsePolicy(
        task_feature_dim=config.get('task_feature_dim'),
        hidden_dim=config.get('hidden_dim'),
        n_hidden_layers=config.get('n_hidden_layers'),
        n_segments=config.get('n_segments'),
        n_controls=config.get('n_controls'),
        output_scale=config.get('output_scale'),
        activation=config.get('activation')
    )

    print(f"Policy task_feature_dim: {policy.task_feature_dim}")

    # Create task features as done in train_meta.py
    task_params = NoiseParameters(
        alpha=1.0,
        A=100.0,
        omega_c=50.0,
        model_type='one_over_f'
    )

    # Convert to array with model (as done in train_meta.py line 116)
    task_array = task_params.to_array(include_model=True)
    print(f"Task array shape: {task_array.shape}")
    print(f"Task array: {task_array}")

    # Create tensor
    task_features = torch.tensor(task_array, dtype=torch.float32)
    print(f"Task features shape: {task_features.shape}")

    # Test forward pass
    print("\nTesting forward pass...")
    try:
        controls = policy(task_features)
        print(f"✓ Forward pass successful!")
        print(f"  Controls shape: {controls.shape}")
        print(f"  Expected: ({config.get('n_segments')}, {config.get('n_controls')})")

        if controls.shape == (config.get('n_segments'), config.get('n_controls')):
            print("\n" + "=" * 70)
            print("✓ DIMENSION FIX VERIFIED!")
            print("=" * 70)
            return True
        else:
            print(f"\n✗ ERROR: Output shape mismatch!")
            return False

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    success = test_dimension_fix()
    sys.exit(0 if success else 1)
