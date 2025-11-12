"""
Test that pre and post adaptation fidelities are tracked correctly.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from metaqctrl.meta_rl.maml import MAML
from metaqctrl.meta_rl.policy import PulsePolicy

def test_fidelity_tracking():
    """Test that both pre and post adaptation fidelities are tracked."""
    print("Testing pre/post adaptation fidelity tracking...")
    print("=" * 70)

    device = torch.device('cpu')

    # Create simple policy
    policy = PulsePolicy(
        task_feature_dim=3,
        hidden_dim=32,
        n_segments=20,
        n_controls=2
    ).to(device)

    # Create MAML with first-order (simpler to test)
    maml = MAML(
        policy=policy,
        inner_lr=0.01,
        inner_steps=3,
        meta_lr=0.001,
        first_order=True,
        device=device
    )

    # Simple dummy loss function (returns value between 0 and 1)
    def loss_fn(policy_model, data):
        """Dummy loss that mimics 1 - fidelity."""
        task_features = data['task_features']
        controls = policy_model(task_features)
        # Return a value between 0 and 1
        loss = torch.sigmoid(torch.mean(controls ** 2))
        return loss

    # Create dummy task batch
    task_batch = []
    for _ in range(4):
        task_batch.append({
            'support': {'task_features': torch.randn(10, 3)},
            'query': {'task_features': torch.randn(10, 3)}
        })

    # Perform meta-training step
    print("\nPerforming meta-training step...")
    metrics = maml.meta_train_step(task_batch, loss_fn, use_higher=False)

    print(f"\nMetrics returned:")
    print(f"  Pre-adapt loss:  {metrics.get('mean_pre_adapt_loss', 'MISSING'):.6f}")
    print(f"  Post-adapt loss: {metrics['meta_loss']:.6f}")
    print(f"  Adaptation gain: {metrics.get('adaptation_gain', 'MISSING'):.6f}")
    print(f"  Gradient norm:   {metrics['grad_norm']:.6f}")

    # Check that pre-adaptation metrics exist
    if 'mean_pre_adapt_loss' not in metrics:
        print("\n✗ FAILURE: Pre-adaptation loss not tracked!")
        return False

    if 'adaptation_gain' not in metrics:
        print("\n✗ FAILURE: Adaptation gain not tracked!")
        return False

    pre_loss = metrics['mean_pre_adapt_loss']
    post_loss = metrics['meta_loss']
    gain = metrics['adaptation_gain']

    # Verify the calculation
    expected_gain = pre_loss - post_loss
    if abs(gain - expected_gain) > 1e-6:
        print(f"\n✗ FAILURE: Adaptation gain calculation incorrect!")
        print(f"  Expected: {expected_gain:.6f}, Got: {gain:.6f}")
        return False

    # Convert to fidelities
    pre_fidelity = 1.0 - pre_loss
    post_fidelity = 1.0 - post_loss

    print(f"\nConverted to fidelities:")
    print(f"  Pre-adapt fidelity:  {pre_fidelity:.6f}")
    print(f"  Post-adapt fidelity: {post_fidelity:.6f}")
    print(f"  Fidelity improvement: {post_fidelity - pre_fidelity:.6f}")

    print("\n" + "=" * 70)
    print("✓ SUCCESS: Pre and post adaptation fidelities are tracked correctly!")
    return True

if __name__ == "__main__":
    success = test_fidelity_tracking()
    sys.exit(0 if success else 1)
