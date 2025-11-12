"""
Quick test to verify second-order MAML produces non-zero gradients.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from metaqctrl.meta_rl.maml import MAML
from metaqctrl.meta_rl.policy import PulsePolicy

def test_second_order_gradients():
    """Test that second-order MAML produces non-zero gradients."""
    print("Testing second-order MAML gradient flow...")
    print("=" * 70)

    device = torch.device('cpu')

    # Create simple policy
    policy = PulsePolicy(
        task_feature_dim=3,
        hidden_dim=32,
        n_segments=20,
        n_controls=2
    ).to(device)

    # Create MAML with second-order
    maml = MAML(
        policy=policy,
        inner_lr=0.01,
        inner_steps=3,
        meta_lr=0.001,
        first_order=False,  # IMPORTANT: Use second-order
        device=device
    )

    # Simple dummy loss function
    def loss_fn(policy_model, data):
        """Dummy loss: minimize control magnitude."""
        task_features = data['task_features']
        controls = policy_model(task_features)
        return torch.mean(controls ** 2)

    # Create dummy task batch
    task_batch = []
    for _ in range(4):
        task_batch.append({
            'support': {'task_features': torch.randn(10, 3)},
            'query': {'task_features': torch.randn(10, 3)}
        })

    # Perform meta-training step
    print("\nPerforming meta-training step with second-order MAML...")
    metrics = maml.meta_train_step(task_batch, loss_fn, use_higher=True)

    print(f"\nResults:")
    print(f"  Meta loss: {metrics['meta_loss']:.6f}")
    print(f"  Gradient norm: {metrics['grad_norm']:.6f}")

    # Check gradients
    print("\nChecking gradients for each parameter:")
    zero_grad_count = 0
    total_params = 0

    for name, param in policy.named_parameters():
        total_params += 1
        if param.grad is None:
            print(f"  ✗ {name}: NO GRADIENT")
            zero_grad_count += 1
        else:
            grad_norm = param.grad.norm().item()
            if grad_norm < 1e-10:
                print(f"  ✗ {name}: ZERO gradient (norm={grad_norm:.2e})")
                zero_grad_count += 1
            else:
                print(f"  ✓ {name}: grad_norm={grad_norm:.4e}")

    print("\n" + "=" * 70)
    if zero_grad_count == 0:
        print(f"✓ SUCCESS: All {total_params} parameters have non-zero gradients!")
        print("Second-order MAML is working correctly.")
        return True
    else:
        print(f"✗ FAILURE: {zero_grad_count}/{total_params} parameters have zero/no gradients")
        print("Second-order MAML is NOT working correctly.")
        return False

if __name__ == "__main__":
    success = test_second_order_gradients()
    sys.exit(0 if success else 1)
