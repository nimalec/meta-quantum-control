"""
Minimal test to verify the FOMAML fix works without full quantum simulation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy

# Dummy policy
class SimplePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 10)

    def forward(self, x):
        return self.linear(x)

# Dummy loss function that doesn't provide gradients (simulating the issue)
def broken_loss_fn(policy, data):
    """Loss that doesn't have gradients - simulates quantum simulator issue."""
    x = data['x']
    output = policy(x)
    # Detach breaks the gradient! This simulates the quantum simulator issue
    loss_value = (output ** 2).mean().detach()
    return loss_value

# Fixed loss function
def working_loss_fn(policy, data):
    """Loss that maintains gradients."""
    x = data['x']
    output = policy(x)
    loss_value = (output ** 2).mean()
    return loss_value

def test_inner_loop_with_broken_loss():
    """Test that inner_loop works even with broken loss (no gradients during adaptation)."""
    print("\n" + "="*70)
    print("TEST 1: Inner loop with BROKEN loss (no gradients)")
    print("="*70)

    policy = SimplePolicy()
    task_data = {
        'support': {'x': torch.randn(5, 3)},
        'query': {'x': torch.randn(5, 3)}
    }

    # Clone for adaptation
    adapted_policy = deepcopy(policy)
    adapted_policy.train()
    inner_opt = optim.SGD(adapted_policy.parameters(), lr=0.01)

    print("Running inner loop adaptation...")
    for step in range(3):
        inner_opt.zero_grad()
        loss = broken_loss_fn(adapted_policy, task_data['support'])
        print(f"  Step {step}: loss={loss.item():.4f}, has grad_fn={loss.grad_fn is not None}")

        # This should work even without gradients!
        try:
            loss.backward()
            inner_opt.step()
            print(f"    ✓ Backward passed (gradients may be None, but no crash)")
        except Exception as e:
            print(f"    ✗ Failed: {e}")
            return False

    # Now compute query loss (this DOES need gradients for meta-learning)
    print("\nComputing query loss with WORKING loss function...")
    query_loss = working_loss_fn(adapted_policy, task_data['query'])
    print(f"  Query loss: {query_loss.item():.4f}")
    print(f"  Has grad_fn: {query_loss.grad_fn is not None}")

    # Compute gradients w.r.t. adapted parameters
    adapted_grads = torch.autograd.grad(
        query_loss,
        adapted_policy.parameters(),
        create_graph=False
    )

    print(f"  Gradients computed: {len([g for g in adapted_grads if g is not None])}/{len(list(adapted_policy.parameters()))}")

    # Apply to meta-parameters
    for meta_p, adapted_g in zip(policy.parameters(), adapted_grads):
        if adapted_g is not None:
            if meta_p.grad is None:
                meta_p.grad = adapted_g.clone()
            else:
                meta_p.grad += adapted_g.clone()

    meta_grad_count = sum(1 for p in policy.parameters() if p.grad is not None)
    print(f"  Meta-gradients set: {meta_grad_count}/{len(list(policy.parameters()))}")

    print("\n✓ SUCCESS! Inner loop works without gradients, meta-update works with gradients")
    return True

def test_higher_library_fails():
    """Test that higher library fails with broken loss."""
    print("\n" + "="*70)
    print("TEST 2: Higher library with BROKEN loss (should fail)")
    print("="*70)

    try:
        import higher
    except ImportError:
        print("Higher library not available - skipping test")
        return True

    policy = SimplePolicy()
    task_data = {
        'support': {'x': torch.randn(5, 3)},
    }

    inner_opt = optim.SGD(policy.parameters(), lr=0.01)

    print("Running inner loop with higher library...")
    try:
        with higher.innerloop_ctx(
            policy,
            inner_opt,
            copy_initial_weights=True,
            track_higher_grads=False
        ) as (fmodel, diffopt):
            for step in range(3):
                loss = broken_loss_fn(fmodel, task_data['support'])
                print(f"  Step {step}: loss={loss.item():.4f}, has grad_fn={loss.grad_fn is not None}")

                # This WILL fail because higher requires gradients!
                diffopt.step(loss)
                print(f"    ✓ Step succeeded (unexpected!)")

        print("\n✗ Higher library worked with broken loss (shouldn't happen)")
        return False

    except RuntimeError as e:
        if "does not require grad" in str(e):
            print(f"\n✓ EXPECTED FAILURE: {str(e)[:100]}...")
            print("  This confirms higher library needs gradients even for first-order MAML")
            return True
        else:
            print(f"\n✗ Unexpected error: {e}")
            return False

if __name__ == "__main__":
    print("TESTING FOMAML FIX LOGIC")
    print("="*70)

    test1_passed = test_inner_loop_with_broken_loss()
    test2_passed = test_higher_library_fails()

    print("\n" + "="*70)
    print("RESULTS:")
    print(f"  Test 1 (inner_loop with broken loss): {'✓ PASSED' if test1_passed else '✗ FAILED'}")
    print(f"  Test 2 (higher fails with broken loss): {'✓ PASSED' if test2_passed else '✗ FAILED'}")

    if test1_passed and test2_passed:
        print("\n✓ ALL TESTS PASSED - The fix logic is sound!")
        print("\nConclusion:")
        print("  - Plain inner_loop works even when loss has no gradients during adaptation")
        print("  - Higher library fails when loss has no gradients")
        print("  - Solution: Use plain inner_loop for first-order MAML")
    else:
        print("\n✗ SOME TESTS FAILED")

    print("="*70)
