"""
QUICK DEBUG SNIPPET

Add this code to your train_meta.py RIGHT BEFORE the trainer.train() call
(around line 295-299)

This will test gradient flow before training starts.
"""

# ============================================================================
# ADD THIS CODE TO train_meta.py BEFORE trainer.train() call
# ============================================================================

print("\n" + "="*70)
print("GRADIENT FLOW PRE-FLIGHT CHECK")
print("="*70)

# Test 1: Sample a task
print("\n[1/3] Sampling test task...")
rng_test = np.random.default_rng(42)
test_task_params = task_sampler(1, 'train', task_dist, rng_test)[0]
print(f"  Task: alpha={test_task_params.alpha:.3f}, A={test_task_params.A:.3f}, omega_c={test_task_params.omega_c:.3f}")

# Test 2: Compute loss
print("\n[2/3] Testing loss computation...")
policy.train()  # CRITICAL: ensure training mode

test_loss = env.compute_loss_differentiable(
    policy,
    test_task_params,
    device,
    use_rk4=config.get('use_rk4_training'),
    dt=config.get('dt_training')
)

print(f"  Loss value: {test_loss.item():.6f}")
print(f"  Loss dtype: {test_loss.dtype}")
print(f"  Loss requires_grad: {test_loss.requires_grad}")
print(f"  Loss grad_fn: {test_loss.grad_fn}")

# CRITICAL CHECK
if test_loss.grad_fn is None and not test_loss.requires_grad:
    print("\n" + "!"*70)
    print("✗ ERROR: Loss has NO gradient connection!")
    print("!"*70)
    print("\nThis is why MAML will fail.")
    print("\nPossible fixes:")
    print("  1. Check compute_loss_differentiable() for .detach() or .item()")
    print("  2. Ensure policy is in train mode (policy.train())")
    print("  3. Check quantum simulator maintains gradient flow")
    print("\nStopping before MAML training to avoid confusing errors.")
    print("="*70 + "\n")
    import sys
    sys.exit(1)
else:
    print("  ✓ Loss has gradient connection - GOOD!")

# Test 3: Test backward
print("\n[3/3] Testing backward pass...")
policy.zero_grad()

try:
    test_loss.backward()

    grad_count = sum(1 for p in policy.parameters() if p.grad is not None and p.grad.norm() > 1e-10)
    param_count = sum(1 for _ in policy.parameters())

    print(f"  ✓ Backward pass succeeded")
    print(f"  Parameters with non-zero gradients: {grad_count}/{param_count}")

    if grad_count == 0:
        print("\n  ⚠ WARNING: All gradients are zero!")
        print("  This could mean the loss doesn't depend on policy parameters.")
    else:
        print("  ✓ Gradients are flowing - GOOD!")

except Exception as e:
    print(f"\n  ✗ Backward pass FAILED: {e}")
    print("\nStopping before MAML training.")
    import sys
    sys.exit(1)

print("\n" + "="*70)
print("✓ PRE-FLIGHT CHECK PASSED - Starting MAML training...")
print("="*70 + "\n")

# ============================================================================
# END OF DEBUG SNIPPET
# ============================================================================

# Now your trainer.train() call will execute
# trainer.train(...)
