# Fix for "element 0 of tensors does not require grad" Error

## Problem Summary

The MAML training is failing with:
```
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
```

This occurs because even with `first_order=true`, the code was using `inner_loop_higher()` which calls `diffopt.step(loss)`, requiring gradients during the inner loop adaptation.

## Root Cause

The quantum simulator's `compute_loss_differentiable()` may not maintain full gradient flow, causing `diffopt.step()` in the higher library to fail.

## Solution

Modify `metaqctrl/meta_rl/maml.py` to force first-order MAML to use the plain `inner_loop()` method instead of `inner_loop_higher()`.

---

## Required Changes

### Change 1: Force first-order MAML to use plain inner_loop

**File:** `metaqctrl/meta_rl/maml.py`

**Find this code** (around line 360-395):

```python
else:
    # First-order MAML - use manual gradient computation
    use_manual_grads = True

    if HIGHER_AVAILABLE:
        # Use higher library for inner loop
        fmodel, inner_losses = self.inner_loop_higher(task_data, loss_fn)

        # Compute query loss
        query_loss = loss_fn(fmodel, task_data['query'])

        # CRITICAL FIX FOR FOMAML:
        # Manually compute gradients and apply to meta-parameters
        # This is what makes it "first-order" - we ignore gradients through adaptation
        meta_params = list(self.policy.parameters())
        adapted_params = list(fmodel.parameters())

        # Compute gradients w.r.t adapted parameters
        adapted_grads = autograd.grad(
            query_loss,
            adapted_params,
            create_graph=False,  # First-order: don't need second derivatives
            allow_unused=True
        )

        # Apply these gradients to meta-parameters
        # (accumulate since we're batching over tasks)
        for meta_param, adapted_grad in zip(meta_params, adapted_grads):
            if adapted_grad is not None:
                if meta_param.grad is None:
                    meta_param.grad = adapted_grad.clone()
                else:
                    meta_param.grad += adapted_grad.clone()

    else:
        # Manual first-order MAML (only if higher not available)
        adapted_policy, inner_losses = self.inner_loop(task_data, loss_fn)

        # Evaluate on query set
        query_loss = loss_fn(adapted_policy, task_data['query'])

        # For FOMAML without higher: manually compute gradients
        meta_params = list(self.policy.parameters())
        adapted_params = list(adapted_policy.parameters())

        adapted_grads = autograd.grad(
            query_loss,
            adapted_params,
            create_graph=False,
            allow_unused=True
        )

        for meta_param, adapted_grad in zip(meta_params, adapted_grads):
            if adapted_grad is not None:
                if meta_param.grad is None:
                    meta_param.grad = adapted_grad.clone()
                else:
                    meta_param.grad += adapted_grad.clone()
```

**Replace with:**

```python
else:
    # First-order MAML - use manual gradient computation
    use_manual_grads = True

    # CRITICAL FIX: For first-order MAML, ALWAYS use the plain inner_loop
    # instead of inner_loop_higher, because inner_loop_higher requires
    # gradients for diffopt.step() even though we don't use them for meta-updates
    #
    # The plain inner_loop uses deepcopy and regular SGD optimizer,
    # which doesn't require the loss to have gradients during adaptation.
    # We only need gradients for the final query loss.

    # Use plain inner loop (no higher library)
    adapted_policy, inner_losses = self.inner_loop(task_data, loss_fn)

    # Evaluate on query set
    query_loss = loss_fn(adapted_policy, task_data['query'])

    # For FOMAML: manually compute gradients
    meta_params = list(self.policy.parameters())
    adapted_params = list(adapted_policy.parameters())

    adapted_grads = autograd.grad(
        query_loss,
        adapted_params,
        create_graph=False,
        allow_unused=True
    )

    for meta_param, adapted_grad in zip(meta_params, adapted_grads):
        if adapted_grad is not None:
            if meta_param.grad is None:
                meta_param.grad = adapted_grad.clone()
            else:
                meta_param.grad += adapted_grad.clone()
```

### Change 2: Verify config has first_order=true

**File:** `configs/experiment_config.yaml`

Ensure this line exists:
```yaml
first_order: true  # Use FOMAML instead of second-order MAML
```

---

## Why This Works

1. **Plain inner_loop** uses `deepcopy()` and regular PyTorch `optimizer.step()`, which doesn't require the loss to have a computational graph during adaptation

2. **Gradients only needed for query loss**: First-order MAML only needs gradients when computing the meta-gradient from the query loss, not during the inner loop adaptation

3. **Avoids higher library issues**: The `higher` library's `diffopt.step()` always requires gradients, even with `track_higher_grads=False`

---

## Testing

After making the change:

1. Clear Python cache:
   ```bash
   cd "/path/to/meta-quantum-control"
   find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
   find . -name "*.pyc" -delete
   ```

2. Run training:
   ```bash
   python experiments/train_scripts/train_meta.py --config configs/experiment_config.yaml
   ```

3. You should see:
   ```
   [MAML] Running in First-Order (FOMAML) mode
     self.first_order: True
     use_higher: True
     HIGHER_AVAILABLE: True
   ```

4. Training should proceed without the gradient error

---

## Additional Diagnostics

If you want to verify gradient flow independently, run:
```bash
python experiments/test_gradient_flow.py
```

This will test if `compute_loss_differentiable()` produces gradients correctly.

---

## What Changed

**Before:**
- First-order MAML used `inner_loop_higher()`
- This called `diffopt.step(loss)` which requires gradients
- Loss from quantum simulator doesn't have gradients → ERROR

**After:**
- First-order MAML uses `inner_loop()`
- This uses regular `optimizer.step()` which works without gradients
- Only the final query loss needs gradients → SUCCESS

---

## Performance Impact

✅ **No performance loss** - First-order MAML is actually faster than second-order MAML
✅ **Still effective** - FOMAML works well in practice (proven in literature)
✅ **More stable** - Avoids numerical issues with complex quantum simulations

---

## If Still Failing

If the error persists:

1. Check that `self.first_order = True` is printed in the MAML mode output
2. Verify the quantum simulator is being called correctly
3. Run the diagnostic script: `python experiments/test_gradient_flow.py`
4. Check that `compute_loss_differentiable()` doesn't have `.detach()` or `.item()` calls
