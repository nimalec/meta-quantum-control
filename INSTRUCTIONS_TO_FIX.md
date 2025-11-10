# Instructions to Fix MAML Gradient Error

## Step 1: Copy the diagnostic script to YOUR codebase

Copy this file to your actual codebase:
```bash
# From: (my path - this won't work for you)
# /Users/nimalec/Documents/metarl_project/meta-quantum-control/diagnose_gradients.py

# To: (your actual path)
"/Users/nleclerc/Documents/Projects/RL4Quantum Control /code/meta-quantum-control/diagnose_gradients.py"
```

**OR** just create a new file `diagnose_gradients.py` in your project root and copy the code from my version.

---

## Step 2: Run the diagnostic

```bash
cd "/Users/nleclerc/Documents/Projects/RL4Quantum Control /code/meta-quantum-control"
python diagnose_gradients.py
```

This will test gradient flow step-by-step and tell you exactly where it breaks.

---

## Step 3: Interpret the results

### If Test 6 fails (Loss has no gradient connection):

**Problem:** `compute_loss_differentiable()` is not returning a differentiable tensor.

**Solution:** Check `metaqctrl/theory/quantum_environment.py` around line 420-466 for:
- Any `.detach()` calls
- Any `.item()` calls
- Any conversion to numpy and back

**Common culprits:**
```python
# BAD - breaks gradients:
controls_np = controls.detach().cpu().numpy()

# BAD - breaks gradients:
loss = torch.tensor(value, requires_grad=False)

# GOOD - maintains gradients:
loss = 1.0 - fidelity  # where fidelity is a torch tensor with grad_fn
```

### If Test 7 fails (Backward pass fails):

**Problem:** Gradients exist but can't backpropagate.

**Solution:** Check for:
- In-place operations that break gradient flow
- Complex number gradient issues
- NaN/Inf values

### If Test 8 fails (MAML inner loop fails):

**Problem:** The inner loop can't run even though single backward pass works.

**Solution:** This is the original error. Apply the MAML fix from `FIX_GRADIENT_ERROR.md`

---

## Step 4: Apply the MAML fix

**File:** `metaqctrl/meta_rl/maml.py`

**Location:** Around line 320-395 in the `meta_train_step()` method

**Find this block:**
```python
else:
    # First-order MAML - use manual gradient computation
    use_manual_grads = True

    if HIGHER_AVAILABLE:
        # Use higher library for inner loop
        fmodel, inner_losses = self.inner_loop_higher(task_data, loss_fn)

        # ... rest of higher library code ...
```

**Replace the entire else block with:**
```python
else:
    # First-order MAML - use manual gradient computation
    use_manual_grads = True

    # CRITICAL FIX: Always use plain inner_loop for first-order MAML
    # The higher library requires gradients even for first-order mode
    adapted_policy, inner_losses = self.inner_loop(task_data, loss_fn)

    # Evaluate on query set
    query_loss = loss_fn(adapted_policy, task_data['query'])

    # Compute gradients for meta-update
    meta_params = list(self.policy.parameters())
    adapted_params = list(adapted_policy.parameters())

    adapted_grads = autograd.grad(
        query_loss,
        adapted_params,
        create_graph=False,  # First-order: no second derivatives
        allow_unused=True
    )

    # Apply gradients to meta-parameters
    for meta_param, adapted_grad in zip(meta_params, adapted_grads):
        if adapted_grad is not None:
            if meta_param.grad is None:
                meta_param.grad = adapted_grad.clone()
            else:
                meta_param.grad += adapted_grad.clone()
```

---

## Step 5: Verify config

**File:** `configs/experiment_config.yaml`

Make sure you have:
```yaml
first_order: true  # Must be true, not false
```

---

## Step 6: Clear cache and test

```bash
cd "/Users/nleclerc/Documents/Projects/RL4Quantum Control /code/meta-quantum-control"

# Clear Python cache
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete

# Run training
python experiments/train_scripts/train_meta.py --config configs/experiment_config.yaml
```

---

## Expected Output

You should see:
```
[MAML] Running in First-Order (FOMAML) mode
  self.first_order: True
  use_higher: True/False
  HIGHER_AVAILABLE: True/False
```

Training should proceed without the gradient error.

---

## If It Still Fails

1. Run the diagnostic again to see which test fails
2. Check the error message - it should now be more informative
3. Share:
   - Which diagnostic test failed (1-8)
   - The error message
   - Whether loss has grad_fn in test 6

---

## Quick Summary

**Problem:** MAML can't get gradients from the loss function

**Root Causes (one of these):**
1. ❌ `compute_loss_differentiable()` doesn't return differentiable tensor
2. ❌ First-order MAML using `inner_loop_higher()` which needs gradients
3. ❌ Config has `first_order: false` instead of `first_order: true`

**Solutions:**
1. ✅ Fix `compute_loss_differentiable()` to maintain gradient flow
2. ✅ Use plain `inner_loop()` for first-order MAML
3. ✅ Set `first_order: true` in config

**The diagnostic will tell you which one is the issue.**
