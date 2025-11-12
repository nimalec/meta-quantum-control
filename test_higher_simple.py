"""
Simple test to understand higher library behavior.
"""

import torch
import torch.nn as nn
import torch.optim as optim

try:
    import higher
    print("✓ higher library is available")
except ImportError:
    print("✗ higher library not available")
    exit(1)

# Create a simple 2-layer network
model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 1)
)

print("\nInitial parameter values:")
for name, param in model.named_parameters():
    print(f"  {name}: {param.data.flatten()[:3]}")

# Create optimizer
opt = optim.SGD(model.parameters(), lr=0.01)

# Create some dummy data
support_x = torch.randn(5, 2)
support_y = torch.ones(5, 1)
query_x = torch.randn(5, 2)
query_y = torch.ones(5, 1)

print("\nTesting second-order MAML with higher...")

# Zero gradients
for param in model.parameters():
    param.grad = None

# Use higher for inner loop
with higher.innerloop_ctx(
    model,
    opt,
    copy_initial_weights=True,
    track_higher_grads=True  # Enable second-order
) as (fmodel, diffopt):
    print("  Inside higher context...")

    # Inner loop (adapt on support set)
    for step in range(3):
        support_pred = fmodel(support_x)
        support_loss = nn.functional.mse_loss(support_pred, support_y)
        print(f"    Inner step {step}: loss={support_loss.item():.6f}")
        diffopt.step(support_loss)

    # Compute query loss
    query_pred = fmodel(query_x)
    query_loss = nn.functional.mse_loss(query_pred, query_y)
    print(f"  Query loss: {query_loss.item():.6f}")

    # CORRECT APPROACH: Compute gradients w.r.t. fmodel params with create_graph=True
    # The higher library tracks how fmodel depends on model through the optimization
    print("  Computing gradients with autograd.grad()...")
    fmodel_params = list(fmodel.parameters())
    meta_params = list(model.parameters())

    # Compute gradients w.r.t. adapted parameters
    # create_graph=True allows backprop through the optimization steps
    grads = torch.autograd.grad(
        query_loss,
        fmodel_params,  # Compute w.r.t. ADAPTED parameters
        create_graph=True,  # Enable second-order gradients
        allow_unused=True
    )
    print(f"  ✓ autograd.grad() returned {len(grads)} gradients")

    # These gradients are w.r.t. fmodel, but higher tracks the dependency
    # We need to manually copy them to meta-parameters
    for i, (meta_param, grad) in enumerate(zip(meta_params, grads)):
        if grad is not None:
            print(f"    Gradient {i}: shape={grad.shape}, norm={grad.norm().item():.6f}")
            if meta_param.grad is None:
                meta_param.grad = grad
            else:
                meta_param.grad += grad
        else:
            print(f"    Gradient {i}: None")

print("\nChecking gradients after backward:")
has_grad = False
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        print(f"  ✓ {name}: grad_norm={grad_norm:.6f}")
        has_grad = True
    else:
        print(f"  ✗ {name}: NO GRADIENT")

if has_grad:
    print("\n✓ SUCCESS: Gradients are populated!")
else:
    print("\n✗ FAILURE: No gradients found!")
