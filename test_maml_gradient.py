"""
Test MAML gradient flow with the higher library.
This replicates the actual MAML training step.
"""

import torch
import torch.optim as optim
from torch import autograd
import numpy as np
import yaml
from pathlib import Path

# Try to import higher
try:
    import higher
    HIGHER_AVAILABLE = True
except ImportError:
    HIGHER_AVAILABLE = False
    print("WARNING: higher library not available")

from metaqctrl.theory.quantum_environment import create_quantum_environment
from metaqctrl.meta_rl.policy import PulsePolicy
from metaqctrl.quantum.noise_adapter import NoiseParameters

def test_maml_gradient_flow():
    """Test that gradients flow through MAML inner loop."""

    print("=" * 70)
    print("Testing MAML Gradient Flow")
    print("=" * 70)

    # Load config
    config_path = Path(__file__).parent / "configs" / "experiment_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    print(f"Higher available: {HIGHER_AVAILABLE}")

    # Create environment
    print("\n1. Creating quantum environment...")
    env = create_quantum_environment(config, target_state=None)

    # Create policy
    print("\n2. Creating policy...")
    policy = PulsePolicy(
        task_feature_dim=config.get('task_feature_dim'),
        hidden_dim=config.get('hidden_dim'),
        n_hidden_layers=config.get('n_hidden_layers'),
        n_segments=config.get('n_segments'),
        n_controls=config.get('n_controls'),
        output_scale=config.get('output_scale'),
        activation=config.get('activation')
    ).to(device)
    policy.train()

    # Create loss function
    dt = config.get('dt_training', 0.01)
    use_rk4 = config.get('use_rk4_training', True)

    def loss_fn(policy_module, task_params):
        """Loss function for a single task."""
        loss = env.compute_loss_differentiable(
            policy_module,
            task_params,
            device,
            use_rk4=use_rk4,
            dt=dt
        )
        return loss

    # Create sample task
    print("\n3. Creating sample task...")
    task_params = NoiseParameters(
        alpha=1.0,
        A=100.0,
        omega_c=50.0,
        model_type='one_over_f'
    )

    # Test 1: Direct loss computation (should work)
    print("\n4. Test 1: Direct loss computation...")
    loss_direct = loss_fn(policy, task_params)
    print(f"   Loss: {loss_direct.item():.6f}")
    print(f"   Has grad_fn: {loss_direct.grad_fn is not None}")

    # Test 2: Loss after manual gradient step
    print("\n5. Test 2: Manual gradient step...")
    optimizer = optim.SGD(policy.parameters(), lr=0.01)
    optimizer.zero_grad()
    loss_direct.backward()
    optimizer.step()
    print("   ✓ Manual gradient step successful")

    # Test 3: MAML-style inner loop WITHOUT higher
    print("\n6. Test 3: Inner loop without higher...")
    from copy import deepcopy
    adapted_policy = deepcopy(policy)
    inner_opt = optim.SGD(adapted_policy.parameters(), lr=0.01)

    # Do 1 inner step
    inner_opt.zero_grad()
    loss_support = loss_fn(adapted_policy, task_params)
    print(f"   Support loss: {loss_support.item():.6f}")
    loss_support.backward()
    inner_opt.step()

    # Query loss
    loss_query = loss_fn(adapted_policy, task_params)
    print(f"   Query loss: {loss_query.item():.6f}")
    print(f"   Query has grad_fn: {loss_query.grad_fn is not None}")

    # Try to compute gradients w.r.t. adapted policy
    adapted_params = list(adapted_policy.parameters())
    try:
        query_grads = autograd.grad(
            loss_query,
            adapted_params,
            create_graph=False,
            allow_unused=True
        )
        grad_found = any(g is not None and g.abs().max() > 0 for g in query_grads)
        if grad_found:
            print("   ✓ Gradients computed successfully (without higher)")
        else:
            print("   ✗ WARNING: All gradients are None or zero")
    except Exception as e:
        print(f"   ✗ ERROR: {e}")
        return False

    # Test 4: MAML-style inner loop WITH higher
    if HIGHER_AVAILABLE:
        print("\n7. Test 4: Inner loop WITH higher...")
        inner_opt2 = optim.SGD(policy.parameters(), lr=0.01)

        with higher.innerloop_ctx(
            policy,
            inner_opt2,
            copy_initial_weights=True,
            track_higher_grads=True
        ) as (fmodel, diffopt):

            # Inner step
            loss_support = loss_fn(fmodel, task_params)
            print(f"   Support loss: {loss_support.item():.6f}")
            print(f"   Support loss grad_fn: {loss_support.grad_fn}")
            print(f"   Support loss requires_grad: {loss_support.requires_grad}")

            diffopt.step(loss_support)

            # Query loss
            loss_query = loss_fn(fmodel, task_params)
            print(f"   Query loss: {loss_query.item():.6f}")
            print(f"   Query loss grad_fn: {loss_query.grad_fn}")
            print(f"   Query loss requires_grad: {loss_query.requires_grad}")

            # Try to compute gradients
            fmodel_params = list(fmodel.parameters())
            try:
                query_grads = autograd.grad(
                    loss_query,
                    fmodel_params,
                    create_graph=True,
                    allow_unused=True
                )
                non_none_grads = sum(1 for g in query_grads if g is not None)
                total_grads = len(query_grads)
                print(f"   Non-None gradients: {non_none_grads}/{total_grads}")

                if non_none_grads == 0:
                    print("   ✗ ERROR: All gradients are None!")
                    print("   This means the loss is not connected to fmodel parameters")
                    return False
                else:
                    max_grad = max(g.abs().max().item() for g in query_grads if g is not None)
                    print(f"   Max gradient magnitude: {max_grad:.6e}")
                    if max_grad > 0:
                        print("   ✓ Gradients computed successfully (with higher)")
                    else:
                        print("   ✗ WARNING: All gradients are zero")

            except Exception as e:
                print(f"   ✗ ERROR computing gradients: {e}")
                import traceback
                traceback.print_exc()
                return False

    print("\n" + "=" * 70)
    print("✓ MAML GRADIENT FLOW TEST PASSED!")
    print("=" * 70)
    return True

if __name__ == "__main__":
    import sys
    success = test_maml_gradient_flow()
    sys.exit(0 if success else 1)
