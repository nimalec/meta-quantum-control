"""
Debug Training Script - Verbose diagnostics for gradient flow
"""

import torch
import numpy as np
import yaml
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from metaqctrl.quantum.noise_models import TaskDistribution, NoisePSDModel, PSDToLindblad, NoiseParameters
from metaqctrl.quantum.gates import TargetGates
from metaqctrl.meta_rl.policy import PulsePolicy
from metaqctrl.meta_rl.maml import MAML
from metaqctrl.theory.quantum_environment import create_quantum_environment


def main():
    # Load config
    config_path = Path(__file__).parent.parent / "configs" / "experiment_config_gpu.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("=" * 70)
    print("DEBUG: Single MAML Step with Verbose Logging")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Create environment
    ket_0 = np.array([1, 0], dtype=complex)
    U_target = TargetGates.pauli_x()
    target_state = np.outer(U_target @ ket_0, (U_target @ ket_0).conj())

    env = create_quantum_environment(config, target_state)

    # Create task distribution
    task_dist = TaskDistribution(
        dist_type='uniform',
        ranges={
            'alpha': tuple(config.get('alpha_range', [0.5, 2.0])),
            'A': tuple(config.get('A_range', [0.001, 0.01])),
            'omega_c': tuple(config.get('omega_c_range', [100, 1000]))
        }
    )

    # Create policy
    policy = PulsePolicy(
        task_feature_dim=3,
        hidden_dim=128,
        n_hidden_layers=2,
        n_segments=20,
        n_controls=2,
        output_scale=2.0
    ).to(device)

    print(f"Policy parameters: {policy.count_parameters():,}")

    # Create MAML
    maml = MAML(
        policy=policy,
        inner_lr=config.get('inner_lr', 0.01),
        inner_steps=config.get('inner_steps', 5),
        meta_lr=config.get('meta_lr', 0.001),
        first_order=config.get('first_order', False),
        device=device
    )

    print(f"\nMAML Configuration:")
    print(f"  Inner LR: {maml.inner_lr}")
    print(f"  Inner steps: {maml.inner_steps}")
    print(f"  Meta LR: {maml.meta_lr}")
    print(f"  First-order: {maml.first_order}")

    # Sample a few tasks
    rng = np.random.default_rng(42)
    tasks = task_dist.sample(3, rng)

    # Loss function
    dt = config.get('dt_training', 0.02)
    use_rk4 = config.get('use_rk4_training', False)

    def loss_fn(pol, data):
        task_params = data['task_params']
        return env.compute_loss_differentiable(pol, task_params, device, use_rk4=use_rk4, dt=dt)

    # Generate task batch
    print(f"\n" + "=" * 70)
    print("Generating task batch...")
    print("=" * 70)

    task_batch = []
    for i, task_params in enumerate(tasks):
        print(f"\nTask {i}: α={task_params.alpha:.2f}, A={task_params.A:.4f}, ω_c={task_params.omega_c:.1f}")

        task_features = torch.tensor(
            task_params.to_array(),
            dtype=torch.float32,
            device=device
        )

        data = {
            'task_features': task_features.unsqueeze(0).repeat(2, 1),
            'task_params': task_params
        }

        task_batch.append({
            'support': data,
            'query': data
        })

    # Test initial loss evaluation
    print(f"\n" + "=" * 70)
    print("Testing initial loss computation...")
    print("=" * 70)

    for i, task_data in enumerate(task_batch):
        loss = loss_fn(policy, task_data['support'])
        print(f"Task {i} initial loss: {loss.item():.6f}")
        print(f"  requires_grad: {loss.requires_grad}")
        print(f"  grad_fn: {loss.grad_fn}")

    # Run one meta-training step with verbose logging
    print(f"\n" + "=" * 70)
    print("Running MAML meta-training step...")
    print("=" * 70)

    # Manually replicate meta_train_step with verbose output
    maml.meta_optimizer.zero_grad()

    try:
        import higher
        HIGHER_AVAILABLE = True
    except ImportError:
        HIGHER_AVAILABLE = False
        print("WARNING: 'higher' library not available!")

    from torch import autograd, optim

    for task_idx, task_data in enumerate(task_batch):
        print(f"\n--- Task {task_idx} ---")

        if maml.first_order and HIGHER_AVAILABLE:
            # FOMAML with higher
            support_data = task_data['support']
            query_data = task_data['query']

            inner_losses = []
            inner_opt = optim.SGD(policy.parameters(), lr=maml.inner_lr)

            with higher.innerloop_ctx(
                policy,
                inner_opt,
                copy_initial_weights=True,
                track_higher_grads=False  # FOMAML
            ) as (fmodel, diffopt):

                # Inner loop
                print(f"  Inner loop adaptation ({maml.inner_steps} steps):")
                for step in range(maml.inner_steps):
                    loss = loss_fn(fmodel, support_data)
                    inner_losses.append(loss.item())
                    print(f"    Step {step}: loss = {loss.item():.6f}")
                    diffopt.step(loss)

                # Query loss
                query_loss = loss_fn(fmodel, query_data)
                print(f"  Query loss: {query_loss.item():.6f}")
                print(f"    requires_grad: {query_loss.requires_grad}")
                print(f"    grad_fn: {query_loss.grad_fn}")

                # Compute gradients
                meta_params = list(policy.parameters())
                adapted_params = list(fmodel.parameters())

                print(f"  Computing gradients w.r.t. {len(adapted_params)} adapted parameters...")

                adapted_grads = autograd.grad(
                    query_loss,
                    adapted_params,
                    create_graph=False,
                    allow_unused=True
                )

                # Diagnostics
                non_none = sum(1 for g in adapted_grads if g is not None)
                zero_grads = sum(1 for g in adapted_grads if g is not None and g.abs().max() < 1e-10)
                grad_norm = torch.sqrt(sum((g ** 2).sum() for g in adapted_grads if g is not None))

                print(f"    Non-None gradients: {non_none}/{len(adapted_grads)}")
                print(f"    Zero gradients: {zero_grads}")
                print(f"    Gradient norm (before assignment): {grad_norm.item():.6e}")

                # Assign to meta-parameters
                for param_idx, (meta_param, adapted_grad) in enumerate(zip(meta_params, adapted_grads)):
                    if adapted_grad is not None:
                        if meta_param.grad is None:
                            meta_param.grad = adapted_grad.clone()
                        else:
                            meta_param.grad = meta_param.grad + adapted_grad.clone()

    # Check accumulated gradients
    print(f"\n" + "=" * 70)
    print("Accumulated gradients across tasks:")
    print("=" * 70)

    n_tasks = len(task_batch)
    grad_stats = []

    for name, param in policy.named_parameters():
        if param.grad is not None:
            g_norm = param.grad.abs().max().item()
            g_mean = param.grad.abs().mean().item()
            grad_stats.append((name, g_norm, g_mean))

    # Average gradients
    for param in policy.parameters():
        if param.grad is not None:
            param.grad = param.grad / n_tasks

    print(f"\nGradients (BEFORE averaging):")
    for name, g_max, g_mean in grad_stats[:5]:  # Show first 5
        print(f"  {name:30s}: max={g_max:.6e}, mean={g_mean:.6e}")

    # After averaging
    overall_norm_before_clip = torch.sqrt(sum(
        (p.grad ** 2).sum() for p in policy.parameters() if p.grad is not None
    ))

    print(f"\nGradients (AFTER averaging over {n_tasks} tasks):")
    print(f"  Overall norm: {overall_norm_before_clip.item():.6e}")

    # Clip
    grad_norm_after_clip = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
    print(f"  After clipping (max_norm=1.0): {grad_norm_after_clip.item():.6e}")

    # Check if gradients are too small
    if grad_norm_after_clip < 1e-6:
        print(f"\n⚠ WARNING: Gradient norm is very small ({grad_norm_after_clip:.6e})")
        print("  This will lead to very slow or no learning!")

    if grad_norm_after_clip < 1e-10:
        print(f"\n✗ ERROR: Gradients are effectively zero!")

    # Optimizer step
    print(f"\nApplying optimizer step...")
    maml.meta_optimizer.step()

    # Check parameter change
    print(f"\nChecking if parameters changed...")
    with torch.no_grad():
        test_loss_after = loss_fn(policy, task_batch[0]['support'])

    print(f"  Task 0 loss before: {inner_losses[0]:.6f}")
    print(f"  Task 0 loss after meta-update: {test_loss_after.item():.6f}")
    print(f"  Change: {test_loss_after.item() - inner_losses[0]:.6e}")


if __name__ == "__main__":
    main()
