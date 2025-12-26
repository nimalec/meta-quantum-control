"""
Adaptation Steps Ablation - Using Pretrained Gamma Checkpoint
  
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from copy import deepcopy
import json
import argparse

from metaqctrl.meta_rl.policy_gamma import GammaPulsePolicy
from metaqctrl.quantum.lindblad_torch import DifferentiableLindbladSimulator

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})


def create_single_qubit_system(gamma_deph=0.05, gamma_relax=0.025, device='cpu'):
    """Create simulator with direct gamma rates."""
    sigma_x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device=device)
    sigma_y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64, device=device)
    sigma_z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=device)
    sigma_p = torch.tensor([[0, 1], [0, 0]], dtype=torch.complex64, device=device)

    H0 = 0.0 * sigma_z
    H_controls = [sigma_x, sigma_y]

    L_operators = []
    if gamma_deph > 0:
        L_operators.append(np.sqrt(gamma_deph / 2.0) * sigma_z)
    if gamma_relax > 0:
        L_operators.append(np.sqrt(gamma_relax) * sigma_p)

    if not L_operators:
        L_operators.append(torch.zeros(2, 2, dtype=torch.complex64, device=device))

    sim = DifferentiableLindbladSimulator(
        H0=H0, H_controls=H_controls, L_operators=L_operators,
        dt=0.05, method='rk4', device=device
    )
    return sim


def compute_loss_gamma(policy, gamma_deph, gamma_relax, device='cpu'):
    """Compute loss using gamma-rate task features."""
    sim = create_single_qubit_system(gamma_deph, gamma_relax, device=device)

    task_features = torch.tensor([
        gamma_deph / 0.1,
        gamma_relax / 0.05,
        (gamma_deph + gamma_relax) / 0.15
    ], dtype=torch.float32, device=device)

    controls = policy(task_features)

    rho0 = torch.zeros(2, 2, dtype=torch.complex64, device=device)
    rho0[0, 0] = 1.0

    rho_final = sim.forward(rho0, controls, T=1.0)

    target = torch.zeros(2, 2, dtype=rho_final.dtype, device=device)
    target[1, 1] = 1.0

    fidelity = torch.abs(torch.trace(target @ rho_final)).real
    return 1.0 - fidelity


def load_pretrained_gamma_policy(checkpoint_path, device='cpu'):
    """Load pretrained gamma policy from checkpoint."""
    print(f"Loading pretrained policy from: {checkpoint_path}")

    policy = GammaPulsePolicy(
        task_feature_dim=3, hidden_dim=64, n_hidden_layers=2,
        n_segments=20, n_controls=2, output_scale=1.0
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'policy_state_dict' in checkpoint:
        policy.load_state_dict(checkpoint['policy_state_dict'])
    else:
        policy.load_state_dict(checkpoint)

    policy.eval()
    return policy


def sample_gamma_tasks(n_tasks, rng=None):
    """Sample gamma-rate tasks with UNIFORM distribution matching training.

    Training distribution:
        gamma_deph: uniform [0.02, 0.15]
        gamma_relax: uniform [0.01, 0.08]
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Match the training distribution exactly
    gamma_deph_range = (0.02, 0.15)
    gamma_relax_range = (0.01, 0.08)

    tasks = []
    for _ in range(n_tasks):
        gamma_deph = rng.uniform(gamma_deph_range[0], gamma_deph_range[1])
        gamma_relax = rng.uniform(gamma_relax_range[0], gamma_relax_range[1])
        tasks.append((gamma_deph, gamma_relax))

    return tasks


def adapt_and_evaluate(robust_policy, task_params, K_adapt, inner_lr=0.01, device='cpu'):
    """Adapt policy for K steps and return final fidelity."""
    gamma_deph, gamma_relax = task_params

    adapted_policy = deepcopy(robust_policy)
    adapted_policy.train()
    inner_opt = optim.Adam(adapted_policy.parameters(), lr=inner_lr)

    for k in range(K_adapt):
        inner_opt.zero_grad()
        loss = compute_loss_gamma(adapted_policy, gamma_deph, gamma_relax, device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapted_policy.parameters(), max_norm=1.0)
        inner_opt.step()

    with torch.no_grad():
        final_loss = compute_loss_gamma(adapted_policy, gamma_deph, gamma_relax, device).item()
        final_fidelity = 1.0 - final_loss

    return final_fidelity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_tasks', type=int, default=50)
    parser.add_argument('--max_K', type=int, default=30)
    parser.add_argument('--n_trials', type=int, default=3)
    parser.add_argument('--inner_lr', type=float, default=0.01)
    parser.add_argument('--checkpoint', type=str,
                        default='../../../checkpoints_gamma/maml_gamma_pauli_x.pt')
    args = parser.parse_args()

    device = torch.device('cpu')
    print(f"Using device: {device}")

    # Load pretrained policy
    checkpoint_path = Path(__file__).parent / args.checkpoint
    if not checkpoint_path.exists():
        checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return

    robust_policy = load_pretrained_gamma_policy(str(checkpoint_path), device=device)

    K_values = list(range(0, args.max_K + 1, 2))  # Every 2 steps

    print("\n" + "=" * 60)
    print("Computing fidelity vs adaptation steps...")
    print("=" * 60)

    all_fidelities = {K: [] for K in K_values}

    for trial in range(args.n_trials):
        print(f"\nTrial {trial + 1}/{args.n_trials}")
        rng = np.random.default_rng(42 + trial)
        task_params_list = sample_gamma_tasks(args.n_tasks, rng=rng)

        for K in K_values:
            fidelities = []
            for task_params in task_params_list:
                fid = adapt_and_evaluate(robust_policy, task_params, K,
                                         inner_lr=args.inner_lr, device=device)
                fidelities.append(fid)
            mean_fid = np.mean(fidelities)
            all_fidelities[K].append(mean_fid)
            print(f"  K={K}: Fidelity = {mean_fid:.4f}")

    # Compute statistics
    K_array = np.array(K_values)
    mean_fids = np.array([np.mean(all_fidelities[K]) for K in K_values])
    std_fids = np.array([np.std(all_fidelities[K]) for K in K_values])

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.errorbar(K_array, mean_fids, yerr=std_fids, fmt='o-', color='#3498db',
                capsize=3, markersize=6, linewidth=2, label='MAML + Adaptation')
    ax.axhline(y=mean_fids[0], color='#e74c3c', linestyle='--', linewidth=1.5,
               label=f'No Adaptation (K=0): {mean_fids[0]:.3f}')

    ax.set_xlabel('Adaptation Steps $K$')
    ax.set_ylabel('Mean Fidelity $\\mathcal{F}$')
    ax.set_title('Fidelity vs Adaptation Steps (Gamma Parameterization)')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([min(0.9, mean_fids.min() - 0.02), 1.0])

    plt.tight_layout()

    output_dir = Path(__file__).parent
    save_path = str(output_dir / "adaptation_steps_gamma_checkpoint.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"\nFigure saved to: {save_path}")

    # Save data
    data_path = str(output_dir / "adaptation_steps_gamma_checkpoint_data.json")
    data_to_save = {
        'K_values': K_values,
        'mean_fidelities': mean_fids.tolist(),
        'std_fidelities': std_fids.tolist(),
        'n_trials': args.n_trials,
        'n_tasks': args.n_tasks,
    }
    with open(data_path, 'w') as f:
        json.dump(data_to_save, f, indent=2)

    print("\n" + "=" * 60)
    print("Summary: Fidelity vs Adaptation Steps")
    print("=" * 60)
    print(f"  K=0:  Fidelity = {mean_fids[0]:.4f}")
    print(f"  K={K_values[-1]}: Fidelity = {mean_fids[-1]:.4f}")
    print(f"  Improvement: +{mean_fids[-1] - mean_fids[0]:.4f}")


if __name__ == '__main__':
    main()
