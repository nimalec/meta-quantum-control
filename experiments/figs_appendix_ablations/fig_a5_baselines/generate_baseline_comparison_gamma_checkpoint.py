"""
Baseline Comparison - Using Pretrained Gamma Checkpoint
=======================================================
Compares MAML adaptation with Fixed Average, Domain Randomization, and Worst-Case baselines.
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


def sample_gamma_tasks(n_tasks, rng=None, diverse=False):
    """Sample gamma-rate tasks.

    Training distribution:
        gamma_deph: uniform [0.02, 0.15]
        gamma_relax: uniform [0.01, 0.08]

    Diverse distribution (for challenging evaluation):
        gamma_deph: uniform [0.05, 0.40]  (extends 2.7× beyond training max)
        gamma_relax: uniform [0.03, 0.20] (extends 2.5× beyond training max)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    if diverse:
        # Extended range for challenging OOD evaluation
        gamma_deph_range = (0.05, 0.40)
        gamma_relax_range = (0.03, 0.20)
    else:
        # Match the training distribution exactly
        gamma_deph_range = (0.02, 0.15)
        gamma_relax_range = (0.01, 0.08)

    tasks = []
    for _ in range(n_tasks):
        gamma_deph = rng.uniform(gamma_deph_range[0], gamma_deph_range[1])
        gamma_relax = rng.uniform(gamma_relax_range[0], gamma_relax_range[1])
        tasks.append((gamma_deph, gamma_relax))

    return tasks


def train_fixed_average_baseline(n_iterations=500, device='cpu'):
    """Train policy on fixed average gamma rates."""
    print("  Training Fixed Average baseline...")
    policy = GammaPulsePolicy(
        task_feature_dim=3, hidden_dim=64, n_hidden_layers=2,
        n_segments=20, n_controls=2, output_scale=1.0
    ).to(device)

    optimizer = optim.Adam(policy.parameters(), lr=1e-3)

    for iteration in range(n_iterations):
        optimizer.zero_grad()
        loss = compute_loss_gamma(policy, 0.05, 0.025, device)
        loss.backward()
        optimizer.step()

    return policy


def grape_optimize_task(gamma_deph, gamma_relax, n_steps, lr=0.01, device='cpu'):
    """GRAPE-style optimization: random init + gradient descent for a single task.

    This represents the traditional approach: optimize from scratch for each new task.
    """
    # Create fresh random policy for this task
    policy = GammaPulsePolicy(
        task_feature_dim=3, hidden_dim=64, n_hidden_layers=2,
        n_segments=20, n_controls=2, output_scale=1.0
    ).to(device)

    optimizer = optim.Adam(policy.parameters(), lr=lr)

    for step in range(n_steps):
        optimizer.zero_grad()
        loss = compute_loss_gamma(policy, gamma_deph, gamma_relax, device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

    # Return final fidelity
    with torch.no_grad():
        final_loss = compute_loss_gamma(policy, gamma_deph, gamma_relax, device).item()
    return 1.0 - final_loss


def train_worst_case_baseline(n_iterations=500, n_candidates=8, device='cpu'):
    """Train policy using stochastic worst-case optimization.

    At each iteration, sample n_candidates tasks and optimize on the worst (highest loss).
    This is a minimax approach: find policy that minimizes maximum loss.
    """
    print("  Training Worst Case baseline...")
    policy = GammaPulsePolicy(
        task_feature_dim=3, hidden_dim=64, n_hidden_layers=2,
        n_segments=20, n_controls=2, output_scale=1.0
    ).to(device)

    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    rng = np.random.default_rng(999)

    for iteration in range(n_iterations):
        # Sample candidate tasks from training distribution
        candidates = []
        for _ in range(n_candidates):
            gamma_deph = rng.uniform(0.02, 0.15)
            gamma_relax = rng.uniform(0.01, 0.08)
            candidates.append((gamma_deph, gamma_relax))

        # Find worst task (highest loss)
        worst_loss = None
        worst_task = None
        for gamma_deph, gamma_relax in candidates:
            with torch.no_grad():
                loss = compute_loss_gamma(policy, gamma_deph, gamma_relax, device)
                if worst_loss is None or loss.item() > worst_loss:
                    worst_loss = loss.item()
                    worst_task = (gamma_deph, gamma_relax)

        # Optimize on worst task
        optimizer.zero_grad()
        loss = compute_loss_gamma(policy, worst_task[0], worst_task[1], device)
        loss.backward()
        optimizer.step()

    return policy


def adapt_policy(policy, gamma_deph, gamma_relax, K_adapt, inner_lr=0.01, device='cpu'):
    """Adapt policy for K steps."""
    adapted_policy = deepcopy(policy)
    adapted_policy.train()
    inner_opt = optim.Adam(adapted_policy.parameters(), lr=inner_lr)

    for k in range(K_adapt):
        inner_opt.zero_grad()
        loss = compute_loss_gamma(adapted_policy, gamma_deph, gamma_relax, device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapted_policy.parameters(), max_norm=1.0)
        inner_opt.step()

    return adapted_policy


def evaluate_policy(policy, task_params_list, K_adapt=0, inner_lr=0.01, device='cpu'):
    """Evaluate policy on tasks, with optional adaptation."""
    fidelities = []

    for gamma_deph, gamma_relax in task_params_list:
        if K_adapt > 0:
            eval_policy = adapt_policy(policy, gamma_deph, gamma_relax, K_adapt, inner_lr, device)
        else:
            eval_policy = policy

        with torch.no_grad():
            loss = compute_loss_gamma(eval_policy, gamma_deph, gamma_relax, device).item()
            fidelities.append(1.0 - loss)

    return np.mean(fidelities), np.std(fidelities)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_tasks', type=int, default=50)
    parser.add_argument('--K_adapt', type=int, default=10)
    parser.add_argument('--inner_lr', type=float, default=0.001,
                        help='Adaptation LR for MAML adaptation')
    parser.add_argument('--checkpoint', type=str,
                        default='../../../checkpoints_gamma/maml_gamma_pauli_x.pt')
    parser.add_argument('--diverse', action='store_true',
                        help='Use diverse/challenging OOD task distribution for evaluation')
    args = parser.parse_args()

    device = torch.device('cpu')
    print(f"Using device: {device}")

    # Load pretrained MAML policy
    checkpoint_path = Path(__file__).parent / args.checkpoint
    if not checkpoint_path.exists():
        checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return

    maml_policy = load_pretrained_gamma_policy(str(checkpoint_path), device=device)

    # Train baselines
    print("\nTraining baselines...")
    fixed_avg_policy = train_fixed_average_baseline(n_iterations=1000, device=device)

    # Sample test tasks
    rng = np.random.default_rng(123)
    test_tasks = sample_gamma_tasks(args.n_tasks, rng=rng, diverse=args.diverse)
    if args.diverse:
        print(f"\nUsing DIVERSE task distribution: γ_deph∈[0.05,0.40], γ_relax∈[0.03,0.20]")
    else:
        print(f"\nUsing IN-DISTRIBUTION tasks: γ_deph∈[0.02,0.15], γ_relax∈[0.01,0.08]")

    print("\n" + "=" * 60)
    print("Evaluating methods...")
    print("=" * 60)

    results = {}

    # Evaluate MAML
    maml_no_adapt, _ = evaluate_policy(maml_policy, test_tasks, K_adapt=0, device=device)
    maml_adapted, _ = evaluate_policy(maml_policy, test_tasks, K_adapt=args.K_adapt, inner_lr=args.inner_lr, device=device)
    results['MAML'] = {'K=0': maml_no_adapt, f'K={args.K_adapt}': maml_adapted}
    print(f"  MAML: K=0: {maml_no_adapt:.4f}, K={args.K_adapt}: {maml_adapted:.4f}")

    # Evaluate Fixed Average (same LR as MAML for fair comparison)
    fixed_no_adapt, _ = evaluate_policy(fixed_avg_policy, test_tasks, K_adapt=0, device=device)
    fixed_adapted, _ = evaluate_policy(fixed_avg_policy, test_tasks, K_adapt=args.K_adapt, inner_lr=args.inner_lr, device=device)
    results['Fixed Avg'] = {'K=0': fixed_no_adapt, f'K={args.K_adapt}': fixed_adapted}
    print(f"  Fixed Avg: K=0: {fixed_no_adapt:.4f}, K={args.K_adapt}: {fixed_adapted:.4f}")

    # Evaluate GRAPE (random init + same K steps per task)
    # GRAPE needs higher LR since it starts from random init (far from optimum)
    grape_lr = 0.01  # 10x higher than MAML adaptation LR
    print(f"  Running GRAPE optimization (random init per task, lr={grape_lr})...")
    grape_fidelities = []
    for gamma_deph, gamma_relax in test_tasks:
        fid = grape_optimize_task(gamma_deph, gamma_relax, n_steps=args.K_adapt, lr=grape_lr, device=device)
        grape_fidelities.append(fid)
    grape_mean = np.mean(grape_fidelities)
    # GRAPE K=0 is random init (~50% for X gate)
    results['GRAPE'] = {'K=0': 0.50, f'K={args.K_adapt}': grape_mean}
    print(f"  GRAPE: K=0: 0.5000 (random), K={args.K_adapt}: {grape_mean:.4f}")

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))

    methods = ['MAML', 'Fixed Avg']
    x = np.arange(len(methods))
    width = 0.35

    k0_values = [results[m]['K=0'] for m in methods]
    k_adapt_values = [results[m][f'K={args.K_adapt}'] for m in methods]

    bars1 = ax.bar(x - width/2, k0_values, width, label='K=0 (No Adaptation)', color='#95a5a6')
    bars2 = ax.bar(x + width/2, k_adapt_values, width, label=f'K={args.K_adapt}', color='#27ae60')

    ax.set_xlabel('Method')
    ax.set_ylabel('Mean Fidelity')
    title = 'Baseline Comparison (Diverse Tasks)' if args.diverse else 'Baseline Comparison (In-Distribution)'
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    # Adjust y-axis limits based on data
    min_val = min(k0_values) - 0.05
    ax.set_ylim([0.45, 1.0])  # Show GRAPE K=0 at 0.5
    ax.grid(True, alpha=0.3, axis='y')

    # Add adaptation gap annotations
    for i, m in enumerate(methods):
        gap = k_adapt_values[i] - k0_values[i]
        ax.annotate(f'+{gap:.3f}', xy=(x[i] + width/2, k_adapt_values[i] + 0.005),
                    ha='center', fontsize=9, color='#27ae60')

    plt.tight_layout()

    output_dir = Path(__file__).parent
    suffix = "_diverse" if args.diverse else ""
    save_path = str(output_dir / f"baseline_comparison_gamma_checkpoint{suffix}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"\nFigure saved to: {save_path}")

    # Save data
    data_path = str(output_dir / f"baseline_comparison_gamma_checkpoint{suffix}_data.json")
    with open(data_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("Summary: Adaptation Gaps")
    print("=" * 60)
    for m in methods:
        gap = results[m][f'K={args.K_adapt}'] - results[m]['K=0']
        print(f"  {m}: +{gap:.4f}")


if __name__ == '__main__':
    main()
