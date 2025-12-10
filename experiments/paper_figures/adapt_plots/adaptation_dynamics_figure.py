import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from copy import deepcopy

# Import project modules
from metaqctrl.meta_rl.policy import PulsePolicy
from metaqctrl.quantum.lindblad_torch import DifferentiableLindbladSimulator
from metaqctrl.quantum.gates import TargetGates
from metaqctrl.utils.checkpoint_utils import load_policy_from_checkpoint

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'font.family': 'serif',
    'figure.dpi': 150,
})

## Single qubit 
def create_single_qubit_system(gamma_deph=0.05, gamma_relax=0.02, device='cpu'):
    """Create a single-qubit Lindblad simulator with given noise rates."""
    # Pauli matrices
    sigma_x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
    sigma_y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)
    sigma_z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)

    # Drift Hamiltonian
    H0 = 0.5 * sigma_z

    # Control Hamiltonians  
    H_controls = [sigma_x, sigma_y] 
    # Lindblad operators
    L_ops = []
    if gamma_deph > 0:
        L_ops.append(np.sqrt(gamma_deph) * sigma_z)
    if gamma_relax > 0:
        L_ops.append(np.sqrt(gamma_relax) * torch.tensor([[0, 1], [0, 0]], dtype=torch.complex64))

    sim = DifferentiableLindbladSimulator(
        H0=H0,
        H_controls=H_controls,
        L_operators=L_ops,
        dt=0.01,
        method='rk4',
        device=torch.device(device)
    )

    return sim


def compute_fidelity(rho, target_rho):
    """Compute state fidelity F(rho, target)."""
    fid = torch.trace(rho @ target_rho).real
    return torch.clamp(fid, 0, 1)


def compute_loss(policy, task_data, return_fidelity=False):
    """
    Compute loss for a task: L = 1 - F(rho_final, rho_target).

    Args:
        policy: PulsePolicy network
        task_data: Dict with 'task_features', 'simulator', 'rho0', 'target_rho', 'T'
    """
    task_features = task_data['task_features']
    sim = task_data['simulator']
    rho0 = task_data['rho0']
    target_rho = task_data['target_rho']
    T = task_data['T']

    # Generate control sequence
    controls = policy(task_features)

    # Simulate
    rho_final, _ = sim.evolve(rho0, controls, T)

    # Compute fidelity and loss
    fidelity = compute_fidelity(rho_final, target_rho)
    loss = 1.0 - fidelity

    if return_fidelity:
        return loss, fidelity
    return loss


# =============================================================================
# Task Sampling
# =============================================================================

def sample_tasks(n_tasks, device='cpu', seed=None):
    """
    Sample tasks with varying noise parameters.

    Each task has different dephasing/relaxation rates.
    Returns list of task dictionaries.
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    tasks = []
  
    gamma_deph_range = (0.0001, 0.1)  # Wide range: 0.0001 to 0.1
    gamma_relax_range = (0.0001, 0.05)  # Proportionally wide range

    # Initial state |0> and target |1> (Pauli X gate)
    rho0 = torch.tensor([[1, 0], [0, 0]], dtype=torch.complex64, device=device)
    target_rho = torch.tensor([[0, 0], [0, 1]], dtype=torch.complex64, device=device)

    for i in range(n_tasks):
        gamma_deph = np.random.uniform(*gamma_deph_range)
        gamma_relax = np.random.uniform(*gamma_relax_range)

        sim = create_single_qubit_system(gamma_deph, gamma_relax, device)

        # Task features: normalized noise parameters (adjusted for new ranges)
        task_features = torch.tensor([
            gamma_deph / 0.1,  #normalize 
            gamma_relax / 0.05, #normalize  
            (gamma_deph + gamma_relax) / 0.15  #normalize  
        ], dtype=torch.float32, device=device)

        tasks.append({
            'task_features': task_features,
            'simulator': sim,
            'rho0': rho0,
            'target_rho': target_rho,
            'T': 1.0,
            'gamma_deph': gamma_deph,
            'gamma_relax': gamma_relax,
            'task_id': i
        })

    return tasks




def train_robust_policy(n_iterations=300, device='cpu', config=None):
    """
    Train a robust policy on a FIXED average noise level (no adaptation).

    This represents a controller optimized for "typical" conditions that cannot
    adapt to different noise environments - showing the benefit of meta-learning.
    """
    print("Training robust baseline policy on FIXED average noise...")

    # Use config if provided, otherwise defaults matching experiment_config.yaml
    if config is None:
        config = {
            'task_feature_dim': 3,
            'hidden_dim': 128,
            'n_hidden_layers': 2,
            'n_segments': 60,
            'n_controls': 2,
            'output_scale': 1.0
        }

    policy = PulsePolicy(
        task_feature_dim=config['task_feature_dim'],
        hidden_dim=config['hidden_dim'],
        n_hidden_layers=config['n_hidden_layers'],
        n_segments=config['n_segments'],
        n_controls=config['n_controls'],
        output_scale=config['output_scale']
    ).to(device)

    optimizer = optim.Adam(policy.parameters(), lr=1e-3)

    avg_gamma_deph = 0.02
    avg_gamma_relax = 0.01

    sim = create_single_qubit_system(avg_gamma_deph, avg_gamma_relax, device)
    rho0 = torch.tensor([[1, 0], [0, 0]], dtype=torch.complex64, device=device)
    target_rho = torch.tensor([[0, 0], [0, 1]], dtype=torch.complex64, device=device)

    fixed_task = {
        'task_features': torch.tensor([
            avg_gamma_deph / 0.1,
            avg_gamma_relax / 0.05,
            (avg_gamma_deph + avg_gamma_relax) / 0.15
        ], dtype=torch.float32, device=device),
        'simulator': sim,
        'rho0': rho0,
        'target_rho': target_rho,
        'T': 1.0,
    }

    for iteration in range(n_iterations):
        optimizer.zero_grad() 
        total_loss = compute_loss(policy, fixed_task)  
        total_loss.backward()
        optimizer.step()

        if iteration % 100 == 0:
            print(f"  Iter {iteration}: Loss = {total_loss.item():.4f}")

    print(f"Robust training complete (fixed gamma_deph={avg_gamma_deph}, gamma_relax={avg_gamma_relax}).")
    return policy


# =============================================================================
# Panel (a): Loss vs K for multiple tasks
# =============================================================================

def generate_panel_a_data(meta_policy, max_K=15, inner_lr=0.05, device='cpu'):
    """
    Generate adaptation curves for 4 diverse tasks spanning the noise range.
    Uses representative tasks to avoid crowding while showing task diversity.
    """
    print("Generating Panel (a) data: Loss vs K...")

    # Create 4 diverse tasks spanning the full gamma_deph range [0.0001, 0.1]
    diverse_noise_levels = [
        (0.001, 0.0005, 'Very low'),    # Near minimum
        (0.02, 0.01, 'Low-mid'),         # Lower-middle
        (0.05, 0.025, 'Mid-high'),       # Upper-middle
        (0.09, 0.045, 'High'),           # Near maximum
    ]

    rho0 = torch.tensor([[1, 0], [0, 0]], dtype=torch.complex64, device=device)
    target_rho = torch.tensor([[0, 0], [0, 1]], dtype=torch.complex64, device=device)

    tasks = []
    for i, (gd, gr, label) in enumerate(diverse_noise_levels):
        sim = create_single_qubit_system(gd, gr, device)
        task_features = torch.tensor([gd/0.1, gr/0.05, (gd+gr)/0.15],
                                      dtype=torch.float32, device=device)
        tasks.append({
            'task_features': task_features,
            'simulator': sim,
            'rho0': rho0,
            'target_rho': target_rho,
            'T': 1.0,
            'gamma_deph': gd,
            'gamma_relax': gr,
            'task_id': i,
            'label': label
        })

    losses_by_task = {}

    for task in tasks:
        task_id = task['task_id']
        losses = []

        # Clone policy
        adapted_policy = deepcopy(meta_policy)
        inner_opt = optim.SGD(adapted_policy.parameters(), lr=inner_lr)

        # Loss at K=0 (before adaptation)
        with torch.no_grad():
            loss_k0 = compute_loss(adapted_policy, task)
            losses.append(loss_k0.item())

        # Adaptation steps
        for k in range(max_K):
            inner_opt.zero_grad()
            loss = compute_loss(adapted_policy, task)
            loss.backward()
            inner_opt.step()

            with torch.no_grad():
                loss_post = compute_loss(adapted_policy, task)
                losses.append(loss_post.item())

        losses_by_task[task_id] = {
            'losses': losses,
            'gamma_deph': task['gamma_deph'],
            'gamma_relax': task['gamma_relax'],
            'label': task.get('label', f"Task {task_id}")
        }

    return losses_by_task


# =============================================================================
# Panel (b): Fidelity distributions
# =============================================================================

def generate_panel_b_data(meta_policy, robust_policy, n_tasks=50, K_adapt=5, inner_lr=0.05, device='cpu'):

    print("Generating Panel (b) data: Fidelity distributions...")

    tasks = sample_tasks(n_tasks, device=device, seed=123)

    fidelities = {
        'robust': [],
        'meta_init': [],
        'adapted': []
    }

    for task in tasks:
        # Robust policy (no adaptation)
        with torch.no_grad():
            _, fid_robust = compute_loss(robust_policy, task, return_fidelity=True)
            fidelities['robust'].append(fid_robust.item())

        # Meta-init (before adaptation)
        with torch.no_grad():
            _, fid_meta = compute_loss(meta_policy, task, return_fidelity=True)
            fidelities['meta_init'].append(fid_meta.item())

        # Adapted (after K steps)
        adapted_policy = deepcopy(meta_policy)
        inner_opt = optim.SGD(adapted_policy.parameters(), lr=inner_lr)

        for _ in range(K_adapt):
            inner_opt.zero_grad()
            loss = compute_loss(adapted_policy, task)
            loss.backward()
            inner_opt.step()

        with torch.no_grad():
            _, fid_adapted = compute_loss(adapted_policy, task, return_fidelity=True)
            fidelities['adapted'].append(fid_adapted.item())

    return fidelities


# =============================================================================
# Panel (c): Pulse sequences
# =============================================================================

def generate_panel_c_data(meta_policy, n_tasks=3, K_adapt=5, inner_lr=0.05, device='cpu'):
    """
    Generate data for Panel (c): Pulse sequences for different tasks.

    Returns:
        pulses: dict mapping task_id -> {'controls': (n_seg, 2), 'task_info': dict}
    """
    print("Generating Panel (c) data: Pulse sequences...")

    # Sample tasks with distinct noise levels spanning the full range
    np.random.seed(789)
    tasks = []

    # Low, medium, high noise tasks - spanning 0.0001 to 0.1 range
    noise_levels = [
        (0.001, 0.0005, 'Low noise'),      # Very low noise
        (0.02, 0.01, 'Medium noise'),      # Moderate noise
        (0.08, 0.04, 'High noise')         # High noise
    ]

    rho0 = torch.tensor([[1, 0], [0, 0]], dtype=torch.complex64, device=device)
    target_rho = torch.tensor([[0, 0], [0, 1]], dtype=torch.complex64, device=device)  # Pauli X target

    for i, (gd, gr, label) in enumerate(noise_levels):
        sim = create_single_qubit_system(gd, gr, device)
        task_features = torch.tensor([gd/0.1, gr/0.05, (gd+gr)/0.15],
                                      dtype=torch.float32, device=device)
        tasks.append({
            'task_features': task_features,
            'simulator': sim,
            'rho0': rho0,
            'target_rho': target_rho,
            'T': 1.0,
            'gamma_deph': gd,
            'gamma_relax': gr,
            'label': label,
            'task_id': i
        })

    pulses = {}

    for task in tasks:
        # Adapt policy to this task
        adapted_policy = deepcopy(meta_policy)
        inner_opt = optim.SGD(adapted_policy.parameters(), lr=inner_lr)

        for _ in range(K_adapt):
            inner_opt.zero_grad()
            loss = compute_loss(adapted_policy, task)
            loss.backward()
            inner_opt.step()

        # Get adapted control sequence
        with torch.no_grad():
            controls = adapted_policy(task['task_features'])

        pulses[task['task_id']] = {
            'controls': controls.cpu().numpy(),
            'label': task['label'],
            'gamma_deph': task['gamma_deph'],
            'gamma_relax': task['gamma_relax']
        }

    return pulses


# =============================================================================
# Plotting
# =============================================================================

def create_figure(losses_data, fidelity_data, pulse_data, save_path=None):
    """Create the 3-panel figure."""

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    # Color palette
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(losses_data)))

    # --- Panel (a): Loss vs K ---
    ax = axes[0]

    for i, (task_id, data) in enumerate(losses_data.items()):
        losses = data['losses']
        K_vals = np.arange(len(losses))
        # Use gamma_phi value with appropriate precision based on magnitude
        gamma_val = data['gamma_deph']
        if gamma_val < 0.01:
            label_str = f"$\\gamma_\\phi$={gamma_val:.3f}"
        else:
            label_str = f"$\\gamma_\\phi$={gamma_val:.2f}"
        ax.plot(K_vals, losses, 'o-', color=colors[i], markersize=4,
                linewidth=1.5, alpha=0.8,
                label=label_str)

    ax.set_xlabel('Adaptation Steps $K$')
    ax.set_ylabel('Loss $\\mathcal{L}(\\theta_K)$')
    ax.set_title('(a) Per-Task Adaptation Dynamics')
    ax.set_xlim(-0.5, len(losses_data[0]['losses']) - 0.5)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)

    # --- Panel (b): Fidelity Distribution ---
    ax = axes[1]

    positions = [1, 2, 3]
    labels = ['Robust', 'Meta-init', 'Adapted']
    data_lists = [fidelity_data['robust'], fidelity_data['meta_init'], fidelity_data['adapted']]
    colors_violin = ['#e74c3c', '#3498db', '#2ecc71']

    parts = ax.violinplot(data_lists, positions=positions, showmeans=True, showmedians=True)

    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors_violin[i])
        pc.set_alpha(0.7)

    parts['cmeans'].set_color('black')
    parts['cmedians'].set_color('white')

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Fidelity')
    ax.set_title('(b) Fidelity Distribution')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')

    # Add statistics
    for i, (pos, data) in enumerate(zip(positions, data_lists)):
        mean = np.mean(data)
        std = np.std(data)
        ax.text(pos, 0.05, f'{mean:.2f}$\\pm${std:.2f}',
                ha='center', fontsize=8, color=colors_violin[i])

    # --- Panel (c): Pulse Sequences ---
    ax = axes[2]

    n_segments = list(pulse_data.values())[0]['controls'].shape[0]
    t = np.linspace(0, 1, n_segments)

    colors_pulse = ['#2ecc71', '#f39c12', '#e74c3c']

    for i, (task_id, data) in enumerate(pulse_data.items()):
        controls = data['controls']
        label = data['label']

        # Plot X control (solid line) and Y control (dashed line) with same color
        ax.plot(t, controls[:, 0], linestyle='-', color=colors_pulse[i],
                linewidth=2, label=f'{label}: $u_x$')
        ax.plot(t, controls[:, 1], linestyle='--', color=colors_pulse[i],
                linewidth=1.5, label=f'{label}: $u_y$')

    ax.set_xlabel('Time $t/T$')
    ax.set_ylabel('Control Amplitude')
    ax.set_title('(c) Task-Adapted Pulse Sequences')
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left', fontsize=7, ncol=2)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.close()
    return fig




def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}") 
    config = {
        'task_feature_dim': 3,
        'hidden_dim': 128,
        'n_hidden_layers': 2,
        'n_segments': 60,
        'n_controls': 2,
        'output_scale': 1.0,
        'inner_lr': 0.01,  # From experiment_config.yaml
        'horizon': 1.0,
        'target_gate': 'pauli_x',
    }

    # Specific checkpoint path - using maml_best_pauli_x_best.pt
    checkpoint_path = project_root / "experiments" / "checkpoints" / "maml_best_pauli_x_best.pt" 
    print(f"Looking for checkpoint: {checkpoint_path}") 
    # Load or train meta-policy
    if checkpoint_path.exists():
        print("Loading pre-trained meta-policy...")
        print(f"  Path: {checkpoint_path}")
        # Use load_policy_from_checkpoint for automatic architecture detection
        meta_policy = load_policy_from_checkpoint(
            str(checkpoint_path),
            config,
            device=torch.device(device),
            eval_mode=False,  # We need gradients for adaptation
            verbose=True
        )
        print(f"  Loaded successfully! Parameters: {meta_policy.count_parameters():,}")
    else:
        print(f"Checkpoint not found at {checkpoint_path}")
        print("Training quick meta-policy...")
        pass  


    # Train robust baseline with matching config
    robust_policy = train_robust_policy(n_iterations=300, device=device, config=config)

    # Get inner_lr from config
    inner_lr = config['inner_lr']
    print(f"\nUsing inner_lr={inner_lr} (from experiment_config.yaml)")

    # Generate data for each panel
    print("\n" + "-" * 50)
    losses_data = generate_panel_a_data(meta_policy, max_K=12, inner_lr=inner_lr, device=device)

    print("-" * 50)
    fidelity_data = generate_panel_b_data(meta_policy, robust_policy, n_tasks=50, K_adapt=5, inner_lr=inner_lr, device=device)

    print("-" * 50)
    pulse_data = generate_panel_c_data(meta_policy, n_tasks=3, K_adapt=5, inner_lr=inner_lr, device=device)

    # Create figure
    print("\n" + "-" * 50)
    print("Creating figure...")

    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    save_path = str(output_dir / "adaptation_dynamics_figure_2.png")

    create_figure(losses_data, fidelity_data, pulse_data, save_path=save_path)

    # Print summary statistics
    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)

    print("\nPanel (a) - Loss reduction per task:")
    for task_id, data in losses_data.items():
        initial = data['losses'][0]
        final = data['losses'][-1]
        reduction = (initial - final) / initial * 100
        print(f"  Task {task_id}: {initial:.4f} -> {final:.4f} ({reduction:.1f}% reduction)")

    print("\nPanel (b) - Fidelity statistics:")
    for name, fids in fidelity_data.items():
        print(f"  {name:12s}: {np.mean(fids):.4f} +/- {np.std(fids):.4f}")

    print("\nPanel (c) - Control amplitude ranges:")
    for task_id, data in pulse_data.items():
        controls = data['controls']
        print(f"  {data['label']:12s}: X=[{controls[:,0].min():.2f}, {controls[:,0].max():.2f}], "
              f"Y=[{controls[:,1].min():.2f}, {controls[:,1].max():.2f}]")

    print("\n" + "=" * 70)
    print("Figure generation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
