"""
Train GRAPE Baseline Policy

Train a GRAPE-based baseline for comparison with meta-learning and robust policies.
Unlike neural network policies, GRAPE directly optimizes control pulses per task.
"""

import torch
import numpy as np
import yaml
from pathlib import Path
from datetime import datetime
import argparse
import pickle

from metaqctrl.baselines.robust_control import GRAPEOptimizer

# Import system creation from train_meta
from train_meta import (
    create_quantum_system,
    create_task_distribution,
    task_sampler
)
from metaqctrl.quantum.gates import TargetGates


def main(config_path: str):
    """Main GRAPE training loop."""

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("=" * 70)
    print("GRAPE Baseline Training")
    print("=" * 70)
    print(f"Config: {config_path}\n")

    # Set random seeds
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Create quantum system
    print("Setting up quantum system...")
    quantum_system = create_quantum_system(config)

    # Create task distribution
    print("Creating task distribution...")
    task_dist = create_task_distribution(config)
    variance = task_dist.compute_variance()
    print(f"  Task variance σ²_θ = {variance:.4f}\n")

    # Target gate
    target_gate_name = config.get('target_gate', 'hadamard')
    if target_gate_name == 'hadamard':
        U_target = TargetGates.hadamard()
    elif target_gate_name == 'pauli_x':
        U_target = TargetGates.pauli_x()
    else:
        raise ValueError(f"Unknown target gate: {target_gate_name}")

    ket_0 = np.array([1, 0], dtype=complex)
    target_state = np.outer(U_target @ ket_0, (U_target @ ket_0).conj())
    print(f"Target gate: {target_gate_name}")

    # Create quantum environment
    from metaqctrl.theory.quantum_environment import QuantumEnvironment
    env = QuantumEnvironment(
        H0=quantum_system['H0'],
        H_controls=quantum_system['H_controls'],
        psd_to_lindblad=quantum_system['psd_to_lindblad'],
        target_state=target_state,
        T=config.get('horizon', 1.0),
        method=config.get('integration_method', 'RK45')
    )

    # GRAPE optimizer setup
    print("\nInitializing GRAPE optimizer...")
    n_segments = config.get('n_segments', 20)
    n_controls = config.get('n_controls', 2)

    # Training parameters
    n_train_tasks = config.get('grape_train_tasks', 50)
    grape_iterations = config.get('grape_iterations', 200)

    print(f"  Training on {n_train_tasks} tasks")
    print(f"  {grape_iterations} iterations per task")
    print(f"  Control segments: {n_segments}")
    print(f"  Control channels: {n_controls}")

    # Sample training tasks
    print(f"\nSampling {n_train_tasks} training tasks...")
    train_tasks = task_sampler(n_train_tasks, 'train', task_dist, rng)

    # Store optimized controls for each task
    grape_controls = {}
    fidelities = []

    # Create save directory
    save_dir = Path(config.get('save_dir', 'checkpoints'))
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = save_dir / f"grape_{timestamp}.pkl"

    print(f"\nResults will be saved to: {save_path}")

    # Train GRAPE on each task
    print("\n" + "=" * 70)
    print("Starting GRAPE optimization...")
    print("=" * 70 + "\n")

    for i, task in enumerate(train_tasks):
        print(f"\n[Task {i+1}/{n_train_tasks}]")
        print(f"  α={task.alpha:.3f}, A={task.A:.3f}, ω_c={task.omega_c:.3f}")

        # Create GRAPE optimizer for this task
        grape = GRAPEOptimizer(
            n_segments=n_segments,
            n_controls=n_controls,
            T=config.get('horizon', 1.0),
            learning_rate=config.get('grape_lr', 0.1),
            method=config.get('grape_method', 'adam'),
            device=device
        )

        # Define simulation function for GRAPE
        def simulate_fn(controls_np, task_params):
            return env.compute_fidelity(controls_np, task_params)

        # Optimize with GRAPE
        optimal_controls = grape.optimize(
            simulate_fn=simulate_fn,
            task_params=task,
            max_iterations=grape_iterations,
            verbose=(i % 10 == 0)  # Print every 10th task
        )

        # Compute final fidelity
        final_fidelity = env.compute_fidelity(optimal_controls, task)
        fidelities.append(final_fidelity)

        # Store controls (keyed by task parameters)
        task_key = (task.alpha, task.A, task.omega_c)
        grape_controls[task_key] = optimal_controls

        print(f"  Final fidelity: {final_fidelity:.6f}")

        # Save intermediate results every 10 tasks
        if (i + 1) % 10 == 0:
            temp_save = {
                'controls': grape_controls,
                'fidelities': fidelities,
                'config': config,
                'n_segments': n_segments,
                'n_controls': n_controls
            }
            with open(str(save_path).replace('.pkl', '_temp.pkl'), 'wb') as f:
                pickle.dump(temp_save, f)

    # Save final results
    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)

    results = {
        'controls': grape_controls,
        'fidelities': fidelities,
        'train_tasks': train_tasks,
        'config': config,
        'n_segments': n_segments,
        'n_controls': n_controls,
        'mean_fidelity': np.mean(fidelities),
        'std_fidelity': np.std(fidelities),
        'min_fidelity': np.min(fidelities),
        'max_fidelity': np.max(fidelities)
    }

    with open(save_path, 'wb') as f:
        pickle.dump(results, f)

    print(f"\nGRAPE baseline saved to: {save_path}")
    print(f"\nStatistics:")
    print(f"  Mean fidelity: {results['mean_fidelity']:.6f}")
    print(f"  Std fidelity:  {results['std_fidelity']:.6f}")
    print(f"  Min fidelity:  {results['min_fidelity']:.6f}")
    print(f"  Max fidelity:  {results['max_fidelity']:.6f}")

    # Also save in a format compatible with paper_results scripts
    # Create a simple "best" version
    best_save_path = save_dir / "grape_best.pkl"
    with open(best_save_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nBest model saved to: {best_save_path}")

    print("\nNext steps:")
    print("1. Evaluate GRAPE baseline using experiment_gap_vs_k.py")
    print("2. Compare with meta-learned and robust policies")
    print("\nNote: GRAPE optimizes per-task, so for new tasks you must")
    print("      run GRAPE optimization again (unlike policy-based methods)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train GRAPE baseline')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/experiment_config.yaml',
        help='Path to config file'
    )

    args = parser.parse_args()

    main(args.config)
