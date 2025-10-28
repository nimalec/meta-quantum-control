"""
Utilities for loading policy checkpoints with automatic architecture detection.

Handles cases where checkpoints were saved with different architecture parameters
than the current config (e.g., different n_hidden_layers or hidden_dim).
"""

import torch
from typing import Dict, Optional
from metaqctrl.meta_rl.policy import PulsePolicy


def infer_policy_architecture_from_checkpoint(
    checkpoint_path: str,
    config: Dict,
    verbose: bool = True
) -> Dict:
    """
    Infer the policy architecture from a checkpoint file.

    This handles cases where the checkpoint was saved with different architecture
    parameters than the current config (e.g., different n_hidden_layers).

    Args:
        checkpoint_path: Path to the checkpoint file
        config: Configuration dictionary (used for n_segments, n_controls)
        verbose: Whether to print architecture info

    Returns:
        arch_config: Dictionary with inferred architecture parameters
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract state dict
    if isinstance(checkpoint, dict) and 'policy_state_dict' in checkpoint:
        state_dict = checkpoint['policy_state_dict']
    elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Count network layers to infer n_hidden_layers
    # Network structure: input layer + n_hidden_layers * (linear + activation) + output layer
    # So we have layers: 0 (input), 1 (act), 2 (hidden1), 3 (act), ..., final (output)
    max_layer_idx = max([
        int(key.split('.')[1])
        for key in state_dict.keys()
        if key.startswith('network.') and '.weight' in key
    ])

    # Get dimensions from the layers
    input_layer_out = state_dict['network.0.weight'].shape[0]  # hidden_dim
    output_layer_in = state_dict[f'network.{max_layer_idx}.weight'].shape[1]  # should be hidden_dim
    output_dim = state_dict[f'network.{max_layer_idx}.weight'].shape[0]

    # Calculate n_hidden_layers
    # Formula: max_layer_idx = 2 + 2 * n_hidden_layers
    # Because: input(0) + act(1) + n_hidden_layers * 2 (each has linear + act) + output
    # For n_hidden_layers=1: 0 (input), 1 (act), 2 (hidden), 3 (act), 4 (output) -> max_layer_idx=4
    # For n_hidden_layers=2: 0 (input), 1 (act), 2 (hidden1), 3 (act), 4 (hidden2), 5 (act), 6 (output) -> max_layer_idx=6
    n_hidden_layers = (max_layer_idx - 2) // 2

    # Calculate expected dimensions
    n_segments = config['n_segments']
    n_controls = config['n_controls']
    expected_output_dim = n_segments * n_controls

    # Verify consistency
    if output_dim != expected_output_dim and verbose:
        print(f"WARNING: Checkpoint output_dim ({output_dim}) doesn't match expected ({expected_output_dim})")
        print(f"         This might indicate different n_segments or n_controls in checkpoint")

    arch_config = {
        'task_feature_dim': config.get('task_feature_dim', 3),
        'hidden_dim': input_layer_out,
        'n_hidden_layers': n_hidden_layers,
        'n_segments': config['n_segments'],
        'n_controls': config['n_controls']
    }

    if verbose:
        print(f"Inferred architecture from checkpoint:")
        print(f"  hidden_dim: {arch_config['hidden_dim']}")
        print(f"  n_hidden_layers: {arch_config['n_hidden_layers']}")
        print(f"  output_dim: {output_dim} (n_segments={n_segments} × n_controls={n_controls})")

    return arch_config


def load_policy_from_checkpoint(
    checkpoint_path: str,
    config: Dict,
    device: torch.device = torch.device('cpu'),
    eval_mode: bool = True,
    verbose: bool = True
) -> PulsePolicy:
    """
    Load a PulsePolicy from checkpoint with automatic architecture detection.

    Args:
        checkpoint_path: Path to the checkpoint file
        config: Configuration dictionary
        device: Device to load the model on
        eval_mode: Whether to set the policy to eval mode
        verbose: Whether to print loading info

    Returns:
        Loaded PulsePolicy
    """
    # Infer architecture from checkpoint
    arch_config = infer_policy_architecture_from_checkpoint(
        checkpoint_path, config, verbose=verbose
    )

    # Create policy with detected architecture
    policy = PulsePolicy(
        task_feature_dim=arch_config['task_feature_dim'],
        hidden_dim=arch_config['hidden_dim'],
        n_hidden_layers=arch_config['n_hidden_layers'],
        n_segments=arch_config['n_segments'],
        n_controls=arch_config['n_controls']
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract state dict
    if isinstance(checkpoint, dict) and 'policy_state_dict' in checkpoint:
        state_dict = checkpoint['policy_state_dict']
    elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Load state dict
    policy.load_state_dict(state_dict)

    if eval_mode:
        policy.eval()

    if verbose:
        print(f"Successfully loaded policy from {checkpoint_path}")

    return policy
