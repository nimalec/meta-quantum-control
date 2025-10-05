"""
Policy Networks for Quantum Control

Maps task features → control pulse sequences
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class PulsePolicy(nn.Module):
    """
    Neural network policy that outputs control pulse sequences.
    
    Architecture:
    task_features → MLP → (n_segments × n_controls) control amplitudes
    """
    
    def __init__(
        self,
        task_feature_dim: int = 3,  # (α, A, ωc)
        hidden_dim: int = 128,
        n_hidden_layers: int = 2,
        n_segments: int = 20,
        n_controls: int = 2,  # Typically 2 for single qubit (X, Y)
        output_scale: float = 1.0,
        activation: str = 'tanh'
    ):
        """
        Args:
            task_feature_dim: Dimension of task encoding
            hidden_dim: Hidden layer width
            n_hidden_layers: Number of hidden layers
            n_segments: Number of pulse segments
            n_controls: Number of control channels
            output_scale: Scale factor for control amplitudes
            activation: 'tanh', 'relu', 'elu'
        """
        super().__init__()
        
        self.task_feature_dim = task_feature_dim
        self.hidden_dim = hidden_dim
        self.n_segments = n_segments
        self.n_controls = n_controls
        self.output_dim = n_segments * n_controls
        self.output_scale = output_scale
        
        # Build MLP
        layers = []
        
        # Input layer
        layers.append(nn.Linear(task_feature_dim, hidden_dim))
        layers.append(self._get_activation(activation))
        
        # Hidden layers
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(self._get_activation(activation))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, self.output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function."""
        if name == 'tanh':
            return nn.Tanh()
        elif name == 'relu':
            return nn.ReLU()
        elif name == 'elu':
            return nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {name}")
    
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
    
    def forward(self, task_features: torch.Tensor) -> torch.Tensor:
        """
        Generate control pulses for given task.
        
        Args:
            task_features: (batch_size, task_feature_dim) or (task_feature_dim,)
            
        Returns:
            controls: (batch_size, n_segments, n_controls) or (n_segments, n_controls)
        """
        single_input = task_features.ndim == 1
        if single_input:
            task_features = task_features.unsqueeze(0)
        
        # Forward pass
        output = self.network(task_features)
        
        # Reshape to (batch, n_segments, n_controls)
        controls = output.view(-1, self.n_segments, self.n_controls)
        
        # Scale outputs
        controls = self.output_scale * controls
        
        if single_input:
            controls = controls.squeeze(0)
        
        return controls
    
    def get_lipschitz_constant(self) -> float:
        """
        Estimate Lipschitz constant L_net via spectral norms.
        L_net ≤ ∏ℓ ||Wℓ||₂
        """
        lipschitz = 1.0
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Compute spectral norm via power iteration
                W = module.weight.data
                spectral_norm = torch.linalg.matrix_norm(W, ord=2).item()
                lipschitz *= spectral_norm
        
        # Account for activation Lipschitz constants (tanh: 1, ReLU: 1)
        return lipschitz
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TaskFeatureEncoder(nn.Module):
    """
    Optional: Learn task representations from raw noise parameters.
    
    Can include:
    - Fourier features
    - Learned embeddings
    - PSD fingerprints
    """
    
    def __init__(
        self,
        raw_dim: int = 3,
        feature_dim: int = 16,
        use_fourier: bool = True,
        fourier_scale: float = 1.0
    ):
        super().__init__()
        
        self.raw_dim = raw_dim
        self.feature_dim = feature_dim
        self.use_fourier = use_fourier
        
        if use_fourier:
            # Random Fourier features: φ(x) = [cos(Bx), sin(Bx)]
            # where B ~ N(0, σ²I)
            self.register_buffer(
                'B',
                torch.randn(raw_dim, feature_dim // 2) * fourier_scale
            )
            final_dim = feature_dim
        else:
            # Simple MLP encoder
            self.encoder = nn.Sequential(
                nn.Linear(raw_dim, 32),
                nn.ReLU(),
                nn.Linear(32, feature_dim)
            )
            final_dim = feature_dim
        
        self.output_dim = final_dim
    
    def forward(self, raw_features: torch.Tensor) -> torch.Tensor:
        """
        Encode raw task parameters.
        
        Args:
            raw_features: (batch, raw_dim) - e.g., (α, A, ωc)
            
        Returns:
            encoded: (batch, feature_dim)
        """
        if self.use_fourier:
            # Fourier features
            x_proj = raw_features @ self.B
            encoded = torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)
        else:
            encoded = self.encoder(raw_features)
        
        return encoded


class ValueNetwork(nn.Module):
    """
    Value function V(s, θ) for advantage estimation in policy gradient.
    Optional for actor-critic style meta-RL.
    """
    
    def __init__(
        self,
        state_dim: int,
        task_feature_dim: int,
        hidden_dim: int = 128
    ):
        super().__init__()
        
        input_dim = state_dim + task_feature_dim
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state: torch.Tensor, task_features: torch.Tensor) -> torch.Tensor:
        """
        Estimate value V(s, θ).
        
        Args:
            state: Current quantum state representation
            task_features: Task encoding
            
        Returns:
            value: Scalar value estimate
        """
        x = torch.cat([state, task_features], dim=-1)
        return self.network(x).squeeze(-1)


def create_policy(
    config: dict,
    device: torch.device = torch.device('cpu')
) -> PulsePolicy:
    """
    Factory function to create policy from config.
    
    Args:
        config: Dictionary with policy hyperparameters
        device: torch device
        
    Returns:
        policy: Initialized policy network
    """
    policy = PulsePolicy(
        task_feature_dim=config.get('task_feature_dim', 3),
        hidden_dim=config.get('hidden_dim', 128),
        n_hidden_layers=config.get('n_hidden_layers', 2),
        n_segments=config.get('n_segments', 20),
        n_controls=config.get('n_controls', 2),
        output_scale=config.get('output_scale', 1.0),
        activation=config.get('activation', 'tanh')
    )
    
    policy = policy.to(device)
    
    print(f"Created policy with {policy.count_parameters():,} parameters")
    print(f"Estimated Lipschitz constant: {policy.get_lipschitz_constant():.2f}")
    
    return policy


# Example usage
if __name__ == "__main__":
    # Create policy
    policy = PulsePolicy(
        task_feature_dim=3,
        hidden_dim=64,
        n_hidden_layers=2,
        n_segments=20,
        n_controls=2,
        output_scale=0.5
    )
    
    print(f"Policy architecture:\n{policy}")
    print(f"\nTotal parameters: {policy.count_parameters():,}")
    
    # Test forward pass
    task_features = torch.tensor([1.0, 0.1, 5.0])  # (α, A, ωc)
    controls = policy(task_features)
    
    print(f"\nTask features shape: {task_features.shape}")
    print(f"Output controls shape: {controls.shape}")
    print(f"Control range: [{controls.min():.3f}, {controls.max():.3f}]")
    
    # Batch forward
    batch_tasks = torch.randn(32, 3)
    batch_controls = policy(batch_tasks)
    print(f"\nBatch input shape: {batch_tasks.shape}")
    print(f"Batch output shape: {batch_controls.shape}")
    
    # Test feature encoder
    encoder = TaskFeatureEncoder(raw_dim=3, feature_dim=16, use_fourier=True)
    encoded_features = encoder(batch_tasks)
    print(f"\nEncoded features shape: {encoded_features.shape}")
