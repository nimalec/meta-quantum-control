# API Reference

Complete API documentation for the Meta-RL Quantum Control codebase.

## Table of Contents

1. [Quantum Simulation](#quantum-simulation)
   - [DifferentiableLindbladSimulator](#differentiablelindbladsimulator)
   - [LindbladSimulator](#lindbla simulator)
   - [Helper Functions](#quantum-helper-functions)

2. [Noise Models](#noise-models)
   - [NoiseParameters](#noiseparameters)
   - [NoisePSDModel](#noisepsdmodel)
   - [PSDToLindblad](#psdtolindblad)
   - [TaskDistribution](#taskdistribution)

3. [Policy Networks](#policy-networks)
   - [PulsePolicy](#pulsepolicy)
   - [TaskFeatureEncoder](#taskfeatureencoder)
   - [ValueNetwork](#valuenetwork)

4. [Meta-Learning](#meta-learning)
   - [MAML](#maml)
   - [MAMLTrainer](#mamltrainer)

5. [Theory & Analysis](#theory--analysis)
   - [OptimalityGapComputer](#optimalitygapcomputer)
   - [GapConstants](#gapconstants)

6. [Baseline Methods](#baseline-methods)
   - [RobustPolicy](#robustpolicy)
   - [GRAPEOptimizer](#grapeoptimizer)

7. [Environment Bridge](#environment-bridge)
   - [QuantumEnvironment](#quantumenvironment)

---

## Quantum Simulation

### DifferentiableLindbladSimulator

**Module:** `src.quantum.lindblad_torch`

Fully differentiable PyTorch implementation of the Lindblad master equation for end-to-end training.

#### Constructor

```python
DifferentiableLindbladSimulator(
    H0: torch.Tensor,
    H_controls: List[torch.Tensor],
    L_operators: List[torch.Tensor],
    dt: float = 0.05,
    method: str = 'rk4',
    device: torch.device = torch.device('cpu')
)
```

**Parameters:**
- `H0` (torch.Tensor): Drift Hamiltonian, shape (d, d), complex64
- `H_controls` (List[torch.Tensor]): Control Hamiltonians, each (d, d), complex64
- `L_operators` (List[torch.Tensor]): Lindblad operators, each (d, d), complex64
- `dt` (float): Time step for numerical integration. Default: 0.05
- `method` (str): Integration method, either 'euler' or 'rk4'. Default: 'rk4'
- `device` (torch.device): PyTorch device. Default: CPU

**Attributes:**
- `d` (int): Hilbert space dimension
- `n_controls` (int): Number of control channels
- `n_lindblad` (int): Number of Lindblad operators

**Example:**

```python
import torch
from src.quantum.lindblad_torch import DifferentiableLindbladSimulator

# Pauli matrices
sigma_x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
sigma_y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)
sigma_z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)

H0 = 0.5 * sigma_z
H_controls = [sigma_x, sigma_y]
L_ops = [torch.sqrt(torch.tensor(0.1)) * sigma_z]

sim = DifferentiableLindbladSimulator(
    H0=H0,
    H_controls=H_controls,
    L_operators=L_ops,
    method='rk4',
    device=torch.device('cuda')
)
```

#### Methods

##### lindbladian()

Compute Lindblad time derivative ρ̇ = L[ρ].

```python
lindbladian(
    rho: torch.Tensor,
    u: torch.Tensor
) -> torch.Tensor
```

**Parameters:**
- `rho` (torch.Tensor): Density matrix, shape (d, d), complex
- `u` (torch.Tensor): Control amplitudes, shape (n_controls,), real

**Returns:**
- `drho_dt` (torch.Tensor): Time derivative, shape (d, d), complex

**Mathematical Formula:**

```
ρ̇ = -i[H_total, ρ] + Σⱼ (Lⱼ ρ L†ⱼ - ½{L†ⱼLⱼ, ρ})
where H_total = H₀ + Σₖ uₖ Hₖ
```

##### step_rk4()

Single RK4 integration step (4th-order Runge-Kutta).

```python
step_rk4(
    rho: torch.Tensor,
    u: torch.Tensor,
    dt: float
) -> torch.Tensor
```

**Parameters:**
- `rho` (torch.Tensor): Current density matrix, shape (d, d)
- `u` (torch.Tensor): Control amplitudes, shape (n_controls,)
- `dt` (float): Time step

**Returns:**
- `rho_next` (torch.Tensor): Density matrix at next time step

**Details:**
- 4th-order accurate: error O(dt⁵)
- Fully differentiable
- More stable than Euler for stiff systems

##### evolve()

Evolve quantum state from t=0 to t=T under piecewise-constant controls.

```python
evolve(
    rho0: torch.Tensor,
    control_sequence: torch.Tensor,
    T: float,
    return_trajectory: bool = False
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]
```

**Parameters:**
- `rho0` (torch.Tensor): Initial state, shape (d, d), complex
- `control_sequence` (torch.Tensor): Control pulses, shape (n_segments, n_controls)
- `T` (float): Total evolution time
- `return_trajectory` (bool): If True, return full trajectory. Default: False

**Returns:**
- `rho_final` (torch.Tensor): Final state, shape (d, d)
- `trajectory` (Optional[torch.Tensor]): Full trajectory if requested, shape (n_segments+1, d, d)

**Important:** This method is **fully differentiable** - gradients flow through the entire simulation!

**Example:**

```python
# Initial state |0⟩
rho0 = torch.tensor([[1, 0], [0, 0]], dtype=torch.complex64)

# Control sequence (learnable parameters)
controls = torch.nn.Parameter(torch.randn(20, 2) * 0.5)

# Forward simulation with gradient tracking
rho_final, trajectory = sim.evolve(rho0, controls, T=1.0, return_trajectory=True)

# Compute loss and backpropagate
purity = torch.trace(rho_final @ rho_final).real
loss = -purity
loss.backward()

# Gradients available!
print(controls.grad.shape)  # (20, 2)
```

##### forward()

PyTorch nn.Module forward pass (alias for evolve without trajectory).

```python
forward(
    rho0: torch.Tensor,
    control_sequence: torch.Tensor,
    T: float
) -> torch.Tensor
```

**Parameters:** Same as `evolve()` (except no `return_trajectory`)

**Returns:**
- `rho_final` (torch.Tensor): Final state, shape (d, d)

---

### Quantum Helper Functions

#### numpy_to_torch_complex()

Convert NumPy complex array to PyTorch complex tensor.

```python
numpy_to_torch_complex(
    array: np.ndarray,
    device: str = 'cpu'
) -> torch.Tensor
```

**Parameters:**
- `array` (np.ndarray): NumPy array (complex or real)
- `device` (str): Target device. Default: 'cpu'

**Returns:**
- `tensor` (torch.Tensor): PyTorch complex64 tensor

**Example:**

```python
import numpy as np
from src.quantum.lindblad_torch import numpy_to_torch_complex

H_np = np.array([[0, 1], [1, 0]], dtype=complex)
H_torch = numpy_to_torch_complex(H_np, device='cuda')
```

#### torch_to_numpy_complex()

Convert PyTorch complex tensor to NumPy complex array.

```python
torch_to_numpy_complex(
    tensor: torch.Tensor
) -> np.ndarray
```

**Parameters:**
- `tensor` (torch.Tensor): PyTorch complex tensor

**Returns:**
- `array` (np.ndarray): NumPy complex array

---

## Noise Models

### NoiseParameters

**Module:** `src.quantum.noise_models`

Dataclass representing task parameters for noise environment.

#### Attributes

```python
@dataclass
class NoiseParameters:
    alpha: float      # Spectral exponent (1/f^α noise)
    A: float          # Amplitude/strength
    omega_c: float    # Cutoff frequency
```

**Physical Meaning:**
- `alpha`: Controls spectral shape (0=white, 1=pink, 2=brown noise)
- `A`: Overall noise strength
- `omega_c`: Characteristic frequency scale

#### Methods

##### to_array()

Convert to numpy array.

```python
to_array() -> np.ndarray
```

**Returns:** Array [alpha, A, omega_c]

##### from_array() (classmethod)

Create from numpy array.

```python
@classmethod
from_array(cls, arr: np.ndarray) -> NoiseParameters
```

**Parameters:**
- `arr` (np.ndarray): Array [alpha, A, omega_c]

**Returns:** NoiseParameters instance

**Example:**

```python
from src.quantum.noise_models import NoiseParameters

# Create noise parameters
theta = NoiseParameters(alpha=1.0, A=0.1, omega_c=5.0)

# Convert to array
arr = theta.to_array()  # [1.0, 0.1, 5.0]

# Reconstruct
theta2 = NoiseParameters.from_array(arr)
```

---

### NoisePSDModel

**Module:** `src.quantum.noise_models`

Power spectral density models for colored noise.

#### Constructor

```python
NoisePSDModel(
    model_type: str = 'one_over_f'
)
```

**Parameters:**
- `model_type` (str): PSD model type. Options:
  - `'one_over_f'`: 1/f^α noise (pink/brown)
  - `'lorentzian'`: Lorentzian (Ornstein-Uhlenbeck)
  - `'double_exp'`: Double-exponential

**Example:**

```python
from src.quantum.noise_models import NoisePSDModel, NoiseParameters
import numpy as np

psd_model = NoisePSDModel(model_type='one_over_f')
theta = NoiseParameters(alpha=1.0, A=0.1, omega_c=5.0)

omega = np.logspace(-1, 2, 1000)
S = psd_model.psd(omega, theta)
```

#### Methods

##### psd()

Compute power spectral density S(ω; θ).

```python
psd(
    omega: np.ndarray,
    theta: NoiseParameters
) -> np.ndarray
```

**Parameters:**
- `omega` (np.ndarray): Frequency array (rad/s)
- `theta` (NoiseParameters): Noise parameters

**Returns:**
- `S` (np.ndarray): PSD values at each frequency

**PSD Formulas:**

**1/f^α noise:**
```
S(ω) = A / (|ω|^α + ω_c^α)
```

**Lorentzian:**
```
S(ω) = A / (ω² + ω_c²)
```

**Double-exponential:**
```
S(ω) = A₁/(ω² + ω²_c1) + A₂/(ω² + ω²_c2)
```

##### correlation_function()

Compute temporal correlation function C(τ).

```python
correlation_function(
    tau: np.ndarray,
    theta: NoiseParameters
) -> np.ndarray
```

**Parameters:**
- `tau` (np.ndarray): Time delay array
- `theta` (NoiseParameters): Noise parameters

**Returns:**
- `C` (np.ndarray): Correlation function values

**Mathematical Definition:**

```
C(τ) = ∫ S(ω) e^{iωτ} dω
```

For Lorentzian (analytical):
```
C(τ) = (A / 2ω_c) exp(-ω_c |τ|)
```

---

### PSDToLindblad

**Module:** `src.quantum.noise_models`

Convert PSD parameters to Lindblad operators.

#### Constructor

```python
PSDToLindblad(
    basis_operators: List[np.ndarray],
    sampling_freqs: np.ndarray,
    psd_model: NoisePSDModel
)
```

**Parameters:**
- `basis_operators` (List[np.ndarray]): Pauli operators [σ_x, σ_y, σ_z] or other basis
- `sampling_freqs` (np.ndarray): Frequencies at which to sample PSD
- `psd_model` (NoisePSDModel): PSD model instance

**Example:**

```python
from src.quantum.noise_models import NoisePSDModel, PSDToLindblad
import numpy as np

# Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

psd_model = NoisePSDModel(model_type='one_over_f')
sampling_freqs = np.array([1.0, 5.0, 10.0])

psd_to_lindblad = PSDToLindblad(
    basis_operators=[sigma_x, sigma_y, sigma_z],
    sampling_freqs=sampling_freqs,
    psd_model=psd_model
)
```

#### Methods

##### get_lindblad_operators()

Get Lindblad operators for given task parameters.

```python
get_lindblad_operators(
    theta: NoiseParameters
) -> List[np.ndarray]
```

**Parameters:**
- `theta` (NoiseParameters): Task parameters

**Returns:**
- `L_ops` (List[np.ndarray]): Lindblad operators [L₁, L₂, ...]

**Mapping:**

```
Γⱼ = S(ω_j; θ)  # Sample PSD at frequency ω_j
Lⱼ = √(Γⱼ) · σⱼ   # Scale basis operator
```

##### get_effective_rates()

Get effective decay rates for each channel.

```python
get_effective_rates(
    theta: NoiseParameters
) -> np.ndarray
```

**Parameters:**
- `theta` (NoiseParameters): Task parameters

**Returns:**
- `rates` (np.ndarray): Effective rates [Γ₁, Γ₂, ...]

---

### TaskDistribution

**Module:** `src.quantum.noise_models`

Distribution P over task parameters Θ.

#### Constructor

```python
TaskDistribution(
    dist_type: str = 'uniform',
    ranges: Dict[str, Tuple[float, float]] = None,
    mean: np.ndarray = None,
    cov: np.ndarray = None
)
```

**Parameters:**
- `dist_type` (str): Distribution type. Options:
  - `'uniform'`: Uniform over box
  - `'gaussian'`: Multivariate Gaussian
- `ranges` (Dict): For uniform, {'alpha': (min, max), 'A': (min, max), 'omega_c': (min, max)}
- `mean` (np.ndarray): For Gaussian, mean vector [3]
- `cov` (np.ndarray): For Gaussian, covariance matrix [3, 3]

**Example:**

```python
from src.quantum.noise_models import TaskDistribution

# Uniform distribution
task_dist = TaskDistribution(
    dist_type='uniform',
    ranges={
        'alpha': (0.5, 2.0),
        'A': (0.05, 0.3),
        'omega_c': (2.0, 8.0)
    }
)

# Sample tasks
tasks = task_dist.sample(n_tasks=100)
```

#### Methods

##### sample()

Sample n tasks from distribution.

```python
sample(
    n_tasks: int,
    rng: np.random.Generator = None
) -> List[NoiseParameters]
```

**Parameters:**
- `n_tasks` (int): Number of tasks to sample
- `rng` (np.random.Generator): Random number generator. Default: new generator

**Returns:**
- `tasks` (List[NoiseParameters]): Sampled tasks

##### compute_variance()

Compute task distribution variance σ²_θ.

```python
compute_variance() -> float
```

**Returns:**
- `sigma_sq` (float): Variance σ²_θ

**Formula:**

For uniform distribution over [a, b]:
```
σ²_θ = Σᵢ (bᵢ - aᵢ)² / 12
```

For Gaussian:
```
σ²_θ = Tr(Cov)
```

---

## Policy Networks

### PulsePolicy

**Module:** `src.meta_rl.policy`

Neural network policy that maps task features → control pulse sequences.

#### Constructor

```python
PulsePolicy(
    task_feature_dim: int = 3,
    hidden_dim: int = 128,
    n_hidden_layers: int = 2,
    n_segments: int = 20,
    n_controls: int = 2,
    output_scale: float = 1.0,
    activation: str = 'tanh'
)
```

**Parameters:**
- `task_feature_dim` (int): Dimension of task encoding (e.g., 3 for [α, A, ω_c]). Default: 3
- `hidden_dim` (int): Hidden layer width. Default: 128
- `n_hidden_layers` (int): Number of hidden layers. Default: 2
- `n_segments` (int): Number of pulse segments. Default: 20
- `n_controls` (int): Number of control channels. Default: 2
- `output_scale` (float): Scale factor for control amplitudes. Default: 1.0
- `activation` (str): Activation function ('tanh', 'relu', 'elu'). Default: 'tanh'

**Architecture:**

```
Input (task_feature_dim)
    ↓
Linear + Activation
    ↓
[Hidden layers] × n_hidden_layers
    ↓
Linear (output_dim = n_segments × n_controls)
    ↓
Reshape to (n_segments, n_controls)
    ↓
Scale by output_scale
```

**Example:**

```python
from src.meta_rl.policy import PulsePolicy
import torch

policy = PulsePolicy(
    task_feature_dim=3,
    hidden_dim=128,
    n_hidden_layers=2,
    n_segments=20,
    n_controls=2,
    output_scale=0.5,
    activation='tanh'
)

print(f"Total parameters: {policy.count_parameters():,}")
```

#### Methods

##### forward()

Generate control pulses for given task.

```python
forward(
    task_features: torch.Tensor
) -> torch.Tensor
```

**Parameters:**
- `task_features` (torch.Tensor): Task encoding, shape (task_feature_dim,) or (batch, task_feature_dim)

**Returns:**
- `controls` (torch.Tensor): Control pulses, shape (n_segments, n_controls) or (batch, n_segments, n_controls)

**Example:**

```python
# Single task
task_features = torch.tensor([1.0, 0.1, 5.0])  # [α, A, ω_c]
controls = policy(task_features)  # Shape: (20, 2)

# Batch of tasks
batch_features = torch.randn(32, 3)
batch_controls = policy(batch_features)  # Shape: (32, 20, 2)
```

##### get_lipschitz_constant()

Estimate Lipschitz constant via spectral norms.

```python
get_lipschitz_constant() -> float
```

**Returns:**
- `L_net` (float): Estimated Lipschitz constant

**Formula:**

```
L_net ≤ ∏ℓ ||Wℓ||₂
```

where ||Wℓ||₂ is the spectral norm of layer ℓ.

##### count_parameters()

Count total trainable parameters.

```python
count_parameters() -> int
```

**Returns:**
- `n_params` (int): Number of parameters

---

### TaskFeatureEncoder

**Module:** `src.meta_rl.policy`

Learn task representations from raw noise parameters using Fourier features or MLPs.

#### Constructor

```python
TaskFeatureEncoder(
    raw_dim: int = 3,
    feature_dim: int = 16,
    use_fourier: bool = True,
    fourier_scale: float = 1.0
)
```

**Parameters:**
- `raw_dim` (int): Raw feature dimension. Default: 3
- `feature_dim` (int): Output feature dimension. Default: 16
- `use_fourier` (bool): Use random Fourier features. Default: True
- `fourier_scale` (float): Scale for Fourier feature matrix. Default: 1.0

**Fourier Features:**

```
φ(x) = [cos(Bx), sin(Bx)]
```

where B ~ N(0, σ²I) is a random matrix.

**Example:**

```python
from src.meta_rl.policy import TaskFeatureEncoder
import torch

encoder = TaskFeatureEncoder(raw_dim=3, feature_dim=16, use_fourier=True)

raw_features = torch.tensor([1.0, 0.1, 5.0])
encoded = encoder(raw_features)  # Shape: (16,)
```

---

## Meta-Learning

### MAML

**Module:** `src.meta_rl.maml`

Model-Agnostic Meta-Learning algorithm for quantum control policies.

#### Constructor

```python
MAML(
    policy: nn.Module,
    inner_lr: float = 0.01,
    inner_steps: int = 5,
    meta_lr: float = 0.001,
    first_order: bool = False,
    device: torch.device = torch.device('cpu')
)
```

**Parameters:**
- `policy` (nn.Module): Policy network (meta-initialization π₀)
- `inner_lr` (float): Inner loop learning rate α. Default: 0.01
- `inner_steps` (int): Number K of inner gradient steps. Default: 5
- `meta_lr` (float): Meta-learning rate β. Default: 0.001
- `first_order` (bool): Use first-order MAML (FOMAML). Default: False
- `device` (torch.device): Torch device. Default: CPU

**Algorithm:**

```
1. Sample tasks θ₁, ..., θ_B ~ P
2. For each task θᵢ:
   a. Clone π₀ → φᵢ
   b. K gradient steps: φᵢ ← φᵢ - α ∇_φ L(φᵢ; θᵢ)
   c. Compute query loss: L_query(φᵢ; θᵢ)
3. Meta-update: π₀ ← π₀ - β ∇_π₀ Σᵢ L_query(φᵢ; θᵢ)
```

**Example:**

```python
from src.meta_rl.maml import MAML
from src.meta_rl.policy import PulsePolicy

policy = PulsePolicy(task_feature_dim=3, hidden_dim=64, n_segments=20)

maml = MAML(
    policy=policy,
    inner_lr=0.01,
    inner_steps=5,
    meta_lr=0.001,
    first_order=False
)
```

#### Methods

##### inner_loop()

Perform K-step adaptation on a single task.

```python
inner_loop(
    task_data: Dict,
    loss_fn: Callable,
    num_steps: Optional[int] = None
) -> Tuple[nn.Module, List[float]]
```

**Parameters:**
- `task_data` (Dict): Dictionary with 'support' and 'query' data
- `loss_fn` (Callable): Loss function L(policy, data) → scalar
- `num_steps` (int): Number of steps. Default: self.inner_steps

**Returns:**
- `adapted_policy` (nn.Module): Policy after K adaptation steps
- `losses` (List[float]): Losses at each step

**Example:**

```python
task_data = {
    'support': {'task_features': torch.randn(10, 3), ...},
    'query': {'task_features': torch.randn(10, 3), ...}
}

def loss_fn(policy, data):
    controls = policy(data['task_features'])
    # ... compute fidelity loss
    return loss

adapted_policy, losses = maml.inner_loop(task_data, loss_fn)
```

##### inner_loop_higher()

Inner loop using `higher` library for second-order MAML.

```python
inner_loop_higher(
    task_data: Dict,
    loss_fn: Callable,
    num_steps: Optional[int] = None
) -> Tuple
```

**Parameters:** Same as `inner_loop()`

**Returns:**
- `fmodel`: Functional model after adaptation (from `higher` library)
- `losses` (List[float]): Inner loop losses

**Note:** Requires `higher` library: `pip install higher`

##### meta_train_step()

Single meta-training step on a batch of tasks.

```python
meta_train_step(
    task_batch: List[Dict],
    loss_fn: Callable,
    use_higher: bool = True
) -> Dict[str, float]
```

**Parameters:**
- `task_batch` (List[Dict]): List of task dictionaries
- `loss_fn` (Callable): Loss function
- `use_higher` (bool): Use higher library for second-order gradients. Default: True

**Returns:**
- `metrics` (Dict[str, float]): Training metrics
  - `'meta_loss'`: Average query loss
  - `'mean_task_loss'`: Mean across tasks
  - `'std_task_loss'`: Std dev across tasks
  - `'min_task_loss'`: Minimum task loss
  - `'max_task_loss'`: Maximum task loss

**Example:**

```python
task_batch = [task_data_1, task_data_2, task_data_3, task_data_4]
metrics = maml.meta_train_step(task_batch, loss_fn, use_higher=True)

print(f"Meta loss: {metrics['meta_loss']:.4f}")
```

##### meta_validate()

Evaluate meta-learned initialization on validation tasks.

```python
meta_validate(
    val_tasks: List[Dict],
    loss_fn: Callable
) -> Dict[str, float]
```

**Parameters:**
- `val_tasks` (List[Dict]): Validation task batch
- `loss_fn` (Callable): Loss function

**Returns:**
- `metrics` (Dict[str, float]): Validation metrics
  - `'val_loss_pre_adapt'`: Loss before adaptation
  - `'val_loss_post_adapt'`: Loss after K adaptation steps
  - `'adaptation_gain'`: Improvement from adaptation
  - `'std_post_adapt'`: Std dev of post-adaptation losses

##### save_checkpoint()

Save meta-learned initialization and training state.

```python
save_checkpoint(
    path: str,
    epoch: int,
    **kwargs
)
```

**Parameters:**
- `path` (str): Save path
- `epoch` (int): Current epoch
- `**kwargs`: Additional metadata to save

**Saved Contents:**
- Policy state dict
- Meta-optimizer state
- Inner loop hyperparameters
- Training history
- Custom metadata

##### load_checkpoint()

Load meta-learned initialization and training state.

```python
load_checkpoint(
    path: str
) -> int
```

**Parameters:**
- `path` (str): Checkpoint path

**Returns:**
- `epoch` (int): Epoch from checkpoint

---

### MAMLTrainer

**Module:** `src.meta_rl.maml`

High-level trainer for MAML experiments with task sampling and logging.

#### Constructor

```python
MAMLTrainer(
    maml: MAML,
    task_sampler: Callable,
    data_generator: Callable,
    loss_fn: Callable,
    n_support: int = 10,
    n_query: int = 10,
    log_interval: int = 10,
    val_interval: int = 50
)
```

**Parameters:**
- `maml` (MAML): MAML instance
- `task_sampler` (Callable): Function that samples tasks from P
- `data_generator` (Callable): Function that generates support/query data
- `loss_fn` (Callable): Loss function
- `n_support` (int): Number of support trajectories per task. Default: 10
- `n_query` (int): Number of query trajectories per task. Default: 10
- `log_interval` (int): Log every N iterations. Default: 10
- `val_interval` (int): Validate every N iterations. Default: 50

#### Methods

##### train()

Main training loop.

```python
train(
    n_iterations: int,
    tasks_per_batch: int = 4,
    val_tasks: int = 20,
    save_path: Optional[str] = None
)
```

**Parameters:**
- `n_iterations` (int): Number of meta-training iterations
- `tasks_per_batch` (int): Tasks per meta-batch. Default: 4
- `val_tasks` (int): Number of validation tasks. Default: 20
- `save_path` (str): Path to save checkpoints. Default: None

**Example:**

```python
from src.meta_rl.maml import MAML, MAMLTrainer

trainer = MAMLTrainer(
    maml=maml,
    task_sampler=task_sampler,
    data_generator=data_generator,
    loss_fn=loss_fn,
    n_support=10,
    n_query=10
)

trainer.train(
    n_iterations=2000,
    tasks_per_batch=4,
    val_tasks=20,
    save_path='checkpoints/meta_policy.pt'
)
```

---

## Theory & Analysis

### OptimalityGapComputer

**Module:** `src.theory.optimality_gap`

Compute and analyze optimality gaps between meta-learning and robust control.

#### Constructor

```python
OptimalityGapComputer(
    quantum_system: Callable,
    fidelity_fn: Callable,
    device: torch.device = torch.device('cpu')
)
```

**Parameters:**
- `quantum_system` (Callable): Function that simulates quantum dynamics
- `fidelity_fn` (Callable): Function that computes fidelity
- `device` (torch.device): Torch device. Default: CPU

#### Methods

##### compute_gap()

Compute empirical optimality gap.

```python
compute_gap(
    meta_policy: torch.nn.Module,
    robust_policy: torch.nn.Module,
    task_distribution: List,
    n_samples: int = 100,
    K: int = 5,
    inner_lr: float = 0.01
) -> Dict[str, float]
```

**Parameters:**
- `meta_policy` (nn.Module): Meta-learned initialization
- `robust_policy` (nn.Module): Robust baseline policy
- `task_distribution` (List): Tasks to sample from
- `n_samples` (int): Number of tasks. Default: 100
- `K` (int): Adaptation steps. Default: 5
- `inner_lr` (float): Inner learning rate. Default: 0.01

**Returns:**
- `results` (Dict[str, float]): Gap metrics
  - `'gap'`: Optimality gap
  - `'meta_fidelity_mean'`: Mean fidelity for meta-policy
  - `'meta_fidelity_std'`: Std dev for meta-policy
  - `'robust_fidelity_mean'`: Mean fidelity for robust policy
  - `'robust_fidelity_std'`: Std dev for robust policy
  - `'meta_fidelities'`: Individual fidelities (list)
  - `'robust_fidelities'`: Individual fidelities (list)

**Gap Formula:**

```
Gap = E_θ[F(AdaptK(π_meta; θ), θ)] - E_θ[F(π_rob, θ)]
```

**Example:**

```python
from src.theory.optimality_gap import OptimalityGapComputer

gap_computer = OptimalityGapComputer(
    quantum_system=quantum_env.simulate,
    fidelity_fn=quantum_env.compute_fidelity
)

results = gap_computer.compute_gap(
    meta_policy=meta_policy,
    robust_policy=robust_policy,
    task_distribution=test_tasks,
    n_samples=100,
    K=5,
    inner_lr=0.01
)

print(f"Optimality Gap: {results['gap']:.4f}")
print(f"Meta fidelity: {results['meta_fidelity_mean']:.4f} ± {results['meta_fidelity_std']:.4f}")
print(f"Robust fidelity: {results['robust_fidelity_mean']:.4f} ± {results['robust_fidelity_std']:.4f}")
```

##### estimate_constants()

Estimate theoretical constants from data.

```python
estimate_constants(
    policy: torch.nn.Module,
    task_distribution: List,
    n_samples: int = 50
) -> GapConstants
```

**Parameters:**
- `policy` (nn.Module): Policy network
- `task_distribution` (List): Task distribution
- `n_samples` (int): Number of samples for estimation. Default: 50

**Returns:**
- `constants` (GapConstants): Estimated constants (C_sep, μ, L, L_F, C_K)

**Example:**

```python
constants = gap_computer.estimate_constants(
    policy=policy,
    task_distribution=train_tasks,
    n_samples=50
)

print(f"C_sep = {constants.C_sep:.4f}")
print(f"μ = {constants.mu:.4f}")
print(f"L = {constants.L:.4f}")
print(f"L_F = {constants.L_F:.4f}")

# Compute theoretical gap bound
sigma_sq = task_dist.compute_variance()
gap_bound = constants.gap_lower_bound(sigma_sq, K=5, eta=0.01)
print(f"Theoretical gap bound: {gap_bound:.4f}")
```

---

### GapConstants

**Module:** `src.theory.optimality_gap`

Dataclass storing constants in optimality gap bound.

#### Attributes

```python
@dataclass
class GapConstants:
    C_sep: float   # Task-optimal policy separation
    mu: float      # Strong convexity / PL constant
    L: float       # Lipschitz constant (fidelity vs task)
    L_F: float     # Lipschitz constant (fidelity vs policy)
    C_K: float     # Inner loop Lipschitz constant
```

#### Methods

##### gap_lower_bound()

Compute theoretical lower bound on gap.

```python
gap_lower_bound(
    sigma_sq: float,
    K: int,
    eta: float
) -> float
```

**Parameters:**
- `sigma_sq` (float): Task variance σ²_θ
- `K` (int): Number of adaptation steps
- `eta` (float): Inner learning rate

**Returns:**
- `gap_bound` (float): Lower bound on gap

**Formula:**

```
Gap(P, K) ≥ c_gap · σ²_θ · (1 - e^(-μηK))
where c_gap = C_sep · L_F · L²
```

---

## Baseline Methods

### RobustPolicy

**Module:** `src.baselines.robust_control`

Train a policy to be robust across task distribution (no adaptation).

#### Constructor

```python
RobustPolicy(
    policy: nn.Module,
    learning_rate: float = 0.001,
    robust_type: str = 'average',
    device: torch.device = torch.device('cpu')
)
```

**Parameters:**
- `policy` (nn.Module): Policy network
- `learning_rate` (float): Learning rate. Default: 0.001
- `robust_type` (str): Robustness criterion. Options:
  - `'average'`: min E_θ[L(π, θ)]
  - `'minimax'`: min max_θ L(π, θ)
  - `'cvar'`: min CVaR_α[L(π, θ)]
- `device` (torch.device): Torch device. Default: CPU

**Example:**

```python
from src.baselines.robust_control import RobustPolicy
from src.meta_rl.policy import PulsePolicy

policy = PulsePolicy(task_feature_dim=3, hidden_dim=64, n_segments=20)

robust = RobustPolicy(
    policy=policy,
    learning_rate=0.001,
    robust_type='average'
)
```

#### Methods

##### train_step()

Single training step.

```python
train_step(
    task_batch: List[Dict],
    loss_fn: Callable
) -> Dict[str, float]
```

**Parameters:**
- `task_batch` (List[Dict]): Batch of tasks with data
- `loss_fn` (Callable): Loss function

**Returns:**
- `metrics` (Dict[str, float]): Training metrics

**Example:**

```python
task_batch = [task_data_1, task_data_2, task_data_3, task_data_4]
metrics = robust.train_step(task_batch, loss_fn)

print(f"Robust loss: {metrics['loss']:.4f}")
```

##### evaluate()

Evaluate robust policy on test tasks.

```python
evaluate(
    test_tasks: List[Dict],
    loss_fn: Callable
) -> Dict[str, float]
```

**Parameters:**
- `test_tasks` (List[Dict]): Test tasks
- `loss_fn` (Callable): Loss function

**Returns:**
- `metrics` (Dict[str, float]): Evaluation metrics

---

### GRAPEOptimizer

**Module:** `src.baselines.robust_control`

Gradient Ascent Pulse Engineering for direct pulse optimization (classical baseline).

See `docs/GRAPE_BASELINE.md` for detailed documentation.

---

## Environment Bridge

### QuantumEnvironment

**Module:** `src.theory.quantum_environment`

Unified quantum environment bridge with caching for efficient simulation.

#### Constructor

```python
QuantumEnvironment(
    H0: np.ndarray,
    H_controls: List[np.ndarray],
    psd_to_lindblad: Callable,
    target_state: np.ndarray,
    T: float = 1.0,
    n_segments: int = 20,
    use_torch: bool = True,
    device: str = 'cpu'
)
```

**Parameters:**
- `H0` (np.ndarray): Drift Hamiltonian
- `H_controls` (List[np.ndarray]): Control Hamiltonians
- `psd_to_lindblad` (Callable): Function mapping task params → Lindblad operators
- `target_state` (np.ndarray): Target quantum state
- `T` (float): Evolution time. Default: 1.0
- `n_segments` (int): Number of control segments. Default: 20
- `use_torch` (bool): Use differentiable PyTorch simulator. Default: True
- `device` (str): Device for simulation. Default: 'cpu'

**Example:**

```python
from src.theory.quantum_environment import QuantumEnvironment
import numpy as np

# Define system
H0 = 0.5 * sigma_z
H_controls = [sigma_x, sigma_y]
target = np.array([1, 0]) / np.sqrt(2)  # |+⟩ state

env = QuantumEnvironment(
    H0=H0,
    H_controls=H_controls,
    psd_to_lindblad=psd_to_lindblad.get_lindblad_operators,
    target_state=target,
    T=1.0,
    n_segments=20,
    use_torch=True,
    device='cuda'
)
```

#### Methods

##### simulate()

Simulate quantum evolution for given controls and task.

```python
simulate(
    controls: np.ndarray,
    task_params: NoiseParameters
) -> np.ndarray
```

**Parameters:**
- `controls` (np.ndarray): Control sequence, shape (n_segments, n_controls)
- `task_params` (NoiseParameters): Task parameters

**Returns:**
- `rho_final` (np.ndarray): Final density matrix

**Features:**
- Automatic caching by task parameters
- Differentiable if use_torch=True

##### compute_fidelity()

Compute fidelity of final state with target.

```python
compute_fidelity(
    rho_final: np.ndarray
) -> float
```

**Parameters:**
- `rho_final` (np.ndarray): Final density matrix

**Returns:**
- `fidelity` (float): State fidelity ⟨ψ_target| ρ |ψ_target⟩

---

## Complete Example

Here's a complete example bringing everything together:

```python
import torch
import numpy as np
from src.quantum.lindblad_torch import DifferentiableLindbladSimulator
from src.quantum.noise_models import (
    TaskDistribution, NoisePSDModel, PSDToLindblad, NoiseParameters
)
from src.meta_rl.policy import PulsePolicy
from src.meta_rl.maml import MAML, MAMLTrainer
from src.theory.optimality_gap import OptimalityGapComputer
from src.baselines.robust_control import RobustPolicy

# 1. Define quantum system (1-qubit)
sigma_x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
sigma_y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)
sigma_z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)

H0 = 0.5 * sigma_z
H_controls = [sigma_x, sigma_y]

# 2. Define task distribution
task_dist = TaskDistribution(
    dist_type='uniform',
    ranges={'alpha': (0.5, 2.0), 'A': (0.05, 0.3), 'omega_c': (2.0, 8.0)}
)

# 3. Create PSD model and mapping
psd_model = NoisePSDModel(model_type='one_over_f')
psd_to_lindblad = PSDToLindblad(
    basis_operators=[sigma_z.numpy()],
    sampling_freqs=np.array([5.0]),
    psd_model=psd_model
)

# 4. Create policy
policy = PulsePolicy(
    task_feature_dim=3,
    hidden_dim=128,
    n_segments=20,
    n_controls=2
)

# 5. Initialize MAML
maml = MAML(
    policy=policy,
    inner_lr=0.01,
    inner_steps=5,
    meta_lr=0.001,
    first_order=False
)

# 6. Create trainer and train
trainer = MAMLTrainer(
    maml=maml,
    task_sampler=task_dist.sample,
    data_generator=your_data_generator,
    loss_fn=your_loss_fn
)

trainer.train(n_iterations=2000, tasks_per_batch=4, save_path='checkpoints/meta.pt')

# 7. Train robust baseline
robust_policy = PulsePolicy(task_feature_dim=3, hidden_dim=128, n_segments=20)
robust = RobustPolicy(robust_policy, robust_type='average')

# ... train robust policy ...

# 8. Evaluate optimality gap
gap_computer = OptimalityGapComputer(
    quantum_system=your_quantum_system,
    fidelity_fn=your_fidelity_fn
)

results = gap_computer.compute_gap(
    meta_policy=maml.policy,
    robust_policy=robust.policy,
    task_distribution=test_tasks,
    K=5
)

print(f"Optimality Gap: {results['gap']:.4f}")

# 9. Estimate constants and compare to theory
constants = gap_computer.estimate_constants(policy, train_tasks)
sigma_sq = task_dist.compute_variance()
theoretical_gap = constants.gap_lower_bound(sigma_sq, K=5, eta=0.01)

print(f"Empirical gap: {results['gap']:.4f}")
print(f"Theoretical bound: {theoretical_gap:.4f}")
```

---

## See Also

- [Theory Guide](THEORY_GUIDE.md) - Mathematical foundations
- [Quick Start Guide](QUICKSTART.md) - Getting started
- [Implementation Summary](IMPLEMENTATION_SUMMARY.md) - Technical details
- [GRAPE Baseline](GRAPE_BASELINE.md) - GRAPE optimizer documentation

---

**Last Updated:** 2025-01-22
**Version:** 1.0.0
