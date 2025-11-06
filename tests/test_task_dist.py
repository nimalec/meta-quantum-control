from metaqctrl.quantum.noise_models_v2 import NoiseParameters
import numpy as np 

# Create task with model type
task = NoiseParameters(
    alpha=1.0,
    A=0.01,
    omega_c=500.0,
    model_type='lorentzian'
)

# Convert to array (4D)
arr = task.to_array(include_model=True)  # [1.0, 0.01, 500.0, 1.0]

# Convert to array (3D, backward compatible)
#arr = task.to_array(include_model=False)  # [1.0, 0.01, 500.0]


# Decode from array
task2 = NoiseParameters.from_array(arr, has_model=True)

from metaqctrl.quantum.noise_models_v2 import TaskDistribution

# Mixed model distribution
dist = TaskDistribution(
    dist_type='uniform',
    ranges={'alpha': (0.5, 2.0), 'A': (0.001, 0.01), 'omega_c': (100, 1000)},
    model_types=['one_over_f', 'lorentzian'],
    model_probs=[0.6, 0.4]  # 60% 1/f, 40% Lorentzian
)

# Sample tasks
tasks = dist.sample(n_tasks=100)
#tasks[i].model_type is randomly selected according to model_probs

# # Compute variance (includes model contribution)
sigma_squared = dist.compute_variance()

from metaqctrl.quantum.noise_adapter import PSDToLindblad
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
# sig_p = np.array([[0, 1], [0, 0]], dtype=complex)

# Dynamic model selection (for mixed models)
# converter = PSDToLindblad(
#     basis_operators=[sigma_z],
#     sampling_freqs=omega_sample,
#     psd_model=None,  # None → dynamic selection based on task.model_type
#     T=1.0,
#     sequence='ramsey',
#     omega0=omega0,
# )
#     g_energy_per_xi=hbar/2

# Automatically uses correct model
#task1 = NoiseParameters(..., model_type='one_over_f')
# #L_ops1 = converter.get_lindblad_operators(task1)  # Uses 1/f PSD

# task2 = NoiseParameters(alpha=1.0, A=0.01, omega_c=500.0, model_type='lorentzian')
# L_ops2 = converter.get_lindblad_operators(task2)  # Uses Lorentzian PSD  