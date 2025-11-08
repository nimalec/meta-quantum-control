import os
import sys
import numpy as np
import torch
import yaml 
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Tuple
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from typer import Typer
from pathlib import Path

from metaqctrl.quantum.lindblad import LindbladSimulator
from metaqctrl.quantum.noise_models import TaskDistribution, NoiseParameters, PSDToLindblad
from metaqctrl.quantum.gates import state_fidelity
from metaqctrl.meta_rl.policy import PulsePolicy
from metaqctrl.meta_rl.maml import MAML
from metaqctrl.theory.quantum_environment import create_quantum_environment
from metaqctrl.theory.optimality_gap import OptimalityGapComputer, GapConstants
from metaqctrl.theory.physics_constants import compute_control_relevant_variance
from metaqctrl.utils.checkpoint_utils import load_policy_from_checkpoint

app = Typer()


def create_task_distributions_with_varying_variance(
    base_config: Dict,
    variance_levels: List[float] = [0.001, 0.002, 0.004, 0.008, 0.016]
) -> List[Tuple[TaskDistribution, float]]:
    """
    Create task distributions with different variances

    FIXED: Use direct range scaling without clipping to ensure monotonic variance

    Strategy: Scale ranges around center proportionally to variance_scale
    Small variance → narrow ranges (tasks are similar)
    Large variance → wide ranges (tasks are diverse)
    """
    distributions = []

    # Base configuration (full ranges from config)
    alpha_full_range = base_config.get('alpha_range', [0.1, 4.0])
    A_full_range = base_config.get('A_range', [10, 100000])
    omega_c_full_range = base_config.get('omega_c_range', [1, 100])

    # Centers
    alpha_center = np.mean(alpha_full_range)
    A_center = np.mean(A_full_range)
    omega_c_center = np.mean(omega_c_full_range)

    # Maximum half-widths
    alpha_max_hw = (alpha_full_range[1] - alpha_full_range[0]) / 2
    A_max_hw = (A_full_range[1] - A_full_range[0]) / 2
    omega_c_max_hw = (omega_c_full_range[1] - omega_c_full_range[0]) / 2
    
    
    

    for var_scale in variance_levels:
        # Normalize var_scale to [0, 1] range
        # Assume max variance corresponds to full parameter ranges
        # Scale factor: sqrt for variance -> std -> range
        #scale_factor = np.sqrt(var_scale / max(variance_levels))
        scale_factor  = var_scale 

        # Scale half-widths proportionally
        alpha_hw = alpha_max_hw * scale_factor
        A_hw = A_max_hw * scale_factor
        omega_c_hw = omega_c_max_hw * scale_factor
 
        task_dist = TaskDistribution(
            dist_type='uniform',
            ranges={
                'alpha': (alpha_full_range[0],   alpha_full_range[1]),          
                'A':  (scale_factor* A_full_range[0], scale_factor*A_full_range[1]),    
                'omega_c': (omega_c_full_range[0], omega_c_full_range[1])
            }
        )        
        

        # Compute expected parameter variance for this distribution
        expected_var = task_dist.compute_variance()  
        distributions.append((task_dist, expected_var))  
    return distributions

def task_sampler(n_tasks: int, task_dist: TaskDistribution):  
    local_rng = np.random.default_rng() 
    return task_dist.sample(n_tasks, local_rng)

# Load config
config_path = '../configs/experiment_config.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
  
variance_levels = [2, 1, 0.5, 0.2, 0.1, 0.01, 0.001, 0.0001]
dist = create_task_distributions_with_varying_variance(config,variance_levels)

#dist_3 = task_sampler(10, dist[7][0])
#print(dist_3)
#print(dist_1, '\n')




