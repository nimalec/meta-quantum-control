<<<<<<< HEAD
"""
Noise Models and PSD Parameterization (Backward Compatibility Wrapper)

This module provides backward compatibility by re-exporting all classes from noise_models_v2.py.

All existing code that imports from this module will automatically get:
- NoiseParameters with model_type support (4D features)
- TaskDistribution with mixed model sampling
- PSDToLindblad with dynamic model selection

For new code, you can import directly from noise_models_v2 or use this module.

IMPORTANT FOR BACKWARD COMPATIBILITY:
------------------------------------
Old code that uses 3D task features will continue to work:
    task.to_array(include_model=False)  # Returns 3D array [alpha, A, omega_c]

New code can use 4D features for mixed models:
    task.to_array(include_model=True)   # Returns 4D array [alpha, A, omega_c, model_encoding]

Mixed Model Support:
-------------------
To enable mixed model sampling in any script, add to your config:
    model_types: ['one_over_f', 'lorentzian']
    model_probs: [0.5, 0.5]  # Optional, defaults to uniform

See MIXED_MODELS_GUIDE.md for complete documentation.
"""

# Re-export everything from noise_models_v2
from metaqctrl.quantum.noise_models_v2 import (
    NoiseParameters,
    NoisePSDModel,
    PSDToLindblad,
    TaskDistribution,
    psd_distance,
    TWO_PI,
    HBAR
)
=======
import numpy as np
from scipy.special import gamma as gamma_func
from typing import Tuple, List, Dict
from dataclasses import dataclass
>>>>>>> 7fb7d9e310f4c4d0cfa3d62371195b2e94509f7d

# Maintain backward compatibility with old imports
__all__ = [
    'NoiseParameters',
    'NoisePSDModel',
    'PSDToLindblad',
    'TaskDistribution',
    'psd_distance',
    'TWO_PI',
    'HBAR'
]

# Version information
__version__ = '2.0.0'  # v2 with backward compatibility
__compatibilty_note__ = """
This is v2 of the noise models with mixed model sampling support.
All v1 code remains compatible. New features:
- NoiseParameters.model_type field
- TaskDistribution.model_types parameter for mixed sampling
- PSDToLindblad dynamic model selection
- 4D task features (backward compatible with 3D)
"""
