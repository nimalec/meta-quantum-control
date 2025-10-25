# Paper Results Generation

This directory contains scripts to generate all core results and figures for the ICML 2026 paper:

**"Meta-Reinforcement Learning for Quantum Control: A Framework for Quantifying Adaptation Gains under Noise Shifts"**

## Overview

The codebase validates the central theoretical prediction:

```
Gap(P, K) ≥ c_quantum * σ²_S * (1 - e^(-μηK))
```

Where:
- **Gap(P, K)**: Optimality gap between meta-learning and robust control
- **σ²_S**: Control-relevant task variance
- **K**: Number of adaptation steps
- **μ**: PL constant (related to spectral gap Δ)
- **c_quantum**: System-dependent constant

## Quick Start

### 1. Train Policies (Prerequisites)

First, train the meta-learned and robust baseline policies:

```bash
# Train meta-learned policy (MAML)
cd experiments
python train_meta.py --config ../configs/experiment_config.yaml --output checkpoints/maml_best.pt

# Train robust baseline
python train_robust.py --config ../configs/experiment_config.yaml --output checkpoints/robust_best.pt
```

### 2. Generate All Results

Run the master script to generate all figures and tables:

```bash
cd experiments/paper_results
python generate_all_results.py --meta_path ../checkpoints/maml_best.pt \
                                --robust_path ../checkpoints/robust_best.pt \
                                --n_tasks 100
```

This will:
- ✓ Run all experiments
- ✓ Generate all figures (PDF format)
- ✓ Compute all constants
- ✓ Validate theoretical predictions
- ✓ Create summary table

**Expected runtime:** ~30-60 minutes (depending on hardware)

## Individual Experiments

You can also run experiments individually:

### Experiment 1: Gap vs Adaptation Steps K

Validates: **Gap(P, K) ∝ (1 - e^(-μηK))**
Expected R²: **≥ 0.96**

```bash
python experiment_gap_vs_k.py
```

**Generates:**
- `results/gap_vs_k/figure.pdf` - Figure showing exponential convergence
- `results/gap_vs_k/results.json` - Raw data and fit parameters

### Experiment 2: Gap vs Task Variance σ²_S

Validates: **Gap(P, K) ∝ σ²_S**
Expected R²: **≥ 0.94**

```bash
python experiment_gap_vs_variance.py
```

**Generates:**
- `results/gap_vs_variance/figure.pdf` - Figure showing linear scaling
- `results/gap_vs_variance/results.json` - Raw data and fit parameters

### Experiment 3: Physics Constants Validation

Validates: **Empirical constants within 2× of theoretical predictions**

```bash
python experiment_constants_validation.py
```

**Generates:**
- `results/constants_validation/constants_visualization.pdf` - Constant distributions
- `results/constants_validation/constants.json` - All estimated constants

**Key Constants Estimated:**
- **Δ_min**: Minimum spectral gap (dissipation timescale)
- **μ_min**: Minimum PL constant (landscape curvature)
- **C_filter**: Filter separation constant (control sensitivity to noise)
- **σ²_S**: Control-relevant task variance
- **c_quantum**: Combined system constant

## Output Structure

After running all experiments, you'll have:

```
results/paper/
├── gap_vs_k/
│   ├── figure.pdf                    # Figure 1: Gap vs K (exponential fit)
│   └── results.json                  # Raw data
├── gap_vs_variance/
│   ├── figure.pdf                    # Figure 2: Gap vs σ²_S (linear fit)
│   └── results.json                  # Raw data
├── constants_validation/
│   ├── constants_visualization.pdf   # Figure 3: Constant distributions
│   └── constants.json                # All constants
└── summary_table.txt                 # Table 1: Complete summary
```

## Expected Results

Based on the paper (Section 5):

| Metric | Target | Interpretation |
|--------|--------|----------------|
| **R² (Gap vs K)** | ≥ 0.96 | Exponential convergence with adaptation |
| **R² (Gap vs σ²_S)** | ≥ 0.94 | Linear scaling with task diversity |
| **μ_empirical / μ_theory** | 0.5-2.0× | PL constant within factor of 2 |
| **c_empirical / c_theory** | 0.5-2.0× | Combined constant within factor of 2 |

### Sample Output

```
==============================================================================
TABLE 1: Theoretical Constants and Empirical Validation
==============================================================================

Physics Constants:
------------------------------------------------------------------------------
  Δ_min (spectral gap)       : 0.1250
  μ_min (PL constant)        : 0.003125
  C_filter (separation)      : 0.0320
  σ²_S (task variance)       : 0.0042
  c_quantum (combined)       : 0.0320

Theoretical Predictions:
------------------------------------------------------------------------------
  μ_theory                   : 0.003906
  c_quantum_theory           : 0.0256

Empirical / Theory Ratios:
------------------------------------------------------------------------------
  μ_empirical / μ_theory     : 0.80x  ✓
  c_empirical / c_theory     : 1.25x  ✓

Scaling Laws Validation:
------------------------------------------------------------------------------
  Gap vs K (exponential):    R² = 0.9612  ✓ (target: 0.96)
    - gap_max                : 0.0450
    - μη                     : 0.0312

  Gap vs σ²_S (linear):      R² = 0.9423  ✓ (target: 0.94)
    - slope (Gap/σ²_S)       : 10.71

==============================================================================

✓✓✓ ALL VALIDATIONS PASSED ✓✓✓
```

## Configuration

Modify parameters in the scripts or pass via config:

### System Parameters
- `num_qubits`: Qubit dimension (default: 1, d=2)
- `num_controls`: Number of control Hamiltonians (default: 2)
- `num_segments`: Control discretization (default: 20)
- `evolution_time`: Gate time T (default: 1.0)

### Task Distribution
- `alpha_range`: Spectral exponent [0.5, 2.0]
- `A_range`: Noise amplitude [0.05, 0.3]
- `omega_c_range`: Cutoff frequency [2.0, 8.0] rad/s

### Meta-Learning
- `inner_lr`: Inner loop learning rate η (default: 0.01)
- `k_values`: Adaptation steps to test (default: [1, 2, 3, 5, 7, 10, 15, 20])

## Troubleshooting

### "ModuleNotFoundError"
Ensure you're running from the `experiments/paper_results/` directory and the `src/` directory is in your Python path.

### "Trained models not found"
Run `train_meta.py` and `train_robust.py` first to generate the checkpoints.

### "R² below target"
- Increase `n_tasks` for more stable estimates (100-200 recommended)
- Check that policies are well-trained (validation fidelity > 0.9)
- Verify environment setup is correct

### "Constants outside 2× bounds"
This is acceptable given heuristic components in the theory (see paper Section 4.2). The framework still provides accurate *scaling* predictions even if absolute constants differ by a factor.

## Citation

If you use these scripts, please cite:

```bibtex
@inproceedings{leclerc2025metarl,
  title={Meta-Reinforcement Learning for Quantum Control: A Framework for Quantifying Adaptation Gains under Noise Shifts},
  author={Leclerc, Nima and Brawand, Nicholas},
  booktitle={Proceedings of the 42nd International Conference on Machine Learning},
  year={2025}
}
```

## Contact

For questions or issues:
- **Email**: nleclerc@mitre.org
- **GitHub Issues**: [Create an issue](https://github.com/yourusername/meta-quantum-control/issues)

## License

Copyright 2025 MITRE Corporation. All rights reserved.
