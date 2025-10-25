# Quick Guide: Generating Paper Results

## TL;DR

```bash
# 1. Train policies (if not already done)
python experiments/train_meta.py --config configs/experiment_config.yaml
python experiments/train_robust.py --config configs/experiment_config.yaml

# 2. Generate all paper results
cd experiments/paper_results
python generate_all_results.py

# 3. View results
ls -R results/paper/
```

## What Gets Generated

### Core Figures (matching paper)

1. **Figure 1: Gap vs Adaptation Steps K**
   - File: `results/paper/gap_vs_k/figure.pdf`
   - Validates: Gap(P, K) ∝ (1 - e^(-μηK))
   - Expected: R² ≈ 0.96

2. **Figure 2: Gap vs Task Variance σ²_S**
   - File: `results/paper/gap_vs_variance/figure.pdf`
   - Validates: Gap(P, K) ∝ σ²_S
   - Expected: R² ≈ 0.94

3. **Figure 3: Constants Validation**
   - File: `results/paper/constants_validation/constants_visualization.pdf`
   - Shows distributions of Δ, μ, C_filter
   - Validates empirical vs theoretical

### Tables

**Table 1: Theoretical Constants and Validation**
- File: `results/paper/summary_table.txt`
- Contains all physics constants (Δ, μ, C_filter, σ²_S, c_quantum)
- Shows empirical/theory ratios
- R² values for scaling laws

## Core Theory Being Validated

From the PDF (pages 1-4), the paper proves:

### Main Theorem (Proposition 4.8)

```
Gap(P, K) ≥ c_quantum * σ²_S * (1 - e^(-μηK))
```

Where:
- **c_quantum = Θ(Δ·C_filter/(d²M²T³))**: System-dependent constant
- **μ = Θ(Δ/(d²M²T))**: PL constant (landscape curvature)
- **σ²_S = Var_θ[∫_Ωcontrol S(ω;θ)χ(ω)dω]**: Control-relevant task variance

### Key Lemmas Validated

1. **Lemma 4.3 (PL Condition)**: μ scaling with spectral gap Δ
2. **Lemma 4.4 (Control-Relevant Variance)**: Bandwidth-localized task diversity
3. **Lemma 4.5 (Filter Separation)**: ||u*_θ - u*_θ'|| ≥ C_filter||S_θ - S_θ'||²

## Experimental Validation Strategy

### 1. Gap vs K Experiment (`experiment_gap_vs_k.py`)

**What it does:**
- Fixes task distribution P (σ²_S constant)
- Varies adaptation steps K ∈ {1, 2, 3, 5, 7, 10, 15, 20}
- Measures Gap(P, K) = E_θ[F_meta(K) - F_robust]
- Fits exponential model: Gap(K) = gap_max · (1 - e^(-μηK))

**Key validation:**
- R² ≥ 0.96 confirms exponential convergence theory
- Fitted μη matches theoretical μ·η within 2×

### 2. Gap vs Variance Experiment (`experiment_gap_vs_variance.py`)

**What it does:**
- Fixes adaptation steps K = 5
- Creates task distributions with varying σ²_S
- Measures Gap for each variance level
- Fits linear model: Gap = slope · σ²_S

**Key validation:**
- R² ≥ 0.94 confirms linear scaling with task diversity
- Slope matches c_quantum prediction within 2×

### 3. Constants Validation (`experiment_constants_validation.py`)

**What it does:**
- Estimates spectral gap Δ(θ) for sampled tasks
- Estimates PL constant μ(θ) via gradient analysis
- Estimates filter constant C_filter via control response
- Computes σ²_S from task distribution
- Compares all empirical constants to theoretical formulas

**Key validation:**
- Δ_min computed from Lindblad superoperator eigenvalues
- μ_empirical / μ_theory ∈ [0.5, 2.0] (within factor of 2)
- c_quantum_empirical / c_quantum_theory ∈ [0.5, 2.0]

## Physics Constants Reference

From the paper (pages 3-4, Appendix A):

### Spectral Gap Δ(θ)
**Physical meaning:** Dissipation timescale (1/Δ = relaxation time)
**Computation:** Eigenvalue gap of Lindblad superoperator L_θ
```python
from theory.physics_constants import compute_spectral_gap
Delta = compute_spectral_gap(env, task)
```

### PL Constant μ(θ)
**Physical meaning:** Landscape curvature (how fast gradient descent converges)
**Theory:** μ = Θ(Δ/(d²M²T)) from Lemma 4.5
**Computation:** Empirical from loss trajectory analysis
```python
from theory.physics_constants import estimate_pl_constant
mu = estimate_pl_constant(env, task, num_samples=10)
```

### Filter Constant C_filter
**Physical meaning:** How strongly different noise PSDs force different optimal controls
**Theory:** C_filter = σ²_min(M) / Σ_j ||W_j||²_2 from Proposition 4.6
**Computation:** Via implicit function theorem on control response
```python
from theory.physics_constants import estimate_filter_constant
C_filter = estimate_filter_constant(env, task_dist, n_samples=50)
```

### Control-Relevant Variance σ²_S
**Physical meaning:** Task diversity within control bandwidth Ω_control
**Theory:** σ²_S = Var_θ[∫_Ωcontrol S(ω;θ)χ(ω)dω] from Definition 4.7
**Computation:** Monte Carlo sampling of tasks
```python
from theory.physics_constants import compute_control_relevant_variance
sigma2_S = compute_control_relevant_variance(tasks, omega_control, chi)
```

## Expected Numerical Values (1-qubit system)

Based on paper Section 5:

| Constant | Typical Value | Units |
|----------|---------------|-------|
| Δ_min | 0.1-0.2 | rad/s |
| μ_min | 0.003-0.005 | 1/(control²) |
| C_filter | 0.03-0.05 | control/PSD² |
| σ²_S | 0.003-0.006 | PSD² |
| c_quantum | 0.02-0.04 | dimensionless |

## Debugging Tips

### If R² is low (<0.90):

1. **Check policy training:**
   ```python
   # Validation fidelity should be > 0.9
   python experiments/train_meta.py --validate
   ```

2. **Increase sample size:**
   ```bash
   python generate_all_results.py --n_tasks 200
   ```

3. **Verify environment:**
   ```python
   python scripts/test_minimal_working.py
   ```

### If constants are outside 2× bounds:

This is **acceptable** per paper Section 4.2 (Remark 4.9):
> "scaling predictions (Lemma 4.5, Proposition 4.6) involve dimensional analysis...
> Proposition 4.8 provides a scaling relationship rather than a tight lower bound"

The theory is **heuristic** for some components but **validated empirically** with R² > 0.94.

## File Structure

```
meta-quantum-control/
├── experiments/
│   ├── paper_results/               # ← All paper result scripts
│   │   ├── README.md
│   │   ├── generate_all_results.py  # ← Run this
│   │   ├── experiment_gap_vs_k.py
│   │   ├── experiment_gap_vs_variance.py
│   │   └── experiment_constants_validation.py
│   ├── train_meta.py                # Train MAML policy
│   ├── train_robust.py              # Train robust baseline
│   └── eval_gap.py                  # Evaluate gap (alternative)
├── src/
│   ├── theory/
│   │   ├── physics_constants.py     # ← Constants estimation
│   │   ├── optimality_gap.py        # ← Gap computation
│   │   └── quantum_environment.py   # ← System setup
│   └── ...
└── configs/
    └── experiment_config.yaml        # ← Full experiment config
```

## Quick Validation Checklist

After running `generate_all_results.py`, verify:

- [ ] `results/paper/gap_vs_k/figure.pdf` exists and shows exponential fit
- [ ] R² for Gap vs K ≥ 0.90 (target: 0.96)
- [ ] `results/paper/gap_vs_variance/figure.pdf` shows linear scaling
- [ ] R² for Gap vs σ²_S ≥ 0.90 (target: 0.94)
- [ ] μ_empirical / μ_theory ∈ [0.5, 2.0]
- [ ] c_quantum_empirical / c_quantum_theory ∈ [0.5, 2.0]
- [ ] `summary_table.txt` shows "✓✓✓ ALL VALIDATIONS PASSED ✓✓✓"

## Next Steps

1. **Run experiments:** `python generate_all_results.py`
2. **Review figures:** Open PDFs in `results/paper/`
3. **Check table:** `cat results/paper/summary_table.txt`
4. **Compare with paper:** Results should match Section 5 predictions

For detailed theoretical background, see the PDF (especially pages 3-5 for the main results and Appendix A for proofs).
