# Paper Validation Experiments

This directory contains scripts to validate the bug fixes and theoretical predictions for the meta-quantum-control paper.

## Overview

Three major fixes were implemented:
1. **PSD to Lindblad Integration**: Fixed noise modeling to properly integrate over control bandwidth
2. **MAML Bug Fixes**: Fixed 4 critical bugs in the meta-learning implementation
3. **Gap Scaling Measurement**: Implemented proper measurement of adaptation gap vs task variance

## Experiments

### 1. PSD to Lindblad Integration Validation (`validate_psd_integration.py`)

**Purpose**: Demonstrate that proper frequency integration gives physically accurate noise modeling.

**What it tests**:
- Comparison of old (single-point) vs new (integrated) methods
- Effect on decay rates across different noise models
- Dependence on control bandwidth
- Impact on quantum gate fidelity

**Outputs**:
- `psd_integration_comparison.png` - Bar chart comparing decay rates
- `bandwidth_dependence.png` - Shows how rates scale with bandwidth
- LaTeX table with numerical results

**Runtime**: ~2-3 minutes

---

### 2. MAML Bug Fixes Validation (`validate_maml_fixes.py`)

**Purpose**: Verify that all 4 MAML bugs are fixed and training is stable.

**Bugs fixed**:
1. Missing `.clone()` in second-order gradient accumulation
2. NaN/Inf handling breaking gradient flow
3. Double gradient averaging
4. Validation loop not exception-safe

**What it tests**:
- Gradient flow (no NaN/Inf)
- Gradient magnitude consistency across batch sizes
- Training stability over 20 steps
- Adaptation performance on validation tasks

**Outputs**:
- `training_curves.png` - Meta-loss and gradient norm over training
- `test_summary.txt` - Pass/fail for each test
- Terminal output with detailed diagnostics

**Runtime**: ~3-5 minutes

---

### 3. Adaptation Gap vs Variance Scaling (`validate_gap_scaling.py`)

**Purpose**: Test the theoretical prediction that Gap ∝ σ²

**What it tests**:
- Gap vs parameter variance σ²_θ (linear fit, R²)
- Gap vs control-relevant variance σ²_S (linear fit, R²)
- Gap vs adaptation steps K (exponential decay)
- Comparison to theoretical bounds

**Outputs**:
- `gap_vs_param_variance.png` - Gap scaling with σ²_θ
- `gap_vs_control_variance.png` - Gap scaling with σ²_S (side-by-side comparison)
- `gap_vs_K.png` - Gap vs number of adaptation steps
- `combined_gap_scaling.png` - Three-panel figure for paper
- LaTeX table with R² values

**Runtime**: ~5-10 minutes

---

## Quick Start

### Run All Experiments

```bash
cd experiments/eval_scripts
python run_all_paper_experiments.py
```

This will:
1. Run all three validation scripts sequentially
2. Generate all figures and tables
3. Save results to `experiments/paper_results/`
4. Print summary of pass/fail for each experiment

**Total runtime**: ~10-20 minutes

---

### Run Individual Experiments

```bash
# PSD integration
python validate_psd_integration.py

# MAML fixes
python validate_maml_fixes.py

# Gap scaling
python validate_gap_scaling.py
```

---

## Results Directory Structure

```
experiments/paper_results/
├── psd_integration/
│   ├── psd_integration_comparison.png
│   └── bandwidth_dependence.png
├── maml_validation/
│   ├── training_curves.png
│   └── test_summary.txt
└── gap_scaling/
    ├── gap_vs_param_variance.png
    ├── gap_vs_control_variance.png
    ├── gap_vs_K.png
    └── combined_gap_scaling.png
```

---

## Expected Results

### PSD Integration

**Old method**: Single-point evaluation
- Arbitrary frequency sampling
- Wrong units
- Misses spectral content

**New method**: Proper integration
- Integrates S(ω) over control bandwidth
- Correct units (rate = ∫ S(ω) dω)
- Physically accurate

**Expected**: New rates are 2-3× smaller and more accurate

---

### MAML Fixes

**Expected outcomes**:
- ✓ Gradient flow test: All gradients finite (no NaN/Inf)
- ✓ Gradient magnitude: Consistent across batch sizes (no double averaging)
- ✓ Training stability: 20 steps without divergence
- ✓ Adaptation: Post-adaptation loss < pre-adaptation loss

**If any test fails**: Check that you're using the fixed MAML implementation

---

### Gap Scaling

**Theoretical prediction**: Gap = c · σ² · (1 - e^(-μηK))

**Expected outcomes**:
- Linear relationship between Gap and σ² (R² > 0.8)
- Either σ²_θ or σ²_S gives good fit (determine which)
- Gap decreases exponentially with K (more adaptation steps → smaller gap)

**If R² < 0.7**:
- May need more samples (increase `n_samples`)
- Check that task variance is actually varying
- Verify meta-policy is trained (not random)

---

## Figures for Paper

### Main Results Figure (3-panel)

`combined_gap_scaling.png` contains:
- **(A)** Gap vs σ²_θ with linear fit and R²
- **(B)** Gap vs σ²_S with linear fit and R²
- **(C)** Gap vs K with exponential fit

Use this as the main results figure.

---

### Supplementary Figures

1. **PSD comparison**: Shows improved noise modeling
2. **Bandwidth dependence**: Shows integration captures frequency content
3. **Training curves**: Demonstrates MAML stability

---

## LaTeX Tables

Both validation scripts print LaTeX tables to the terminal. Copy these into your paper:

**Table 1**: PSD to Lindblad comparison
```latex
\begin{table}[h]
\caption{Comparison of PSD to Lindblad conversion methods...}
...
\end{table}
```

**Table 2**: Gap scaling results
```latex
\begin{table}[h]
\caption{Linear scaling of adaptation gap with task variance...}
...
\end{table}
```

---

## Troubleshooting

### Import errors
```bash
# Add parent directory to Python path
export PYTHONPATH=/path/to/meta-quantum-control:$PYTHONPATH
```

### GPU memory issues
- Scripts default to CPU
- For GPU: modify `device = torch.device('cuda')`
- Reduce batch sizes if OOM

### Slow runtime
- Reduce `n_samples` (default: 30-50)
- Reduce `n_integration_points` (default: 1000)
- Use fewer variance points (default: 8-10)

### Poor R² values
- Increase `n_samples` to reduce noise
- Check meta-policy is trained (not random initialization)
- Verify task distribution has meaningful variance

---

## Citation

If you use these experiments in your paper, please cite:

```bibtex
@article{your-paper-2025,
  title={Meta-Learning for Robust Quantum Control},
  author={Your Name},
  journal={arXiv},
  year={2025}
}
```

---

## Contact

For questions or issues:
- Check the main repository README
- File an issue on GitHub
- Contact: [your email]

---

## Changelog

**2025-01-XX**: Initial release
- PSD integration validation
- MAML bug fix validation
- Gap scaling validation
- Master experiment runner
