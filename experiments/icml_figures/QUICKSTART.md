# Quick Start Guide

## Generate All Figures in 3 Steps

### Step 1: Train Models (if not already done)

```bash
cd /Users/nimalec/Documents/metarl_project/meta-quantum-control

# Train meta-learning policy (~10 hours on CPU, ~2 hours on GPU)
python experiments/train_meta.py

# Train robust baseline (~5 hours on CPU, ~1 hour on GPU)
python experiments/train_robust.py
```

### Step 2: Generate Figures

```bash
cd experiments/icml_figures

# Generate all figures (~1-2 hours total)
python run_all_figures.py
```

### Step 3: View Results

```bash
# Results are in:
open results/icml_figures/figure2_complete.pdf
open results/icml_figures/figure3_training_stability.pdf
open results/icml_figures/figure4_robustness.pdf
open results/icml_figures/appendix_s1_ablation.pdf
```

## Fast Mode (Skip Long Experiments)

If you want quick results with fewer test tasks:

```bash
# Edit the scripts to reduce n_test_tasks
# In figure2_adaptation_scaling.py, change:
#   n_test_tasks=100  →  n_test_tasks=20

# Then run
python run_all_figures.py
```

## Individual Figures

```bash
# Just Figure 2 (main theory validation)
python figure2_adaptation_scaling.py

# Just Figure 3 (training diagnostics, fast)
python figure3_training_stability.py

# Just Figure 4 (OOD robustness)
python figure4_robustness_ood.py

# Just Appendix
python appendix_figures.py
```

## Expected Output

```
results/icml_figures/
├── figure2_complete.pdf         # Main results: theory + experiments
├── figure3_training_stability.pdf  # Training convergence
├── figure4_robustness.pdf       # OOD performance
├── appendix_s1_ablation.pdf     # Sensitivity studies
├── appendix_s2_flat_variance.pdf  # Negative result
└── appendix_s3_classical.pdf    # Generality demo
```

## Troubleshooting

**"ERROR: Trained models not found"**
→ Run Step 1 first, or specify paths:
```bash
python run_all_figures.py \
  --meta-policy /path/to/maml_checkpoint.pt \
  --robust-policy /path/to/robust_checkpoint.pt
```

**"Takes too long"**
→ Reduce `n_test_tasks` in scripts from 100 to 20-50

**"Out of memory"**
→ Scripts use CPU by default. If still issues, run one at a time:
```bash
python run_all_figures.py --figures 2
python run_all_figures.py --figures 3
python run_all_figures.py --figures 4
```

## Key Results to Expect

✓ **Figure 2b**: R² ≈ 0.96 (validates exponential convergence theory)
✓ **Figure 2d**: Monotonic increase (adaptation helps more with diversity)
✓ **Figure 3b**: Query < Support (adaptation works during training)
✓ **Figure 4a**: Meta adapts better than Robust under OOD

## Customization

Edit `config` dict at bottom of each script to change:
- Target gate (hadamard → pauli_x)
- Number of qubits
- Control parameters
- Task distribution ranges

## Complete Documentation

See [README.md](README.md) for full details.
