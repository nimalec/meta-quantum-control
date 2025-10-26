# Plotting Scripts Guide for Paper Results

This directory contains scripts for generating all publication-quality figures and analyses for the meta-learning quantum control paper.

## Overview

### Experiment Scripts (generate data + plots)
1. **`experiment_gap_vs_k.py`** - Optimality gap vs adaptation steps K
2. **`experiment_gap_vs_variance.py`** - Optimality gap vs task variance σ²_S
3. **`experiment_constants_validation.py`** - Physics constants estimation
4. **`experiment_system_scaling.py`** - System size scaling analysis

### Plotting Scripts (use existing checkpoints/data)
1. **`plot_fidelity_vs_k.py`** - Fidelity vs K from checkpoint
2. **`plot_validation_metrics.py`** - Validation metrics from checkpoint
3. **`metaqctrl/utils/plot_training.py`** - Training curves from checkpoint

### Master Scripts
- **`generate_all_results.py`** - Run all experiments and generate all figures
- **`generate_figure4_two_qubit.py`** - Generate two-qubit results

---

## Quick Start

### 1. Generate All Paper Results
```bash
# Run all experiments and generate all figures
python generate_all_results.py \
    --meta_path checkpoints/maml_best.pt \
    --robust_path checkpoints/robust_best.pt \
    --n_tasks 100 \
    --output_dir results/paper
```

### 2. Plot Fidelity vs K from Checkpoint
```bash
# Plot post-adaptation fidelity for different K values
python plot_fidelity_vs_k.py \
    --checkpoint checkpoints/maml_best.pt \
    --config ../../configs/experiment_config.yaml \
    --k_values 1 2 3 5 7 10 15 20 \
    --n_tasks 100 \
    --output_dir results/fidelity_vs_k
```

**Generates:**
- `fidelity_vs_k.pdf` - Mean fidelity vs K with error bars
- `adaptation_gain_vs_k.pdf` - Fidelity improvement vs K
- `fidelity_distributions.pdf` - Histograms for each K
- `combined_analysis.pdf` - 2x2 comprehensive analysis

### 3. Plot Validation Metrics from Checkpoint
```bash
# Analyze validation performance during training
python plot_validation_metrics.py \
    --checkpoint checkpoints/maml_best.pt \
    --output_dir results/validation_analysis
```

**Generates:**
- `validation_evolution.pdf` - Validation loss/fidelity over training
- `train_val_comparison.pdf` - Training vs validation curves
- `validation_summary.pdf` - Comprehensive 2x2 summary

**With checkpoint comparison:**
```bash
python plot_validation_metrics.py \
    --checkpoint checkpoints/maml_best.pt \
    --compare "K=3:checkpoints/maml_K3.pt" "K=5:checkpoints/maml_K5.pt" \
    --output_dir results/validation_comparison
```

### 4. Plot Training Curves from Checkpoint
```bash
# Generate training metrics plots
python -m metaqctrl.utils.plot_training \
    checkpoints/maml_best.pt \
    results/training_plots
```

**Generates:**
- `training_loss.png` - Meta-training loss evolution
- `training_fidelity.png` - Fidelity over iterations
- `validation_loss.png` - Validation performance
- `combined_metrics.png` - 2x2 comprehensive view

---

## Detailed Script Descriptions

### Experiment Scripts

#### `experiment_gap_vs_k.py`
**Purpose:** Validate theoretical prediction Gap(K) ∝ (1 - e^(-μηK))

**Usage:**
```bash
python experiment_gap_vs_k.py
```

**Requirements:**
- Trained meta-policy checkpoint (`checkpoints/maml_best.pt`)
- Trained robust policy checkpoint (`checkpoints/robust_best.pt`)

**Outputs:**
- `results/gap_vs_k/results.json` - Raw data
- `results/gap_vs_k/figure.pdf` - Publication figure
- R² fit quality (target: ≥ 0.96)

---

#### `experiment_gap_vs_variance.py`
**Purpose:** Validate theoretical prediction Gap(K) ∝ σ²_S

**Usage:**
```bash
python experiment_gap_vs_variance.py
```

**Outputs:**
- `results/gap_vs_variance/results.json` - Raw data
- `results/gap_vs_variance/figure.pdf` - Publication figure
- R² fit quality (target: ≥ 0.94)

---

#### `experiment_constants_validation.py`
**Purpose:** Estimate and validate all physics constants from theory

**Usage:**
```bash
python experiment_constants_validation.py
```

**Outputs:**
- `results/constants_validation/constants.json` - All constants
- `results/constants_validation/constants_visualization.pdf` - Validation plots

**Validates:**
- Spectral gap Δ
- PL constant μ
- Filter constant C_filter
- Task variance σ²_S
- Combined constant c_quantum

---

### Plotting Scripts

#### `plot_fidelity_vs_k.py`
**Purpose:** Visualize how post-adaptation fidelity varies with K

**Key Features:**
- Loads checkpoint and evaluates on test tasks
- Compares pre-adaptation vs post-adaptation
- Shows adaptation gain (improvement)
- Provides distribution analysis

**Arguments:**
```bash
--checkpoint PATH       # Path to trained policy checkpoint (required)
--config PATH          # Path to experiment config (default: ../../configs/experiment_config.yaml)
--k_values K1 K2 ...   # K values to test (default: 1 2 3 5 7 10 15 20)
--n_tasks N            # Number of test tasks (default: 100)
--output_dir DIR       # Output directory (default: results/fidelity_vs_k)
--inner_lr LR          # Inner loop learning rate (default: 0.01)
```

**Example with custom K values:**
```bash
python plot_fidelity_vs_k.py \
    --checkpoint checkpoints/maml_final.pt \
    --k_values 1 3 5 10 20 50 \
    --n_tasks 200 \
    --inner_lr 0.005 \
    --output_dir results/fidelity_analysis_custom
```

---

#### `plot_validation_metrics.py`
**Purpose:** Comprehensive validation analysis from checkpoint(s)

**Key Features:**
- Validation evolution during training
- Training vs validation comparison
- Multi-checkpoint comparison
- Statistical summaries

**Arguments:**
```bash
--checkpoint PATH       # Main checkpoint file (required)
--output_dir DIR       # Output directory (default: results/validation_plots)
--compare LABEL:PATH   # Additional checkpoints for comparison (optional)
```

**Example comparing multiple runs:**
```bash
python plot_validation_metrics.py \
    --checkpoint checkpoints/run1/maml_best.pt \
    --compare \
        "Run2:checkpoints/run2/maml_best.pt" \
        "Run3:checkpoints/run3/maml_best.pt" \
    --output_dir results/multi_run_comparison
```

---

### Training Curve Plotting

#### `metaqctrl/utils/plot_training.py`
**Purpose:** Standard training curve visualization

**Usage:**
```bash
python -m metaqctrl.utils.plot_training CHECKPOINT_PATH [OUTPUT_DIR]
```

**Example:**
```bash
python -m metaqctrl.utils.plot_training \
    checkpoints/maml_20241026_150000.pt \
    results/training_curves
```

**Features:**
- Training loss evolution
- Fidelity tracking
- Validation performance
- Smoothed curves with configurable window
- Multi-run comparison

---

## Checkpoint Requirements

### What Checkpoints Should Contain

For full plotting functionality, checkpoints should include:

**Minimum (for basic plots):**
```python
checkpoint = {
    'model_state_dict': policy.state_dict(),  # or 'policy_state_dict'
    'meta_train_losses': [...]  # List of training losses per iteration
}
```

**Recommended (for validation plots):**
```python
checkpoint = {
    'model_state_dict': policy.state_dict(),
    'meta_train_losses': [...],
    'meta_val_losses': [...],      # Validation losses
    'val_interval': 50,             # How often validation runs
    'inner_lr': 0.01,              # Inner loop learning rate
    'inner_steps': 5,              # K value used during training
    'epoch': 100,                  # Current epoch
}
```

**Full (for comprehensive analysis):**
```python
checkpoint = {
    'model_state_dict': policy.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'meta_train_losses': [...],
    'meta_val_losses': [...],
    'val_loss_pre_adapt': float,   # Final pre-adapt validation loss
    'val_loss_post_adapt': float,  # Final post-adapt validation loss
    'adaptation_gain': float,       # Post - Pre
    'val_interval': 50,
    'inner_lr': 0.01,
    'inner_steps': 5,
    'epoch': 100,
    'config': {...}                # Full experiment config
}
```

---

## Workflow Examples

### Workflow 1: Training → Plotting
```bash
# 1. Train meta-learning policy
python experiments/train_meta.py --output checkpoints/maml

# 2. Generate training curves
python -m metaqctrl.utils.plot_training \
    checkpoints/maml/maml_best.pt \
    results/training

# 3. Analyze fidelity vs K
python experiments/paper_results/plot_fidelity_vs_k.py \
    --checkpoint checkpoints/maml/maml_best.pt \
    --output_dir results/fidelity_analysis

# 4. Analyze validation metrics
python experiments/paper_results/plot_validation_metrics.py \
    --checkpoint checkpoints/maml/maml_best.pt \
    --output_dir results/validation_analysis
```

### Workflow 2: Paper Figure Generation
```bash
# 1. Train both policies if not done
python experiments/train_meta.py --output checkpoints/
python experiments/train_robust.py --output checkpoints/

# 2. Generate all paper results
python experiments/paper_results/generate_all_results.py \
    --meta_path checkpoints/maml_best.pt \
    --robust_path checkpoints/robust_best.pt \
    --n_tasks 100 \
    --output_dir results/paper_final
```

This generates:
- Figure 1: Gap vs K (with R² fit)
- Figure 2: Gap vs Variance (with R² fit)
- Table 1: Constants validation
- All supporting data as JSON

### Workflow 3: Hyperparameter Comparison
```bash
# Compare different K values during training
python experiments/paper_results/plot_validation_metrics.py \
    --checkpoint checkpoints/K3/maml_best.pt \
    --compare \
        "K=5:checkpoints/K5/maml_best.pt" \
        "K=7:checkpoints/K7/maml_best.pt" \
        "K=10:checkpoints/K10/maml_best.pt" \
    --output_dir results/K_comparison

# Compare different learning rates
python experiments/paper_results/plot_fidelity_vs_k.py \
    --checkpoint checkpoints/lr_0.01/maml_best.pt \
    --output_dir results/lr_0.01_analysis

python experiments/paper_results/plot_fidelity_vs_k.py \
    --checkpoint checkpoints/lr_0.005/maml_best.pt \
    --output_dir results/lr_0.005_analysis
```

---

## Output File Structure

After running all scripts, you'll have:

```
results/
├── paper/                          # From generate_all_results.py
│   ├── gap_vs_k/
│   │   ├── results.json
│   │   └── figure.pdf
│   ├── gap_vs_variance/
│   │   ├── results.json
│   │   └── figure.pdf
│   ├── constants_validation/
│   │   ├── constants.json
│   │   └── constants_visualization.pdf
│   └── summary_table.txt
│
├── fidelity_vs_k/                 # From plot_fidelity_vs_k.py
│   ├── results.json
│   ├── fidelity_vs_k.pdf
│   ├── adaptation_gain_vs_k.pdf
│   ├── fidelity_distributions.pdf
│   └── combined_analysis.pdf
│
├── validation_plots/              # From plot_validation_metrics.py
│   ├── validation_evolution.pdf
│   ├── train_val_comparison.pdf
│   └── validation_summary.pdf
│
└── training_curves/               # From plot_training.py
    ├── training_loss.png
    ├── training_fidelity.png
    ├── validation_loss.png
    └── combined_metrics.png
```

---

## Tips and Best Practices

### For Publication-Quality Figures
1. **Use high DPI:** All scripts default to 300 DPI for publication
2. **PDF format:** Use PDF for vector graphics (scalable)
3. **Font consistency:** Scripts use consistent fonts and sizes
4. **Color schemes:** Colorblind-friendly palettes via seaborn

### For Reproducibility
1. **Save configs:** Always save experiment configs with results
2. **Random seeds:** Set seeds in training scripts
3. **Checkpoint everything:** Save full checkpoints during training
4. **Log parameters:** Include all hyperparameters in filenames/logs

### For Performance
1. **GPU usage:** Scripts automatically detect and use GPU
2. **Parallel evaluation:** Use multiprocessing for large n_tasks
3. **Caching:** Cache expensive computations (e.g., spectral gaps)
4. **Batch sizes:** Adjust based on available memory

---

## Troubleshooting

### "No validation data found in checkpoint"
**Solution:** Ensure your training script saves `meta_val_losses` to checkpoints.

### "Checkpoint loading failed"
**Solution:** Check checkpoint format. Try:
```python
checkpoint = torch.load(path, map_location='cpu')
print(checkpoint.keys())  # See what's available
```

### "ImportError: No module named metaqctrl"
**Solution:** Run from project root or add to PYTHONPATH:
```bash
export PYTHONPATH=/path/to/meta-quantum-control:$PYTHONPATH
```

### "Out of memory during evaluation"
**Solution:** Reduce `--n_tasks` or use smaller batches:
```bash
python plot_fidelity_vs_k.py --n_tasks 50  # Instead of 100
```

---

## Contact

For questions or issues with plotting scripts, please open an issue in the repository.
