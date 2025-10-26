# MAML Training Visualization Guide

This guide explains how to generate publication-quality plots from your MAML training checkpoints for meta-RL papers.

## Quick Start

### 1. After Training

Once you've run a training session with `experiments/train_meta.py`, you'll have checkpoint files in the `checkpoints/` directory:

```bash
checkpoints/
├── maml_20241025_120000.pt        # Final checkpoint
└── maml_20241025_120000_best.pt   # Best validation checkpoint
```

### 2. Generate All Standard Plots

The easiest way to generate all plots is to use the command-line interface:

```bash
python metaqctrl/utils/plot_training.py checkpoints/maml_20241025_120000.pt results/figures/
```

This will create:
- `training_loss.png` - Meta-training loss over iterations
- `training_fidelity.png` - Fidelity (1 - loss) over iterations
- `validation_loss.png` - Post-adaptation validation loss
- `combined_metrics.png` - 2x2 grid with all metrics and summary statistics

### 3. Programmatic Use

```python
from metaqctrl.utils.plot_training import (
    plot_training_loss,
    plot_fidelity,
    plot_combined_metrics
)

# Generate individual plots
plot_training_loss('checkpoints/maml_best.pt', 'results/figures/loss.png')
plot_fidelity('checkpoints/maml_best.pt', 'results/figures/fidelity.png')

# Generate combined figure (recommended for papers)
plot_combined_metrics('checkpoints/maml_best.pt', 'results/figures/combined.png')
```

## Available Plots

### 1. Training Loss

```python
plot_training_loss(
    checkpoint_path='checkpoints/maml.pt',
    save_path='figures/training_loss.png',
    smooth_window=10,        # Moving average window
    show_smoothed=True       # Show both raw and smoothed
)
```

**What it shows:** Meta-training loss decreasing over iterations. This is the loss computed on the query set after K inner-loop adaptation steps.

**Interpretation:**
- Downward trend = meta-learning is working
- Smoothed curve removes noise for clearer visualization
- Raw curve shows actual training dynamics

---

### 2. Fidelity

```python
plot_fidelity(
    checkpoint_path='checkpoints/maml.pt',
    save_path='figures/fidelity.png',
    smooth_window=10
)
```

**What it shows:** Gate fidelity (1 - loss) over iterations. Fidelity = 1.0 means perfect gate implementation.

**Interpretation:**
- Higher is better (closer to 1.0)
- For quantum control: fidelity > 0.99 is typically considered high-fidelity
- Shows how well the meta-learned policy performs

---

### 3. Validation Loss

```python
plot_validation_losses(
    checkpoint_path='checkpoints/maml.pt',
    save_path='figures/validation_loss.png'
)
```

**What it shows:** Validation loss computed on held-out tasks at regular intervals.

**Interpretation:**
- Lower is better
- Should track training loss (no overfitting)
- Used for early stopping / model selection

---

### 4. Combined Metrics (Recommended for Papers)

```python
plot_combined_metrics(
    checkpoint_path='checkpoints/maml.pt',
    save_path='figures/combined.png',
    smooth_window=20,
    figsize=(14, 10),
    dpi=600  # High resolution for publication
)
```

**What it shows:** 2×2 grid with:
- (a) Training loss with smoothing
- (b) Training fidelity with smoothing
- (c) Validation loss over iterations
- (d) Summary statistics table

**Why use this:** Comprehensive single-figure summary of training, perfect for papers or presentations.

---

### 5. Comparison Across Runs

```python
plot_loss_comparison(
    checkpoint_paths={
        'MAML (K=1)': 'checkpoints/maml_k1.pt',
        'MAML (K=5)': 'checkpoints/maml_k5.pt',
        'Baseline': 'checkpoints/baseline.pt'
    },
    save_path='figures/comparison.png',
    metric='fidelity'  # or 'loss'
)
```

**What it shows:** Overlay multiple training runs to compare hyperparameters or algorithms.

**Use cases:**
- Hyperparameter sensitivity (e.g., inner learning rate)
- Ablation studies (e.g., first-order vs second-order MAML)
- Baseline comparisons (e.g., MAML vs robust control)

---

## Data Available in Checkpoints

Checkpoints saved by MAML contain:

| Key | Type | Description |
|-----|------|-------------|
| `meta_train_losses` | List[float] | Training loss at each iteration |
| `meta_val_losses` | List[float] | Validation loss at each validation interval |
| `epoch` | int | Total number of iterations completed |
| `inner_lr` | float | Inner loop learning rate (α) |
| `inner_steps` | int | Number of inner gradient steps (K) |
| `policy_state_dict` | dict | Trained policy weights |
| `meta_optimizer_state_dict` | dict | Meta-optimizer state |

Additional fields in validation checkpoints:
- `val_loss_pre_adapt` - Loss before adaptation
- `val_loss_post_adapt` - Loss after K adaptation steps
- `adaptation_gain` - Improvement from adaptation
- `std_post_adapt` - Standard deviation of post-adaptation losses

### Inspecting Checkpoint Contents

```python
from metaqctrl.utils.plot_training import print_checkpoint_summary

print_checkpoint_summary('checkpoints/maml_best.pt')
```

Output:
```
============================================================
Checkpoint: checkpoints/maml_best.pt
============================================================

Available Keys:
  meta_train_losses: list of length 1000
  meta_val_losses: list of length 20
  epoch: 1000
  policy_state_dict: dict with 6 items
  ...

Training Metrics:
  Total iterations: 1000
  Final training loss: 0.0234
  Min training loss: 0.0123
  Max training loss: 0.5678

Validation Metrics:
  Validation points: 20
  Final validation loss: 0.0256
  Best validation loss: 0.0198

MAML Configuration:
  Inner LR: 0.01
  Inner Steps: 5
  Epoch: 1000
============================================================
```

---

## Custom Plotting

For advanced users, you can access checkpoint data directly:

```python
import torch
import matplotlib.pyplot as plt
import numpy as np

# Load checkpoint
checkpoint = torch.load('checkpoints/maml.pt', map_location='cpu')
train_losses = checkpoint['meta_train_losses']

# Custom analysis
iterations = np.arange(len(train_losses))

# Example: Plot loss on log scale
plt.semilogy(iterations, train_losses)
plt.xlabel('Iteration')
plt.ylabel('Loss (log scale)')
plt.title('Training Loss (Log Scale)')
plt.grid(True, alpha=0.3)
plt.savefig('figures/loss_log_scale.png', dpi=300)
```

### Example: Compute Convergence Point

```python
import numpy as np

# Find when loss reaches within 5% of final value
final_loss = train_losses[-1]
threshold = final_loss * 1.05

converged_idx = next(i for i, loss in enumerate(train_losses)
                     if all(l <= threshold for l in train_losses[i:]))

print(f"Converged at iteration {converged_idx} (loss={train_losses[converged_idx]:.4f})")
```

---

## Publication-Quality Settings

For journal submissions, use these settings:

```python
plot_combined_metrics(
    checkpoint_path='checkpoints/maml_best.pt',
    save_path='figures/figure3_training.png',
    smooth_window=20,      # Smoother curves
    figsize=(12, 8),       # 2-column journal width
    dpi=600                # High resolution
)
```

**Recommended formats:**
- **PNG**: 600 DPI, RGB color
- **PDF**: Vector format (convert with `convert file.png file.pdf`)
- **Font**: Times New Roman (serif) - already configured
- **Size**: 7-12 inches wide (2-column format)

**Converting to PDF (lossless):**
```bash
# Using ImageMagick
convert -density 600 figure.png figure.pdf

# Or using Python
from PIL import Image
img = Image.open('figure.png')
img.save('figure.pdf', 'PDF', resolution=600.0)
```

---

## Tips for Meta-RL Papers

### 1. Learning Curves
Always show smoothed curves with raw data faintly in background (default behavior).

### 2. Error Bars / Confidence Intervals
If you run multiple seeds, plot mean ± std:

```python
import matplotlib.pyplot as plt
import numpy as np

# Assuming you have multiple runs
runs = [
    torch.load(f'checkpoints/seed{i}.pt')['meta_train_losses']
    for i in range(5)
]

# Convert to array (assuming same length)
runs_array = np.array(runs)
mean = runs_array.mean(axis=0)
std = runs_array.std(axis=0)

iterations = np.arange(len(mean))

plt.plot(iterations, mean, linewidth=2, label='MAML')
plt.fill_between(iterations, mean - std, mean + std, alpha=0.3)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)
```

### 3. Adaptation Gain
Show pre-adaptation vs post-adaptation performance:

```python
# This data is saved in validation checkpoints
checkpoint = torch.load('checkpoints/maml_best.pt')

pre_adapt = checkpoint.get('val_loss_pre_adapt', None)
post_adapt = checkpoint.get('val_loss_post_adapt', None)
gain = checkpoint.get('adaptation_gain', None)

if gain:
    print(f"Adaptation improves loss by {gain:.4f}")
    print(f"Pre-adapt: {pre_adapt:.4f} → Post-adapt: {post_adapt:.4f}")
```

### 4. Sample Efficiency
Plot data efficiency by showing performance vs number of support samples:

```python
# Compare checkpoints with different n_support
checkpoints = {
    '2 support': 'checkpoints/support2.pt',
    '5 support': 'checkpoints/support5.pt',
    '10 support': 'checkpoints/support10.pt',
}

plot_loss_comparison(checkpoints, save_path='figures/sample_efficiency.png')
```

---

## Common Issues

### No validation losses in checkpoint

**Problem:** `plot_validation_losses()` shows "No validation losses found"

**Cause:** Training was run without validation (val_interval not set or too large)

**Solution:**
- Use `plot_training_loss()` and `plot_fidelity()` instead
- Re-run training with validation enabled in config:
  ```yaml
  val_interval: 50  # Validate every 50 iterations
  ```

### Checkpoint file not found

**Problem:** `FileNotFoundError: checkpoints/maml.pt`

**Cause:** Haven't run training yet or checkpoint saved elsewhere

**Solution:**
- Run training: `python experiments/train_meta.py configs/experiment_config.yaml`
- Check `save_dir` in your config file
- Use absolute path: `plot_training_loss('/full/path/to/checkpoint.pt')`

### Plots look noisy

**Problem:** Training curves are very jagged

**Solution:**
- Increase smoothing window: `smooth_window=50`
- Use only smoothed curve: `show_smoothed=True` (default)
- Average over multiple seeds (see Tips section)

---

## Examples

See `examples/plot_training_example.py` for complete working examples:

```bash
python examples/plot_training_example.py
```

This includes:
1. Single checkpoint analysis
2. Comparing multiple runs
3. Custom analysis
4. Publication-ready figures

---

## Integration with Training

To automatically generate plots after training, modify `experiments/train_meta.py`:

```python
from metaqctrl.utils.plot_training import plot_combined_metrics

# At the end of training
trainer.train(n_iterations=1000, save_path='checkpoints/maml.pt')

# Auto-generate plots
plot_combined_metrics(
    'checkpoints/maml_best.pt',
    save_path='results/figures/training_summary.png'
)
```

---

## Questions?

For issues or feature requests, please open an issue on GitHub or refer to the main documentation.

**Related files:**
- Implementation: `metaqctrl/utils/plot_training.py`
- Examples: `examples/plot_training_example.py`
- MAML trainer: `metaqctrl/meta_rl/maml.py`
