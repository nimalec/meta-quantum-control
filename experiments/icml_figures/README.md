# ICML Figure Generation Scripts

This directory contains scripts to generate all figures for the ICML submission "Meta-Reinforcement Learning for Quantum Control: A Framework for Quantifying Adaptation Gains under Noise Shifts".

## Overview

The scripts generate:
- **Figure 2**: Fast Adaptation Scaling (Theory Meets Experiment)
- **Figure 3**: Training Stability and Generalization
- **Figure 4**: Robustness to Distribution Shift
- **Appendix S1**: Sensitivity/Ablation Studies
- **Appendix S2**: Flat Gap vs PSD Variance
- **Appendix S3**: Classical Control Toy Example

## Quick Start

### 1. Prerequisites

Ensure you have trained models:
```bash
# Train meta-learning policy
python experiments/train_meta.py --config configs/experiment_config.yaml

# Train robust baseline
python experiments/train_robust.py --config configs/experiment_config.yaml
```

### 2. Generate All Figures

```bash
cd experiments/icml_figures
python run_all_figures.py
```

This will generate all figures and save them to `results/icml_figures/`.

### 3. Generate Individual Figures

```bash
# Figure 2 only
python run_all_figures.py --figures 2

# Figure 3 only
python run_all_figures.py --figures 3

# Multiple figures
python run_all_figures.py --figures 2 4 appendix
```

### 4. Use Cached Data (Skip Experiments)

If you've already run experiments and want to regenerate plots only:
```bash
python run_all_figures.py --skip-experiments
```

## Detailed Usage

### Figure 2: Fast Adaptation Scaling

```bash
python figure2_adaptation_scaling.py
```

**Panels:**
- (a) Schematic of adaptation gain vs K
- (b) Empirical gap vs K (high variance regime)
- (c) Empirical gap vs K (low variance regime)
- (d) Gap vs detuning spread
- (e) Gap vs decoherence/actuator spread

**Runtime:** ~30-60 minutes (depending on `n_test_tasks`)

**Key Parameters:**
- `n_test_tasks`: Number of test tasks per condition (default: 100)
- `k_values`: Adaptation steps to evaluate (default: [1, 2, 3, 5, 7, 10, 15, 20])

### Figure 3: Training Stability

```bash
python figure3_training_stability.py
```

**Panels:**
- (a) Meta-loss vs outer iterations
- (b) Support vs Query loss curves
- (c) Validation on held-out tasks
- (d) Stability diagnostics (gradient norms, NaN incidents)

**Requirements:**
This figure requires training logs. If not available, synthetic data will be generated for demonstration.

**To generate real logs:**
Add logging to `experiments/train_meta.py`:
```bash
python figure3_training_stability.py --show-snippet
```
This will print code to add to your training script.

### Figure 4: Robustness and OOD

```bash
python figure4_robustness_ood.py
```

**Panels:**
- (a) Performance under OOD detuning mismatch
- (b) Per-task heatmap (hardest regime)
- (c) Baseline comparison bars
- (d) Cross-family transfer (optional)

**Runtime:** ~20-40 minutes

**Key Parameters:**
- `ood_detuning_levels`: OOD test levels (default: [1.0, 1.5, 2.0, 2.5, 3.0, 4.0])
- `n_hard_tasks`: Tasks for heatmap (default: 20)
- `include_grape`: Whether to compute GRAPE baseline (default: False, slow if True)

### Appendix Figures

```bash
python appendix_figures.py
```

**Generates:**
- Appendix S1: Ablation studies (inner LR, first/second-order, K saturation)
- Appendix S2: Flat gap vs raw PSD variance
- Appendix S3: Classical control toy example

**Runtime:** ~15-30 minutes

## Advanced Options

### Specify Custom Checkpoints

```bash
python run_all_figures.py \
  --meta-policy /path/to/maml_checkpoint.pt \
  --robust-policy /path/to/robust_checkpoint.pt
```

### Custom Output Directory

```bash
python run_all_figures.py --output-dir /path/to/output
```

### Provide Training Log for Figure 3

```bash
python run_all_figures.py --training-log checkpoints/training_log.json
```

## Output Files

After running, you'll find in `results/icml_figures/`:

```
results/icml_figures/
├── figure2_complete.pdf
├── figure2_complete.png
├── figure2_data.json
├── figure3_training_stability.pdf
├── figure3_training_stability.png
├── figure4_robustness.pdf
├── figure4_robustness.png
├── figure4_data.json
├── appendix_s1_ablation.pdf
├── appendix_s1_ablation.png
├── appendix_s2_flat_variance.pdf
├── appendix_s2_flat_variance.png
├── appendix_s3_classical.pdf
├── appendix_s3_classical.png
└── appendix_data.json
```

## Troubleshooting

### "ERROR: Trained models not found"

Train models first:
```bash
python experiments/train_meta.py
python experiments/train_robust.py
```

Or specify paths manually:
```bash
python run_all_figures.py \
  --meta-policy checkpoints/your_meta_policy.pt \
  --robust-policy checkpoints/your_robust_policy.pt
```

### Figure 3 uses synthetic data

This is expected if you haven't added logging to your training script. To use real data:

1. Run `python figure3_training_stability.py --show-snippet`
2. Add the printed code to `experiments/train_meta.py`
3. Re-run training
4. Regenerate Figure 3

### Experiments take too long

Reduce `n_test_tasks` in the scripts:
- Figure 2: Change `n_test_tasks=100` to `n_test_tasks=50`
- Figure 4: Change `n_test_tasks=100` to `n_test_tasks=50`

Or use cached data:
```bash
python run_all_figures.py --skip-experiments
```

### Out of memory

If running on a machine with limited RAM:
1. Use CPU instead of GPU (scripts default to CPU)
2. Reduce `n_test_tasks`
3. Process figures one at a time:
   ```bash
   python run_all_figures.py --figures 2
   python run_all_figures.py --figures 3
   python run_all_figures.py --figures 4
   python run_all_figures.py --figures appendix
   ```

## Customization

### Modify Experimental Parameters

Edit the `config` dictionary at the bottom of each script:

```python
config = {
    'num_qubits': 1,
    'n_controls': 2,
    'n_segments': 20,
    'horizon': 1.0,
    'target_gate': 'hadamard',
    'hidden_dim': 128,
    'n_hidden_layers': 2,
    'inner_lr': 0.01,
    'noise_frequencies': [1.0, 5.0, 10.0]
}
```

### Modify Plot Aesthetics

All plotting functions use seaborn and matplotlib. Customize in the `plot_*` functions:
- Colors: Change `color='steelblue'` to your preference
- Styles: Modify `sns.set_style("whitegrid")`
- Sizes: Adjust `figsize=(18, 10)`

## Performance Tips

1. **Parallel Execution**: Run figures in parallel on different machines/GPUs
2. **Caching**: Use `--skip-experiments` after first run
3. **Reduced Tasks**: Lower `n_test_tasks` for faster iteration
4. **GRAPE Baseline**: Set `include_grape=False` (GRAPE is slow)

## Expected Results

### Figure 2
- Panel (b): R² ≈ 0.96 for exponential fit (high variance)
- Panel (c): R² ≈ 0.88 for exponential fit (low variance)
- Panels (d,e): Monotonically increasing gap with diversity

### Figure 3
- Support loss > Query loss (adaptation helps)
- Validation fidelity increasing over training
- Gradient norms stabilize, few NaN incidents

### Figure 4
- Meta+Adapt outperforms Robust baseline under OOD
- Heatmap shows Meta-Adapt recovers performance
- Bar chart: Meta+Adapt ≈ GRAPE > Robust

### Appendix
- S1: Results stable across inner LRs, first/second-order similar
- S2: Flat relationship (validates theory focuses on controller-relevant diversity)
- S3: Classical system shows same adaptation scaling

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
- Nima Leclerc: nleclerc@mitre.org
- GitHub Issues: [link to repo]
