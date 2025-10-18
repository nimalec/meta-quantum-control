# Makefile for generating publication-quality figures
# Usage: make <target>
# Example: make results/figures/noise.png

.PHONY: help clean all-figures

clean:
	rm -f results/figures/*.png

.venv:
	uv venv
	uv pip install -r requirements.txt

# ============================================================================
# Noise PSD Figures
# ============================================================================

# Default noise plot with 5 tasks
results/figures/one_over_f.png: .venv src/utils/plots.py src/quantum/noise_models.py
	uv run src/utils/plots.py noise $@ --model one_over_f --n-tasks 5 --seed 42

# Noise plot with different models
results/figures/noise_lorentzian.png: .venv src/utils/plots.py src/quantum/noise_models.py
	uv run src/utils/plots.py noise $@ --model lorentzian --n-tasks 5 --seed 42

results/figures/noise_double_exp.png: .venv src/utils/plots.py src/quantum/noise_models.py
	uv run src/utils/plots.py noise $@ --model double_exp --n-tasks 5 --seed 42

# Noise plot with distance calculations
results/figures/noise_with_distances.png: .venv src/utils/plots.py src/quantum/noise_models.py
	uv run src/utils/plots.py noise $@ --n-tasks 5 --seed 42 --distances

# ============================================================================
# PSD Comparison Figures
# ============================================================================

# Default PSD comparison
results/figures/psd_comparison.png: .venv src/utils/plots.py src/quantum/noise_models.py
	uv run src/utils/plots.py psd-comparison $@

# PSD comparison with custom parameters
results/figures/psd_comparison_alpha1.5.png: .venv src/utils/plots.py src/quantum/noise_models.py
	uv run src/utils/plots.py psd-comparison $@ --alpha 1.5 --amplitude 0.2 --cutoff 10.0

# Generate all standard figures
all-figures: \
	results/figures/one_over_f.png \
	results/figures/psd_comparison.png \
	results/figures/noise_lorentzian.png \
	results/figures/noise_double_exp.png

