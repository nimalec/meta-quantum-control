# Makefile for generating publication-quality figures
# Usage: make <target>
# Example: make results/figures/noise.png

.PHONY: help clean all-figures test test-verbose test-coverage

clean:
	rm -f results/figures/*.png

.venv:
	uv sync --all-groups

# ============================================================================
# Testing
# ============================================================================

# Run all tests
test: .venv
	@echo "Running tests..."
	uv run pytest tests/

# Run tests with verbose output
test-verbose: .venv
	@echo "Running tests (verbose)..."
	uv run pytest -v tests/

# Run tests with coverage report
test-coverage: .venv
	@echo "Running tests with coverage..."
	uv run pytest --cov=metaqctrl --cov-report=term-missing --cov-report=html tests/
	@echo "Coverage report generated in htmlcov/index.html"

# ============================================================================
# Noise PSD Figures
# ============================================================================

# Default noise plot with 5 tasks
results/figures/one_over_f.png: .venv metaqctrl/utils/plots.py metaqctrl/quantum/noise_models.py
	uv run python metaqctrl/utils/plots.py noise $@ --model one_over_f --n-tasks 5 --seed 42

# Noise plot with different models
results/figures/noise_lorentzian.png: .venv metaqctrl/utils/plots.py metaqctrl/quantum/noise_models.py
	uv run python metaqctrl/utils/plots.py noise $@ --model lorentzian --n-tasks 5 --seed 42

results/figures/noise_double_exp.png: .venv metaqctrl/utils/plots.py metaqctrl/quantum/noise_models.py
	uv run python metaqctrl/utils/plots.py noise $@ --model double_exp --n-tasks 5 --seed 42

# Noise plot with distance calculations
results/figures/noise_with_distances.png: .venv metaqctrl/utils/plots.py metaqctrl/quantum/noise_models.py
	uv run python metaqctrl/utils/plots.py noise $@ --n-tasks 5 --seed 42 --distances

# ============================================================================
# PSD Comparison Figures
# ============================================================================

# Default PSD comparison
results/figures/psd_comparison.png: .venv metaqctrl/utils/plots.py metaqctrl/quantum/noise_models.py
	uv run python metaqctrl/utils/plots.py psd-comparison $@

# PSD comparison with custom parameters
results/figures/psd_comparison_alpha1.5.png: .venv metaqctrl/utils/plots.py metaqctrl/quantum/noise_models.py
	uv run python metaqctrl/utils/plots.py psd-comparison $@ --alpha 1.5 --amplitude 0.2 --cutoff 10.0

# ============================================================================
# Effective Decay Rates Figures
# ============================================================================

# Default effective rates plot
results/figures/effective_rates.png: .venv metaqctrl/utils/plots.py metaqctrl/quantum/noise_models.py
	uv run python metaqctrl/utils/plots.py effective-rates $@ --n-tasks 5 --seed 42

# Effective rates with more sampling frequencies
results/figures/effective_rates_detailed.png: .venv metaqctrl/utils/plots.py metaqctrl/quantum/noise_models.py
	uv run python metaqctrl/utils/plots.py effective-rates $@ --n-tasks 5 --n-freqs 20 --seed 42

# Generate all standard figures
all-figures: \
	results/figures/one_over_f.png \
	results/figures/psd_comparison.png \
	results/figures/noise_lorentzian.png \
	results/figures/noise_double_exp.png \
	results/figures/effective_rates.png


results/gap_vs_k/figure.pdf:
	uv run experiments/paper_results/experiment_gap_vs_k.py \
		--meta-path experiments/train_scripts/checkpoints/maml_best_pauli_x_policy.pt  \
		--robust-path experiments/train_scripts/checkpoints/robust_minimax_20251103_184653_best_policy.pt
