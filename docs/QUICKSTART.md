# Quick Start Guide

## Prerequisites

Install uv (modern Python package manager):

```bash
# Mac/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/meta-quantum-control.git
cd meta-quantum-control

# Install dependencies and create virtual environment
uv sync
```

## Test

Run the test suite:

```bash
# Run all tests
make test

# Or directly with uv
uv run pytest tests/

# Run minimal working example
uv run python scripts/test_minimal_working.py
```

Expected: `✓✓✓ ALL TESTS PASSED ✓✓✓`

## Train

```bash
# Run meta-learning training
uv run python experiments/train_meta.py --config configs/test_config.yaml

# Or activate environment first
source .venv/bin/activate  # Mac/Linux
python experiments/train_meta.py --config configs/test_config.yaml
```

## Generate Figures

```bash
# Generate all publication-quality figures
make all-figures
```

## Troubleshooting

See `DEBUG_PROTOCOL.md`
