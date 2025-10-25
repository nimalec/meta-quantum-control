#!/bin/bash
set -e

echo "Installing meta-quantum-control with uv..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "Please restart your shell or run: source $HOME/.cargo/env"
    exit 1
fi

# Install dependencies and create virtual environment (editable mode)
echo "Installing dependencies in editable mode..."
uv sync --all-groups

echo ""
echo "✓ Installation complete!"
echo ""
echo "To activate the environment:"
echo "  source .venv/bin/activate"
echo ""
echo "To run tests:"
echo "  make test"
echo "  uv run python scripts/test_minimal_working.py"
echo ""
echo "To run experiments:"
echo "  uv run python experiments/train_meta.py --config configs/experiment_config.yaml"
