#!/bin/bash
set -e

echo "Installing meta-quantum-control..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install package
pip install -e ".[dev]"

echo "✓ Installation complete!"
echo "Activate: source venv/bin/activate"
echo "Test: python scripts/test_minimal_working.py"
