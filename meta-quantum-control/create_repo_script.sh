#!/bin/bash
# Script to create complete repository structure
# Run this to generate the full meta-quantum-control directory

set -e

REPO_NAME="meta-quantum-control"

echo "Creating repository: $REPO_NAME"
echo "================================"

# Create directory structure
mkdir -p $REPO_NAME/{configs,src/{quantum,meta_rl,baselines,theory,utils},experiments,tests,scripts,docs,notebooks,.github/workflows}

cd $REPO_NAME

# Create __init__.py files
touch src/__init__.py
touch src/quantum/__init__.py
touch src/meta_rl/__init__.py
touch src/baselines/__init__.py
touch src/theory/__init__.py
touch src/utils/__init__.py
touch experiments/__init__.py
touch tests/__init__.py

echo "✓ Directory structure created"

# Create .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# Experiments
checkpoints/
checkpoints_test/
results/
figures/
wandb/
*.pt
*.pth

# Data
data/
*.csv
*.h5

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/
EOF

echo "✓ .gitignore created"

# Create LICENSE
cat > LICENSE << 'EOF'
MIT License

Copyright (c) 2025 Nima Leclerc

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOF

echo "✓ LICENSE created"

# Create pyproject.toml
cat > pyproject.toml << 'EOF'
[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "meta-quantum-control"
version = "1.0.0"
description = "Meta-RL for Quantum Control with Optimality Gap Theory"
authors = [{name = "Nima Leclerc", email = "nleclerc@mitre.org"}]
readme = "README.md"
requires-python = ">=3.8"
license = {text = "MIT"}

[tool.black]
line-length = 100
target-version = ['py38', 'py39', 'py310']

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
EOF

echo "✓ pyproject.toml created"

# Create installation script
cat > scripts/install.sh << 'EOF'
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
EOF

chmod +x scripts/install.sh

echo "✓ Installation script created"

# Create test config
cat > configs/test_config.yaml << 'EOF'
# Quick test configuration
seed: 42
psd_model: 'one_over_f'
horizon: 1.0
target_gate: 'hadamard'

task_dist_type: 'uniform'
alpha_range: [0.5, 2.0]
A_range: [0.05, 0.3]
omega_c_range: [2.0, 8.0]

task_feature_dim: 3
hidden_dim: 32
n_hidden_layers: 2
n_segments: 10
n_controls: 2
output_scale: 0.5

inner_lr: 0.01
inner_steps: 3
meta_lr: 0.001
first_order: true

n_iterations: 10
tasks_per_batch: 2
n_support: 5
n_query: 5
log_interval: 5
val_interval: 10

save_dir: 'checkpoints_test'
integration_method: 'RK45'
EOF

echo "✓ Test config created"

# Create GitHub Actions
mkdir -p .github/workflows
cat > .github/workflows/test.yml << 'EOF'
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.8', '3.9', '3.10']
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
    
    - name: Run tests
      run: |
        pytest tests/ -v
EOF

echo "✓ GitHub Actions created"

# Create quick start doc
cat > docs/QUICKSTART.md << 'EOF'
# Quick Start Guide

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/meta-quantum-control.git
cd meta-quantum-control
./scripts/install.sh
```

## Test

```bash
python scripts/test_minimal_working.py
```

Expected: `✓✓✓ ALL TESTS PASSED ✓✓✓`

## Train

```bash
python experiments/train_meta.py --config configs/test_config.yaml
```

## Troubleshooting

See `DEBUG_PROTOCOL.md`
EOF

echo "✓ Documentation created"

# Create placeholder README
cat > README.md << 'EOF'
# Meta-RL for Quantum Control

[![Tests](https://github.com/YOUR_USERNAME/meta-quantum-control/workflows/Tests/badge.svg)](https://github.com/YOUR_USERNAME/meta-quantum-control/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> Implementation of "Meta-Reinforcement Learning for Quantum Control: Generalization and Robustness under Noise Shifts"

## Quick Start

```bash
./scripts/install.sh
python scripts/test_minimal_working.py
```

## Documentation

See `docs/` directory for complete documentation.

## Citation

```bibtex
@inproceedings{leclerc2025meta,
  title={Meta-Reinforcement Learning for Quantum Control},
  author={Leclerc, Nima},
  booktitle={ICML},
  year={2025}
}
```

## License

MIT License - see LICENSE file
EOF

echo "✓ README created"

echo ""
echo "================================"
echo "Repository structure created!"
echo "================================"
echo ""
echo "Next steps:"
echo "1. Copy your source files into src/ directories"
echo "2. Copy experiment scripts into experiments/"
echo "3. Copy test files into tests/"
echo "4. Copy documentation into docs/"
echo ""
echo "Files to copy:"
echo "  - All artifacts from this conversation"
echo "  - Specifically the FIXED versions of:"
echo "    * src/theory/quantum_environment.py"
echo "    * src/theory/physics_constants.py"
echo "    * src/quantum/noise_models.py (fixed PSD mapping)"
echo "    * experiments/train_meta.py (fixed)"
echo "    * scripts/test_minimal_working.py"
echo "    * docs/DEBUG_PROTOCOL.md"
echo ""
echo "Then run:"
echo "  cd $REPO_NAME"
echo "  git init"
echo "  git add ."
echo "  git commit -m 'Initial commit'"
echo "  git remote add origin https://github.com/YOUR_USERNAME/$REPO_NAME.git"
echo "  git push -u origin main"
