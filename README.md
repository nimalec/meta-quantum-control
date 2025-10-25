# Meta-RL for Quantum Control
Implementation of "Meta-Reinforcement Learning for Quantum Control: Generalization and Robustness under Noise Shifts"

## Quick Start

### Prerequisites
You must have uv installed:
```bash
# Mac/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Installation & Usage
**Option 1: Quick start (recommended)**
```bash
# Automatically sets up environment and generates figures
make all-figures
```

**Option 2: Manual setup**
```bash
# Install dependencies (creates .venv and installs all packages from pyproject.toml)
uv sync

# Run experiments
uv run python experiments/train_meta.py

# Or activate the environment
source .venv/bin/activate  # Mac/Linux
.venv\Scripts\activate     # Windows
```

## Testing

```bash
# Run all tests
make test

# Or directly with uv
uv run pytest tests/

# Run tests with verbose output
make test-verbose

# Run tests with coverage report
make test-coverage
```

## Documentation
See `docs/` directory for complete documentation.

## Contributing
```bash
# Adding packages
uv add <package>

# Running experiments
uv run python experiments/train_meta.py --config configs/experiment_config.yaml
```

## Citation

```bibtex
@inproceedings{leclerc2025meta,
  title={Meta-Reinforcement Learning for Quantum Control},
  author={Leclerc, Nima; Brawand, Nicholas},
  booktitle={ICML},
  year={2025}
}
```

## License

MIT License - see LICENSE file
