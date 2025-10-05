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
