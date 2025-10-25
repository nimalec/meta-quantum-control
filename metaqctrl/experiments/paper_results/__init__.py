"""
Paper Results Generation Package

This package contains all scripts to generate figures and validate theoretical
predictions for the ICML 2026 paper on Meta-RL for Quantum Control.

Main modules:
- experiment_gap_vs_k: Validates Gap(P,K) ∝ (1 - e^(-μηK))
- experiment_gap_vs_variance: Validates Gap(P,K) ∝ σ²_S
- experiment_constants_validation: Estimates and validates physics constants
- generate_all_results: Master script to run all experiments

Usage:
    python generate_all_results.py --meta_path path/to/maml.pt \
                                    --robust_path path/to/robust.pt
"""

__version__ = "1.0.0"
__author__ = "Nima Leclerc, Nicholas Brawand"
