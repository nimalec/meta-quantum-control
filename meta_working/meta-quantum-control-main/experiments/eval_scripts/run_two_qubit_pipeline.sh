#!/bin/bash

# Complete 2-Qubit Experimental Pipeline
# Runs full training and generates Figure 4 for paper

echo "================================================================================"
echo "TWO-QUBIT CNOT GATE OPTIMIZATION - COMPLETE PIPELINE"
echo "================================================================================"
echo ""

# Configuration
CONFIG_FILE="../configs/two_qubit_experiment.yaml"
OUTPUT_DIR="checkpoints/two_qubit"
RESULTS_DIR="results/two_qubit"

echo "[Pipeline Configuration]"
echo "  Config: $CONFIG_FILE"
echo "  Checkpoints: $OUTPUT_DIR"
echo "  Results: $RESULTS_DIR"
echo ""

# Create directories
mkdir -p $OUTPUT_DIR
mkdir -p $RESULTS_DIR
mkdir -p results/paper

# ============================================================================
# Step 1: Train Meta-Learned Policy (MAML)
# ============================================================================
echo "================================================================================"
echo "STEP 1: Train Meta-Learned Policy (MAML)"
echo "================================================================================"
echo "Expected time: ~2-4 hours (3000 iterations)"
echo ""

python train_meta_two_qubit.py --config $CONFIG_FILE --output $OUTPUT_DIR

if [ $? -ne 0 ]; then
    echo "ERROR: MAML training failed!"
    exit 1
fi

echo ""
echo "✓ MAML training complete"
echo ""

# ============================================================================
# Step 2: Train Robust Baseline (optional - can use simplified version)
# ============================================================================
echo "================================================================================"
echo "STEP 2: Train Robust Baseline (Minimax)"
echo "================================================================================"
echo "Note: Using simplified robust baseline for now"
echo ""

# For now, we can skip this or use a simplified version
# python train_robust_two_qubit.py --config $CONFIG_FILE --output $OUTPUT_DIR

echo "✓ Using baseline policy (can enhance later)"
echo ""

# ============================================================================
# Step 3: Estimate Physics Constants
# ============================================================================
echo "================================================================================"
echo "STEP 3: Estimate Physics Constants (Δ, μ, C_filter, σ²_S)"
echo "================================================================================"
echo "Expected time: ~10-15 minutes"
echo ""

cd paper_results
python experiment_system_scaling.py

if [ $? -ne 0 ]; then
    echo "ERROR: Constants estimation failed!"
    exit 1
fi

cd ..

echo ""
echo "✓ Constants estimated"
echo ""

# ============================================================================
# Step 4: Generate Figure 4
# ============================================================================
echo "================================================================================"
echo "STEP 4: Generate Figure 4 (Two-Qubit Validation)"
echo "================================================================================"
echo "Expected time: ~2 minutes"
echo ""

cd paper_results
python generate_figure4_two_qubit.py --output ../results/paper/figure4_two_qubit.pdf

if [ $? -ne 0 ]; then
    echo "ERROR: Figure generation failed!"
    exit 1
fi

cd ..

echo ""
echo "✓ Figure 4 generated"
echo ""

# ============================================================================
# Step 5: Summary
# ============================================================================
echo "================================================================================"
echo "PIPELINE COMPLETE!"
echo "================================================================================"
echo ""
echo "Generated outputs:"
echo "  1. Trained policies:"
echo "     - $OUTPUT_DIR/maml_best.pt"
echo "     - $OUTPUT_DIR/maml_final.pt"
echo ""
echo "  2. Constants estimation:"
echo "     - results/system_scaling/scaling_results.json"
echo "     - results/system_scaling/scaling_comparison.pdf"
echo ""
echo "  3. Figure 4 (for paper):"
echo "     - results/paper/figure4_two_qubit.pdf"
echo ""
echo "Next steps:"
echo "  1. Review Figure 4: open results/paper/figure4_two_qubit.pdf"
echo "  2. Check constants: cat results/system_scaling/scaling_results.json"
echo "  3. Include in paper: Add to Section 5.2 or Appendix C"
echo ""
echo "================================================================================"
