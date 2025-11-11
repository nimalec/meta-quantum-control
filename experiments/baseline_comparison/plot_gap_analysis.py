"""
Plot gap analysis results comparing baselines.

This script generates publication-quality figures for:
1. Gap vs K (adaptation steps)
2. Gap vs Variance (task diversity)

Usage:
    python plot_gap_analysis.py --results_dir results/baseline_comparison
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import argparse
from typing import Dict


def plot_gap_vs_k(results: Dict, output_path: str):
    """
    Plot gap vs K for both baselines.

    Shows:
    - Meta K=0 baseline (horizontal line)
    - Average task baseline (horizontal line)
    - Meta adapted performance vs K
    - Gap relative to each baseline
    """
    sns.set_style("whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    k_values = np.array(results['k_values'])

    # Extract data
    meta_k0_mean = results['meta_k0_mean'][0]
    meta_k0_std = results['meta_k0_std'][0]

    avg_task_mean = results['average_task_mean'][0]
    avg_task_std = results['average_task_std'][0]

    meta_adapted_mean = np.array(results['meta_adapted_mean'])
    meta_adapted_std = np.array(results['meta_adapted_std'])

    gap_vs_k0_mean = np.array(results['gap_vs_meta_k0_mean'])
    gap_vs_k0_std = np.array(results['gap_vs_meta_k0_std'])

    gap_vs_avg_mean = np.array(results['gap_vs_average_mean'])
    gap_vs_avg_std = np.array(results['gap_vs_average_std'])

    # Plot 1: Absolute Fidelities
    ax1.axhline(
        y=meta_k0_mean,
        color='blue',
        linestyle='--',
        linewidth=2,
        label=f'Meta K=0 (no adaptation): {meta_k0_mean:.4f}'
    )
    ax1.fill_between(
        k_values,
        meta_k0_mean - meta_k0_std,
        meta_k0_mean + meta_k0_std,
        color='blue',
        alpha=0.2
    )

    ax1.axhline(
        y=avg_task_mean,
        color='green',
        linestyle='--',
        linewidth=2,
        label=f'Average Task: {avg_task_mean:.4f}'
    )
    ax1.fill_between(
        k_values,
        avg_task_mean - avg_task_std,
        avg_task_mean + avg_task_std,
        color='green',
        alpha=0.2
    )

    ax1.errorbar(
        k_values,
        meta_adapted_mean,
        yerr=meta_adapted_std,
        fmt='o-',
        markersize=8,
        capsize=5,
        capthick=2,
        label='Meta Adapted',
        color='red',
        linewidth=2
    )

    ax1.set_xlabel('Adaptation Steps (K)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Fidelity', fontsize=14, fontweight='bold')
    ax1.set_title('Performance vs Adaptation Steps', fontsize=16, fontweight='bold')
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=12)

    # Plot 2: Gaps
    ax2.errorbar(
        k_values,
        gap_vs_k0_mean,
        yerr=gap_vs_k0_std,
        fmt='o-',
        markersize=8,
        capsize=5,
        capthick=2,
        label='Gap vs Meta K=0',
        color='blue',
        linewidth=2
    )

    ax2.errorbar(
        k_values,
        gap_vs_avg_mean,
        yerr=gap_vs_avg_std,
        fmt='s-',
        markersize=8,
        capsize=5,
        capthick=2,
        label='Gap vs Average Task',
        color='green',
        linewidth=2
    )

    ax2.axhline(y=0, color='black', linestyle=':', linewidth=1)

    ax2.set_xlabel('Adaptation Steps (K)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Fidelity Gap', fontsize=14, fontweight='bold')
    ax2.set_title('Adaptation Gain vs Baselines', fontsize=16, fontweight='bold')
    ax2.legend(fontsize=11, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {output_path}")

    # Also save PNG
    png_path = output_path.replace('.pdf', '.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {png_path}")

    plt.close()


def plot_gap_vs_variance(results: Dict, output_path: str):
    """
    Plot gap vs variance for both baselines.

    Shows how the gap scales with task diversity (variance).
    """
    sns.set_style("whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    variances = np.array(results['variances'])
    K_fixed = results['K_fixed']

    # Extract data
    meta_k0_mean = np.array(results['meta_k0_mean'])
    meta_k0_std = np.array(results['meta_k0_std'])

    avg_task_mean = np.array(results['average_task_mean'])
    avg_task_std = np.array(results['average_task_std'])

    meta_adapted_mean = np.array(results['meta_adapted_mean'])
    meta_adapted_std = np.array(results['meta_adapted_std'])

    gap_vs_k0_mean = np.array(results['gap_vs_meta_k0_mean'])
    gap_vs_k0_std = np.array(results['gap_vs_meta_k0_std'])

    gap_vs_avg_mean = np.array(results['gap_vs_average_mean'])
    gap_vs_avg_std = np.array(results['gap_vs_average_std'])

    # Plot 1: Absolute Fidelities vs Variance
    ax1.errorbar(
        variances,
        meta_k0_mean,
        yerr=meta_k0_std,
        fmt='o-',
        markersize=8,
        capsize=5,
        capthick=2,
        label='Meta K=0',
        color='blue',
        linewidth=2
    )

    ax1.errorbar(
        variances,
        avg_task_mean,
        yerr=avg_task_std,
        fmt='s-',
        markersize=8,
        capsize=5,
        capthick=2,
        label='Average Task',
        color='green',
        linewidth=2
    )

    ax1.errorbar(
        variances,
        meta_adapted_mean,
        yerr=meta_adapted_std,
        fmt='^-',
        markersize=8,
        capsize=5,
        capthick=2,
        label=f'Meta K={K_fixed}',
        color='red',
        linewidth=2
    )

    ax1.set_xlabel('Task Variance ($\\sigma^2$)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Fidelity', fontsize=14, fontweight='bold')
    ax1.set_title('Performance vs Task Diversity', fontsize=16, fontweight='bold')
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=12)

    # Plot 2: Gaps vs Variance
    ax2.errorbar(
        variances,
        gap_vs_k0_mean,
        yerr=gap_vs_k0_std,
        fmt='o-',
        markersize=8,
        capsize=5,
        capthick=2,
        label='Gap vs Meta K=0',
        color='blue',
        linewidth=2
    )

    ax2.errorbar(
        variances,
        gap_vs_avg_mean,
        yerr=gap_vs_avg_std,
        fmt='s-',
        markersize=8,
        capsize=5,
        capthick=2,
        label='Gap vs Average Task',
        color='green',
        linewidth=2
    )

    # Fit linear models
    from scipy.optimize import curve_fit

    def linear_model(x, slope):
        return slope * x

    try:
        # Fit gap vs K=0
        popt_k0, _ = curve_fit(
            linear_model,
            variances,
            gap_vs_k0_mean,
            sigma=gap_vs_k0_std,
            absolute_sigma=True
        )
        var_fine = np.linspace(0, max(variances), 100)
        ax2.plot(
            var_fine,
            linear_model(var_fine, popt_k0[0]),
            '--',
            color='blue',
            linewidth=2,
            alpha=0.5,
            label=f'Linear fit (vs K=0): slope={popt_k0[0]:.2f}'
        )

        # Fit gap vs Average
        popt_avg, _ = curve_fit(
            linear_model,
            variances,
            gap_vs_avg_mean,
            sigma=gap_vs_avg_std,
            absolute_sigma=True
        )
        ax2.plot(
            var_fine,
            linear_model(var_fine, popt_avg[0]),
            '--',
            color='green',
            linewidth=2,
            alpha=0.5,
            label=f'Linear fit (vs Avg): slope={popt_avg[0]:.2f}'
        )
    except Exception as e:
        print(f"Warning: Could not fit linear models: {e}")

    ax2.axhline(y=0, color='black', linestyle=':', linewidth=1)

    ax2.set_xlabel('Task Variance ($\\sigma^2$)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Fidelity Gap', fontsize=14, fontweight='bold')
    ax2.set_title(f'Adaptation Gain vs Task Diversity (K={K_fixed})', fontsize=16, fontweight='bold')
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {output_path}")

    # Also save PNG
    png_path = output_path.replace('.pdf', '.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {png_path}")

    plt.close()


def plot_combined_summary(
    gap_vs_k_results: Dict,
    gap_vs_var_results: Dict,
    output_path: str
):
    """
    Create a combined 2x2 summary figure.
    """
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Top left: Fidelity vs K
    ax1 = fig.add_subplot(gs[0, 0])
    k_values = np.array(gap_vs_k_results['k_values'])
    meta_k0_mean = gap_vs_k_results['meta_k0_mean'][0]
    avg_task_mean = gap_vs_k_results['average_task_mean'][0]
    meta_adapted_mean = np.array(gap_vs_k_results['meta_adapted_mean'])

    ax1.axhline(y=meta_k0_mean, color='blue', linestyle='--', linewidth=2, label='Meta K=0')
    ax1.axhline(y=avg_task_mean, color='green', linestyle='--', linewidth=2, label='Average Task')
    ax1.plot(k_values, meta_adapted_mean, 'ro-', markersize=8, linewidth=2, label='Meta Adapted')

    ax1.set_xlabel('Adaptation Steps (K)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Fidelity', fontsize=12, fontweight='bold')
    ax1.set_title('(A) Performance vs K', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Top right: Gap vs K
    ax2 = fig.add_subplot(gs[0, 1])
    gap_vs_k0_mean = np.array(gap_vs_k_results['gap_vs_meta_k0_mean'])
    gap_vs_avg_mean = np.array(gap_vs_k_results['gap_vs_average_mean'])

    ax2.plot(k_values, gap_vs_k0_mean, 'bo-', markersize=8, linewidth=2, label='vs Meta K=0')
    ax2.plot(k_values, gap_vs_avg_mean, 'gs-', markersize=8, linewidth=2, label='vs Average Task')
    ax2.axhline(y=0, color='black', linestyle=':', linewidth=1)

    ax2.set_xlabel('Adaptation Steps (K)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Fidelity Gap', fontsize=12, fontweight='bold')
    ax2.set_title('(B) Adaptation Gain vs K', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Bottom left: Fidelity vs Variance
    ax3 = fig.add_subplot(gs[1, 0])
    variances = np.array(gap_vs_var_results['variances'])
    meta_k0_var_mean = np.array(gap_vs_var_results['meta_k0_mean'])
    avg_task_var_mean = np.array(gap_vs_var_results['average_task_mean'])
    meta_adapted_var_mean = np.array(gap_vs_var_results['meta_adapted_mean'])

    ax3.plot(variances, meta_k0_var_mean, 'bo-', markersize=8, linewidth=2, label='Meta K=0')
    ax3.plot(variances, avg_task_var_mean, 'gs-', markersize=8, linewidth=2, label='Average Task')
    ax3.plot(variances, meta_adapted_var_mean, 'r^-', markersize=8, linewidth=2,
             label=f'Meta K={gap_vs_var_results["K_fixed"]}')

    ax3.set_xlabel('Task Variance ($\\sigma^2$)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Fidelity', fontsize=12, fontweight='bold')
    ax3.set_title('(C) Performance vs Variance', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Bottom right: Gap vs Variance
    ax4 = fig.add_subplot(gs[1, 1])
    gap_vs_k0_var_mean = np.array(gap_vs_var_results['gap_vs_meta_k0_mean'])
    gap_vs_avg_var_mean = np.array(gap_vs_var_results['gap_vs_average_mean'])

    ax4.plot(variances, gap_vs_k0_var_mean, 'bo-', markersize=8, linewidth=2, label='vs Meta K=0')
    ax4.plot(variances, gap_vs_avg_var_mean, 'gs-', markersize=8, linewidth=2, label='vs Average Task')
    ax4.axhline(y=0, color='black', linestyle=':', linewidth=1)

    ax4.set_xlabel('Task Variance ($\\sigma^2$)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Fidelity Gap', fontsize=12, fontweight='bold')
    ax4.set_title('(D) Adaptation Gain vs Variance', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Combined figure saved to {output_path}")

    # Also save PNG
    png_path = output_path.replace('.pdf', '.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {png_path}")

    plt.close()


def main(args):
    """Main plotting function."""

    print("=" * 70)
    print("PLOTTING BASELINE COMPARISON RESULTS")
    print("=" * 70)

    results_dir = Path(args.results_dir)

    # Load results
    gap_vs_k_path = results_dir / 'gap_vs_k_results.json'
    gap_vs_var_path = results_dir / 'gap_vs_variance_results.json'

    if gap_vs_k_path.exists():
        print(f"\nLoading gap vs K results from {gap_vs_k_path}...")
        with open(gap_vs_k_path, 'r') as f:
            gap_vs_k_results = json.load(f)

        # Plot gap vs K
        output_path = results_dir / 'gap_vs_k.pdf'
        plot_gap_vs_k(gap_vs_k_results, str(output_path))

    else:
        print(f"\nWarning: {gap_vs_k_path} not found. Skipping gap vs K plot.")
        gap_vs_k_results = None

    if gap_vs_var_path.exists():
        print(f"\nLoading gap vs variance results from {gap_vs_var_path}...")
        with open(gap_vs_var_path, 'r') as f:
            gap_vs_var_results = json.load(f)

        # Plot gap vs variance
        output_path = results_dir / 'gap_vs_variance.pdf'
        plot_gap_vs_variance(gap_vs_var_results, str(output_path))

    else:
        print(f"\nWarning: {gap_vs_var_path} not found. Skipping gap vs variance plot.")
        gap_vs_var_results = None

    # Create combined summary if both available
    if gap_vs_k_results is not None and gap_vs_var_results is not None:
        print("\nCreating combined summary figure...")
        output_path = results_dir / 'combined_summary.pdf'
        plot_combined_summary(gap_vs_k_results, gap_vs_var_results, str(output_path))

    print("\n" + "=" * 70)
    print("PLOTTING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Plot baseline comparison results'
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default='results/baseline_comparison',
        help='Directory containing results JSON files'
    )

    args = parser.parse_args()
    main(args)
