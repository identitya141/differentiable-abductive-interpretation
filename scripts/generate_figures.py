#!/usr/bin/env python3
"""
Generate Paper Figures from Experiment Results

Creates publication-ready figures for the DAI paper.

Usage:
    python scripts/generate_figures.py --results-dir results/ --output-dir docs/figures/
    python scripts/generate_figures.py --figure gen_gap  # Generate specific figure
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Plotting setup
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Publication-quality settings
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'serif',
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.figsize': (6, 4),
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
    })
    
    # Colorblind-friendly palette
    COLORS = sns.color_palette("colorblind")
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("Warning: matplotlib/seaborn not installed. Skipping figure generation.")


def load_training_logs(log_dir: Path) -> Dict:
    """Load training logs for plotting."""
    logs = {}
    
    for log_file in log_dir.glob("**/*.json"):
        with open(log_file) as f:
            logs[log_file.stem] = json.load(f)
    
    return logs


def load_depth_series(results_dir: Path) -> Dict[str, List[Tuple[int, float, float]]]:
    """Load real mean and sample-standard-deviation accuracy by depth."""
    series = {}
    for report_path in sorted(results_dir.glob("*_aggregated.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        depth_metrics = report.get("final_evaluation", {}).get(
            "accuracy_by_depth", {}
        )
        points = [
            (
                int(depth),
                float(summary["mean"]),
                float(summary.get("std", 0.0)),
            )
            for depth, summary in depth_metrics.items()
            if isinstance(summary, dict) and "mean" in summary
        ]
        if points:
            method_name = report_path.stem.removesuffix("_aggregated")
            series[method_name] = sorted(points)

    if not series:
        raise ValueError(
            f"No aggregated depth metrics found under {results_dir}. "
            "Run scripts/aggregate_results.py first."
        )
    return series


def figure_depth_accuracy(results_dir: Path, output_path: Path):
    """Plot OOD exact-match accuracy against compositional depth."""
    if not HAS_PLOTTING:
        raise RuntimeError("matplotlib and seaborn are required")

    fig, axis = plt.subplots(1, 1, figsize=(7.2, 4.5))
    for method_name, points in load_depth_series(results_dir).items():
        depths = [point[0] for point in points]
        means = [100.0 * point[1] for point in points]
        standard_deviations = [100.0 * point[2] for point in points]
        line = axis.plot(
            depths, means, marker="o", linewidth=1.8, label=method_name
        )[0]
        lower = [
            max(0.0, mean - deviation)
            for mean, deviation in zip(means, standard_deviations)
        ]
        upper = [
            min(100.0, mean + deviation)
            for mean, deviation in zip(means, standard_deviations)
        ]
        axis.fill_between(
            depths, lower, upper, color=line.get_color(), alpha=0.15
        )

    axis.set_xlabel("Compositional depth")
    axis.set_ylabel("OOD exact match (%)")
    axis.set_ylim(0.0, 100.0)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {output_path}")


def figure_generalization_gap(results_dir: Path, output_path: Path):
    """
    Figure 2: Generalization Gap Analysis
    
    Shows how the gap between IID and OOD accuracy evolves during training
    for baseline vs. DAI.
    """
    if not HAS_PLOTTING:
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    
    # Simulated data for template (replace with actual data)
    steps = np.arange(0, 10001, 100)
    
    # Baseline
    baseline_iid = 0.5 + 0.45 * (1 - np.exp(-steps / 2000))
    baseline_ood = 0.3 + 0.15 * (1 - np.exp(-steps / 2000))
    baseline_gap = baseline_iid - baseline_ood
    
    # DAI
    dai_iid = 0.5 + 0.43 * (1 - np.exp(-steps / 2000))
    dai_ood = 0.3 + 0.50 * (1 - np.exp(-steps / 3000))
    dai_gap = dai_iid - dai_ood
    
    # Plot IID and OOD for both
    ax.plot(steps, baseline_iid * 100, '--', color=COLORS[0], label='Baseline IID')
    ax.plot(steps, baseline_ood * 100, '-', color=COLORS[0], label='Baseline OOD')
    ax.fill_between(steps, baseline_ood * 100, baseline_iid * 100, 
                    color=COLORS[0], alpha=0.2)
    
    ax.plot(steps, dai_iid * 100, '--', color=COLORS[1], label='DAI IID')
    ax.plot(steps, dai_ood * 100, '-', color=COLORS[1], label='DAI OOD')
    ax.fill_between(steps, dai_ood * 100, dai_iid * 100, 
                    color=COLORS[1], alpha=0.2)
    
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Generalization Gap During Training')
    ax.legend(loc='lower right')
    ax.set_xlim(0, 10000)
    ax.set_ylim(0, 100)
    
    plt.savefig(output_path)
    plt.close()
    print(f"Figure saved to {output_path}")


def figure_abstraction_comparison(results_dir: Path, output_path: Path):
    """
    Figure 3: Abstract Domain Comparison
    
    Bar chart comparing different abstract domains across datasets.
    """
    if not HAS_PLOTTING:
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    
    # Data (replace with actual results)
    domains = ['None', 'Type', 'Interval', 'Mono', 'TypeMono']
    datasets = ['SCAN', 'COGS', 'CFQ', 'CLUTRR']
    
    # Simulated data
    data = {
        'SCAN': [20, 65, 45, 55, 85],
        'COGS': [35, 70, 50, 60, 80],
        'CFQ': [15, 35, 25, 30, 45],
        'CLUTRR': [40, 60, 55, 65, 75],
    }
    
    x = np.arange(len(domains))
    width = 0.2
    
    for i, dataset in enumerate(datasets):
        offset = (i - 1.5) * width
        ax.bar(x + offset, data[dataset], width, label=dataset, color=COLORS[i])
    
    ax.set_xlabel('Abstract Domain')
    ax.set_ylabel('OOD Accuracy (%)')
    ax.set_title('Effect of Abstract Domain Choice')
    ax.set_xticks(x)
    ax.set_xticklabels(domains)
    ax.legend()
    ax.set_ylim(0, 100)
    
    plt.savefig(output_path)
    plt.close()
    print(f"Figure saved to {output_path}")


def figure_scaling_analysis(results_dir: Path, output_path: Path):
    """
    Figure 4: Scaling Analysis
    
    Shows how accuracy degrades with compositional depth for baseline vs. DAI.
    """
    if not HAS_PLOTTING:
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    
    # Data
    depths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    # Simulated data (replace with actual)
    baseline_acc = [95, 90, 80, 65, 50, 40, 30, 25, 20, 18]
    dai_acc = [95, 92, 88, 82, 75, 70, 65, 60, 55, 50]
    
    ax.plot(depths, baseline_acc, 'o-', color=COLORS[0], 
            label='Baseline', markersize=6)
    ax.plot(depths, dai_acc, 's-', color=COLORS[1], 
            label='DAI', markersize=6)
    
    # Add training regime indicator
    ax.axvspan(1, 5, alpha=0.1, color='gray', label='In-distribution')
    ax.axvline(x=5, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Compositional Depth')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy vs. Compositional Depth')
    ax.legend()
    ax.set_xlim(1, 10)
    ax.set_ylim(0, 100)
    ax.set_xticks(depths)
    
    plt.savefig(output_path)
    plt.close()
    print(f"Figure saved to {output_path}")


def figure_lambda_sensitivity(results_dir: Path, output_path: Path):
    """
    Figure 5: Abstraction Loss Weight Sensitivity
    
    Shows how performance varies with lambda, highlighting the sweet spot.
    """
    if not HAS_PLOTTING:
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    
    # Data
    lambdas = [0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    
    # Simulated data
    iid_acc = [95, 94, 93, 92, 90, 85, 75, 60]
    ood_acc = [20, 35, 55, 75, 80, 78, 65, 45]
    
    ax.plot(lambdas, iid_acc, 'o-', color=COLORS[0], 
            label='IID Accuracy', markersize=6)
    ax.plot(lambdas, ood_acc, 's-', color=COLORS[1], 
            label='OOD Accuracy', markersize=6)
    
    # Highlight sweet spot
    best_idx = np.argmax(ood_acc)
    ax.axvline(x=lambdas[best_idx], color=COLORS[2], linestyle='--', 
               alpha=0.5, label=f'Best λ={lambdas[best_idx]}')
    
    ax.set_xlabel('Abstraction Loss Weight (λ)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Effect of Abstraction Loss Weight')
    ax.legend()
    ax.set_xscale('symlog', linthresh=0.01)
    ax.set_ylim(0, 100)
    
    plt.savefig(output_path)
    plt.close()
    print(f"Figure saved to {output_path}")


def figure_representation_tsne(results_dir: Path, output_path: Path):
    """
    Figure 6: Representation Visualization
    
    t-SNE visualization of hidden representations colored by abstract type.
    """
    if not HAS_PLOTTING:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Simulated data (replace with actual representations)
    np.random.seed(42)
    n_points = 200
    n_types = 4
    
    # Baseline: mixed representations
    baseline_x = np.random.randn(n_points)
    baseline_y = np.random.randn(n_points)
    baseline_types = np.random.randint(0, n_types, n_points)
    
    # DAI: clustered by type
    dai_x = []
    dai_y = []
    dai_types = []
    
    centers = [(0, 3), (3, 0), (0, -3), (-3, 0)]
    for i in range(n_types):
        cx, cy = centers[i]
        dai_x.extend(np.random.randn(n_points // n_types) * 0.5 + cx)
        dai_y.extend(np.random.randn(n_points // n_types) * 0.5 + cy)
        dai_types.extend([i] * (n_points // n_types))
    
    dai_x = np.array(dai_x)
    dai_y = np.array(dai_y)
    dai_types = np.array(dai_types)
    
    # Plot baseline
    for t in range(n_types):
        mask = baseline_types == t
        axes[0].scatter(baseline_x[mask], baseline_y[mask], 
                       c=[COLORS[t]], alpha=0.6, s=20, label=f'Type {t}')
    axes[0].set_title('Baseline (Vanilla T5)')
    axes[0].set_xlabel('t-SNE 1')
    axes[0].set_ylabel('t-SNE 2')
    axes[0].legend(markerscale=2)
    
    # Plot DAI
    for t in range(n_types):
        mask = dai_types == t
        axes[1].scatter(dai_x[mask], dai_y[mask], 
                       c=[COLORS[t]], alpha=0.6, s=20, label=f'Type {t}')
    axes[1].set_title('DAI (TypeMonotonicity)')
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')
    axes[1].legend(markerscale=2)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Figure saved to {output_path}")


def generate_all_figures(results_dir: Path, output_dir: Path):
    """Generate every artifact-backed figure."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    figures = [("depth_accuracy", figure_depth_accuracy)]
    
    for name, func in figures:
        output_path = output_dir / f"{name}.pdf"
        try:
            func(results_dir, output_path)
        except Exception as e:
            print(f"Error generating {name}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Directory containing result files")
    parser.add_argument("--output-dir", type=str, default="docs/figures",
                        help="Output directory for figures")
    parser.add_argument("--figure", type=str, default=None,
                        help="Generate specific figure only")
    
    args = parser.parse_args()
    
    if not HAS_PLOTTING:
        print("Error: matplotlib and seaborn required for figure generation")
        print("Install with: pip install matplotlib seaborn")
        return
    
    if args.figure:
        # Generate specific figure
        figures = {
            "depth_accuracy": figure_depth_accuracy,
        }
        
        if args.figure in figures:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            figures[args.figure](
                Path(args.results_dir), 
                output_dir / f"{args.figure}.pdf"
            )
        else:
            print(f"Unknown figure: {args.figure}")
            print(f"Available: {list(figures.keys())}")
    else:
        generate_all_figures(Path(args.results_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()
