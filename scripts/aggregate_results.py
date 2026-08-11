#!/usr/bin/env python3
"""
Aggregate Results Across Seeds

Computes mean and standard deviation across multiple random seeds.

Usage:
    python scripts/aggregate_results.py --input-pattern "results/dai_seed*.json" --output results/dai_aggregated.json
"""

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List
import re

import numpy as np


def load_results(pattern: str) -> List[Dict]:
    """Load all result files matching pattern."""
    files = glob.glob(pattern)
    results = []
    
    for file_path in sorted(files):
        with open(file_path) as f:
            data = json.load(f)
            # Extract seed from filename
            match = re.search(r'seed(\d+)', file_path)
            if match:
                data['seed'] = int(match.group(1))
            results.append(data)
    
    print(f"Loaded {len(results)} result files")
    return results


def aggregate_metric(values: List[float]) -> Dict:
    """Compute summary statistics using sample standard deviation."""
    if not values:
        return {"mean": 0.0, "std": 0.0, "n": 0}
    
    values = [v for v in values if v is not None]
    if not values:
        return {"mean": 0.0, "std": 0.0, "n": 0}
    
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "n": len(values),
    }


def aggregate_current_results(results: List[Dict]) -> Dict:
    """Aggregate the result schema emitted by scripts/train.py."""
    if not results:
        return {}

    seeds = [result.get("seed") for result in results]
    if any(seed is None for seed in seeds):
        raise ValueError("Every result must have a seed")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Duplicate seeds found: {seeds}")

    aggregated = {
        "schema_version": 1,
        "seeds": sorted(seeds),
        "n_seeds": len(seeds),
    }

    for section_name in ("iid_evaluation", "final_evaluation"):
        section = _aggregate_evaluation_section(results, section_name)
        if section:
            aggregated[section_name] = section

    efficiency_metrics = (
        "optimizer_updates",
        "examples_seen",
        "training_wall_clock_seconds",
        "accelerator_hours",
        "peak_cuda_memory_bytes",
    )
    efficiency = {}
    for metric_name in efficiency_metrics:
        values = [
            result.get(metric_name)
            for result in results
            if isinstance(result.get(metric_name), (int, float))
        ]
        if values:
            efficiency[metric_name] = aggregate_metric(values)
    if efficiency:
        aggregated["efficiency"] = efficiency

    parameter_sections = [result.get("model_parameters") for result in results]
    if all(isinstance(section, dict) for section in parameter_sections):
        parameters = {}
        for metric_name in ("total", "trainable"):
            values = [section.get(metric_name) for section in parameter_sections]
            if all(isinstance(value, int) for value in values):
                if len(set(values)) != 1:
                    raise ValueError(
                        f"Parameter count {metric_name} differs across seeds: {values}"
                    )
                parameters[metric_name] = values[0]
        if parameters:
            aggregated["model_parameters"] = parameters

    return aggregated


def _aggregate_evaluation_section(results: List[Dict], section_name: str) -> Dict:
    sections = [
        result[section_name]
        for result in results
        if isinstance(result.get(section_name), dict)
    ]
    if not sections:
        return {}

    aggregated = {}
    scalar_names = sorted({
        key
        for section in sections
        for key, value in section.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    })
    for metric_name in scalar_names:
        values = [
            section[metric_name]
            for section in sections
            if isinstance(section.get(metric_name), (int, float))
        ]
        aggregated[metric_name] = aggregate_metric(values)

    for group_name in ("accuracy_by_depth", "accuracy_by_category"):
        groups = [
            section[group_name]
            for section in sections
            if isinstance(section.get(group_name), dict)
        ]
        group_keys = sorted({str(key) for group in groups for key in group})
        if group_keys:
            aggregated[group_name] = {}
        for group_key in group_keys:
            values = [
                group[group_key]
                for group in groups
                if group_key in group
                and isinstance(group[group_key], (int, float))
            ]
            if values:
                aggregated[group_name][group_key] = aggregate_metric(values)
    return aggregated


def aggregate_results(results: List[Dict]) -> Dict:
    """Aggregate results across seeds."""
    if not results:
        return {}
    
    aggregated = {
        "seeds": [r.get("seed") for r in results],
        "n_seeds": len(results),
        "datasets": {},
    }
    
    # Get all datasets
    all_datasets = set()
    for r in results:
        if "datasets" in r:
            all_datasets.update(r["datasets"].keys())
    
    # Aggregate each dataset
    for dataset in all_datasets:
        dataset_results = {}
        
        # Collect metrics from all seeds
        metrics_by_seed = []
        for r in results:
            if "datasets" in r and dataset in r["datasets"]:
                metrics_by_seed.append(r["datasets"][dataset])
        
        if not metrics_by_seed:
            continue
        
        # Aggregate test metrics
        if "test" in metrics_by_seed[0]:
            test_metrics = [m["test"] for m in metrics_by_seed if "test" in m]
            dataset_results["test"] = {}
            
            # Get all metric names
            metric_names = set()
            for tm in test_metrics:
                if isinstance(tm, dict):
                    metric_names.update(tm.keys())
            
            for metric_name in metric_names:
                values = []
                for tm in test_metrics:
                    if isinstance(tm, dict) and metric_name in tm:
                        val = tm[metric_name]
                        if isinstance(val, (int, float)):
                            values.append(val)
                
                if values:
                    dataset_results["test"][metric_name] = aggregate_metric(values)
        
        # Aggregate OOD splits
        if "ood_splits" in metrics_by_seed[0]:
            ood_splits = [m["ood_splits"] for m in metrics_by_seed if "ood_splits" in m]
            dataset_results["ood_splits"] = {}
            
            # Get all split names
            split_names = set()
            for os in ood_splits:
                if isinstance(os, dict):
                    split_names.update(os.keys())
            
            for split_name in split_names:
                split_metrics = {}
                
                # Get metrics for this split
                split_results = []
                for os in ood_splits:
                    if isinstance(os, dict) and split_name in os:
                        split_results.append(os[split_name])
                
                if not split_results:
                    continue
                
                # Aggregate each metric
                metric_names = set()
                for sr in split_results:
                    if isinstance(sr, dict):
                        metric_names.update(sr.keys())
                
                for metric_name in metric_names:
                    values = []
                    for sr in split_results:
                        if isinstance(sr, dict) and metric_name in sr:
                            val = sr[metric_name]
                            if isinstance(val, (int, float)):
                                values.append(val)
                    
                    if values:
                        split_metrics[metric_name] = aggregate_metric(values)
                
                dataset_results["ood_splits"][split_name] = split_metrics
        
        aggregated["datasets"][dataset] = dataset_results
    
    return aggregated


def format_summary(aggregated: Dict) -> str:
    """Format aggregated results as a summary string."""
    lines = []
    lines.append("="*60)
    lines.append("AGGREGATED RESULTS SUMMARY")
    lines.append(f"Seeds: {aggregated.get('seeds', [])}")
    lines.append("="*60)
    
    for dataset, results in aggregated.get("datasets", {}).items():
        lines.append(f"\n{dataset.upper()}")
        lines.append("-"*40)
        
        if "test" in results:
            for metric, values in results["test"].items():
                if isinstance(values, dict) and "mean" in values:
                    lines.append(f"  {metric}: {values['mean']:.4f} ± {values['std']:.4f}")
        
        if "ood_splits" in results:
            lines.append("  OOD Splits:")
            for split, metrics in results["ood_splits"].items():
                if "exact_match" in metrics:
                    em = metrics["exact_match"]
                    lines.append(f"    {split}: {em['mean']:.4f} ± {em['std']:.4f}")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Aggregate results across seeds")
    parser.add_argument("--input-pattern", type=str, required=True,
                        help="Glob pattern for input files")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for aggregated results")
    
    args = parser.parse_args()
    
    # Load and aggregate
    results = load_results(args.input_pattern)
    aggregated = (
        aggregate_current_results(results)
        if results and "final_evaluation" in results[0]
        else aggregate_results(results)
    )
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(aggregated, f, indent=2)
    
    print(f"\nAggregated results saved to {output_path}")
    print(format_summary(aggregated))


if __name__ == "__main__":
    main()
