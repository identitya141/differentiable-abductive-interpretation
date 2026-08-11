#!/usr/bin/env python3
"""Expand and query the canonical six-benchmark publication matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from src.models.baselines import BASELINE_REGISTRY


STRUCTURAL_OVERRIDES = {
    "random_structure": (
        "data.composition_structure_mode=random"
    ),
    "shuffled_structure": (
        "data.composition_structure_mode=shuffled"
    ),
    "simple_consistency": (
        "model.cross_layer_consistency=false,model.consistency_weight=0.0,"
        "abstraction.concretization_weight=0.0,abstraction.composition_weight=1.0,"
        "abstraction.consistency_weight=0.0,abstraction.entropy_regularization=0.0,"
        "abstraction.contrastive_weight=0.0,abstraction.structural_contrastive_weight=0.0"
    ),
}


def load_manifest(path: Path) -> Dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2:
        raise ValueError("Unified matrix requires schema_version=2")
    if manifest.get("baseline_registry") != "all":
        raise ValueError("baseline_registry must be 'all'")
    seeds = manifest.get("seeds")
    required_seeds = [42, 123, 456, 789, 1024, 2027, 4099, 7919, 104729, 130363]
    if seeds != required_seeds or len(set(seeds)) != len(required_seeds):
        raise ValueError("The publication protocol requires the ten paired seeds")
    benchmarks = manifest.get("benchmarks")
    if not isinstance(benchmarks, list) or len(benchmarks) != 6:
        raise ValueError("The unified matrix requires exactly six benchmark settings")
    identities = {(row["dataset"], row["split"]) for row in benchmarks}
    if len(identities) != 6:
        raise ValueError("Benchmark dataset/split identities must be unique")
    return manifest


def expand_configurations(manifest: Dict[str, Any]) -> List[Dict[str, str]]:
    configurations: List[Dict[str, str]] = []
    proposed = str(manifest["proposed_method"])
    for benchmark in manifest["benchmarks"]:
        for method, spec in BASELINE_REGISTRY.items():
            runner = spec.runner
            if runner == "dai_control":
                config = benchmark["grounded_config"]
                override = STRUCTURAL_OVERRIDES[method]
            else:
                config = f"configs/baselines/{spec.config_name}"
                override = ""
            configurations.append({
                "dataset": benchmark["dataset"],
                "split": benchmark["split"],
                "method": method,
                "runner": runner,
                "config": config,
                "override": override,
                "data_subdir": benchmark["data_subdir"],
            })
        configurations.append({
            "dataset": benchmark["dataset"],
            "split": benchmark["split"],
            "method": proposed,
            "runner": "proposed",
            "config": benchmark["full_config"],
            "override": "",
            "data_subdir": benchmark["data_subdir"],
        })
    return configurations


def expand_runs(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expand configurations to one independently retryable row per seed."""
    return [
        {**configuration, "seed": seed}
        for configuration in expand_configurations(manifest)
        for seed in manifest["seeds"]
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--count", action="store_true")
    group.add_argument("--row", type=int)
    group.add_argument("--methods", action="store_true")
    group.add_argument("--benchmarks", action="store_true")
    group.add_argument("--proposed", action="store_true")
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    runs = expand_runs(manifest)
    if args.count:
        print(len(runs))
    elif args.row is not None:
        if not 0 <= args.row < len(runs):
            raise SystemExit(f"Row must be in 0..{len(runs) - 1}")
        for key in ("dataset", "split", "method", "runner", "config", "override", "data_subdir"):
            print(runs[args.row][key])
        print(runs[args.row]["seed"])
    elif args.methods:
        print(" ".join([*BASELINE_REGISTRY, manifest["proposed_method"]]))
    elif args.benchmarks:
        for row in manifest["benchmarks"]:
            print(f"{row['dataset']}/{row['split']}")
    else:
        print(manifest["proposed_method"])


if __name__ == "__main__":
    main()
