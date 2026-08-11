#!/usr/bin/env python3
"""Compare two methods using paired seed and per-example statistics."""

import argparse
import glob
import itertools
import json
import math
import random
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

ArtifactKey = Tuple[str, str, int, str]
SeedArtifacts = Dict[int, Dict[ArtifactKey, bool]]


def expand_paths(patterns: Iterable[str]) -> List[Path]:
    paths = sorted({Path(path) for pattern in patterns for path in glob.glob(pattern)})
    if not paths:
        raise FileNotFoundError(f"No files matched: {list(patterns)}")
    return paths


def load_artifacts(paths: Sequence[Path]) -> Tuple[str, SeedArtifacts]:
    method_names = set()
    by_seed: SeedArtifacts = {}

    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        if not rows:
            raise ValueError(f"Artifact file is empty: {path}")

        seed = int(rows[0]["seed"])
        if seed in by_seed:
            raise ValueError(f"Duplicate artifact file for seed {seed}")
        method_names.update(str(row.get("method", row["experiment_name"])) for row in rows)
        if any(int(row["seed"]) != seed for row in rows):
            raise ValueError(f"Mixed seeds in artifact file: {path}")

        examples = {}
        for row in rows:
            key = (
                str(row["dataset"]),
                str(row["split"]),
                int(row["example_index"]),
                str(row["normalized_target"]),
            )
            if key in examples:
                raise ValueError(f"Duplicate example key in {path}: {key}")
            examples[key] = bool(row["correct"])
        by_seed[seed] = examples

    if len(method_names) != 1:
        raise ValueError(f"Expected one experiment name, found {sorted(method_names)}")
    return method_names.pop(), by_seed


def mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("Cannot compute a percentile of no values")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_mean_interval(
    values: Sequence[float],
    samples: int = 10000,
    confidence: float = 0.95,
    seed: int = 2026,
) -> Tuple[float, float]:
    if not values:
        raise ValueError("Bootstrap requires at least one value")
    generator = random.Random(seed)
    bootstrap_means = [
        mean([generator.choice(values) for _ in values])
        for _ in range(samples)
    ]
    tail = (1.0 - confidence) / 2.0
    return percentile(bootstrap_means, tail), percentile(bootstrap_means, 1.0 - tail)


def paired_permutation_pvalue(
    differences: Sequence[float],
    samples: int = 100000,
    seed: int = 2026,
) -> float:
    if not differences:
        raise ValueError("Permutation test requires paired differences")
    observed = abs(mean(differences))
    tolerance = 1e-15

    if len(differences) <= 20:
        extreme = 0
        total = 0
        for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
            permuted = abs(mean([sign * value for sign, value in zip(signs, differences)]))
            extreme += permuted + tolerance >= observed
            total += 1
        return extreme / total

    generator = random.Random(seed)
    extreme = 0
    for _ in range(samples):
        permuted = abs(
            mean([generator.choice((-1.0, 1.0)) * value for value in differences])
        )
        extreme += permuted + tolerance >= observed
    return (extreme + 1) / (samples + 1)


def exact_mcnemar(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> Dict[str, float]:
    if len(correct_a) != len(correct_b):
        raise ValueError("McNemar inputs must be paired")
    a_only = sum(left and not right for left, right in zip(correct_a, correct_b))
    b_only = sum(right and not left for left, right in zip(correct_a, correct_b))
    discordant = a_only + b_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail_count = min(a_only, b_only)
        tail_probability = sum(
            math.comb(discordant, index) for index in range(tail_count + 1)
        ) / (2 ** discordant)
        p_value = min(1.0, 2.0 * tail_probability)
    return {
        "a_only_correct": a_only,
        "b_only_correct": b_only,
        "discordant": discordant,
        "exact_p_value": p_value,
    }


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    count = len(p_values)
    ordered_indices = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [0.0] * count
    running_maximum = 0.0
    for rank, index in enumerate(ordered_indices):
        candidate = min(1.0, (count - rank) * p_values[index])
        running_maximum = max(running_maximum, candidate)
        adjusted[index] = running_maximum
    return adjusted


def compare(method_a: Tuple[str, SeedArtifacts], method_b: Tuple[str, SeedArtifacts]) -> Dict:
    name_a, artifacts_a = method_a
    name_b, artifacts_b = method_b
    seeds = sorted(set(artifacts_a) & set(artifacts_b))
    if len(seeds) < 2:
        raise ValueError("At least two paired seeds are required")

    per_seed = []
    differences = []
    mcnemar_by_seed = {}
    for seed in seeds:
        examples_a = artifacts_a[seed]
        examples_b = artifacts_b[seed]
        if set(examples_a) != set(examples_b):
            missing_a = len(set(examples_b) - set(examples_a))
            missing_b = len(set(examples_a) - set(examples_b))
            raise ValueError(
                f"Unpaired examples for seed {seed}: missing_a={missing_a}, missing_b={missing_b}"
            )
        keys = sorted(examples_a)
        correct_a = [examples_a[key] for key in keys]
        correct_b = [examples_b[key] for key in keys]
        accuracy_a = mean([float(value) for value in correct_a])
        accuracy_b = mean([float(value) for value in correct_b])
        difference = accuracy_a - accuracy_b
        differences.append(difference)
        per_seed.append(
            {
                "seed": seed,
                "examples": len(keys),
                "accuracy_a": accuracy_a,
                "accuracy_b": accuracy_b,
                "difference_a_minus_b": difference,
            }
        )
        mcnemar_by_seed[str(seed)] = exact_mcnemar(correct_a, correct_b)

    difference_std = statistics.stdev(differences) if len(differences) > 1 else 0.0
    effect_size = mean(differences) / difference_std if difference_std > 0 else None
    confidence_interval = bootstrap_mean_interval(differences)
    permutation_p = paired_permutation_pvalue(differences)

    benchmark_ids = {
        (key[0], key[1])
        for artifacts in artifacts_a.values()
        for key in artifacts
    }
    if len(benchmark_ids) != 1:
        raise ValueError(f"Expected one benchmark setting, found {sorted(benchmark_ids)}")
    dataset, split = benchmark_ids.pop()

    return {
        "method_a": name_a,
        "method_b": name_b,
        "benchmark": {"dataset": dataset, "split": split},
        "paired_seeds": seeds,
        "per_seed": per_seed,
        "seed_level": {
            "mean_accuracy_a": mean([row["accuracy_a"] for row in per_seed]),
            "mean_accuracy_b": mean([row["accuracy_b"] for row in per_seed]),
            "mean_difference_a_minus_b": mean(differences),
            "sample_std_difference": difference_std,
            "bootstrap_95_percent_ci": list(confidence_interval),
            "paired_permutation_p_value": permutation_p,
            "paired_cohens_dz": effect_size,
        },
        "example_level_mcnemar_by_seed": mcnemar_by_seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-a", nargs="+", required=True)
    parser.add_argument("--method-b", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    method_a = load_artifacts(expand_paths(args.method_a))
    method_b = load_artifacts(expand_paths(args.method_b))
    report = compare(method_a, method_b)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
