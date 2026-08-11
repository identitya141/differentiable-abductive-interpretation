#!/usr/bin/env python3
"""Generate exploratory failure summaries from validated prediction artifacts."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Dict, Iterable, List


def categorize_error(prediction: str, target: str) -> str:
    """Assign a deterministic sequence-level error category."""
    prediction_tokens = prediction.split()
    target_tokens = target.split()
    if len(prediction_tokens) < len(target_tokens):
        return "truncation"
    if len(prediction_tokens) > len(target_tokens):
        return "over_generation"
    if Counter(prediction_tokens) == Counter(target_tokens):
        return "order_error"
    if not set(prediction_tokens).intersection(target_tokens):
        return "completely_wrong"
    return "partial_error"


def _group_summary(counts: Dict[str, List[int]]) -> Dict[str, Dict[str, float]]:
    return {
        key: {
            "examples": values[0],
            "failures": values[1],
            "failure_rate": values[1] / values[0],
        }
        for key, values in sorted(counts.items())
    }


def analyze_prediction_files(paths: Iterable[Path], max_samples: int = 25) -> Dict:
    """Summarize failures by method, seed, depth, and generalization category."""
    rows_by_method = defaultdict(list)
    for path in sorted(paths):
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        if not rows:
            raise ValueError(f"Prediction artifact is empty: {path}")
        for row in rows:
            rows_by_method[str(row["experiment_name"])].append(row)

    if not rows_by_method:
        raise ValueError("No prediction artifacts were provided")

    methods = {}
    for method, rows in sorted(rows_by_method.items()):
        error_types = Counter()
        by_depth = defaultdict(lambda: [0, 0])
        by_category = defaultdict(lambda: [0, 0])
        samples = []
        failures = 0
        seeds = set()

        for row in rows:
            correct = bool(row["correct"])
            seeds.add(int(row["seed"]))
            depth = str(row.get("composition_depth", "unknown"))
            category = str(row.get("generalization_category") or "uncategorized")
            by_depth[depth][0] += 1
            by_category[category][0] += 1
            if correct:
                continue

            failures += 1
            by_depth[depth][1] += 1
            by_category[category][1] += 1
            error_type = categorize_error(
                str(row["normalized_prediction"]),
                str(row["normalized_target"]),
            )
            error_types[error_type] += 1
            if len(samples) < max_samples:
                samples.append(
                    {
                        "seed": int(row["seed"]),
                        "example_index": int(row["example_index"]),
                        "input": str(row.get("input", "")),
                        "prediction": str(row["normalized_prediction"]),
                        "target": str(row["normalized_target"]),
                        "composition_depth": row.get("composition_depth"),
                        "generalization_category": row.get("generalization_category"),
                        "error_type": error_type,
                    }
                )

        methods[method] = {
            "seeds": sorted(seeds),
            "examples": len(rows),
            "failures": failures,
            "failure_rate": failures / len(rows),
            "error_types": dict(sorted(error_types.items())),
            "by_composition_depth": _group_summary(by_depth),
            "by_generalization_category": _group_summary(by_category),
            "failure_samples": samples,
        }

    return {
        "schema_version": 1,
        "analysis_status": "exploratory",
        "methods": methods,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=25)
    args = parser.parse_args()
    if args.max_samples < 0:
        parser.error("--max-samples must be non-negative")

    report = analyze_prediction_files(args.predictions, args.max_samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Failure analysis written to {args.output}")


if __name__ == "__main__":
    main()