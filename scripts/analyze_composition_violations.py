#!/usr/bin/env python3
"""Analyze held-out abstract composition violation from prediction artifacts."""

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List, Optional


def _summary(values: List[float]) -> Dict[str, Optional[float]]:
    return {
        "count": len(values),
        "mean": mean(values) if values else None,
        "sample_std": stdev(values) if len(values) > 1 else None,
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
    }


def _point_biserial(values: List[float], correctness: List[bool]) -> Optional[float]:
    if len(values) < 2 or len(set(correctness)) < 2:
        return None
    numeric_correctness = [float(value) for value in correctness]
    values_mean = mean(values)
    correctness_mean = mean(numeric_correctness)
    numerator = sum(
        (value - values_mean) * (correct - correctness_mean)
        for value, correct in zip(values, numeric_correctness)
    )
    value_scale = math.sqrt(sum((value - values_mean) ** 2 for value in values))
    correctness_scale = math.sqrt(
        sum((correct - correctness_mean) ** 2 for correct in numeric_correctness)
    )
    return numerator / (value_scale * correctness_scale)


def analyze_rows(rows: Iterable[Dict]) -> Dict:
    rows = list(rows)
    violations = []
    correctness = []
    missing_count = 0

    for row in rows:
        value = row.get("composition_violation")
        if value is None:
            missing_count += 1
            continue
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("composition_violation values must be finite")
        if "correct" not in row:
            raise ValueError("prediction row is missing correctness")
        violations.append(value)
        correctness.append(bool(row["correct"]))

    correct_values = [
        value for value, is_correct in zip(violations, correctness) if is_correct
    ]
    incorrect_values = [
        value for value, is_correct in zip(violations, correctness) if not is_correct
    ]
    return {
        "analysis_class": "exploratory",
        "total_examples": len(rows),
        "examples_with_composition": len(violations),
        "examples_without_composition": missing_count,
        "coverage": len(violations) / len(rows) if rows else 0.0,
        "all": _summary(violations),
        "correct": _summary(correct_values),
        "incorrect": _summary(incorrect_values),
        "incorrect_minus_correct_mean": (
            mean(incorrect_values) - mean(correct_values)
            if correct_values and incorrect_values
            else None
        ),
        "point_biserial_correlation_with_correctness": _point_biserial(
            violations, correctness
        ),
    }


def load_rows(paths: Iterable[Path]) -> List[Dict]:
    rows = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = analyze_rows(load_rows(args.predictions))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
