#!/usr/bin/env python3
"""Validate official SLOG categories, recursion depths, and grounded roles."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from typing import Dict

from src.data.cogs_composition import extract_cogs_composition_specs
from src.data.slog_dataset import SLOG_CATEGORIES, infer_slog_depth


def validate_slog_corpus(data_dir: Path) -> Dict:
    path = data_dir / "generalization_sets" / "gen_cogsLF.tsv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing SLOG generalization set: {path}")

    category_counts = Counter()
    depth_counts = Counter()
    operator_counts = Counter()
    errors = []
    annotated_examples = 0
    composition_count = 0
    with path.open(encoding="utf-8", newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle, delimiter="\t"), 1):
            if len(row) != 3:
                errors.append(f"line {line_number}: expected three TSV columns")
                continue
            sentence, logical_form, category = row
            category_counts[category] += 1
            if category not in SLOG_CATEGORIES:
                errors.append(f"line {line_number}: unknown category {category!r}")
                continue
            depth = infer_slog_depth(category, logical_form)
            if depth is not None:
                depth_counts[f"{category}:{depth}"] += 1
                expected = {3} if category.endswith("_3") else set(range(5, 13))
                if depth not in expected:
                    errors.append(
                        f"line {line_number}: depth {depth} invalid for {category}"
                    )
            try:
                specs = extract_cogs_composition_specs(sentence, logical_form)
            except ValueError as error:
                errors.append(f"line {line_number}: {error}")
                continue
            if specs:
                annotated_examples += 1
            composition_count += len(specs)
            operator_counts.update(spec.operator for spec in specs)

    for category in SLOG_CATEGORIES:
        if category_counts[category] != 1000:
            errors.append(
                f"category {category} has {category_counts[category]} examples, expected 1000"
            )
    unexpected = sorted(set(category_counts) - set(SLOG_CATEGORIES))
    report = {
        "dataset": "slog",
        "source": "https://github.com/bingzhilee/SLOG",
        "examples": sum(category_counts.values()),
        "category_counts": dict(sorted(category_counts.items())),
        "unexpected_categories": unexpected,
        "depth_counts": dict(sorted(depth_counts.items())),
        "annotated_examples": annotated_examples,
        "annotation_coverage": (
            annotated_examples / sum(category_counts.values())
            if category_counts
            else 0.0
        ),
        "composition_count": composition_count,
        "operator_counts": dict(sorted(operator_counts.items())),
        "errors": errors,
        "passed": not errors and set(category_counts) == set(SLOG_CATEGORIES),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = validate_slog_corpus(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not report["passed"]:
        raise SystemExit(
            f"SLOG validation failed with {len(report['errors'])} errors"
        )


if __name__ == "__main__":
    main()