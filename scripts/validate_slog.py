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


def validate_slog_corpus(data_dir: Path, minimum_coverage: float = 0.5) -> Dict:
    path = data_dir / "generalization_sets" / "gen_cogsLF.tsv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing SLOG generalization set: {path}")

    category_counts = Counter()
    depth_counts = Counter()
    operator_counts = Counter()
    errors = []
    annotated_examples = 0
    composition_count = 0
    category_annotated = Counter()
    category_relations = Counter()
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
                category_annotated[category] += 1
            composition_count += len(specs)
            category_relations[category] += len(specs)
            operator_counts.update(spec.operator for spec in specs)

    for category in SLOG_CATEGORIES:
        if category_counts[category] != 1000:
            errors.append(
                f"category {category} has {category_counts[category]} examples, expected 1000"
            )
    unexpected = sorted(set(category_counts) - set(SLOG_CATEGORIES))
    coverage_by_category = {
        category: {
            "examples": count,
            "annotated_examples": category_annotated[category],
            "annotation_coverage": category_annotated[category] / count,
            "mean_relations_per_example": category_relations[category] / count,
        }
        for category, count in sorted(category_counts.items())
    }
    categories_below_minimum = sorted(
        category for category, values in coverage_by_category.items()
        if values["annotation_coverage"] < minimum_coverage
    )
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
        "coverage_by_category": coverage_by_category,
        "categories_below_minimum_coverage": categories_below_minimum,
        "errors": errors,
        "minimum_annotation_coverage": minimum_coverage,
        "passed": (
            not errors and set(category_counts) == set(SLOG_CATEGORIES)
            and composition_count > 0
            and annotated_examples / max(1, sum(category_counts.values())) >= minimum_coverage
            and not categories_below_minimum
        ),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-coverage", type=float, default=0.5)
    args = parser.parse_args()

    report = validate_slog_corpus(args.data_dir, args.minimum_coverage)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not report["passed"]:
        raise SystemExit(
            f"SLOG validation failed with {len(report['errors'])} errors"
        )


if __name__ == "__main__":
    main()
