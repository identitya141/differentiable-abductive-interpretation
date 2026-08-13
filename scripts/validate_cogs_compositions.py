#!/usr/bin/env python3
"""Validate deterministic grounded COGS role annotations over local TSV data."""

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Dict

from src.data.cogs_composition import extract_cogs_composition_specs


def validate_cogs_corpus(data_dir: Path, minimum_coverage: float = 0.5) -> Dict:
    report = {
        "dataset": "cogs",
        "splits": {},
        "total_examples": 0,
        "primitive_entries": 0,
        "eligible_examples": 0,
        "annotated_examples": 0,
        "composition_count": 0,
        "operator_counts": Counter(),
        "coverage_by_category": {},
        "errors": [],
    }
    for split, filename in (
        ("train", "train.tsv"),
        ("dev", "dev.tsv"),
        ("gen", "gen.tsv"),
    ):
        path = data_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing COGS split: {path}")
        split_total = 0
        split_primitives = 0
        split_annotated = 0
        split_compositions = 0
        category_totals = Counter()
        category_annotated = Counter()
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    report["errors"].append(
                        f"{filename}:{line_number}: expected sentence and logical form"
                    )
                    continue
                split_total += 1
                category = parts[2] if len(parts) >= 3 else None
                category_key = category or "uncategorized"
                category_totals[category_key] += 1
                if category == "primitive":
                    split_primitives += 1
                    continue
                try:
                    specs = extract_cogs_composition_specs(parts[0], parts[1])
                except ValueError as exc:
                    report["errors"].append(
                        f"{filename}:{line_number}: {exc}"
                    )
                    continue
                if specs:
                    split_annotated += 1
                    category_annotated[category_key] += 1
                split_compositions += len(specs)
                report["operator_counts"].update(
                    spec.operator for spec in specs
                )
        report["splits"][split] = {
            "examples": split_total,
            "primitive_entries": split_primitives,
            "eligible_examples": split_total - split_primitives,
            "annotated_examples": split_annotated,
            "annotation_coverage": (
                split_annotated / (split_total - split_primitives)
                if split_total > split_primitives
                else 0.0
            ),
            "composition_count": split_compositions,
            "coverage_by_category": {
                category: {
                    "examples": total,
                    "annotated_examples": category_annotated[category],
                    "annotation_coverage": category_annotated[category] / total,
                }
                for category, total in sorted(category_totals.items())
            },
        }
        report["total_examples"] += split_total
        report["primitive_entries"] += split_primitives
        report["eligible_examples"] += split_total - split_primitives
        report["annotated_examples"] += split_annotated
        report["composition_count"] += split_compositions

    total = report["eligible_examples"]
    report["annotation_coverage"] = (
        report["annotated_examples"] / total if total else 0.0
    )
    report["operator_counts"] = dict(sorted(report["operator_counts"].items()))
    combined_totals = Counter()
    combined_annotated = Counter()
    for split_report in report["splits"].values():
        for category, values in split_report["coverage_by_category"].items():
            combined_totals[category] += values["examples"]
            combined_annotated[category] += values["annotated_examples"]
    report["coverage_by_category"] = {
        category: {
            "examples": total,
            "annotated_examples": combined_annotated[category],
            "annotation_coverage": combined_annotated[category] / total,
        }
        for category, total in sorted(combined_totals.items())
    }
    report["minimum_annotation_coverage"] = minimum_coverage
    report["categories_below_minimum_coverage"] = sorted(
        category for category, values in report["coverage_by_category"].items()
        if category != "primitive"
        and values["annotation_coverage"] < minimum_coverage
    )
    report["passed"] = (
        not report["errors"] and report["composition_count"] > 0
        and report["annotation_coverage"] >= minimum_coverage
        and not report["categories_below_minimum_coverage"]
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-coverage", type=float, default=0.5)
    args = parser.parse_args()

    report = validate_cogs_corpus(args.data_dir, args.minimum_coverage)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not report["passed"]:
        raise SystemExit(
            f"COGS composition validation failed with {len(report['errors'])} errors"
        )


if __name__ == "__main__":
    main()
