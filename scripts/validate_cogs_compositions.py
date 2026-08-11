#!/usr/bin/env python3
"""Validate deterministic grounded COGS role annotations over local TSV data."""

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Dict

from src.data.cogs_composition import extract_cogs_composition_specs


def validate_cogs_corpus(data_dir: Path) -> Dict:
    report = {
        "dataset": "cogs",
        "splits": {},
        "total_examples": 0,
        "primitive_entries": 0,
        "eligible_examples": 0,
        "annotated_examples": 0,
        "composition_count": 0,
        "operator_counts": Counter(),
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
    report["passed"] = not report["errors"] and report["composition_count"] > 0
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = validate_cogs_corpus(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not report["passed"]:
        raise SystemExit(
            f"COGS composition validation failed with {len(report['errors'])} errors"
        )


if __name__ == "__main__":
    main()
