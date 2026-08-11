#!/usr/bin/env python3
"""Validate grounded relation chains in official CLUTRR CSV files."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from typing import Dict

from src.data.clutrr_composition import (
    extract_clutrr_composition_specs,
    extract_clutrr_query_path,
)


REQUIRED_FIELDS = (
    "story",
    "text_query",
    "text_target",
    "story_edges",
    "edge_types",
    "query_edge",
    "genders",
    "task_name",
)


def validate_clutrr_corpus(data_dir: Path) -> Dict:
    files = sorted(data_dir.rglob("*.csv"))
    split_counts = Counter()
    hop_counts = Counter()
    operator_counts = Counter()
    errors = []
    examples = 0
    annotated_examples = 0
    composition_count = 0

    for path in files:
        split = next(
            (name for name in ("train", "test") if path.name.endswith(f"_{name}.csv")),
            None,
        )
        if split is None:
            continue
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            missing_columns = [
                field for field in REQUIRED_FIELDS if field not in (reader.fieldnames or [])
            ]
            if missing_columns:
                errors.append(
                    f"{path}: missing columns {', '.join(missing_columns)}"
                )
                continue
            for line_number, row in enumerate(reader, 2):
                examples += 1
                split_counts[split] += 1
                missing_values = [field for field in REQUIRED_FIELDS if not row[field]]
                if missing_values:
                    errors.append(
                        f"{path}:{line_number}: empty fields {', '.join(missing_values)}"
                    )
                    continue
                try:
                    hop = int(row["task_name"].rsplit(".", 1)[1])
                except (IndexError, ValueError):
                    errors.append(
                        f"{path}:{line_number}: invalid task_name {row['task_name']!r}"
                    )
                    continue
                hop_counts[f"{split}:k={hop}"] += 1
                try:
                    path_nodes, _ = extract_clutrr_query_path(
                        row["story_edges"],
                        row["edge_types"],
                        row["query_edge"],
                        row["genders"],
                    )
                    if len(path_nodes) - 1 != hop:
                        raise ValueError(
                            f"task_name hop {hop} differs from query path "
                            f"hop {len(path_nodes) - 1}"
                        )
                    specs = extract_clutrr_composition_specs(
                        row["story"],
                        row["story_edges"],
                        row["edge_types"],
                        row["query_edge"],
                        row["genders"],
                    )
                except ValueError as error:
                    errors.append(f"{path}:{line_number}: {error}")
                    continue
                if specs:
                    annotated_examples += 1
                composition_count += len(specs)
                operator_counts.update(spec.operator for spec in specs)

    used_files = [
        str(path.relative_to(data_dir))
        for path in files
        if path.name.endswith(("_train.csv", "_test.csv"))
    ]
    if not used_files:
        errors.append(f"no official *_train.csv or *_test.csv files under {data_dir}")
    return {
        "dataset": "clutrr",
        "source": "https://github.com/facebookresearch/clutrr",
        "official_csv_files": used_files,
        "examples": examples,
        "split_counts": dict(sorted(split_counts.items())),
        "hop_counts": dict(sorted(hop_counts.items())),
        "annotated_examples": annotated_examples,
        "annotation_coverage": annotated_examples / examples if examples else 0.0,
        "composition_count": composition_count,
        "operator_counts": dict(sorted(operator_counts.items())),
        "errors": errors,
        "passed": bool(used_files) and examples > 0 and not errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = validate_clutrr_corpus(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not report["passed"]:
        raise SystemExit(
            f"CLUTRR validation failed with {len(report['errors'])} errors"
        )


if __name__ == "__main__":
    main()