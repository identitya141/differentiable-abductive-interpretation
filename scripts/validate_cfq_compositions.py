#!/usr/bin/env python3
"""Validate deterministic grounded CFQ relations over local MCD splits."""

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Dict

from datasets import load_from_disk

from src.data.cfq_composition import extract_cfq_composition_specs


def validate_cfq_corpus(data_dir: Path) -> Dict:
    report = {
        "dataset": "cfq",
        "mcd_splits": {},
        "total_examples": 0,
        "annotated_examples": 0,
        "composition_count": 0,
        "operator_counts": Counter(),
        "errors": [],
    }
    for mcd_path in sorted(data_dir.glob("mcd[123]")):
        dataset = load_from_disk(str(mcd_path))
        mcd_report = {}
        for split in ("train", "test"):
            split_total = len(dataset[split])
            split_annotated = 0
            split_compositions = 0
            for index, item in enumerate(dataset[split]):
                try:
                    specs = extract_cfq_composition_specs(
                        item["question"], item["query"]
                    )
                except ValueError as exc:
                    report["errors"].append(
                        f"{mcd_path.name}/{split}:{index}: {exc}"
                    )
                    continue
                if specs:
                    split_annotated += 1
                split_compositions += len(specs)
                report["operator_counts"].update(
                    spec.operator for spec in specs
                )
            mcd_report[split] = {
                "examples": split_total,
                "annotated_examples": split_annotated,
                "annotation_coverage": (
                    split_annotated / split_total if split_total else 0.0
                ),
                "composition_count": split_compositions,
            }
            report["total_examples"] += split_total
            report["annotated_examples"] += split_annotated
            report["composition_count"] += split_compositions
        report["mcd_splits"][mcd_path.name] = mcd_report

    total = report["total_examples"]
    report["annotation_coverage"] = (
        report["annotated_examples"] / total if total else 0.0
    )
    report["operator_counts"] = dict(sorted(report["operator_counts"].items()))
    report["passed"] = (
        bool(report["mcd_splits"])
        and not report["errors"]
        and report["composition_count"] > 0
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = validate_cfq_corpus(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not report["passed"]:
        raise SystemExit(
            f"CFQ composition validation failed with {len(report['errors'])} errors"
        )


if __name__ == "__main__":
    main()