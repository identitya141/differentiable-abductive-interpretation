#!/usr/bin/env python3
"""Validate deterministic composition coverage over local SCAN files."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable

from src.data.scan_composition import extract_composition_specs, parse_scan_command


def iter_commands(paths: Iterable[Path]):
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                if not line.startswith("IN:") or "OUT:" not in line:
                    raise ValueError(f"Malformed SCAN row at {path}:{line_number}")
                command = line.split("OUT:", 1)[0].removeprefix("IN:").strip()
                yield path, line_number, command


def validate_scan_compositions(data_dir: Path, split: str) -> Dict[str, object]:
    paths = sorted((data_dir / split).glob(f"tasks_*_{split}.txt"))
    if not paths:
        raise FileNotFoundError(f"No SCAN files found under {data_dir / split}")

    operator_counts: Counter[str] = Counter()
    total_examples = 0
    compositional_examples = 0
    total_compositions = 0

    for path, line_number, command in iter_commands(paths):
        try:
            specs = extract_composition_specs(parse_scan_command(command))
        except Exception as error:
            raise ValueError(
                f"Could not parse SCAN command at {path}:{line_number}: {command!r}"
            ) from error
        total_examples += 1
        if specs:
            compositional_examples += 1
        total_compositions += len(specs)
        operator_counts.update(spec.operator for spec in specs)

    return {
        "data_dir": str(data_dir),
        "split": split,
        "files": [str(path) for path in paths],
        "total_examples": total_examples,
        "compositional_examples": compositional_examples,
        "primitive_only_examples": total_examples - compositional_examples,
        "composition_coverage": compositional_examples / total_examples,
        "total_compositions": total_compositions,
        "operator_counts": dict(sorted(operator_counts.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/scan"))
    parser.add_argument("--split", default="length")
    parser.add_argument("--minimum-coverage", type=float, default=0.99)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = validate_scan_compositions(args.data_dir, args.split)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")

    if report["composition_coverage"] < args.minimum_coverage:
        raise SystemExit(
            "Composition coverage "
            f"{report['composition_coverage']:.2%} is below "
            f"{args.minimum_coverage:.2%}"
        )


if __name__ == "__main__":
    main()
