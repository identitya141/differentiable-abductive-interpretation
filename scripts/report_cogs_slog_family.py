#!/usr/bin/env python3
"""Bundle COGS and SLOG result artifacts without conflating their metrics."""

import argparse
import json
from pathlib import Path
from typing import Dict


def build_family_report(cogs: Dict, slog: Dict) -> Dict:
    """Represent the related benchmark family with separate dataset results."""
    return {
        "schema_version": 1,
        "benchmark_family": "COGS/SLOG",
        "reporting_rule": (
            "COGS and SLOG are related semantic-parsing benchmarks; all metrics "
            "remain dataset-specific and no pooled score is a primary endpoint."
        ),
        "datasets": {"cogs": cogs, "slog": slog},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cogs", type=Path, required=True)
    parser.add_argument("--slog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = build_family_report(
        json.loads(args.cogs.read_text()),
        json.loads(args.slog.read_text()),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()