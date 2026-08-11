#!/usr/bin/env python3
"""Combine pairwise reports and apply Holm correction across the full family."""

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Sequence

from compare_experiments import holm_adjust


def summarize_reports(reports: Sequence[Dict]) -> Dict:
    if not reports:
        raise ValueError("At least one comparison report is required")

    method_names = {str(report["method_a"]) for report in reports}
    if len(method_names) != 1:
        raise ValueError(f"Expected one focal method, found {sorted(method_names)}")

    comparison_keys = [
        (
            str(report.get("benchmark", {}).get("dataset", "")),
            str(report.get("benchmark", {}).get("split", "")),
            str(report["method_b"]),
        )
        for report in reports
    ]
    if len(set(comparison_keys)) != len(comparison_keys):
        raise ValueError("Comparison reports contain duplicate control entries for a benchmark")

    p_values = [
        float(report["seed_level"]["paired_permutation_p_value"])
        for report in reports
    ]
    adjusted_p_values = holm_adjust(p_values)
    comparisons: List[Dict] = []
    for report, adjusted_p_value in zip(reports, adjusted_p_values):
        comparison = deepcopy(report)
        comparison["seed_level"]["holm_adjusted_p_value"] = adjusted_p_value
        comparison["all_paired_seeds_improved"] = all(
            float(row["difference_a_minus_b"]) > 0.0
            for row in comparison["per_seed"]
        )
        comparisons.append(comparison)

    return {
        "focal_method": method_names.pop(),
        "family_size": len(comparisons),
        "correction": "Holm",
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reports = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in args.reports
    ]
    summary = summarize_reports(reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
