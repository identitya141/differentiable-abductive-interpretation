#!/usr/bin/env python3
"""Generate depth, operator, and category tables from prediction artifacts."""

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.scan_composition import extract_composition_specs, parse_scan_command


def _scan_operators(command: str) -> List[str]:
    specs = extract_composition_specs(parse_scan_command(command))
    return sorted({spec.operator for spec in specs})


def _summary(values: Sequence[float]) -> Dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "n_seeds": len(values),
    }


def load_rows(
    experiment_dir: Path,
    methods: Sequence[str],
    seeds: Sequence[int],
) -> Dict[str, Dict[int, List[Dict[str, Any]]]]:
    runs = {}
    for method in methods:
        method_runs = {}
        for seed in seeds:
            path = (
                experiment_dir
                / method
                / f"seed_{seed}"
                / f"predictions_seed{seed}.jsonl"
            )
            if not path.is_file():
                raise ValueError(f"Missing prediction artifact: {path}")
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if not rows:
                raise ValueError(f"Empty prediction artifact: {path}")
            if any(row.get("seed") != seed for row in rows):
                raise ValueError(f"Wrong or mixed seed in {path}")
            method_runs[seed] = rows
        runs[method] = method_runs
    return runs


def build_breakdown_report(
    runs: Dict[str, Dict[int, List[Dict[str, Any]]]],
    expected_seeds: Sequence[int],
) -> Dict[str, Any]:
    report = {"schema_version": 1, "seeds": list(expected_seeds), "methods": {}}
    for method, seed_rows in runs.items():
        if set(seed_rows) != set(expected_seeds):
            raise ValueError(f"Method {method} does not contain all expected seeds")

        per_group = {
            "depth": defaultdict(dict),
            "operator": defaultdict(dict),
            "category": defaultdict(dict),
        }
        for seed, rows in seed_rows.items():
            counts = {
                "depth": defaultdict(lambda: [0, 0]),
                "operator": defaultdict(lambda: [0, 0]),
                "category": defaultdict(lambda: [0, 0]),
            }
            for row_index, row in enumerate(rows):
                location = f"method={method}, seed={seed}, row={row_index}"
                correct = row.get("correct")
                if not isinstance(correct, bool):
                    raise ValueError(f"Missing correctness at {location}")
                depth = row.get("composition_depth")
                if not isinstance(depth, int) or depth < 0:
                    raise ValueError(f"Missing or invalid composition depth at {location}")
                _record(counts["depth"], str(depth), correct)

                dataset = str(row.get("dataset", "")).lower()
                if dataset == "scan":
                    input_text = row.get("input")
                    if not isinstance(input_text, str):
                        raise ValueError(f"Missing SCAN input at {location}")
                    for operator in _scan_operators(input_text):
                        _record(counts["operator"], operator, correct)
                elif dataset == "cogs":
                    category = row.get("generalization_category")
                    if not isinstance(category, str) or not category.strip():
                        raise ValueError(f"Missing COGS category at {location}")
                    _record(counts["category"], category, correct)
                else:
                    category = row.get("generalization_category")
                    if isinstance(category, str) and category.strip():
                        _record(counts["category"], category, correct)

            for dimension, dimension_counts in counts.items():
                for group, (correct_count, total_count) in dimension_counts.items():
                    per_group[dimension][group][seed] = correct_count / total_count

        method_report = {}
        for dimension, groups in per_group.items():
            if not groups:
                continue
            method_report[dimension] = {}
            for group, values_by_seed in sorted(groups.items()):
                if set(values_by_seed) != set(expected_seeds):
                    raise ValueError(
                        f"Incomplete seed coverage for {method}/{dimension}/{group}: "
                        f"{sorted(values_by_seed)}"
                    )
                values = [values_by_seed[seed] for seed in expected_seeds]
                method_report[dimension][group] = {
                    **_summary(values),
                    "per_seed": {
                        str(seed): values_by_seed[seed] for seed in expected_seeds
                    },
                }
        report["methods"][method] = method_report
    return report


def _record(counts: Dict[str, List[int]], group: str, correct: bool) -> None:
    counts[group][0] += int(correct)
    counts[group][1] += 1


def iter_summary_rows(report: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for method, method_report in report["methods"].items():
        for dimension, groups in method_report.items():
            for group, summary in groups.items():
                yield {
                    "method": method,
                    "dimension": dimension,
                    "group": group,
                    "mean": summary["mean"],
                    "std": summary["std"],
                    "n_seeds": summary["n_seeds"],
                }


def write_outputs(report: Dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "breakdowns.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    rows = list(iter_summary_rows(report))
    with (output_dir / "breakdowns.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("method", "dimension", "group", "mean", "std", "n_seeds"),
        )
        writer.writeheader()
        writer.writerows(rows)

    latex = [
        "\\begin{longtable}{lllcc}",
        "\\toprule",
        "Method & Breakdown & Group & Accuracy (\\%) & Seeds \\\\",
        "\\midrule",
        "\\endhead",
    ]
    for row in rows:
        method = _latex_escape(str(row["method"]))
        dimension = _latex_escape(str(row["dimension"]))
        group = _latex_escape(str(row["group"]))
        accuracy = f"{100 * row['mean']:.1f} $\\pm$ {100 * row['std']:.1f}"
        latex.append(
            f"{method} & {dimension} & {group} & {accuracy} & {row['n_seeds']} \\\\"
        )
    latex.extend(("\\bottomrule", "\\end{longtable}"))
    (output_dir / "breakdowns.tex").write_text(
        "\n".join(latex) + "\n", encoding="utf-8"
    )


def _latex_escape(value: str) -> str:
    return value.replace("_", "\\_").replace("%", "\\%")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    runs = load_rows(args.experiment_dir, args.methods, args.seeds)
    report = build_breakdown_report(runs, args.seeds)
    write_outputs(report, args.output_dir)
    print(f"Breakdown tables written to {args.output_dir}")


if __name__ == "__main__":
    main()