#!/usr/bin/env python3
"""Run matched experiment configurations over paired random seeds."""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

DEFAULT_SEEDS = (42, 123, 456, 789, 1024)


def build_commands(
    python_executable: str,
    configs: Sequence[Path],
    seeds: Sequence[int],
) -> List[List[str]]:
    return [
        [
            python_executable,
            "scripts/train.py",
            "--config",
            str(config),
            "--seed",
            str(seed),
        ]
        for config in configs
        for seed in seeds
    ]


def run_matrix(commands: Sequence[Sequence[str]], dry_run: bool) -> List[Dict]:
    records = []
    for command in commands:
        started_at = datetime.now(timezone.utc).isoformat()
        print(" ".join(command), flush=True)
        if dry_run:
            return_code = None
            status = "dry_run"
        else:
            completed = subprocess.run(list(command), check=False)
            return_code = completed.returncode
            status = "completed" if return_code == 0 else "failed"
        records.append(
            {
                "command": list(command),
                "started_at": started_at,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "return_code": return_code,
                "status": status,
            }
        )
        if return_code not in (None, 0):
            break
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("experiments/experiment_matrix_manifest.json"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    missing_configs = [str(config) for config in args.configs if not config.is_file()]
    if missing_configs:
        raise FileNotFoundError(f"Missing experiment configs: {missing_configs}")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must be unique for paired comparisons")

    commands = build_commands(args.python, args.configs, args.seeds)
    records = run_matrix(commands, args.dry_run)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "configs": [str(config) for config in args.configs],
        "seeds": args.seeds,
        "dry_run": args.dry_run,
        "runs": records,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    failed = [record for record in records if record["status"] == "failed"]
    if failed:
        raise SystemExit(failed[0]["return_code"])


if __name__ == "__main__":
    main()
