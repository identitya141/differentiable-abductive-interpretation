#!/usr/bin/env python3
"""Audit whether GSM8K rationales provide source-grounded composition steps."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from typing import Dict, Iterable, Mapping


EQUATION_PATTERN = re.compile(r"<<([^<>]+?)=([^<>]+?)>>")


def assess_examples(examples: Iterable[Mapping[str, str]]) -> Dict:
    total = 0
    examples_with_equations = 0
    equation_steps = 0
    source_grounded_steps = 0
    fully_source_grounded_examples = 0
    step_counts = Counter()
    malformed_answers = 0

    for example in examples:
        total += 1
        question = example.get("question", "")
        answer = example.get("answer", "")
        if not question or "####" not in answer:
            malformed_answers += 1
        equations = EQUATION_PATTERN.findall(answer)
        step_counts[len(equations)] += 1
        if equations:
            examples_with_equations += 1
        grounded = 0
        for expression, result in equations:
            equation_steps += 1
            canonical = f"{expression}={result}"
            if question.count(canonical) == 1:
                grounded += 1
                source_grounded_steps += 1
        if equations and grounded == len(equations):
            fully_source_grounded_examples += 1

    return {
        "examples": total,
        "examples_with_target_equations": examples_with_equations,
        "target_equation_steps": equation_steps,
        "source_grounded_equation_steps": source_grounded_steps,
        "fully_source_grounded_examples": fully_source_grounded_examples,
        "source_grounded_example_coverage": (
            fully_source_grounded_examples / total if total else 0.0
        ),
        "target_step_count_distribution": {
            str(key): value for key, value in sorted(step_counts.items())
        },
        "malformed_answers": malformed_answers,
    }


def _load_split(data_dir: Path, split: str) -> Iterable[Mapping[str, str]]:
    jsonl_path = data_dir / f"{split}.jsonl"
    if jsonl_path.is_file():
        with jsonl_path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    arrow_files = sorted((data_dir / split).glob("*.arrow"))
    if arrow_files:
        import pyarrow.ipc as ipc

        rows = []
        for arrow_path in arrow_files:
            with arrow_path.open("rb") as handle:
                rows.extend(ipc.open_stream(handle).read_all().to_pylist())
        return rows
    raise FileNotFoundError(
        f"No {split}.jsonl or {split}/*.arrow GSM8K data under {data_dir}"
    )


def validate_gsm8k(data_dir: Path) -> Dict:
    splits = {
        split: assess_examples(_load_split(data_dir, split))
        for split in ("train", "test")
    }
    total_examples = sum(report["examples"] for report in splits.values())
    grounded_examples = sum(
        report["fully_source_grounded_examples"] for report in splits.values()
    )
    target_equation_steps = sum(
        report["target_equation_steps"] for report in splits.values()
    )
    return {
        "dataset": "gsm8k",
        "criterion": (
            "Every reasoning equation must have an exact unique source substring "
            "before it can define encoder-span child/parent supervision."
        ),
        "splits": splits,
        "examples": total_examples,
        "fully_source_grounded_examples": grounded_examples,
        "target_equation_steps": target_equation_steps,
        "source_grounded_example_coverage": (
            grounded_examples / total_examples if total_examples else 0.0
        ),
        "publication_structure_eligible": (
            total_examples > 0 and grounded_examples == total_examples
        ),
        "decision": (
            "include"
            if total_examples > 0 and grounded_examples == total_examples
            else "exclude_target_only_reasoning_annotations"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = validate_gsm8k(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()