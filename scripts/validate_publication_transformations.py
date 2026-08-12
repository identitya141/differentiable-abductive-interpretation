#!/usr/bin/env python3
"""Instantiate every publication dataset with the real tokenizer and contract."""

import argparse
import json
from collections import Counter
from pathlib import Path

from transformers import T5Tokenizer

from src.data.scan_dataset import SCANDataset
from src.data.cogs_dataset import COGSDataset
from src.data.slog_dataset import SLOGDataset
from src.data.cfq_dataset import CFQDataset
from src.utils.benchmark_contract import get_benchmark_contract
from src.utils.tokenizer_utils import extend_tokenizer_for_dataset


def summarize(dataset, batch_size):
    relation_counts = [len(row.get("composition_specs", [])) for row in dataset._tokenized_examples]
    operators = Counter(
        spec.operator
        for row in dataset._tokenized_examples
        for spec in row.get("composition_specs", [])
    )
    categories = Counter()
    category_annotated = Counter()
    category_relations = Counter()
    for example, count in zip(dataset.examples, relation_counts):
        category = example.generalization_category or "uncategorized"
        categories[category] += 1
        category_annotated[category] += int(count > 0)
        category_relations[category] += count
    anchors = anchors_with_negative = negative_total = 0
    for start in range(0, len(dataset._tokenized_examples), batch_size):
        batch_operators = [
            spec.operator
            for row in dataset._tokenized_examples[start:start + batch_size]
            for spec in row.get("composition_specs", [])
        ]
        for operator in batch_operators:
            negative_count = sum(other != operator for other in batch_operators)
            anchors += 1
            anchors_with_negative += int(negative_count > 0)
            negative_total += negative_count
    return {
        "examples": len(dataset),
        "annotated_examples": sum(count > 0 for count in relation_counts),
        "zero_relation_examples": sum(count == 0 for count in relation_counts),
        "mean_relations_per_example": sum(relation_counts) / max(1, len(relation_counts)),
        "median_relations_per_example": sorted(relation_counts)[len(relation_counts) // 2] if relation_counts else 0,
        "operator_distribution": dict(sorted(operators.items())),
        "structural_contrastive_coverage": {
            "anchor_count": anchors,
            "anchors_with_negative": anchors_with_negative,
            "zero_negative_fraction": 1.0 - anchors_with_negative / anchors if anchors else 0.0,
            "mean_negative_count": negative_total / anchors_with_negative if anchors_with_negative else 0.0,
        },
        "coverage_by_generalization_category": {
            category: {
                "examples": count,
                "annotated_examples": category_annotated[category],
                "annotation_coverage": category_annotated[category] / count,
                "mean_relations_per_example": category_relations[category] / count,
            }
            for category, count in sorted(categories.items())
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model", default="t5-small")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"schema_version": 1, "model": args.model, "benchmarks": {}}
    benchmark_specs = [
        ("scan", "length", SCANDataset, {"scan_split": "length"}, args.data_root / "scan", ("train", "test")),
        ("cogs", "generalization", COGSDataset, {}, args.data_root / "cogs/COGS-main/data", ("train", "dev", "iid_test", "test")),
        ("slog", "structural_generalization", SLOGDataset, {}, args.data_root / "slog", ("train", "dev", "iid_test", "test")),
    ]
    for dataset_name, benchmark_split, cls, extra, data_dir, splits in benchmark_specs:
        contract = get_benchmark_contract(dataset_name, benchmark_split)
        tokenizer = T5Tokenizer.from_pretrained(args.model)
        tokenizer, _ = extend_tokenizer_for_dataset(tokenizer, dataset_name, verbose=False)
        key = f"{dataset_name}/{benchmark_split}"
        report["benchmarks"][key] = {}
        for split in splits:
            dataset = cls(
                tokenizer=tokenizer, split=split, data_dir=str(data_dir),
                max_source_length=contract.max_source_length,
                max_target_length=contract.max_target_length, **extra,
            )
            report["benchmarks"][key][split] = summarize(dataset, contract.train_batch_size)
    for cfq_split in ("mcd1", "mcd2", "mcd3"):
        contract = get_benchmark_contract("cfq", cfq_split)
        tokenizer = T5Tokenizer.from_pretrained(args.model)
        tokenizer, _ = extend_tokenizer_for_dataset(tokenizer, "cfq", verbose=False)
        key = f"cfq/{cfq_split}"
        report["benchmarks"][key] = {}
        for split in ("train", "test"):
            dataset = CFQDataset(
                tokenizer=tokenizer, split=split, cfq_split=cfq_split,
                data_dir=str(args.data_root / "cfq"),
                max_source_length=contract.max_source_length,
                max_target_length=contract.max_target_length,
            )
            report["benchmarks"][key][split] = summarize(dataset, contract.train_batch_size)
    report["passed"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
