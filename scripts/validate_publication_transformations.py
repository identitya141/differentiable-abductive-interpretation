#!/usr/bin/env python3
"""Instantiate every publication dataset with the real tokenizer and contract."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer, T5Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.scan_dataset import SCANDataset
from src.data.cogs_dataset import COGSDataset
from src.data.slog_dataset import SLOGDataset
from src.data.cfq_dataset import CFQDataset
from src.utils.benchmark_contract import get_benchmark_contract
from src.utils.tokenizer_utils import extend_tokenizer_for_dataset
from src.models.baselines import BASELINE_REGISTRY, linearize_source_only_tree
from src.models.baselines.baseline_models import TinyLlamaBaseline
from src.data.scan_composition import linearize_scan_command


def _reasoning_and_answer(target):
    if "####" in target:
        reasoning, answer = target.rsplit("####", 1)
        return reasoning.strip(), answer.strip()
    tokens = target.split()
    return " ; ".join(f"step {i}: {token}" for i, token in enumerate(tokens, 1)) or "direct", target


def _transformed_text(method, dataset_name, source, target):
    if method == "tree_linearized_t5":
        source = linearize_scan_command(source) if dataset_name == "scan" else linearize_source_only_tree(source)
    elif method == "cot":
        reasoning, answer = _reasoning_and_answer(target)
        source = "Let's think step by step. " + source
        target = f"{reasoning} Therefore, the answer is: {answer}"
    elif method == "scratchpad":
        reasoning, answer = _reasoning_and_answer(target)
        source = f"{source} <scratch>"
        target = f"{reasoning} </scratch> {answer}"
    elif method == "tinyllama_lora":
        instruction = TinyLlamaBaseline.DATASET_INSTRUCTIONS[dataset_name]
        source = TinyLlamaBaseline.INSTRUCTION_TEMPLATE.format(
            instruction=instruction, input=source
        )
    return source, target


def summarize_baseline_transformations(datasets, dataset_name, contract, t5_model):
    summaries = {}
    examples = [example for dataset in datasets for example in dataset.examples]
    transformation_cache = {}
    for method in BASELINE_REGISTRY:
        transform_key = method if method in {
            "tree_linearized_t5", "cot", "scratchpad", "tinyllama_lora"
        } else "unmodified"
        if transform_key in transformation_cache:
            summaries[method] = dict(transformation_cache[transform_key])
            continue
        if method == "tinyllama_lora":
            tokenizer = AutoTokenizer.from_pretrained(
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            )
        else:
            tokenizer = T5Tokenizer.from_pretrained(t5_model)
            tokenizer, _ = extend_tokenizer_for_dataset(tokenizer, dataset_name, verbose=False)
            if method == "scratchpad":
                tokenizer.add_special_tokens(
                    {"additional_special_tokens": ["<scratch>", "</scratch>"]}
                )
        transformed = [
            _transformed_text(method, dataset_name, row.input_text, row.target_text)
            for row in examples
        ]
        source_lengths, target_lengths = [], []
        for start in range(0, len(transformed), 1024):
            batch = transformed[start:start + 1024]
            source_lengths.extend(len(ids) for ids in tokenizer(
                [row[0] for row in batch], truncation=False, add_special_tokens=True,
            )["input_ids"])
            target_lengths.extend(len(ids) for ids in tokenizer(
                text_target=[row[1] for row in batch], truncation=False,
                add_special_tokens=True,
            )["input_ids"])
        source_sorted, target_sorted = sorted(source_lengths), sorted(target_lengths)
        percentile_index = max(0, int(0.99 * len(examples)) - 1)
        summary = {
            "examples": len(examples),
            "max_source_tokens": max(source_lengths),
            "max_target_tokens": max(target_lengths),
            "p99_source_tokens": source_sorted[percentile_index],
            "p99_target_tokens": target_sorted[percentile_index],
            "sources_exceeding_contract": sum(
                length > contract.max_source_length for length in source_lengths
            ),
            "targets_exceeding_contract": sum(
                length > contract.max_target_length for length in target_lengths
            ),
        }
        transformation_cache[transform_key] = summary
        summaries[method] = dict(summary)
    return summaries


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
        loaded_datasets = []
        for split in splits:
            dataset = cls(
                tokenizer=tokenizer, split=split, data_dir=str(data_dir),
                max_source_length=contract.max_source_length,
                max_target_length=contract.max_target_length, **extra,
            )
            loaded_datasets.append(dataset)
            report["benchmarks"][key][split] = summarize(dataset, contract.train_batch_size)
        report["benchmarks"][key]["baseline_transformations"] = summarize_baseline_transformations(
            loaded_datasets, dataset_name, contract, args.model
        )
    for cfq_split in ("mcd1", "mcd2", "mcd3"):
        contract = get_benchmark_contract("cfq", cfq_split)
        tokenizer = T5Tokenizer.from_pretrained(args.model)
        tokenizer, _ = extend_tokenizer_for_dataset(tokenizer, "cfq", verbose=False)
        key = f"cfq/{cfq_split}"
        report["benchmarks"][key] = {}
        loaded_datasets = []
        for split in ("train", "test"):
            dataset = CFQDataset(
                tokenizer=tokenizer, split=split, cfq_split=cfq_split,
                data_dir=str(args.data_root / "cfq"),
                max_source_length=contract.max_source_length,
                max_target_length=contract.max_target_length,
            )
            loaded_datasets.append(dataset)
            report["benchmarks"][key][split] = summarize(dataset, contract.train_batch_size)
        report["benchmarks"][key]["baseline_transformations"] = summarize_baseline_transformations(
            loaded_datasets, "cfq", contract, args.model
        )
    failures = [
        f"{benchmark}/{method}"
        for benchmark, sections in report["benchmarks"].items()
        for method, values in sections["baseline_transformations"].items()
        if values["sources_exceeding_contract"] or values["targets_exceeding_contract"]
    ]
    report["failed_baseline_transformations"] = failures
    report["passed"] = not failures
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not report["passed"]:
        raise SystemExit(
            "Baseline transformations exceed publication contracts: " + ", ".join(failures)
        )


if __name__ == "__main__":
    main()
