#!/usr/bin/env python3
"""
Baseline Training Script

Trains baseline models for comparison with DAI.

FAIR COMPARISON PROTOCOL:
========================
This script is configured to ensure apples-to-apples comparison with DAI:

MATCHED (controlled variables):
  - Base model: t5-small (60.5M params)
  - Tokenizer: T5Tokenizer with same preprocessing
  - Dataset/split: Same (e.g., SCAN length)
  - Optimizer: AdamW with same betas, epsilon
  - Learning rate: 3e-4 (same as DAI)
  - LR schedule: Cosine with 10% warmup (same as DAI)
  - Weight decay: 0.01 (same as DAI)
  - Batch size: 32 train / 64 eval (same as DAI)
  - Training epochs: 20 (same as DAI)
  - Max grad norm: 1.0 (via HF trainer default)
  - Generation: beam search (8 beams), length_penalty=1.2, max=256
  - Evaluation: Same exact_match metric code

DIFFERENT (the point of the comparison):
  - Training objective: Cross-entropy only (no abstraction loss)
  - No λ warmup/scheduling (not applicable)
  - No over-constraint detection (not applicable)

COMPUTE TRACKING:
  - Wall-clock time logged for matched-compute analysis
  - FLOPs can be estimated from training logs

Usage:
    python scripts/train_baseline.py --baseline vanilla --dataset scan --split length --output-dir checkpoints/vanilla
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    Trainer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.baselines import (
    BaselineConfig,
    BASELINE_ALIASES,
    BASELINE_REGISTRY,
    canonical_baseline_name,
    create_baseline,
    linearize_source_only_tree,
)
from src.evaluation.compositional_metrics import CompositionParser
from src.evaluation.metrics import normalize_batch_for_eval, normalize_for_eval
from src.utils.reproducibility import set_seed
from src.utils.benchmark_contract import get_benchmark_contract, paired_holdout_indices
from src.utils.tokenizer_utils import (
    extend_tokenizer_for_dataset,
    resize_with_deterministic_added_token_init,
)
from src.utils.schedulers import hf_scheduler_name
from src.utils.generation import apply_generation_contract
import logging
import numpy as np
import yaml


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def build_prediction_artifact_rows(
    *, metadata_rows, predictions, targets, normalized_predictions,
    normalized_targets, experiment_name, method, dataset_name, split, seed,
):
    """Build the runner-independent prediction schema used by paper analyses."""
    if not (
        len(metadata_rows) == len(predictions) == len(targets)
        == len(normalized_predictions) == len(normalized_targets)
    ):
        raise ValueError("Prediction outputs and metadata must have identical lengths")
    parser = CompositionParser(dataset_name)
    rows = []
    for index, (metadata, prediction, target, normalized_prediction, normalized_target) in enumerate(zip(
        metadata_rows, predictions, targets, normalized_predictions, normalized_targets
    )):
        input_text = metadata.get("input_text")
        if not isinstance(input_text, str) or not input_text:
            raise ValueError(f"Missing input_text metadata for prediction row {index}")
        depth = metadata.get("composition_depth")
        if not isinstance(depth, int) or depth < 0:
            depth = parser.get_depth(input_text)
        rows.append({
            "example_index": index, "experiment_name": experiment_name,
            "method": method, "dataset": dataset_name, "split": split,
            "seed": seed, "input": input_text, "composition_depth": depth,
            "generalization_category": metadata.get("generalization_category"),
            "prediction": prediction, "target": target,
            "normalized_prediction": normalized_prediction,
            "normalized_target": normalized_target,
            "correct": normalized_prediction == normalized_target,
            "composition_violation": None,
        })
    return rows

def setup_file_logging(output_dir: Path) -> logging.FileHandler:
    """Set up file handler for logging to .txt file for easy copying."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "baseline_training_log.txt"
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    # Add to root logger so all loggers write to this file
    logging.getLogger().addHandler(file_handler)
    logger.info(f"Logging to file: {log_file}")
    return file_handler


def log_baseline_config_verification(
    baseline_type: str,
    base_model: str,
    dataset_name: str,
    split: str,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    lr_scheduler: str,
    warmup_ratio: float,
    generation_num_beams: int,
    generation_max_length: int,
    seed: int,
    output_dir: Path,
):
    """Log comprehensive config verification at startup for debugging."""
    logger.info("=" * 60)
    logger.info("BASELINE CONFIG VERIFICATION (for debugging reproducibility)")
    logger.info("=" * 60)
    
    logger.info("[BASELINE CONFIG]")
    logger.info(f"  baseline_type:   {baseline_type}")
    logger.info(f"  base_model:      {base_model}")
    
    logger.info("[DATA CONFIG]")
    logger.info(f"  dataset:         {dataset_name}")
    logger.info(f"  split:           {split}")
    
    logger.info("[TRAINING CONFIG]")
    logger.info(f"  num_epochs:      {num_epochs}")
    logger.info(f"  batch_size:      {batch_size}")
    logger.info(f"  eval_batch_size: {batch_size * 2} (2x train)")
    logger.info(f"  learning_rate:   {learning_rate}")
    logger.info(f"  lr_scheduler:    {lr_scheduler}")
    logger.info(f"  warmup_ratio:    {warmup_ratio}")
    logger.info(f"  weight_decay:    0.01")
    logger.info(f"  seed:            {seed}")
    
    logger.info("[GENERATION CONFIG]")
    logger.info(f"  num_beams:       {generation_num_beams}")
    logger.info(f"  max_length:      {generation_max_length}")
    
    logger.info("[OUTPUT]")
    logger.info(f"  output_dir:      {output_dir}")
    
    logger.info("[FAIR COMPARISON NOTES]")
    logger.info("  - Same base model, optimizer, LR schedule as DAI")
    logger.info("  - NO abstraction loss (baseline uses CE only)")
    logger.info("  - NO λ warmup/scheduling (not applicable)")
    
    logger.info("=" * 60)


def load_baseline_dataset(
    dataset_name: str, 
    data_dir: Path, 
    tokenizer, 
    split: str = None,
    max_length: int = 128,
    max_target_length: int = 128,
    baseline_type: str = "vanilla",
    model=None,
    split_seed: int = 42,
    publication_mode: bool = True,
):
    """
    Load and tokenize dataset for baseline training.
    
    Args:
        dataset_name: Dataset name (scan, cogs, cfq, clutrr, gsm8k)
        data_dir: Directory containing datasets
        tokenizer: HuggingFace tokenizer
        split: Dataset-specific split (e.g., 'length' for SCAN)
        max_length: Maximum sequence length
        
    Returns:
        Tokenized dataset with train/test splits
    """
    from datasets import Dataset, load_from_disk, DatasetDict
    from datasets import load_dataset as hf_load
    
    def rows_dataset(rows):
        return Dataset.from_dict({
            "input": [row[0] for row in rows],
            "output": [row[1] for row in rows],
            "category": [row[2] if len(row) > 2 else None for row in rows],
        })

    def tsv_rows(path):
        rows = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 2:
                    raise ValueError(f"Expected at least two TSV columns at {path}:{line_number}")
                rows.append((fields[0], fields[1], fields[2] if len(fields) > 2 else None))
        return rows

    def local_publication_dataset():
        """Read the exact staged corpora used by the DAI data modules."""
        root = Path(data_dir)
        if dataset_name == "scan":
            train_path = root / split / f"tasks_train_{split}.txt"
            test_path = root / split / f"tasks_test_{split}.txt"
            if not train_path.is_file() or not test_path.is_file():
                return None
            def scan_rows(path):
                rows = []
                for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                    if not line.startswith("IN:") or "OUT:" not in line:
                        raise ValueError(f"Malformed SCAN row at {path}:{line_number}")
                    source, target = line.split("OUT:", 1)
                    rows.append((source.removeprefix("IN:").strip(), target.strip(), "length"))
                return rows
            train = rows_dataset(scan_rows(train_path))
            training_indices, validation_indices = paired_holdout_indices(
                len(train), 0.1, split_seed
            )
            return DatasetDict({
                "train": train.select(training_indices),
                "validation": train.select(validation_indices),
                "iid_test": train.select(validation_indices),
                "test": rows_dataset(scan_rows(test_path)),
            })
        if dataset_name == "cogs":
            required = {name: root / filename for name, filename in {
                "train": "train.tsv", "validation": "dev.tsv",
                "iid_test": "test.tsv", "test": "gen.tsv",
            }.items()}
            if not all(path.is_file() for path in required.values()):
                return None
            return DatasetDict({name: rows_dataset(tsv_rows(path)) for name, path in required.items()})
        if dataset_name == "slog":
            candidates = {
                "train": ("cogs_LF/train.tsv", "train.tsv"),
                "validation": ("cogs_LF/dev.tsv", "dev.tsv"),
                "iid_test": ("cogs_LF/test.tsv", "test.tsv"),
                "test": ("generalization_sets/gen_cogsLF.tsv", "gen_cogsLF.tsv"),
            }
            paths = {}
            for name, options in candidates.items():
                matches = [root / option for option in options if (root / option).is_file()]
                if not matches:
                    return None
                paths[name] = matches[0]
            return DatasetDict({name: rows_dataset(tsv_rows(path)) for name, path in paths.items()})
        if dataset_name == "cfq":
            split_root = root / split
            if (split_root / "dataset_dict.json").is_file():
                loaded = load_from_disk(str(split_root))
            elif all((split_root / f"{name}.json").is_file() for name in ("train", "test")):
                def json_rows(name):
                    values = json.loads((split_root / f"{name}.json").read_text(encoding="utf-8"))
                    return [(item["question"], item["query"], split) for item in values]
                loaded = DatasetDict({name: rows_dataset(json_rows(name)) for name in ("train", "test")})
            else:
                return None
            train = loaded["train"]
            training_indices, validation_indices = paired_holdout_indices(
                len(train), 0.1, split_seed
            )
            return DatasetDict({
                "train": train.select(training_indices),
                "validation": train.select(validation_indices),
                "iid_test": train.select(validation_indices),
                "test": loaded["test"],
            })
        return None

    dataset = local_publication_dataset()

    # Build path with split if provided
    if split:
        dataset_path = data_dir / dataset_name / split
    else:
        dataset_path = data_dir / dataset_name
    
    # Try loading from saved HF dataset (various formats)
    for subdir in ["", "hf_dataset", "main"] if dataset is None else []:
        check_path = dataset_path / subdir if subdir else dataset_path
        if check_path.exists() and (check_path / "dataset_dict.json").exists():
            print(f"Loading dataset from {check_path}")
            dataset = load_from_disk(str(check_path))
            break
        # Check for train/test subdirectories
        if (check_path / "train").exists():
            print(f"Loading dataset from {check_path}")
            dataset = load_from_disk(str(check_path))
            break
    
    # Fall back to HuggingFace hub
    if dataset is None and publication_mode:
        raise FileNotFoundError(
            f"Canonical staged {dataset_name}/{split or ''} dataset was not found "
            f"under {data_dir}; publication runs forbid Internet fallback"
        )
    if dataset is None:
        print(f"Loading dataset '{dataset_name}' from HuggingFace Hub")
        if split:
            # 'split' here is the dataset config name (e.g., 'length' for SCAN)
            # not the train/test split - HF uses 'name' parameter for this
            dataset = hf_load(dataset_name, name=split, trust_remote_code=True)
        else:
            dataset = hf_load(dataset_name, trust_remote_code=True)
    
    # Tokenization function
    def raw_text(examples):
        # Handle different dataset formats
        if "commands" in examples:  # SCAN
            inputs = examples["commands"]
            targets = examples["actions"]
        elif "source" in examples:  # COGS
            inputs = examples["source"]
            targets = examples["target"]
        elif "question" in examples:  # CFQ, GSM8K
            inputs = examples["question"]
            if "answer" in examples:
                targets = examples["answer"]
            else:
                targets = examples["query"]
        else:
            # Generic format
            inputs = examples.get("input", examples.get("text", []))
            targets = examples.get("output", examples.get("label", []))
        
        return list(inputs), list(targets)

    def reasoning_and_answer(target: str):
        """Use supplied rationales when present; otherwise make the trace explicit."""
        if "####" in target:
            reasoning, answer = target.rsplit("####", 1)
            return reasoning.strip(), answer.strip()
        tokens = target.split()
        trace = " ; ".join(
            f"step {i}: {token}" for i, token in enumerate(tokens, start=1)
        )
        return trace or "direct", target

    def tokenize_function(examples):
        inputs, targets = raw_text(examples)
        categories = list(examples.get("category", [None] * len(inputs)))
        if baseline_type == "tree_linearized_t5":
            if dataset_name == "scan":
                from src.data.scan_composition import linearize_scan_command
                inputs = [linearize_scan_command(text) for text in inputs]
            else:
                inputs = [linearize_source_only_tree(text) for text in inputs]
        if baseline_type == "cot":
            inputs = [model.preprocess_input(text) for text in inputs]
            targets = [
                model.format_target_with_cot(*reasoning_and_answer(target))
                for target in targets
            ]
        elif baseline_type == "scratchpad":
            inputs = [model.preprocess_input(text) for text in inputs]
            targets = [
                model.format_target(*reasoning_and_answer(target))
                for target in targets
            ]

        untruncated_sources = tokenizer(
            list(inputs), truncation=False, add_special_tokens=True
        )["input_ids"]
        too_long_sources = [len(ids) for ids in untruncated_sources if len(ids) > max_length]
        if too_long_sources:
            raise ValueError(
                f"Source requires up to {max(too_long_sources)} tokens but the benchmark "
                f"contract permits {max_length}; refusing truncation"
            )
        untruncated_targets = tokenizer(
            text_target=list(targets), truncation=False, add_special_tokens=True
        )["input_ids"]
        too_long = [len(ids) for ids in untruncated_targets if len(ids) > max_target_length]
        if too_long:
            raise ValueError(
                f"Gold target requires up to {max(too_long)} tokens but the benchmark "
                f"contract permits {max_target_length}; refusing truncation"
            )

        model_inputs = tokenizer(
            inputs,
            max_length=max_length,
            truncation=False,
            padding="max_length",
        )
        
        labels = tokenizer(
            targets,
            max_length=max_target_length,
            truncation=False,
            padding="max_length",
        )
        
        # Mask PAD tokens in labels with -100 so they're ignored by loss
        pad_token_id = tokenizer.pad_token_id
        label_ids = [
            [(tok if tok != pad_token_id else -100) for tok in seq]
            for seq in labels["input_ids"]
        ]
        model_inputs["labels"] = label_ids
        model_inputs["input_text"] = list(inputs)
        model_inputs["generalization_category"] = categories
        parser = CompositionParser(dataset_name)
        model_inputs["composition_depth"] = [parser.get_depth(text) for text in inputs]
        return model_inputs
    
    # Get column names to remove after tokenization
    # DatasetDict.column_names returns {split: columns} dict, not list
    if isinstance(dataset, DatasetDict):
        first_key = list(dataset.keys())[0]
        cols = dataset[first_key].column_names
    else:
        # Single Dataset
        cols = dataset.column_names
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=cols,
    )
    
    return tokenized_dataset


def load_config_file(config_path):
    if config_path is None:
        return {}
    with open(config_path, "r") as handle:
        return yaml.safe_load(handle) or {}


def load_llama_dataset(
    dataset_name, data_dir, split, model, max_source_length, max_target_length,
    *, publication_mode=True, split_seed=42,
):
    """Build causal-LM examples with loss masked over the instruction prompt."""
    from datasets import Dataset, DatasetDict, load_from_disk, load_dataset as hf_load
    # ``data_dir`` is already the benchmark root selected by the matrix
    # (for example .../data/scan), so appending dataset_name would duplicate it.
    dataset_path = Path(data_dir) / split if dataset_name == "scan" and split else Path(data_dir)
    dataset = None
    def rows_dataset(rows):
        return Dataset.from_dict({
            "input": [row[0] for row in rows],
            "output": [row[1] for row in rows],
            "category": [row[2] for row in rows],
        })
    if dataset_name == "scan":
        train_path = dataset_path / f"tasks_train_{split}.txt"
        test_path = dataset_path / f"tasks_test_{split}.txt"
        if train_path.is_file() and test_path.is_file():
            def scan_rows(path):
                rows = []
                for line in path.read_text(encoding="utf-8").splitlines():
                    source, target = line.split("OUT:", 1)
                    rows.append((source.removeprefix("IN:").strip(), target.strip(), "length"))
                return rows
            train = rows_dataset(scan_rows(train_path))
            train_ids, val_ids = paired_holdout_indices(len(train), 0.1, split_seed)
            dataset = DatasetDict({
                "train": train.select(train_ids), "validation": train.select(val_ids),
                "iid_test": train.select(val_ids), "test": rows_dataset(scan_rows(test_path)),
            })
    elif dataset_name in {"cogs", "slog"}:
        if dataset_name == "cogs":
            names = {"train": "train.tsv", "validation": "dev.tsv", "iid_test": "test.tsv", "test": "gen.tsv"}
        else:
            names = {"train": "cogs_LF/train.tsv", "validation": "cogs_LF/dev.tsv", "iid_test": "cogs_LF/test.tsv", "test": "generalization_sets/gen_cogsLF.tsv"}
        if all((dataset_path / value).is_file() for value in names.values()):
            def tsv_rows(path):
                return [
                    (fields[0], fields[1], fields[2] if len(fields) > 2 else None)
                    for fields in (
                        line.rstrip("\n").split("\t")
                        for line in path.read_text(encoding="utf-8").splitlines()
                    )
                ]
            dataset = DatasetDict({key: rows_dataset(tsv_rows(dataset_path / value)) for key, value in names.items()})
    for subdir in ["", "hf_dataset", "main"] if dataset is None else []:
        candidate = dataset_path / subdir if subdir else dataset_path
        if candidate.exists() and ((candidate / "dataset_dict.json").exists() or (candidate / "train").exists()):
            dataset = load_from_disk(str(candidate))
            break
    if dataset is None and publication_mode:
        raise FileNotFoundError(
            f"Canonical staged {dataset_name}/{split or ''} dataset was not found "
            f"under {data_dir}; publication runs forbid Internet fallback"
        )
    if dataset is None:
        dataset = hf_load(dataset_name, name=split, trust_remote_code=True) if split else hf_load(dataset_name, trust_remote_code=True)

    tokenizer = model.tokenizer
    eos = tokenizer.eos_token or ""

    def encode(examples):
        if "commands" in examples:
            sources, targets = examples["commands"], examples["actions"]
        elif "source" in examples:
            sources, targets = examples["source"], examples["target"]
        elif "question" in examples:
            sources = examples["question"]
            targets = examples.get("answer", examples.get("query"))
        else:
            sources = examples.get("input", examples.get("text", []))
            targets = examples.get("output", examples.get("label", []))
        result = {"input_ids": [], "attention_mask": [], "labels": [], "prompt_length": [], "target_ids": [], "input_text": [], "generalization_category": [], "composition_depth": []}
        categories = examples.get("category", [None] * len(sources))
        parser = CompositionParser(dataset_name)
        for source, target, category in zip(sources, targets, categories):
            prompt_ids = tokenizer(model.format_prompt(source), truncation=False, add_special_tokens=True)["input_ids"]
            target_ids = tokenizer(str(target) + eos, truncation=False, add_special_tokens=False)["input_ids"]
            if len(prompt_ids) > max_source_length:
                raise ValueError(f"Source requires {len(prompt_ids)} tokens; limit is {max_source_length}")
            if len(target_ids) > max_target_length:
                raise ValueError(f"Target requires {len(target_ids)} tokens; limit is {max_target_length}")
            full_ids = prompt_ids + target_ids
            result["input_ids"].append(full_ids)
            result["attention_mask"].append([1] * len(full_ids))
            result["labels"].append([-100] * len(prompt_ids) + target_ids)
            result["prompt_length"].append(len(prompt_ids))
            result["target_ids"].append(target_ids)
            result["input_text"].append(str(source))
            result["generalization_category"].append(category)
            result["composition_depth"].append(parser.get_depth(str(source)))
        return result

    columns = dataset[next(iter(dataset))].column_names
    return dataset.map(encode, batched=True, remove_columns=columns)


class CausalBaselineCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        max_full = max(len(x["input_ids"]) for x in features)
        max_target = max(len(x["target_ids"]) for x in features)
        pad = self.tokenizer.pad_token_id
        batch = {key: [] for key in ("input_ids", "attention_mask", "labels", "target_ids")}
        batch["prompt_length"] = []
        for item in features:
            gap = max_full - len(item["input_ids"])
            batch["input_ids"].append(item["input_ids"] + [pad] * gap)
            batch["attention_mask"].append(item["attention_mask"] + [0] * gap)
            batch["labels"].append(item["labels"] + [-100] * gap)
            batch["target_ids"].append(item["target_ids"] + [-100] * (max_target - len(item["target_ids"])))
            batch["prompt_length"].append(item["prompt_length"])
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


class CausalGenerationTrainer(Trainer):
    """Trainer that evaluates decoder-only baselines without exposing gold answers."""
    def compute_loss(self, model, inputs, return_outputs=False):
        inputs = dict(inputs)
        inputs.pop("prompt_length", None)
        inputs.pop("target_ids", None)
        outputs = model(**inputs)
        return (outputs.loss, outputs) if return_outputs else outputs.loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss = self.compute_loss(model, inputs).detach()
        if prediction_loss_only:
            return loss, None, None
        lengths = inputs["prompt_length"].tolist()
        prompts = [inputs["input_ids"][i, :length] for i, length in enumerate(lengths)]
        max_prompt = max(lengths)
        pad = model.tokenizer.pad_token_id
        prompt_ids = torch.full((len(prompts), max_prompt), pad, dtype=torch.long, device=inputs["input_ids"].device)
        prompt_mask = torch.zeros_like(prompt_ids)
        for i, prompt in enumerate(prompts):
            prompt_ids[i, -len(prompt):] = prompt
            prompt_mask[i, -len(prompt):] = 1
        generated = model.generate(input_ids=prompt_ids, attention_mask=prompt_mask, max_new_tokens=model.baseline_config.max_target_length)
        generated = generated[:, max_prompt:]
        return loss, generated, inputs["target_ids"]


def train_baseline(
    baseline_type: str,
    output_dir: Path,
    data_dir: Path,
    dataset_name: str = "scan",
    split: str = None,
    seed: int = 42,
    num_epochs: int = 20,  # Match DAI: 20 epochs
    batch_size: int = 32,
    learning_rate: float = 3e-4,  # Match DAI: 3e-4 (not HF default 5e-5)
    base_model: str = "t5-small",
    max_source_length: int = 128,
    max_target_length: int = 128,
    eval_batch_size: int = None,
    gradient_accumulation_steps: int = 1,
    lr_scheduler: str = "cosine",
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    generation_num_beams: int = 8,
    generation_max_length: int = 256,
    constrain_to_training_targets: bool = False,
    record_eos_diagnostics: bool = True,
    model_kwargs: dict = None,
    publication_mode: bool = True,
):
    """Train a baseline model."""
    
    baseline_type = canonical_baseline_name(baseline_type)

    contract = get_benchmark_contract(dataset_name, split)
    max_source_length = contract.max_source_length
    max_target_length = contract.max_target_length
    batch_size = contract.train_batch_size
    eval_batch_size = contract.eval_batch_size
    generation_num_beams = contract.generation_num_beams
    generation_max_length = contract.generation_max_new_tokens

    # Set seed for reproducibility
    set_seed(seed)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up file logging for easy copying of job outputs
    file_handler = setup_file_logging(output_dir)
    
    # Log comprehensive config verification (critical for debugging)
    log_baseline_config_verification(
        baseline_type=baseline_type,
        base_model=base_model,
        dataset_name=dataset_name,
        split=split,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        lr_scheduler=lr_scheduler,
        warmup_ratio=warmup_ratio,
        generation_num_beams=generation_num_beams,
        generation_max_length=generation_max_length,
        seed=seed,
        output_dir=output_dir,
    )
    
    logger.info(f"Training Baseline: {baseline_type.upper()}")
    logger.info(f"Dataset: {dataset_name}")
    if split:
        logger.info(f"Split: {split}")
    logger.info(f"Base model: {base_model}")
    logger.info(f"Seed: {seed}")
    logger.info(f"Output: {output_dir}")
    
    print("="*60)
    print(f"Training Baseline: {baseline_type.upper()}")
    print("="*60)
    print(f"Dataset: {dataset_name}")
    if split:
        print(f"Split: {split}")
    print(f"Base model: {base_model}")
    print(f"Seed: {seed}")
    print(f"Output: {output_dir}")
    
    # Create baseline model
    config = BaselineConfig(
        base_model=base_model,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
    )
    model_kwargs = model_kwargs or {}
    if baseline_type == "tinyllama_lora":
        model = create_baseline(
            baseline_type, config, dataset_type=dataset_name,
            model_name=base_model, **model_kwargs,
        )
        tokenizer = model.tokenizer
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        tokenizer, _ = extend_tokenizer_for_dataset(
            tokenizer, dataset_name, verbose=True
        )
        if baseline_type in ("cot", "scratchpad", "symbolic"):
            model_kwargs["tokenizer"] = tokenizer
        model = create_baseline(
            baseline_type, config, dataset_type=dataset_name, **model_kwargs
        )
        resize_with_deterministic_added_token_init(
            model._hf_model, len(tokenizer), seed=contract.data_split_seed
        )
    
    # Load dataset
    print("\nLoading dataset...")
    if baseline_type == "tinyllama_lora":
        dataset = load_llama_dataset(
            dataset_name, data_dir, split, model,
            max_source_length, max_target_length,
            publication_mode=publication_mode,
            split_seed=contract.data_split_seed,
        )
    else:
        dataset = load_baseline_dataset(
            dataset_name, data_dir, tokenizer, split=split,
            max_length=max_source_length,
            max_target_length=max_target_length,
            baseline_type=baseline_type,
            model=model,
            split_seed=contract.data_split_seed,
            publication_mode=publication_mode,
        )

    if constrain_to_training_targets:
        if baseline_type != "random_init_t5":
            raise ValueError(
                "Target-vocabulary constraints are currently supported only for random_init_t5"
            )
        allowed_output_ids = {
            int(token_id)
            for labels in dataset["train"]["labels"]
            for token_id in labels
            if int(token_id) != -100
        }
        model.set_allowed_output_token_ids(allowed_output_ids)
        logger.info(
            "Random-init decoding constrained to %d training-target token IDs",
            len(model.allowed_output_token_ids),
        )
    
    # Data collator
    data_collator = (
        CausalBaselineCollator(tokenizer)
        if baseline_type == "tinyllama_lora"
        else DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model.t5, padding=True)
    )
    
    # Compute metrics function for exact match accuracy
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        
        # Decode predictions
        if isinstance(preds, tuple):
            preds = preds[0]
        
        # Replace -100 in labels (we set them for padding)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        
        # Decode to strings
        keep_markers = baseline_type == "scratchpad"
        decoded_preds = tokenizer.batch_decode(
            preds, skip_special_tokens=not keep_markers
        )
        decoded_labels = tokenizer.batch_decode(
            labels, skip_special_tokens=not keep_markers
        )
        if keep_markers:
            removable = filter(None, [
                tokenizer.pad_token,
                tokenizer.eos_token,
                tokenizer.bos_token,
            ])
            removable = tuple(removable)
            for token in removable:
                decoded_preds = [text.replace(token, "") for text in decoded_preds]
                decoded_labels = [text.replace(token, "") for text in decoded_labels]
        
        decoded_preds = normalize_batch_for_eval(
            decoded_preds,
            dataset_type=dataset_name,
            baseline_type=baseline_type,
        )
        decoded_labels = normalize_batch_for_eval(
            decoded_labels,
            dataset_type=dataset_name,
            baseline_type=baseline_type,
        )
        
        # Compute exact match
        exact_matches = sum(p == l for p, l in zip(decoded_preds, decoded_labels))
        exact_match = exact_matches / len(decoded_labels) if decoded_labels else 0.0
        
        # Token-level accuracy (for debugging)
        token_correct = 0
        token_total = 0
        for pred, label in zip(decoded_preds, decoded_labels):
            pred_tokens = pred.split()
            label_tokens = label.split()
            for i, tok in enumerate(label_tokens):
                token_total += 1
                if i < len(pred_tokens) and pred_tokens[i] == tok:
                    token_correct += 1
        token_accuracy = token_correct / token_total if token_total > 0 else 0.0
        
        return {
            "exact_match": exact_match,
            "token_accuracy": token_accuracy,
        }
    
    # Training arguments - MATCHED TO DAI FOR FAIR COMPARISON
    # Same: optimizer (AdamW), LR (3e-4), warmup (10%), schedule (cosine),
    #       batch size (32), epochs (20), weight_decay (0.01)
    # Different: No abstraction loss (that's the point of the baseline)
    if baseline_type != "tinyllama_lora":
        apply_generation_contract(model, dataset_name, tokenizer=tokenizer)
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size or batch_size * 2,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=hf_scheduler_name(lr_scheduler),
        logging_dir=str(output_dir / "logs"),
        logging_steps=100,
        evaluation_strategy="epoch",  # Fixed: was 'eval_strategy'
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_exact_match",  # Use exact match as primary metric
        greater_is_better=True,  # Higher exact match is better
        predict_with_generate=True,
        # The model generation config uses max_new_tokens, matching DAI exactly.
        generation_max_length=None,
        generation_num_beams=generation_num_beams,
        gradient_accumulation_steps=gradient_accumulation_steps,
        remove_unused_columns=baseline_type != "tinyllama_lora",
        seed=seed,
        report_to="tensorboard",
    )
    
    # Select eval dataset (prefer validation, fall back to test)
    if "validation" in dataset:
        eval_dataset = dataset["validation"]
    elif "test" in dataset:
        eval_dataset = dataset["test"]
    else:
        raise ValueError(f"No validation or test split found in dataset. Available: {list(dataset.keys())}")
    
    # Create trainer
    trainer_class = CausalGenerationTrainer if baseline_type == "tinyllama_lora" else Seq2SeqTrainer
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train with timing for compute comparison
    logger.info("Starting training...")
    print("\nStarting training...")
    train_start = time.time()
    train_result = trainer.train()
    train_time = time.time() - train_start
    
    # Save best model
    best_model_path = output_dir / "best_model"
    trainer.save_model(str(best_model_path))
    tokenizer.save_pretrained(str(best_model_path / "tokenizer"))

    def generation_diagnostics(raw_predictions, decoded_predictions, decoded_targets):
        raw_predictions = np.asarray(raw_predictions)
        eos_token_id = tokenizer.eos_token_id
        eos_emitted = (
            np.any(raw_predictions == eos_token_id, axis=1)
            if eos_token_id is not None else np.zeros(len(raw_predictions), dtype=bool)
        )
        prediction_lengths = [len(text.strip().split()) for text in decoded_predictions]
        target_lengths = [len(text.strip().split()) for text in decoded_targets]
        return {
            "eos_emission_rate": float(np.mean(eos_emitted)) if len(eos_emitted) else 0.0,
            "generation_limit_rate": float(np.mean(~eos_emitted)) if len(eos_emitted) else 0.0,
            "average_prediction_length": float(np.mean(prediction_lengths)) if prediction_lengths else 0.0,
            "average_target_length": float(np.mean(target_lengths)) if target_lengths else 0.0,
            "over_generation_rate": float(np.mean([
                prediction > 2 * max(1, target)
                for prediction, target in zip(prediction_lengths, target_lengths)
            ])) if target_lengths else 0.0,
        }

    def evaluate_split(eval_data):
        prediction_output = trainer.predict(eval_data)
        predictions = prediction_output.predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        labels = prediction_output.label_ids
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        keep_markers = baseline_type == "scratchpad"
        decoded_predictions = tokenizer.batch_decode(
            predictions, skip_special_tokens=not keep_markers
        )
        decoded_targets = tokenizer.batch_decode(
            labels, skip_special_tokens=not keep_markers
        )
        normalized_predictions = normalize_batch_for_eval(
            decoded_predictions, dataset_type=dataset_name, baseline_type=baseline_type
        )
        normalized_targets = normalize_batch_for_eval(
            decoded_targets, dataset_type=dataset_name, baseline_type=baseline_type
        )
        correct = [a == b for a, b in zip(normalized_predictions, normalized_targets)]
        metrics = {
            "accuracy": sum(correct) / len(correct),
            "exact_match": sum(correct) / len(correct),
            "num_examples": len(correct),
            "num_correct": sum(correct),
        }
        if record_eos_diagnostics:
            metrics["generation_diagnostics"] = generation_diagnostics(
                predictions, decoded_predictions, decoded_targets
            )
        return metrics, decoded_predictions, decoded_targets, normalized_predictions, normalized_targets

    iid_dataset = dataset.get("iid_test", dataset.get("validation"))
    ood_dataset = dataset.get("test")
    if iid_dataset is None or ood_dataset is None:
        raise ValueError("Publication baselines require IID and OOD evaluation splits")
    iid_metrics, *_ = evaluate_split(iid_dataset)
    final_metrics, predictions, targets, normalized_predictions, normalized_targets = evaluate_split(ood_dataset)

    experiment_name = f"{dataset_name}_{split}_{baseline_type}"
    prediction_path = output_dir / f"predictions_seed{seed}.jsonl"
    metadata_rows = [ood_dataset[index] for index in range(len(ood_dataset))]
    artifact_rows = build_prediction_artifact_rows(
        metadata_rows=metadata_rows, predictions=predictions, targets=targets,
        normalized_predictions=normalized_predictions,
        normalized_targets=normalized_targets, experiment_name=experiment_name,
        method=baseline_type, dataset_name=dataset_name, split=split, seed=seed,
    )
    with prediction_path.open("w", encoding="utf-8") as handle:
        for row in artifact_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    
    # Compute training stats for fair comparison reporting
    total_steps = train_result.global_step
    samples_per_second = len(dataset["train"]) * num_epochs / train_time
    
    # Save training config with compute metrics
    config_dict = {
        "baseline_type": baseline_type,
        "base_model": base_model,
        "dataset": dataset_name,
        "split": split,
        "seed": seed,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "lr_scheduler": lr_scheduler,
        "warmup_ratio": warmup_ratio,
        "weight_decay": weight_decay,
        "generation_num_beams": generation_num_beams,
        "generation_max_length": generation_max_length,
        "constrain_to_training_targets": constrain_to_training_targets,
        "benchmark_contract": contract.__dict__,
        # Compute tracking for fair comparison
        "compute_metrics": {
            "total_steps": total_steps,
            "train_time_seconds": train_time,
            "train_time_hours": train_time / 3600,
            "samples_per_second": samples_per_second,
            "train_samples": len(dataset["train"]),
            "eval_samples": len(eval_dataset),
        },
        # Document matched hyperparameters
        "fair_comparison_notes": {
            "optimization_budget": (
                "scratch-specific" if baseline_type == "random_init_t5" else "matched to DAI"
            ),
            "output_constraint": (
                "training-target vocabulary only" if constrain_to_training_targets else "none"
            ),
        },
    }
    with open(output_dir / "training_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    publication_result = {
        "schema_version": 1,
        "experiment_name": experiment_name,
        "dataset": dataset_name,
        "split": split,
        "method": baseline_type,
        "seed": seed,
        "optimizer_updates": int(train_result.global_step),
        "examples_seen": int(len(dataset["train"]) * num_epochs),
        "training_wall_clock_seconds": float(train_time),
        "accelerator_hours": float(train_time / 3600 if torch.cuda.is_available() else 0.0),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0),
        "model_parameters": {"total": total_parameters, "trainable": trainable_parameters},
        "iid_evaluation": iid_metrics,
        "final_evaluation": {
            **final_metrics,
            "ood_accuracy": final_metrics["accuracy"],
            "generalization_gap": iid_metrics["accuracy"] - final_metrics["accuracy"],
        },
        "prediction_artifact": str(prediction_path),
        "compute_metrics": config_dict["compute_metrics"],
        "source_snapshot_id": os.environ.get("SOURCE_SNAPSHOT_ID"),
        "source_git_revision": os.environ.get("SOURCE_GIT_REVISION"),
        "config_sha256": os.environ.get("CONFIG_SHA256"),
        "data_sha256": os.environ.get("DATA_SHA256"),
    }
    with (output_dir / f"results_seed{seed}.json").open("w", encoding="utf-8") as handle:
        json.dump(publication_result, handle, indent=2, sort_keys=True)
    
    # Log training summary (goes to both console and file)
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Model saved to: {best_model_path}")
    logger.info(f"Total steps: {total_steps}")
    logger.info(f"Training time: {train_time/3600:.2f} hours")
    logger.info(f"Samples/second: {samples_per_second:.1f}")
    
    print(f"\n{'='*60}")
    print(f"✓ Training complete!")
    print(f"{'='*60}")
    print(f"Model saved to: {best_model_path}")
    print(f"Total steps: {total_steps}")
    print(f"Training time: {train_time/3600:.2f} hours")
    print(f"Samples/second: {samples_per_second:.1f}")
    
    return trainer


def main():
    parser = argparse.ArgumentParser(description="Train baseline models")
    parser.add_argument("--baseline", type=str, default=None,
                        choices=sorted(set(BASELINE_REGISTRY) | set(BASELINE_ALIASES)),
                        help="Baseline type to train")
    parser.add_argument("--output-dir", "--output_dir", type=str, default=None,
                        help="Output directory for checkpoints")
    parser.add_argument("--data-dir", "--data_dir", type=str, default=None,
                        help="Directory containing datasets")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["scan", "cogs", "slog", "cfq", "clutrr", "gsm8k"],
                        help="Dataset to train on")
    parser.add_argument("--split", type=str, default=None,
                        help="Dataset split (e.g., 'length' for SCAN, 'mcd1' for CFQ)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of training epochs (default: 20 to match DAI)")
    parser.add_argument("--batch-size", "--batch_size", type=int, default=None,
                        help="Training batch size")
    parser.add_argument("--lr", "--learning_rate", type=float, default=None,
                        help="Learning rate (default: 3e-4 to match DAI)")
    parser.add_argument("--base-model", "--base_model", type=str, default=None,
                        help="Base model to use")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to baseline config YAML file")
    parser.add_argument(
        "--allow-network-data-fallback", action="store_true",
        help="Permit non-publication exploratory runs to download missing data",
    )
    
    args = parser.parse_args()
    file_config = load_config_file(args.config)
    model_config = file_config.get("model", {})
    training_config = file_config.get("training", {})
    generation_config = file_config.get("generation", {})
    baseline_type = args.baseline or file_config.get("baseline_type")
    if baseline_type is None:
        parser.error("--baseline is required unless baseline_type is set by --config")
    output_dir = args.output_dir or file_config.get("output_dir")
    if output_dir is None:
        parser.error("--output-dir is required unless output_dir is set by --config")

    model_kwargs = {}
    baseline_type = canonical_baseline_name(baseline_type)
    if baseline_type == "modular":
        modules = file_config.get("modules", {})
        model_kwargs = {
            key: modules[key]
            for key in ("num_modules", "module_dim", "num_composition_steps")
            if key in modules
        }
    elif baseline_type == "tinyllama_lora":
        lora = file_config.get("lora", {})
        quantization = file_config.get("quantization", {})
        model_kwargs = {
            "use_lora": lora.get("enabled", True),
            "use_4bit": quantization.get("use_4bit", True),
            "lora_r": lora.get("r", 16),
            "lora_alpha": lora.get("alpha", 32),
            "lora_dropout": lora.get("dropout", 0.05),
        }

    train_baseline(
        baseline_type=baseline_type,
        output_dir=Path(output_dir),
        data_dir=Path(args.data_dir or file_config.get("data_dir", "data")),
        dataset_name=args.dataset or file_config.get("dataset", "scan"),
        split=args.split,
        seed=args.seed if args.seed is not None else file_config.get("seed", 42),
        num_epochs=args.epochs if args.epochs is not None else training_config.get("num_epochs", 20),
        batch_size=args.batch_size if args.batch_size is not None else training_config.get("train_batch_size", 32),
        eval_batch_size=training_config.get("eval_batch_size"),
        learning_rate=args.lr if args.lr is not None else training_config.get("learning_rate", 3e-4),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 1),
        lr_scheduler=training_config.get("lr_scheduler", "cosine"),
        warmup_ratio=training_config.get("warmup_ratio", 0.1),
        weight_decay=training_config.get("weight_decay", 0.01),
        generation_num_beams=generation_config.get("num_beams", 8),
        generation_max_length=generation_config.get("max_length", 256),
        constrain_to_training_targets=generation_config.get(
            "constrain_to_training_targets", False
        ),
        record_eos_diagnostics=generation_config.get("record_eos_diagnostics", True),
        base_model=args.base_model or model_config.get("base_model", "t5-small"),
        max_source_length=model_config.get("max_source_length", 128),
        max_target_length=model_config.get("max_target_length", 128),
        model_kwargs=model_kwargs,
        publication_mode=not args.allow_network_data_fallback,
    )


if __name__ == "__main__":
    main()
