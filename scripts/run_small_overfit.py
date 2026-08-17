#!/usr/bin/env python3
"""Run the bounded SCAN small-data overfitting publication gate."""

import argparse
import json
import random
import sys
import traceback
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from transformers import T5Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.train import create_model, get_data_module
from src.evaluation.metrics import evaluate_model
from src.evaluation.overfit_gate import evaluate_overfit_gate
from src.training.trainer import DAITrainer, TrainingConfig
from src.utils.config import load_config
from src.utils.reproducibility import set_seed
from src.utils.tokenizer_utils import extend_tokenizer_for_dataset


def select_subset_indices(available_examples: int, subset_size: int, seed: int):
    if not 16 <= subset_size <= 32:
        raise ValueError("subset_size must be between 16 and 32")
    if available_examples < subset_size:
        raise ValueError(
            f"Requested {subset_size} examples, found {available_examples}"
        )
    return sorted(random.Random(seed).sample(range(available_examples), subset_size))


def _training_config(config, output_dir: Path, seed: int) -> TrainingConfig:
    return TrainingConfig(
        experiment_name=config.experiment_name,
        run_id="overfit_run",
        seed=seed,
        num_epochs=config.training.num_epochs,
        max_steps=config.training.max_steps,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        max_grad_norm=config.training.max_grad_norm,
        train_batch_size=config.training.train_batch_size,
        eval_batch_size=config.training.eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        warmup_ratio=config.training.warmup_ratio,
        lr_scheduler=config.training.lr_scheduler,
        fp16=config.training.fp16,
        fp16_initial_scale=config.training.fp16_initial_scale,
        abstraction_loss_weight=config.abstraction.abstraction_loss_weight,
        abstraction_warmup_epochs=config.abstraction.warmup_epochs,
        abstraction_ramp_epochs=config.abstraction.ramp_epochs,
        abstraction_use_step_schedule=config.abstraction.use_step_schedule,
        abstraction_warmup_steps=config.abstraction.warmup_steps,
        abstraction_ramp_steps=config.abstraction.ramp_steps,
        abstraction_max_abs_task_ratio=config.abstraction.max_abs_task_ratio,
        abstraction_backoff_enabled=config.abstraction.backoff_enabled,
        abstraction_backoff_trigger_count=config.abstraction.backoff_trigger_count,
        abstraction_backoff_steps=config.abstraction.backoff_steps,
        eval_strategy="no",
        save_strategy="best",
        logging_steps=config.logging_steps,
        output_dir=str(output_dir),
    )


def _subset_loader(data_module, subset, batch_size: int, shuffle: bool, seed: int):
    generator = torch.Generator()
    generator.manual_seed(seed)
    base_dataset = data_module.train_dataset.dataset
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=base_dataset.collate_fn,
        generator=generator if shuffle else None,
    )


def _disable_dropout(model: torch.nn.Module) -> int:
    """Disable stochastic regularization for the bounded memorization gate."""
    dropout_modules = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
            dropout_modules += 1
    return dropout_modules


def run_gate(args) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to run the overfit gate")
    device_name = torch.cuda.get_device_name(0)
    if "H100" not in device_name.upper():
        raise RuntimeError(f"H100 required, found: {device_name}")

    config = load_config(args.config)
    if config.training.max_steps is None or config.training.max_steps <= 0:
        raise ValueError("Overfit config must define a positive training.max_steps")

    config.data.data_dir = str(args.data_dir)
    config.data.num_workers = 0
    config.training.seed = args.seed
    set_seed(args.seed)

    tokenizer = T5Tokenizer.from_pretrained(config.model.base_model)
    tokenizer, _ = extend_tokenizer_for_dataset(tokenizer, "scan", verbose=False)
    data_module = get_data_module("scan", tokenizer, config)
    data_module.setup()

    available_examples = len(data_module.train_dataset)
    selected_indices = select_subset_indices(
        available_examples, args.subset_size, args.seed
    )
    train_subset = Subset(data_module.train_dataset, selected_indices)
    train_loader = _subset_loader(
        data_module,
        train_subset,
        config.training.train_batch_size,
        shuffle=True,
        seed=args.seed,
    )
    evaluation_loader = _subset_loader(
        data_module,
        train_subset,
        config.training.eval_batch_size,
        shuffle=False,
        seed=args.seed,
    )

    model = create_model(config)
    model.resize_token_embeddings(len(tokenizer))
    disabled_dropout_modules = _disable_dropout(model)
    trainer = DAITrainer(
        model=model,
        config=_training_config(config, args.output_dir, args.seed),
        train_dataloader=train_loader,
        eval_dataloader=None,
        tokenizer=tokenizer,
    )
    training_results = trainer.train()
    training_evaluation = evaluate_model(
        model=model,
        dataloader=evaluation_loader,
        tokenizer=tokenizer,
        device=torch.device("cuda"),
        dataset_type="scan",
    )
    gate = evaluate_overfit_gate(
        train_log=training_results["train_log"],
        exact_match=training_evaluation.exact_match,
        optimizer_updates=training_results["optimizer_updates"],
        max_steps=config.training.max_steps,
        minimum_exact_match=args.minimum_exact_match,
        maximum_loss_ratio=args.maximum_loss_ratio,
    )
    gate.update(
        {
            "status": "passed" if gate["passed"] else "failed",
            "seed": args.seed,
            "subset_size": args.subset_size,
            "selected_training_indices": selected_indices,
            "gpu": device_name,
            "config": str(args.config),
            "data_dir": str(args.data_dir),
            "disabled_dropout_modules": disabled_dropout_modules,
            "prediction_samples": [
                {
                    "input": input_text,
                    "prediction": prediction,
                    "target": target,
                }
                for input_text, prediction, target in zip(
                    training_evaluation.inputs or [],
                    training_evaluation.predictions or [],
                    training_evaluation.targets or [],
                )
            ][:5],
        }
    )
    return gate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset-size", type=int, default=24)
    parser.add_argument("--minimum-exact-match", type=float, default=0.95)
    parser.add_argument("--maximum-loss-ratio", type=float, default=0.25)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = run_gate(args)
    except Exception as error:
        report = {
            "passed": False,
            "status": "error",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "seed": args.seed,
            "subset_size": args.subset_size,
        }
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report.get("passed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
