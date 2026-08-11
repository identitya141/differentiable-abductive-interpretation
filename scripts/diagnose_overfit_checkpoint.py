#!/usr/bin/env python3
"""Inspect predictions and teacher-forced accuracy for an overfit checkpoint."""

import argparse
import json
import os
from pathlib import Path
import site
import sys

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
venv_dir = os.environ.get("VENV_DIR")
if venv_dir:
    sys.path.insert(
        0,
        str(
            Path(venv_dir)
            / "lib"
            / f"python{sys.version_info.major}.{sys.version_info.minor}"
            / "site-packages"
        ),
    )
sys.path.append(site.getusersitepackages())

from transformers import T5Tokenizer

from scripts.run_small_overfit import _subset_loader
from scripts.train import create_model, get_data_module
from src.evaluation.metrics import evaluate_model, normalize_for_eval
from src.utils.config import load_config
from src.utils.tokenizer_utils import extend_tokenizer_for_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gate-report", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    config.data.data_dir = str(args.data_dir)
    config.data.num_workers = 0
    gate_report = json.loads(args.gate_report.read_text(encoding="utf-8"))

    tokenizer = T5Tokenizer.from_pretrained(config.model.base_model)
    tokenizer, _ = extend_tokenizer_for_dataset(tokenizer, "scan", verbose=False)
    data_module = get_data_module("scan", tokenizer, config)
    data_module.setup()
    subset = Subset(
        data_module.train_dataset,
        gate_report["selected_training_indices"],
    )
    loader: DataLoader = _subset_loader(
        data_module,
        subset,
        config.training.eval_batch_size,
        shuffle=False,
        seed=gate_report["seed"],
    )

    model = create_model(config)
    model.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    device = torch.device("cuda")
    model.to(device)

    evaluation = evaluate_model(
        model=model,
        dataloader=loader,
        tokenizer=tokenizer,
        device=device,
        dataset_type="scan",
    )

    teacher_correct = 0
    teacher_total = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            labels = batch.labels.to(device)
            outputs = model(
                input_ids=batch.input_ids.to(device),
                attention_mask=batch.attention_mask.to(device),
                labels=labels,
                compute_abstraction_loss=False,
            )
            predictions = outputs.logits.argmax(dim=-1)
            mask = labels.ne(-100)
            teacher_correct += int(predictions.eq(labels).logical_and(mask).sum().item())
            teacher_total += int(mask.sum().item())

    rows = []
    for input_text, prediction, target in zip(
        evaluation.inputs or [],
        evaluation.predictions or [],
        evaluation.targets or [],
    ):
        rows.append(
            {
                "input": input_text,
                "prediction": prediction,
                "target": target,
                "normalized_prediction": normalize_for_eval(prediction, "scan"),
                "normalized_target": normalize_for_eval(target, "scan"),
            }
        )
    print(
        json.dumps(
            {
                "exact_match": evaluation.exact_match,
                "token_accuracy": evaluation.token_accuracy,
                "teacher_forced_token_accuracy": (
                    teacher_correct / teacher_total if teacher_total else 0.0
                ),
                "avg_pred_length": evaluation.avg_pred_length,
                "avg_target_length": evaluation.avg_target_length,
                "examples": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()