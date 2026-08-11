#!/usr/bin/env python3
"""Diagnose first-batch FP16 gradients for task and abstraction branches."""

import argparse

import torch
from torch.amp import autocast
from transformers import T5Tokenizer

from scripts.train import create_model, get_data_module
from src.utils.config import load_config
from src.utils.tokenizer_utils import extend_tokenizer_for_dataset


def gradient_stats(model):
    results = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        finite = torch.isfinite(gradient)
        results.append(
            {
                "name": name,
                "nan": bool(torch.isnan(gradient).any()),
                "inf": bool(torch.isinf(gradient).any()),
                "max_abs": (
                    float(gradient[finite].abs().max())
                    if finite.any()
                    else float("nan")
                ),
            }
        )
    return results


def run_backward(model, batch, mode):
    model.zero_grad(set_to_none=True)
    torch.manual_seed(2026)
    with autocast("cuda", dtype=torch.float16):
        if mode == "reference_task_only":
            output = model.t5(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            task_loss = output.loss
            raw_abstraction_loss = None
            loss = task_loss
        else:
            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                composition_specs=batch["composition_specs"],
                compute_abstraction_loss=True,
            )
            task_loss = output.task_loss
            raw_abstraction_loss = output.raw_abstraction_loss

        if mode == "compute_raw_but_ignore":
            loss = task_loss
        elif mode == "zero_times_raw":
            loss = output.task_loss + 0.0 * output.raw_abstraction_loss
        elif mode != "reference_task_only":
            raise ValueError(mode)

    raw_value = (
        f"{raw_abstraction_loss.detach().float().item():.6f}"
        if raw_abstraction_loss is not None
        else "not_computed"
    )
    print(
        f"{mode}: task={task_loss.detach().float().item():.6f} "
        f"raw_abs={raw_value} "
        f"loss_finite={bool(torch.isfinite(loss))}"
    )
    loss.backward()
    stats = gradient_stats(model)
    affected = [item for item in stats if item["nan"] or item["inf"]]
    print(f"{mode}: nonfinite_gradient_count={len(affected)}")
    for item in affected[:20]:
        print(
            f"  {item['name']}: nan={item['nan']} inf={item['inf']} "
            f"max_finite_abs={item['max_abs']:.6g}"
        )
    largest = sorted(
        (item for item in stats if not (item["nan"] or item["inf"])),
        key=lambda item: item["max_abs"],
        reverse=True,
    )[:10]
    print(f"{mode}: largest_finite_gradients")
    for item in largest:
        print(f"  {item['name']}: {item['max_abs']:.6g}")


def compare_reference_logits(model, batch):
    model.eval()
    with torch.no_grad(), autocast("cuda", dtype=torch.float16):
        reference = model.t5(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        custom = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            composition_specs=batch["composition_specs"],
            compute_abstraction_loss=False,
        )
    difference = (custom.logits.float() - reference.logits.float()).abs()
    print(
        "reference_equivalence: "
        f"max_abs_logit_difference={difference.max().item():.6g} "
        f"mean_abs_logit_difference={difference.mean().item():.6g}"
    )
    model.train()


def run_anomaly_detection(model, batch):
    model.zero_grad(set_to_none=True)
    torch.manual_seed(2026)
    try:
        with torch.autograd.detect_anomaly():
            with autocast("cuda", dtype=torch.float16):
                output = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    composition_specs=batch["composition_specs"],
                    compute_abstraction_loss=True,
                )
                loss = output.task_loss + 0.0 * output.raw_abstraction_loss
            print("anomaly_detection: starting zero_times_raw backward")
            loss.backward()
    except RuntimeError as error:
        print(f"anomaly_detection: {error}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    tokenizer = T5Tokenizer.from_pretrained(config.model.base_model)
    tokenizer, _ = extend_tokenizer_for_dataset(tokenizer, "scan", verbose=False)
    data_module = get_data_module("scan", tokenizer, config)
    data_module.setup()
    batch = next(iter(data_module.train_dataloader()))
    batch = {
        key: value.cuda() if isinstance(value, torch.Tensor) else value
        for key, value in vars(batch).items()
    }

    model = create_model(config).cuda().train()
    model.resize_token_embeddings(len(tokenizer))
    compare_reference_logits(model, batch)
    for mode in (
        "reference_task_only",
        "compute_raw_but_ignore",
        "zero_times_raw",
    ):
        run_backward(model, batch, mode)
    run_anomaly_detection(model, batch)


if __name__ == "__main__":
    main()