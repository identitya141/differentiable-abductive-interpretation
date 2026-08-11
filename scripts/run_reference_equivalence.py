#!/usr/bin/env python3
"""Gate publication runs on DAI/reference T5 logit equivalence."""

import argparse
import json
import sys
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.train import create_model
from src.evaluation.reference_equivalence import evaluate_reference_equivalence
from src.utils.config import load_config
from src.utils.reproducibility import set_seed


def deterministic_batch(device: torch.device):
    return {
        "input_ids": torch.tensor(
            [[71, 123, 19, 1, 0, 0], [8, 42, 91, 37, 1, 0]],
            dtype=torch.long,
            device=device,
        ),
        "attention_mask": torch.tensor(
            [[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0]],
            dtype=torch.long,
            device=device,
        ),
        "labels": torch.tensor(
            [[11, 29, 1, -100, -100], [53, 7, 31, 1, -100]],
            dtype=torch.long,
            device=device,
        ),
    }


def run_gate(
    config_path: Path,
    maximum_absolute_error: float,
    device_name: str,
) -> dict:
    config = load_config(config_path)
    if config.model.architecture != "dai":
        raise ValueError("Reference equivalence requires a DAI architecture config")
    if config.model.apply_projection:
        raise ValueError("Reference equivalence requires model.apply_projection=false")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    set_seed(config.training.seed)
    device = torch.device(device_name)
    model = create_model(config).to(device).eval()
    batch = deterministic_batch(device)

    with torch.inference_mode():
        reference = model.t5(**batch, return_dict=True)
        custom = model(
            **batch,
            compute_abstraction_loss=False,
            composition_specs=None,
        )

    report = evaluate_reference_equivalence(
        custom_logits=custom.logits,
        reference_logits=reference.logits,
        maximum_absolute_error=maximum_absolute_error,
    )
    report.update(
        {
            "status": "passed" if report["passed"] else "failed",
            "config": str(config_path),
            "base_model": config.model.base_model,
            "device": (
                torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
            ),
            "dtype": str(custom.logits.dtype),
            "training_mode": model.training,
            "apply_projection": config.model.apply_projection,
            "shared_encoder_object": model.dai_encoder.t5_encoder is model.t5.encoder,
        }
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--maximum-absolute-error", type=float, default=1e-5)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    args.report.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = run_gate(args.config, args.maximum_absolute_error, args.device)
    except Exception as error:
        report = {
            "passed": False,
            "status": "error",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
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