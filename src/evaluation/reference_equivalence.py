"""Reference-equivalence checks for the DAI task-only forward path."""

import math
from typing import Any, Dict

import torch


def evaluate_reference_equivalence(
    custom_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    maximum_absolute_error: float = 1e-5,
) -> Dict[str, Any]:
    if maximum_absolute_error <= 0 or not math.isfinite(maximum_absolute_error):
        raise ValueError("maximum_absolute_error must be positive and finite")
    if custom_logits.shape != reference_logits.shape:
        raise ValueError(
            "Custom/reference logit shapes differ: "
            f"{tuple(custom_logits.shape)} != {tuple(reference_logits.shape)}"
        )

    custom = custom_logits.detach().float()
    reference = reference_logits.detach().float()
    finite_logits = bool(torch.isfinite(custom).all() and torch.isfinite(reference).all())
    if finite_logits:
        difference = (custom - reference).abs()
        maximum_error = float(difference.max()) if difference.numel() else 0.0
        mean_error = float(difference.mean()) if difference.numel() else 0.0
    else:
        maximum_error = math.inf
        mean_error = math.inf

    criteria = {
        "finite_logits": finite_logits,
        "maximum_absolute_error_within_tolerance": (
            finite_logits and maximum_error <= maximum_absolute_error
        ),
    }
    return {
        "passed": all(criteria.values()),
        "criteria": criteria,
        "metrics": {
            "maximum_absolute_logit_error": maximum_error,
            "mean_absolute_logit_error": mean_error,
            "compared_logit_count": custom.numel(),
            "logit_shape": list(custom.shape),
        },
        "thresholds": {
            "maximum_absolute_logit_error": maximum_absolute_error,
        },
    }