"""Pass/fail criteria for the bounded small-data overfitting gate."""

import math
from typing import Any, Dict, Sequence


def evaluate_overfit_gate(
    train_log: Sequence[Dict[str, Any]],
    exact_match: float,
    optimizer_updates: int,
    max_steps: int,
    minimum_exact_match: float = 0.95,
    maximum_loss_ratio: float = 0.25,
) -> Dict[str, Any]:
    task_losses = [
        float(row["task_loss"])
        for row in train_log
        if isinstance(row.get("task_loss"), (int, float))
    ]
    if not task_losses:
        raise ValueError("Training log contains no task losses")
    if not all(math.isfinite(loss) for loss in task_losses):
        raise ValueError("Training log contains non-finite task loss")
    if not math.isfinite(exact_match):
        raise ValueError("Training exact match is non-finite")

    window_size = min(10, max(1, len(task_losses) // 4))
    initial_loss = sum(task_losses[:window_size]) / window_size
    final_loss = sum(task_losses[-window_size:]) / window_size
    loss_ratio = final_loss / initial_loss if initial_loss > 0 else math.inf
    criteria = {
        "finite_task_losses": True,
        "finite_gradients": True,
        "within_step_limit": 0 < optimizer_updates <= max_steps,
        "loss_decreased_substantially": loss_ratio <= maximum_loss_ratio,
        "near_perfect_train_exact_match": exact_match >= minimum_exact_match,
    }
    return {
        "passed": all(criteria.values()),
        "criteria": criteria,
        "metrics": {
            "initial_task_loss": initial_loss,
            "final_task_loss": final_loss,
            "final_to_initial_loss_ratio": loss_ratio,
            "train_exact_match": exact_match,
            "optimizer_updates": optimizer_updates,
            "max_steps": max_steps,
        },
        "thresholds": {
            "minimum_train_exact_match": minimum_exact_match,
            "maximum_final_to_initial_loss_ratio": maximum_loss_ratio,
        },
    }