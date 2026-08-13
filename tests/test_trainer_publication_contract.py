"""Regression tests for publication evaluation and checkpoint contracts."""

import os
os.environ.setdefault("USE_TF", "0")

from types import SimpleNamespace
from unittest.mock import patch

import torch

from src.training.trainer import DAITrainer, TrainingConfig


class _EvaluationModel:
    def eval(self):
        return self

    def train(self):
        return self

    def __call__(self, **kwargs):
        return SimpleNamespace(loss=torch.tensor(0.0))

    def generate(self, **kwargs):
        return torch.ones((kwargs["input_ids"].shape[0], 2), dtype=torch.long)


def test_non_scan_validation_uses_its_full_contract_ceiling():
    trainer = DAITrainer.__new__(DAITrainer)
    trainer.eval_dataloader = [{
        "input_ids": torch.ones((1, 3), dtype=torch.long),
        "attention_mask": torch.ones((1, 3), dtype=torch.long),
        "labels": torch.ones((1, 3), dtype=torch.long),
    }]
    trainer.model = _EvaluationModel()
    trainer.config = TrainingConfig(dataset_type="cogs")
    trainer.tokenizer = object()
    trainer.compute_metrics = lambda predictions, labels: {"exact_match": 0.0}
    trainer.device = torch.device("cpu")

    with patch(
        "src.training.trainer.generate_scan_optimized",
        return_value=torch.ones((1, 2), dtype=torch.long),
    ) as generate:
        trainer.evaluate()

    assert generate.call_args.kwargs["max_new_tokens"] == 1024


def test_checkpoint_pruning_preserves_best_final_and_newest(tmp_path):
    trainer = DAITrainer.__new__(DAITrainer)
    trainer.config = SimpleNamespace(
        output_path=tmp_path, save_total_limit=3,
    )
    root = tmp_path / "checkpoints"
    for name in ("best", "epoch_1", "epoch_2", "final"):
        (root / name).mkdir(parents=True)
    os.utime(root / "epoch_1", (1, 1))
    os.utime(root / "epoch_2", (2, 2))
    trainer._prune_checkpoints()

    assert (root / "best").is_dir()
    assert (root / "final").is_dir()
    assert (root / "epoch_2").is_dir()
    assert not (root / "epoch_1").exists()


def test_training_config_carries_dataset_identity():
    assert TrainingConfig(dataset_type="cfq").dataset_type == "cfq"
