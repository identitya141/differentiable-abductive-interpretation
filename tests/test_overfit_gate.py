"""Tests for bounded small-data overfit pass/fail criteria."""

import math
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from src.evaluation.overfit_gate import evaluate_overfit_gate
from src.training.trainer import DAITrainer
from src.utils.config import load_config
from scripts.run_small_overfit import _training_config, select_subset_indices


class OverfitGateTests(unittest.TestCase):
    def test_passes_substantial_loss_drop_and_near_perfect_exact_match(self):
        report = evaluate_overfit_gate(
            train_log=[
                {"task_loss": loss}
                for loss in (4.0, 3.8, 1.0, 0.5, 0.2, 0.1)
            ],
            exact_match=1.0,
            optimizer_updates=600,
            max_steps=1000,
        )

        self.assertTrue(report["passed"])
        self.assertTrue(report["criteria"]["finite_gradients"])
        self.assertTrue(report["criteria"]["within_step_limit"])

    def test_fails_when_memorization_or_loss_drop_is_insufficient(self):
        report = evaluate_overfit_gate(
            train_log=[{"task_loss": 4.0}, {"task_loss": 2.0}],
            exact_match=0.9,
            optimizer_updates=1000,
            max_steps=1000,
        )

        self.assertFalse(report["passed"])
        self.assertFalse(report["criteria"]["loss_decreased_substantially"])
        self.assertFalse(report["criteria"]["near_perfect_train_exact_match"])

    def test_rejects_nonfinite_task_loss(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            evaluate_overfit_gate(
                train_log=[{"task_loss": math.nan}],
                exact_match=1.0,
                optimizer_updates=1,
                max_steps=10,
            )

    def test_config_is_bounded_and_uses_full_structural_objective(self):
        config = load_config("configs/experiments/scan_small_overfit.json")

        self.assertEqual(config.training.max_steps, 6000)
        self.assertEqual(config.training.lr_scheduler, "linear")
        self.assertTrue(config.training.fp16)
        self.assertEqual(config.training.fp16_initial_scale, 1024.0)
        self.assertGreater(config.abstraction.structural_contrastive_weight, 0.0)
        self.assertEqual(config.eval_strategy, "steps")
        self.assertEqual(config.save_strategy, "best")
        trainer_config = _training_config(config, Path("test-output"), seed=42)
        self.assertTrue(trainer_config.abstraction_use_step_schedule)
        self.assertEqual(trainer_config.abstraction_warmup_steps, 50)
        self.assertEqual(trainer_config.abstraction_ramp_steps, 100)
        self.assertEqual(trainer_config.eval_strategy, "steps")
        self.assertEqual(trainer_config.eval_steps, 250)

    def test_constant_scheduler_uses_optimizer_learning_rate(self):
        trainer = DAITrainer.__new__(DAITrainer)
        trainer.config = SimpleNamespace(
            lr_scheduler="constant", warmup_steps=None, warmup_ratio=0.0
        )
        trainer.total_steps = 10
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        trainer.optimizer = torch.optim.AdamW([parameter], lr=0.001)

        trainer.lr_scheduler = trainer._create_scheduler()

        self.assertIsNone(trainer.lr_scheduler)
        self.assertEqual(trainer._current_learning_rate(), 0.001)

    def test_subset_selection_is_deterministic_and_bounded(self):
        first = select_subset_indices(100, 24, seed=42)
        second = select_subset_indices(100, 24, seed=42)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 24)
        self.assertEqual(len(set(first)), 24)
        with self.assertRaisesRegex(ValueError, "between 16 and 32"):
            select_subset_indices(100, 15, seed=42)


if __name__ == "__main__":
    unittest.main()
