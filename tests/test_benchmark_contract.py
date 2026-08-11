"""Regression tests for the shared publication fairness contract."""

import inspect
import unittest

from pathlib import Path

from src.models.baselines import linearize_source_only_tree
from src.utils.benchmark_contract import (
    apply_benchmark_contract,
    get_benchmark_contract,
    paired_holdout_indices,
)
from src.utils.config import ExperimentConfig
from src.utils.generation import get_generation_config


class BenchmarkContractTests(unittest.TestCase):
    def test_dataset_limits_and_decoding_are_shared(self):
        expected = {
            "scan": (64, 128),
            "cogs": (128, 256),
            "slog": (256, 512),
            "cfq": (128, 256),
        }
        for dataset, lengths in expected.items():
            with self.subTest(dataset=dataset):
                contract = get_benchmark_contract(dataset)
                self.assertEqual(
                    (contract.max_source_length, contract.max_target_length), lengths
                )
                generation = get_generation_config(dataset)
                self.assertEqual(generation["num_beams"], 1)
                self.assertEqual(generation["min_new_tokens"], 0)
                self.assertFalse(generation["use_eos_ban"])
                self.assertEqual(
                    generation["max_new_tokens"], contract.max_target_length
                )

    def test_contract_separates_model_seed_from_split_seed_and_eval_loading(self):
        config = ExperimentConfig()
        config.training.seed = 130363
        contract = apply_benchmark_contract(config)
        self.assertEqual(config.training.seed, 130363)
        self.assertEqual(config.data.split_seed, 42)
        self.assertEqual(config.training.eval_batch_size, 64)
        self.assertEqual(config.data.eval_num_workers, 0)
        self.assertEqual(contract.data_split_seed, 42)

    def test_paired_holdout_indices_are_runner_independent(self):
        first = paired_holdout_indices(101, 0.1, 42)
        second = paired_holdout_indices(101, 0.1, 42)
        self.assertEqual(first, second)
        self.assertTrue(set(first[0]).isdisjoint(first[1]))
        self.assertEqual(sorted([*first[0], *first[1]]), list(range(101)))

    def test_non_scan_tree_linearization_cannot_read_a_gold_target(self):
        self.assertEqual(
            list(inspect.signature(linearize_source_only_tree).parameters), ["text"]
        )
        tree = linearize_source_only_tree("a student liked the book")
        self.assertIn("<TREE>", tree)
        self.assertIn("( SEQ", tree)

    def test_dai_runner_reloads_validation_best_before_final_test(self):
        source = Path("scripts/train.py").read_text(encoding="utf-8")
        self.assertLess(
            source.index('trainer.load_checkpoint("best")'),
            source.index("Running final held-out IID evaluation"),
        )


if __name__ == "__main__":
    unittest.main()
