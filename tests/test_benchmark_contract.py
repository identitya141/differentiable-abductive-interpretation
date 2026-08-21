"""Regression tests for the shared publication fairness contract."""

import inspect
import unittest

from pathlib import Path

from src.models.baselines import linearize_source_only_tree
from scripts.validate_publication_transformations import _transformed_text
from src.utils.benchmark_contract import (
    apply_benchmark_contract,
    get_baseline_contract,
    get_benchmark_contract,
    paired_holdout_indices,
)
from src.utils.config import ExperimentConfig
from src.utils.generation import get_generation_config


class BenchmarkContractTests(unittest.TestCase):
    def test_dataset_limits_and_decoding_are_shared(self):
        expected = {
            "scan": (64, 128),
            "cogs": (128, 1024),
            "slog": (256, 1024),
            "cfq": (128, 640),
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

    def test_transformed_baselines_get_measured_no_truncation_ceilings(self):
        expected_tree_sources = {
            "scan": 128,
            "cogs": 512,
            "slog": 512,
            "cfq": 256,
        }
        for dataset, source_length in expected_tree_sources.items():
            with self.subTest(dataset=dataset):
                transformed = get_baseline_contract(
                    dataset, "tree_linearized_t5"
                )
                raw = get_benchmark_contract(dataset)
                self.assertEqual(transformed.max_source_length, source_length)
                self.assertEqual(
                    transformed.max_target_length, raw.max_target_length
                )

        tinyllama = get_baseline_contract("scan", "tinyllama_lora")
        self.assertEqual(
            (tinyllama.max_source_length, tinyllama.max_target_length),
            (64, 320),
        )

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

    def test_unannotated_reasoning_baselines_do_not_copy_gold_into_a_trace(self):
        target = "I_WALK I_JUMP"
        _, cot_target = _transformed_text("cot", "scan", "walk and jump", target)
        _, scratch_target = _transformed_text(
            "scratchpad", "scan", "walk and jump", target
        )

        self.assertEqual(cot_target, f"Therefore, the answer is: {target}")
        self.assertEqual(scratch_target, f"[/SCRATCH] {target}")
        self.assertNotIn("step 1:", cot_target)
        self.assertNotIn("step 1:", scratch_target)

    def test_dai_runner_reloads_validation_best_before_final_test(self):
        source = Path("scripts/train.py").read_text(encoding="utf-8")
        self.assertLess(
            source.index('trainer.load_checkpoint("best")'),
            source.index("Running final held-out IID evaluation"),
        )

    def test_reviewer_docs_disclose_supervision_and_primary_type_domain(self):
        formal = Path("docs/FORMAL_GUARANTEES.md").read_text(encoding="utf-8")
        protocol = Path("docs/EXPERIMENT_PROTOCOL.md").read_text(encoding="utf-8")
        specification = Path("docs/RESEARCH_SPECIFICATION.md").read_text(
            encoding="utf-8"
        )

        for document in (formal, protocol, specification):
            self.assertIn("gold-derived", document)
            self.assertIn("input-only", document)
        self.assertIn("primary learned abstract representation", formal)
        self.assertIn("A = \\Delta^{T-1}", formal)
        self.assertIn("oracle", formal)
        self.assertIn("2026-08-11", protocol)
        self.assertNotIn(
            "chosen abstract domain (type-monotonicity)", specification
        )


if __name__ == "__main__":
    unittest.main()
