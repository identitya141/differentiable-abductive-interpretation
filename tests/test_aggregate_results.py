"""Tests for current-schema multi-seed result aggregation."""

import importlib.util
import sys
import unittest
from pathlib import Path


def _load_aggregation_module():
    path = Path("scripts/aggregate_results.py")
    spec = importlib.util.spec_from_file_location("aggregate_results", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


aggregation = _load_aggregation_module()


class CurrentResultAggregationTests(unittest.TestCase):
    def test_aggregates_metrics_depth_efficiency_and_parameters(self):
        results = [
            self._result(42, 0.6, 0.4, 100.0),
            self._result(123, 0.8, 0.6, 140.0),
        ]

        summary = aggregation.aggregate_current_results(results)

        self.assertEqual(summary["seeds"], [42, 123])
        self.assertAlmostEqual(
            summary["final_evaluation"]["exact_match"]["mean"], 0.5
        )
        self.assertAlmostEqual(
            summary["final_evaluation"]["accuracy_by_depth"]["3"]["mean"],
            0.7,
        )
        self.assertAlmostEqual(
            summary["efficiency"]["training_wall_clock_seconds"]["mean"],
            120.0,
        )
        self.assertEqual(summary["model_parameters"]["total"], 1000)
        self.assertGreater(
            summary["final_evaluation"]["exact_match"]["std"], 0.0
        )

    def test_rejects_duplicate_seeds(self):
        results = [self._result(42, 0.6, 0.4, 100.0)] * 2

        with self.assertRaisesRegex(ValueError, "Duplicate seeds"):
            aggregation.aggregate_current_results(results)

    @staticmethod
    def _result(seed, iid_accuracy, ood_accuracy, wall_time):
        return {
            "seed": seed,
            "iid_evaluation": {
                "accuracy": iid_accuracy,
                "exact_match": iid_accuracy,
                "accuracy_by_depth": {"3": iid_accuracy},
                "accuracy_by_category": {},
                "num_examples": 10,
                "num_correct": int(iid_accuracy * 10),
            },
            "final_evaluation": {
                "accuracy": ood_accuracy,
                "exact_match": ood_accuracy,
                "ood_accuracy": ood_accuracy,
                "generalization_gap": iid_accuracy - ood_accuracy,
                "accuracy_by_depth": {"3": iid_accuracy},
                "accuracy_by_category": {"structural": ood_accuracy},
                "num_examples": 10,
                "num_correct": int(ood_accuracy * 10),
            },
            "optimizer_updates": 200,
            "examples_seen": 3200,
            "training_wall_clock_seconds": wall_time,
            "accelerator_hours": wall_time / 3600.0,
            "peak_cuda_memory_bytes": 1024,
            "model_parameters": {"total": 1000, "trainable": 900},
        }


if __name__ == "__main__":
    unittest.main()