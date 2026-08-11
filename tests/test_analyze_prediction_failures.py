"""Tests for artifact-based publication failure analysis."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


def _load_module():
    path = Path("scripts/analyze_prediction_failures.py")
    spec = importlib.util.spec_from_file_location("analyze_prediction_failures", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


analysis = _load_module()


class PredictionFailureAnalysisTests(unittest.TestCase):
    def test_categorizes_sequence_errors(self):
        self.assertEqual(analysis.categorize_error("A", "A B"), "truncation")
        self.assertEqual(analysis.categorize_error("A B C", "A B"), "over_generation")
        self.assertEqual(analysis.categorize_error("B A A", "A B A"), "order_error")
        self.assertEqual(analysis.categorize_error("X Y", "A B"), "completely_wrong")
        self.assertEqual(analysis.categorize_error("A X", "A B"), "partial_error")

    def test_groups_failures_and_marks_report_exploratory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "predictions.jsonl"
            rows = [
                self._row(42, 0, "A B", "A B", True, 1),
                self._row(42, 1, "A", "A B", False, 2),
                self._row(123, 0, "B A", "A B", False, 2),
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            report = analysis.analyze_prediction_files([path], max_samples=1)

        method = report["methods"]["scan_full_contrastive"]
        self.assertEqual(report["analysis_status"], "exploratory")
        self.assertEqual(method["seeds"], [42, 123])
        self.assertEqual(method["failures"], 2)
        self.assertEqual(method["error_types"], {"order_error": 1, "truncation": 1})
        self.assertEqual(method["by_composition_depth"]["2"]["failures"], 2)
        self.assertEqual(len(method["failure_samples"]), 1)

    @staticmethod
    def _row(seed, index, prediction, target, correct, depth):
        return {
            "experiment_name": "scan_full_contrastive",
            "seed": seed,
            "example_index": index,
            "input": f"command {index}",
            "normalized_prediction": prediction,
            "normalized_target": target,
            "correct": correct,
            "composition_depth": depth,
            "generalization_category": "length",
        }


if __name__ == "__main__":
    unittest.main()