"""Standard-library tests for paired experiment statistics."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


def _load_comparison_module():
    path = Path("scripts/compare_experiments.py")
    spec = importlib.util.spec_from_file_location("compare_experiments", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


comparison = _load_comparison_module()


class ExperimentComparisonTests(unittest.TestCase):
    def test_compares_paired_seed_and_example_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            method_a_paths = []
            method_b_paths = []
            for seed in (42, 123, 456):
                method_a_path = root / f"a_seed{seed}.jsonl"
                method_b_path = root / f"b_seed{seed}.jsonl"
                method_a_paths.append(method_a_path)
                method_b_paths.append(method_b_path)
                self._write_artifact(
                    method_a_path,
                    "grounded",
                    seed,
                    [True, True, True, False],
                )
                self._write_artifact(
                    method_b_path,
                    "bottleneck",
                    seed,
                    [True, False, False, False],
                )

            report = comparison.compare(
                comparison.load_artifacts(method_a_paths),
                comparison.load_artifacts(method_b_paths),
            )

            self.assertEqual(report["paired_seeds"], [42, 123, 456])
            self.assertEqual(
                report["seed_level"]["mean_difference_a_minus_b"], 0.5
            )
            self.assertEqual(
                report["example_level_mcnemar_by_seed"]["42"]["a_only_correct"],
                2,
            )
            low, high = report["seed_level"]["bootstrap_95_percent_ci"]
            self.assertLessEqual(low, 0.5)
            self.assertGreaterEqual(high, 0.5)
            self.assertGreaterEqual(
                report["seed_level"]["paired_permutation_p_value"], 0.0
            )
            self.assertLessEqual(
                report["seed_level"]["paired_permutation_p_value"], 1.0
            )

    def test_holm_adjustment_is_monotone_in_rank(self):
        adjusted = comparison.holm_adjust([0.01, 0.04, 0.03])

        self.assertEqual(adjusted, [0.03, 0.06, 0.06])

    def test_pairs_different_input_representations_by_example_and_target(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            plain_path = root / "plain_seed42.jsonl"
            tree_path = root / "tree_seed42.jsonl"
            self._write_artifact(plain_path, "plain", 42, [True, False])
            self._write_artifact(tree_path, "tree", 42, [False, False])
            rows = [json.loads(line) for line in tree_path.read_text().splitlines()]
            for row in rows:
                row["input"] = f"(TREE {row['input']})"
            tree_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            _, plain = comparison.load_artifacts([plain_path])
            _, tree = comparison.load_artifacts([tree_path])

            self.assertEqual(set(plain[42]), set(tree[42]))

    @staticmethod
    def _write_artifact(path, experiment_name, seed, correctness):
        with path.open("w", encoding="utf-8") as handle:
            for index, correct in enumerate(correctness):
                row = {
                    "example_index": index,
                    "experiment_name": experiment_name,
                    "dataset": "scan",
                    "split": "length",
                    "seed": seed,
                    "input": f"command {index}",
                    "normalized_target": f"target {index}",
                    "correct": correct,
                }
                handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    unittest.main()
