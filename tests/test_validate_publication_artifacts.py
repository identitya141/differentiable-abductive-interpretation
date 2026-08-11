"""Tests for publication artifact validation."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


def _load_validator_module():
    path = Path("scripts/validate_publication_artifacts.py")
    spec = importlib.util.spec_from_file_location(
        "validate_publication_artifacts", path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


validator = _load_validator_module()


class PublicationArtifactValidationTests(unittest.TestCase):
    def test_validates_complete_paired_runs_with_different_inputs(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for method in ("plain", "tree"):
                for seed in (42, 123):
                    self._write_run(root, method, seed, transformed=method == "tree")

            manifest = validator.validate_artifacts(
                root, ("plain", "tree"), (42, 123)
            )

            self.assertEqual(manifest["validated_runs"], 4)
            self.assertEqual(
                manifest["artifacts"]["plain"]["42"]["prediction_rows"], 2
            )
            self.assertEqual(
                len(manifest["artifacts"]["tree"]["42"]["result_sha256"]),
                64,
            )

    def test_rejects_nonfinite_result_values(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._write_run(root, "method", 42)
            result_path = root / "method/seed_42/results_seed42.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["final_loss"] = float("nan")
            result_path.write_text(json.dumps(result), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Non-finite number"):
                validator.validate_artifacts(root, ("method",), (42,))

    def test_rejects_cross_method_target_misalignment(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._write_run(root, "first", 42)
            self._write_run(root, "second", 42, target_suffix=" changed")

            with self.assertRaisesRegex(ValueError, "Unpaired examples"):
                validator.validate_artifacts(root, ("first", "second"), (42,))

    def test_rejects_metrics_that_disagree_with_predictions(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._write_run(root, "method", 42)
            result_path = root / "method/seed_42/results_seed42.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["final_evaluation"]["num_correct"] = 2
            result_path.write_text(json.dumps(result), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Prediction correctness differs"):
                validator.validate_artifacts(root, ("method",), (42,))

    @staticmethod
    def _write_run(
        root,
        method,
        seed,
        transformed=False,
        target_suffix="",
    ):
        run_dir = root / method / f"seed_{seed}"
        run_dir.mkdir(parents=True)
        result = {
            "final_loss": 1.0,
            "optimizer_updates": 200,
            "examples_seen": 3200,
            "training_wall_clock_seconds": 100.0,
            "accelerator_hours": 100.0 / 3600.0,
            "peak_cuda_memory_bytes": 1024,
            "model_parameters": {"total": 1000, "trainable": 900},
            "iid_evaluation": {
                "accuracy": 0.5,
                "exact_match": 0.5,
                "num_examples": 2,
                "num_correct": 1,
            },
            "final_evaluation": {
                "accuracy": 0.5,
                "exact_match": 0.5,
                "num_examples": 2,
                "num_correct": 1,
            },
        }
        (run_dir / f"results_seed{seed}.json").write_text(
            json.dumps(result), encoding="utf-8"
        )

        rows = []
        for index in range(2):
            target = f"TARGET {index}{target_suffix}"
            prediction = target if index == 0 else "WRONG"
            input_text = f"command {index}"
            if transformed:
                input_text = f"(TREE {input_text})"
            rows.append(
                {
                    "experiment_name": f"experiment_{method}",
                    "dataset": "scan",
                    "split": "length",
                    "seed": seed,
                    "example_index": index,
                    "input": input_text,
                    "normalized_prediction": prediction,
                    "normalized_target": target,
                    "correct": prediction == target,
                }
            )
        (run_dir / f"predictions_seed{seed}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()