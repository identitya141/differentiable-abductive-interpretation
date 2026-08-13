#!/usr/bin/env python3
"""Validate publication result and prediction artifacts before analysis."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

ExampleKey = Tuple[str, str, int, str]

REQUIRED_RESULT_NUMBERS = (
    "optimizer_updates",
    "examples_seen",
    "training_wall_clock_seconds",
    "accelerator_hours",
    "peak_cuda_memory_bytes",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_finite_numbers(value: Any, location: str) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError(f"Non-finite number at {location}: {value}")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            _require_finite_numbers(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _require_finite_numbers(child, f"{location}[{index}]")


def _validate_result(path: Path) -> Dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    _require_finite_numbers(result, str(path))

    for section_name in ("iid_evaluation", "final_evaluation"):
        section = result.get(section_name)
        if not isinstance(section, dict):
            raise ValueError(f"Missing {section_name} in {path}")
        for metric_name in ("accuracy", "exact_match"):
            value = section.get(metric_name)
            if not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
                raise ValueError(f"Invalid {section_name}.{metric_name} in {path}: {value}")
        examples = section.get("num_examples")
        correct = section.get("num_correct")
        if not isinstance(examples, int) or examples <= 0:
            raise ValueError(f"Invalid {section_name}.num_examples in {path}: {examples}")
        if not isinstance(correct, int) or not 0 <= correct <= examples:
            raise ValueError(f"Invalid {section_name}.num_correct in {path}: {correct}")

    for metric_name in REQUIRED_RESULT_NUMBERS:
        value = result.get(metric_name)
        if not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"Missing or invalid {metric_name} in {path}: {value}")

    parameters = result.get("model_parameters")
    if not isinstance(parameters, dict):
        raise ValueError(f"Missing model_parameters in {path}")
    for metric_name in ("total", "trainable"):
        value = parameters.get(metric_name)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"Invalid model_parameters.{metric_name} in {path}: {value}")
    return result


def _validate_predictions(path: Path, expected_seed: int) -> Tuple[str, Dict[ExampleKey, bool]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"Prediction artifact is empty: {path}")

    experiment_names = set()
    examples = {}
    for row_index, row in enumerate(rows):
        location = f"{path}:{row_index + 1}"
        if row.get("seed") != expected_seed:
            raise ValueError(f"Wrong seed at {location}: {row.get('seed')}")
        experiment_names.add(str(row.get("experiment_name")))
        prediction = row.get("normalized_prediction")
        target = row.get("normalized_target")
        correct = row.get("correct")
        if not isinstance(prediction, str) or not isinstance(target, str):
            raise ValueError(f"Missing normalized text at {location}")
        if not isinstance(correct, bool) or correct != (prediction == target):
            raise ValueError(f"Incorrect correctness flag at {location}")
        key = (
            str(row.get("dataset")),
            str(row.get("split")),
            int(row["example_index"]),
            target,
        )
        if key in examples:
            raise ValueError(f"Duplicate example at {location}: {key}")
        examples[key] = correct

    if len(experiment_names) != 1:
        raise ValueError(f"Mixed experiment names in {path}: {sorted(experiment_names)}")
    return experiment_names.pop(), examples


def validate_artifacts(
    experiment_dir: Path,
    methods: Sequence[str],
    seeds: Sequence[int],
    *,
    require_provenance: bool = False,
) -> Dict:
    if len(set(methods)) != len(methods):
        raise ValueError("Method list contains duplicates")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seed list contains duplicates")

    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "methods": list(methods),
        "seeds": list(seeds),
        "artifacts": {},
    }
    reference_keys: Dict[int, set] = {}
    experiment_names = set()

    for method in methods:
        method_manifest = {}
        method_experiment_names = set()
        for seed in seeds:
            run_dir = experiment_dir / method / f"seed_{seed}"
            result_path = run_dir / f"results_seed{seed}.json"
            prediction_path = run_dir / f"predictions_seed{seed}.jsonl"
            for path in (result_path, prediction_path):
                if not path.is_file() or path.stat().st_size == 0:
                    raise ValueError(f"Missing or empty artifact: {path}")

            result = _validate_result(result_path)
            if require_provenance:
                contract_path = run_dir / "run_contract.json"
                if not contract_path.is_file():
                    raise ValueError(f"Missing run contract: {contract_path}")
                contract = json.loads(contract_path.read_text(encoding="utf-8"))
                expected = {
                    "source_snapshot_id": contract.get("source_snapshot_id"),
                    "source_git_revision": contract.get("source_git_revision"),
                    "config_sha256": contract.get("config_sha256"),
                    "data_sha256": contract.get("data_sha256"),
                    "seed": seed,
                    "method": method,
                }
                for field, value in expected.items():
                    if value in (None, ""):
                        raise ValueError(f"Missing {field} in {contract_path}")
                    if result.get(field) != value:
                        raise ValueError(
                            f"Result/contract mismatch for {field} in {result_path}: "
                            f"{result.get(field)!r} != {value!r}"
                        )
            experiment_name, examples = _validate_predictions(prediction_path, seed)
            method_experiment_names.add(experiment_name)
            final_evaluation = result["final_evaluation"]
            if final_evaluation["num_examples"] != len(examples):
                raise ValueError(
                    f"Prediction count differs from final_evaluation.num_examples: {prediction_path}"
                )
            prediction_correct = sum(examples.values())
            if final_evaluation["num_correct"] != prediction_correct:
                raise ValueError(
                    f"Prediction correctness differs from final_evaluation.num_correct: {prediction_path}"
                )
            prediction_accuracy = prediction_correct / len(examples)
            if not math.isclose(
                float(final_evaluation["exact_match"]),
                prediction_accuracy,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"Prediction correctness differs from final_evaluation.exact_match: {prediction_path}"
                )

            keys = set(examples)
            if seed not in reference_keys:
                reference_keys[seed] = keys
            elif keys != reference_keys[seed]:
                raise ValueError(
                    f"Unpaired examples for method={method}, seed={seed}: "
                    f"expected={len(reference_keys[seed])}, found={len(keys)}"
                )

            method_manifest[str(seed)] = {
                "result_sha256": _sha256(result_path),
                "predictions_sha256": _sha256(prediction_path),
                "prediction_rows": len(examples),
            }

        if len(method_experiment_names) != 1:
            raise ValueError(
                f"Method {method} has inconsistent experiment names: "
                f"{sorted(method_experiment_names)}"
            )
        experiment_name = method_experiment_names.pop()
        if experiment_name in experiment_names:
            raise ValueError(f"Duplicate experiment name across methods: {experiment_name}")
        experiment_names.add(experiment_name)
        method_manifest["experiment_name"] = experiment_name
        manifest["artifacts"][method] = method_manifest

    manifest["validated_runs"] = len(methods) * len(seeds)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-provenance", action="store_true")
    args = parser.parse_args()

    manifest = validate_artifacts(
        args.experiment_dir, args.methods, args.seeds,
        require_provenance=args.require_provenance,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Validated {manifest['validated_runs']} runs; manifest: {args.output}")


if __name__ == "__main__":
    main()
