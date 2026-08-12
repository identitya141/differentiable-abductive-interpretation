import os
os.environ.setdefault("USE_TF", "0")

import tempfile
from pathlib import Path
from unittest.mock import patch

import yaml
from transformers import Seq2SeqTrainingArguments

from scripts.generate_breakdown_tables import build_breakdown_report
from scripts.train_baseline import (
    build_prediction_artifact_rows,
    hf_scheduler_name,
    load_baseline_dataset,
)
from src.models.baselines import BASELINE_REGISTRY


def test_every_baseline_config_constructs_hf_training_arguments():
    with tempfile.TemporaryDirectory() as directory:
        for spec in BASELINE_REGISTRY.values():
            config = yaml.safe_load(
                (Path("configs/baselines") / spec.config_name).read_text()
            )
            training = config.get("training", {})
            args = Seq2SeqTrainingArguments(
                output_dir=str(Path(directory) / spec.key),
                num_train_epochs=training.get("num_epochs", 20),
                learning_rate=training.get("learning_rate", 3e-4),
                warmup_ratio=training.get("warmup_ratio", 0.1),
                lr_scheduler_type=hf_scheduler_name(
                    training.get("lr_scheduler", "cosine")
                ),
                report_to=[],
            )
            assert args.output_dir.endswith(spec.key)


def test_publication_loader_raises_without_attempting_network_access():
    with tempfile.TemporaryDirectory() as directory, patch(
        "datasets.load_dataset"
    ) as network_loader:
        try:
            load_baseline_dataset(
                "scan", Path(directory), tokenizer=None, split="length",
                publication_mode=True,
            )
        except FileNotFoundError as error:
            assert "forbid Internet fallback" in str(error)
        else:
            raise AssertionError("Missing staged publication data was accepted")
        network_loader.assert_not_called()


def test_baseline_writer_output_passes_breakdown_contract():
    rows = build_prediction_artifact_rows(
        metadata_rows=[{
            "input_text": "jump twice", "composition_depth": 2,
            "generalization_category": "length",
        }],
        predictions=["I_JUMP I_JUMP"], targets=["I_JUMP I_JUMP"],
        normalized_predictions=["I_JUMP I_JUMP"],
        normalized_targets=["I_JUMP I_JUMP"],
        experiment_name="scan_length_reference_t5", method="reference_t5",
        dataset_name="scan", split="length", seed=42,
    )
    report = build_breakdown_report({"reference_t5": {42: rows}}, [42])
    assert report["methods"]["reference_t5"]["depth"]["2"]["mean"] == 1.0
