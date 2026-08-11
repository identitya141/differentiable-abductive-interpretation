"""Tests for the custom/reference T5 logit-equivalence gate."""

import math
from unittest.mock import patch

import pytest
import torch
from transformers import T5Config

from src.evaluation.reference_equivalence import evaluate_reference_equivalence
from src.models.dai_transformer import DAIConfig, DAITransformer


def test_equivalence_evaluator_passes_negligible_error():
    reference = torch.tensor([[[1.0, 2.0]]])
    custom = reference + 1e-6

    report = evaluate_reference_equivalence(custom, reference, 1e-5)

    assert report["passed"]
    assert report["metrics"]["maximum_absolute_logit_error"] <= 1e-5


@pytest.mark.parametrize("invalid_value", [math.nan, math.inf])
def test_equivalence_evaluator_rejects_nonfinite_logits(invalid_value):
    report = evaluate_reference_equivalence(
        torch.tensor([invalid_value]), torch.zeros(1), 1e-5
    )

    assert not report["passed"]
    assert not report["criteria"]["finite_logits"]


def test_dai_task_logits_match_shared_reference_t5():
    t5_config = T5Config(
        vocab_size=128,
        d_model=16,
        d_ff=32,
        num_layers=2,
        num_decoder_layers=2,
        num_heads=2,
        d_kv=8,
        dropout_rate=0.0,
        decoder_start_token_id=0,
        pad_token_id=0,
        eos_token_id=1,
    )
    dai_config = DAIConfig(
        domain_type="type",
        constrained_layers=[0, 1],
        num_types=4,
        type_embed_dim=4,
        apply_projection=False,
    )
    with patch.object(T5Config, "from_pretrained", return_value=t5_config):
        model = DAITransformer(config=dai_config, pretrained=False).eval()

    batch = {
        "input_ids": torch.tensor([[2, 3, 1, 0], [4, 5, 6, 1]]),
        "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]]),
        "labels": torch.tensor([[7, 1, -100], [8, 9, 1]]),
    }
    with torch.inference_mode():
        reference = model.t5(**batch, return_dict=True)
        custom = model(**batch, compute_abstraction_loss=False)

    report = evaluate_reference_equivalence(custom.logits, reference.logits, 1e-5)

    assert model.dai_encoder.t5_encoder is model.t5.encoder
    assert report["passed"], report