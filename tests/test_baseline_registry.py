from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch


def tiny_t5_config():
    from transformers import T5Config

    return T5Config(
        vocab_size=32,
        d_model=8,
        d_ff=16,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        d_kv=4,
        dropout_rate=0.0,
        decoder_start_token_id=0,
        pad_token_id=0,
        eos_token_id=1,
    )


def test_registry_is_the_canonical_eleven_baselines():
    from src.models.baselines import BASELINE_REGISTRY

    assert tuple(BASELINE_REGISTRY) == (
        "reference_t5",
        "random_init_t5",
        "tree_linearized_t5",
        "random_structure",
        "shuffled_structure",
        "simple_consistency",
        "cot",
        "scratchpad",
        "modular",
        "symbolic",
        "tinyllama_lora",
    )
    assert len(BASELINE_REGISTRY) == 11


def test_legacy_names_do_not_create_duplicate_baselines():
    from src.models.baselines import canonical_baseline_name

    assert canonical_baseline_name("vanilla") == "reference_t5"
    assert canonical_baseline_name("nesy") == "modular"
    assert canonical_baseline_name("llama") == "tinyllama_lora"


def test_previously_config_only_controls_have_model_classes():
    from src.models.baselines import (
        BASELINE_REGISTRY,
        RandomInitT5,
        SimpleConsistencyT5,
        TreeLinearizedT5,
    )

    assert BASELINE_REGISTRY["random_init_t5"].model_class is RandomInitT5
    assert BASELINE_REGISTRY["tree_linearized_t5"].model_class is TreeLinearizedT5
    assert BASELINE_REGISTRY["simple_consistency"].model_class is SimpleConsistencyT5


def test_random_init_uses_scratch_specific_publication_profile():
    import yaml
    from pathlib import Path

    config_path = Path(__file__).resolve().parents[1] / "configs/baselines/random_init_t5.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["training"]["num_epochs"] == 60
    assert config["training"]["learning_rate"] == 1e-3
    assert config["training"]["lr_scheduler"] == "inverse_sqrt"
    assert config["generation"]["constrain_to_training_targets"] is True
    assert config["generation"]["record_eos_diagnostics"] is True


def test_random_init_output_constraint_keeps_target_tokens_and_eos():
    from src.models.baselines import BaselineConfig, RandomInitT5

    config = tiny_t5_config()
    with patch(
        "src.models.baselines.baseline_models.T5Config.from_pretrained",
        return_value=config,
    ):
        model = RandomInitT5(BaselineConfig(base_model="tiny-local-t5"))

    model.set_allowed_output_token_ids([7, 8, -100])
    assert model.allowed_output_token_ids == (1, 7, 8)


def test_proposed_dai_method_is_not_registered_as_a_baseline():
    from src.models.baselines import BASELINE_REGISTRY

    assert "full_contrastive" not in BASELINE_REGISTRY
    assert "dai" not in BASELINE_REGISTRY


def test_dai_save_load_round_trip_is_download_free(tmp_path):
    from src.models.dai_transformer import DAIConfig, DAITransformer

    config = DAIConfig(
        base_model_name="tiny-local-t5",
        domain_type="type",
        constrained_layers=[0],
        num_types=4,
        type_embed_dim=4,
        abstraction_loss_weight=0.0,
    )
    local_config = tiny_t5_config()
    with patch(
        "src.models.dai_transformer.T5Config.from_pretrained",
        return_value=local_config,
    ):
        model = DAITransformer(config, pretrained=False).eval()
        input_ids = torch.tensor([[2, 3, 1]])
        attention_mask = torch.ones_like(input_ids)
        decoder_input_ids = torch.tensor([[0, 4]])
        with torch.no_grad():
            expected = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                compute_abstraction_loss=False,
            ).logits

        save_dir = tmp_path / "checkpoint"
        model.save_pretrained(str(save_dir))
        restored = DAITransformer.from_pretrained(str(save_dir)).eval()
        with torch.no_grad():
            actual = restored(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                compute_abstraction_loss=False,
            ).logits

    assert (save_dir / "dai_config.json").is_file()
    assert (save_dir / "pytorch_model.bin").is_file()
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "baseline_key",
    (
        "reference_t5",
        "random_init_t5",
        "tree_linearized_t5",
        "random_structure",
        "shuffled_structure",
        "simple_consistency",
        "cot",
        "scratchpad",
        "modular",
        "symbolic",
        "tinyllama_lora",
    ),
)
def test_every_registered_baseline_performs_optimizer_step_without_download(
    baseline_key,
):
    from transformers import GPT2Config, GPT2LMHeadModel, T5ForConditionalGeneration
    from src.models.baselines import BaselineConfig, create_baseline

    t5_config = tiny_t5_config()
    causal_model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=32,
            n_embd=8,
            n_layer=1,
            n_head=2,
            bos_token_id=1,
            eos_token_id=1,
            pad_token_id=0,
        )
    )
    tokenizer = SimpleNamespace(
        padding_side="right",
        pad_token="<pad>",
        eos_token="</s>",
        pad_token_id=0,
        eos_token_id=1,
    )
    kwargs = {}
    if baseline_key == "tinyllama_lora":
        kwargs.update(use_lora=False, use_4bit=False, model_name="tiny-local-causal")

    with (
        patch(
            "src.models.baselines.baseline_models.T5ForConditionalGeneration.from_pretrained",
            side_effect=lambda *_args, **_kwargs: T5ForConditionalGeneration(t5_config),
        ),
        patch(
            "src.models.baselines.baseline_models.T5Config.from_pretrained",
            return_value=t5_config,
        ),
        patch(
            "src.models.dai_transformer.T5ForConditionalGeneration.from_pretrained",
            side_effect=lambda *_args, **_kwargs: T5ForConditionalGeneration(t5_config),
        ),
        patch(
            "src.models.baselines.baseline_models.AutoModelForCausalLM.from_pretrained",
            return_value=causal_model,
        ),
        patch(
            "src.models.baselines.baseline_models.AutoTokenizer.from_pretrained",
            return_value=tokenizer,
        ),
    ):
        model = create_baseline(
            baseline_key,
            BaselineConfig(base_model="tiny-local-t5"),
            **kwargs,
        )

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    input_ids = torch.tensor([[2, 3, 1], [4, 5, 1]])
    attention_mask = torch.ones_like(input_ids)
    labels = torch.tensor([[6, 7, 1], [8, 9, 1]])
    if baseline_key == "tinyllama_lora":
        labels = input_ids.clone()

    optimizer.zero_grad()
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    assert output.loss is not None
    assert torch.isfinite(output.loss)
    output.loss.backward()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert any(parameter.grad is not None for parameter in trainable)
    before = [parameter.detach().clone() for parameter in trainable]
    optimizer.step()
    assert any(
        not torch.equal(previous, parameter.detach())
        for previous, parameter in zip(before, trainable)
    )
