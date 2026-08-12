import torch
from transformers import T5Config, T5ForConditionalGeneration

from src.utils.tokenizer_utils import resize_with_deterministic_added_token_init


def test_added_token_rows_are_paired_despite_prior_rng_consumption():
    config = T5Config(
        vocab_size=32, d_model=8, d_ff=16, num_layers=1, num_heads=2, d_kv=4,
        pad_token_id=0, eos_token_id=1, decoder_start_token_id=0,
    )
    torch.manual_seed(5)
    first = T5ForConditionalGeneration(config)
    second = T5ForConditionalGeneration(config)
    second.load_state_dict(first.state_dict())
    _ = torch.randn(1000)
    resize_with_deterministic_added_token_init(first, 38, seed=42)
    _ = torch.randn(2000)
    resize_with_deterministic_added_token_init(second, 38, seed=42)
    assert torch.equal(first.shared.weight[32:], second.shared.weight[32:])
