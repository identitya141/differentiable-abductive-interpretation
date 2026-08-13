"""
Generation Utilities for Compositional Generalization

Provides optimized generation strategies for SCAN/COGS/CFQ tasks,
including EOS-ban processors and length-aware decoding.
"""

import torch
from transformers import LogitsProcessor, LogitsProcessorList
from typing import Optional, List, Union


# =============================================================================
# SCAN Action Tokens (for atomic tokenization)
# =============================================================================

SCAN_ACTION_TOKENS = [
    "I_WALK", "I_RUN", "I_JUMP", "I_LOOK", 
    "I_TURN_LEFT", "I_TURN_RIGHT"
]


def add_scan_action_tokens(tokenizer, model):
    """
    Add SCAN action tokens as *regular vocab tokens* for atomic tokenization.
    
    This reduces early EOS by making each action a single token instead
    of multiple subword pieces.
    
    IMPORTANT: We use add_tokens() NOT add_special_tokens() because
    special tokens get stripped when decode(skip_special_tokens=True),
    which breaks metrics by producing empty strings.
    
    Args:
        tokenizer: HuggingFace tokenizer
        model: HuggingFace model (will resize embeddings)
        
    Returns:
        Number of tokens added
    """
    # Do NOT use add_special_tokens - those get stripped during decoding!
    num_added = tokenizer.add_tokens(SCAN_ACTION_TOKENS, special_tokens=False)
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))
    return num_added


def is_atomic_scan(tokenizer) -> bool:
    """
    Check if tokenizer has SCAN action tokens as atomic (single) tokens.
    
    This is used to dynamically configure generation parameters:
    - Atomic mode: shorter sequences (~25-40 tokens), less EOS-ban needed
    - Subword mode: longer sequences (~120-130 tokens), needs EOS-ban
    
    Args:
        tokenizer: HuggingFace tokenizer
        
    Returns:
        True if all SCAN action tokens are single tokens in vocab
    """
    for token in SCAN_ACTION_TOKENS:
        ids = tokenizer.encode(token, add_special_tokens=False)
        if len(ids) != 1:
            return False
    return True


# =============================================================================
# EOS-Ban LogitsProcessor
# =============================================================================

class MinNewTokensEosBan(LogitsProcessor):
    """
    Blocks EOS token until minimum number of new tokens have been generated.
    
    This fixes the early EOS problem observed in SCAN length split where
    the model produces correct prefixes but terminates prematurely.
    """
    
    def __init__(self, eos_token_id: int, start_len: int, min_new_tokens: int):
        """
        Args:
            eos_token_id: Token ID of EOS
            start_len: Initial length of decoder input (typically 1 for T5)
            min_new_tokens: Minimum tokens before EOS is allowed
        """
        self.eos_token_id = eos_token_id
        self.start_len = start_len
        self.min_new_tokens = min_new_tokens

    def __call__(
        self, 
        input_ids: torch.LongTensor, 
        scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Block EOS if we haven't generated enough tokens yet."""
        cur_len = input_ids.shape[1]
        new_tokens = cur_len - self.start_len
        
        if new_tokens < self.min_new_tokens:
            scores[:, self.eos_token_id] = float("-inf")
        
        return scores


class EosLogitsBias(LogitsProcessor):
    """
    Apply a constant bias to EOS logits to discourage early termination.
    
    Negative bias discourages EOS, positive encourages it.
    """
    
    def __init__(self, eos_token_id: int, bias: float = -2.0):
        """
        Args:
            eos_token_id: Token ID of EOS
            bias: Logit bias (negative = discourage EOS)
        """
        self.eos_token_id = eos_token_id
        self.bias = bias

    def __call__(
        self, 
        input_ids: torch.LongTensor, 
        scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Apply bias to EOS logits."""
        scores[:, self.eos_token_id] += self.bias
        return scores


# =============================================================================
# Optimized Generation Functions
# =============================================================================

def generate_scan_optimized(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int = 256,
    min_new_tokens: Optional[int] = None,
    num_beams: int = 8,
    length_penalty: float = 1.2,
    eos_bias: float = 0.0,
    use_eos_ban: bool = True,
    **kwargs,
) -> torch.Tensor:
    """
    Optimized generation for SCAN dataset with EOS-ban and length control.
    
    Addresses the early EOS problem by:
    1. Blocking EOS until min_new_tokens are generated
    2. Using beam search with length penalty
    3. Optionally applying EOS logit bias
    
    Args:
        model: DAITransformer or T5 model
        tokenizer: T5Tokenizer
        input_ids: Encoder input IDs [batch, seq_len]
        attention_mask: Encoder attention mask [batch, seq_len]
        max_new_tokens: Maximum tokens to generate
        min_new_tokens: Minimum tokens before EOS allowed (default: 0.85 * max_new_tokens)
        num_beams: Number of beams for beam search
        length_penalty: Length penalty (>1 = prefer longer)
        eos_bias: Bias to add to EOS logits (negative = discourage)
        use_eos_ban: Whether to use EOS-ban processor
        **kwargs: Additional generate() arguments
        
    Returns:
        Generated token IDs [batch, seq_len]
    """
    # Safe default: no EOS-ban unless explicitly requested via config.
    # The old default of 0.85*max was dangerous (forced 217 tokens for max=256).
    if min_new_tokens is None:
        min_new_tokens = 0
    
    # Build logits processors
    processors = LogitsProcessorList()
    
    # Decoder start length for T5 is typically 1 (decoder_start_token)
    start_len = 1
    
    if use_eos_ban and min_new_tokens > 0:
        processors.append(MinNewTokensEosBan(
            eos_token_id=tokenizer.eos_token_id,
            start_len=start_len,
            min_new_tokens=min_new_tokens,
        ))
    
    if eos_bias != 0.0:
        processors.append(EosLogitsBias(
            eos_token_id=tokenizer.eos_token_id,
            bias=eos_bias,
        ))
    
    # Generate with optimized settings
    return model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        length_penalty=length_penalty,
        early_stopping=False,  # Don't stop early with EOS-ban
        logits_processor=processors if len(processors) > 0 else None,
        **kwargs,
    )


def generate_with_length_hint(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_length: int,
    length_tolerance: int = 10,
    num_beams: int = 8,
    length_penalty: float = 1.2,
    **kwargs,
) -> torch.Tensor:
    """
    Generate with a specific target length hint.
    
    Uses min/max constraints to guide generation toward target length.
    
    Args:
        model: DAITransformer or T5 model
        tokenizer: T5Tokenizer
        input_ids: Encoder input IDs
        attention_mask: Encoder attention mask
        target_length: Expected output length in tokens
        length_tolerance: Allowed deviation from target
        num_beams: Number of beams
        length_penalty: Length penalty
        **kwargs: Additional generate() arguments
        
    Returns:
        Generated token IDs
    """
    min_new_tokens = max(1, target_length - length_tolerance)
    max_new_tokens = target_length + length_tolerance * 2
    
    return generate_scan_optimized(
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        num_beams=num_beams,
        length_penalty=length_penalty,
        **kwargs,
    )


# =============================================================================
# Dataset-specific generation configs
# =============================================================================

GENERATION_CONFIGS = {
    "scan": {
        # Base config - will be adjusted by get_generation_config() based on
        # whether tokenizer uses atomic tokens or subwords
        "max_new_tokens": 256,
        "min_new_tokens": 0,  # Safe default, overridden per mode below
        # Mode-specific settings (selected dynamically):
        "min_new_tokens_atomic": 0,      # Atomic: ~25-40 token outputs, no ban needed
        "min_new_tokens_subword": 80,    # Subword: ~120-130 tokens, ban prevents early EOS
        "max_new_tokens_atomic": 80,     # Atomic: keep tight for speed
        "max_new_tokens_subword": 256,   # Subword: need full length
        "num_beams": 4,  # Reduced from 8 for faster training eval
        "num_beams_final": 8,  # Use this for final evaluation/reporting
        "length_penalty": 1.2,
        "use_eos_ban": True,
    },
    "cogs": {
        "max_new_tokens": 256,
        "min_new_tokens": 50,
        "num_beams": 4,
        "length_penalty": 1.0,
        "use_eos_ban": True,
    },
    "cfq": {
        "max_new_tokens": 512,
        "min_new_tokens": 100,
        "num_beams": 4,
        "length_penalty": 1.0,
        "use_eos_ban": True,
    },
    "gsm8k": {
        "max_new_tokens": 256,
        "min_new_tokens": 20,
        "num_beams": 1,  # Greedy for math
        "length_penalty": 1.0,
        "use_eos_ban": False,  # Math needs natural stopping
    },
    "clutrr": {
        "max_new_tokens": 64,
        "min_new_tokens": 1,
        "num_beams": 4,
        "length_penalty": 1.0,
        "use_eos_ban": False,  # Short answers
    },
}


def get_generation_config(dataset_type: str, tokenizer=None, final_eval: bool = False) -> dict:
    """
    Get optimized generation config for a dataset type.
    
    For SCAN, dynamically selects atomic vs subword mode based on tokenizer.
    
    Args:
        dataset_type: Dataset type (e.g., "scan", "scan_length", "cogs")
        tokenizer: Optional tokenizer to detect atomic mode for SCAN
        final_eval: If True, use higher beam count for final reporting
        
    Returns:
        Dictionary of generation parameters
    """
    dataset_type = dataset_type.lower().split("_")[0]  # "scan_length" -> "scan"
    if dataset_type in {"scan", "cogs", "slog", "cfq"}:
        from src.utils.benchmark_contract import get_benchmark_contract

        contract = get_benchmark_contract(dataset_type)
        return {
            "max_new_tokens": contract.generation_max_new_tokens,
            "min_new_tokens": contract.generation_min_new_tokens,
            "num_beams": contract.generation_num_beams,
            "length_penalty": contract.generation_length_penalty,
            "use_eos_ban": contract.generation_use_eos_ban,
        }
    config = GENERATION_CONFIGS.get(dataset_type, GENERATION_CONFIGS["scan"]).copy()
    
    # SCAN: dynamically select atomic vs subword config
    if dataset_type == "scan" and tokenizer is not None:
        if is_atomic_scan(tokenizer):
            # Atomic mode: short sequences, no EOS-ban needed
            config["min_new_tokens"] = config.get("min_new_tokens_atomic", 0)
            config["max_new_tokens"] = config.get("max_new_tokens_atomic", 80)
            config["num_beams"] = 1  # Greedy is fast and works well for atomic
        else:
            # Subword mode: long sequences, need EOS-ban
            config["min_new_tokens"] = config.get("min_new_tokens_subword", 80)
            config["max_new_tokens"] = config.get("max_new_tokens_subword", 256)
    
    # Use higher beam count for final evaluation/reporting
    if final_eval and "num_beams_final" in config:
        config["num_beams"] = config["num_beams_final"]
    
    # Clean up mode-specific keys that shouldn't be passed to generate()
    for key in ["min_new_tokens_atomic", "min_new_tokens_subword",
                "max_new_tokens_atomic", "max_new_tokens_subword", "num_beams_final"]:
        config.pop(key, None)
    
    return config


def apply_generation_contract(model, dataset_type: str, tokenizer=None) -> dict:
    """Apply the shared max-new-token decoding contract to an HF model."""
    config = get_generation_config(dataset_type, tokenizer=tokenizer)
    generation_config = model.generation_config
    generation_config.max_new_tokens = config["max_new_tokens"]
    generation_config.max_length = None
    generation_config.min_new_tokens = config["min_new_tokens"]
    generation_config.num_beams = config["num_beams"]
    generation_config.length_penalty = config["length_penalty"]
    return config
