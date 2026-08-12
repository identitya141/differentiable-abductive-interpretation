"""
Tokenizer utilities for dataset-specific token handling.

For SCAN and similar compositional tasks, adding action tokens as regular vocab
tokens (NOT special tokens) dramatically improves generation quality by:
1. Making each action atomic (1 token instead of 5-7 subwords)
2. Eliminating beam search hallucinations like "I_TURN_RUN"
3. Reducing sequence length and EOS pressure

IMPORTANT: We use tokenizer.add_tokens() NOT add_special_tokens() because
special tokens get stripped when decode(skip_special_tokens=True), which
breaks metrics by producing empty strings.
"""

from contextlib import contextmanager
from typing import List, Optional, Tuple
import random

import numpy as np
import torch
from transformers import PreTrainedTokenizer, T5Tokenizer


# SCAN action vocabulary - these should be single tokens
SCAN_ACTION_TOKENS = [
    "I_WALK",
    "I_RUN", 
    "I_JUMP",
    "I_LOOK",
    "I_TURN_LEFT",
    "I_TURN_RIGHT",
]

# COGS logical form tokens (common predicates/functions)
COGS_SPECIAL_TOKENS = [
    "lambda", "AND", "agent", "theme", "recipient", "goal", "source",
    "ccomp", "xcomp", "nmod.on", "nmod.in", "nmod.beside", "nmod.to",
]

# CFQ SPARQL tokens
CFQ_SPECIAL_TOKENS = [
    "SELECT", "WHERE", "FILTER", "OPTIONAL", "DISTINCT",
    "?x0", "?x1", "?x2", "?x3", "?x4", "?x5",
    "M0", "M1", "M2", "M3", "M4", "M5",
]


def get_dataset_special_tokens(dataset_type: str) -> List[str]:
    """
    Get special tokens for a specific dataset type.
    
    Args:
        dataset_type: One of "scan", "cogs", "cfq"
        
    Returns:
        List of special tokens to add
    """
    dataset_type = dataset_type.lower().split("_")[0]  # "scan_length" -> "scan"
    
    if dataset_type == "scan":
        return SCAN_ACTION_TOKENS
    elif dataset_type in {"cogs", "slog"}:
        return COGS_SPECIAL_TOKENS
    elif dataset_type == "cfq":
        return CFQ_SPECIAL_TOKENS
    else:
        return []


def extend_tokenizer_for_dataset(
    tokenizer: PreTrainedTokenizer,
    dataset_type: str,
    verbose: bool = True,
) -> Tuple[PreTrainedTokenizer, int]:
    """
    Extend tokenizer with dataset-specific special tokens.
    
    Args:
        tokenizer: Base tokenizer (e.g., T5Tokenizer)
        dataset_type: Dataset type for selecting tokens
        verbose: Whether to print token info
        
    Returns:
        Tuple of (extended tokenizer, number of tokens added)
    """
    special_tokens = get_dataset_special_tokens(dataset_type)
    
    if not special_tokens:
        if verbose:
            print(f"No special tokens defined for dataset: {dataset_type}")
        return tokenizer, 0
    
    # Check which tokens are not already in vocabulary
    tokens_to_add = []
    for token in special_tokens:
        # Check if token exists as single token
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if len(encoded) > 1:
            tokens_to_add.append(token)
            if verbose:
                print(f"  '{token}' currently encodes to {len(encoded)} tokens: {encoded}")
    
    if not tokens_to_add:
        if verbose:
            print(f"All {len(special_tokens)} tokens already atomic in vocabulary")
        return tokenizer, 0
    
    # Add tokens as REGULAR vocab tokens, NOT special tokens.
    # CRITICAL: add_special_tokens() would make skip_special_tokens=True
    # strip them during decoding, breaking metrics (empty strings = false 100% EM).
    num_added = tokenizer.add_tokens(tokens_to_add, special_tokens=False)
    
    if verbose:
        print(f"Added {num_added} vocab tokens (NOT special tokens):")
        for token in tokens_to_add:
            token_id = tokenizer.convert_tokens_to_ids(token)
            print(f"  '{token}' -> {token_id}")
        print(f"New vocabulary size: {len(tokenizer)}")
    
    return tokenizer, num_added


@contextmanager
def _preserve_rng_state():
    """Preserve every RNG touched by model embedding initialization."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def resize_with_deterministic_added_token_init(
    model,
    new_num_tokens: int,
    *,
    seed: int,
):
    """Resize embeddings using a dedicated paired seed, then restore RNG state.

    This makes newly added dataset-token rows identical for paired DAI and T5
    runs even when DAI constructed auxiliary modules before the resize.
    """
    get_embeddings = getattr(model, "get_input_embeddings", None)
    if get_embeddings is None and hasattr(model, "t5"):
        get_embeddings = model.t5.get_input_embeddings
    if get_embeddings is None and hasattr(model, "_hf_model"):
        get_embeddings = model._hf_model.get_input_embeddings
    if get_embeddings is None:
        raise TypeError("Model does not expose input embeddings")
    old_num_tokens = int(get_embeddings().weight.shape[0])
    if new_num_tokens <= old_num_tokens:
        return model.resize_token_embeddings(new_num_tokens)
    with _preserve_rng_state():
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return model.resize_token_embeddings(new_num_tokens)


def verify_tokenization(
    tokenizer: PreTrainedTokenizer,
    dataset_type: str,
) -> bool:
    """
    Verify that special tokens are correctly tokenized as single tokens.
    
    Args:
        tokenizer: Tokenizer to verify
        dataset_type: Dataset type
        
    Returns:
        True if all tokens are atomic
    """
    special_tokens = get_dataset_special_tokens(dataset_type)
    all_atomic = True
    
    for token in special_tokens:
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if len(encoded) != 1:
            print(f"WARNING: '{token}' not atomic, encodes to {len(encoded)} tokens")
            all_atomic = False
    
    return all_atomic


def create_scan_tokenizer(
    base_model: str = "google-t5/t5-small",
    verbose: bool = True,
) -> T5Tokenizer:
    """
    Create a T5 tokenizer extended with SCAN action tokens.
    
    Args:
        base_model: Base model name
        verbose: Whether to print info
        
    Returns:
        Extended tokenizer
    """
    if verbose:
        print(f"Creating SCAN tokenizer from {base_model}")
        print(f"Adding {len(SCAN_ACTION_TOKENS)} action tokens as special tokens")
    
    tokenizer = T5Tokenizer.from_pretrained(base_model)
    
    if verbose:
        print(f"Original vocabulary size: {len(tokenizer)}")
        print("\nBefore extension:")
        for token in SCAN_ACTION_TOKENS:
            encoded = tokenizer.encode(token, add_special_tokens=False)
            decoded_pieces = [tokenizer.decode([t]) for t in encoded]
            print(f"  '{token}' -> {len(encoded)} tokens: {decoded_pieces}")
    
    tokenizer, num_added = extend_tokenizer_for_dataset(
        tokenizer, "scan", verbose=verbose
    )
    
    if verbose:
        print("\nAfter extension:")
        for token in SCAN_ACTION_TOKENS:
            encoded = tokenizer.encode(token, add_special_tokens=False)
            print(f"  '{token}' -> {len(encoded)} token(s): {encoded}")
    
    return tokenizer


# Example usage / verification
if __name__ == "__main__":
    print("=" * 60)
    print("Testing SCAN tokenizer extension")
    print("=" * 60)
    
    tokenizer = create_scan_tokenizer()
    
    print("\n" + "=" * 60)
    print("Testing example sequence")
    print("=" * 60)
    
    test_seq = "I_TURN_LEFT I_WALK I_TURN_LEFT I_WALK I_TURN_RIGHT I_JUMP"
    tokens = tokenizer.encode(test_seq, add_special_tokens=False)
    print(f"\nSequence: {test_seq}")
    print(f"Tokens: {tokens}")
    print(f"Decoded: {[tokenizer.decode([t]) for t in tokens]}")
    print(f"Length: {len(tokens)} tokens (should be 6 with atomic actions)")
