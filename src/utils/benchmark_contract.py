"""Immutable fairness contract shared by DAI and publication baselines."""

from dataclasses import dataclass, replace
import random


@dataclass(frozen=True)
class BenchmarkContract:
    dataset: str
    split: str
    max_source_length: int
    max_target_length: int
    train_batch_size: int = 32
    eval_batch_size: int = 64
    train_num_workers: int = 4
    eval_num_workers: int = 0
    data_split_seed: int = 42
    tokenizer_mode: str = "dataset_atomic"
    generation_num_beams: int = 1
    generation_min_new_tokens: int = 0
    generation_length_penalty: float = 1.0
    generation_use_eos_ban: bool = False

    @property
    def generation_max_new_tokens(self) -> int:
        return self.max_target_length


_LENGTHS = {
    "scan": (64, 128),
    # Measured with the pinned dataset-atomic T5 tokenizer. Maxima are
    # COGS=743, SLOG=972, CFQ=531 target tokens; ceilings include headroom.
    "cogs": (128, 1024),
    "slog": (256, 1024),
    "cfq": (128, 640),
}

# Representation/tokenizer-specific ceilings measured by
# scripts/validate_publication_transformations.py over every publication split.
# These prevent truncation without changing the raw-data contract used by DAI
# and unmodified T5 baselines.
_BASELINE_LENGTH_OVERRIDES = {
    "tree_linearized_t5": {
        "scan": (128, 128),
        "cogs": (512, 1024),
        "slog": (512, 1024),
        "cfq": (256, 640),
    },
    # TinyLlama tokenizes SCAN actions less compactly than the dataset-atomic
    # T5 tokenizer (observed maximum: 289 tokens).
    "tinyllama_lora": {
        "scan": (64, 320),
    },
}

_DEFAULT_SPLITS = {
    "scan": "length",
    "cogs": "generalization",
    "slog": "structural_generalization",
    "cfq": "mcd1",
}


def get_benchmark_contract(dataset: str, split: str | None = None) -> BenchmarkContract:
    """Return the frozen publication contract for one benchmark setting."""
    dataset = dataset.lower().split("_", 1)[0]
    if dataset not in _LENGTHS:
        raise ValueError(f"No publication benchmark contract for {dataset!r}")
    source_length, target_length = _LENGTHS[dataset]
    return BenchmarkContract(
        dataset=dataset,
        split=split or _DEFAULT_SPLITS[dataset],
        max_source_length=source_length,
        max_target_length=target_length,
    )


def get_baseline_contract(
    dataset: str,
    baseline_type: str,
    split: str | None = None,
) -> BenchmarkContract:
    """Return the no-truncation contract for a transformed baseline.

    Batch sizes, split seeds, and decoding settings remain identical to the
    benchmark contract. Only a representation's measured token ceilings may
    differ.
    """
    contract = get_benchmark_contract(dataset, split)
    lengths = _BASELINE_LENGTH_OVERRIDES.get(baseline_type, {}).get(
        contract.dataset
    )
    if lengths is None:
        return contract
    return replace(
        contract,
        max_source_length=lengths[0],
        max_target_length=lengths[1],
    )


def apply_benchmark_contract(config):
    """Apply controlled benchmark variables to an ExperimentConfig in place."""
    contract = get_benchmark_contract(config.data.dataset, config.data.dataset_split)
    config.data.max_source_length = contract.max_source_length
    config.data.max_target_length = contract.max_target_length
    config.data.num_workers = contract.train_num_workers
    config.data.eval_num_workers = contract.eval_num_workers
    config.data.split_seed = contract.data_split_seed
    config.training.train_batch_size = contract.train_batch_size
    config.training.eval_batch_size = contract.eval_batch_size
    return contract


def paired_holdout_indices(
    dataset_size: int, validation_fraction: float, split_seed: int
) -> tuple[list[int], list[int]]:
    """Return identical train/validation indices for every model runner."""
    if dataset_size < 2:
        raise ValueError("A paired holdout requires at least two examples")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    indices = list(range(dataset_size))
    random.Random(split_seed).shuffle(indices)
    validation_size = max(1, round(dataset_size * validation_fraction))
    validation = indices[:validation_size]
    training = indices[validation_size:]
    if not training:
        raise ValueError("Training split is empty after paired holdout")
    return training, validation
