"""Immutable fairness contract shared by DAI and publication baselines."""

from dataclasses import dataclass
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
    "cogs": (128, 256),
    "slog": (256, 512),
    "cfq": (128, 256),
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
