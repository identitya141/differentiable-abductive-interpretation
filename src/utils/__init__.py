"""Utilities package - Configuration, reproducibility, and helpers."""

from importlib import import_module
from typing import Any

__all__ = [
    "set_seed",
    "get_reproducibility_info",
    "load_config",
    "merge_configs",
    "extend_tokenizer_for_dataset",
    "get_dataset_special_tokens",
    "create_scan_tokenizer",
    "SCAN_ACTION_TOKENS",
]

_EXPORT_MODULES = {
    "set_seed": "src.utils.reproducibility",
    "get_reproducibility_info": "src.utils.reproducibility",
    "load_config": "src.utils.config",
    "merge_configs": "src.utils.config",
    "extend_tokenizer_for_dataset": "src.utils.tokenizer_utils",
    "get_dataset_special_tokens": "src.utils.tokenizer_utils",
    "create_scan_tokenizer": "src.utils.tokenizer_utils",
    "SCAN_ACTION_TOKENS": "src.utils.tokenizer_utils",
}


def __getattr__(name: str) -> Any:
    """Load utilities and their optional dependencies only when requested."""
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
