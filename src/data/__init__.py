"""Data package - Dataset loaders for compositional generalization benchmarks."""

from importlib import import_module
from typing import Any

__all__ = [
    "BaseCompositionalDataset",
    "SCANDataset",
    "COGSDataset",
    "SLOGDataset",
    "CFQDataset",
    "CLUTRRDataset",
    "GSM8KDataset",
]

_EXPORT_MODULES = {
    "BaseCompositionalDataset": "src.data.base_dataset",
    "SCANDataset": "src.data.scan_dataset",
    "COGSDataset": "src.data.cogs_dataset",
    "SLOGDataset": "src.data.slog_dataset",
    "CFQDataset": "src.data.cfq_dataset",
    "CLUTRRDataset": "src.data.clutrr_dataset",
    "GSM8KDataset": "src.data.gsm8k_dataset",
}


def __getattr__(name: str) -> Any:
    """Load dataset implementations only when requested."""
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
