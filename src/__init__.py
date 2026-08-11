"""DAI research package."""

from importlib import import_module
from typing import Any

__version__ = "1.0.0"
__all__ = [
    "DAITransformer",
    "DAIConfig",
    "create_dai_model",
    "AbstractDomain",
    "TypeDomain",
    "IntervalDomain",
    "MonotonicityDomain",
    "TypeMonotonicityDomain",
]

_EXPORT_MODULES = {
    "DAITransformer": "src.models.dai_transformer",
    "DAIConfig": "src.models.dai_transformer",
    "create_dai_model": "src.models.dai_transformer",
    "AbstractDomain": "src.models.abstract_domains",
    "TypeDomain": "src.models.abstract_domains",
    "IntervalDomain": "src.models.abstract_domains",
    "MonotonicityDomain": "src.models.abstract_domains",
    "TypeMonotonicityDomain": "src.models.abstract_domains",
}


def __getattr__(name: str) -> Any:
    """Load optional ML exports only when requested."""
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
