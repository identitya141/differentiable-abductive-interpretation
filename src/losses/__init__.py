"""Losses package - Abstraction and compositional losses."""

from importlib import import_module
from typing import Any

__all__ = [
    "AbstractionLoss",
    "AbstractionLossOutput",
    "CompositionAwareAbstractionLoss",
    "HierarchicalAbstractionLoss",
    "OverConstraintDetector",
    "SubConstituentLoss",
]


def __getattr__(name: str) -> Any:
    """Load loss implementations only when requested."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module("src.losses.abstraction_loss"), name)
    globals()[name] = value
    return value
