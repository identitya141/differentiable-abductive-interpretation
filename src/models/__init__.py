"""Models package - DAI transformer and abstract domains."""

from importlib import import_module
from typing import Any

__all__ = [
    # Domains
    "AbstractDomain",
    "TypeDomain",
    "IntervalDomain",
    "MonotonicityDomain",
    "TypeMonotonicityDomain",
    "get_abstract_domain",
    # Layers
    "AbstractionLayer",
    "MultiLayerAbstractionModule",
    "AbstractionScheduler",
    # Models
    "DAIConfig",
    "DAIModelOutput",
    "DAITransformer",
    "DAITransformerForSequenceClassification",
    "create_dai_model",
]

_DOMAIN_EXPORTS = {
    "AbstractDomain",
    "TypeDomain",
    "IntervalDomain",
    "MonotonicityDomain",
    "TypeMonotonicityDomain",
    "get_abstract_domain",
}
_LAYER_EXPORTS = {
    "AbstractionLayer",
    "MultiLayerAbstractionModule",
    "AbstractionScheduler",
}
_MODEL_EXPORTS = {
    "DAIConfig",
    "DAIModelOutput",
    "DAITransformer",
    "DAITransformerForSequenceClassification",
    "create_dai_model",
}


def __getattr__(name: str) -> Any:
    """Load model components only when requested."""
    if name in _DOMAIN_EXPORTS:
        module_name = "src.models.abstract_domains"
    elif name in _LAYER_EXPORTS:
        module_name = "src.models.abstraction_layer"
    elif name in _MODEL_EXPORTS:
        module_name = "src.models.dai_transformer"
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
