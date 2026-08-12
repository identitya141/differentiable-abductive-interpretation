"""Shared scheduler-name boundary helpers."""

HF_SCHEDULER_ALIASES = {"cosine_with_warmup": "cosine"}


def hf_scheduler_name(name: str) -> str:
    """Translate project scheduler names at the Hugging Face boundary."""
    return HF_SCHEDULER_ALIASES.get(name, name)
