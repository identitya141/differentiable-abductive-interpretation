"""Deterministic grounded composition relations for COGS logical forms."""

from dataclasses import dataclass
import re
from typing import Any, Sequence, Tuple


@dataclass(frozen=True)
class COGSCompositionSpec:
    """A semantic-role relation grounded to half-open sentence word spans."""

    left_span: Tuple[int, int]
    right_span: Tuple[int, int]
    parent_span: Tuple[int, int]
    operator: str


class COGSCompositionError(ValueError):
    """Raised when a COGS logical form cannot be grounded to its sentence."""


_ROLE_PATTERN = re.compile(
    r"[A-Za-z][A-Za-z0-9_-]*\s*\.\s*"
    r"(?:(?P<nmod>nmod)\s*\.\s*[A-Za-z][A-Za-z0-9_-]*|"
    r"(?P<role>agent|theme|recipient|ccomp|xcomp))\s*"
    r"\(\s*(?P<left>x\s*_\s*\d+|[A-Za-z][A-Za-z0-9_-]*)\s*,\s*"
    r"(?P<right>x\s*_\s*\d+|[A-Za-z][A-Za-z0-9_-]*)\s*\)"
)
_VARIABLE_PATTERN = re.compile(r"^x\s*_\s*(\d+)$")


def extract_cogs_composition_specs(
    sentence: str,
    logical_form: str,
) -> Tuple[COGSCompositionSpec, ...]:
    """Extract semantic-role edges using COGS token-index variables."""
    if logical_form.count("(") != logical_form.count(")"):
        raise COGSCompositionError("Unbalanced parentheses in COGS logical form")
    if logical_form.lstrip().startswith("LAMBDA "):
        return ()

    words = sentence.split()

    def argument_span(argument: str) -> Tuple[int, int]:
        variable = _VARIABLE_PATTERN.match(argument)
        if variable:
            index = int(variable.group(1))
            if index >= len(words):
                raise COGSCompositionError(
                    f"Variable {argument!r} is outside {len(words)} sentence tokens"
                )
            return index, index + 1

        matches = [
            index
            for index, word in enumerate(words)
            if word.casefold() == argument.casefold()
        ]
        if len(matches) != 1:
            raise COGSCompositionError(
                f"Named argument {argument!r} has {len(matches)} sentence matches"
            )
        return matches[0], matches[0] + 1

    specs = []
    for match in _ROLE_PATTERN.finditer(logical_form):
        left_span = argument_span(match.group("left"))
        right_span = argument_span(match.group("right"))
        parent_span = (
            min(left_span[0], right_span[0]),
            max(left_span[1], right_span[1]),
        )
        specs.append(
            COGSCompositionSpec(
                left_span=left_span,
                right_span=right_span,
                parent_span=parent_span,
                operator="nmod" if match.group("nmod") else match.group("role"),
            )
        )

    return tuple(specs)


def align_cogs_composition_specs_to_tokens(
    sentence: str,
    specs: Sequence[COGSCompositionSpec],
    tokenizer: Any,
) -> Tuple[COGSCompositionSpec, ...]:
    """Map COGS word spans to tokenizer spans with exact verification."""
    words = sentence.split()
    word_token_ids = [
        tokenizer.encode(word, add_special_tokens=False) for word in words
    ]
    flattened_ids = [token for tokens in word_token_ids for token in tokens]
    sentence_ids = tokenizer.encode(sentence, add_special_tokens=False)
    if flattened_ids != sentence_ids:
        raise COGSCompositionError(
            f"Tokenizer cannot align COGS words independently for {sentence!r}"
        )

    boundaries = [0]
    for token_ids in word_token_ids:
        boundaries.append(boundaries[-1] + len(token_ids))

    def token_span(word_span: Tuple[int, int]) -> Tuple[int, int]:
        start, end = word_span
        if start < 0 or start >= end or end > len(words):
            raise COGSCompositionError(
                f"Invalid COGS word span {word_span} for {sentence!r}"
            )
        return boundaries[start], boundaries[end]

    return tuple(
        COGSCompositionSpec(
            left_span=token_span(spec.left_span),
            right_span=token_span(spec.right_span),
            parent_span=token_span(spec.parent_span),
            operator=spec.operator,
        )
        for spec in specs
    )
