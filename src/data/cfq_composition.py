"""Deterministic grounded relations for CFQ question-query pairs."""

from dataclasses import dataclass
import re
from typing import Any, Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class CFQCompositionSpec:
    """A SPARQL relation grounded to half-open question word spans."""

    left_span: Tuple[int, int]
    right_span: Tuple[int, int]
    parent_span: Tuple[int, int]
    operator: str


class CFQCompositionError(ValueError):
    """Raised when a grounded CFQ relation is internally inconsistent."""


_TRIPLE_PATTERN = re.compile(
    r"^(?P<left>\?x\d+|M\d+)\s+(?P<predicate>\S+)\s+"
    r"(?P<right>\?x\d+|M\d+|ns:\S+)\s*\.?$"
)
_TYPE_PATTERN = re.compile(r"^(?P<variable>\?x\d+)\s+a\s+ns:(?P<type>\S+)\s*\.?$")
_SELECT_PATTERN = re.compile(r"SELECT\s+DISTINCT\s+(?P<variable>\?x\d+)")
_WH_WORDS = {"who", "what", "which", "where", "when"}


def _unique_span(words: Sequence[str], phrase: Sequence[str]) -> Tuple[int, int] | None:
    folded_words = [word.casefold() for word in words]
    folded_phrase = [word.casefold() for word in phrase]
    starts = [
        start
        for start in range(len(words) - len(phrase) + 1)
        if folded_words[start : start + len(phrase)] == folded_phrase
    ]
    if len(starts) != 1:
        return None
    return starts[0], starts[0] + len(phrase)


def _type_phrase(type_uri: str) -> Tuple[str, ...]:
    return tuple(type_uri.rsplit(".", 1)[-1].replace("_", " ").split())


def extract_cfq_composition_specs(
    question: str,
    query: str,
) -> Tuple[CFQCompositionSpec, ...]:
    """Ground query triples whose endpoints have unique question spans."""
    if query.count("{") != query.count("}"):
        raise CFQCompositionError("Unbalanced braces in CFQ query")

    words = question.split()
    anchors: Dict[str, Tuple[int, int]] = {}
    for entity in set(re.findall(r"\bM\d+\b", query)):
        span = _unique_span(words, (entity,))
        if span is not None:
            anchors[entity] = span

    lines = [line.strip() for line in query.splitlines()]
    for line in lines:
        match = _TYPE_PATTERN.match(line)
        if match:
            span = _unique_span(words, _type_phrase(match.group("type")))
            if span is not None:
                anchors.setdefault(match.group("variable"), span)

    selected = _SELECT_PATTERN.search(query)
    wh_spans = [
        (index, index + 1)
        for index, word in enumerate(words)
        if word.casefold() in _WH_WORDS
    ]
    if selected and len(wh_spans) == 1:
        anchors.setdefault(selected.group("variable"), wh_spans[0])

    specs: List[CFQCompositionSpec] = []
    for line in lines:
        match = _TRIPLE_PATTERN.match(line)
        if not match:
            continue
        left_span = anchors.get(match.group("left"))
        right_span = anchors.get(match.group("right"))
        if left_span is None or right_span is None or left_span == right_span:
            continue
        predicate = match.group("predicate")
        specs.append(
            CFQCompositionSpec(
                left_span=left_span,
                right_span=right_span,
                parent_span=(
                    min(left_span[0], right_span[0]),
                    max(left_span[1], right_span[1]),
                ),
                operator="join" if "/" in predicate or "|" in predicate else "relation",
            )
        )
    return tuple(specs)


def align_cfq_composition_specs_to_tokens(
    question: str,
    specs: Sequence[CFQCompositionSpec],
    tokenizer: Any,
) -> Tuple[CFQCompositionSpec, ...]:
    """Map verified CFQ word spans to tokenizer spans."""
    words = question.split()
    word_token_ids = [tokenizer.encode(word, add_special_tokens=False) for word in words]
    if [token for ids in word_token_ids for token in ids] != tokenizer.encode(
        question, add_special_tokens=False
    ):
        raise CFQCompositionError(f"Tokenizer cannot align CFQ words for {question!r}")
    boundaries = [0]
    for token_ids in word_token_ids:
        boundaries.append(boundaries[-1] + len(token_ids))

    def token_span(span: Tuple[int, int]) -> Tuple[int, int]:
        start, end = span
        if start < 0 or start >= end or end > len(words):
            raise CFQCompositionError(f"Invalid CFQ word span {span}")
        return boundaries[start], boundaries[end]

    return tuple(
        CFQCompositionSpec(
            left_span=token_span(spec.left_span),
            right_span=token_span(spec.right_span),
            parent_span=token_span(spec.parent_span),
            operator=spec.operator,
        )
        for spec in specs
    )