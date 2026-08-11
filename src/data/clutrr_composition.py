"""Conservative grounded composition relations for official CLUTRR rows."""

import ast
from collections import deque
from dataclasses import dataclass
import re
from typing import Any, Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class CLUTRRCompositionSpec:
    """A relation or path join grounded to half-open source word spans."""

    left_span: Tuple[int, int]
    right_span: Tuple[int, int]
    parent_span: Tuple[int, int]
    operator: str


class CLUTRRCompositionError(ValueError):
    """Raised when official CLUTRR metadata is malformed or inconsistent."""


def _parse_literal(value: Any, field: str) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError) as error:
        raise CLUTRRCompositionError(f"Invalid {field} literal") from error


def _entity_names(genders: str) -> Dict[int, str]:
    names: Dict[int, str] = {}
    for index, entry in enumerate(genders.split(",")):
        if not entry or ":" not in entry:
            raise CLUTRRCompositionError("Invalid genders metadata")
        name, _ = entry.rsplit(":", 1)
        if not name:
            raise CLUTRRCompositionError("Empty entity name in genders metadata")
        names[index] = name
    return names


def _unique_name_span(words: Sequence[str], name: str) -> Tuple[int, int] | None:
    name_words = name.split()
    normalized_words = [re.sub(r"^\W+|\W+$", "", word).casefold() for word in words]
    normalized_name = [word.casefold() for word in name_words]
    matches = [
        (start, start + len(normalized_name))
        for start in range(len(words) - len(normalized_name) + 1)
        if normalized_words[start : start + len(normalized_name)] == normalized_name
    ]
    return matches[0] if len(matches) == 1 else None


def _shortest_path(
    edges: Sequence[Tuple[int, int]], query_edge: Tuple[int, int]
) -> Tuple[int, ...]:
    adjacency: Dict[int, List[int]] = {}
    for left, right in edges:
        adjacency.setdefault(left, []).append(right)
        adjacency.setdefault(right, []).append(left)

    start, goal = query_edge
    queue = deque([(start, (start,))])
    visited = {start}
    while queue:
        node, path = queue.popleft()
        if node == goal:
            return path
        for neighbor in adjacency.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + (neighbor,)))
    raise CLUTRRCompositionError("Query endpoints are disconnected from the story graph")


def extract_clutrr_composition_specs(
    source_text: str,
    story_edges: Any,
    edge_types: Any,
    query_edge: Any,
    genders: str,
) -> Tuple[CLUTRRCompositionSpec, ...]:
    """Ground the official metadata path when all path entities are unambiguous."""
    path, names = extract_clutrr_query_path(
        story_edges, edge_types, query_edge, genders
    )

    words = source_text.split()
    spans = {node: _unique_name_span(words, names[node]) for node in path}
    if any(span is None for span in spans.values()):
        return ()

    relation_spans: List[Tuple[int, int]] = []
    specs: List[CLUTRRCompositionSpec] = []
    for left_node, right_node in zip(path, path[1:]):
        left_span = spans[left_node]
        right_span = spans[right_node]
        assert left_span is not None and right_span is not None
        parent_span = (
            min(left_span[0], right_span[0]),
            max(left_span[1], right_span[1]),
        )
        relation_spans.append(parent_span)
        specs.append(CLUTRRCompositionSpec(left_span, right_span, parent_span, "relation"))

    composed_span = relation_spans[0]
    for relation_span in relation_spans[1:]:
        parent_span = (
            min(composed_span[0], relation_span[0]),
            max(composed_span[1], relation_span[1]),
        )
        specs.append(
            CLUTRRCompositionSpec(composed_span, relation_span, parent_span, "join")
        )
        composed_span = parent_span
    return tuple(specs)


def extract_clutrr_query_path(
    story_edges: Any,
    edge_types: Any,
    query_edge: Any,
    genders: str,
) -> Tuple[Tuple[int, ...], Dict[int, str]]:
    """Parse official metadata and return its shortest query path and entities."""
    parsed_edges = _parse_literal(story_edges, "story_edges")
    parsed_types = _parse_literal(edge_types, "edge_types")
    parsed_query = _parse_literal(query_edge, "query_edge")
    if not isinstance(parsed_edges, (list, tuple)) or not all(
        isinstance(edge, (list, tuple)) and len(edge) == 2 for edge in parsed_edges
    ):
        raise CLUTRRCompositionError("story_edges must be a sequence of pairs")
    edges = [tuple(int(node) for node in edge) for edge in parsed_edges]
    if not isinstance(parsed_types, (list, tuple)) or len(parsed_types) != len(edges):
        raise CLUTRRCompositionError("edge_types must align one-to-one with story_edges")
    if not isinstance(parsed_query, (list, tuple)) or len(parsed_query) != 2:
        raise CLUTRRCompositionError("query_edge must be a pair")

    names = _entity_names(genders)
    if any(node not in names for edge in edges for node in edge):
        raise CLUTRRCompositionError("story_edges reference an unknown entity")
    query = tuple(int(node) for node in parsed_query)
    path = _shortest_path(edges, query)
    if len(path) < 2 or any(node not in names for node in path):
        raise CLUTRRCompositionError("Path references an unknown entity")
    return path, names


def align_clutrr_composition_specs_to_tokens(
    source_text: str,
    specs: Sequence[CLUTRRCompositionSpec],
    tokenizer: Any,
) -> Tuple[CLUTRRCompositionSpec, ...]:
    """Map whitespace-word spans to tokenizer spans with exact verification."""
    words = source_text.split()
    word_token_ids = [tokenizer.encode(word, add_special_tokens=False) for word in words]
    if [token for tokens in word_token_ids for token in tokens] != tokenizer.encode(
        source_text, add_special_tokens=False
    ):
        raise CLUTRRCompositionError(
            f"Tokenizer cannot align CLUTRR words independently for {source_text!r}"
        )

    boundaries = [0]
    for token_ids in word_token_ids:
        boundaries.append(boundaries[-1] + len(token_ids))

    def token_span(word_span: Tuple[int, int]) -> Tuple[int, int]:
        start, end = word_span
        if start < 0 or start >= end or end > len(words):
            raise CLUTRRCompositionError(
                f"Invalid CLUTRR word span {word_span} for {source_text!r}"
            )
        return boundaries[start], boundaries[end]

    return tuple(
        CLUTRRCompositionSpec(
            token_span(spec.left_span),
            token_span(spec.right_span),
            token_span(spec.parent_span),
            spec.operator,
        )
        for spec in specs
    )