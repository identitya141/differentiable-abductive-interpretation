"""Deterministic grammar structures for SCAN commands."""

from dataclasses import dataclass
import random
from typing import Any, Optional, Sequence, Tuple


@dataclass(frozen=True)
class SCANCompositionNode:
    """A constituent in a SCAN command, indexed by half-open word spans."""

    operator: str
    span: Tuple[int, int]
    left: Optional["SCANCompositionNode"] = None
    right: Optional["SCANCompositionNode"] = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


@dataclass(frozen=True)
class SCANCompositionSpec:
    """A binary composition relation extracted from a SCAN parse tree."""

    left_span: Tuple[int, int]
    right_span: Tuple[int, int]
    parent_span: Tuple[int, int]
    operator: str


class SCANParseError(ValueError):
    """Raised when a command is outside the supported SCAN grammar."""


class SCANAlignmentError(ValueError):
    """Raised when word spans cannot be mapped to tokenizer positions exactly."""


_PRIMITIVES = {"walk", "look", "run", "jump"}
_DIRECTIONS = {"left", "right"}
_MODIFIERS = {"opposite", "around"}
_REPETITIONS = {"twice", "thrice"}
_CONNECTIVES = {"and", "after"}
_NONCE_PRIMITIVES = {
    "walk": "dax",
    "look": "kiki",
    "run": "zup",
    "jump": "wif",
    "turn": "blick",
}


def parse_scan_command(command: str) -> SCANCompositionNode:
    """Parse a SCAN command into its deterministic compositional structure."""
    tokens = tuple(command.lower().split())
    if not tokens:
        raise SCANParseError("SCAN command cannot be empty")
    return _parse_command(tokens, 0)


def extract_composition_specs(root: SCANCompositionNode) -> Tuple[SCANCompositionSpec, ...]:
    """Return child-to-parent relations in bottom-up order."""
    specs = []

    def visit(node: SCANCompositionNode) -> None:
        if node.left is not None:
            visit(node.left)
        if node.right is not None:
            visit(node.right)
        if node.left is not None and node.right is not None:
            specs.append(
                SCANCompositionSpec(
                    left_span=node.left.span,
                    right_span=node.right.span,
                    parent_span=node.span,
                    operator=node.operator,
                )
            )

    visit(root)
    return tuple(specs)


def linearize_scan_command(command: str) -> str:
    """Render a SCAN command as an operator-labeled bracketed tree."""
    tokens = tuple(command.lower().split())
    root = parse_scan_command(command)

    def render(node: SCANCompositionNode) -> str:
        if node.is_leaf:
            return f"[{tokens[node.span[0]]}]"
        children = " ".join(
            render(child)
            for child in (node.left, node.right)
            if child is not None
        )
        return f"[{node.operator.upper()} {children}]"

    return render(root)


def replace_scan_primitives_with_nonce_words(command: str) -> str:
    """Replace lexical primitives while preserving operators and word spans."""
    return " ".join(
        _NONCE_PRIMITIVES.get(token, token)
        for token in command.lower().split()
    )


def align_composition_specs_to_tokens(
    command: str,
    specs: Sequence[SCANCompositionSpec],
    tokenizer: Any,
) -> Tuple[SCANCompositionSpec, ...]:
    """Map word-indexed spans to token-indexed spans with exact verification."""
    words = command.split()
    word_token_ids = [
        tokenizer.encode(word, add_special_tokens=False)
        for word in words
    ]
    flattened_ids = [token_id for token_ids in word_token_ids for token_id in token_ids]
    command_ids = tokenizer.encode(command, add_special_tokens=False)

    if flattened_ids != command_ids:
        raise SCANAlignmentError(
            "Tokenizer cannot align SCAN words independently: "
            f"{command!r} produced {command_ids}, but words produced {flattened_ids}"
        )

    boundaries = [0]
    for token_ids in word_token_ids:
        boundaries.append(boundaries[-1] + len(token_ids))

    def token_span(word_span: Tuple[int, int]) -> Tuple[int, int]:
        start, end = word_span
        if start < 0 or end > len(words) or start >= end:
            raise SCANAlignmentError(f"Invalid word span {word_span} for {command!r}")
        return boundaries[start], boundaries[end]

    return tuple(
        SCANCompositionSpec(
            left_span=token_span(spec.left_span),
            right_span=token_span(spec.right_span),
            parent_span=token_span(spec.parent_span),
            operator=spec.operator,
        )
        for spec in specs
    )


def transform_composition_specs(
    specs: Sequence[SCANCompositionSpec],
    token_count: int,
    mode: str,
    seed: str,
    corruption_probability: float = 0.0,
) -> Tuple[SCANCompositionSpec, ...]:
    """Create deterministic structural controls while preserving span lengths."""
    if not 0.0 <= corruption_probability <= 1.0:
        raise ValueError("corruption_probability must be between 0 and 1")
    if not specs:
        return tuple(specs)
    if mode not in {"shuffled", "random"}:
        if mode != "grounded":
            raise ValueError(f"Unknown composition structure mode: {mode}")

    generator = random.Random(seed)

    operator_pool = sorted({spec.operator for spec in specs} | {
        "direction", "opposite", "around", "twice", "thrice", "and", "after",
    })

    def different_operator(current: str) -> str:
        return generator.choice([operator for operator in operator_pool if operator != current])

    def random_spec(spec: SCANCompositionSpec) -> SCANCompositionSpec:
        return SCANCompositionSpec(
            left_span=spec.left_span,
            right_span=spec.right_span,
            parent_span=spec.parent_span,
            operator=different_operator(spec.operator),
        )

    if mode == "grounded":
        transformed = tuple(specs)
    elif mode == "shuffled":
        operators = [spec.operator for spec in specs]
        if len(set(operators)) > 1:
            operators = operators[1:] + operators[:1]
        else:
            operators = [different_operator(spec.operator) for spec in specs]
        transformed = tuple(
            SCANCompositionSpec(
                left_span=spec.left_span,
                right_span=spec.right_span,
                parent_span=spec.parent_span,
                operator=operator,
            )
            for spec, operator in zip(specs, operators)
        )
    else:
        transformed = tuple(random_spec(spec) for spec in specs)

    if corruption_probability == 0.0:
        return transformed
    if mode != "grounded":
        raise ValueError("percentage corruption requires grounded structure mode")

    return tuple(
        random_spec(spec) if generator.random() < corruption_probability else spec
        for spec in transformed
    )


def _parse_command(tokens: Sequence[str], offset: int) -> SCANCompositionNode:
    connective_index = _find_connective(tokens)
    if connective_index is not None:
        connective = tokens[connective_index]
        left = _parse_simple(tokens[:connective_index], offset)
        right = _parse_simple(tokens[connective_index + 1 :], offset + connective_index + 1)

        if connective == "after":
            # Children are stored in semantic execution order.  SCAN's
            # ``X after Y`` therefore executes Y (left child) before X.
            left, right = right, left

        return SCANCompositionNode(
            operator=connective,
            span=(offset, offset + len(tokens)),
            left=left,
            right=right,
        )

    return _parse_simple(tokens, offset)


def _parse_simple(tokens: Sequence[str], offset: int) -> SCANCompositionNode:
    if not tokens:
        raise SCANParseError("Missing SCAN constituent")

    if tokens[-1] in _REPETITIONS:
        child = _parse_verb(tokens[:-1], offset)
        repetition = SCANCompositionNode(
            operator=tokens[-1],
            span=(offset + len(tokens) - 1, offset + len(tokens)),
        )
        return SCANCompositionNode(
            operator=tokens[-1],
            span=(offset, offset + len(tokens)),
            left=child,
            right=repetition,
        )

    return _parse_verb(tokens, offset)


def _parse_verb(tokens: Sequence[str], offset: int) -> SCANCompositionNode:
    if len(tokens) == 1 and tokens[0] in _PRIMITIVES:
        return SCANCompositionNode(operator="primitive", span=(offset, offset + 1))

    if len(tokens) == 2 and tokens[0] == "turn" and tokens[1] in _DIRECTIONS:
        return _binary_node(tokens, offset, "direction")

    if len(tokens) == 2 and tokens[0] in _PRIMITIVES and tokens[1] in _DIRECTIONS:
        return _binary_node(tokens, offset, "direction")

    if (
        len(tokens) == 3
        and tokens[0] in _PRIMITIVES | {"turn"}
        and tokens[1] in _MODIFIERS
        and tokens[2] in _DIRECTIONS
    ):
        base = SCANCompositionNode(operator="primitive", span=(offset, offset + 1))
        modifier = SCANCompositionNode(
            operator=tokens[1],
            span=(offset + 1, offset + 2),
        )
        modified = SCANCompositionNode(
            operator=tokens[1],
            span=(offset, offset + 2),
            left=base,
            right=modifier,
        )
        direction = SCANCompositionNode(
            operator=tokens[2],
            span=(offset + 2, offset + 3),
        )
        return SCANCompositionNode(
            operator="direction",
            span=(offset, offset + 3),
            left=modified,
            right=direction,
        )

    raise SCANParseError(f"Unsupported SCAN constituent: {' '.join(tokens)}")


def _binary_node(tokens: Sequence[str], offset: int, operator: str) -> SCANCompositionNode:
    return SCANCompositionNode(
        operator=operator,
        span=(offset, offset + 2),
        left=SCANCompositionNode(operator="primitive", span=(offset, offset + 1)),
        right=SCANCompositionNode(operator=tokens[1], span=(offset + 1, offset + 2)),
    )


def _find_connective(tokens: Sequence[str]) -> Optional[int]:
    indices = [index for index, token in enumerate(tokens) if token in _CONNECTIVES]
    if len(indices) > 1:
        raise SCANParseError("SCAN command contains multiple top-level connectives")
    return indices[0] if indices else None
