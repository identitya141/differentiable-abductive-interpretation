"""Dataset-agnostic corruption controls for aligned composition specifications."""

from __future__ import annotations

import random
from typing import Any, Sequence, Tuple


def transform_aligned_specs(
    specs: Sequence[Any],
    *,
    token_count: int,
    mode: str,
    seed: str,
) -> Tuple[Any, ...]:
    if mode not in {"grounded", "random", "shuffled"}:
        raise ValueError(f"Unknown composition structure mode: {mode}")
    if mode == "grounded" or not specs:
        return tuple(specs)
    generator = random.Random(seed)

    for spec in specs:
        for span in (spec.left_span, spec.right_span, spec.parent_span):
            if not 0 <= span[0] < span[1] <= token_count:
                raise ValueError("Composition span is outside the tokenized input")

    operator_pool = sorted({spec.operator for spec in specs} | {
        "direction", "opposite", "around", "twice", "thrice", "and", "after",
        "agent", "theme", "recipient", "ccomp", "xcomp", "nmod", "relation", "join",
    })

    def different_operator(current: str) -> str:
        choices = [operator for operator in operator_pool if operator != current]
        return generator.choice(choices)

    if mode == "shuffled":
        operators = [spec.operator for spec in specs]
        if len(set(operators)) > 1:
            operators = operators[1:] + operators[:1]
        else:
            operators = [different_operator(spec.operator) for spec in specs]
        return tuple(
            type(spec)(
                left_span=spec.left_span,
                right_span=spec.right_span,
                parent_span=spec.parent_span,
                operator=operator,
            )
            for spec, operator in zip(specs, operators)
        )
    return tuple(
        type(spec)(
            left_span=spec.left_span,
            right_span=spec.right_span,
            parent_span=spec.parent_span,
            operator=different_operator(spec.operator),
        )
        for spec in specs
    )
