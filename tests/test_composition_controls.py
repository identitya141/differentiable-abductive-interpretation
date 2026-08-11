from dataclasses import dataclass

from src.data.composition_controls import transform_aligned_specs


@dataclass(frozen=True)
class Spec:
    left_span: tuple[int, int]
    right_span: tuple[int, int]
    parent_span: tuple[int, int]
    operator: str


SPECS = (
    Spec((0, 1), (2, 3), (0, 3), "agent"),
    Spec((1, 2), (3, 4), (1, 4), "theme"),
)


def test_grounded_control_preserves_specs():
    assert transform_aligned_specs(SPECS, token_count=6, mode="grounded", seed="x") == SPECS


def test_shuffled_control_preserves_valid_tree_and_changes_semantics():
    transformed = transform_aligned_specs(SPECS, token_count=6, mode="shuffled", seed="x")
    assert [spec.left_span for spec in transformed] == [spec.left_span for spec in SPECS]
    assert [spec.right_span for spec in transformed] == [spec.right_span for spec in SPECS]
    assert [spec.parent_span for spec in transformed] == [spec.parent_span for spec in SPECS]
    assert [spec.operator for spec in transformed] == ["theme", "agent"]
    assert all(isinstance(spec, Spec) for spec in transformed)


def test_random_control_is_deterministic_and_in_bounds():
    first = transform_aligned_specs(SPECS, token_count=6, mode="random", seed="paired-seed")
    second = transform_aligned_specs(SPECS, token_count=6, mode="random", seed="paired-seed")
    assert first == second
    assert first != SPECS
    for original, spec in zip(SPECS, first):
        assert spec.left_span == original.left_span
        assert spec.right_span == original.right_span
        assert spec.parent_span == original.parent_span
        assert spec.operator != original.operator
        for span in (spec.left_span, spec.right_span, spec.parent_span):
            assert 0 <= span[0] < span[1] <= 6
