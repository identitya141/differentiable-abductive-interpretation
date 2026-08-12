from dataclasses import dataclass

from src.data.composition_controls import transform_aligned_specs
from src.data.scan_composition import SCANCompositionSpec, transform_composition_specs


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


def test_random_control_never_crosses_dataset_operator_vocabulary():
    transformed = transform_aligned_specs(
        SPECS, token_count=6, mode="random", seed="paired-seed", dataset="cogs"
    )
    allowed = {"agent", "theme", "recipient", "ccomp", "xcomp", "nmod"}
    assert {spec.operator for spec in transformed} <= allowed
    assert not {"twice", "join"} & {spec.operator for spec in transformed}


def test_topology_control_preserves_marginals_but_changes_assignments():
    specs = (
        SCANCompositionSpec((0, 1), (1, 2), (0, 2), "twice"),
        SCANCompositionSpec((2, 3), (3, 4), (2, 4), "and"),
    )
    transformed = transform_composition_specs(
        specs, token_count=4, mode="topology", seed="paired"
    )
    assert transformed != specs
    assert [x.parent_span for x in transformed] == [x.parent_span for x in specs]
    assert [x.operator for x in transformed] == [x.operator for x in specs]
    assert sorted((x.left_span, x.right_span) for x in transformed) == sorted(
        (x.left_span, x.right_span) for x in specs
    )
