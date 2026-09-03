"""Tests for the canonical cross-method evaluation normalizer."""

from src.evaluation.metrics import (
    compute_exact_match_accuracy,
    normalize_batch_for_eval,
    normalize_for_eval,
)


def test_scan_normalization_is_case_and_whitespace_invariant():
    assert normalize_for_eval("  i_walk   i_jump ", "scan") == "I_WALK I_JUMP"
    assert compute_exact_match_accuracy(
        ["i_walk   i_jump"], ["I_WALK I_JUMP"], dataset_type="scan"
    ) == 1.0


def test_scan_normalization_repairs_adjacent_added_tokens():
    assert normalize_for_eval("I_TURN_LEFTI_RUNI_RUN", "scan") == (
        "I_TURN_LEFT I_RUN I_RUN"
    )
    assert compute_exact_match_accuracy(
        ["I_TURN_LEFTI_RUN"], ["I_TURN_LEFT I_RUN"], dataset_type="scan"
    ) == 1.0


def test_scan_normalization_preserves_unknown_residue():
    assert normalize_for_eval("I_WALK nonsense I_JUMP", "scan") == (
        "I_WALK NONSENSE I_JUMP"
    )


def test_scratchpad_normalization_removes_decoded_special_tokens_after_answer():
    prediction = "<pad> [SCRATCH] think [/SCRATCH] I_JUMP </s> <pad> <pad>"
    target = "[/SCRATCH] I_JUMP </s> <pad>"
    assert normalize_for_eval(prediction, "scan", "scratchpad") == "I_JUMP"
    assert normalize_for_eval(target, "scan", "scratchpad") == "I_JUMP"


def test_baseline_reasoning_is_removed_before_dataset_normalization():
    predictions = normalize_batch_for_eval(
        ["Reasoning here. Therefore, the answer is: i_jump"],
        dataset_type="scan",
        baseline_type="cot",
    )
    targets = normalize_batch_for_eval(["I_JUMP"], dataset_type="scan")

    assert predictions == targets == ["I_JUMP"]


def test_cfq_normalization_canonicalizes_keywords_and_punctuation_spacing():
    left = normalize_for_eval("select ?x where{ ?x ns:p ?y .}", "cfq")
    right = normalize_for_eval("SELECT ?x WHERE { ?x ns:p ?y. }", "cfq")

    assert left == right
