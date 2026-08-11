"""Tests for deterministic SCAN composition extraction."""

import unittest

import torch

from src.data.scan_composition import (
    SCANAlignmentError,
    SCANCompositionSpec,
    align_composition_specs_to_tokens,
    SCANParseError,
    extract_composition_specs,
    linearize_scan_command,
    parse_scan_command,
    replace_scan_primitives_with_nonce_words,
    transform_composition_specs,
)
from src.losses.abstraction_loss import AbstractionLoss
from src.models.abstract_domains import AbstractDomain, TypeElement


class _FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        pieces = {"walk": [1], "around": [2, 3], "right": [4]}
        return [token for word in text.split() for token in pieces[word]]


class _IdentityCompositionDomain(AbstractDomain):
    def abstract(self, hidden_states):
        return TypeElement(type_logits=hidden_states)

    def concretize_loss(self, hidden_states, abstraction, attention_mask=None):
        del abstraction
        return hidden_states.new_zeros(())

    def compose(self, left, right, composition_type):
        del composition_type
        return TypeElement(type_logits=(left.type_logits + right.type_logits) / 2)

    def consistency_loss(self, left, right):
        return torch.nn.functional.mse_loss(left.type_logits, right.type_logits)


class SCANCompositionTests(unittest.TestCase):
    def test_extract_composition_specs(self):
        cases = [
            ("jump", []),
            ("jump twice", [((0, 1), (1, 2), (0, 2), "twice")]),
            (
                "walk around right",
                [
                    ((0, 1), (1, 2), (0, 2), "around"),
                    ((0, 2), (2, 3), (0, 3), "direction"),
                ],
            ),
            (
                "walk and run twice",
                [
                    ((2, 3), (3, 4), (2, 4), "twice"),
                    ((0, 1), (2, 4), (0, 4), "and"),
                ],
            ),
            ("walk after run", [((2, 3), (0, 1), (0, 3), "after")]),
        ]

        for command, expected in cases:
            with self.subTest(command=command):
                actual = [
                    (spec.left_span, spec.right_span, spec.parent_span, spec.operator)
                    for spec in extract_composition_specs(parse_scan_command(command))
                ]
                self.assertEqual(actual, expected)

    def test_rejects_unsupported_command(self):
        with self.assertRaises(SCANParseError):
            parse_scan_command("jump beside walk")

    def test_linearizes_operator_labeled_tree(self):
        self.assertEqual(
            linearize_scan_command("walk twice and turn left"),
            "[AND [TWICE [walk] [twice]] [DIRECTION [turn] [left]]]",
        )

    def test_nonce_control_preserves_composition_words(self):
        self.assertEqual(
            replace_scan_primitives_with_nonce_words(
                "walk twice and jump after turn left"
            ),
            "dax twice and wif after blick left",
        )

    def test_aligns_word_spans_to_token_spans(self):
        specs = (
            SCANCompositionSpec((0, 1), (1, 2), (0, 2), "around"),
            SCANCompositionSpec((0, 2), (2, 3), (0, 3), "direction"),
        )

        aligned = align_composition_specs_to_tokens(
            "walk around right", specs, _FakeTokenizer()
        )

        self.assertEqual(
            aligned,
            (
                SCANCompositionSpec((0, 1), (1, 3), (0, 3), "around"),
                SCANCompositionSpec((0, 3), (3, 4), (0, 4), "direction"),
            ),
        )

    def test_rejects_ambiguous_token_alignment(self):
        class AmbiguousTokenizer(_FakeTokenizer):
            def encode(self, text, add_special_tokens=False):
                if " " in text:
                    return [99]
                return super().encode(text, add_special_tokens=add_special_tokens)

        with self.assertRaises(SCANAlignmentError):
            align_composition_specs_to_tokens(
                "walk around right",
                [SCANCompositionSpec((0, 1), (1, 2), (0, 2), "around")],
                AmbiguousTokenizer(),
            )

    def test_shuffled_control_changes_structure(self):
        specs = (
            SCANCompositionSpec((0, 1), (1, 2), (0, 2), "around"),
            SCANCompositionSpec((0, 2), (2, 3), (0, 3), "direction"),
        )

        shuffled = transform_composition_specs(specs, 3, "shuffled", "seed")

        self.assertNotEqual(shuffled, specs)
        self.assertEqual(len(shuffled), len(specs))
        self.assertEqual(
            [spec.operator for spec in shuffled],
            ["direction", "around"],
        )
        self.assertEqual(
            [spec.parent_span for spec in shuffled],
            [spec.parent_span for spec in specs],
        )
        self.assertEqual(
            [spec.left_span for spec in shuffled],
            [spec.left_span for spec in specs],
        )
        self.assertEqual([spec.right_span for spec in shuffled], [spec.right_span for spec in specs])

    def test_grounded_structure_has_lower_loss_than_shuffled_structure(self):
        specs = (
            SCANCompositionSpec((0, 1), (1, 2), (0, 2), "and"),
            SCANCompositionSpec((2, 3), (3, 4), (2, 4), "after"),
        )
        shuffled = transform_composition_specs(specs, 4, "shuffled", "seed")
        hidden_states = torch.tensor(
            [[[1.0, 0.0], [3.0, 0.0], [0.0, 2.0], [0.0, 6.0]]]
        )
        loss_fn = AbstractionLoss(
            abstract_domain=_IdentityCompositionDomain(hidden_dim=2),
            concretization_weight=0.0,
            composition_weight=1.0,
            consistency_weight=0.0,
            entropy_regularization=0.0,
        )

        grounded_loss = loss_fn(
            hidden_states,
            attention_mask=torch.ones(1, 4),
            composition_specs=[list(specs)],
        ).composition_loss
        shuffled_loss = loss_fn(
            hidden_states,
            attention_mask=torch.ones(1, 4),
            composition_specs=[list(shuffled)],
        ).composition_loss

        self.assertIsNotNone(grounded_loss)
        self.assertIsNotNone(shuffled_loss)
        self.assertTrue(torch.isfinite(grounded_loss))
        self.assertTrue(torch.isfinite(shuffled_loss))
        self.assertNotEqual(shuffled, specs)

    def test_random_control_is_deterministic_and_length_matched(self):
        specs = (
            SCANCompositionSpec((0, 1), (1, 3), (0, 3), "around"),
            SCANCompositionSpec((0, 3), (3, 4), (0, 4), "direction"),
        )

        first = transform_composition_specs(specs, 6, "random", "fixed-seed")
        second = transform_composition_specs(specs, 6, "random", "fixed-seed")

        self.assertEqual(first, second)
        self.assertNotEqual(first, specs)
        self.assertEqual(len(first), len(specs))
        self.assertEqual(
            [spec.operator for spec in first],
            [spec.operator for spec in second],
        )
        for original, transformed in zip(specs, first):
            self.assertNotEqual(original.operator, transformed.operator)
            for original_span, transformed_span in zip(
                (original.left_span, original.right_span, original.parent_span),
                (transformed.left_span, transformed.right_span, transformed.parent_span),
            ):
                self.assertEqual(
                    original_span,
                    transformed_span,
                )

    def test_percentage_corruption_is_deterministic_and_length_matched(self):
        specs = (
            SCANCompositionSpec((0, 1), (1, 2), (0, 2), "twice"),
            SCANCompositionSpec((0, 2), (3, 4), (0, 4), "and"),
        )

        unchanged = transform_composition_specs(
            specs, 6, "grounded", "fixed-seed", corruption_probability=0.0
        )
        first = transform_composition_specs(
            specs, 6, "grounded", "fixed-seed", corruption_probability=1.0
        )
        second = transform_composition_specs(
            specs, 6, "grounded", "fixed-seed", corruption_probability=1.0
        )

        self.assertEqual(unchanged, specs)
        self.assertEqual(first, second)
        self.assertNotEqual(first, specs)
        for original, transformed in zip(specs, first):
            for original_span, transformed_span in zip(
                (original.left_span, original.right_span, original.parent_span),
                (transformed.left_span, transformed.right_span, transformed.parent_span),
            ):
                self.assertEqual(
                    original_span[1] - original_span[0],
                    transformed_span[1] - transformed_span[0],
                )

    def test_rejects_corruption_with_non_grounded_mode(self):
        specs = (SCANCompositionSpec((0, 1), (1, 2), (0, 2), "twice"),)

        with self.assertRaises(ValueError):
            transform_composition_specs(
                specs, 3, "random", "seed", corruption_probability=0.5
            )


if __name__ == "__main__":
    unittest.main()
