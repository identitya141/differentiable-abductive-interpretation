"""Tests for grounded COGS semantic-role composition annotations."""

import unittest

from src.data.cogs_composition import (
    COGSCompositionError,
    COGSCompositionSpec,
    align_cogs_composition_specs_to_tokens,
    extract_cogs_composition_specs,
)


class _FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        ids = []
        for word in text.split():
            ids.extend([len(word), sum(ord(character) for character in word) % 97])
        return ids


class COGSCompositionTests(unittest.TestCase):
    def test_extracts_transitive_agent_and_theme_roles(self):
        specs = extract_cogs_composition_specs(
            "The sailor dusted a boy .",
            "* sailor ( x _ 1 ) ; dust . agent ( x _ 2 , x _ 1 ) "
            "AND dust . theme ( x _ 2 , x _ 4 ) AND boy ( x _ 4 )",
        )

        self.assertEqual(
            specs,
            (
                COGSCompositionSpec((2, 3), (1, 2), (1, 3), "agent"),
                COGSCompositionSpec((2, 3), (4, 5), (2, 5), "theme"),
            ),
        )

    def test_extracts_named_agent_and_nominal_modifier(self):
        specs = extract_cogs_composition_specs(
            "Emma ate the ring beside a bed .",
            "* ring ( x _ 3 ) ; eat . agent ( x _ 1 , Emma ) "
            "AND eat . theme ( x _ 1 , x _ 3 ) "
            "AND ring . nmod . beside ( x _ 3 , x _ 6 ) "
            "AND bed ( x _ 6 )",
        )

        self.assertEqual([spec.operator for spec in specs], ["agent", "theme", "nmod"])
        self.assertEqual(specs[0].right_span, (0, 1))
        self.assertEqual(specs[2].parent_span, (3, 7))

    def test_primitive_lambda_entry_has_no_grounded_compositions(self):
        specs = extract_cogs_composition_specs(
            "touch",
            "LAMBDA a . LAMBDA b . LAMBDA e . touch . agent ( e , b ) "
            "AND touch . theme ( e , a )",
        )

        self.assertEqual(specs, ())

    def test_aligns_word_spans_to_token_spans(self):
        aligned = align_cogs_composition_specs_to_tokens(
            "The sailor dusted .",
            [COGSCompositionSpec((2, 3), (1, 2), (1, 3), "agent")],
            _FakeTokenizer(),
        )

        self.assertEqual(
            aligned,
            (COGSCompositionSpec((4, 6), (2, 4), (2, 6), "agent"),),
        )

    def test_rejects_unbalanced_logical_form(self):
        with self.assertRaisesRegex(COGSCompositionError, "Unbalanced"):
            extract_cogs_composition_specs(
                "The sailor dusted .",
                "dust . agent ( x _ 2 , x _ 1",
            )

    def test_rejects_variable_outside_sentence(self):
        with self.assertRaisesRegex(COGSCompositionError, "outside"):
            extract_cogs_composition_specs(
                "The sailor dusted .",
                "dust . agent ( x _ 2 , x _ 99 )",
            )


if __name__ == "__main__":
    unittest.main()
