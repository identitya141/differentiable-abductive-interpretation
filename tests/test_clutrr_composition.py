"""Tests for conservative CLUTRR relation-chain annotations."""

import unittest

from src.data.clutrr_composition import (
    CLUTRRCompositionError,
    CLUTRRCompositionSpec,
    align_clutrr_composition_specs_to_tokens,
    extract_clutrr_composition_specs,
)


class _FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        ids = []
        for word in text.split():
            ids.extend([len(word), sum(ord(character) for character in word) % 97])
        return ids


class CLUTRRCompositionTests(unittest.TestCase):
    def test_extracts_query_path_and_binary_joins(self):
        specs = extract_clutrr_composition_specs(
            "Alice knows Bob and Carol waits elsewhere",
            "[(0, 1), (1, 2), (2, 3)]",
            "['mother', 'brother', 'daughter']",
            "(0, 2)",
            "Alice:female,Bob:male,Carol:female,David:male",
        )

        self.assertEqual(
            specs,
            (
                CLUTRRCompositionSpec((0, 1), (2, 3), (0, 3), "relation"),
                CLUTRRCompositionSpec((2, 3), (4, 5), (2, 5), "relation"),
                CLUTRRCompositionSpec((0, 3), (2, 5), (0, 5), "join"),
            ),
        )

    def test_ignores_ambiguous_entity_mentions(self):
        specs = extract_clutrr_composition_specs(
            "Alice knows Bob and Alice asks Bob",
            [(0, 1)],
            ["mother"],
            (0, 1),
            "Alice:female,Bob:male",
        )

        self.assertEqual(specs, ())

    def test_rejects_disconnected_query(self):
        with self.assertRaisesRegex(CLUTRRCompositionError, "disconnected"):
            extract_clutrr_composition_specs(
                "Alice knows Bob while Carol knows David",
                [(0, 1), (2, 3)],
                ["mother", "father"],
                (0, 3),
                "Alice:female,Bob:male,Carol:female,David:male",
            )

    def test_rejects_misaligned_edge_types(self):
        with self.assertRaisesRegex(CLUTRRCompositionError, "one-to-one"):
            extract_clutrr_composition_specs(
                "Alice knows Bob",
                [(0, 1)],
                [],
                (0, 1),
                "Alice:female,Bob:male",
            )

    def test_aligns_word_spans_to_token_spans(self):
        aligned = align_clutrr_composition_specs_to_tokens(
            "Alice knows Bob",
            [CLUTRRCompositionSpec((0, 1), (2, 3), (0, 3), "relation")],
            _FakeTokenizer(),
        )

        self.assertEqual(
            aligned,
            (CLUTRRCompositionSpec((0, 2), (4, 6), (0, 6), "relation"),),
        )


if __name__ == "__main__":
    unittest.main()