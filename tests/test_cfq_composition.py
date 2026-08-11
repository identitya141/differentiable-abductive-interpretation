"""Tests for grounded CFQ query-graph composition annotations."""

import unittest
from unittest.mock import patch

import torch
from torch.utils.data import Dataset

from src.data.cfq_composition import (
    CFQCompositionError,
    CFQCompositionSpec,
    align_cfq_composition_specs_to_tokens,
    extract_cfq_composition_specs,
)
from src.data.cfq_dataset import CFQDataModule


class _FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [len(word) for word in text.split()]


class _FakeCFQDataset(Dataset):
    def __init__(self, tokenizer, split, **kwargs):
        del tokenizer, kwargs
        self.split = split
        self.size = 10 if split == "train" else 3

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return index

    @staticmethod
    def collate_fn(batch):
        return torch.tensor(batch)


class CFQCompositionTests(unittest.TestCase):
    def test_data_module_uses_deterministic_train_only_validation(self):
        with patch("src.data.cfq_dataset.CFQDataset", _FakeCFQDataset):
            first = CFQDataModule(
                object(), data_dir="data/cfq", validation_fraction=0.2, seed=42
            )
            second = CFQDataModule(
                object(), data_dir="data/cfq", validation_fraction=0.2, seed=42
            )
            first.setup()
            second.setup()

        self.assertEqual(first.train_dataset.indices, second.train_dataset.indices)
        self.assertEqual(
            first.validation_dataset.indices, second.validation_dataset.indices
        )
        self.assertEqual(len(first.train_dataset), 8)
        self.assertEqual(len(first.validation_dataset), 2)
        self.assertEqual(len(first.test_dataset), 3)
        self.assertTrue(
            set(first.train_dataset.indices).isdisjoint(
                first.validation_dataset.indices
            )
        )

    def test_grounds_typed_variables_and_entity(self):
        specs = extract_cfq_composition_specs(
            "Who wrote M1 and wrote a film",
            "SELECT DISTINCT ?x0 WHERE {\n"
            "?x0 a ns:people.person .\n"
            "?x0 ns:film.writer.film ?x1 .\n"
            "?x0 ns:film.writer.film M1 .\n"
            "?x1 a ns:film.film\n}",
        )

        self.assertEqual(
            specs,
            (
                CFQCompositionSpec((0, 1), (6, 7), (0, 7), "relation"),
                CFQCompositionSpec((0, 1), (2, 3), (0, 3), "relation"),
            ),
        )

    def test_marks_property_paths_as_joins(self):
        specs = extract_cfq_composition_specs(
            "Did a person marry a cinematographer",
            "SELECT count(*) WHERE {\n"
            "?x0 a ns:people.person .\n"
            "?x0 ns:people.person.spouse_s/ns:people.marriage.spouse ?x1 .\n"
            "?x1 a ns:film.cinematographer .\n}",
        )

        self.assertEqual(specs[0].operator, "join")
        self.assertEqual(specs[0].parent_span, (2, 6))

    def test_skips_ambiguous_or_ungrounded_endpoints(self):
        specs = extract_cfq_composition_specs(
            "Did a person influence a person",
            "SELECT count(*) WHERE {\n?x0 a ns:people.person .\n"
            "?x0 ns:influence.influence_node.influenced ?x1 .\n"
            "?x1 a ns:people.person\n}",
        )
        self.assertEqual(specs, ())

    def test_aligns_word_spans(self):
        aligned = align_cfq_composition_specs_to_tokens(
            "Who wrote M1",
            [CFQCompositionSpec((0, 1), (2, 3), (0, 3), "relation")],
            _FakeTokenizer(),
        )
        self.assertEqual(
            aligned,
            (CFQCompositionSpec((0, 1), (2, 3), (0, 3), "relation"),),
        )

    def test_rejects_unbalanced_query(self):
        with self.assertRaisesRegex(CFQCompositionError, "Unbalanced"):
            extract_cfq_composition_specs("Who wrote M1", "SELECT WHERE {")


if __name__ == "__main__":
    unittest.main()