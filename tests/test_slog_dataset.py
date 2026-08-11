"""Tests for official SLOG schema and depth metadata."""

import csv
import tempfile
import unittest
from pathlib import Path

from src.data.slog_dataset import SLOG_CATEGORIES, SLOGDataset, infer_slog_depth


class _FakeEncoding:
    def __init__(self, values):
        import torch

        self.input_ids = torch.tensor([values])
        self.attention_mask = torch.ones_like(self.input_ids)


class _FakeTokenizer:
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [len(word) for word in text.split()]

    def __call__(self, text=None, text_target=None, max_length=128, **kwargs):
        del kwargs
        value = text if text is not None else text_target
        ids = self.encode(value)[:max_length]
        return _FakeEncoding(ids + [0] * (max_length - len(ids)))


class SLOGDatasetTests(unittest.TestCase):
    def test_defines_exactly_the_official_17_categories(self):
        self.assertEqual(len(SLOG_CATEGORIES), 17)
        self.assertEqual(len(set(SLOG_CATEGORIES)), 17)

    def test_infers_each_recursive_depth_family(self):
        self.assertEqual(
            infer_slog_depth("PP_3", "a . nmod . on ( x , y ) AND b . nmod . in ( y , z )"),
            2,
        )
        self.assertEqual(
            infer_slog_depth("CP_3", "a . ccomp ( x , y ) AND b . ccomp ( y , z )"),
            2,
        )
        self.assertEqual(
            infer_slog_depth("center_embed_3", "a . nmod ( x , y )"),
            1,
        )
        self.assertIsNone(infer_slog_depth("PP_modif_subj", "a . nmod . on ( x , y )"))
        self.assertIsNone(infer_slog_depth("Q_subj_active", "ask ( x )"))

    def test_loads_generalization_category_and_depth(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "generalization_sets" / "gen_cogsLF.tsv"
            path.parent.mkdir()
            with path.open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle, delimiter="\t").writerow(
                    [
                        "A girl saw a cat in a room .",
                        "see . agent ( x _ 2 , x _ 1 ) AND cat . nmod . in ( x _ 4 , x _ 7 )",
                        "PP_3",
                    ]
                )

            dataset = SLOGDataset(
                _FakeTokenizer(), split="test", data_dir=directory
            )

            example = dataset.get_example(0)
            self.assertTrue(example.is_ood)
            self.assertEqual(example.generalization_category, "PP_3")
            self.assertEqual(example.compositional_structure, "depth:1")

    def test_rejects_unknown_generalization_category(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gen_cogsLF.tsv"
            path.write_text("A cat slept .\tsleep ( cat )\tunknown\n")
            with self.assertRaisesRegex(ValueError, "Unknown SLOG category"):
                SLOGDataset(_FakeTokenizer(), split="test", data_dir=directory)


if __name__ == "__main__":
    unittest.main()