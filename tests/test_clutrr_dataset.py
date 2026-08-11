"""Tests for loading the official CLUTRR CSV contract."""

import csv
import tempfile
import unittest
from pathlib import Path

from src.data.clutrr_dataset import CLUTRRDataset


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


class CLUTRRDatasetTests(unittest.TestCase):
    def test_loads_official_columns_and_grounded_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task_1.2_train.csv"
            fields = [
                "story",
                "text_query",
                "text_target",
                "story_edges",
                "edge_types",
                "query_edge",
                "genders",
                "task_name",
            ]
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(
                    {
                        "story": "Alice knows Bob while Carol waits",
                        "text_query": "How is Alice related to Bob?",
                        "text_target": "Alice is Bob's mother.",
                        "story_edges": "[(0, 1)]",
                        "edge_types": "['mother']",
                        "query_edge": "(0, 1)",
                        "genders": "Alice:female,Bob:male,Carol:female",
                        "task_name": "task_1.2",
                    }
                )

            dataset = CLUTRRDataset(
                _FakeTokenizer(),
                split="train",
                train_hops=[2],
                data_dir=directory,
                max_source_length=32,
                max_target_length=16,
            )

            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset.get_example(0).generalization_category, "k=2")
            self.assertEqual(
                [spec.operator for spec in dataset[0]["composition_specs"]],
                ["relation"],
            )

    def test_missing_official_data_does_not_silently_generate(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                CLUTRRDataset(
                    _FakeTokenizer(),
                    split="train",
                    train_hops=[2],
                    data_dir=directory,
                )


if __name__ == "__main__":
    unittest.main()