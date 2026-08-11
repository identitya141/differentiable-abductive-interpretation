"""Focused tests for official SLOG evaluation metadata."""

import unittest
import json
import tempfile
from pathlib import Path

import torch

from scripts.evaluate import evaluate_model, save_results
from src.data.base_dataset import CompositionalBatch


class _Tokenizer:
    pad_token_id = 0

    def batch_decode(self, rows, skip_special_tokens=True):
        return ["correct" if int(row[0]) == 1 else "wrong" for row in rows]


class _Model:
    def to(self, device):
        return self

    def eval(self):
        return self

    def generate(self, input_ids, **kwargs):
        return input_ids


class _Dataset(torch.utils.data.Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        return index

    def collate_fn(self, batch):
        size = len(batch)
        return CompositionalBatch(
            input_ids=torch.tensor([[1], [2]][:size]),
            attention_mask=torch.ones(size, 1, dtype=torch.long),
            labels=torch.tensor([[1], [1]][:size]),
            input_texts=["first", "second"][:size],
            original_input_texts=["first", "second"][:size],
            is_ood=torch.tensor([True, True][:size]),
            generalization_categories=["PP_5-12", "Q_long_mv"][:size],
            composition_depths=[5, None][:size],
        )


class SLOGEvaluationTests(unittest.TestCase):
    def test_preserves_official_category_and_depth_breakdowns(self):
        result = evaluate_model(
            _Model(),
            _Tokenizer(),
            _Dataset(),
            torch.device("cpu"),
            batch_size=2,
            dataset_type="slog",
        )

        self.assertEqual(result.exact_match, 0.5)
        self.assertEqual(result.accuracy_by_category["PP_5-12"], 1.0)
        self.assertEqual(result.accuracy_by_category["Q_long_mv"], 0.0)
        self.assertEqual(result.accuracy_by_depth, {5: 1.0})
        self.assertEqual(result.inputs, ["first", "second"])

    def test_quick_subset_keeps_custom_collator(self):
        result = evaluate_model(
            _Model(),
            _Tokenizer(),
            _Dataset(),
            torch.device("cpu"),
            max_samples=1,
            batch_size=1,
            dataset_type="slog",
        )

        self.assertEqual(result.num_examples, 1)
        self.assertEqual(result.accuracy_by_depth, {5: 1.0})

    def test_result_dataclass_is_json_serializable(self):
        result = evaluate_model(
            _Model(), _Tokenizer(), _Dataset(), torch.device("cpu"), batch_size=2
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            save_results({"datasets": {"slog": {"test": result}}}, output)
            saved = json.loads(output.read_text())

        self.assertEqual(saved["datasets"]["slog"]["test"]["num_examples"], 2)


if __name__ == "__main__":
    unittest.main()