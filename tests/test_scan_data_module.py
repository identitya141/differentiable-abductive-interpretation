"""Tests for local SCAN loading and deterministic validation splitting."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from src.data.scan_dataset import SCANDataModule


class _FakeTokenizer:
    pad_token_id = 0

    def __init__(self):
        self._ids = {}

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [self._token_id(word) for word in text.split()]

    def __call__(
        self,
        text=None,
        text_target=None,
        max_length=16,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ):
        del padding, return_tensors
        value = text_target if text_target is not None else text
        token_ids = self.encode(value)[:max_length] if truncation else self.encode(value)
        attention_mask = [1] * len(token_ids)
        padding_size = max_length - len(token_ids)
        token_ids.extend([self.pad_token_id] * padding_size)
        attention_mask.extend([0] * padding_size)
        return SimpleNamespace(
            input_ids=torch.tensor([token_ids]),
            attention_mask=torch.tensor([attention_mask]),
        )

    def _token_id(self, token):
        if token not in self._ids:
            self._ids[token] = len(self._ids) + 1
        return self._ids[token]


class SCANDataModuleTests(unittest.TestCase):
    def test_local_data_uses_deterministic_validation_holdout(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            split_dir = data_dir / "length"
            split_dir.mkdir()
            (split_dir / "tasks_train_length.txt").write_text(
                "\n".join(
                    [
                        "IN: jump twice OUT: JUMP JUMP",
                        "IN: walk and run OUT: WALK RUN",
                        "IN: look after jump OUT: JUMP LOOK",
                        "IN: run around left OUT: LTURN RUN",
                        "IN: turn opposite right OUT: LTURN LTURN",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (split_dir / "tasks_test_length.txt").write_text(
                "IN: walk thrice OUT: WALK WALK WALK\n",
                encoding="utf-8",
            )

            first = self._build_module(data_dir)
            second = self._build_module(data_dir)

            self.assertEqual(len(first.train_dataset), 4)
            self.assertEqual(len(first.validation_dataset), 1)
            self.assertEqual(len(first.test_dataset), 1)
            self.assertEqual(first.train_dataset.indices, second.train_dataset.indices)
            self.assertEqual(
                first.validation_dataset.indices,
                second.validation_dataset.indices,
            )
            self.assertTrue(
                set(first.train_dataset.indices).isdisjoint(
                    first.validation_dataset.indices
                )
            )

            batch = next(iter(first.validation_dataloader()))
            self.assertEqual(len(batch.composition_specs), 1)
            self.assertGreater(len(batch.composition_specs[0]), 0)
            self.assertEqual(first.validation_dataloader().num_workers, 0)

    def test_tree_linearization_removes_span_constraints(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = self._write_minimal_data(Path(temporary_directory))
            module = self._build_module(
                data_dir,
                input_representation="tree_linearized",
            )

            example = module.train_dataset.dataset.get_example(0)
            self.assertTrue(example.input_text.startswith("["))
            self.assertEqual(example.composition_specs, [])

    def test_nonce_control_preserves_grounded_constraints(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = self._write_minimal_data(Path(temporary_directory))
            module = self._build_module(data_dir, nonce_primitives=True)

            example = module.train_dataset.dataset.get_example(0)
            self.assertIn("wif", example.input_text)
            self.assertGreater(len(example.composition_specs), 0)

    @staticmethod
    def _write_minimal_data(data_dir):
        split_dir = data_dir / "length"
        split_dir.mkdir()
        (split_dir / "tasks_train_length.txt").write_text(
            "IN: jump twice OUT: JUMP JUMP\n"
            "IN: walk and run OUT: WALK RUN\n",
            encoding="utf-8",
        )
        (split_dir / "tasks_test_length.txt").write_text(
            "IN: walk thrice OUT: WALK WALK WALK\n",
            encoding="utf-8",
        )
        return data_dir

    @staticmethod
    def _build_module(data_dir, **overrides):
        module = SCANDataModule(
            tokenizer=_FakeTokenizer(),
            scan_split="length",
            batch_size=2,
            max_source_length=16,
            max_target_length=16,
            num_workers=0,
            data_dir=str(data_dir),
            validation_fraction=0.2,
            seed=123,
            **overrides,
        )
        module.setup()
        return module


if __name__ == "__main__":
    unittest.main()
