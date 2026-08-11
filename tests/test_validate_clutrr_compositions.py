"""Tests for official-only CLUTRR corpus validation."""

import csv
from pathlib import Path
import tempfile
import unittest

from scripts.validate_clutrr_compositions import (
    REQUIRED_FIELDS,
    validate_clutrr_corpus,
)


class CLUTRRCorpusValidationTests(unittest.TestCase):
    def _write(self, root: Path, row, name: str = "task_1.2_train.csv") -> None:
        with (root / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=REQUIRED_FIELDS)
            writer.writeheader()
            writer.writerow(row)

    def test_valid_official_row_reports_grounding(self):
        row = {
            "story": "Alice knows Bob.",
            "text_query": "What is Alice to Bob?",
            "text_target": "mother",
            "story_edges": "[(0, 1)]",
            "edge_types": "['mother']",
            "query_edge": "(0, 1)",
            "genders": "Alice:female,Bob:male",
            "task_name": "task_1.1",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write(root, row)
            report = validate_clutrr_corpus(root)

        self.assertTrue(report["passed"])
        self.assertEqual(report["annotated_examples"], 1)
        self.assertEqual(report["operator_counts"], {"relation": 1})
        self.assertEqual(report["hop_counts"], {"train:k=1": 1})

    def test_ambiguous_names_are_unannotated_not_invented(self):
        row = {
            "story": "Alice met Bob. Alice is Bob's mother.",
            "text_query": "What is Alice to Bob?",
            "text_target": "mother",
            "story_edges": "[(0, 1)]",
            "edge_types": "['mother']",
            "query_edge": "(0, 1)",
            "genders": "Alice:female,Bob:male",
            "task_name": "task_1.1",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write(root, row)
            report = validate_clutrr_corpus(root)

        self.assertTrue(report["passed"])
        self.assertEqual(report["annotated_examples"], 0)

    def test_requires_official_split_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            report = validate_clutrr_corpus(Path(directory))

        self.assertFalse(report["passed"])
        self.assertEqual(report["examples"], 0)

    def test_rejects_task_hop_that_disagrees_with_query_path(self):
        row = {
            "story": "Alice knows Bob.",
            "text_query": "What is Alice to Bob?",
            "text_target": "mother",
            "story_edges": "[(0, 1)]",
            "edge_types": "['mother']",
            "query_edge": "(0, 1)",
            "genders": "Alice:female,Bob:male",
            "task_name": "task_1.2",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write(root, row)
            report = validate_clutrr_corpus(root)

        self.assertFalse(report["passed"])
        self.assertIn("differs from query path", report["errors"][0])


if __name__ == "__main__":
    unittest.main()