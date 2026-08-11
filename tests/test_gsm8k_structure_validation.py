"""Tests for strict GSM8K source-grounding eligibility."""

import unittest
import json
from pathlib import Path
import tempfile

from scripts.validate_gsm8k_structure import assess_examples, validate_gsm8k


class GSM8KStructureValidationTests(unittest.TestCase):
    def test_target_only_equation_is_not_source_grounded(self):
        report = assess_examples(
            [
                {
                    "question": "There are 2 bags with 3 apples each.",
                    "answer": "Multiply. <<2*3=6>>\n#### 6",
                }
            ]
        )

        self.assertEqual(report["target_equation_steps"], 1)
        self.assertEqual(report["source_grounded_equation_steps"], 0)
        self.assertEqual(report["fully_source_grounded_examples"], 0)

    def test_exact_source_equation_is_counted_conservatively(self):
        report = assess_examples(
            [
                {
                    "question": "Evaluate 2*3=6 once.",
                    "answer": "Compute <<2*3=6>>\n#### 6",
                }
            ]
        )

        self.assertEqual(report["fully_source_grounded_examples"], 1)

    def test_tracks_malformed_answer_contract(self):
        report = assess_examples([{"question": "Q", "answer": "No final marker"}])

        self.assertEqual(report["malformed_answers"], 1)
        self.assertEqual(report["target_step_count_distribution"], {"0": 1})

    def test_offline_jsonl_corpus_excludes_target_only_steps(self):
        example = {
            "question": "There are 2 bags with 3 apples each.",
            "answer": "Multiply. <<2*3=6>>\n#### 6",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ("train", "test"):
                (root / f"{split}.jsonl").write_text(json.dumps(example) + "\n")
            report = validate_gsm8k(root)

        self.assertFalse(report["publication_structure_eligible"])
        self.assertEqual(report["decision"], "exclude_target_only_reasoning_annotations")
        self.assertEqual(report["target_equation_steps"], 2)


if __name__ == "__main__":
    unittest.main()