"""Tests for family-wise publication comparison summaries."""

import importlib.util
import sys
import unittest
from pathlib import Path


def _load_summary_module():
    scripts_dir = str(Path("scripts").resolve())
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    path = Path("scripts/summarize_comparisons.py")
    spec = importlib.util.spec_from_file_location("summarize_comparisons", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


summary_module = _load_summary_module()


class ComparisonSummaryTests(unittest.TestCase):
    def test_applies_holm_across_the_complete_family(self):
        reports = [
            self._report("control_a", 0.01, [0.2, 0.1]),
            self._report("control_b", 0.04, [0.1, -0.1]),
            self._report("control_c", 0.03, [0.3, 0.2]),
        ]

        summary = summary_module.summarize_reports(reports)

        self.assertEqual(summary["family_size"], 3)
        adjusted = [
            report["seed_level"]["holm_adjusted_p_value"]
            for report in summary["comparisons"]
        ]
        self.assertEqual(adjusted, [0.03, 0.06, 0.06])
        self.assertTrue(summary["comparisons"][0]["all_paired_seeds_improved"])
        self.assertFalse(summary["comparisons"][1]["all_paired_seeds_improved"])

    def test_rejects_duplicate_controls(self):
        report = self._report("control", 0.05, [0.1, 0.2])

        with self.assertRaisesRegex(ValueError, "duplicate control"):
            summary_module.summarize_reports([report, report])

    def test_primary_family_contract_is_enforced(self):
        reports = [self._report(f"control_{i}", 0.1, [0.1]) for i in range(6)]
        for report in reports:
            report["benchmark"] = {"dataset": "scan", "split": "length"}
        summary = summary_module.summarize_reports(
            reports, expected_dataset="scan", expected_split="length",
            expected_family_size=6,
        )
        self.assertEqual(summary["family_size"], 6)
        with self.assertRaisesRegex(ValueError, "family size"):
            summary_module.summarize_reports(reports[:5], expected_family_size=6)

    @staticmethod
    def _report(control, p_value, differences):
        return {
            "method_a": "full_contrastive",
            "method_b": control,
            "seed_level": {
                "paired_permutation_p_value": p_value,
            },
            "per_seed": [
                {"seed": index, "difference_a_minus_b": difference}
                for index, difference in enumerate(differences)
            ],
        }


if __name__ == "__main__":
    unittest.main()
