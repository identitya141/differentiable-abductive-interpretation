"""Tests for artifact-backed composition-violation analysis."""

import math
import unittest

from scripts.analyze_composition_violations import analyze_rows


class CompositionViolationAnalysisTests(unittest.TestCase):
    def test_reports_bounds_groups_coverage_and_correlation(self):
        report = analyze_rows(
            [
                {"correct": True, "composition_violation": 0.1},
                {"correct": True, "composition_violation": 0.2},
                {"correct": False, "composition_violation": 0.8},
                {"correct": False, "composition_violation": 0.9},
                {"correct": False, "composition_violation": None},
            ]
        )

        self.assertEqual(report["analysis_class"], "exploratory")
        self.assertEqual(report["examples_with_composition"], 4)
        self.assertEqual(report["examples_without_composition"], 1)
        self.assertEqual(report["all"]["minimum"], 0.1)
        self.assertEqual(report["all"]["maximum"], 0.9)
        self.assertAlmostEqual(report["correct"]["mean"], 0.15)
        self.assertAlmostEqual(report["incorrect"]["mean"], 0.85)
        self.assertLess(
            report["point_biserial_correlation_with_correctness"], 0.0
        )

    def test_rejects_non_finite_violation(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            analyze_rows([{"correct": True, "composition_violation": math.nan}])


if __name__ == "__main__":
    unittest.main()
