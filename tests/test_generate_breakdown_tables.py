"""Tests for artifact-backed depth, operator, and category tables."""

import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path


def _load_breakdown_module():
    path = Path("scripts/generate_breakdown_tables.py")
    spec = importlib.util.spec_from_file_location("generate_breakdown_tables", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


breakdowns = _load_breakdown_module()


class BreakdownTableTests(unittest.TestCase):
    def test_aggregates_scan_depth_and_each_present_operator(self):
        runs = {
            "proposed": {
                42: [self._scan_row(True)],
                123: [self._scan_row(False)],
            }
        }

        report = breakdowns.build_breakdown_report(runs, (42, 123))
        method = report["methods"]["proposed"]

        self.assertEqual(set(method["operator"]), {"around", "direction", "twice"})
        self.assertEqual(method["depth"]["3"]["mean"], 0.5)
        self.assertTrue(
            math.isclose(method["operator"]["around"]["std"], math.sqrt(0.5))
        )

    def test_requires_cogs_generalization_categories(self):
        row = {
            "dataset": "cogs",
            "composition_depth": 2,
            "correct": True,
            "generalization_category": None,
        }
        runs = {"cogs_method": {42: [row]}}

        with self.assertRaisesRegex(ValueError, "Missing COGS category"):
            breakdowns.build_breakdown_report(runs, (42,))

    def test_writes_json_csv_and_latex_from_real_report(self):
        runs = {"proposed": {42: [self._scan_row(True)]}}
        report = breakdowns.build_breakdown_report(runs, (42,))

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            breakdowns.write_outputs(report, output_dir)

            self.assertTrue((output_dir / "breakdowns.json").is_file())
            self.assertIn("operator", (output_dir / "breakdowns.csv").read_text())
            self.assertIn("\\begin{longtable}", (output_dir / "breakdowns.tex").read_text())
            loaded = json.loads((output_dir / "breakdowns.json").read_text())
            self.assertEqual(loaded["methods"]["proposed"]["depth"]["3"]["mean"], 1.0)

    @staticmethod
    def _scan_row(correct):
        return {
            "dataset": "scan",
            "input": "jump around left twice",
            "composition_depth": 3,
            "correct": correct,
            "generalization_category": None,
        }


if __name__ == "__main__":
    unittest.main()