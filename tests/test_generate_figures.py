"""Tests for artifact-backed figure data loading."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


def _load_figure_module():
    path = Path("scripts/generate_figures.py")
    spec = importlib.util.spec_from_file_location("generate_figures", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


figures = _load_figure_module()


class FigureArtifactTests(unittest.TestCase):
    def test_loads_and_sorts_real_depth_summaries(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            report = {
                "final_evaluation": {
                    "accuracy_by_depth": {
                        "4": {"mean": 0.5, "std": 0.1},
                        "2": {"mean": 0.8, "std": 0.05},
                    }
                }
            }
            (root / "proposed_aggregated.json").write_text(
                json.dumps(report), encoding="utf-8"
            )

            series = figures.load_depth_series(root)

            self.assertEqual(
                series["proposed"],
                [(2, 0.8, 0.05), (4, 0.5, 0.1)],
            )

    def test_rejects_missing_real_depth_data(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(ValueError, "No aggregated depth metrics"):
                figures.load_depth_series(Path(temporary_directory))


if __name__ == "__main__":
    unittest.main()