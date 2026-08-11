"""Tests for release and benchmark-family reporting utilities."""

import hashlib
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from scripts.build_release_manifest import build_manifest
from scripts.report_cogs_slog_family import build_family_report


class ReleaseReportingTests(unittest.TestCase):
    def test_family_report_preserves_separate_dataset_artifacts(self):
        cogs = {"final_evaluation": {"exact_match": 0.5}}
        slog = {"final_evaluation": {"exact_match": 0.25}}

        report = build_family_report(cogs, slog)

        self.assertEqual(report["benchmark_family"], "COGS/SLOG")
        self.assertIs(report["datasets"]["cogs"], cogs)
        self.assertIs(report["datasets"]["slog"], slog)
        self.assertNotIn("exact_match", report)

    def test_release_manifest_is_sorted_and_checksum_backed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src").mkdir()
            (root / "src" / "b.py").write_text("b\n")
            (root / "src" / "a.py").write_text("a\n")

            with patch(
                "scripts.build_release_manifest._git_value",
                side_effect=["revision", ""],
            ):
                manifest = build_manifest(root, ("src",))

            self.assertEqual(
                [entry["path"] for entry in manifest["files"]],
                ["src/a.py", "src/b.py"],
            )
            self.assertEqual(
                manifest["files"][0]["sha256"],
                hashlib.sha256(b"a\n").hexdigest(),
            )

    def test_release_manifest_rejects_missing_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                build_manifest(Path(directory), ("missing",))

    def test_manifest_requires_git_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "README.md").write_text("release\n")
            with patch(
                "scripts.build_release_manifest._git_value", return_value=None
            ):
                with self.assertRaisesRegex(RuntimeError, "Git worktree"):
                    build_manifest(root, includes=("README.md",))


if __name__ == "__main__":
    unittest.main()