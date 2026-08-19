"""Regression tests for collision-free multi-dataset publication wiring."""

import json
from pathlib import Path
import unittest
import yaml

from scripts.unified_experiment_matrix import expand_runs, load_manifest


ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = ROOT / "configs/publication/multidataset_matrix.json"


class MultidatasetMatrixTests(unittest.TestCase):
    def setUp(self):
        self.matrix = load_manifest(MATRIX_PATH)
        self.runs = expand_runs(self.matrix)

    def test_every_run_has_unique_identity_name_and_output(self):
        identities = []
        for run in self.runs:
            identity = (run["dataset"], run["split"], run["method"], run["seed"])
            identities.append(identity)
            self.assertTrue((ROOT / run["config"]).is_file())

        self.assertEqual(len(identities), len(set(identities)))
        self.assertEqual(len(identities), 720)
        self.assertEqual(
            {(row["dataset"], row["split"]) for row in self.runs},
            {
                ("scan", "length"),
                ("cogs", "generalization"),
                ("slog", "structural_generalization"),
                ("cfq", "mcd1"),
                ("cfq", "mcd2"),
                ("cfq", "mcd3"),
            },
        )

    def test_every_unique_matrix_configuration_parses_before_submission(self):
        checked = set()
        for run in self.runs:
            path = ROOT / run["config"]
            if path in checked:
                continue
            checked.add(path)
            if path.suffix == ".json":
                payload = json.loads(path.read_text())
            else:
                payload = yaml.safe_load(path.read_text())
            self.assertIsInstance(payload, dict)
            self.assertTrue(payload)
            self.assertIn(run["runner"], {"baseline", "dai_control", "proposed"})

    def test_seeds_are_paired_and_unique(self):
        self.assertEqual(
            self.matrix["seeds"],
            [42, 123, 456, 789, 1024, 2027, 4099, 7919, 104729, 130363],
        )
        self.assertEqual(len(self.matrix["seeds"]), len(set(self.matrix["seeds"])))

    def test_ineligible_datasets_are_explicitly_not_scheduled(self):
        scheduled = {run["dataset"] for run in self.runs}
        self.assertEqual(scheduled, {"scan", "cogs", "slog", "cfq"})

    def test_launcher_uses_immutable_snapshot_and_hierarchical_output(self):
        launcher = (ROOT / "slurm/slurm_multidataset_matrix.sh").read_text()
        submitter = (ROOT / "slurm/submit_multidataset_pipeline.sh").read_text()
        self.assertIn('if [[ -w "$PROJECT_DIR" ]]', launcher)
        self.assertIn(
            'experiments/publication/$DATASET/$SPLIT/$METHOD', launcher
        )
        self.assertIn('case "$RUNNER" in', launcher)
        self.assertIn("scripts/train_baseline.py", launcher)
        self.assertIn("dai_control|proposed", launcher)
        self.assertIn("scripts/unified_experiment_matrix.py", submitter)
        self.assertIn("slurm/slurm_test_suite.sh", submitter)
        self.assertIn("slurm/slurm_validate_publication_datasets.sh", submitter)
        self.assertIn("slurm/slurm_reference_equivalence.sh", submitter)
        self.assertIn("slurm/slurm_numerical_smoke.sh", submitter)
        self.assertIn("slurm/slurm_small_overfit.sh", submitter)
        self.assertIn("slurm/slurm_prepare_publication_gates.sh", submitter)
        self.assertIn('--dependency="afterok:$VALIDATION_JOB:$GATE_JOB"', submitter)
        self.assertIn('f"seed_{seed}"', (ROOT / "scripts/train.py").read_text())
        self.assertNotIn("training.fp16=true", launcher)
        self.assertIn("sha256sum --check SHA256SUMS", launcher)
        self.assertIn('"data_sha256": os.environ["DATA_SHA256"]', launcher)
        self.assertIn("Existing run contract differs; refusing unsafe resume", launcher)
        self.assertIn("Resuming baseline from newest checkpoint", launcher)
        self.assertIn("Resuming DAI from", launcher)
        self.assertIn("#SBATCH --partition=gpu_30d_p", launcher)
        self.assertIn("#SBATCH --gres=gpu:H100:1", launcher)
        self.assertIn("#SBATCH --time=14-00:00:00", launcher)
        self.assertIn("chmod -R a-w", submitter)
        self.assertIn("--exclude='data/**'", submitter)
        self.assertIn("--exclude='logs/**'", submitter)
        self.assertIn("--exclude='results/**'", submitter)
        self.assertIn("module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1", submitter)
        self.assertEqual(submitter.count('"$VENV_DIR/bin/python"'), 5)
        self.assertNotIn(" SOURCE_SNAPSHOT_ID=\"$SOURCE_SNAPSHOT_ID\" python", submitter)
        self.assertIn('--dependency="afterok:$MATRIX_JOB"', submitter)
        self.assertIn("#SBATCH --partition=batch", submitter)
        self.assertIn('if [[ -z "${SLURM_JOB_ID:-}" ]]', submitter)
        self.assertIn("Submit this coordinator with sbatch", submitter)
        self.assertNotIn("Run this submitter on a Sapelo login node", submitter)

    def test_every_setting_has_eleven_baselines_and_proposed_method(self):
        by_setting = {}
        for run in self.runs:
            by_setting.setdefault((run["dataset"], run["split"]), []).append(run)
        for rows in by_setting.values():
            self.assertEqual(len(rows), 120)
            self.assertEqual(sum(row["runner"] == "baseline" for row in rows), 80)
            self.assertEqual(sum(row["runner"] == "dai_control" for row in rows), 30)
            self.assertEqual(sum(row["runner"] == "proposed" for row in rows), 10)
            for method in {row["method"] for row in rows}:
                self.assertEqual(
                    sorted(row["seed"] for row in rows if row["method"] == method),
                    [42, 123, 456, 789, 1024, 2027, 4099, 7919, 104729, 130363],
                )


if __name__ == "__main__":
    unittest.main()
