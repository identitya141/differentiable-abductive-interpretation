"""Standard-library tests for experiment matrix orchestration."""

import importlib.util
import sys
import unittest
from pathlib import Path


def _load_matrix_module():
    path = Path("scripts/run_experiment_matrix.py")
    spec = importlib.util.spec_from_file_location("run_experiment_matrix", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


matrix = _load_matrix_module()


class ExperimentMatrixTests(unittest.TestCase):
    def test_slurm_launcher_activates_python_before_gate_validation(self):
        launcher = Path("slurm/slurm_publication_matrix.sh").read_text(
            encoding="utf-8"
        )
        first_python = launcher.index('python - "$REFERENCE_EQUIVALENCE_REPORT"')

        self.assertLess(launcher.index("module load PyTorch/"), first_python)
        self.assertLess(launcher.index('source "$VENV_DIR/bin/activate"'), first_python)
        self.assertNotIn("training.fp16=true", launcher)

    def test_builds_config_major_paired_seed_commands(self):
        commands = matrix.build_commands(
            "/python",
            [Path("a.json"), Path("b.json")],
            [42, 123],
        )

        self.assertEqual(
            commands,
            [
                ["/python", "scripts/train.py", "--config", "a.json", "--seed", "42"],
                ["/python", "scripts/train.py", "--config", "a.json", "--seed", "123"],
                ["/python", "scripts/train.py", "--config", "b.json", "--seed", "42"],
                ["/python", "scripts/train.py", "--config", "b.json", "--seed", "123"],
            ],
        )

    def test_dry_run_records_every_command(self):
        commands = [["python", "train.py", "--seed", "42"]]

        records = matrix.run_matrix(commands, dry_run=True)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["status"], "dry_run")
        self.assertIsNone(records[0]["return_code"])


if __name__ == "__main__":
    unittest.main()
