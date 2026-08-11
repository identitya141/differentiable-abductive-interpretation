#!/bin/bash
#SBATCH --job-name=dai_slog_validate
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/slog_validate_%j.out
#SBATCH --error=logs/slog_validate_%j.err

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: Submit this script with sbatch; do not run it on a login node." >&2
    exit 2
fi

PROJECT_DIR=${PROJECT_DIR:-"$HOME/dai-research"}
VENV_DIR=${VENV_DIR:-"$PROJECT_DIR/venv"}
DATA_DIR=${DATA_DIR:-"$PROJECT_DIR/data/slog"}
REPORT_PATH=${REPORT_PATH:-"$PROJECT_DIR/results/validation/slog.json"}

module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
export PYTHONNOUSERSITE=1
export PROJECT_DIR VENV_DIR DATA_DIR REPORT_PATH

cd "$PROJECT_DIR"
python - <<'PY'
import json
import os
from pathlib import Path
import site
import sys

import torch

sys.path.insert(0, os.environ["PROJECT_DIR"])
sys.path.insert(
    1,
    str(
        Path(os.environ["VENV_DIR"])
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    ),
)
sys.path.append(site.getusersitepackages())

from scripts.download_data import download_slog
from scripts.validate_slog import validate_slog_corpus

data_dir = Path(os.environ["DATA_DIR"])
report_path = Path(os.environ["REPORT_PATH"])
download_slog(data_dir)
report = validate_slog_corpus(data_dir)
report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
if not report["passed"]:
    raise SystemExit(f"SLOG validation failed with {len(report['errors'])} errors")
print(json.dumps(report, indent=2, sort_keys=True))
PY