#!/bin/bash
#SBATCH --job-name=dai_clutrr_validate
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/clutrr_validate_%j.out
#SBATCH --error=logs/clutrr_validate_%j.err

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: Submit this script with sbatch; do not run it on a login node." >&2
    exit 2
fi

PROJECT_DIR=${PROJECT_DIR:-"$HOME/dai-research"}
DATA_DIR=${DATA_DIR:-"$PROJECT_DIR/data/clutrr"}
REPORT_PATH=${REPORT_PATH:-"$PROJECT_DIR/results/validation/clutrr_compositions.json"}

if [[ ! -d "$DATA_DIR" ]]; then
    echo "ERROR: CLUTRR data directory not found: $DATA_DIR" >&2
    exit 1
fi

module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
export PYTHONNOUSERSITE=1

cd "$PROJECT_DIR"
"$PROJECT_DIR/venv/bin/python" scripts/validate_clutrr_compositions.py \
    --data-dir "$DATA_DIR" \
    --output "$REPORT_PATH"