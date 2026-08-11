#!/bin/bash
#SBATCH --job-name=dai_gsm8k_structure
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/gsm8k_structure_%j.out
#SBATCH --error=logs/gsm8k_structure_%j.err

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: Submit this script with sbatch; do not run it on a login node." >&2
    exit 2
fi

PROJECT_DIR=${PROJECT_DIR:-"$HOME/dai-research"}
VENV_DIR=${VENV_DIR:-"$PROJECT_DIR/venv"}
DATA_DIR=${DATA_DIR:-"$PROJECT_DIR/data/gsm8k/main"}
REPORT_PATH=${REPORT_PATH:-"$PROJECT_DIR/results/validation/gsm8k_structure.json"}

module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
export PYTHONNOUSERSITE=1

cd "$PROJECT_DIR"
"$VENV_DIR/bin/python" scripts/validate_gsm8k_structure.py \
    --data-dir "$DATA_DIR" \
    --output "$REPORT_PATH"