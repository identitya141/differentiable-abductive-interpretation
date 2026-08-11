#!/bin/bash
#SBATCH --job-name=dai_release_manifest
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=logs/release_manifest_%j.out
#SBATCH --error=logs/release_manifest_%j.err

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: Submit this script with sbatch; do not run it on a login node." >&2
    exit 2
fi

PROJECT_DIR=${PROJECT_DIR:-"$HOME/dai-research"}
OUTPUT=${OUTPUT:-"$PROJECT_DIR/results/release/source_manifest.json"}

module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

cd "$PROJECT_DIR"
"$PROJECT_DIR/venv/bin/python" scripts/build_release_manifest.py \
    --root "$PROJECT_DIR" \
    --output "$OUTPUT"