#!/bin/bash
#SBATCH --job-name=dai_tests
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/tests_%j.out
#SBATCH --error=logs/tests_%j.err

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: Submit this script with sbatch; do not run it on a login node." >&2
    exit 2
fi

PROJECT_DIR=${PROJECT_DIR:-"$HOME/dai-research"}
SCRATCH_DIR=${SCRATCH_DIR:-"/scratch/$USER/dai-research"}
VENV_DIR=${VENV_DIR:-"$PROJECT_DIR/venv"}
JOB_TMP_DIR="$SCRATCH_DIR/tmp/$SLURM_JOB_ID"

mkdir -p "$JOB_TMP_DIR" "$SCRATCH_DIR/test-results" "$SCRATCH_DIR/cache/huggingface"

module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

export HF_HOME="$SCRATCH_DIR/cache/huggingface"
export TRANSFORMERS_CACHE="$SCRATCH_DIR/cache/huggingface/transformers"
export HF_DATASETS_CACHE="$SCRATCH_DIR/cache/huggingface/datasets"
export TMPDIR="$JOB_TMP_DIR"
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PROJECT_DIR VENV_DIR

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR"
read -ra TEST_PATH_ARGS <<< "${TEST_PATHS:-tests}"
"$VENV_DIR/bin/python" -m pytest "${TEST_PATH_ARGS[@]}" -q \
    -o "cache_dir=$JOB_TMP_DIR/pytest-cache" \
    2>&1 | tee "$SCRATCH_DIR/test-results/tests_$SLURM_JOB_ID.log"
