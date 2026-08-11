#!/bin/bash
#SBATCH --job-name=dai_nan_diag
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:12:00
#SBATCH --output=logs/nan_diag_%j.out
#SBATCH --error=logs/nan_diag_%j.err

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: Submit this script with sbatch; do not run it on a login node." >&2
    exit 2
fi

PROJECT_DIR=${PROJECT_DIR:-"$HOME/dai-research"}
SCRATCH_DIR=${SCRATCH_DIR:-"/scratch/$USER/dai-research"}
VENV_DIR=${VENV_DIR:-"$PROJECT_DIR/venv"}
HF_CACHE_DIR="$SCRATCH_DIR/cache/huggingface"
JOB_TMP_DIR="$SCRATCH_DIR/tmp/$SLURM_JOB_ID"

mkdir -p "$HF_CACHE_DIR" "$JOB_TMP_DIR" "$SCRATCH_DIR/logs"

module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
source "$VENV_DIR/bin/activate"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HOME="$HF_CACHE_DIR"
export TRANSFORMERS_CACHE="$HF_CACHE_DIR/transformers"
export HF_DATASETS_CACHE="$HF_CACHE_DIR/datasets"
export TMPDIR="$JOB_TMP_DIR"
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=2026

cd "$PROJECT_DIR"

if ! python -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
    echo "ERROR: CUDA is unavailable; refusing to run the diagnostic." >&2
    exit 2
fi

python scripts/diagnose_fp16_gradients.py \
    --config configs/experiments/scan_grounded.json \
    2>&1 | tee "$SCRATCH_DIR/logs/nan_diag_$SLURM_JOB_ID.log"