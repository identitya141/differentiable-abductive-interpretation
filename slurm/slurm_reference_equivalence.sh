#!/bin/bash
#SBATCH --job-name=dai_ref_equiv
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=logs/reference_equivalence_%j.out
#SBATCH --error=logs/reference_equivalence_%j.err

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: Submit this script with sbatch; do not run it on a login node." >&2
    exit 2
fi

PROJECT_DIR=${PROJECT_DIR:-"$HOME/dai-research"}
SCRATCH_DIR=${SCRATCH_DIR:-"/scratch/$USER/dai-research"}
VENV_DIR=${VENV_DIR:-"$PROJECT_DIR/venv"}
REPORT_PATH="$SCRATCH_DIR/results/gates/reference_equivalence_$SLURM_JOB_ID.json"
JOB_TMP_DIR="$SCRATCH_DIR/tmp/$SLURM_JOB_ID"
HF_CACHE_DIR="$SCRATCH_DIR/cache/huggingface"

mkdir -p "$(dirname "$REPORT_PATH")" "$JOB_TMP_DIR" "$HF_CACHE_DIR"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "ERROR: Sapelo environment is missing: $VENV_DIR" >&2
    exit 2
fi

module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
source "$VENV_DIR/bin/activate"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$VENV_DIR/lib/python3.11/site-packages:$PROJECT_DIR"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HOME="$HF_CACHE_DIR"
export TRANSFORMERS_CACHE="$HF_CACHE_DIR/transformers"
export HF_DATASETS_CACHE="$HF_CACHE_DIR/datasets"
export TMPDIR="$JOB_TMP_DIR"
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=42

cd "$PROJECT_DIR"

python scripts/run_reference_equivalence.py \
    --config configs/experiments/scan_full_contrastive.json \
    --report "$REPORT_PATH" \
    --maximum-absolute-error 1e-5 \
    --device cpu

echo "REFERENCE_EQUIVALENCE_REPORT=$REPORT_PATH"
