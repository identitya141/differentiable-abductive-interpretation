#!/bin/bash
#SBATCH --job-name=dai_overfit_gate
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:H100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/small_overfit_%j.out
#SBATCH --error=logs/small_overfit_%j.err

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: Submit this script with sbatch; do not run it on a login node." >&2
    exit 2
fi

PROJECT_DIR=${PROJECT_DIR:-"$HOME/dai-research"}
SCRATCH_DIR=${SCRATCH_DIR:-"/scratch/$USER/dai-research"}
VENV_DIR=${VENV_DIR:-"$PROJECT_DIR/venv"}
SCAN_DATA_DIR="$SCRATCH_DIR/data/scan"
OUTPUT_DIR="$SCRATCH_DIR/experiments/gates/small_overfit_$SLURM_JOB_ID"
REPORT_PATH="$SCRATCH_DIR/results/gates/small_overfit_$SLURM_JOB_ID.json"
JOB_TMP_DIR="$SCRATCH_DIR/tmp/$SLURM_JOB_ID"
HF_CACHE_DIR="$SCRATCH_DIR/cache/huggingface"

mkdir -p \
    "$OUTPUT_DIR" \
    "$(dirname "$REPORT_PATH")" \
    "$JOB_TMP_DIR" \
    "$HF_CACHE_DIR" \
    "$SCRATCH_DIR/logs"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "ERROR: Sapelo environment is missing: $VENV_DIR" >&2
    exit 2
fi
if [[ ! -s "$SCAN_DATA_DIR/length/tasks_train_length.txt" ]]; then
    echo "ERROR: SCAN data is missing from $SCAN_DATA_DIR" >&2
    exit 2
fi

module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
source "$VENV_DIR/bin/activate"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HOME="$HF_CACHE_DIR"
export TRANSFORMERS_CACHE="$HF_CACHE_DIR/transformers"
export HF_DATASETS_CACHE="$HF_CACHE_DIR/datasets"
export TMPDIR="$JOB_TMP_DIR"
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=42

cd "$PROJECT_DIR"

python scripts/run_small_overfit.py \
    --config configs/experiments/scan_small_overfit.json \
    --data-dir "$SCAN_DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --report "$REPORT_PATH" \
    --seed 42 \
    --subset-size 24 \
    --minimum-exact-match 0.95 \
    --maximum-loss-ratio 0.25 \
    2>&1 | tee "$SCRATCH_DIR/logs/small_overfit_${SLURM_JOB_ID}.log"

echo "OVERFIT_GATE_REPORT=$REPORT_PATH"
