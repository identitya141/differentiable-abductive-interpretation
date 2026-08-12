#!/bin/bash
#SBATCH --job-name=dai_scan_secondary
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-13%5
#SBATCH --output=logs/scan_secondary_%A_%a.out
#SBATCH --error=logs/scan_secondary_%A_%a.err

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: Submit this script with sbatch; do not run it on a login node." >&2
    exit 2
fi

PROJECT_DIR=${PROJECT_DIR:-"$HOME/dai-research"}
SCRATCH_DIR=${SCRATCH_DIR:-"/scratch/$USER/dai-research"}
VENV_DIR=${VENV_DIR:-"$PROJECT_DIR/venv"}
SCAN_DATA_DIR="$SCRATCH_DIR/data/scan"
HF_CACHE_DIR="$SCRATCH_DIR/cache/huggingface"

CONFIGS=(
    scan_corruption_000
    scan_corruption_010
    scan_corruption_025
    scan_corruption_050
    scan_corruption_075
    scan_corruption_100
    scan_lambda_0000005
    scan_lambda_000001
    scan_lambda_000002
    scan_lambda_000004
    scan_lambda_000008
    scan_nonce_reference_t5
    scan_nonce_full_contrastive
    scan_topology_corruption
)
SEEDS=(42 123 456 789 1024)

if (( SLURM_ARRAY_TASK_ID >= ${#CONFIGS[@]} )); then
    echo "ERROR: Array task $SLURM_ARRAY_TASK_ID is outside the matrix." >&2
    exit 2
fi

CONFIG_NAME=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
CONFIG_PATH="$PROJECT_DIR/configs/experiments/$CONFIG_NAME.json"
OUTPUT_DIR="$SCRATCH_DIR/experiments/scan_secondary/$CONFIG_NAME"
TASK_LOG_DIR="$SCRATCH_DIR/logs/scan_secondary"

mkdir -p "$OUTPUT_DIR" "$TASK_LOG_DIR" "$HF_CACHE_DIR"

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
export TOKENIZERS_PARALLELISM=false

cd "$PROJECT_DIR"

if ! python -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
    echo "ERROR: CUDA is unavailable; refusing to train." >&2
    exit 2
fi

for SEED in "${SEEDS[@]}"; do
    RESULT_PATH="$OUTPUT_DIR/seed_$SEED/results_seed${SEED}.json"
    if [[ -s "$RESULT_PATH" ]]; then
        echo "Skipping completed seed $SEED: $RESULT_PATH"
        continue
    fi
    export PYTHONHASHSEED=$SEED
    python scripts/train.py \
        --config "$CONFIG_PATH" \
        --seed "$SEED" \
        --override "output_dir=$OUTPUT_DIR,training.fp16=true,data.num_workers=8,data.data_dir=$SCAN_DATA_DIR" \
        2>&1 | tee "$TASK_LOG_DIR/${CONFIG_NAME}_seed${SEED}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"
done
