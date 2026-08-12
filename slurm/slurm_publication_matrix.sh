#!/bin/bash
#SBATCH --job-name=dai_pub_matrix
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-6%5
#SBATCH --output=logs/publication_%A_%a.out
#SBATCH --error=logs/publication_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=adetayo@uga.edu

# Run the seven primary SCAN methods over five paired seeds. Each array
# task owns one configuration and runs its five seeds sequentially on one A100.
#
# Submit all 35 runs as seven configuration tasks:
#   mkdir -p logs
#   sbatch slurm/slurm_publication_matrix.sh
#
# Retry one failed configuration task; completed seed artifacts are skipped:
#   sbatch --array=TASK_ID slurm/slurm_publication_matrix.sh
#
# Override the maximum concurrent tasks at submission time:
#   sbatch --array=0-6%3 slurm/slurm_publication_matrix.sh

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: Submit this script with sbatch; do not run it on a login node." >&2
    exit 2
fi

PROJECT_DIR=${PROJECT_DIR:-"$HOME/dai-research"}
SCRATCH_DIR=${SCRATCH_DIR:-"/scratch/$USER/dai-research"}
VENV_DIR=${VENV_DIR:-"$PROJECT_DIR/venv"}

if [[ ! -d "$PROJECT_DIR" ]]; then
    echo "ERROR: Project directory does not exist: $PROJECT_DIR" >&2
    exit 2
fi
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "ERROR: Sapelo virtual environment is missing: $VENV_DIR" >&2
    exit 2
fi

module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
source "$VENV_DIR/bin/activate"

if [[ -z "${REFERENCE_EQUIVALENCE_REPORT:-}" ]]; then
    echo "ERROR: REFERENCE_EQUIVALENCE_REPORT is required; submit through slurm/submit_publication_pipeline.sh." >&2
    exit 2
fi
if [[ ! -s "$REFERENCE_EQUIVALENCE_REPORT" ]]; then
    echo "ERROR: Reference-equivalence report is missing or empty: $REFERENCE_EQUIVALENCE_REPORT" >&2
    exit 2
fi
python - "$REFERENCE_EQUIVALENCE_REPORT" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    report = json.load(handle)
if report.get("passed") is not True or report.get("status") != "passed":
    raise SystemExit(f"Reference-equivalence gate did not pass: {sys.argv[1]}")
PY

if [[ -z "${OVERFIT_REPORT:-}" ]]; then
    echo "ERROR: OVERFIT_REPORT is required; submit through slurm/submit_publication_pipeline.sh." >&2
    exit 2
fi
if [[ ! -s "$OVERFIT_REPORT" ]]; then
    echo "ERROR: Overfit gate report is missing or empty: $OVERFIT_REPORT" >&2
    exit 2
fi
python - "$OVERFIT_REPORT" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    report = json.load(handle)
if report.get("passed") is not True or report.get("status") != "passed":
    raise SystemExit(f"Overfit gate did not pass: {sys.argv[1]}")
PY

CONFIGS=(
    scan_random_init_t5
    scan_reference_t5
    scan_tree_linearized
    scan_random_structure
    scan_shuffled_structure
    scan_simple_consistency
    scan_full_contrastive
)
SEEDS=(42 123 456 789 1024)

if (( SLURM_ARRAY_TASK_ID >= ${#CONFIGS[@]} )); then
    echo "ERROR: Array task $SLURM_ARRAY_TASK_ID is outside the experiment matrix." >&2
    exit 2
fi

CONFIG_NAME=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
CONFIG_PATH="$PROJECT_DIR/configs/experiments/${CONFIG_NAME}.json"
OUTPUT_DIR="$SCRATCH_DIR/experiments/publication/${CONFIG_NAME}"
TASK_LOG_DIR="$SCRATCH_DIR/logs/publication"
HF_CACHE_DIR="$SCRATCH_DIR/cache/huggingface"
SCAN_DATA_DIR="$SCRATCH_DIR/data/scan"
SCAN_TRAIN="$SCAN_DATA_DIR/length/tasks_train_length.txt"
SCAN_TEST="$SCAN_DATA_DIR/length/tasks_test_length.txt"

mkdir -p "$OUTPUT_DIR" "$TASK_LOG_DIR" "$HF_CACHE_DIR"

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "ERROR: Experiment config does not exist: $CONFIG_PATH" >&2
    exit 2
fi
if [[ ! -s "$SCAN_TRAIN" || ! -s "$SCAN_TEST" ]]; then
    echo "ERROR: SCAN length data is not staged under $SCAN_DATA_DIR/length." >&2
    echo "Run slurm/copy_to_sapelo.sh from the local project before submitting." >&2
    exit 2
fi

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HOME="$HF_CACHE_DIR"
export TRANSFORMERS_CACHE="$HF_CACHE_DIR/transformers"
export HF_DATASETS_CACHE="$HF_CACHE_DIR/datasets"
export TOKENIZERS_PARALLELISM=false

cd "$PROJECT_DIR"

echo "========================================================================"
echo "DAI publication experiment"
echo "========================================================================"
echo "Job:          $SLURM_JOB_ID"
echo "Array task:   $SLURM_ARRAY_TASK_ID / 6"
echo "Node:         $SLURM_NODELIST"
echo "Config:       $CONFIG_NAME"
echo "Seeds:        ${SEEDS[*]}"
echo "Output root:  $OUTPUT_DIR"
echo "Started:      $(date --iso-8601=seconds)"
echo "========================================================================"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; refusing to train on a non-GPU node")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
PY

for SEED in "${SEEDS[@]}"; do
    RESULT_PATH="$OUTPUT_DIR/seed_$SEED/results_seed${SEED}.json"
    TASK_LOG="$TASK_LOG_DIR/${CONFIG_NAME}_seed${SEED}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"

    if [[ -s "$RESULT_PATH" ]]; then
        echo "Skipping completed seed $SEED: $RESULT_PATH"
        continue
    fi

    export PYTHONHASHSEED=$SEED
    echo "Starting $CONFIG_NAME seed $SEED at $(date --iso-8601=seconds)"
    python scripts/train.py \
        --config "$CONFIG_PATH" \
        --seed "$SEED" \
        --override "output_dir=$OUTPUT_DIR,data.num_workers=8,data.data_dir=$SCAN_DATA_DIR" \
        2>&1 | tee "$TASK_LOG"

    if [[ ! -s "$RESULT_PATH" ]]; then
        echo "ERROR: Training exited without producing $RESULT_PATH" >&2
        exit 1
    fi
    echo "Completed seed $SEED: $RESULT_PATH"
done

echo "Completed configuration: $CONFIG_NAME at $(date --iso-8601=seconds)"