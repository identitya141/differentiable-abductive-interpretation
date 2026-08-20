#!/bin/bash
#SBATCH --job-name=dai_num_smoke
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:H100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/numerical_smoke_%j.out
#SBATCH --error=logs/numerical_smoke_%j.err

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: Submit this script with sbatch; do not run it on a login node." >&2
    exit 2
fi

PROJECT_DIR=${PROJECT_DIR:-"$HOME/dai-research"}
SCRATCH_DIR=${SCRATCH_DIR:-"/scratch/$USER/dai-research"}
VENV_DIR=${VENV_DIR:-"$PROJECT_DIR/venv"}
SCAN_DATA_DIR="$SCRATCH_DIR/data/scan"
OUTPUT_DIR="$SCRATCH_DIR/experiments/numerical_smoke"
JOB_TMP_DIR="$SCRATCH_DIR/tmp/$SLURM_JOB_ID"
HF_CACHE_DIR="$SCRATCH_DIR/cache/huggingface"

mkdir -p "$OUTPUT_DIR" "$JOB_TMP_DIR" "$HF_CACHE_DIR" "$SCRATCH_DIR/logs"

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
export PYTHONNOUSERSITE=1

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HOME="$HF_CACHE_DIR"
export TRANSFORMERS_CACHE="$HF_CACHE_DIR/transformers"
export HF_DATASETS_CACHE="$HF_CACHE_DIR/datasets"
export TMPDIR="$JOB_TMP_DIR"
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=42

cd "$PROJECT_DIR"

python -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' || {
    echo "ERROR: CUDA is unavailable; refusing to train." >&2
    exit 2
}

python scripts/train.py \
    --config configs/experiments/scan_numerical_smoke.json \
    --seed 42 \
    --override "output_dir=$OUTPUT_DIR,data.data_dir=$SCAN_DATA_DIR,data.num_workers=8" \
    2>&1 | tee "$SCRATCH_DIR/logs/numerical_smoke_${SLURM_JOB_ID}.log"

RESULT_PATH="$OUTPUT_DIR/seed_42/results_seed42.json"
python - "$RESULT_PATH" <<'PY'
import json
import math
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    results = json.load(handle)

if results.get("optimizer_updates") != 200:
    raise SystemExit(f"Expected 200 optimizer updates, got {results.get('optimizer_updates')}")

for key in ("final_loss", "training_wall_clock_seconds", "peak_cuda_memory_bytes"):
    value = results.get(key)
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise SystemExit(f"Non-finite or missing result {key}: {value}")

print("NUMERICAL_SMOKE_PASSED")
PY
