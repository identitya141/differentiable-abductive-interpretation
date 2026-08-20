#!/bin/bash
#SBATCH --job-name=dai_multi_matrix
#SBATCH --partition=gpu_30d_p
#SBATCH --gres=gpu:H100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=14-00:00:00
# The coordinator supplies the exact --array range from the canonical manifest.
#SBATCH --array=0-0
#SBATCH --output=logs/multidataset_%A_%a.out
#SBATCH --error=logs/multidataset_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=adetayo@uga.edu

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: Submit this script with sbatch; do not run it on a login node." >&2
    exit 2
fi

PROJECT_DIR=${PROJECT_DIR:?PROJECT_DIR must identify an immutable source snapshot}
SOURCE_SNAPSHOT_ID=${SOURCE_SNAPSHOT_ID:?SOURCE_SNAPSHOT_ID is required}
SOURCE_GIT_REVISION=${SOURCE_GIT_REVISION:?SOURCE_GIT_REVISION is required}
SCRATCH_DIR=${SCRATCH_DIR:-"/scratch/$USER/dai-research"}
VENV_DIR=${VENV_DIR:-"$HOME/dai-research/venv"}
MATRIX_PATH=${MATRIX_PATH:-"$PROJECT_DIR/configs/publication/multidataset_matrix.json"}
VALIDATION_DIR=${VALIDATION_DIR:-"$SCRATCH_DIR/results/publication_validations"}
GATE_DIR=${GATE_DIR:-"$SCRATCH_DIR/results/publication_gates"}
REFERENCE_EQUIVALENCE_REPORT="$GATE_DIR/reference_equivalence.json"
OVERFIT_REPORT="$GATE_DIR/small_overfit.json"

if [[ -w "$PROJECT_DIR" ]]; then
    echo "ERROR: Source snapshot must be read-only: $PROJECT_DIR" >&2
    exit 2
fi
if [[ ! -x "$VENV_DIR/bin/python" || ! -s "$MATRIX_PATH" ]]; then
    echo "ERROR: Missing environment or matrix manifest." >&2
    exit 2
fi

module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
source "$VENV_DIR/bin/activate"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$VENV_DIR/lib/python3.11/site-packages:$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_DIR"

if [[ ! -s "$GATE_DIR/SHA256SUMS" || ! -s "$VALIDATION_DIR/SHA256SUMS" ]]; then
    echo "ERROR: Missing checksum manifests for gates or dataset validations." >&2
    exit 2
fi
(cd "$GATE_DIR" && sha256sum --check SHA256SUMS)
(cd "$VALIDATION_DIR" && sha256sum --check SHA256SUMS)
python - "$REFERENCE_EQUIVALENCE_REPORT" "$OVERFIT_REPORT" <<'PY'
import json
import sys

for path in sys.argv[1:]:
    report = json.load(open(path, encoding="utf-8"))
    if report.get("passed") is not True or report.get("status") != "passed":
        raise SystemExit(f"Publication gate did not pass: {path}")
PY

mapfile -t RUN_FIELDS < <(python scripts/unified_experiment_matrix.py \
    "$MATRIX_PATH" --row "$SLURM_ARRAY_TASK_ID")

if [[ ${#RUN_FIELDS[@]} -ne 8 ]]; then
    echo "ERROR: Invalid matrix row $SLURM_ARRAY_TASK_ID" >&2
    exit 2
fi

DATASET=${RUN_FIELDS[0]}
SPLIT=${RUN_FIELDS[1]}
METHOD=${RUN_FIELDS[2]}
RUNNER=${RUN_FIELDS[3]}
CONFIG_PATH="$PROJECT_DIR/${RUN_FIELDS[4]}"
METHOD_OVERRIDE=${RUN_FIELDS[5]}
DATA_DIR="$SCRATCH_DIR/data/${RUN_FIELDS[6]}"
SEED=${RUN_FIELDS[7]}
OUTPUT_DIR="$SCRATCH_DIR/experiments/publication/$DATASET/$SPLIT/$METHOD"
REPORT_DIR="$SCRATCH_DIR/results/publication_multidataset/$DATASET/$SPLIT"
LOG_DIR="$SCRATCH_DIR/logs/publication_multidataset"

if [[ ! -s "$CONFIG_PATH" || ! -d "$DATA_DIR" ]]; then
    echo "ERROR: Missing config or staged dataset: $CONFIG_PATH ; $DATA_DIR" >&2
    exit 2
fi

python - "$VALIDATION_DIR" "$DATASET" <<'PY'
import json
from pathlib import Path
import sys

directory = Path(sys.argv[1])
dataset = sys.argv[2]
matches = []
for path in directory.glob("*.json"):
    report = json.loads(path.read_text(encoding="utf-8"))
    if str(report.get("dataset", "")).lower() == dataset and report.get("passed") is True:
        matches.append(path)
if not matches:
    raise SystemExit(f"No passed validation report for {dataset} under {directory}")
print(f"Validated prerequisite: {matches[0]}")
PY

mkdir -p "$OUTPUT_DIR" "$REPORT_DIR" "$LOG_DIR" "$SCRATCH_DIR/cache/huggingface"
CONFIG_SHA256=$(sha256sum "$CONFIG_PATH" | awk '{print $1}')
MATRIX_SHA256=$(sha256sum "$MATRIX_PATH" | awk '{print $1}')
GATE_MANIFEST_SHA256=$(sha256sum "$GATE_DIR/SHA256SUMS" | awk '{print $1}')
VALIDATION_MANIFEST_SHA256=$(sha256sum "$VALIDATION_DIR/SHA256SUMS" | awk '{print $1}')
HASH_ROOT="$DATA_DIR"
if [[ "$DATASET" == "cfq" ]]; then
    HASH_ROOT="$DATA_DIR/$SPLIT"
fi
if [[ ! -d "$HASH_ROOT" || -z "$(find "$HASH_ROOT" -type f -print -quit)" ]]; then
    echo "ERROR: Expected dataset split directory is missing: $HASH_ROOT" >&2
    exit 2
fi
DATA_SHA256=$(
    find "$HASH_ROOT" -type f -print0 \
        | sort -z \
        | xargs -0 -r sha256sum \
        | sha256sum \
        | awk '{print $1}'
)
export CONFIG_SHA256 MATRIX_SHA256 DATA_SHA256 GATE_MANIFEST_SHA256 VALIDATION_MANIFEST_SHA256 SOURCE_SNAPSHOT_ID SOURCE_GIT_REVISION DATASET SPLIT METHOD RUNNER METHOD_OVERRIDE DATA_DIR

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HOME="$SCRATCH_DIR/cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TOKENIZERS_PARALLELISM=false

cd "$PROJECT_DIR"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; refusing to train")
print(torch.__version__, torch.cuda.get_device_name(0))
PY

for SEED in "$SEED"; do
    RUN_DIR="$OUTPUT_DIR/seed_$SEED"
    RESULT_PATH="$RUN_DIR/results_seed${SEED}.json"
    PREDICTION_PATH="$RUN_DIR/predictions_seed${SEED}.jsonl"
    TASK_LOG="$LOG_DIR/${DATASET}_${SPLIT}_${METHOD}_seed${SEED}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"
    if [[ -s "$RESULT_PATH" && -s "$PREDICTION_PATH" ]]; then
        echo "Skipping completed seed $SEED: $RESULT_PATH"
        continue
    fi
    mkdir -p "$RUN_DIR"
    CONTRACT_PATH="$RUN_DIR/run_contract.json"
    export RUN_CONTRACT_PATH="$CONTRACT_PATH"
    python - "$CONTRACT_PATH" "$SEED" <<'PY'
import json
import os
from pathlib import Path
import sys

path = Path(sys.argv[1])
contract = {
    "schema_version": 1,
    "source_snapshot_id": os.environ["SOURCE_SNAPSHOT_ID"],
    "source_git_revision": os.environ["SOURCE_GIT_REVISION"],
    "matrix_sha256": os.environ["MATRIX_SHA256"],
    "config_sha256": os.environ["CONFIG_SHA256"],
    "data_sha256": os.environ["DATA_SHA256"],
    "gate_manifest_sha256": os.environ["GATE_MANIFEST_SHA256"],
    "validation_manifest_sha256": os.environ["VALIDATION_MANIFEST_SHA256"],
    "dataset": os.environ["DATASET"], "split": os.environ["SPLIT"],
    "method": os.environ["METHOD"], "seed": int(sys.argv[2]),
    "runner": os.environ["RUNNER"], "override": os.environ["METHOD_OVERRIDE"],
    "data_dir": os.environ["DATA_DIR"],
}
if path.exists():
    existing = json.loads(path.read_text(encoding="utf-8"))
    identity_keys = {
        "source_snapshot_id", "source_git_revision", "matrix_sha256",
        "config_sha256", "data_sha256", "dataset", "split", "method",
        "seed", "runner", "override", "data_dir",
    }
    if any(existing.get(key) != contract.get(key) for key in identity_keys):
        raise SystemExit(f"Existing run contract differs; refusing unsafe resume: {path}")
else:
    path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
    export PYTHONHASHSEED=$SEED
    export PUBLICATION_METHOD=$METHOD
    case "$RUNNER" in
        baseline)
            RESUME_ARGS=()
            if compgen -G "$RUN_DIR/checkpoint-*" >/dev/null; then
                RESUME_ARGS+=(--resume-from-checkpoint)
                echo "Resuming baseline from newest checkpoint in $RUN_DIR"
            fi
            python scripts/train_baseline.py \
                --baseline "$METHOD" \
                --config "$CONFIG_PATH" \
                --dataset "$DATASET" \
                --split "$SPLIT" \
                --data-dir "$DATA_DIR" \
                --output-dir "$RUN_DIR" \
                --seed "$SEED" \
                "${RESUME_ARGS[@]}" \
                2>&1 | tee "$TASK_LOG"
            ;;
        dai_control|proposed)
            OVERRIDE="output_dir=$OUTPUT_DIR,data.data_dir=$DATA_DIR"
            if [[ -n "$METHOD_OVERRIDE" ]]; then
                OVERRIDE="$OVERRIDE,$METHOD_OVERRIDE"
            fi
            RESUME_ARGS=()
            LATEST_CHECKPOINT=$(find "$RUN_DIR/checkpoints" -mindepth 1 -maxdepth 1 -type d -name 'epoch_*' -printf '%f\n' 2>/dev/null | sort -t_ -k2,2n | tail -1)
            if [[ -n "$LATEST_CHECKPOINT" ]]; then
                RESUME_ARGS+=(--resume-from-checkpoint "$LATEST_CHECKPOINT")
                echo "Resuming DAI from $LATEST_CHECKPOINT"
            fi
            python scripts/train.py \
                --config "$CONFIG_PATH" \
                --seed "$SEED" \
                --override "$OVERRIDE" \
                "${RESUME_ARGS[@]}" \
                2>&1 | tee "$TASK_LOG"
            ;;
        *)
            echo "ERROR: Unknown runner '$RUNNER' for method '$METHOD'." >&2
            exit 2
            ;;
    esac
    if [[ ! -s "$RESULT_PATH" || ! -s "$PREDICTION_PATH" ]]; then
        echo "ERROR: Missing result or predictions after training: $RUN_DIR" >&2
        exit 1
    fi
done
