#!/bin/bash
#SBATCH --job-name=dai_multi_submit
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=logs/multidataset_submit_%j.out
#SBATCH --error=logs/multidataset_submit_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=adetayo@uga.edu

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-"$HOME/dai-research"}
SCRATCH_DIR=${SCRATCH_DIR:-"/scratch/$USER/dai-research"}
VENV_DIR=${VENV_DIR:-"$PROJECT_DIR/venv"}
MATRIX_RELATIVE="configs/publication/multidataset_matrix.json"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: Submit this coordinator with sbatch; do not run it directly." >&2
    exit 2
fi
if [[ ! -d "$PROJECT_DIR/.git" || ! -x "$VENV_DIR/bin/python" ]]; then
    echo "ERROR: PROJECT_DIR or its virtual environment is invalid." >&2
    exit 2
fi

cd "$PROJECT_DIR"
echo "Coordinator job: $SLURM_JOB_ID on $SLURM_NODELIST"
SOURCE_REVISION=$(git rev-parse HEAD)
SOURCE_STATE_HASH=$(
    {
        git diff --binary HEAD
        # Runtime data and scratch symlinks are deliberately absent from the
        # immutable source snapshot and must not participate in its hash.
        git ls-files --others --exclude-standard -z \
            --exclude=data --exclude='data/**' \
            --exclude=checkpoints --exclude='checkpoints/**' \
            --exclude=experiments --exclude='experiments/**' \
            --exclude=logs --exclude='logs/**' \
            --exclude=results --exclude='results/**' \
            --exclude=venv --exclude='venv/**' \
            --exclude=.venv --exclude='.venv/**' \
            | sort -z \
            | xargs -0 -r sha256sum
    } | sha256sum | awk '{print $1}'
)
SOURCE_SNAPSHOT_ID="${SOURCE_REVISION:0:12}-${SOURCE_STATE_HASH:0:12}"
SNAPSHOT_ROOT="$SCRATCH_DIR/source_snapshots"
SNAPSHOT_DIR="$SNAPSHOT_ROOT/$SOURCE_SNAPSHOT_ID"

if [[ ! -d "$SNAPSHOT_DIR" ]]; then
    TEMP_SNAPSHOT="$SNAPSHOT_ROOT/.${SOURCE_SNAPSHOT_ID}.$$"
    mkdir -p "$TEMP_SNAPSHOT"
    rsync -a \
        --exclude=.git \
        --exclude=.venv \
        --exclude=venv \
        --exclude=data \
        --exclude=checkpoints \
        --exclude=experiments \
        --exclude=logs \
        --exclude=results \
        "$PROJECT_DIR/" "$TEMP_SNAPSHOT/"
    SOURCE_REVISION="$SOURCE_REVISION" SOURCE_STATE_HASH="$SOURCE_STATE_HASH" \
        SOURCE_SNAPSHOT_ID="$SOURCE_SNAPSHOT_ID" python - "$TEMP_SNAPSHOT/source_manifest.json" <<'PY'
import json
import os
from pathlib import Path
import subprocess
import sys

manifest = {
    "schema_version": 1,
    "source_snapshot_id": os.environ["SOURCE_SNAPSHOT_ID"],
    "git_revision": os.environ["SOURCE_REVISION"],
    "source_state_sha256": os.environ["SOURCE_STATE_HASH"],
    "git_status": subprocess.check_output(
        ["git", "status", "--short"], text=True
    ).splitlines(),
}
Path(sys.argv[1]).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
PY
    chmod -R a-w "$TEMP_SNAPSHOT"
    mv "$TEMP_SNAPSHOT" "$SNAPSHOT_DIR"
fi

RUN_COUNT=$(cd "$SNAPSHOT_DIR" && "$VENV_DIR/bin/python" \
    scripts/unified_experiment_matrix.py "$MATRIX_RELATIVE" --count)
if [[ "$RUN_COUNT" -lt 1 ]]; then
    echo "ERROR: Matrix has no runs." >&2
    exit 2
fi

VALIDATION_DIR="$SCRATCH_DIR/results/publication_validations/${SOURCE_SNAPSHOT_ID}_${SLURM_JOB_ID}"
GATE_DIR="$SCRATCH_DIR/results/publication_gates/${SOURCE_SNAPSHOT_ID}_${SLURM_JOB_ID}"
SEED_COUNT=$(cd "$SNAPSHOT_DIR" && "$VENV_DIR/bin/python" - "$MATRIX_RELATIVE" <<'PY'
import json
import sys
print(len(json.load(open(sys.argv[1], encoding="utf-8"))["seeds"]))
PY
)
EXPORTS="ALL,PROJECT_DIR=$SNAPSHOT_DIR,SOURCE_SNAPSHOT_ID=$SOURCE_SNAPSHOT_ID,SOURCE_GIT_REVISION=$SOURCE_REVISION,SCRATCH_DIR=$SCRATCH_DIR,VENV_DIR=$VENV_DIR,VALIDATION_DIR=$VALIDATION_DIR,GATE_DIR=$GATE_DIR"
submit_job() {
    local submission
    submission=$(sbatch --parsable "$@")
    printf '%s\n' "${submission%%;*}"
}

TEST_JOB=$(submit_job --export="$EXPORTS" \
    "$SNAPSHOT_DIR/slurm/slurm_test_suite.sh")
REFERENCE_JOB=$(submit_job --dependency="afterok:$TEST_JOB" --export="$EXPORTS" \
    "$SNAPSHOT_DIR/slurm/slurm_reference_equivalence.sh")
VALIDATION_JOB=$(submit_job --dependency="afterok:$TEST_JOB" --export="$EXPORTS" \
    "$SNAPSHOT_DIR/slurm/slurm_validate_publication_datasets.sh")
SMOKE_JOB=$(submit_job --dependency="afterok:$REFERENCE_JOB" --export="$EXPORTS" \
    "$SNAPSHOT_DIR/slurm/slurm_numerical_smoke.sh")
OVERFIT_JOB=$(submit_job --dependency="afterok:$SMOKE_JOB" --export="$EXPORTS" \
    "$SNAPSHOT_DIR/slurm/slurm_small_overfit.sh")
REFERENCE_SOURCE="$SCRATCH_DIR/results/gates/reference_equivalence_${REFERENCE_JOB}.json"
OVERFIT_SOURCE="$SCRATCH_DIR/results/gates/small_overfit_${OVERFIT_JOB}.json"
GATE_JOB=$(submit_job --dependency="afterok:$REFERENCE_JOB:$OVERFIT_JOB" \
    --export="$EXPORTS,REFERENCE_SOURCE=$REFERENCE_SOURCE,OVERFIT_SOURCE=$OVERFIT_SOURCE" \
    "$SNAPSHOT_DIR/slurm/slurm_prepare_publication_gates.sh")
MATRIX_JOB=$(submit_job \
    --dependency="afterok:$VALIDATION_JOB:$GATE_JOB" \
    --array="0-$((RUN_COUNT - 1))%${MAX_CONCURRENT:-5}" \
    --export="$EXPORTS" \
    "$SNAPSHOT_DIR/slurm/slurm_multidataset_matrix.sh")
ANALYSIS_JOB=$(submit_job \
    --dependency="afterok:$MATRIX_JOB" \
    --export="$EXPORTS" \
    "$SNAPSHOT_DIR/slurm/slurm_multidataset_analysis.sh")

WORKFLOW_DIR="$SCRATCH_DIR/workflows/$SOURCE_SNAPSHOT_ID"
mkdir -p "$WORKFLOW_DIR"
TEST_JOB="$TEST_JOB" REFERENCE_JOB="$REFERENCE_JOB" VALIDATION_JOB="$VALIDATION_JOB" \
SMOKE_JOB="$SMOKE_JOB" OVERFIT_JOB="$OVERFIT_JOB" GATE_JOB="$GATE_JOB" \
MATRIX_JOB="$MATRIX_JOB" ANALYSIS_JOB="$ANALYSIS_JOB" \
    SOURCE_SNAPSHOT_ID="$SOURCE_SNAPSHOT_ID" SNAPSHOT_DIR="$SNAPSHOT_DIR" \
    python - "$WORKFLOW_DIR/submission_${MATRIX_JOB}.json" <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys

record = {
    "schema_version": 1,
    "submitted_at": datetime.now(timezone.utc).isoformat(),
    "coordinator_job_id": os.environ["SLURM_JOB_ID"],
    "source_snapshot_id": os.environ["SOURCE_SNAPSHOT_ID"],
    "source_snapshot_dir": os.environ["SNAPSHOT_DIR"],
    "test_job_id": os.environ["TEST_JOB"],
    "reference_job_id": os.environ["REFERENCE_JOB"],
    "validation_job_id": os.environ["VALIDATION_JOB"],
    "numerical_smoke_job_id": os.environ["SMOKE_JOB"],
    "overfit_job_id": os.environ["OVERFIT_JOB"],
    "gate_manifest_job_id": os.environ["GATE_JOB"],
    "matrix_job_id": os.environ["MATRIX_JOB"],
    "analysis_job_id": os.environ["ANALYSIS_JOB"],
}
path = Path(sys.argv[1])
with path.open("x", encoding="utf-8") as handle:
    json.dump(record, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

echo "Source snapshot: $SOURCE_SNAPSHOT_ID"
echo "Configurations: $((RUN_COUNT / SEED_COUNT)) (6 benchmarks x 12 methods)"
echo "Training runs:  $RUN_COUNT (ten paired seeds)"
echo "Tests:           $TEST_JOB"
echo "Reference gate:  $REFERENCE_JOB"
echo "Data validation: $VALIDATION_JOB"
echo "Numerical smoke: $SMOKE_JOB"
echo "Overfit gate:    $OVERFIT_JOB"
echo "Gate manifest:   $GATE_JOB"
echo "Matrix job:     $MATRIX_JOB"
echo "Analysis job:   $ANALYSIS_JOB (afterok:$MATRIX_JOB)"
