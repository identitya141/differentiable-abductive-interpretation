#!/bin/bash
#SBATCH --job-name=dai_matrix_submit
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/matrix_submit_%j.out
#SBATCH --error=logs/matrix_submit_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=adetayo@uga.edu

# This second-stage compute-node coordinator runs only after all gates finish.
# At that point the prerequisite jobs have left the queue, leaving enough QOS
# capacity for the five-worker H100 array and its dependent analysis job.

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: Submit this stage through sbatch." >&2
    exit 2
fi

PROJECT_DIR=${PROJECT_DIR:?PROJECT_DIR is required}
SOURCE_SNAPSHOT_ID=${SOURCE_SNAPSHOT_ID:?SOURCE_SNAPSHOT_ID is required}
SCRATCH_DIR=${SCRATCH_DIR:-"/scratch/$USER/dai-research"}
RUN_COUNT=${RUN_COUNT:?RUN_COUNT is required}
MATRIX_WORKERS=${MATRIX_WORKERS:?MATRIX_WORKERS is required}

if (( MATRIX_WORKERS < 1 || MATRIX_WORKERS > 5 )); then
    echo "ERROR: MATRIX_WORKERS must be between 1 and 5." >&2
    exit 2
fi

submit_job() {
    local submission
    if ! submission=$(sbatch --parsable "$@"); then
        echo "ERROR: sbatch submission failed: $*" >&2
        return 1
    fi
    [[ -n "$submission" ]] || return 1
    printf '%s\n' "${submission%%;*}"
}

MATRIX_JOB=$(submit_job \
    --array="0-$((MATRIX_WORKERS - 1))%$MATRIX_WORKERS" \
    --export=ALL,PROJECT_DIR,SOURCE_SNAPSHOT_ID,SOURCE_GIT_REVISION,SCRATCH_DIR,VENV_DIR,VALIDATION_DIR,GATE_DIR,RUN_COUNT,MATRIX_WORKERS \
    "$PROJECT_DIR/slurm/slurm_multidataset_worker.sh")
ANALYSIS_JOB=$(submit_job \
    --dependency="afterok:$MATRIX_JOB" \
    --export=ALL,PROJECT_DIR,SOURCE_SNAPSHOT_ID,SOURCE_GIT_REVISION,SCRATCH_DIR,VENV_DIR,VALIDATION_DIR,GATE_DIR \
    "$PROJECT_DIR/slurm/slurm_multidataset_analysis.sh")

WORKFLOW_DIR="$SCRATCH_DIR/workflows/$SOURCE_SNAPSHOT_ID"
mkdir -p "$WORKFLOW_DIR"
printf 'matrix_job_id=%s\nanalysis_job_id=%s\n' "$MATRIX_JOB" "$ANALYSIS_JOB" \
    > "$WORKFLOW_DIR/matrix_stage_${SLURM_JOB_ID}.txt"

echo "Matrix workers: $MATRIX_JOB (array 0-$((MATRIX_WORKERS - 1)))"
echo "Analysis:       $ANALYSIS_JOB (afterok:$MATRIX_JOB)"
