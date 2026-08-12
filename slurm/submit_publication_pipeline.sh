#!/bin/bash
# Submit the complete publication pipeline without running project computation.
# Dependency chain:
#   tests -> reference equivalence -> numerical smoke -> small overfit
#   -> publication matrix -> analysis

set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: Run this submission script from a Sapelo login node." >&2
    exit 2
fi
if ! command -v sbatch >/dev/null 2>&1; then
    echo "ERROR: sbatch is unavailable; run this script on Sapelo." >&2
    exit 2
fi

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SCRATCH_DIR=${SCRATCH_DIR:-"/scratch/$USER/dai-research"}
export PROJECT_DIR SCRATCH_DIR
cd "$PROJECT_DIR"
mkdir -p logs

submit_job() {
    local submission
    submission=$(sbatch --parsable "$@")
    printf '%s\n' "${submission%%;*}"
}

TEST_JOB=$(submit_job slurm/slurm_test_suite.sh)
REFERENCE_JOB=$(submit_job \
    --dependency="afterok:$TEST_JOB" \
    slurm/slurm_reference_equivalence.sh)
REFERENCE_REPORT="$SCRATCH_DIR/results/gates/reference_equivalence_${REFERENCE_JOB}.json"
SMOKE_JOB=$(submit_job \
    --dependency="afterok:$REFERENCE_JOB" \
    slurm/slurm_numerical_smoke.sh)
OVERFIT_JOB=$(submit_job \
    --dependency="afterok:$SMOKE_JOB" \
    slurm/slurm_small_overfit.sh)
OVERFIT_REPORT="$SCRATCH_DIR/results/gates/small_overfit_${OVERFIT_JOB}.json"
MATRIX_JOB=$(submit_job \
    --dependency="afterok:$OVERFIT_JOB" \
    --export="ALL,REFERENCE_EQUIVALENCE_REPORT=$REFERENCE_REPORT,OVERFIT_REPORT=$OVERFIT_REPORT" \
    slurm/slurm_publication_matrix.sh)
ANALYSIS_JOB=$(submit_job \
    --dependency="afterok:$MATRIX_JOB" \
    slurm/slurm_publication_analysis.sh)

cat <<EOF
Publication pipeline submitted:
  tests:              $TEST_JOB
    reference gate:     $REFERENCE_JOB
  numerical smoke:    $SMOKE_JOB
  small overfit gate: $OVERFIT_JOB
  publication matrix: $MATRIX_JOB
  analysis:           $ANALYSIS_JOB
    reference report:   $REFERENCE_REPORT
  overfit report:     $OVERFIT_REPORT

Monitor with:
    squeue -j $TEST_JOB,$REFERENCE_JOB,$SMOKE_JOB,$OVERFIT_JOB,$MATRIX_JOB,$ANALYSIS_JOB
EOF