#!/bin/bash
#SBATCH --job-name=dai_multi_matrix
#SBATCH --partition=gpu_30d_p
#SBATCH --gres=gpu:H100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=14-00:00:00
#SBATCH --array=0-4%5
#SBATCH --output=logs/multidataset_worker_%A_%a.out
#SBATCH --error=logs/multidataset_worker_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=adetayo@uga.edu

# Keep the number of submitted GPU jobs within Sapelo's per-user QOS limit.
# Each of five H100 workers owns a deterministic strided subset of the 720
# independently checkpointed matrix rows. Re-submission skips completed rows
# and resumes compatible partial rows.

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: Submit this worker with sbatch." >&2
    exit 2
fi

PROJECT_DIR=${PROJECT_DIR:?PROJECT_DIR is required}
RUN_COUNT=${RUN_COUNT:?RUN_COUNT is required}
MATRIX_WORKERS=${MATRIX_WORKERS:?MATRIX_WORKERS is required}

if (( SLURM_ARRAY_TASK_ID >= MATRIX_WORKERS )); then
    echo "ERROR: Worker index is outside MATRIX_WORKERS." >&2
    exit 2
fi

for (( MATRIX_ROW=SLURM_ARRAY_TASK_ID; MATRIX_ROW<RUN_COUNT; MATRIX_ROW+=MATRIX_WORKERS )); do
    echo "Worker $SLURM_ARRAY_TASK_ID starting matrix row $MATRIX_ROW / $((RUN_COUNT - 1))"
    SLURM_ARRAY_TASK_ID=$MATRIX_ROW \
        bash "$PROJECT_DIR/slurm/slurm_multidataset_matrix.sh"
done
