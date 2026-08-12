#!/bin/bash
#SBATCH --job-name=dai_pub_analysis
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/publication_analysis_%j.out
#SBATCH --error=logs/publication_analysis_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=adetayo@uga.edu

# Generate paired seed-level and example-level reports after every matrix task
# succeeds. Submit with an afterok dependency on the array job:
#   MATRIX_JOB=$(sbatch --parsable slurm/slurm_publication_matrix.sh)
#   sbatch --dependency="afterok:$MATRIX_JOB" slurm/slurm_publication_analysis.sh

set -euo pipefail
shopt -s nullglob

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: Submit this script with sbatch; do not run it on a login node." >&2
    exit 2
fi

PROJECT_DIR=${PROJECT_DIR:-"$HOME/dai-research"}
SCRATCH_DIR=${SCRATCH_DIR:-"/scratch/$USER/dai-research"}
VENV_DIR=${VENV_DIR:-"$PROJECT_DIR/venv"}
EXPERIMENT_DIR="$SCRATCH_DIR/experiments/publication"
REPORT_DIR="$SCRATCH_DIR/results/publication_statistics"

CONTROLS=(
    scan_random_init_t5
    scan_reference_t5
    scan_tree_linearized
    scan_random_structure
    scan_shuffled_structure
    scan_simple_consistency
)
METHODS=(scan_full_contrastive "${CONTROLS[@]}")

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "ERROR: Sapelo virtual environment is missing: $VENV_DIR" >&2
    exit 2
fi

module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
source "$VENV_DIR/bin/activate"

mkdir -p "$REPORT_DIR"
cd "$PROJECT_DIR"

GROUND_FILES=("$EXPERIMENT_DIR"/scan_full_contrastive/seed_*/predictions_seed*.jsonl)
GROUND_COUNT=${#GROUND_FILES[@]}
if [[ "$GROUND_COUNT" -ne 5 ]]; then
    echo "ERROR: Expected 5 grounded prediction files, found $GROUND_COUNT." >&2
    exit 1
fi

echo "Generating publication statistics on $SLURM_NODELIST"
echo "Grounded artifacts: $GROUND_COUNT"

python scripts/validate_publication_artifacts.py \
    --experiment-dir "$EXPERIMENT_DIR" \
    --methods "${METHODS[@]}" \
    --seeds 42 123 456 789 1024 \
    --output "$REPORT_DIR/artifact_manifest.json"
python scripts/analyze_composition_violations.py \
    --predictions "${GROUND_FILES[@]}" \
    --output "$REPORT_DIR/composition_violation_analysis.json"
ALL_PREDICTION_FILES=("$EXPERIMENT_DIR"/scan_*/seed_*/predictions_seed*.jsonl)
python scripts/analyze_prediction_failures.py \
    --predictions "${ALL_PREDICTION_FILES[@]}" \
    --output "$REPORT_DIR/exploratory_failure_analysis.json"

for METHOD in "${METHODS[@]}"; do
    RESULT_FILES=("$EXPERIMENT_DIR"/"$METHOD"/seed_*/results_seed*.json)
    if [[ "${#RESULT_FILES[@]}" -ne 5 ]]; then
        echo "ERROR: Expected 5 result files for $METHOD, found ${#RESULT_FILES[@]}." >&2
        exit 1
    fi
    python scripts/aggregate_results.py \
        --input-pattern "$EXPERIMENT_DIR/$METHOD/seed_*/results_seed*.json" \
        --output "$REPORT_DIR/${METHOD}_aggregated.json"
done

for CONTROL in "${CONTROLS[@]}"; do
    CONTROL_FILES=("$EXPERIMENT_DIR"/"$CONTROL"/seed_*/predictions_seed*.jsonl)
    CONTROL_COUNT=${#CONTROL_FILES[@]}
    if [[ "$CONTROL_COUNT" -ne 5 ]]; then
        echo "ERROR: Expected 5 prediction files for $CONTROL, found $CONTROL_COUNT." >&2
        exit 1
    fi

    echo "Comparing scan_full_contrastive with $CONTROL"
    python scripts/compare_experiments.py \
        --method-a "${GROUND_FILES[@]}" \
        --method-b "${CONTROL_FILES[@]}" \
        --output "$REPORT_DIR/full_contrastive_vs_${CONTROL}.json"
done

COMPARISON_REPORTS=("$REPORT_DIR"/full_contrastive_vs_*.json)
if [[ "${#COMPARISON_REPORTS[@]}" -ne "${#CONTROLS[@]}" ]]; then
    echo "ERROR: Expected ${#CONTROLS[@]} pairwise reports, found ${#COMPARISON_REPORTS[@]}." >&2
    exit 1
fi
python scripts/summarize_comparisons.py \
    --reports "${COMPARISON_REPORTS[@]}" \
    --output "$REPORT_DIR/primary_comparisons_holm.json"
python scripts/generate_figures.py \
    --results-dir "$REPORT_DIR" \
    --output-dir "$REPORT_DIR/figures" \
    --figure depth_accuracy
python scripts/generate_breakdown_tables.py \
    --experiment-dir "$EXPERIMENT_DIR" \
    --methods "${METHODS[@]}" \
    --seeds 42 123 456 789 1024 \
    --output-dir "$REPORT_DIR/tables"

echo "Statistical reports written to $REPORT_DIR"