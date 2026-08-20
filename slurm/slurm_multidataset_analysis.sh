#!/bin/bash
#SBATCH --job-name=dai_multi_analysis
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/multidataset_analysis_%j.out
#SBATCH --error=logs/multidataset_analysis_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=adetayo@uga.edu

set -euo pipefail
shopt -s nullglob

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: Submit this script with sbatch." >&2
    exit 2
fi

PROJECT_DIR=${PROJECT_DIR:?PROJECT_DIR must identify the matrix source snapshot}
SOURCE_SNAPSHOT_ID=${SOURCE_SNAPSHOT_ID:?SOURCE_SNAPSHOT_ID is required}
SCRATCH_DIR=${SCRATCH_DIR:-"/scratch/$USER/dai-research"}
VENV_DIR=${VENV_DIR:-"$HOME/dai-research/venv"}
EXPERIMENT_ROOT="$SCRATCH_DIR/experiments/publication"
REPORT_ROOT="$SCRATCH_DIR/results/publication_multidataset/$SOURCE_SNAPSHOT_ID/analysis_$SLURM_JOB_ID"
SEEDS=(42 123 456 789 1024 2027 4099 7919 104729 130363)
MATRIX_PATH="$PROJECT_DIR/configs/publication/multidataset_matrix.json"

module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
source "$VENV_DIR/bin/activate"
export PYTHONNOUSERSITE=1
cd "$PROJECT_DIR"
read -ra METHODS <<< "$(python scripts/unified_experiment_matrix.py "$MATRIX_PATH" --methods)"
mapfile -t GROUPS < <(python scripts/unified_experiment_matrix.py "$MATRIX_PATH" --benchmarks)
PROPOSED_METHOD=$(python scripts/unified_experiment_matrix.py "$MATRIX_PATH" --proposed)
if [[ ${#METHODS[@]} -ne 12 || ${#GROUPS[@]} -ne 6 ]]; then
    echo "ERROR: Expected 12 methods and 6 benchmark settings." >&2
    exit 2
fi

PRIMARY_CONTROLS=(random_init_t5 reference_t5 tree_linearized_t5 random_structure shuffled_structure simple_consistency)
PRIMARY_REPORTS=()
for GROUP in "${GROUPS[@]}"; do
    EXPERIMENT_DIR="$EXPERIMENT_ROOT/$GROUP"
    REPORT_DIR="$REPORT_ROOT/$GROUP"
    mkdir -p "$REPORT_DIR"
    python scripts/validate_publication_artifacts.py \
        --require-provenance \
        --experiment-dir "$EXPERIMENT_DIR" \
        --methods "${METHODS[@]}" \
        --seeds "${SEEDS[@]}" \
        --output "$REPORT_DIR/artifact_manifest.json"
    for METHOD in "${METHODS[@]}"; do
        python scripts/aggregate_results.py \
            --input-pattern "$EXPERIMENT_DIR/$METHOD/seed_*/results_seed*.json" \
            --output "$REPORT_DIR/${METHOD}_aggregated.json"
    done
    PROPOSED_FILES=("$EXPERIMENT_DIR"/$PROPOSED_METHOD/seed_*/predictions_seed*.jsonl)
    for CONTROL in "${METHODS[@]}"; do
        if [[ "$CONTROL" == "$PROPOSED_METHOD" ]]; then
            continue
        fi
        CONTROL_FILES=("$EXPERIMENT_DIR"/$CONTROL/seed_*/predictions_seed*.jsonl)
        COMPARISON="$REPORT_DIR/${PROPOSED_METHOD}_vs_${CONTROL}.json"
        python scripts/compare_experiments.py \
            --method-a "${PROPOSED_FILES[@]}" \
            --method-b "${CONTROL_FILES[@]}" \
            --output "$COMPARISON"
        if [[ "$GROUP" == "scan/length" ]]; then
            for PRIMARY_CONTROL in "${PRIMARY_CONTROLS[@]}"; do
                if [[ "$CONTROL" == "$PRIMARY_CONTROL" ]]; then
                    PRIMARY_REPORTS+=("$COMPARISON")
                fi
            done
        fi
    done
    python scripts/analyze_composition_violations.py \
        --predictions "${PROPOSED_FILES[@]}" \
        --output "$REPORT_DIR/composition_violation_analysis.json"
done

python scripts/summarize_comparisons.py \
    --reports "${PRIMARY_REPORTS[@]}" \
    --expected-dataset scan \
    --expected-split length \
    --expected-family-size 6 \
    --output "$REPORT_ROOT/primary_scan_length_holm_m6.json"

# Secondary families are corrected separately within each benchmark setting.
for GROUP in "${GROUPS[@]}"; do
    REPORTS=("$REPORT_ROOT/$GROUP/${PROPOSED_METHOD}_vs_"*.json)
    python scripts/summarize_comparisons.py \
        --reports "${REPORTS[@]}" \
        --expected-family-size 11 \
        --output "$REPORT_ROOT/$GROUP/secondary_holm_m11.json"
done
