#!/bin/bash
#SBATCH --job-name=dai_data_validate
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/publication_data_validate_%j.out
#SBATCH --error=logs/publication_data_validate_%j.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:?PROJECT_DIR is required}
SCRATCH_DIR=${SCRATCH_DIR:-"/scratch/$USER/dai-research"}
VENV_DIR=${VENV_DIR:-"$HOME/dai-research/venv"}
OUTPUT_DIR=${VALIDATION_DIR:-"$SCRATCH_DIR/results/publication_validations"}

module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
source "$VENV_DIR/bin/activate"
export PYTHONNOUSERSITE=1
cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_DIR"

python scripts/validate_scan_compositions.py \
    --data-dir "$SCRATCH_DIR/data/scan" --split length \
    --output "$OUTPUT_DIR/scan.json"
python scripts/validate_cogs_compositions.py \
    --data-dir "$SCRATCH_DIR/data/cogs/COGS-main/data" \
    --output "$OUTPUT_DIR/cogs.json"
python scripts/validate_slog.py \
    --data-dir "$SCRATCH_DIR/data/slog" \
    --output "$OUTPUT_DIR/slog.json"
python scripts/validate_cfq_compositions.py \
    --data-dir "$SCRATCH_DIR/data/cfq" \
    --output "$OUTPUT_DIR/cfq.json"
python scripts/validate_publication_transformations.py \
    --data-root "$SCRATCH_DIR/data" \
    --output "$OUTPUT_DIR/tokenized_transformations.json"

python - "$OUTPUT_DIR" <<'PY'
import json
from pathlib import Path
import sys

directory = Path(sys.argv[1])
for dataset in ("scan", "cogs", "slog", "cfq"):
    path = directory / f"{dataset}.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["dataset"] = dataset
    if dataset == "scan":
        report["passed"] = report.get("composition_coverage", 0.0) >= 0.99
    if report.get("passed") is not True:
        raise SystemExit(f"Dataset validation did not pass: {path}")
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

(cd "$OUTPUT_DIR" && sha256sum scan.json cogs.json slog.json cfq.json tokenized_transformations.json > SHA256SUMS)
(cd "$OUTPUT_DIR" && sha256sum --check SHA256SUMS)
