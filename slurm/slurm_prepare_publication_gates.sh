#!/bin/bash
#SBATCH --job-name=dai_gate_manifest
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=logs/publication_gate_manifest_%j.out
#SBATCH --error=logs/publication_gate_manifest_%j.err

set -euo pipefail

REFERENCE_SOURCE=${REFERENCE_SOURCE:?REFERENCE_SOURCE is required}
OVERFIT_SOURCE=${OVERFIT_SOURCE:?OVERFIT_SOURCE is required}
SCRATCH_DIR=${SCRATCH_DIR:-"/scratch/$USER/dai-research"}
OUTPUT_DIR=${GATE_DIR:-"$SCRATCH_DIR/results/publication_gates"}

mkdir -p "$OUTPUT_DIR"
cp "$REFERENCE_SOURCE" "$OUTPUT_DIR/reference_equivalence.json"
cp "$OVERFIT_SOURCE" "$OUTPUT_DIR/small_overfit.json"
python - "$OUTPUT_DIR/reference_equivalence.json" "$OUTPUT_DIR/small_overfit.json" <<'PY'
import json
import sys
for path in sys.argv[1:]:
    report = json.load(open(path, encoding="utf-8"))
    if report.get("passed") is not True or report.get("status") != "passed":
        raise SystemExit(f"Gate did not pass: {path}")
PY
(cd "$OUTPUT_DIR" && sha256sum reference_equivalence.json small_overfit.json > SHA256SUMS)
(cd "$OUTPUT_DIR" && sha256sum --check SHA256SUMS)
