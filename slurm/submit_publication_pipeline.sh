#!/bin/bash
# Backward-compatible alias for the sole canonical publication coordinator.

set -euo pipefail

if ! command -v sbatch >/dev/null 2>&1; then
    echo "ERROR: sbatch is unavailable; run this script on Sapelo." >&2
    exit 2
fi

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_DIR"
echo "submit_publication_pipeline.sh is an alias for the canonical ten-seed multidataset pipeline."
exec sbatch slurm/submit_multidataset_pipeline.sh
