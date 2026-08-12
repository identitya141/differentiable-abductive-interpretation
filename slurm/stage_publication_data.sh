#!/bin/bash
# Stage the four canonical publication datasets and write a SHA-256 manifest.
set -euo pipefail

SAPELO_HOST="${1:-sapelo2}"
LOCAL_DATA_ROOT="${LOCAL_DATA_ROOT:-data}"
REMOTE_USER=$(ssh "$SAPELO_HOST" 'id -un')
REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-/scratch/$REMOTE_USER/dai-research/data}"

required=(
    "scan/length/tasks_train_length.txt"
    "scan/length/tasks_test_length.txt"
    "cogs/COGS-main/data/train.tsv"
    "cogs/COGS-main/data/dev.tsv"
    "cogs/COGS-main/data/test.tsv"
    "cogs/COGS-main/data/gen.tsv"
    "slog/cogs_LF/train.tsv"
    "slog/cogs_LF/dev.tsv"
    "slog/cogs_LF/test.tsv"
    "slog/generalization_sets/gen_cogsLF.tsv"
)
for relative in "${required[@]}"; do
    if [[ ! -s "$LOCAL_DATA_ROOT/$relative" ]]; then
        echo "ERROR: Missing canonical dataset file: $LOCAL_DATA_ROOT/$relative" >&2
        exit 2
    fi
done
for split in mcd1 mcd2 mcd3; do
    if [[ ! -d "$LOCAL_DATA_ROOT/cfq/$split" ]]; then
        echo "ERROR: Missing canonical CFQ split: $LOCAL_DATA_ROOT/cfq/$split" >&2
        exit 2
    fi
done

ssh "$SAPELO_HOST" "mkdir -p '$REMOTE_DATA_ROOT'"
for dataset in scan cogs slog cfq; do
    rsync -az --delete "$LOCAL_DATA_ROOT/$dataset/" "$SAPELO_HOST:$REMOTE_DATA_ROOT/$dataset/"
done
ssh "$SAPELO_HOST" "cd '$REMOTE_DATA_ROOT' && find scan cogs slog cfq -type f -print0 | sort -z | xargs -0 sha256sum > publication_sha256.txt"
echo "Staged and hashed publication data at $SAPELO_HOST:$REMOTE_DATA_ROOT"
