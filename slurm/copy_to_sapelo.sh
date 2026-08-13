#!/bin/bash
#
# Copy project to Sapelo2 and prepare for execution
#
# Usage:
#   bash copy_to_sapelo.sh [ssh_alias]
#   If no argument provided, uses 'sapelo2' as default

set -e

SAPELO_HOST="${1:-sapelo2}"  # Default to 'sapelo2' SSH alias
LOCAL_DIR="."
REMOTE_DIR="~/dai-research"

echo "=================================================="
echo "Copying DAI Project to Sapelo2"
echo "=================================================="
echo "SSH Host: $SAPELO_HOST"
echo "Remote directory: $REMOTE_DIR"
echo ""

# Check if ssh works
echo "Testing connection to Sapelo2..."
if ! ssh -o BatchMode=yes -o ConnectTimeout=5 "$SAPELO_HOST" exit 2>/dev/null; then
    echo "Warning: Cannot connect to Sapelo2 without password."
    echo "You may be prompted for your password multiple times."
    echo "Consider setting up SSH keys for passwordless access."
    echo ""
fi

SOURCE_REVISION=$(git -C "$LOCAL_DIR" rev-parse HEAD)
SOURCE_REMOTE=$(git -C "$LOCAL_DIR" remote get-url origin)
if [[ -n "$(git -C "$LOCAL_DIR" status --porcelain --untracked-files=no)" ]]; then
    echo "ERROR: Commit tracked changes before deployment; Sapelo runs use an immutable Git revision." >&2
    exit 2
fi

echo "Cloning/updating the versioned source tree at $SOURCE_REVISION..."
ssh "$SAPELO_HOST" bash -s -- "$REMOTE_DIR" "$SOURCE_REMOTE" "$SOURCE_REVISION" <<'REMOTE'
set -euo pipefail
remote_dir=$1
source_remote=$2
source_revision=$3
if [[ ! -d "$remote_dir/.git" ]]; then
    if [[ -e "$remote_dir" && -n "$(find "$remote_dir" -mindepth 1 -print -quit)" ]]; then
        echo "ERROR: $remote_dir exists but is not a Git clone; move it aside first." >&2
        exit 2
    fi
    git clone "$source_remote" "$remote_dir"
fi
git -C "$remote_dir" fetch origin
git -C "$remote_dir" checkout --detach "$source_revision"
REMOTE

echo "Staging all canonical publication datasets..."
bash slurm/stage_publication_data.sh "$SAPELO_HOST"

# Make scripts executable
echo ""
echo "Making scripts executable..."
ssh "$SAPELO_HOST" "cd $REMOTE_DIR && chmod +x slurm/*.sh scripts/*.py"

echo ""
echo "=================================================="
echo "Versioned deployment complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Log into Sapelo2:"
echo "   ssh $SAPELO_HOST"
echo ""
echo "2. Run setup (first time only):"
echo "   cd ~/dai-research"
echo "   bash slurm/setup_sapelo.sh"
echo ""
echo "3. Submit a training job:"
echo "   sbatch slurm/slurm_train.sh"
echo ""
echo "4. Check job status:"
echo "   squeue -u $MYID"
echo ""
echo "5. View logs:"
echo "   tail -f ~/dai-research/logs/train_JOBID.out"
echo ""
