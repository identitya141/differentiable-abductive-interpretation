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

# Create remote directory
echo "Creating remote directory..."
ssh "$SAPELO_HOST" "mkdir -p $REMOTE_DIR"

# Copy files using rsync (more efficient than scp)
echo "Copying files (this may take a while)..."
rsync -avz --progress \
    --exclude-from='.slurmignore' \
    --exclude='/venv/' \
    --exclude='/.venv/' \
    --exclude='__pycache__/' \
    --exclude='/.git/' \
    --exclude='*.pyc' \
    --exclude='/checkpoints/' \
    --exclude='/logs/' \
    --exclude='/results/' \
    --exclude='/data/' \
    "$LOCAL_DIR/" "$SAPELO_HOST:$REMOTE_DIR/"

# The publication matrix uses the local SCAN grammar files. Stage only this
# compact corpus to scratch; larger datasets remain excluded from project sync.
REMOTE_USER=$(ssh "$SAPELO_HOST" 'id -un')
REMOTE_SCAN_DIR="/scratch/$REMOTE_USER/dai-research/data/scan"
echo "Staging SCAN data to $REMOTE_SCAN_DIR..."
ssh "$SAPELO_HOST" "mkdir -p '$REMOTE_SCAN_DIR'"
rsync -avz --progress \
    "$LOCAL_DIR/data/scan/" "$SAPELO_HOST:$REMOTE_SCAN_DIR/"

# Make scripts executable
echo ""
echo "Making scripts executable..."
ssh "$SAPELO_HOST" "cd $REMOTE_DIR && chmod +x slurm/*.sh scripts/*.py"

echo ""
echo "=================================================="
echo "Copy complete!"
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
