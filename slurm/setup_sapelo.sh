#!/bin/bash
#
# Setup script for Sapelo2 HPC cluster
# Run this once to set up your environment on Sapelo
#
# Usage:
#   bash setup_sapelo.sh

set -e  # Exit on error

echo "=================================================="
echo "DAI Research - Sapelo2 Setup"
echo "=================================================="

# Configuration
PROJECT_DIR="$HOME/dai-research"              # Code stays in home (persistent)
SCRATCH_DIR="/scratch/$USER/dai-research"   # Large files go to scratch (fast I/O)
DATA_DIR="$SCRATCH_DIR/data"
VENV_DIR="$PROJECT_DIR/venv"

# Check if on Sapelo
if [[ ! -f /etc/banner ]] || ! grep -q "Sapelo2" /etc/banner 2>/dev/null; then
    echo "Warning: This doesn't appear to be Sapelo2"
    echo "Continuing anyway..."
fi

echo ""
echo "Step 1: Loading required modules..."
module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

echo ""
echo "Step 2: Creating project directories..."
mkdir -p "$PROJECT_DIR"
mkdir -p "$SCRATCH_DIR"
mkdir -p "$DATA_DIR"
mkdir -p "$SCRATCH_DIR/logs"
mkdir -p "$SCRATCH_DIR/checkpoints"
mkdir -p "$SCRATCH_DIR/results"

# Create symlinks from project dir to scratch for convenience
ln -sfn "$SCRATCH_DIR/logs" "$PROJECT_DIR/logs"
ln -sfn "$SCRATCH_DIR/checkpoints" "$PROJECT_DIR/checkpoints"
ln -sfn "$SCRATCH_DIR/results" "$PROJECT_DIR/results"
ln -sfn "$SCRATCH_DIR/data" "$PROJECT_DIR/data"

echo ""
echo "Step 3: Creating Python virtual environment..."
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at $VENV_DIR"
    echo "Skipping creation (will use existing environment)..."
else
    python -m venv --system-site-packages "$VENV_DIR"
    echo "Virtual environment created successfully"
fi

echo ""
echo "Step 4: Activating virtual environment and installing dependencies..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
pip install --upgrade pip setuptools wheel

# PyTorch is provided by the cluster module. Install all other reproducibility
# pins without allowing pip to replace the module's CUDA-enabled Torch stack.
echo "Installing pinned non-PyTorch dependencies..."
grep -Ev '^(torch|torchvision|torchaudio)==' \
    "$PROJECT_DIR/requirements.txt" > "$SCRATCH_DIR/requirements-sapelo.txt"
pip install -r "$SCRATCH_DIR/requirements-sapelo.txt"

# Install project in editable mode
if [ -f "$PROJECT_DIR/setup.py" ] || [ -f "$PROJECT_DIR/pyproject.toml" ]; then
    echo "Installing project package..."
    pip install --no-deps -e "$PROJECT_DIR"
fi

echo ""
echo "Step 5: Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
python -c "import datasets; print(f'Datasets version: {datasets.__version__}')"

echo ""
echo "=================================================="
echo "Setup complete!"
echo "=================================================="
echo ""
echo "Project directory (code): $PROJECT_DIR"
echo "Scratch directory (data): $SCRATCH_DIR"
echo "Virtual environment: $VENV_DIR"
echo "Data directory: $DATA_DIR"
echo ""
echo "Storage layout:"
echo "  - Code: $PROJECT_DIR (persistent, backed up)"
echo "  - Checkpoints: $SCRATCH_DIR/checkpoints (fast I/O, temporary)"
echo "  - Data: $SCRATCH_DIR/data (fast I/O, temporary)"
echo "  - Logs: $SCRATCH_DIR/logs (fast I/O, temporary)"
echo "  Note: Symlinks created in project dir for easy access"
echo ""
echo "Next steps:"
echo "1. Copy your code to: $PROJECT_DIR"
echo "2. Activate environment: source $VENV_DIR/bin/activate"
echo "3. Submit jobs using: sbatch slurm_train.sh"
echo ""
echo "To activate in future sessions:"
echo "  module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1"
echo "  source $VENV_DIR/bin/activate"
echo ""
