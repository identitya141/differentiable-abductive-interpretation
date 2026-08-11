# =============================================================================
# DAI Research - Docker Environment
# =============================================================================
#
# This Dockerfile provides a fully reproducible environment for running
# all experiments. It pins all system dependencies and Python packages.
#
# Usage:
#   docker build -t dai-research .
#   docker run --gpus all -v $(pwd):/workspace dai-research make reproduce-all
#
# =============================================================================

# Use NVIDIA's official CUDA image with Ubuntu 22.04
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /workspace

# =============================================================================
# System Dependencies
# =============================================================================

RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build essentials
    build-essential \
    cmake \
    git \
    wget \
    curl \
    unzip \
    # Python
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    # For some packages
    libffi-dev \
    libssl-dev \
    # Visualization (optional, for generating figures)
    libgl1-mesa-glx \
    libglib2.0-0 \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# =============================================================================
# Python Environment
# =============================================================================

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Copy requirements first for better caching
COPY requirements.txt /workspace/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA 11.8 support
RUN pip install --no-cache-dir \
    torch==2.1.2+cu118 \
    torchvision==0.16.2+cu118 \
    torchaudio==2.1.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# =============================================================================
# Project Setup
# =============================================================================

# Copy project files
COPY . /workspace/

# Install project in development mode
RUN pip install -e .

# Create necessary directories
RUN mkdir -p /workspace/data \
             /workspace/checkpoints \
             /workspace/results \
             /workspace/logs

# =============================================================================
# Environment Variables
# =============================================================================

# Set CUDA visible devices (can be overridden at runtime)
ENV CUDA_VISIBLE_DEVICES=0

# Set Python environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/workspace:$PYTHONPATH

# Disable tokenizers parallelism warning
ENV TOKENIZERS_PARALLELISM=false

# Set random seed for reproducibility
ENV DAI_SEED=42

# =============================================================================
# Health Check
# =============================================================================

RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}')" && \
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" && \
    python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"

# =============================================================================
# Entry Point
# =============================================================================

# Default command: validate the released implementation.
CMD ["pytest", "-q"]
