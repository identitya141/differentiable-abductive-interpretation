# DAI Research - Sapelo2 Complete Guide

Complete guide for running DAI experiments on UGA's Sapelo2 HPC cluster.

## Canonical H100 end-to-end workflow

The single supported end-to-end entrypoint is:

```bash
mkdir -p logs
sbatch slurm/submit_multidataset_pipeline.sh
```

`submit_publication_pipeline.sh` is a backward-compatible alias. The
coordinator submits tests, reference equivalence, dataset validation, an H100
numerical smoke test, an H100 overfit gate, the complete 720-run publication
matrix, and dependent analysis. Matrix tasks request one H100 from
`gpu_30d_p` for 14 days and run at most five tasks concurrently by default.

Completed result/prediction pairs are skipped on a subsequent submission.
Incomplete baseline runs resume from the newest Hugging Face checkpoint, and
incomplete DAI runs resume from the newest completed epoch checkpoint. Resume
is refused if the source snapshot, matrix, config, dataset, method, or seed has
changed. Therefore, after a failure, redeploy the same committed revision and
submit the same coordinator command again.

Monitor the workflow with:

```bash
squeue -u "$USER"
sacct -X --starttime today --format=JobID,JobName,State,Elapsed,ExitCode
```

The older single-dataset examples later in this guide are retained for
historical reference and are not the canonical publication workflow.

---

## 📋 Prerequisites

- UGA MyID and Sapelo2 account
- SSH access to Sapelo2
- This project on your local machine

---

## 🚀 Quick Start (3 Steps)

### 1. Copy Project to Sapelo2

From your local machine:
```bash
cd "/home/adetayo/Documents/CSCI Forms/Adetayo Research/ Neurosymbolic Program Induction via Differentiable Abstract Interpretation "

# Copy to Sapelo (replace 'adetayo' with your MyID)
bash slurm/copy_to_sapelo.sh adetayo
```

**Alternative: Manual copy**
```bash
# From your local terminal
scp -r "/home/adetayo/Documents/CSCI Forms/Adetayo Research/ Neurosymbolic Program Induction via Differentiable Abstract Interpretation" your_myid@sapelo2.gacrc.uga.edu:~/dai-research/

# Or clone from GitHub (if using version control)
# ssh your_myid@sapelo2.gacrc.uga.edu
# git clone https://github.com/yourusername/dai-research.git ~/dai-research
```

### 2. Setup Environment (First Time Only)

Log into Sapelo2 and run setup:
```bash
ssh adetayo@sapelo2.gacrc.uga.edu
cd ~/dai-research
bash slurm/setup_sapelo.sh
```

**This takes ~10-15 minutes and:**
- Loads necessary modules (Python, CUDA, cuDNN)
- Creates a virtual environment
- Installs all dependencies with correct versions
- Verifies GPU availability
- Creates scratch space directories

### 3. Submit Your First Job

```bash
# Quick test run (SCAN dataset, 1 seed)
sbatch slurm/slurm_train.sh

# Check job status
squeue -u $USER

# View output in real-time
tail -f logs/train_*.out
```

---

## 📊 Running Experiments

### Publication Matrix: 7 Configurations x 5 Seeds

The publication pipeline uses a seven-task A100 job array. Each task runs one
configuration over five paired seeds sequentially, producing 35 total runs
while respecting Sapelo's GPU submission limit. A dependent CPU job produces
the paired statistical reports only after all seven tasks succeed.

First, sync the current code and compact SCAN corpus from the local project:

```bash
bash slurm/copy_to_sapelo.sh sapelo2
```

Then log in and submit the pipeline. These commands only submit jobs; training
and analysis do not execute on the login node.

```bash
ssh sapelo2
cd ~/dai-research
mkdir -p logs

MATRIX_JOB=$(sbatch --parsable slurm/slurm_publication_matrix.sh)
sbatch --dependency="afterok:$MATRIX_JOB" slurm/slurm_publication_analysis.sh
echo "Submitted publication matrix job $MATRIX_JOB"
```

The default array runs at most five A100 tasks concurrently. To reduce queue or
allocation pressure, override the concurrency limit:

```bash
MATRIX_JOB=$(sbatch --parsable --array=0-6%3 slurm/slurm_publication_matrix.sh)
sbatch --dependency="afterok:$MATRIX_JOB" slurm/slurm_publication_analysis.sh
```

Array task mapping:

| Task ID | Configuration | Seeds in order |
|----------|---------------|----------------|
| 0 | Grounded composition | 42, 123, 456, 789, 1024 |
| 1 | Matched bottleneck | 42, 123, 456, 789, 1024 |
| 2 | Bottleneck + entropy | 42, 123, 456, 789, 1024 |
| 3 | Contrastive control | 42, 123, 456, 789, 1024 |
| 4 | No auxiliary objective | 42, 123, 456, 789, 1024 |
| 5 | Shuffled-operator control | 42, 123, 456, 789, 1024 |
| 6 | Random-operator control | 42, 123, 456, 789, 1024 |

Monitor the array and its dependent analysis job:

```bash
squeue -u "$USER"
sacct -j "$MATRIX_JOB" --format=JobID,State,Elapsed,MaxRSS,ExitCode
tail -f logs/publication_${MATRIX_JOB}_0.out
```

Retry a failed task using the same task ID, then resubmit analysis after the
retry succeeds:

```bash
RETRY_JOB=$(sbatch --parsable --array=TASK_ID slurm/slurm_publication_matrix.sh)
sbatch --dependency="afterok:$RETRY_JOB" slurm/slurm_publication_analysis.sh
```

Artifacts are written under scratch:

```text
/scratch/$USER/dai-research/experiments/publication/<config>/seed_<seed>/
/scratch/$USER/dai-research/logs/publication/
/scratch/$USER/dai-research/results/publication_statistics/
```

The matrix job refuses to run outside SLURM, without CUDA, or without the two
local SCAN length files. Scratch artifacts remain subject to Sapelo's retention
policy and should be archived after analysis.

### Single Training Run

```bash
# Train on SCAN dataset with default settings
sbatch slurm/slurm_train.sh

# Train on specific dataset and split
sbatch --export=DATASET=cogs slurm/slurm_train.sh
sbatch --export=DATASET=scan,SPLIT=addprim_jump slurm/slurm_train.sh

# Train with specific seed
sbatch --export=DATASET=scan,SEED=42 slurm/slurm_train.sh

# CFQ with MCD split
sbatch --export=DATASET=cfq,SPLIT=mcd1 slurm/slurm_train.sh
```

### Multi-Seed Training (Recommended for Publications)

For reproducible results with statistical significance:
```bash
# Run with 3 seeds (42, 1337, 2024)
sbatch slurm/slurm_train_multi_seed.sh

# Specify custom seeds
sbatch --export=SEEDS="42 1337 2024 9999" slurm/slurm_train_multi_seed.sh

# Different dataset with multiple seeds
sbatch --export=DATASET=cogs,SEEDS="42 1337 2024" slurm/slurm_train_multi_seed.sh
```

This automatically:
- Runs training with each seed sequentially
- Aggregates results across seeds
- Computes mean and standard deviation
- Saves summary to `results/`

### All Datasets

```bash
# Submit jobs for all benchmarks
for dataset in scan cogs cfq clutrr gsm8k; do
    sbatch --export=DATASET=$dataset slurm/slurm_train_multi_seed.sh
done
```

### Evaluation

```bash
# Evaluate a trained checkpoint
sbatch --export=CHECKPOINT=/path/to/checkpoint slurm/slurm_evaluate.sh

# Example
sbatch --export=CHECKPOINT=/scratch/$USER/dai-research/checkpoints/scan_length_seed42/checkpoint-best slurm/slurm_evaluate.sh
```

---

## 📝 Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# Detailed job information
scontrol show job JOBID

# View logs in real-time
tail -f ~/dai-research/logs/train_JOBID.out
tail -f ~/dai-research/logs/train_JOBID.err

# Check GPU usage (from compute node)
ssh gpu-node-name  # Get node name from squeue
nvidia-smi
watch -n 1 nvidia-smi  # Continuous monitoring

# Check resource usage
sacct -j JOBID --format=JobID,MaxRSS,MaxVMSize,Elapsed
```

---

## 📁 File Structure on Sapelo2

### Project Layout
```
~/dai-research/                           # Home directory (code, persistent)
├── src/                                  # Source code
├── scripts/                              # Training scripts
├── slurm/                                # SLURM job scripts
├── setup_sapelo.sh                       # Setup script
├── venv/                                 # Virtual environment
├── checkpoints/ -> /scratch/.../checkpoints/   # Symlink to scratch
├── logs/ -> /scratch/.../logs/                 # Symlink to scratch
├── data/ -> /scratch/.../data/                 # Symlink to scratch
└── results/ -> /scratch/.../results/           # Symlink to scratch

/scratch/$USER/dai-research/              # Scratch space (data, fast I/O)
├── checkpoints/                          # Model checkpoints (1-10GB each)
│   ├── scan_length_seed42/
│   ├── scan_length_seed1337/
│   └── cogs_seed42/
├── logs/                                 # SLURM logs
│   ├── train_12345.out
│   └── train_12345.err
├── results/                              # Evaluation results
│   └── scan_length_aggregated.json
└── data/                                 # Cached datasets
```

### Storage Strategy

| Location | Purpose | Quota | Persistence | Speed |
|----------|---------|-------|-------------|-------|
| `~/dai-research` | Code, venv | ~100GB | Permanent | Standard |
| `/scratch/$USER` | Data, checkpoints | ~1TB+ | 30 days | Fast I/O |

⚠️ **Important:** Files in `/scratch` are deleted after 30 days! Archive results regularly.

---

## 🔧 Configuration & Customization

### Update Email Notifications

```bash
cd ~/dai-research
sed -i 's/adetayo@uga.edu/YOUR_MYID@uga.edu/g' slurm/slurm_*.sh
```

### Adjust SLURM Resources

Edit `slurm/*.sh` headers to modify:

```bash
#SBATCH --mem=64G                # Increase memory (default: 32G)
#SBATCH --time=48:00:00          # Extend time limit (default: 24h)
#SBATCH --gres=gpu:A100:2        # Request 2 GPUs (default: 1)
#SBATCH --cpus-per-task=32       # Adjust CPUs (default: 64)
```

### Modify Training Parameters

Edit `slurm/slurm_train.sh` to change:
```bash
--num_epochs 20                  # More/fewer epochs
--train_batch_size 32            # Batch size
--learning_rate 3e-4             # Learning rate
--abstraction_loss_weight 1.0    # DAI loss weight
--warmup_ratio 0.1              # Warmup schedule
```

### Use Job Arrays (Advanced)

For running many configurations efficiently:
```bash
# Add to slurm/slurm_train.sh
#SBATCH --array=1-10

# Then use $SLURM_ARRAY_TASK_ID in your script
SEED=$((42 * $SLURM_ARRAY_TASK_ID))
```

---

## 🎯 Understanding Sapelo2 Resources

### GPU Types Available

| GPU | Memory | Compute | Best For | Command |
|-----|--------|---------|----------|---------|
| **A100** | 40GB | Ampere | **Recommended** | `--gres=gpu:A100:1` |
| V100 | 32GB | Volta | Older, still good | `--gres=gpu:V100:1` |

### Resource Recommendations

| Scenario | GPU | CPUs | RAM | Time | Notes |
|----------|-----|------|-----|------|-------|
| **Quick test** | 1 A100 | 64 | 32GB | 2h | Single epoch |
| **Single seed** | 1 A100 | 64 | 32GB | 24h | Standard |
| **Multi-seed (3)** | 1 A100 | 64 | 32GB | 72h | Full experiment |
| **Large model** | 1 A100 | 64 | 64GB | 48h | T5-large/base |
| **Evaluation only** | 1 A100 | 64 | 16GB | 2h | Fast |

### Check Available Resources

```bash
# See GPU partition info
sinfo -p gpu_p -o "%n %c %m %G"

# Check specific node details
scontrol show node gpu-node-name

# View partition limits
sinfo -p gpu_p -l
```

---

## ⏱️ Expected Runtimes

| Dataset | Single Seed | 3 Seeds | Resources | Notes |
|---------|-------------|---------|-----------|-------|
| **SCAN** | 2-4 hours | 6-12h | 1 A100, 32GB | Fast, good for testing |
| **COGS** | 6-8 hours | 18-24h | 1 A100, 32GB | Moderate complexity |
| **CFQ** | 12-16 hours | 36-48h | 1 A100, 32GB | Longer sequences |
| **CLUTRR** | 8-10 hours | 24-30h | 1 A100, 32GB | Relational reasoning |
| **GSM8K** | 20-24 hours | 60-72h | 1 A100, 32GB | Math problems, longest |

*Times are estimates for 20 epochs with batch size 32*

---

## 🐛 Troubleshooting

### Out of Memory (OOM) Errors

**Problem:** Job crashes with CUDA out of memory
```bash
# Solution 1: Reduce batch size
--train_batch_size 16              # Instead of 32
--gradient_accumulation_steps 2    # Maintain effective batch size

# Solution 2: Use smaller model
--model_name t5-small              # Instead of t5-base

# Solution 3: Reduce sequence length
--max_source_length 64             # Instead of 128
--max_target_length 64
```

### Time Limit Exceeded

**Problem:** Job killed before completion
```bash
# Solution: Increase time limit in SLURM header
#SBATCH --time=48:00:00            # Instead of 24:00:00

# Or use checkpointing to resume
--resume_from_checkpoint /path/to/checkpoint
```

### Module Not Found Errors

**Problem:** Python packages not found
```bash
# Solution: Ensure environment is activated
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/11.7.0
module load cuDNN/8.4.1.50-CUDA-11.7.0
source ~/dai-research/venv/bin/activate

# Verify installation
python -c "import torch; print(torch.__version__)"
python -c "import transformers; print(transformers.__version__)"
```

### Job Stuck in Queue

**Problem:** Job waiting too long
```bash
# Check queue
squeue -p gpu_p

# Solution 1: Reduce resource requirements
#SBATCH --time=12:00:00           # Request less time
#SBATCH --mem=16G                 # Request less memory

# Solution 2: Use different partition (if available)
#SBATCH --partition=gpu_p

# Solution 3: Check job priority
sprio -j JOBID
```

### Scratch Files Deleted

**Problem:** Checkpoints gone after 30 days
```bash
# Prevention: Archive regularly
cd /scratch/$USER/dai-research
tar -czf ~/dai-results-$(date +%Y%m%d).tar.gz checkpoints/ results/

# Recovery: No recovery possible, re-run experiments
# Best practice: Copy important results to home immediately after completion
```

### Slow Data Loading

**Problem:** Training bottlenecked by data
```bash
# Solution: Increase num_workers in dataset config
--num_workers 8                    # Match CPUs allocated

# Or: Pre-download datasets to scratch
python scripts/download_data.py --all --output data
bash slurm/stage_publication_data.sh sapelo2
```

---

## 📊 Retrieving Results

### Copy Results to Local Machine

```bash
# From your local terminal
scp -r adetayo@sapelo2.gacrc.uga.edu:/scratch/$USER/dai-research/results ./local_results/
scp -r adetayo@sapelo2.gacrc.uga.edu:/scratch/$USER/dai-research/checkpoints ./local_checkpoints/

# Or use rsync for efficiency
rsync -avz --progress adetayo@sapelo2.gacrc.uga.edu:/scratch/$USER/dai-research/results ./local_results/
```

### Archive Results on Sapelo

```bash
# On Sapelo2
cd /scratch/$USER/dai-research

# Archive all results
tar -czf ~/dai-results-$(date +%Y%m%d).tar.gz results/ checkpoints/

# Archive specific experiment
tar -czf ~/scan-results.tar.gz results/*scan* checkpoints/*scan*

# Verify archive
tar -tzf ~/dai-results-*.tar.gz | head -20
```

---

## ✅ Reproducibility Features

All scripts ensure complete reproducibility through:

### 1. Fixed Random Seeds
```bash
export PYTHONHASHSEED=$SEED
# PyTorch seeds set in code: torch.manual_seed(seed)
```

### 2. Deterministic Operations
```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_LAUNCH_BLOCKING=1
# Python deterministic mode enabled in code
```

### 3. Pinned Dependencies
- All package versions specified in `requirements.txt`
- PyTorch 2.1.2 with CUDA 11.7
- Transformers 4.36.2

### 4. Environment Logging
- Git commit hash (if using version control)
- Full package list saved with results
- Hardware configuration logged

### 5. Multi-Seed Statistics
- Default seeds: 42, 1337, 2024
- Mean and standard deviation computed
- Full per-seed results saved

---

## 🎯 Recommended Workflow

### For Development/Testing
1. **Local quick test** (if you have GPU): `--num_epochs 1 --max_steps 100`
2. **Sapelo test**: `sbatch slurm/slurm_train.sh` (single seed, SCAN, 1 epoch)
3. **Verify outputs**: Check logs and results format
4. **Full experiment**: Proceed with multi-seed runs

### For Publication Results
1. **Single seed pilot**: Test on SCAN with one seed
2. **Multi-seed SCAN**: `sbatch --export=DATASET=scan slurm/slurm_train_multi_seed.sh`
3. **All datasets**: Loop through all benchmarks with 3 seeds
4. **Aggregate & analyze**: Use `scripts/aggregate_results.py` and `scripts/generate_tables.py`
5. **Archive results**: Copy to permanent storage immediately

### Batch Processing All Datasets
```bash
# Submit all experiments at once
for dataset in scan cogs cfq clutrr gsm8k; do
    for split in length addprim_jump; do  # Adjust per dataset
        sbatch --export=DATASET=$dataset,SPLIT=$split slurm/slurm_train_multi_seed.sh
    done
done

# Monitor progress
watch -n 30 'squeue -u $USER'
```

---

## 🔄 Best Practices

### 1. Use Scratch Space Efficiently
```bash
# Symlinks already created by setup script
ls -la ~/dai-research/checkpoints  # Points to /scratch
```

### 2. Test Before Full Runs
```bash
# Quick test with reduced parameters
sbatch --export=DATASET=scan,MAX_STEPS=100 slurm/slurm_train.sh
```

### 3. Monitor Resource Usage
```bash
# Check after job completes
sacct -j JOBID --format=JobID,JobName,MaxRSS,MaxVMSize,Elapsed,State

# During job
ssh gpu-node-name
watch -n 1 nvidia-smi
```

### 4. Save Results Immediately
```bash
# Add to your workflow
scp -r sapelo2:/scratch/$USER/dai-research/results ./results-backup-$(date +%Y%m%d)/
```

### 5. Clean Up Old Files
```bash
# Remove old checkpoints (keep best only)
find /scratch/$USER/dai-research/checkpoints -name "checkpoint-epoch-*" -mtime +7 -delete
```

---

## 📚 Additional Resources

### Documentation
- **Sapelo2 wiki**: https://wiki.gacrc.uga.edu/wiki/Sapelo2
- **SLURM docs**: https://slurm.schedmd.com/documentation.html
- **GACRC training**: https://gacrc.uga.edu/training/

### Support
- **Technical issues**: gacrc-help@uga.edu
- **Queue info**: `sinfo -p gpu_p`
- **System status**: https://status.gacrc.uga.edu/

### Useful Commands Reference
```bash
# Job management
sbatch <script>              # Submit job
squeue -u $USER              # Check your jobs
scancel JOBID                # Cancel job
scontrol show job JOBID      # Job details

# Resource info
sinfo -p gpu_p               # Partition info
squeue -p gpu_p              # Jobs in partition
sprio -j JOBID               # Job priority

# Storage
quota -s                     # Check home quota
du -sh /scratch/$USER        # Check scratch usage
lfs quota -h /scratch        # Scratch quota (if applicable)
```

---

## 📝 Summary Cheat Sheet

```bash
# One-time setup
bash slurm/copy_to_sapelo.sh adetayo
ssh adetayo@sapelo2.gacrc.uga.edu
cd ~/dai-research && bash slurm/setup_sapelo.sh

# Submit jobs
sbatch slurm/slurm_train.sh                              # Single run
sbatch slurm/slurm_train_multi_seed.sh                   # Multi-seed
sbatch --export=DATASET=cogs slurm/slurm_train.sh        # Different dataset

# Monitor
squeue -u $USER                                          # Job status
tail -f logs/train_*.out                                 # View logs
ssh gpu-node && nvidia-smi                               # Check GPU

# Retrieve
scp -r adetayo@sapelo2:/scratch/$USER/dai-research/results ./

# Customize
vi slurm/slurm_train.sh                                  # Edit parameters
sed -i 's/OLD/NEW/g' slurm/*.sh                         # Batch changes
```

---

**Questions?** Check the troubleshooting section or contact GACRC support at gacrc-help@uga.edu
