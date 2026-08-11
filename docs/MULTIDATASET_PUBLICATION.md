# Multi-Dataset Publication Workflow

The frozen cross-dataset matrix is defined by
`configs/publication/multidataset_matrix.json`. It schedules paired seeds
42, 123, 456, 789, 1024, 2027, 4099, 7919, 104729, and 130363 for all eleven registered baselines and the
proposed full structural-contrastive DAI method on:

- SCAN length
- COGS generalization
- SLOG structural generalization
- CFQ MCD1, MCD2, and MCD3

This expands to 720 independently retryable array tasks (72 method/benchmark
configurations times ten seeds) and 720 paired-seed training
runs. Baseline rows dispatch to `scripts/train_baseline.py`; structural DAI
controls and the proposed method dispatch to `scripts/train.py`.

## Safety Contract

Do not sync these changes into `~/dai-research` while an existing job may
still load files from that checkout. After the active SCAN matrix completes,
sync the repository and submit from a Sapelo login node:

```bash
cd ~/dai-research
sbatch slurm/submit_multidataset_pipeline.sh
```

The login node only sends this `sbatch` request. Git hashing, source
snapshotting, manifest creation, and all downstream job submissions run in
the short `batch` coordinator allocation.

Set `MAX_CONCURRENT` to limit simultaneous A100 allocations:

```bash
sbatch --export=ALL,MAX_CONCURRENT=3 slurm/submit_multidataset_pipeline.sh
```

The coordinator creates a read-only, content-addressed source snapshot and
then submits a 720-task GPU array plus an `afterok` analysis job. A submission
JSON records the coordinator, matrix, and analysis job IDs under:

```text
/scratch/$USER/dai-research/workflows/<source-snapshot-id>/
```

Each run is isolated at:

```text
/scratch/$USER/dai-research/experiments/publication/
  <dataset>/<split>/<method>/seed_<seed>/
```

Before training, the launcher verifies the numerical gate checksums, passed
dataset-validation checksums, config and matrix hashes, source snapshot, and
the staged dataset content hash. `run_contract.json` prevents a retry from
reusing an output directory with different code, configuration, or data.

Analysis artifacts are isolated by source snapshot at:

```text
/scratch/$USER/dai-research/results/publication_multidataset/
  <source-snapshot-id>/analysis_<job-id>/<dataset>/<split>/
```
