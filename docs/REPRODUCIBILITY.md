# Reproducibility and Release Contract

## Evidence Boundary

Code, tests, validators, and frozen protocols may be released before final
experiments. Accuracy, improvement, statistical significance, and
cross-benchmark claims require validated result artifacts. Missing results are
shown as pending, never replaced by expected values.

## Sapelo Workflow

Repository computation is submitted to a Sapelo compute node. The hardened
test harness accepts focused paths without running tests on a login node:

```bash
TEST_PATHS="tests/test_slog_dataset.py tests/test_slog_evaluation.py" \
  sbatch slurm/slurm_test_suite.sh
```

Publication jobs remain blocked by reference equivalence, numerical smoke, and
the strict 95% small-data overfit gate. The gate is not weakened or bypassed.
The frozen matrix and dependent analysis use the scripts under `slurm/` and
the paired seeds in `docs/EXPERIMENT_PROTOCOL.md`.

## Dataset Validation

Run validators against official corpora and retain their JSON reports:

```bash
sbatch slurm/slurm_validate_slog.sh
sbatch slurm/slurm_validate_clutrr.sh
sbatch slurm/slurm_validate_gsm8k_structure.sh
```

SCAN, COGS, and CFQ validators are under `scripts/`. CLUTRR synthetic fallback
is for tests only and is never publication evidence. GSM8K target rationales
do not qualify as encoder-span supervision unless the validator proves exact,
unique source grounding.

The GSM8K audit over 8,792 official examples found 27,998 target-side equation
steps and zero fully source-grounded examples (Sapelo job 47336265). GSM8K is
therefore excluded from structure-supervised publication experiments unless a
new, independently validated source-grounded annotation resource is declared
by protocol amendment.

## Required Run Artifacts

Each reported run must retain:

- resolved configuration, method, dataset, split, seed, and source revision;
- parameter counts, optimizer updates, examples seen, wall time, accelerator
  time, and peak accelerator memory;
- raw and normalized per-example predictions, targets, correctness, category,
  and available exact depth metadata;
- checkpoint-selection records that do not use OOD test labels;
- validation and statistical-analysis reports, including failed runs.

Published third-party aggregate values are stored and labeled separately from
reproduced per-seed artifacts. They are not inputs to paired tests.

## Release Manifest

After source and documentation are frozen, generate the deterministic manifest:

```bash
sbatch slurm/slurm_release_manifest.sh
```

This writes `results/release/source_manifest.json` with sorted paths, SHA-256
checksums, sizes, Git state, and environment metadata. Regenerate it after any
source, configuration, test, script, Slurm, or documentation change.
The inventory also includes the retained GSM8K eligibility report at
`results/validation/gsm8k_structure.json`.
Manifest generation fails if the Sapelo project directory is not a Git
worktree; source revision and dirty state must never be emitted as null.

Paper tables and figures are generated only from validated result files:

```bash
make generate-tables
make generate-figures
```