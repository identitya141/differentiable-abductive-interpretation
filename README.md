# Neurosymbolic Program Induction via Differentiable Abstract Interpretation (DAI)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1+-ee4c2c.svg)](https://pytorch.org/)

**A training-time framework that studies differentiable abstract-domain consistency as an inductive bias for compositional transformer representations.**

## Key Contributions

1. **Grounded Consistency Objective**: A differentiable objective compares each parent-span abstraction with the operator-specific composition of its child-span abstractions.

2. **Task-Appropriate Abstract Domains**: The primary SCAN study uses learned type distributions; optional interval, monotonicity, product, and relational domains support tasks where those abstractions are justified.

3. **Layer-Wise Constraints and Controls**: Configurable constrained layers, fixed or learned rules, shared or operator-specific rules, and matched structural controls support falsification experiments.

4. **Reproducible Evaluation Infrastructure**: Paired-seed configurations, per-example artifacts, canonical normalization, compute accounting, and dependent statistical analysis jobs support publication experiments. Empirical improvement claims remain pending those experiments.

Supervision scope: SCAN structures are parsed from inputs alone. COGS/SLOG
logical forms and CFQ SPARQL targets provide training-time gold-derived
structural annotations aligned to source spans; they are not used during
generation. See the formal-scope document for the corresponding claim limits.

The exact implemented objects, limited construction guarantees, and explicit
non-guarantees are documented in [Formal Scope and Guarantees](docs/FORMAL_GUARANTEES.md).
The frozen analysis and release rules are in
[Experiment Protocol](docs/EXPERIMENT_PROTOCOL.md) and
[Reproducibility](docs/REPRODUCIBILITY.md).

## 📁 Repository Structure

```
├── configs/       # Base, dataset, baseline, and frozen experiment configs
├── data/          # Local official benchmark corpora (not release source)
├── docs/          # Protocol, guarantees, readiness, and artifact templates
├── scripts/       # Training, evaluation, validation, analysis, and reporting
├── slurm/         # Sapelo compute-node jobs and orchestration
├── src/           # Data, models, losses, training, evaluation, and utilities
├── tests/         # Focused and integration tests
├── pyproject.toml
└── requirements.txt
```

## 🚀 Quick Start

### Publication Workflow

```bash
# Environment and official datasets
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python scripts/download_data.py --all --output data
# copy_to_sapelo.sh invokes the all-dataset canonical staging + SHA-256 step
bash slurm/copy_to_sapelo.sh sapelo2

# Result-independent validation and release inventory
sbatch slurm/slurm_test_suite.sh
sbatch slurm/slurm_release_manifest.sh

# Paper artifacts after validated runs exist
python scripts/generate_breakdown_tables.py --help
python scripts/generate_figures.py --help
```

Full publication experiments are submitted with the scripts under `slurm/`;
they are not run on login nodes. See the reproducibility contract for focused
tests, corpus validators, required raw artifacts, and gate ordering.

```bash
sbatch slurm/submit_multidataset_pipeline.sh
```

Publication runners are offline-only for data: a missing canonical staged file
is a hard error, never an implicit Hugging Face download.

## Results Status

Publication results are not reported until the numerical smoke, reference
equivalence, and small-data overfit gates pass and the paired-seed artifacts
complete validation. See [Publication Readiness TODO](docs/PUBLICATION_TODO.md)
for the current evidence status.

## 🔧 Requirements

- Python 3.10+
- PyTorch 2.1+
- CUDA 11.8+ (for GPU training)
- 32GB+ RAM (for large datasets)
- 1-4 A100 GPUs (or equivalent)

## 📖 Citation

```bibtex
@inproceedings{author2026dai,
  title={Neurosymbolic Program Induction via Differentiable Abstract Interpretation},
  author={Okunoye, Adetayo},
  booktitle={Proceedings of NeurIPS},
  year={2026}
}
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.








