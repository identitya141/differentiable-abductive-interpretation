# Baseline Configurations

Configuration files for baseline models used in comparison with DAI.

## Available Baselines

| Config File | Baseline | Description |
|-------------|----------|-------------|
| `vanilla_t5.yaml` | Vanilla T5 | Standard T5 fine-tuning without modifications |
| `random_init_t5.yaml` | Random-init T5 | Architecture-matched scratch T5 with a 60-epoch inverse-square-root schedule, target-vocabulary decoding constraint, and EOS diagnostics |
| `tree_linearized_t5.yaml` | Tree-linearized T5 | T5 trained on linearized parse-tree inputs |
| `random_structure.yaml` | Random operator | Matched-span control with dataset-valid random operator labels |
| `shuffled_structure.yaml` | Shuffled operator | Matched-span control with operator labels shuffled within examples |
| `simple_consistency.yaml` | Simple consistency | Composition consistency without structural contrastive loss |
| `cot_t5.yaml` | Chain-of-Thought | T5 with "Let's think step by step" prompting |
| `scratchpad_t5.yaml` | Scratchpad | T5 with intermediate computation space |
| `modular_nn.yaml` | Neural Module Network | T5 with learned modular composition |
| `symbolic_rules.yaml` | Symbolic Rules | T5 augmented with hand-coded rules (SCAN only) |
| `tinyllama_lora.yaml` | TinyLlama + LoRA | TinyLlama 1.1B with efficient LoRA fine-tuning |

The authoritative eleven-baseline identity and class registry is
`src/models/baselines/baseline_models.py`. `vanilla` and `reference_t5` are
aliases for one baseline and must not be counted twice. The proposed full DAI
method is intentionally not part of the baseline registry.

## Usage

### Training a Single Baseline

```bash
# On Sapelo2
sbatch slurm/slurm_train_baselines.sh                              # Vanilla on SCAN length (default)
sbatch --export=BASELINE=cot slurm/slurm_train_baselines.sh        # Chain-of-Thought
sbatch --export=BASELINE=scratchpad slurm/slurm_train_baselines.sh # Scratchpad
```

### Training All Baselines

```bash
sbatch --export=BASELINE=all slurm/slurm_train_baselines.sh
```

### Different Datasets

```bash
sbatch --export=BASELINE=vanilla,DATASET=cogs,SPLIT=gen slurm/slurm_train_baselines.sh
sbatch --export=BASELINE=vanilla,DATASET=cfq,SPLIT=mcd1 slurm/slurm_train_baselines.sh
```

### Local Training (not recommended for full experiments)

```bash
python scripts/train_baseline.py \
    --baseline vanilla \
    --dataset scan \
    --split length \
    --output_dir checkpoints/baselines/vanilla_scan_length \
    --epochs 30
```

## Baseline Details

### Vanilla T5
Standard T5-small fine-tuned on the task. Represents the neural baseline without compositional inductive biases.

### Chain-of-Thought (CoT)
Adds "Let's think step by step." prefix to encourage reasoning chains. Better for multi-step problems.

### Scratchpad
Uses `[SCRATCH]...[/SCRATCH]` markers for intermediate computation. More structured than CoT.

### Neural Module Network
8 learned MLP modules with attention-based selection. Inspired by Neural Module Networks (Andreas et al., 2016).

### Symbolic Rules (SCAN only)
Full SCAN grammar implemented as hand-coded rules. Should achieve ~100% on SCAN if rules match.

### TinyLlama + LoRA
TinyLlama 1.1B Chat with LoRA provides the decoder-only instruction baseline at substantially lower compute cost than LLaMA-2 7B. It remains larger than T5-small, so parameter counts and accelerator-hours must be reported.

## Comparison with DAI

These baselines establish the performance floor and ceiling for comparison:
- **Floor**: Vanilla T5 (what standard transformers achieve)
- **Ceiling**: Symbolic Rules (what perfect compositional knowledge achieves)
- **DAI Goal**: Match or exceed NeSy baselines, approach symbolic ceiling
