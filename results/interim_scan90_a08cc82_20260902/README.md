# Interim audit of 90 completed SCAN-Length runs

Date: 2026-09-02

Scope: nine completed methods, ten paired seeds per method. This is a
diagnostic interim analysis. It excludes the proposed `full_contrastive` DAI
method, symbolic results, TinyLlama, and every non-SCAN benchmark.

## Completion

- 90 result JSON files and 90 prediction JSONL files are present in the
  canonical `scan/length` experiment directory.
- The runs cover ten seeds for each of nine methods.
- Thirty runs use source revision `d6c393d`; sixty use `9edca0d`.
- Twenty-nine additional legacy results exist outside the canonical matrix and
  are excluded.

## Provisional scalar results

Percentages are mean exact match over ten seeds, with sample standard deviation.

| Method | IID EM | OOD EM | Generalization gap | Mean accelerator hours |
|---|---:|---:|---:|---:|
| Modular | 95.52 ± 2.39 | 10.53 ± 2.29 | 84.98 | 0.308 |
| CoT | 98.39 ± 0.59 | 8.61 ± 2.11 | 89.78 | 0.266 |
| Tree-linearized T5 | 92.10 ± 2.94 | 7.84 ± 3.47 | 84.26 | 0.300 |
| Shuffled structure | 83.26 ± 5.34 | 7.38 ± 2.20 | 75.88 | 0.795 |
| Reference T5 | 84.91 ± 3.99 | 7.10 ± 2.70 | 77.81 | 0.177 |
| Random structure | 85.46 ± 5.57 | 6.57 ± 2.23 | 78.88 | 0.813 |
| Simple consistency | 84.99 ± 4.06 | 6.38 ± 1.27 | 78.62 | 0.802 |
| Random-init T5 | 0.21 ± 0.12 | 0.00 ± 0.00 | 0.21 | 1.259 |
| Scratchpad | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 | 0.266 |

Scratchpad values are invalid as scientific performance estimates because its
stored normalized targets retain special-token strings such as `</S><PAD>`.

## Pairwise comparisons that passed the current key-pairing code

Each difference is method minus reference T5 in absolute accuracy units.

| Method | Mean difference | Bootstrap 95% CI | Paired permutation p | Cohen's dz |
|---|---:|---:|---:|---:|
| Modular | +0.0343 | [0.0107, 0.0575] | 0.0273 | 0.86 |
| CoT | +0.0151 | [-0.0068, 0.0368] | 0.2344 | 0.40 |
| Tree-linearized T5 | +0.0073 | [-0.0217, 0.0377] | 0.6660 | 0.14 |
| Random-init T5 | -0.0710 | [-0.0864, -0.0547] | 0.0020 | -2.63 |

These are exploratory comparisons, not the predeclared DAI comparison family.
If Holm correction is applied across these four available comparisons, the
modular comparison does not remain below 0.05 (adjusted p approximately 0.082).

## Depth and operator diagnostics

The breakdown script completed for modular, random-init, random-structure,
reference T5, shuffled-structure, and simple-consistency.

- Every one of these methods has 0% exact match on the OOD depth-3 group.
- All nonzero OOD performance comes from the depth-4 group.
- Modular has the best depth-4 mean (10.55%), followed by shuffled structure
  (7.40%), reference T5 (7.12%), random structure (6.58%), and simple
  consistency (6.39%).
- Across these methods, `twice` examples are easier than `thrice` examples.
  For reference T5 the means are 14.60% versus 3.32%; for modular they are
  22.07% versus 6.56%.
- These operator groups overlap and must not be interpreted as independent
  samples.

## Formal validation failures

The current 90-run set does not pass the checked-in publication validator.

1. Structural-control result JSON files omit required top-level provenance
   fields, beginning with `seed`. Their contracts contain those fields, but the
   result artifacts do not.
2. Random structure, shuffled structure, and simple consistency store SCAN
   normalized targets with spaces, while reference/baseline artifacts store
   the equivalent action sequences without spaces. The comparison script
   therefore treats all 3,920 examples as unpaired.
3. Scratchpad normalized targets retain special-token text, producing zero
   exact match and preventing valid pairing.
4. CoT prediction artifacts save the prefixed model input (`Let's think step by
   step...`) rather than the original SCAN command.
5. Tree-linearized artifacts save the transformed tree string rather than the
   original command. Scratchpad similarly saves its transformed input.
6. Consequently, the all-method depth/operator generator cannot parse CoT,
   scratchpad, or tree-linearized inputs.
7. Generation length diagnostics report one token for concatenated SCAN action
   strings and should not be used until action-aware length measurement is
   implemented.

## Scientific interpretation

The completed baselines confirm that SCAN-Length has a large IID-to-OOD gap.
They do not test the project's central hypothesis because no canonical
`full_contrastive` DAI result exists yet. The current structural controls also
do not show a compelling advantage over reference T5: shuffled structure is
only 0.28 percentage points above reference on the scalar aggregate, while
random structure and simple consistency are lower. The modular baseline's
provisional advantage is interesting but is not evidence for DAI's grounded
abstract-consistency objective.

No conference claim should be based on this interim analysis. Artifact schema,
normalization, pairing, and original-input preservation must be repaired and
validated before the completed baselines are accepted into the final analysis.

## Generated files

- `*_aggregated.json`: per-method scalar and efficiency summaries.
- `comparisons/*.json`: four pairwise comparisons that passed current pairing.
- `breakdowns_parseable6/`: JSON, CSV, and LaTeX depth/operator summaries for
  the six methods whose artifacts retain parseable original inputs.
