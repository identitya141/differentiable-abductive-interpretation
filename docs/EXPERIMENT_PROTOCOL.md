# Predeclared Experiment Protocol

## Status and Change Control

This protocol is frozen before the final publication matrix is run. Any later
change must be appended under Amendments with its rationale and whether it was
made before or after inspecting final OOD outcomes. Existing entries are not
silently rewritten.

The numerical smoke, reference-equivalence, and small-data overfit gates are
eligibility checks, not evidence for the primary hypothesis. The final matrix
must remain blocked until all gates pass.

## Primary Question

Does `scan_full_contrastive` improve OOD exact match on the SCAN Length test
split relative to each predeclared control under paired seeds?

## Primary Endpoint

The primary endpoint is exact string match on the official OOD test split after
canonical dataset-specific normalization. Checkpoint selection uses only the
deterministic training/validation split and never the OOD test labels.

## Methods and Seeds

The primary method is `scan_full_contrastive`. The six primary controls are:

1. `scan_random_init_t5`
2. `scan_reference_t5`
3. `scan_tree_linearized`
4. `scan_random_structure` (matched-span random-operator corruption)
5. `scan_shuffled_structure` (matched-span shuffled-operator corruption)
6. `scan_simple_consistency`

The separately reported secondary experiment
`scan_topology_corruption_control` changes child-parent assignments while
preserving relation count, parent spans, child-span marginals, and operator
frequencies. It is not added post hoc to the six-comparison primary family.

All methods use the ten paired seeds `42`, `123`, `456`, `789`, `1024`,
`2027`, `4099`, `7919`, `104729`, and `130363`. The expansion from five to
ten was frozen before inspecting final OOD results because an exact two-sided
sign-flip test with five pairs has minimum attainable p-value 0.0625 and cannot
support the predeclared Holm-corrected inference. Failed runs
are rerun with the same seed and frozen configuration; seeds are not replaced
based on outcomes.

## Primary Analysis

For each control comparison, report method and control mean, sample standard
deviation, paired mean difference, 95% bootstrap confidence interval, paired
Cohen's $d_z$, paired sign-flip permutation test, and exact McNemar test over
paired examples. Apply Holm correction jointly across the six primary control
comparisons. Also report the paired difference for every individual seed.

No result is selected by best seed, best checkpoint on the OOD split, or best
post-hoc normalization rule. Missing or non-finite artifacts fail validation
rather than being omitted.

## Secondary and Exploratory Analyses

The following are secondary descriptive endpoints: IID exact match,
IID-to-OOD generalization gap, accuracy by composition depth, category/operator
accuracy, compute accounting, and held-out composition violation.

The following are exploratory unless promoted by a dated amendment made before
final outcomes are inspected: correlations between structural agreement and
correctness, qualitative failure categories, lambda and corruption sweeps,
layer/objective ablations, nonce controls, and analyses on additional benchmark
families. Exploratory intervals and adjusted p-values are reported as such and
do not redefine the primary claim.

## Benchmark Nomenclature

SCAN experiments are labeled by their implemented split, such as Length or
AddPrim. The term MCD is reserved for the official CFQ MCD1, MCD2, and MCD3
partitions; no SCAN result is labeled MCD without a separately implemented and
documented compound-divergence partition.

## Structural-Supervision Disclosure

SCAN structures are deterministically parsed from the input command alone.
COGS and SLOG semantic-role relations are extracted from each paired gold
logical form and aligned to source spans. CFQ relations are extracted from the
paired gold SPARQL query and conservatively aligned to question spans. The
latter benchmarks therefore use **training-time gold-derived structural
supervision**. They must not be described as using an input-only parser.

No structure annotation is provided to generation. Held-out structural
violation analysis may use gold-derived relations only as an explicitly
oracle, gold-assisted secondary diagnostic; it is never an input to prediction,
checkpoint selection, or the primary exact-match endpoint.

## Published Structural Comparator

The predeclared structure-aware comparator for the COGS/SLOG family is the
AM-Parser evaluated by Li et al. (2023), following the COGS adaptation of
Weißenhorn et al. (2022). It predicts an Apply-Modify dependency tree and then
converts that tree to the graph-based COGS logical form. A tree-linearized T5
control is not AM-Parser and must never be labeled as such.

Published AM-Parser values may appear only in a clearly labeled external
reference column with their original citation, metric, split, and reported
uncertainty. They are contextual evidence, not paired observations, and are
excluded from this repository's paired confidence intervals and hypothesis
tests.

A reproduced comparison may be labeled `AM-Parser (reproduced)` only after the
external implementation and revision are recorded, the official COGS-LF SLOG
files are used unchanged, and predictions are scored with the paper's
reformatted exact match. Report the official IID and generalization splits,
all 17 SLOG categories, the six recursion families by exact depth, all seeds,
and failures. Do not substitute a projective parser variant for a
non-projective one without naming the variant. Model capacity, training data,
training updates, stopping rule, hardware, and decoding settings must accompany
the result. Until those artifacts exist, the comparison remains pending.

## Artifact Requirements

Every run must record the resolved configuration, method, split, seed, source
revision, parameter counts, updates, examples seen, wall time, accelerator time,
peak memory, and per-example raw and normalized predictions. The dependent
analysis job validates method/seed coverage and writes SHA-256 manifests before
statistics or figures are generated.

## Amendments

### 2026-08-11 — Fairness contract and seed-count amendment

Before submitting the new publication matrix, tokenizer mode, literal
train/validation indices, source/target limits, evaluation loading, decoding,
and validation-based checkpoint selection were unified across DAI and the
applicable baselines through one benchmark contract. The paired seed set was
expanded from five to the ten seeds listed above because a two-sided exact
sign-flip test with five pairs has minimum attainable p-value 0.0625 and cannot
support the predeclared Holm-corrected analysis. This amendment followed
preliminary legacy SCAN runs but preceded the corrected 720-run matrix and was
not selected based on results from that matrix. Previous artifacts are not
silently relabeled as results under the amended contract.
