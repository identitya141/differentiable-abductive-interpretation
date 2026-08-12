# Publication Readiness TODO

Target claim:

> Using differentiable abstract-domain consistency objectives as an inductive bias for transformer compositional generalization.

## Phase 0: Numerical Integrity and Reference Equivalence

- [x] Restore the tied-embedding output scaling used by the reference T5 implementation.
- [x] Add fail-fast checks for non-finite forward losses and gradient norms.
- [x] Add a one-batch FP16 diagnostic that separates reference T5, compute-and-ignore abstraction, and zero-weight abstraction paths.
- [x] Complete the Sapelo diagnostic and identify `LogBackward0` in the zero-weight abstraction path as the first non-finite operation.
- [x] Disconnect the abstraction graph whenever its scheduled weight is zero while retaining detached diagnostics.
- [x] Implement custom/reference T5 logit comparison with shared weights, deterministic padded inputs, identical masks, disabled dropout, and evaluation mode.
- [x] Mirror Hugging Face T5 encoder dropout and relative-position-bias propagation in the custom DAI encoder path.
- [x] Add a no-download two-layer regression test and a machine-readable pretrained CPU compute-node equivalence gate.
- [x] Require custom/reference maximum absolute logit error to be numerically negligible before publication runs; passed with maximum error `0.0` against a `1e-5` threshold.
- [x] Run a 100-500-step A100 numerical smoke test with finite losses, gradients, activations, and decreasing training loss; 200-step Sapelo job 47336063 passed, with logged loss decreasing from 27.19 at step 10 to 6.59 at step 200.
- [ ] Run a small-data overfitting test before launching the full experiment matrix.
- [x] Implement a deterministic 24-example SCAN overfit harness with an A100 guard, 3,000-update ceiling, disabled dropout, constant learning rate, finite loss/gradient checks, loss-reduction threshold, and 95% training exact-match threshold.
- [x] Write a machine-readable overfit report on pass, criterion failure, or runtime error.
- [x] Block the publication matrix with both `afterok` and validation of the exact job-specific passing overfit report.
- [x] Add a single submission-only dependency chain for tests, reference equivalence, numerical smoke, overfit, publication matrix, and analysis.
- [x] Keep all full publication runs stopped until the numerical and reference-equivalence gates pass; the matrix remains additionally blocked on the overfit gate.

## Phase 1: Grounded Compositional Constraints

- [x] Define a deterministic SCAN grammar parser that returns typed composition trees.
- [x] Emit child and parent spans for every internal SCAN composition.
- [x] Add focused parser tests for primitives, modifiers, repetition, conjunction, and `after`.
- [x] Add composition metadata to `CompositionalExample` and `CompositionalBatch`.
- [x] Align word spans with tokenizer positions and reject ambiguous spans.
- [x] Preserve per-example composition metadata through the data collator.
- [x] Replace global token triplets with batched span-level composition specifications.
- [x] Pool child and parent encoder states using mean pooling.
- [x] Compute abstract consistency between the parent abstraction and composed child abstractions.
- [x] Support operator-specific abstract composition for direction, modifier, repetition, sequence, and `after`.
- [x] Fail loudly when grounded composition is required but a training batch has no metadata.
- [x] Log composition count, raw/weighted loss, and operator-table gradient norm; validate corpus coverage with a preflight script.
- [x] Add a test showing correct structures produce lower loss than shuffled structures.
- [x] Add a no-download one-batch test proving composition loss produces nonzero operator gradients and an optimizer update.
- [x] Add a small-data overfitting check before full training; execution remains pending the numerical gates.

## Phase 2: Formal Scope and Guarantees

- [x] Define the concrete representation space, abstract space, abstraction map, abstract composition operators, and divergence.
- [x] State the approximate commutativity objective precisely.
- [x] Document construction guarantees, empirical obligations, and non-guarantees.
- [x] State and justify the zero-divergence agreement proposition.
- [ ] Report empirical composition-violation bounds on held-out data.
- [x] Remove unsupported implementation and paper-template claims of a Galois connection and sound over-approximation.
- [x] Rename `concretize_loss` in paper-facing terminology to representation reconstruction consistency while retaining the internal API for compatibility.
- [x] Decide not to implement a finite type-powerset lattice without externally grounded concrete type semantics; document it as a separate future method.
- [x] Mark lattice-law and conservative-transfer tests as not applicable because the current method makes no lattice or soundness claim.
- [x] Update `RESEARCH_SPECIFICATION.md`, `PAPER_ARTIFACTS.md`, and the README to match implemented guarantees.

## Phase 3: Matched Controls

- [x] Establish inherited JSON configurations for shared model, data, optimizer, schedule, and seed settings.
- [x] Route DAI and all controls through the same canonical evaluator normalization functions.
- [x] Implement a parameter-matched bottleneck reconstruction control.
- [x] Implement a bottleneck plus entropy control.
- [x] Implement a supervised contrastive control over operator-labeled parent abstractions.
- [x] Implement the full abstract-domain consistency model.
- [x] Add an independently executed reference Hugging Face T5 baseline; do not treat zero-weight DAI as sufficient evidence of equivalence.
- [x] Add a randomly initialized Transformer or randomly initialized T5 architecture with a matched training budget.
- [x] Add a tree-linearized T5 baseline with no structural loss.
- [x] Add an isolated simple parent/child structural-consistency baseline.
- [x] Implement abstract composition consistency plus contrastive learning as the proposed full model.
- [x] Ensure the contrastive objective compares composed child representations with the correct parent and explicit negatives.
- [x] Freeze the final seven primary SCAN comparisons: random-init Transformer, reference T5, tree-linearized T5, random structure, shuffled structure, simple consistency, and the full proposed method.
- [x] Implement result-artifact recording for total/trainable parameters across every system; verify populated values in publication runs.
- [x] Implement recording for updates, examples seen, wall-clock time, accelerator hours, and peak memory; verify populated values in publication runs.

## Phase 4: Ablations and Falsification

- [x] Remove composition loss while retaining the same abstraction networks.
- [x] Add a deterministic shuffled-structure training mode.
- [x] Add deterministic random span pairs matched in number and length.
- [x] Remove entropy regularization.
- [x] Remove reconstruction consistency.
- [x] Test each constrained layer independently.
- [x] Compare learned and fixed abstract composition rules.
- [x] Compare operator-specific and operator-agnostic composition.
- [x] Confirm operator-embedding removal is not applicable: the implemented architecture selects operator-specific transfer tensors and contains no operator embedding component.
- [x] Implement matched MSE, cosine, and contrastive structural-consistency objective configurations; empirical comparison remains pending the numerical gates.
- [x] Freeze five structural-loss lambda configurations around the selected value; run the sweep after numerical gates pass.
- [x] Freeze 0%, 10%, 25%, 50%, 75%, and 100% structure-corruption configurations with a shared budget; run after numerical gates pass.
- [ ] Verify that OOD performance degrades as structure corruption increases.
- [x] Confirm that random spans preserve the number, operators, and individual lengths of grounded spans.
- [x] Confirm that shuffled controls preserve composition count, operators, parent hierarchy, binary branching, and the right-span length multiset while breaking input-structure alignment.
- [ ] Verify that genuine structures outperform shuffled and random structures.

## Phase 5: Fair Evaluation

- [x] Create a deterministic SCAN validation split and prohibit SCAN test-set checkpoint selection.
- [x] Use one canonical output-normalization path for every method.
- [x] Save input, target, raw/normalized prediction, correctness, method, split, and seed as JSONL.
- [x] Predeclare OOD exact match as the primary endpoint.
- [ ] Report novel-composition exact match and accuracy by composition depth.
- [ ] Report in-distribution exact match, generalization gap, and compositional-distance accuracy.
- [ ] Report abstract composition violation and representation diagnostics.
- [ ] Run SCAN IID/simple as a sanity check and report the IID-OOD gap.
- [ ] Run SCAN Length, AddPrim Jump, AddPrim Turn Left, and selected template splits.
- [x] Reserve MCD terminology for CFQ MCD1-3 and label SCAN experiments by their implemented split.
- [x] Build grounded COGS composition annotations and validate their corpus coverage: semantic-role arguments map to sentence-token spans, parent spans use the contiguous argument hull, and corpus validation covers 48,012/48,012 non-primitive train/dev/gen examples with 152,591 `agent`, `theme`, `recipient`, `ccomp`, `xcomp`, and `nmod` relations (Sapelo job 47336076; primitive lexical entries are ineligible).
- [x] Preserve COGS IID and generalization case labels and implement category-level reporting; full runs remain pending.
- [x] Implement official SLOG loading, validate all 17 structural categories and six exact recursion-depth families, and implement overall/category/depth reporting; empirical values remain pending.
- [x] Treat COGS and SLOG as one related benchmark family while reporting each dataset separately.
- [x] Build conservative grounded CFQ composition annotations and validate corpus coverage: exact `M#`, unique RDF-type phrase, and unique selected-WH anchors ground 479,977 query relations in 165,060/323,133 MCD1-3 examples (51.1%, zero parser errors; Sapelo job 47336095). Treat the remaining examples as unannotated rather than inferring ambiguous variable spans.
- [ ] Evaluate CFQ MCD1, MCD2, and MCD3 separately and report their mean.
- [ ] After SCAN screening, run at least reference T5, random structure, shuffled structure, and the proposed method on COGS/SLOG and CFQ.
- [ ] Add CLUTRR only after the official CSV corpus passes `scripts/validate_clutrr_compositions.py`; the validator is implemented, but generated or pre-generated official CSV evidence is pending.
- [x] Exclude GSM8K from structure-supervised publication experiments: the strict audit found 0/8,792 fully source-grounded examples despite 27,998 target equation steps (Sapelo job 47336265). Target-only chain-of-thought is not valid encoder-span composition supervision.

## Phase 5A: Pretraining and Lexical Controls

- [x] Create a deterministic nonce-word mapping for SCAN primitives and preserve original commands in run artifacts.
- [x] Freeze pretrained reference-T5 and proposed-method nonce configurations; comparisons remain pending execution.
- [ ] Compare the proposed structural objective with a random-init backbone or controlled novel embeddings.
- [ ] Verify that any structural advantage persists when pretrained lexical familiarity is removed.

## Phase 5B: Mechanistic and Error Analysis

- [x] Implement artifact-backed accuracy-versus-depth tables and figures for every primary method; generate final artifacts after runs complete.
- [ ] Test whether the proposed method's advantage grows with compositional depth or difficulty.
- [ ] Compare structural-consistency scores for correct and incorrect OOD predictions.
- [ ] Quantify the correlation between held-out structural agreement and OOD correctness.
- [x] Implement JSON/CSV/LaTeX reporting by SCAN operator and available benchmark category; populate COGS/SLOG/CFQ tables after runs complete.
- [ ] Categorize representative failures by operator, depth, output length, and structural novelty.
- [ ] Include qualitative examples showing successes, remaining failures, and differences from shuffled/random controls.

## Phase 6: Statistical Evidence

- [x] Add a Sapelo A100 job array for the seven-config, five-seed matrix and a dependent compute-node statistics job.
- [x] Add a pre-analysis artifact validator for method/seed coverage, finite metrics, paired examples, correctness totals, and SHA-256 manifests.
- [x] Freeze the protocol before final runs in `EXPERIMENT_PROTOCOL.md`, with explicit amendment rules.
- [x] Freeze five paired seeds for the primary matrix; expanding to ten remains optional and must not depend on observed outcomes.
- [x] Implement mean, sample standard deviation, paired difference, and bootstrap confidence intervals.
- [x] Implement an exact/Monte Carlo paired sign-flip permutation test across seeds.
- [x] Implement paired Cohen's dz effect size.
- [x] Implement exact McNemar tests over paired example predictions.
- [x] Apply Holm correction jointly across the primary comparison family.
- [x] Extend `scripts/aggregate_results.py` for current seed artifacts, depth/category metrics, compute accounting, and sample standard deviation.
- [x] Mark exploratory analyses separately from preregistered primary analyses.
- [ ] Report mean, sample standard deviation, paired difference, 95% confidence interval, effect size, and statistical test for every primary comparison.
- [ ] Report whether the improvement occurs for every paired seed rather than only the best seed.

## Phase 6A: Structural Baseline and Release Protocol

- [x] Freeze AM-Parser as the published structure-aware COGS/SLOG comparator and prohibit labeling tree-linearized T5 as AM-Parser.
- [x] Separate cited published values from reproduced artifacts and exclude external aggregate values from paired statistics.
- [x] Require a pinned external implementation, official COGS-LF splits, reformatted exact match, category/depth breakdowns, seeds, capacity, budget, and decoder variant before claiming reproduction.
- [x] Implement deterministic source-release checksums and COGS/SLOG family reporting.
- [ ] Execute and validate the reproduced AM-Parser comparison; until then, report only a clearly cited published reference.

## Phase 7: Publication Gate

- [ ] DAI beats the matched bottleneck on OOD exact match on at least two benchmarks.
- [ ] The paired 95% confidence interval for the primary comparison excludes zero.
- [ ] Correct composition structures beat shuffled and random controls.
- [ ] Composition agreement improves on held-out data and relates to compositional generalization.
- [ ] Claims in the abstract and conclusion do not exceed the formal or empirical evidence.
- [ ] Every headline table and figure is reproducible from checked-in scripts and raw result files.
- [ ] Demonstrate the central ordering: correct structure outperforms shuffled/random structure and the no-structure baseline.
- [ ] Demonstrate that the effect appears across SCAN, COGS/SLOG, and CFQ rather than only one synthetic split.
- [ ] Demonstrate that the effect survives the nonce-word or random-init pretraining control.
- [ ] Demonstrate increasing or sustained benefit with greater composition depth.
- [ ] Release code, frozen configurations, seeds, environment details, raw predictions, and artifact-generation commands.
- [ ] Compare against at least one previously published structural or compositional-generalization method using a fair protocol.
- [ ] Clearly distinguish the consistency principle from prior grammar-aware, tree-aware, and structure-linearization approaches.
