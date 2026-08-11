# Formal Scope and Guarantees

## Intended Claim

This project studies differentiable abstract-domain consistency objectives as
an inductive bias for transformer compositional generalization. It does not
claim to invent differentiable abstract interpretation.

## Implemented Objects

Let the concrete representation space be:

$$
H = \mathbb{R}^{d}.
$$

For a token span $s$, the concrete span representation is mean pooling:

$$
h_s = \frac{1}{|s|}\sum_{i \in s} h_i.
$$

The primary learned abstract representation used by the frozen SCAN, COGS,
SLOG, and CFQ experiments is a type distribution:

$$
A = \Delta^{T-1},
$$

where $\Delta^{T-1}$ is a distribution over learned semantic types. The
differentiable abstraction map is:

$$
\alpha_\theta : H \rightarrow A.
$$

For each grounded composition operator $o$, the primary implementation learns
an operator-specific type transfer table:

$$
C_{A,o} : A \times A \rightarrow A.
$$

Grounded composition triples $(L,R,P,o)$ are extracted deterministically from
the SCAN grammar and aligned exactly to tokenizer spans. The consistency
objective is:

$$
\mathcal{L}_{\mathrm{comp}} =
\frac{1}{N}\sum_{(L,R,P,o)}
D_A\left(
\alpha_\theta(h_P),
C_{A,o}(\alpha_\theta(h_L),\alpha_\theta(h_R))
\right).
$$

The primary type-domain divergence is symmetric KL divergence between type
distributions. Interval, monotonicity, type-monotonicity product, and
relational domains remain implemented research variants; they are not the
abstract representation used by the frozen primary matrix.

## Structural-Supervision Boundary

SCAN composition relations are derived entirely from the input command by its
deterministic grammar. COGS and SLOG relations are different: semantic roles
such as `agent`, `theme`, and `nmod` are identified from the gold logical form
and then aligned to spans in the source sentence. CFQ relations are identified
from the gold SPARQL query and conservatively grounded to source-question
spans. These three benchmarks therefore use **training-time gold-derived
structural supervision**, not an input-only structural parser.

Gold-derived relations are not consumed during generation and are unnecessary
for deployment-time prediction. When they are used on held-out examples to
measure composition violation, that quantity is explicitly an oracle,
gold-assisted diagnostic; it is not an inference-time signal or capability.
Shuffled and random matched-span controls test whether the correct structural
annotation matters beyond receiving an auxiliary loss.

## Guaranteed by Construction

1. The grounded SCAN parser is deterministic for a fixed command.
2. Word-to-token alignment is accepted only when independent word encodings
   exactly reproduce the full command encoding.
3. Invalid or truncated spans raise an error instead of contributing a silent
   zero loss.
4. The composition objective is differentiable with respect to encoder states,
   abstraction networks, and the selected learned type transfer table.
5. Each operator uses a distinct learned type transfer table.
6. For the primary nonnegative divergence, zero type consistency loss implies
   equality of the compared type distributions, up to numerical precision.

The sixth statement follows because symmetric KL divergence is nonnegative and
equals zero only when the compared probability distributions agree.

## Not Guaranteed

The current implementation does not establish:

- a Galois connection;
- a lattice abstraction or concretization map;
- sound over-approximation of concrete transformer operations;
- formal program correctness;
- guaranteed compositional generalization;
- semantic identifiability of learned latent types.

`concretize_loss` is retained as an internal API name for compatibility. In
paper-facing text it should be called representation reconstruction consistency,
not mathematical concretization.

## Finite-Lattice Decision

The present method does not add a finite type-powerset lattice. Its type labels
are learned latent categories without externally specified concrete semantics,
so declaring subset order or conservative transfer rules would not establish a
meaningful soundness result. A lattice variant would be a separate method: it
must first define task-level concrete type sets and conservative transfers, then
test order, join, meet, monotonicity, and transfer conservativeness. Those laws
are therefore not claimed or tested for the current learned type domain.

## Empirical Obligations

Claims beyond the construction above require experiments showing that:

- grounded structures outperform shuffled and random matched-span controls;
- the full objective outperforms matched bottleneck, entropy, and contrastive
  controls;
- improvements hold across paired seeds and benchmarks;
- composition agreement improves on held-out examples;
- gains are not explained only by parameter count, compute, or checkpoint
  selection.
