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

The primary learned abstract representation is a product:

$$
A = \Delta^{T-1} \times \Delta^2,
$$

where $\Delta^{T-1}$ is a distribution over learned semantic types and
$\Delta^2$ is a distribution over decreasing, non-monotonic, and increasing
classes. The differentiable abstraction map is:

$$
\alpha_\theta : H \rightarrow A.
$$

For each SCAN grammar operator $o$, the implementation learns a type transfer
table and applies a differentiable monotonicity transfer:

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

The implemented product divergence is symmetric KL divergence for type
distributions plus mean squared distance for monotonicity distributions.

## Guaranteed by Construction

1. The grounded SCAN parser is deterministic for a fixed command.
2. Word-to-token alignment is accepted only when independent word encodings
   exactly reproduce the full command encoding.
3. Invalid or truncated spans raise an error instead of contributing a silent
   zero loss.
4. The composition objective is differentiable with respect to encoder states,
   abstraction networks, and the selected learned type transfer table.
5. Each operator uses a distinct learned type transfer table.
6. For the implemented nonnegative divergences, zero product consistency loss
   implies equality of the compared type and monotonicity distributions, up to
   numerical precision.

The sixth statement follows because both symmetric KL divergence and squared
distance are nonnegative and equal zero only when their compared probability
distributions agree.

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
are therefore not claimed or tested for the current learned product domain.

## Empirical Obligations

Claims beyond the construction above require experiments showing that:

- grounded structures outperform shuffled and random matched-span controls;
- the full objective outperforms matched bottleneck, entropy, and contrastive
  controls;
- improvements hold across paired seeds and benchmarks;
- composition agreement improves on held-out examples;
- gains are not explained only by parameter count, compute, or checkpoint
  selection.