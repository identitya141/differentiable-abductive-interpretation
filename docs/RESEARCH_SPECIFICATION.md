# Research Specification: Differentiable Abstract Interpretation for Compositional Generalization

## 1. Formal Problem Definition

### 1.1 Compositional Generalization Failure

**Definition 1 (Compositional Language).** Let $\mathcal{V}$ be a vocabulary of primitive symbols and $\mathcal{R}$ be a set of composition rules. A compositional language $\mathcal{L}$ is the closure of $\mathcal{V}$ under $\mathcal{R}$:

$$\mathcal{L} = \text{Closure}(\mathcal{V}, \mathcal{R})$$

**Definition 2 (Compositional Generalization).** A model $f_\theta$ exhibits compositional generalization if for any novel composition $c = r(c_1, c_2, \ldots, c_k)$ where $r \in \mathcal{R}$ and each $c_i$ was seen during training:

$$f_\theta(c) = r(f_\theta(c_1), f_\theta(c_2), \ldots, f_\theta(c_k))$$

That is, the model's behavior on novel compositions is predictable from its behavior on constituents and the composition rule.

**Definition 3 (Compositional Generalization Failure).** A model $f_\theta$ fails to compositionally generalize when there exists a distribution shift between training distribution $\mathcal{D}_{\text{train}}$ and test distribution $\mathcal{D}_{\text{test}}$ such that:

1. **Structural Novelty**: $\mathcal{D}_{\text{test}}$ contains compositions of primitives not seen together during training
2. **Performance Gap**: $\mathbb{E}_{(x,y) \sim \mathcal{D}_{\text{test}}}[\mathcal{L}(f_\theta(x), y)] \gg \mathbb{E}_{(x,y) \sim \mathcal{D}_{\text{train}}}[\mathcal{L}(f_\theta(x), y)]$
3. **Primitive Competence**: The model correctly handles individual primitives: $\forall v \in \mathcal{V}: f_\theta(v) \approx \text{correct}$

**Formal Characterization of the Gap:**

Let $\mathcal{C}_{\text{train}} \subseteq \mathcal{L}$ be the set of compositions seen during training. Define the compositional distance:

$$d_{\text{comp}}(x, \mathcal{C}_{\text{train}}) = \min_{c \in \mathcal{C}_{\text{train}}} |\text{parse}(x) \triangle \text{parse}(c)|$$

where $\triangle$ denotes symmetric difference of parse tree structures. Compositional generalization failure occurs when:

$$\text{Acc}(f_\theta, x) \propto \exp(-\alpha \cdot d_{\text{comp}}(x, \mathcal{C}_{\text{train}}))$$

for some $\alpha > 0$, indicating exponential degradation with compositional novelty.

---

### 1.2 Program-Like Representations

**Definition 4 (Program-Like Representation).** An internal representation $h \in \mathcal{H}$ is program-like if it satisfies the following properties:

1. **Compositionality**: There exists a composition operator $\oplus: \mathcal{H} \times \mathcal{H} \to \mathcal{H}$ such that:
   $$h_{\text{compose}(x_1, x_2)} = h_{x_1} \oplus h_{x_2}$$

2. **Abstraction**: The representation respects type/category abstraction:
   $$\forall x_1, x_2: \text{type}(x_1) = \text{type}(x_2) \implies \text{Abstract}(h_{x_1}) = \text{Abstract}(h_{x_2})$$

3. **Locality**: Modifications to one component have bounded effect on others:
   $$\|h_{\text{compose}(x_1', x_2)} - h_{\text{compose}(x_1, x_2)}\|_2 \leq L \cdot \|h_{x_1'} - h_{x_1}\|_2$$

**Definition 5 (Classical Abstract Interpretation Target).** Given:
- Concrete domain: $\mathcal{H} \subseteq \mathbb{R}^d$ (hidden representations)
- Abstract domain: $\mathcal{A}$ (symbolic abstractions like types, ranges)
- Abstraction function: $\alpha: \mathcal{H} \to \mathcal{A}$
- Concretization function: $\gamma: \mathcal{A} \to 2^{\mathcal{H}}$

Classical sound abstract interpretation would require a Galois connection:

$$\alpha(h) \sqsubseteq a \iff h \in \gamma(a)$$

and neural operations respect abstract semantics:

$$\forall h_1, h_2 \in \mathcal{H}: \alpha(\text{NeuralOp}(h_1, h_2)) \sqsubseteq \text{AbstractOp}(\alpha(h_1), \alpha(h_2))$$

The current implementation does not establish these properties. It implements
the differentiable consistency objective specified in `FORMAL_GUARANTEES.md`.

---

### 1.3 Why Likelihood-Based Training Fails

**Theorem 1 (Insufficiency of MLE for Compositionality).** Maximum likelihood estimation on finite compositional data does not guarantee compositional generalization.

**Proof Sketch:**

Consider a seq2seq model trained with cross-entropy loss:

$$\mathcal{L}_{\text{CE}} = -\sum_{(x,y) \in \mathcal{D}} \log p_\theta(y|x)$$

MLE finds parameters $\theta^*$ that maximize likelihood on $\mathcal{D}_{\text{train}}$. However:

1. **Multiple Minima**: Many parameter settings achieve similar training likelihood but differ drastically on compositional test cases
2. **Shortcut Learning**: The optimization landscape favors solutions that exploit surface statistics rather than compositional structure
3. **No Structural Inductive Bias**: Cross-entropy provides no gradient signal encouraging compositional representations

**Concrete Example:**

For SCAN "jump around right twice", MLE can achieve perfect training accuracy by:
- Memorizing "around right" → "RTURN RTURN RTURN RTURN"
- Memorizing "jump twice" → "JUMP JUMP"

But fail on "jump around left twice" because the composition was never seen, even though constituents were.

**What DAI Adds:**

Our abstraction loss provides explicit gradient signal that:
1. Encourages type consistency across representations
2. Penalizes disagreement between grounded parent abstractions and composed child abstractions
3. Tests whether this representation bias supports novel recombination

---

## 2. Scientific Hypotheses

### H1: Abstraction Constraint Hypothesis
Neural representations trained with the grounded abstract-domain consistency objective will exhibit improved OOD compositional generalization compared with matched no-structure controls.

**Measurable Prediction**: The paired-seed OOD exact-match difference is positive, with a 95% confidence interval reported for every primary comparison.

### H2: Type Consistency Hypothesis
Representations of semantically equivalent subexpressions will cluster in abstract space when trained with abstraction loss.

**Measurable Prediction**: Held-out grounded parent/child compositions have lower consistency violation than shuffled and random matched-span controls.

### H3: Structural Agreement Hypothesis
Predictions with stronger held-out grounded composition agreement will be more likely to be correct on OOD examples.

**Measurable Prediction**: Structural-consistency scores differ between correct and incorrect OOD predictions and correlate with per-example correctness.

### H4: Layer-Specificity Hypothesis
Different transformer layers benefit from different abstraction constraints; early layers benefit from type constraints, later layers from operational constraints.

**Measurable Prediction**: Predeclared single-layer ablations produce distinguishable paired OOD exact-match and held-out consistency estimates.

---

## 3. Assumptions and Scope

### Explicit Assumptions

1. **Data Assumption**: Training data contains sufficient primitive examples to learn basic operations
2. **Architecture Assumption**: Transformer attention can represent compositional operations given appropriate inductive bias
3. **Abstract Domain Assumption**: The primary learned type domain captures
   relevant compositional structure for the evaluated tasks; optional product
   and monotonicity domains are variants rather than primary-matrix claims
4. **Optimization Assumption**: The abstraction loss landscape is smooth enough for gradient-based optimization

### Scope Limitations

1. **Not Claimed**: DAI does not guarantee zero-shot generalization to arbitrary novel compositions
2. **Not Claimed**: DAI does not replace the need for diverse training data
3. **Not Claimed**: DAI is optimal for all compositional generalization tasks

### Potential Failure Modes

1. **Over-Constraint**: Too strong abstraction may prevent learning flexible representations
2. **Domain Mismatch**: Chosen abstract domain may not align with task structure
3. **Optimization Difficulty**: Joint optimization of task loss and abstraction loss may have conflicting gradients

---

## 4. Relationship to Prior Work

### 4.1 Differences from Standard Regularization

| Aspect | L2/Dropout Regularization | DAI Abstraction Loss |
|--------|---------------------------|---------------------|
| **Target** | Weight magnitudes | Representation structure |
| **Inductive Bias** | Simplicity/sparsity | Compositional hierarchy |
| **Theoretical Basis** | Bayesian priors | Abstract interpretation |
| **Failure Penalty** | All weights equally | Compositional violations specifically |

### 4.2 Differences from Neuro-Symbolic Methods

| Aspect | Traditional NeSy | DAI |
|--------|------------------|-----|
| **Symbolic Component** | Explicit programs/rules | Dataset-grounded composition annotations and learned abstract transfer rules |
| **Integration** | Modular/hybrid | End-to-end differentiable |
| **Supervision** | Often requires program labels | Input-output pairs plus deterministic composition structures |
| **Flexibility** | Limited by symbolic grammar | Soft constraints allow exceptions |

### 4.3 Differences from Structured Attention

| Aspect | Syntactic Attention | DAI |
|--------|---------------------|-----|
| **Parse Requirement** | Usually requires syntactic parses | Requires dataset-specific grounded composition structures during training |
| **Constraint Type** | Attention sparsity patterns | Representation semantics |
| **Layer Application** | Attention weights | Hidden representations |

### 4.4 Differences from Grammar-Aware and Tree-Based Methods

| Aspect | Grammar-aware parser / AM-Parser | Tree-linearized seq2seq | DAI |
|--------|----------------------------------|-------------------------|-----|
| **Structure at inference** | Predicts an explicit tree or graph | Generates a serialized structure | Generates the task output directly |
| **Training signal** | Gold or derived symbolic structures and parser actions | Source-only tree serialization in this repository | Input-output loss plus grounded span-composition consistency; SCAN structures are input-derived, while COGS/SLOG/CFQ use training-time gold-derived structural supervision |
| **Constraint** | Discrete grammar and decoder validity | Output ordering bias | Soft agreement between child and parent hidden-state abstractions |
| **Guarantee** | Determined by the parser grammar and decoder | No symbolic validity guarantee | No soundness or completeness guarantee for generated programs |

DAI's proposed distinction is therefore a training-time latent consistency
principle, not the use of structure by itself. Dataset grammars, explicit
parsers, structured attention, tree serialization, and classical
neuro-symbolic execution are prior structural approaches. Novelty claims must
be limited to the implemented objective and supported by matched controls;
they must not imply that DAI introduced grammar-aware compositional learning.

For COGS, SLOG, and CFQ, the structural annotations used by DAI are extracted
from the paired gold logical form or SPARQL query during training and aligned
back to source spans. They are not produced by an input-only parser. Generation
uses only the input and trained model. Any held-out composition-violation score
that uses these annotations is reported as a gold-assisted diagnostic rather
than a deployable inference-time measure.

---

## 5. Success Criteria

### Primary Success Metric
Paired-seed OOD exact match under the primary comparisons and multiplicity
control specified in `EXPERIMENT_PROTOCOL.md`.

### Secondary Success Metrics
1. IID-to-OOD generalization gap and accuracy by composition depth
2. Grounded consistency violation relative to matched structural controls
3. Compute, parameter, and optimization accounting for every method

### Minimum Viable Result
The central SCAN comparison is fully reported over all paired seeds, including
negative or inconclusive outcomes. Cross-benchmark claims require separately
validated COGS/SLOG and CFQ experiments and are not inferred from SCAN.
