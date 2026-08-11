# Paper Artifacts Guide

This document provides templates and guidelines for generating publication-ready artifacts from DAI experiment results.

## Table of Contents

1. [Main Results Table](#main-results-table)
2. [Ablation Study Tables](#ablation-study-tables)
3. [Figures](#figures)
4. [Contribution Bullets](#contribution-bullets)
5. [Limitations Section](#limitations-section)
6. [LaTeX Templates](#latex-templates)

---

## Main Results Table

### Template: Compositional Generalization Results

```latex
\begin{table*}[t]
\centering
\caption{Compositional generalization results. We report exact match accuracy
(\%) on out-of-distribution test splits as mean $\pm$ sample standard deviation
over five paired seeds.}
\label{tab:main_results}
\begin{tabular}{lccccccc}
\toprule
\textbf{Method} & \multicolumn{2}{c}{\textbf{SCAN}} & \textbf{COGS} & \textbf{CFQ} & \multicolumn{2}{c}{\textbf{CLUTRR}} & \textbf{GSM8K} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-4} \cmidrule(lr){5-5} \cmidrule(lr){6-7} \cmidrule(lr){8-8}
& Length & AddPrim & Gen & MCD-avg & k≤6 & k>6 & Test \\
\midrule
\multicolumn{8}{l}{\textit{Baselines}} \\
Vanilla T5 & -- & -- & -- & -- & -- & -- & -- \\
Chain-of-Thought & -- & -- & -- & -- & -- & -- & -- \\
Scratchpad & -- & -- & -- & -- & -- & -- & -- \\
NMN-style & -- & -- & -- & -- & -- & -- & -- \\
\midrule
\multicolumn{8}{l}{\textit{Our Method}} \\
DAI (Type) & -- & -- & -- & -- & -- & -- & -- \\
DAI (Interval) & -- & -- & -- & -- & -- & -- & -- \\
DAI (Monotonicity) & -- & -- & -- & -- & -- & -- & -- \\
DAI (TypeMono) & \textbf{--} & \textbf{--} & \textbf{--} & \textbf{--} & \textbf{--} & \textbf{--} & \textbf{--} \\
\bottomrule
\end{tabular}
\end{table*}
```

Populate only benchmark columns backed by completed, validated artifacts. Do
not insert expected percentages into result tables.

---

## Ablation Study Tables

### Table A1: Abstraction Loss Weight

```latex
\begin{table}[h]
\centering
\caption{Effect of abstraction loss weight $\lambda$ on SCAN-Length accuracy.
Too low values fail to enforce structure; too high values over-constrain.}
\label{tab:ablation_lambda}
\begin{tabular}{lcccc}
\toprule
$\lambda$ & 0.01 & 0.1 & 0.5 & 1.0 \\
\midrule
IID Accuracy & -- & -- & -- & -- \\
OOD Accuracy & -- & -- & -- & -- \\
Gen. Gap & -- & -- & -- & -- \\
\bottomrule
\end{tabular}
\end{table}
```

### Table A2: Abstract Domain Comparison

```latex
\begin{table}[h]
\centering
\caption{Comparison of abstract domains across benchmarks.}
\label{tab:ablation_domains}
\begin{tabular}{lcccc}
\toprule
Domain & SCAN & COGS & CFQ & Avg \\
\midrule
None (baseline) & -- & -- & -- & -- \\
Type only & -- & -- & -- & -- \\
Interval only & -- & -- & -- & -- \\
Monotonicity only & -- & -- & -- & -- \\
Type + Monotonicity & -- & -- & -- & -- \\
\bottomrule
\end{tabular}
\end{table}
```

### Table A3: Constrained Layers

```latex
\begin{table}[h]
\centering
\caption{Effect of which transformer layers receive abstraction constraints.}
\label{tab:ablation_layers}
\begin{tabular}{lccc}
\toprule
Constrained Layers & SCAN-Len & COGS-Gen & CFQ-MCD \\
\midrule
None & -- & -- & -- \\
{[}0, 1{]} (early) & -- & -- & -- \\
{[}2, 4{]} (middle) & -- & -- & -- \\
{[}4, 5{]} (late) & -- & -- & -- \\
{[}0, 2, 4{]} (all) & -- & -- & -- \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Figures

### Figure 1: Method Overview Diagram

**Description:** A schematic diagram showing:
1. Input sequence → T5 Encoder with abstraction layers
2. Abstraction layer detail: span pooling, $\alpha_\theta$, abstract composition, and the parent/composed-child divergence
3. Three abstract domain visualizations (Type, Interval, Monotonicity)
4. Combined loss: L_task + λ·L_abstraction

**Generation Script:** `scripts/generate_figures.py --figure method_overview`

### Figure 2: Generalization Gap Analysis

**Description:** Line plot showing:
- X-axis: Training steps
- Y-axis: Accuracy (%)
- Lines: IID accuracy, OOD accuracy for baseline vs. DAI
- Shaded region: Generalization gap

**Data Required:**
- Training logs with per-step evaluation on IID and OOD splits

**Generation Script:** `scripts/generate_figures.py --figure gen_gap`

### Figure 3: Abstract Representation Visualization

**Description:** t-SNE or UMAP visualization of:
- Left: Baseline representations (no clear structure)
- Right: DAI representations (clustered by abstract type)

**Data Required:**
- Hidden states from model on evaluation set
- Ground truth abstract types

**Generation Script:** `scripts/generate_figures.py --figure representation_viz`

### Figure 4: Scaling Analysis

**Description:** Plot showing:
- X-axis: Compositional depth / sequence length
- Y-axis: Accuracy (%)
- Comparison of degradation rate: Baseline vs DAI

**Generation Script:** `scripts/generate_figures.py --figure scaling`

### Figure 5: Over-Constraint Detection

**Description:** Heatmap or bar chart showing:
- X-axis: Abstraction loss weight
- Y-axis: Metrics (IID accuracy, OOD accuracy, constraint violation)
- Clear "sweet spot" visualization

---

## Contribution Bullets

### Primary Contributions (NeurIPS/ICLR Safe)

1. **Grounded Consistency Objective:** We introduce differentiable abstract-domain consistency objectives that bias transformer representations toward dataset-grounded compositional structure.

2. **Formal Scope:** We define the learned abstraction and operator-specific composition maps, prove the limited zero-divergence agreement property, and state explicitly that the method does not establish classical soundness or a Galois connection.

3. **Controlled Evaluation:** We compare against matched no-auxiliary, bottleneck, entropy, and contrastive controls, plus shuffled and random structural falsification conditions. Report benchmark claims only after the preregistered runs are complete.

4. **Reproducible Methodology:** We provide deterministic grammar extraction, exact tokenizer alignment, paired-seed configurations, per-example artifacts, and standard-library statistical comparison tools.

### Secondary Contributions

5. **Analysis of Abstraction Types:** Reserved for results after the corresponding ablations are run.

6. **Efficiency Analysis:** Report measured wall-clock, accelerator time, throughput, and peak memory without asserting an overhead threshold in advance.

---

## Limitations Section

### Honest Limitations (Reviewer-Aware Wording)

```latex
\section{Limitations}

Our work has several limitations that suggest directions for future research:

\paragraph{Abstract Domain Selection.} 
While we provide guidelines for choosing abstract domains, the optimal choice 
remains task-dependent. Automatically learning or discovering appropriate 
abstract domains is an open problem.

\paragraph{Scalability to Larger Models.}
Our experiments focus on T5-small and T5-base. While the method should scale 
to larger models, we have not verified this empirically due to computational 
constraints. The interaction between model capacity and abstraction constraints 
warrants further investigation.

\paragraph{Task Scope.}
Our benchmarks focus on semantic parsing and multi-hop reasoning. 
Compositional generalization in other domains (vision-language, 
code generation) may require different abstract domains.

\paragraph{Over-Constraint Trade-offs.}
Although we introduce detection mechanisms, precisely balancing abstraction 
constraints with expressivity remains challenging. Adaptive constraint 
scheduling could further improve results.

\paragraph{Comparison with Symbolic Methods.}
Pure neuro-symbolic methods that use explicit program synthesis may 
achieve stronger guarantees on specific tasks. Our method trades formal 
guarantees for generality and end-to-end differentiability.
```

### Known Failure Cases

Document these explicitly in supplementary material:

Populate this section only from checked per-example prediction artifacts. At a
minimum, stratify observed failures by operator, composition depth, output
length, and structural novelty; do not state unmeasured thresholds or failure
modes as empirical findings.

---

## LaTeX Templates

### Proposition Environment

```latex
\begin{proposition}[Zero-Loss Abstract Agreement]
\label{thm:main}
Let $D_A$ be the implemented sum of symmetric KL divergence and squared
distance. For a grounded composition $(L,R,P,o)$, if its consistency loss is
zero, then the compared type and monotonicity distributions agree:
\[
\alpha_\theta(h_P) = C_{A,o}(\alpha_\theta(h_L), \alpha_\theta(h_R)).
\]
This proposition does not imply abstract-interpretation soundness or
generalization to unseen compositions.
\end{proposition}
```

### Algorithm Environment

```latex
\begin{algorithm}[t]
\caption{DAI Training}
\label{alg:dai_training}
\begin{algorithmic}[1]
\Require Dataset $\mathcal{D}$, abstraction $\alpha$, weight $\lambda$
\State Initialize T5 model $f_\theta$
\For{epoch $= 1$ to $N$}
    \State $\lambda_t \gets \text{schedule}(\lambda, \text{epoch})$
    \For{batch $(x, y) \in \mathcal{D}$}
        \State $h \gets \text{Encoder}(x)$
        \State $a \gets \alpha(h)$
        \State $\mathcal{L}_\text{task} \gets \text{CrossEntropy}(f_\theta(x), y)$
        \State $\mathcal{L}_\text{abs} \gets \mathcal{L}_\text{consistency}(h, a) + \mathcal{L}_\text{compose}(h, a)$
        \State $\mathcal{L} \gets \mathcal{L}_\text{task} + \lambda_t \cdot \mathcal{L}_\text{abs}$
        \State Update $\theta$ via gradient descent on $\mathcal{L}$
    \EndFor
\EndFor
\Return $f_\theta$
\end{algorithmic}
\end{algorithm}
```

---

## Generating Paper-Ready Artifacts

### Commands

```bash
# Generate all tables from results
make generate-tables

# Generate depth/operator/category breakdown tables
python scripts/generate_breakdown_tables.py \
    --experiment-dir results/publication \
    --methods scan_full_contrastive scan_reference_t5 \
    --seeds 42 123 456 789 1024 \
    --output-dir results/publication/breakdowns

# Inventory the frozen source package on a Sapelo compute node
sbatch slurm/slurm_release_manifest.sh
    --results-dir results/ \
    --output-dir docs/paper/ \
    --format markdown
```

### Checklist Before Submission

- [ ] Primary tables report all five paired seeds, sample standard deviation,
      paired 95% confidence intervals, effect sizes, and corrected tests
- [ ] Figure fonts are ≥8pt
- [ ] Color scheme is colorblind-friendly
- [ ] All hyperparameters documented in appendix
- [ ] Code and data URLs included
- [ ] Reproducibility checklist completed
- [ ] Ethics statement (if required)
- [ ] Compute requirements documented
