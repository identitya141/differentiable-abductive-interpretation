"""
Evaluation Metrics for Compositional Generalization

This module provides comprehensive evaluation metrics specifically designed
for measuring compositional generalization performance.

Metrics Include:
1. In-distribution accuracy
2. Out-of-distribution compositional accuracy
3. Generalization gap
4. Per-primitive accuracy
5. Per-composition-depth accuracy
6. Exact match vs. partial match
7. Training stability metrics
8. Over-constraint failure rate
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import re

import numpy as np
import torch
from torch.utils.data import DataLoader

# Import composition parser for self-sufficient depth/primitive extraction
try:
    from .compositional_metrics import CompositionParser
    HAS_PARSER = True
except ImportError:
    HAS_PARSER = False


# =============================================================================
# Output Normalization for Fair Cross-Baseline Evaluation
# =============================================================================
# These functions ensure EM/compositional metrics are fair across baselines
# (Vanilla, CoT, Scratchpad, LLaMA) and datasets (SCAN, COGS, CFQ).

_COT_MARKERS = [
    "Therefore, the answer is:",
    "Therefore, the answer is",
    "The answer is:",
    "The answer is",
    "Answer:",
    "Final answer:",
    "So the answer is:",
    "So the answer is",
    "####",  # common in math datasets (GSM8K)
]

_LLAMA_RESPONSE_MARKERS = [
    "### Response:",
    "### Response",
    "Response:",
    "Response",
]


def _collapse_ws(s: str) -> str:
    """Collapse all whitespace to single spaces."""
    return " ".join(s.strip().split())


def _strip_llama_scaffold(s: str) -> str:
    """
    Removes typical instruction scaffolding if the model accidentally
    echoes it back (or if you decode a full prompt+response).
    Keeps only the portion after the last response marker if present.
    """
    text = s.strip()
    for marker in _LLAMA_RESPONSE_MARKERS:
        if marker in text:
            text = text.split(marker)[-1].strip()
    return text


def _extract_cot_final_answer(s: str) -> str:
    """
    Extract the answer portion from a chain-of-thought output.
    If no marker found, returns the original string trimmed.
    """
    text = s.strip()
    # Use the *last* marker occurrence to be robust
    found = None
    found_idx = -1
    for m in _COT_MARKERS:
        idx = text.rfind(m)
        if idx > found_idx:
            found = m
            found_idx = idx
    if found is not None and found_idx >= 0:
        return text[found_idx + len(found):].strip(" \n\t:-")
    return text


def _strip_scratchpad(s: str, start: str = "[SCRATCH]", end: str = "[/SCRATCH]") -> str:
    """
    Remove scratchpad region. If end token appears, keep text after it.
    If only start token appears, keep text before it (or after, depending on convention).
    Mimics ScratchpadT5.extract_answer which keeps AFTER [/SCRATCH].
    """
    text = s.strip()
    if end in text:
        return text.split(end)[-1].strip()
    if start in text:
        return text.split(start)[-1].strip()
    return text


def _strip_code_fences(s: str) -> str:
    """Remove accidental triple backticks blocks or markdown artifacts."""
    text = s.strip()
    if text.startswith("```") and text.endswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def _normalize_scan_actions(s: str) -> str:
    """
    SCAN targets are action tokens like: 'LTURN JUMP LTURN JUMP'
    Canonicalize spacing + uppercase (SCAN gold is uppercase).
    """
    text = _collapse_ws(s)
    return text.upper()


def _normalize_cogs_lf(s: str) -> str:
    """
    COGS targets are logical forms. Minimal normalization to avoid
    changing semantics: collapse whitespace only.
    """
    return _collapse_ws(s)


def _normalize_cfq_sparql(s: str) -> str:
    """
    CFQ targets are SPARQL. Exact string match is brittle due to whitespace.
    This normalizer:
      - collapses whitespace,
      - normalizes spacing around punctuation,
      - uppercases SPARQL keywords.
    NOTE: Not full SPARQL canonicalization; pragmatic normalizer.
    """
    text = s.strip()
    text = _collapse_ws(text)

    # Normalize spacing around braces, parens, dots, commas, semicolons
    text = re.sub(r"\s*{\s*", " { ", text)
    text = re.sub(r"\s*}\s*", " } ", text)
    text = re.sub(r"\s*\(\s*", " ( ", text)
    text = re.sub(r"\s*\)\s*", " ) ", text)
    text = re.sub(r"\s*;\s*", " ; ", text)
    text = re.sub(r"\s*,\s*", " , ", text)
    text = re.sub(r"\s*\.\s*", " . ", text)

    # Normalize SPARQL keywords to uppercase
    for kw in ["select", "where", "filter", "optional", "distinct", "ask"]:
        text = re.sub(rf"\b{kw}\b", kw.upper(), text, flags=re.IGNORECASE)

    return _collapse_ws(text)


def _normalize_gsm8k_answer(s: str) -> str:
    """
    GSM8K answers are typically numeric. Extract the final number.
    """
    text = s.strip()
    # Extract after #### if present (GSM8K format)
    if "####" in text:
        text = text.split("####")[-1].strip()
    # Try to extract numeric answer
    numbers = re.findall(r'-?[\d,]+\.?\d*', text)
    if numbers:
        # Return the last number found (typically the final answer)
        return numbers[-1].replace(",", "")
    return _collapse_ws(text)


def normalize_for_eval(
    output: str,
    dataset_type: str,
    baseline_type: Optional[str] = None,
) -> str:
    """
    Normalize a model output string so EM / compositional metrics are fair
    across baselines and datasets.

    This function ensures:
    - CoT outputs have reasoning stripped, only final answer evaluated
    - Scratchpad outputs have scratch regions removed
    - LLaMA outputs have prompt scaffolding removed
    - Dataset-specific canonicalization (SCAN uppercase, CFQ SPARQL normalization)

    Args:
        output: decoded model output
        dataset_type: "scan" | "cogs" | "cfq" | "gsm8k" | "clutrr"
        baseline_type: optional hint ("vanilla"|"cot"|"scratchpad"|"llama"|...)

    Returns:
        normalized string intended for exact-match comparison.
    """
    if output is None:
        return ""

    text = output

    # 1) Remove obvious formatting artifacts first
    text = _strip_code_fences(text)

    # 2) Remove LLaMA scaffolding if present (safe to apply generally)
    text = _strip_llama_scaffold(text)

    # 3) Baseline-specific extraction
    if baseline_type is not None:
        bt = baseline_type.lower()
        if bt in {"cot", "chainofthought", "chain-of-thought", "chain_of_thought"}:
            text = _extract_cot_final_answer(text)
        elif bt in {"scratchpad", "scratch"}:
            text = _strip_scratchpad(text)
        elif bt in {"llama", "llama_baseline", "llama-baseline", "tinyllama", "tinyllama_lora"}:
            # LLaMA sometimes produces reasoning + answer; try CoT extraction as fallback
            extracted = _extract_cot_final_answer(text)
            text = extracted if extracted != text else text
    else:
        # If baseline_type unknown, still apply extraction if markers appear
        if "[/SCRATCH]" in text or "[SCRATCH]" in text:
            text = _strip_scratchpad(text)
        if any(m in text for m in _COT_MARKERS):
            text = _extract_cot_final_answer(text)

    # 4) Dataset-specific canonicalization
    dt = dataset_type.lower()
    if dt == "scan":
        text = _normalize_scan_actions(text)
    elif dt == "cogs":
        text = _normalize_cogs_lf(text)
    elif dt == "cfq":
        text = _normalize_cfq_sparql(text)
    elif dt == "gsm8k":
        text = _normalize_gsm8k_answer(text)
    elif dt == "clutrr":
        # CLUTRR: relationship words, lowercase and collapse whitespace
        text = _collapse_ws(text.lower())
    else:
        text = _collapse_ws(text)

    return text


def normalize_batch_for_eval(
    outputs: List[str],
    dataset_type: str,
    baseline_type: Optional[str] = None,
) -> List[str]:
    """
    Batch version of normalize_for_eval for convenience.
    
    Args:
        outputs: List of decoded model outputs
        dataset_type: "scan" | "cogs" | "cfq" | "gsm8k" | "clutrr"
        baseline_type: optional baseline hint
        
    Returns:
        List of normalized strings
    """
    return [normalize_for_eval(o, dataset_type, baseline_type) for o in outputs]


@dataclass
class EvaluationResult:
    """
    Comprehensive evaluation result.
    
    Note on token_accuracy:
        For generation (model.generate()), token_accuracy is computed as prefix
        accuracy (longest matching prefix ratio) since raw token-by-token comparison
        is meaningless for variable-length outputs. For teacher-forced accuracy,
        use compute_token_accuracy_teacher_forced() on forward pass logits.
    """
    # Core metrics
    accuracy: float
    exact_match: float
    
    # Partial match metrics
    # For generation: this is prefix accuracy (more meaningful than token-by-token)
    # For teacher-forced: use compute_token_accuracy_teacher_forced() separately
    token_accuracy: Optional[float] = None
    avg_pred_length: Optional[float] = None
    avg_target_length: Optional[float] = None
    length_ratio: Optional[float] = None  # pred_len / target_len
    
    # Compositional metrics
    in_distribution_accuracy: float = 0.0
    out_of_distribution_accuracy: float = 0.0
    generalization_gap: float = 0.0
    
    # Breakdown
    accuracy_by_depth: Dict[int, float] = None
    accuracy_by_primitive: Dict[str, float] = None
    accuracy_by_category: Dict[str, float] = None
    
    # Stability
    std_across_seeds: Optional[float] = None
    
    # Detailed
    num_examples: int = 0
    num_correct: int = 0
    predictions: Optional[List] = None
    targets: Optional[List] = None
    inputs: Optional[List[str]] = None
    is_ood: Optional[List[bool]] = None
    categories: Optional[List[Optional[str]]] = None
    composition_violations: Optional[List[Optional[float]]] = None
    
    def __post_init__(self):
        if self.accuracy_by_depth is None:
            self.accuracy_by_depth = {}
        if self.accuracy_by_primitive is None:
            self.accuracy_by_primitive = {}
        if self.accuracy_by_category is None:
            self.accuracy_by_category = {}


def compute_exact_match_accuracy(
    predictions: List[str],
    targets: List[str],
    dataset_type: str = "scan",
) -> float:
    """
    Compute exact match accuracy with dataset-specific normalization.
    
    Args:
        predictions: List of predicted strings
        targets: List of target strings
        dataset_type: Dataset type for normalization (scan, cogs, cfq, gsm8k, clutrr)
        
    Returns:
        Exact match accuracy (0-1)
    """
    if len(predictions) == 0:
        return 0.0
    
    # Apply dataset-specific normalization for fair comparison
    pred_normalized = [normalize_for_eval(p, dataset_type) for p in predictions]
    target_normalized = [normalize_for_eval(t, dataset_type) for t in targets]
    
    correct = sum(p == t for p, t in zip(pred_normalized, target_normalized))
    return correct / len(predictions)


def compute_token_accuracy(
    predictions: List[List[int]],
    targets: List[List[int]],
    ignore_index: int = -100,
) -> float:
    """
    Compute token-level accuracy for aligned sequences.
    
    WARNING: This assumes predictions and targets are aligned token-by-token.
    For generation outputs (model.generate()), use prefix_accuracy instead.
    This function is primarily for teacher-forced predictions from logits.
    
    Args:
        predictions: List of predicted token sequences
        targets: List of target token sequences
        ignore_index: Index to ignore in targets
        
    Returns:
        Token accuracy (0-1)
    """
    total_tokens = 0
    correct_tokens = 0
    
    for pred, target in zip(predictions, targets):
        for p, t in zip(pred, target):
            if t != ignore_index:
                total_tokens += 1
                if p == t:
                    correct_tokens += 1
    
    return correct_tokens / max(1, total_tokens)


def compute_token_accuracy_teacher_forced(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> float:
    """
    Compute token accuracy from teacher-forced forward pass logits.
    
    This is the correct way to compute token accuracy for seq2seq models.
    For generation outputs, use prefix_accuracy or exact_match instead.
    
    Args:
        logits: Model output logits [batch, seq_len, vocab_size]
        labels: Target labels [batch, seq_len]
        ignore_index: Index to ignore in labels (typically -100)
        
    Returns:
        Token accuracy (0-1) computed from argmax(logits) vs labels
    """
    predictions = logits.argmax(dim=-1)  # [batch, seq_len]
    
    # Mask out ignored positions
    mask = labels != ignore_index
    
    correct = (predictions == labels) & mask
    total = mask.sum().item()
    
    if total == 0:
        return 0.0
    
    return correct.sum().item() / total


def compute_length_diagnostics(
    predictions: List[str],
    targets: List[str],
) -> Dict[str, float]:
    """
    Compute length-related diagnostics for debugging generation issues.
    
    Returns:
        Dictionary with avg_pred_length, avg_target_length, length_ratio
    """
    if len(predictions) == 0:
        return {"avg_pred_length": 0.0, "avg_target_length": 0.0, "length_ratio": 0.0}
    
    pred_lengths = [len(p.strip().split()) for p in predictions]
    target_lengths = [len(t.strip().split()) for t in targets]
    
    avg_pred = sum(pred_lengths) / len(pred_lengths)
    avg_target = sum(target_lengths) / len(target_lengths)
    
    return {
        "avg_pred_length": avg_pred,
        "avg_target_length": avg_target,
        "length_ratio": avg_pred / max(1, avg_target),
    }


def compute_prefix_accuracy(
    predictions: List[str],
    targets: List[str],
) -> float:
    """
    Compute prefix accuracy (longest matching prefix).
    
    Useful for seq2seq where partial matches matter.
    """
    if len(predictions) == 0:
        return 0.0
    
    total_ratio = 0.0
    
    for pred, target in zip(predictions, targets):
        pred_tokens = pred.strip().split()
        target_tokens = target.strip().split()
        
        matching = 0
        for p, t in zip(pred_tokens, target_tokens):
            if p == t:
                matching += 1
            else:
                break
        
        if len(target_tokens) > 0:
            total_ratio += matching / len(target_tokens)
    
    return total_ratio / len(predictions)


class CompositionalMetrics:
    """
    Metrics specifically for compositional generalization.
    
    Self-sufficient: can derive depth/primitives from input text using parser,
    without requiring external functions or pre-computed attributes.
    """
    
    def __init__(
        self,
        tokenizer=None,
        dataset_type: str = "scan",
    ):
        """
        Initialize metrics.
        
        Args:
            tokenizer: Tokenizer for decoding predictions
            dataset_type: Type of dataset for parser (scan, cogs, cfq, etc.)
        """
        self.tokenizer = tokenizer
        self.dataset_type = dataset_type
        
        # Use built-in parser for depth/primitive extraction (self-sufficient)
        self.parser = CompositionParser(dataset_type) if HAS_PARSER else None
    
    def compute(
        self,
        predictions: List,
        targets: List,
        inputs: Optional[List[str]] = None,
        is_ood: Optional[List[bool]] = None,
        categories: Optional[List[Optional[str]]] = None,
        depths: Optional[List[Optional[int]]] = None,
    ) -> EvaluationResult:
        """
        Compute comprehensive compositional metrics.
        
        Args:
            predictions: Model predictions (tokens or strings)
            targets: Ground truth targets
            inputs: Input text strings (for depth/primitive analysis)
            is_ood: Whether each example is out-of-distribution
            
        Returns:
            EvaluationResult with all metrics
        """
        # Issue #5: Guard against empty predictions
        if len(predictions) == 0:
            return EvaluationResult(
                accuracy=0.0,
                exact_match=0.0,
                num_examples=0,
                num_correct=0,
            )
        
        # Decode if necessary
        # NOTE: SCAN action tokens (I_WALK, I_RUN, etc.) are added as regular
        # vocab tokens (not special tokens), so skip_special_tokens=True is safe
        # and correctly strips only EOS/PAD/etc.
        if self.tokenizer is not None and isinstance(predictions[0], (list, torch.Tensor)):
            pred_strings = [
                self.tokenizer.decode(p, skip_special_tokens=True)
                for p in predictions
            ]
            # Replace -100 with pad_token_id before decoding (more robust than filtering)
            target_strings = []
            for tgt in targets:
                t = np.array(tgt)
                t = np.where(t == -100, self.tokenizer.pad_token_id, t)
                target_strings.append(self.tokenizer.decode(t, skip_special_tokens=True))
        else:
            pred_strings = predictions
            target_strings = targets
        
        # Normalize predictions and targets for fair comparison
        # This applies dataset-specific canonicalization (SCAN uppercase, CFQ SPARQL, etc.)
        pred_normalized = [normalize_for_eval(p, self.dataset_type) for p in pred_strings]
        target_normalized = [normalize_for_eval(t, self.dataset_type) for t in target_strings]
        
        # Compute exact match with explicit count (fixes issue #6)
        num_correct_count = sum(
            p == t for p, t in zip(pred_normalized, target_normalized)
        )
        exact_match = num_correct_count / len(pred_strings)
        
        # Split by in/out-of-distribution
        if is_ood is not None:
            id_correct = 0
            id_total = 0
            ood_correct = 0
            ood_total = 0
            
            for pred, target, ood in zip(pred_normalized, target_normalized, is_ood):
                correct = pred == target
                if ood:
                    ood_total += 1
                    if correct:
                        ood_correct += 1
                else:
                    id_total += 1
                    if correct:
                        id_correct += 1
            
            id_acc = id_correct / max(1, id_total)
            ood_acc = ood_correct / max(1, ood_total)
        else:
            id_acc = exact_match
            ood_acc = exact_match
        
        # Generalization gap
        gen_gap = id_acc - ood_acc
        
        # Accuracy by composition depth (now uses parser on inputs, not examples)
        # Use normalized strings for consistent comparison
        accuracy_by_depth = self._compute_accuracy_by_depth(
            pred_normalized, target_normalized, inputs, depths
        )
        
        # Accuracy by primitive (now uses parser on inputs, not examples)
        # Use normalized strings for consistent comparison
        accuracy_by_primitive = self._compute_accuracy_by_primitive(
            pred_normalized, target_normalized, inputs
        )
        accuracy_by_category = self._compute_accuracy_by_category(
            pred_normalized, target_normalized, categories
        )
        
        # Length diagnostics (critical for debugging EM issues)
        # Use normalized strings for consistent length comparison
        length_diag = compute_length_diagnostics(pred_normalized, target_normalized)
        
        # Prefix accuracy (more meaningful than misaligned token accuracy for generation)
        # Use normalized strings for consistent comparison
        prefix_acc = compute_prefix_accuracy(pred_normalized, target_normalized)
        
        return EvaluationResult(
            accuracy=exact_match,
            exact_match=exact_match,
            token_accuracy=prefix_acc,  # Using prefix accuracy as proxy (see docstring)
            avg_pred_length=length_diag["avg_pred_length"],
            avg_target_length=length_diag["avg_target_length"],
            length_ratio=length_diag["length_ratio"],
            in_distribution_accuracy=id_acc,
            out_of_distribution_accuracy=ood_acc,
            generalization_gap=gen_gap,
            accuracy_by_depth=accuracy_by_depth,
            accuracy_by_primitive=accuracy_by_primitive,
            accuracy_by_category=accuracy_by_category,
            num_examples=len(predictions),
            num_correct=num_correct_count,  # Use exact count, not float*len
            predictions=pred_strings,
            targets=target_strings,
            inputs=inputs,
            is_ood=is_ood,
            categories=categories,
        )

    @staticmethod
    def _compute_accuracy_by_category(
        predictions: List[str],
        targets: List[str],
        categories: Optional[List[Optional[str]]],
    ) -> Dict[str, float]:
        if categories is None:
            return {}
        correct = defaultdict(int)
        total = defaultdict(int)
        for prediction, target, category in zip(predictions, targets, categories):
            if not category:
                continue
            total[category] += 1
            correct[category] += int(prediction == target)
        return {
            category: correct[category] / count
            for category, count in sorted(total.items())
        }
    
    def _compute_accuracy_by_depth(
        self,
        predictions: List[str],
        targets: List[str],
        inputs: Optional[List[str]],
        depths: Optional[List[Optional[int]]] = None,
    ) -> Dict[int, float]:
        """
        Compute accuracy stratified by composition depth.
        
        Uses parser to extract depth from input text (self-sufficient).
        """
        depth_correct = defaultdict(int)
        depth_total = defaultdict(int)

        if depths is not None:
            items = zip(predictions, targets, depths)
        elif inputs is not None and self.parser is not None:
            items = (
                (prediction, target, self.parser.get_depth(input_text))
                for prediction, target, input_text in zip(
                    predictions, targets, inputs
                )
            )
        else:
            return {}

        for pred, target, depth in items:
            if depth is None:
                continue
            
            depth_total[depth] += 1
            # Strings are already normalized, compare directly
            if pred == target:
                depth_correct[depth] += 1
        
        return {
            d: depth_correct[d] / max(1, depth_total[d])
            for d in sorted(depth_total.keys())
        }
    
    def _compute_accuracy_by_primitive(
        self,
        predictions: List[str],
        targets: List[str],
        inputs: Optional[List[str]],
    ) -> Dict[str, float]:
        """
        Compute accuracy stratified by primitives in input.
        
        Uses parser to extract primitives from input text (self-sufficient).
        """
        if inputs is None or self.parser is None:
            return {}
        
        primitive_correct = defaultdict(int)
        primitive_total = defaultdict(int)
        
        for pred, target, inp in zip(predictions, targets, inputs):
            primitives = self.parser.get_primitives(inp)
            # Strings are already normalized, compare directly
            correct = pred == target
            
            for prim in primitives:
                primitive_total[prim] += 1
                if correct:
                    primitive_correct[prim] += 1
        
        return {
            p: primitive_correct[p] / max(1, primitive_total[p])
            for p in sorted(primitive_total.keys())
        }


class OverConstraintMetrics:
    """
    Metrics for detecting over-constraint from abstraction loss.
    """
    
    def __init__(self):
        self.task_losses = []
        self.abstraction_losses = []
        self.accuracies = []
    
    def update(
        self,
        task_loss: float,
        abstraction_loss: float,
        accuracy: float,
    ):
        """Record metrics for one evaluation."""
        self.task_losses.append(task_loss)
        self.abstraction_losses.append(abstraction_loss)
        self.accuracies.append(accuracy)
    
    def compute_over_constraint_rate(self) -> Dict[str, float]:
        """
        Compute over-constraint metrics.
        
        Returns:
            Dictionary with:
            - over_constraint_rate: Fraction of time accuracy decreased
            - task_loss_correlation: Correlation between abstraction loss and task loss
        """
        if len(self.accuracies) < 2:
            return {"over_constraint_rate": 0.0}
        
        # Compute rate of accuracy decrease when abstraction loss is high
        high_abs_loss_threshold = np.percentile(self.abstraction_losses, 75)
        
        acc_changes = np.diff(self.accuracies)
        high_abs_loss = np.array(self.abstraction_losses[:-1]) > high_abs_loss_threshold
        
        over_constraint_count = np.sum((acc_changes < 0) & high_abs_loss)
        over_constraint_rate = over_constraint_count / max(1, np.sum(high_abs_loss))
        
        # Correlation between abstraction loss and task loss
        if len(self.task_losses) > 1:
            correlation = np.corrcoef(self.task_losses, self.abstraction_losses)[0, 1]
        else:
            correlation = 0.0
        
        return {
            "over_constraint_rate": float(over_constraint_rate),
            "loss_correlation": float(correlation),
            "mean_abstraction_loss": float(np.mean(self.abstraction_losses)),
            "std_abstraction_loss": float(np.std(self.abstraction_losses)),
        }


class TrainingStabilityMetrics:
    """
    Metrics for evaluating training stability.
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.losses = []
        self.accuracies = []
    
    def update(self, loss: float, accuracy: Optional[float] = None):
        """Record training metrics."""
        self.losses.append(loss)
        if accuracy is not None:
            self.accuracies.append(accuracy)
    
    def compute_stability(self) -> Dict[str, float]:
        """
        Compute stability metrics.
        
        Returns:
            Dictionary with:
            - loss_variance: Variance of loss over training
            - loss_trend: Linear trend of loss (negative = decreasing = good)
            - convergence_epoch: Estimated epoch of convergence
        """
        if len(self.losses) < self.window_size:
            return {"loss_variance": 0.0, "loss_trend": 0.0}
        
        losses = np.array(self.losses)
        
        # Rolling variance
        rolling_var = []
        for i in range(len(losses) - self.window_size):
            window = losses[i:i + self.window_size]
            rolling_var.append(np.var(window))
        
        # Linear trend
        x = np.arange(len(losses))
        slope, _ = np.polyfit(x, losses, 1)
        
        # Estimate convergence (when rolling variance drops below threshold)
        var_threshold = np.mean(rolling_var) * 0.1
        convergence_idx = len(losses)  # Default: not converged
        for i, v in enumerate(rolling_var):
            if v < var_threshold:
                convergence_idx = i + self.window_size
                break
        
        return {
            "loss_variance": float(np.mean(rolling_var)),
            "loss_trend": float(slope),
            "convergence_step": convergence_idx,
            "final_loss": float(losses[-1]),
        }


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    tokenizer,
    device: torch.device = None,
    dataset_type: str = "scan",
    max_new_tokens: int = 256,
    use_optimized_generation: bool = True,
) -> EvaluationResult:
    """
    Evaluate a model on a dataloader.
    
    Args:
        model: Model to evaluate
        dataloader: Data to evaluate on
        tokenizer: Tokenizer for decoding
        device: Device to run on
        dataset_type: Type of dataset for parser (scan, cogs, cfq, etc.)
        max_new_tokens: Maximum tokens to generate
        use_optimized_generation: Use EOS-ban and beam search for SCAN
        
    Returns:
        EvaluationResult with comprehensive metrics
    """
    # Import here to avoid circular imports
    from src.utils.generation import generate_scan_optimized, get_generation_config
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    model.to(device)
    
    all_predictions = []
    all_targets = []
    all_inputs = []  # Fixed: actually collect input texts
    all_is_ood = []
    all_categories = []
    all_composition_violations = []
    
    # Get optimized generation config for this dataset
    gen_config = get_generation_config(dataset_type, tokenizer=tokenizer) if use_optimized_generation else {}
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch.input_ids.to(device)
            attention_mask = batch.attention_mask.to(device)
            
            # Generate predictions with optimized settings
            if use_optimized_generation:
                generated = generate_scan_optimized(
                    model=model,
                    tokenizer=tokenizer,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **gen_config,
                )
            else:
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                )
            
            all_predictions.extend(generated.cpu().tolist())
            all_targets.extend(batch.labels.cpu().tolist())
            
            # Fixed issue #1: Collect input texts from batch
            if (
                hasattr(batch, 'original_input_texts')
                and batch.original_input_texts is not None
            ):
                all_inputs.extend(batch.original_input_texts)
            elif hasattr(batch, 'input_texts') and batch.input_texts is not None:
                all_inputs.extend(batch.input_texts)
            elif hasattr(batch, 'text'):
                all_inputs.extend(batch.text)
            else:
                # Fallback: decode input_ids
                for ids in input_ids:
                    decoded = tokenizer.decode(ids, skip_special_tokens=True)
                    all_inputs.append(decoded)
            
            # Collect is_ood if available
            if hasattr(batch, 'is_ood') and batch.is_ood is not None:
                all_is_ood.extend(batch.is_ood.cpu().tolist())
            if (
                hasattr(batch, 'generalization_categories')
                and batch.generalization_categories is not None
            ):
                all_categories.extend(batch.generalization_categories)

            composition_specs = getattr(batch, 'composition_specs', None)
            if (
                composition_specs is not None
                and hasattr(model, 'get_abstraction_diagnostics')
            ):
                model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=batch.labels.to(device),
                    compute_abstraction_loss=True,
                    composition_specs=composition_specs,
                )
                diagnostics = model.get_abstraction_diagnostics()
                violation_sums = [0.0] * input_ids.size(0)
                violation_counts = [0.0] * input_ids.size(0)
                for layer_diagnostics in diagnostics.values():
                    values = layer_diagnostics.get('loss_composition_per_example')
                    counts = layer_diagnostics.get(
                        'loss_composition_count_per_example'
                    )
                    if values is None or counts is None:
                        continue
                    for index, (value, count) in enumerate(zip(values, counts)):
                        count_value = float(count.detach().item())
                        violation_sums[index] += (
                            float(value.detach().item()) * count_value
                        )
                        violation_counts[index] += count_value
                all_composition_violations.extend(
                    total / count if count else None
                    for total, count in zip(violation_sums, violation_counts)
                )
    
    # Compute metrics with inputs for depth/primitive analysis
    metrics = CompositionalMetrics(
        tokenizer=tokenizer,
        dataset_type=dataset_type,
    )
    
    result = metrics.compute(
        predictions=all_predictions,
        targets=all_targets,
        inputs=all_inputs if all_inputs else None,
        is_ood=all_is_ood if all_is_ood else None,
        categories=all_categories if all_categories else None,
    )
    result.composition_violations = (
        all_composition_violations if all_composition_violations else None
    )
    
    model.train()
    return result


def run_comprehensive_evaluation(
    model: torch.nn.Module,
    dataloaders: Dict[str, DataLoader],
    tokenizer,
    device: torch.device = None,
    dataset_type: str = "scan",
) -> Dict[str, EvaluationResult]:
    """
    Run evaluation on multiple splits/datasets.
    
    Args:
        model: Model to evaluate
        dataloaders: Dictionary of name -> DataLoader
        tokenizer: Tokenizer
        device: Device
        dataset_type: Type of dataset for parser
        
    Returns:
        Dictionary of name -> EvaluationResult
    """
    results = {}
    
    for name, dataloader in dataloaders.items():
        print(f"Evaluating on {name}...")
        results[name] = evaluate_model(
            model, dataloader, tokenizer, device, dataset_type=dataset_type
        )
        print(f"  Accuracy: {results[name].accuracy:.4f}")
        print(f"  OOD Accuracy: {results[name].out_of_distribution_accuracy:.4f}")
        print(f"  Gen Gap: {results[name].generalization_gap:.4f}")
        if results[name].accuracy_by_depth:
            print(f"  Depth breakdown: {results[name].accuracy_by_depth}")
    
    return results
