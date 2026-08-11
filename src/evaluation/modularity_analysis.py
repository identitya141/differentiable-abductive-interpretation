"""
Modular vs. Holistic Processing Analysis

This module implements the key diagnostic from the research proposal that analyzes
whether the model processes compositional inputs modularly (independently processing
sub-parts then composing) or holistically (processing the entire input as one unit).

Key Concepts:
- Modular Processing: Sub-expressions are encoded independently, then combined
- Holistic Processing: The entire expression is encoded as a single unit

Diagnostic Methods:
1. Representation similarity analysis (RSA) - using abstraction-level composition
2. Gradient-based localization - with proper span detection and masking
3. Ablation studies (ΔNLL-based, not generation comparison)

Reference: Hupkes et al. (2020) "Compositionality Decomposed"
           Keysers et al. (2020) "Measuring Compositional Generalization"
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import json
import logging

import numpy as np
import torch
import torch.nn.functional as F

from src.evaluation.compositional_metrics import CompositionParser


logger = logging.getLogger(__name__)


# =============================================================================
# Token Span Alignment Utilities
# =============================================================================

def find_subtoken_spans(
    full_tokens: List[str],
    sub_tokens: List[str],
    normalize: bool = True,
) -> List[Tuple[int, int]]:
    """
    Find all occurrences of sub_tokens sequence within full_tokens.
    
    Args:
        full_tokens: Full sequence of tokens
        sub_tokens: Sub-sequence to find
        normalize: Whether to normalize tokens before matching
        
    Returns:
        List of (start, end) spans where sub_tokens appears
    """
    def norm(t: str) -> str:
        if normalize:
            return t.lower().replace("▁", "").replace("Ġ", "").strip()
        return t
    
    full_norm = [norm(t) for t in full_tokens]
    sub_norm = [norm(t) for t in sub_tokens]
    
    if not sub_norm:
        return []
    
    spans = []
    i = 0
    while i <= len(full_norm) - len(sub_norm):
        if full_norm[i:i+len(sub_norm)] == sub_norm:
            spans.append((i, i + len(sub_norm)))
            i += len(sub_norm)  # Non-overlapping
        else:
            i += 1
    
    return spans


def get_sub_expression_spans(
    tokenizer,
    full_text: str,
    sub_expressions: List[str],
) -> Dict[str, List[Tuple[int, int]]]:
    """
    Get token-level spans for each sub-expression within the full text.
    
    Args:
        tokenizer: HuggingFace tokenizer
        full_text: Full input text
        sub_expressions: List of sub-expressions to locate
        
    Returns:
        Dict mapping sub-expression to list of (start, end) token spans
    """
    # Tokenize full text
    full_tokens = tokenizer.tokenize(full_text)
    
    span_map = {}
    for sub_expr in sub_expressions:
        sub_tokens = tokenizer.tokenize(sub_expr)
        spans = find_subtoken_spans(full_tokens, sub_tokens)
        span_map[sub_expr] = spans
    
    return span_map


@dataclass
class ProcessingAnalysisResult:
    """Results from modular vs holistic analysis."""
    
    # Overall metrics
    modularity_score: float = 0.0  # [0, 1] where 1 = fully modular
    holistic_score: float = 0.0    # [0, 1] where 1 = fully holistic
    
    # Sub-expression analysis
    composition_predictability: float = 0.0  # R² of predicting full from parts
    sub_expression_independence: float = 0.0
    composition_alignment: float = 0.0
    
    # By complexity (only depth >= 2)
    modularity_by_depth: Dict[int, float] = field(default_factory=dict)
    count_by_depth: Dict[int, int] = field(default_factory=dict)
    
    # By composition type
    modularity_by_type: Dict[str, float] = field(default_factory=dict)
    
    # Stratified by correctness
    modularity_correct: Optional[float] = None
    modularity_incorrect: Optional[float] = None
    
    # Gradient localization
    gradient_localization_score: float = 0.0
    span_gradient_mass: Dict[str, float] = field(default_factory=dict)
    
    # RSA analysis (abstraction-level)
    rsa_abstraction_consistency: float = 0.0
    
    # Ablation results (ΔNLL-based)
    ablation_sensitivity: Dict[str, float] = field(default_factory=dict)
    ablation_specificity: float = 0.0  # How specific is ablation impact?
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "modularity_score": self.modularity_score,
            "holistic_score": self.holistic_score,
            "composition_predictability": self.composition_predictability,
            "sub_expression_independence": self.sub_expression_independence,
            "modularity_by_depth": self.modularity_by_depth,
            "count_by_depth": self.count_by_depth,
            "modularity_by_type": self.modularity_by_type,
            "modularity_correct": self.modularity_correct,
            "modularity_incorrect": self.modularity_incorrect,
            "gradient_localization_score": self.gradient_localization_score,
            "span_gradient_mass": self.span_gradient_mass,
            "rsa_abstraction_consistency": self.rsa_abstraction_consistency,
            "ablation_sensitivity": self.ablation_sensitivity,
            "ablation_specificity": self.ablation_specificity,
        }


class ModularityAnalyzer:
    """
    Analyze modular vs holistic processing in neural models.
    
    This implements the key diagnostic from the proposal that measures
    whether the model processes compositional structures modularly.
    
    Uses abstraction-level composition when available (DAI models),
    falling back to learned linear composition probe otherwise.
    """
    
    def __init__(
        self,
        dataset_type: str = "scan",
        device: Optional[torch.device] = None,
    ):
        """
        Initialize analyzer.
        
        Args:
            dataset_type: Type of dataset (scan, cogs, cfq)
            device: Computation device
        """
        self.dataset_type = dataset_type
        self.parser = CompositionParser(dataset_type)
        self.device = device or torch.device("cpu")
        
        # Cache for linear composition probe (fitted on demand)
        self._composition_probe = None
        self._probe_fitted = False
    
    def _get_representation(
        self,
        model,
        tokenizer,
        text: str,
        layer_idx: int = -1,
        use_abstraction: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Get pooled representation for text.
        
        Args:
            model: The model
            tokenizer: Tokenizer
            text: Input text
            layer_idx: Which layer to use
            use_abstraction: If True and model has abstraction layer, return both
            
        Returns:
            Tuple of (hidden_repr, abstract_repr) where abstract_repr may be None
        """
        device = self.device
        
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get actual sequence length
        seq_len = int(inputs["attention_mask"][0].sum().item())
        
        with torch.no_grad():
            if hasattr(model, 't5'):
                outputs = model.t5.encoder(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    output_hidden_states=True,
                    return_dict=True,
                )
                hidden = outputs.hidden_states[layer_idx]
            else:
                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    output_hidden_states=True,
                    return_dict=True,
                )
                hidden = outputs.hidden_states[layer_idx] if hasattr(outputs, 'hidden_states') else outputs.last_hidden_state
            
            # Slice to actual sequence length (exclude padding)
            hidden = hidden[:, :seq_len, :]
            mask = inputs["attention_mask"][:, :seq_len].unsqueeze(-1)
            
            # Mean pooling over actual tokens only
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            hidden_repr = pooled.squeeze(0)  # [hidden_dim]
            
            # Try to get abstraction if model has abstraction layer
            abstract_repr = None
            if use_abstraction and hasattr(model, 'abstraction_layer'):
                try:
                    # Get abstract representation from the abstraction layer
                    abstract_repr = model.abstraction_layer(hidden)
                    abstract_repr = abstract_repr.mean(dim=1).squeeze(0)  # Pool
                except Exception:
                    pass
        
        return hidden_repr, abstract_repr
    
    def analyze_representation_similarity(
        self,
        model,
        tokenizer,
        full_expression: str,
        sub_expressions: List[str],
        layer_idx: int = -1,
    ) -> Dict[str, float]:
        """
        Representational Similarity Analysis (RSA) using abstraction-level composition.
        
        Instead of comparing repr(A◦B) to mean(repr(A), repr(B)), we:
        1. If model has abstraction layer: compare γ(h_full) to α(γ(h_A), γ(h_B))
        2. Otherwise: measure predictability of repr(full) from [repr(A); repr(B)]
        
        This directly aligns with the DAI proposal's SubConstituentLoss.
        
        Args:
            model: The encoder model
            tokenizer: Tokenizer
            full_expression: The complete expression (e.g., "jump and walk")
            sub_expressions: List of sub-parts (e.g., ["jump", "walk"])
            layer_idx: Which layer to analyze (-1 for last)
            
        Returns:
            Dict with similarity metrics
        """
        if len(sub_expressions) < 2:
            return {"error": "Need at least 2 sub-expressions", "modularity_estimate": None}
        
        # Get representations (both hidden and abstract if available)
        full_hidden, full_abstract = self._get_representation(
            model, tokenizer, full_expression, layer_idx
        )
        
        sub_hiddens = []
        sub_abstracts = []
        for sub in sub_expressions:
            h, a = self._get_representation(model, tokenizer, sub, layer_idx)
            sub_hiddens.append(h)
            if a is not None:
                sub_abstracts.append(a)
        
        result = {}
        
        # Method 1: Abstraction-level composition (preferred for DAI models)
        if full_abstract is not None and len(sub_abstracts) == len(sub_expressions):
            # If model has abstract composition operator, use it
            if hasattr(model, 'abstraction_layer') and hasattr(model.abstraction_layer, 'composition_operator'):
                try:
                    composed_abstract = model.abstraction_layer.composition_operator(
                        torch.stack(sub_abstracts)
                    )
                except Exception:
                    composed_abstract = torch.stack(sub_abstracts).mean(dim=0)
            else:
                # Fallback: use element-wise operations that preserve abstract domain structure
                # For interval domain: intersection (min of uppers, max of lowers)
                # For general: use mean as approximation
                composed_abstract = torch.stack(sub_abstracts).mean(dim=0)
            
            # Consistency between full abstract and composed abstract
            abstraction_consistency = F.cosine_similarity(
                full_abstract.unsqueeze(0), composed_abstract.unsqueeze(0)
            ).item()
            
            result["abstraction_consistency"] = abstraction_consistency
            result["modularity_estimate"] = abstraction_consistency
        
        # Method 2: Linear composition predictability (for non-DAI models)
        else:
            # Concatenate sub-representations
            concat_subs = torch.cat(sub_hiddens)  # [2 * hidden_dim]
            
            # Measure how well full can be predicted from parts
            # Simple approach: cosine between full and projected concatenation
            # For proper R², would need to fit a linear probe on training data
            
            # Project concatenation to same dim as full (simple linear combination)
            # This is a rough approximation - better would be fitted probe
            if len(sub_hiddens) == 2:
                # Linear combination of parts
                combined = (sub_hiddens[0] + sub_hiddens[1]) / 2
                composition_sim = F.cosine_similarity(
                    full_hidden.unsqueeze(0), combined.unsqueeze(0)
                ).item()
            else:
                combined = torch.stack(sub_hiddens).mean(dim=0)
                composition_sim = F.cosine_similarity(
                    full_hidden.unsqueeze(0), combined.unsqueeze(0)
                ).item()
            
            result["composition_similarity"] = composition_sim
            
            # Predictability gain from using both parts vs. just one
            # Higher gain = more compositional (both parts matter)
            single_sims = [
                F.cosine_similarity(full_hidden.unsqueeze(0), h.unsqueeze(0)).item()
                for h in sub_hiddens
            ]
            best_single = max(single_sims)
            
            # Compositional gain: how much better is using both vs. best single?
            # Positive gain means composition helps
            compositional_gain = composition_sim - best_single
            result["compositional_gain"] = compositional_gain
            
            # Modularity: high if composition helps AND parts are distinguishable
            if len(sub_hiddens) >= 2:
                # Parts should be somewhat independent (not identical)
                # But "independence" shouldn't be orthogonality - measure distinguishability
                part_sim = F.cosine_similarity(
                    sub_hiddens[0].unsqueeze(0), sub_hiddens[1].unsqueeze(0)
                ).item()
                # Parts are distinguishable if similarity < 0.9 (not copies)
                parts_distinguishable = 1.0 if part_sim < 0.9 else (1 - part_sim) * 10
                
                # Modularity = composition helps AND parts contribute distinctly
                modularity = max(0, compositional_gain) * parts_distinguishable
                result["modularity_estimate"] = min(1.0, modularity + 0.5)  # Scale to [0.5, 1]
            else:
                result["modularity_estimate"] = composition_sim
        
        # Additional diagnostics
        result["avg_sub_similarity"] = float(np.mean([
            F.cosine_similarity(full_hidden.unsqueeze(0), h.unsqueeze(0)).item()
            for h in sub_hiddens
        ]))
        
        return result
    
    def compute_gradient_localization(
        self,
        model,
        tokenizer,
        input_text: str,
        target_text: str,
        sub_expressions: Optional[List[str]] = None,
        target_output_spans: Optional[List[Tuple[int, int]]] = None,
    ) -> Dict[str, float]:
        """
        Compute gradient-based localization with proper masking and span detection.
        
        Measures how localized the gradients are to relevant sub-expressions.
        For true modularity testing, we can condition on specific output spans.
        
        Modular: Gradients for output A are localized to input sub-expression A
        Holistic: Gradients are diffuse across the entire input
        
        Args:
            model: The model (will be set to train mode temporarily)
            tokenizer: Tokenizer
            input_text: Input text
            target_text: Target text
            sub_expressions: List of sub-expressions to compute span mass for
            target_output_spans: If provided, only backprop loss from these output positions
            
        Returns:
            Dict with gradient localization metrics
        """
        device = self.device
        was_training = model.training
        model.train()  # Need training mode for gradients
        
        try:
            # Tokenize input
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get actual input sequence length (exclude padding)
            input_seq_len = int(inputs["attention_mask"][0].sum().item())
            
            # Tokenize target
            targets = tokenizer(
                target_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            )
            targets = {k: v.to(device) for k, v in targets.items()}
            target_seq_len = int(targets["attention_mask"][0].sum().item())
            
            # Get input embeddings with gradient tracking
            if hasattr(model, 't5'):
                embeddings = model.t5.encoder.embed_tokens(inputs["input_ids"])
            else:
                embeddings = model.get_input_embeddings()(inputs["input_ids"])
            
            embeddings = embeddings.detach().clone()
            embeddings.requires_grad_(True)
            
            # Forward pass
            if hasattr(model, 't5'):
                outputs = model.t5(
                    inputs_embeds=embeddings,
                    attention_mask=inputs["attention_mask"],
                    labels=targets["input_ids"],
                    return_dict=True,
                )
            else:
                outputs = model(
                    inputs_embeds=embeddings,
                    attention_mask=inputs["attention_mask"],
                    labels=targets["input_ids"],
                    return_dict=True,
                )
            
            # Get loss - optionally mask to specific output positions
            if target_output_spans and hasattr(outputs, 'logits'):
                # Compute per-position loss and mask
                logits = outputs.logits  # [1, seq, vocab]
                labels = targets["input_ids"]  # [1, seq]
                
                # Create position mask
                position_mask = torch.zeros_like(labels, dtype=torch.float)
                for start, end in target_output_spans:
                    position_mask[0, start:end] = 1.0
                
                # Compute masked cross-entropy
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                per_position_loss = loss_fct(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1)
                ).view(labels.shape)
                
                loss = (per_position_loss * position_mask).sum() / (position_mask.sum() + 1e-10)
            else:
                loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Get gradient magnitudes - SLICE TO ACTUAL SEQ LEN (exclude padding)
            grad_magnitudes = embeddings.grad.abs().sum(dim=-1).squeeze(0)  # [padded_seq_len]
            grad_magnitudes = grad_magnitudes[:input_seq_len]  # Slice to actual length
            grad_magnitudes = grad_magnitudes / (grad_magnitudes.sum() + 1e-10)
            
            # Compute localization metrics on actual tokens only
            seq_len = input_seq_len
            
            # Entropy of gradient distribution (lower = more localized)
            grad_entropy = -(grad_magnitudes * (grad_magnitudes + 1e-10).log()).sum()
            max_entropy = torch.log(torch.tensor(float(seq_len), device=device))
            normalized_entropy = grad_entropy / max_entropy
            
            localization_score = 1 - normalized_entropy.item()
            
            # Compute span-specific gradient mass if sub-expressions provided
            span_gradient_mass = {}
            if sub_expressions:
                # Get token-level spans for each sub-expression
                span_map = get_sub_expression_spans(tokenizer, input_text, sub_expressions)
                
                for sub_expr, spans in span_map.items():
                    if spans:
                        # Sum gradient mass in all spans for this sub-expression
                        total_span_mass = 0.0
                        for start, end in spans:
                            if end <= seq_len:
                                total_span_mass += grad_magnitudes[start:end].sum().item()
                        span_gradient_mass[sub_expr] = total_span_mass
            
            result = {
                "localization_score": localization_score,
                "gradient_entropy": grad_entropy.item(),
                "input_seq_len": seq_len,
                "span_gradient_mass": span_gradient_mass,
            }
            
            # Compute contrast ratio if we have span masses
            if span_gradient_mass:
                total_span_mass = sum(span_gradient_mass.values())
                outside_mass = 1.0 - total_span_mass
                contrast_ratio = total_span_mass / (outside_mass + 1e-10)
                result["span_contrast_ratio"] = contrast_ratio
            
            return result
            
        finally:
            # Restore original training mode
            if not was_training:
                model.eval()
    
    def ablation_study(
        self,
        model,
        tokenizer,
        full_expression: str,
        sub_expressions: List[str],
        target_text: str,
    ) -> Dict[str, float]:
        """
        ΔNLL-based ablation study to measure compositional processing.
        
        Uses teacher-forced log-likelihood instead of unstable generation comparison.
        Measures sensitivity (ΔNLL when masking) and specificity (does masking A
        mostly affect output positions corresponding to A?).
        
        Args:
            model: The model
            tokenizer: Tokenizer
            full_expression: Complete expression
            sub_expressions: List of sub-expressions to ablate
            target_text: Target output for NLL computation
            
        Returns:
            Dict with ablation sensitivity and specificity scores
        """
        device = self.device
        model.eval()
        
        def compute_nll(input_text: str, target_text: str) -> Tuple[float, torch.Tensor]:
            """Compute negative log-likelihood via teacher forcing."""
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            )
            targets = tokenizer(
                target_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            targets = {k: v.to(device) for k, v in targets.items()}
            
            with torch.no_grad():
                if hasattr(model, 't5'):
                    outputs = model.t5(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        labels=targets["input_ids"],
                        return_dict=True,
                    )
                else:
                    outputs = model(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        labels=targets["input_ids"],
                        return_dict=True,
                    )
                
                # Get per-position loss for specificity analysis
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                    labels = targets["input_ids"]
                    loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
                    per_position_nll = loss_fct(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1)
                    ).view(labels.shape)
                else:
                    per_position_nll = None
                
                total_nll = outputs.loss.item()
            
            return total_nll, per_position_nll
        
        # Get baseline NLL
        baseline_nll, baseline_per_pos = compute_nll(full_expression, target_text)
        
        # Determine the sentinel token for T5
        if hasattr(tokenizer, 'additional_special_tokens') and tokenizer.additional_special_tokens:
            sentinel_token = "<extra_id_0>"
        else:
            # Fallback to pad token masking
            sentinel_token = tokenizer.pad_token or "<pad>"
        
        sensitivities = {}
        specificities = {}
        
        for i, sub_expr in enumerate(sub_expressions):
            # Create ablated input using proper T5 sentinel tokens
            # Replace the sub-expression tokens with a single sentinel
            ablated_input = full_expression.replace(sub_expr, sentinel_token, 1)
            
            if ablated_input == full_expression:
                # Sub-expression not found
                sensitivities[sub_expr] = 0.0
                continue
            
            try:
                ablated_nll, ablated_per_pos = compute_nll(ablated_input, target_text)
                
                # Sensitivity = ΔNLL (how much does masking hurt?)
                delta_nll = ablated_nll - baseline_nll
                sensitivities[sub_expr] = max(0, delta_nll)
                
                # Specificity: Does masking cause localized vs. diffuse degradation?
                # High specificity = masking A hurts output A, not everything
                if baseline_per_pos is not None and ablated_per_pos is not None:
                    # Compare per-position NLL changes
                    if baseline_per_pos.shape == ablated_per_pos.shape:
                        per_pos_delta = (ablated_per_pos - baseline_per_pos).squeeze(0)
                        
                        # Check if degradation is localized
                        # (For SCAN: if we mask "jump", mainly "JUMP" tokens should degrade)
                        # Measure via entropy of the delta distribution
                        delta_abs = per_pos_delta.abs()
                        if delta_abs.sum() > 1e-10:
                            delta_norm = delta_abs / delta_abs.sum()
                            entropy = -(delta_norm * (delta_norm + 1e-10).log()).sum().item()
                            max_entropy = np.log(len(delta_norm))
                            specificity = 1 - (entropy / max_entropy) if max_entropy > 0 else 0
                            specificities[sub_expr] = specificity
                
            except Exception as e:
                logger.warning(f"Ablation failed for '{sub_expr}': {e}")
                sensitivities[sub_expr] = 0.0
        
        # Aggregate metrics
        sensitivity_values = list(sensitivities.values())
        specificity_values = list(specificities.values()) if specificities else []
        
        result = {
            "baseline_nll": baseline_nll,
            "per_sub_sensitivity": sensitivities,
            "per_sub_specificity": specificities,
            "avg_sensitivity": float(np.mean(sensitivity_values)) if sensitivity_values else 0.0,
            "avg_specificity": float(np.mean(specificity_values)) if specificity_values else 0.0,
        }
        
        # Modularity interpretation:
        # - High sensitivity = each part matters (good)
        # - High specificity = effects are localized (modular)
        # - Low specificity = effects are diffuse (holistic)
        if sensitivity_values and specificity_values:
            result["ablation_modularity"] = float(np.mean(specificity_values))
        elif sensitivity_values:
            # If we only have sensitivity, high sensitivity suggests parts matter
            # but doesn't tell us about modularity vs holism
            result["ablation_modularity"] = 0.5  # Neutral
        else:
            result["ablation_modularity"] = 0.0
        
        return result
    
    def analyze_example(
        self,
        model,
        tokenizer,
        input_text: str,
        target_text: Optional[str] = None,
        is_correct: Optional[bool] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Complete modular/holistic analysis for a single example.
        
        Returns None for depth < 2 examples since modularity is undefined
        for simple (non-compositional) inputs.
        
        Args:
            model: The model
            tokenizer: Tokenizer
            input_text: Input text
            target_text: Optional target text
            is_correct: Whether the model's prediction was correct (for stratification)
            
        Returns:
            Dict with all analysis metrics, or None if depth < 2
        """
        # Get sub-expressions
        sub_expressions = self.parser.get_sub_constituents(input_text)
        composition_type = self.parser.get_composition_type(input_text)
        depth = self.parser.get_depth(input_text)
        
        # Modularity analysis is undefined for simple expressions
        if depth < 2 or len(sub_expressions) < 2:
            logger.debug(f"Skipping non-compositional example (depth={depth}): {input_text[:50]}...")
            return None
        
        result = {
            "input": input_text,
            "depth": depth,
            "composition_type": composition_type,
            "sub_expressions": sub_expressions,
            "is_correct": is_correct,
        }
        
        # RSA analysis with abstraction-level composition
        rsa_result = self.analyze_representation_similarity(
            model, tokenizer, input_text, sub_expressions
        )
        result["rsa"] = rsa_result
        
        # Use abstraction_consistency as primary modularity metric (if available)
        # Fall back to compositional_gain, then modularity_estimate
        modularity = rsa_result.get(
            "abstraction_consistency",
            rsa_result.get(
                "compositional_gain",
                rsa_result.get("modularity_estimate", 0.5)
            )
        )
        result["modularity_score"] = modularity
        
        # Ablation study with ΔNLL
        if target_text and len(sub_expressions) >= 1:
            try:
                ablation_result = self.ablation_study(
                    model, tokenizer, input_text, sub_expressions, target_text
                )
                result["ablation"] = ablation_result
                
                # Include ablation modularity in final score
                if "ablation_modularity" in ablation_result:
                    result["ablation_modularity"] = ablation_result["ablation_modularity"]
            except Exception as e:
                logger.warning(f"Ablation failed: {e}")
                result["ablation"] = {}
        
        return result
    
    def analyze_batch(
        self,
        model,
        tokenizer,
        examples: List[Dict[str, str]],
        max_examples: int = 100,
        include_predictions: bool = False,
    ) -> ProcessingAnalysisResult:
        """
        Analyze modular/holistic processing for a batch of examples.
        
        Properly stratifies results by correctness and depth.
        Only includes depth >= 2 examples in modularity averages.
        
        Args:
            model: The model
            tokenizer: Tokenizer
            examples: List of {"input": ..., "target": ..., "prediction": ... (optional), "is_correct": ... (optional)}
            max_examples: Maximum examples to analyze
            include_predictions: If True, generate predictions if not provided
            
        Returns:
            ProcessingAnalysisResult with aggregate statistics
        """
        results = []
        modularity_by_depth = {}
        modularity_by_type = {}
        modularity_correct = []
        modularity_incorrect = []
        skipped_simple = 0
        
        for i, example in enumerate(examples[:max_examples]):
            try:
                # Determine correctness if not provided
                is_correct = example.get("is_correct")
                if is_correct is None and "prediction" in example and "target" in example:
                    is_correct = example["prediction"].strip() == example["target"].strip()
                
                result = self.analyze_example(
                    model, tokenizer,
                    example["input"],
                    example.get("target"),
                    is_correct=is_correct,
                )
                
                # Skip simple (depth < 2) examples
                if result is None:
                    skipped_simple += 1
                    continue
                
                results.append(result)
                modularity = result["modularity_score"]
                
                # Aggregate by depth (only depth >= 2 examples get here)
                depth = result["depth"]
                if depth not in modularity_by_depth:
                    modularity_by_depth[depth] = []
                modularity_by_depth[depth].append(modularity)
                
                # Aggregate by type
                comp_type = result["composition_type"]
                if comp_type not in modularity_by_type:
                    modularity_by_type[comp_type] = []
                modularity_by_type[comp_type].append(modularity)
                
                # Stratify by correctness
                if result.get("is_correct") is True:
                    modularity_correct.append(modularity)
                elif result.get("is_correct") is False:
                    modularity_incorrect.append(modularity)
                
            except Exception as e:
                logger.warning(f"Analysis failed for example {i}: {e}")
        
        # Compute aggregates (only from depth >= 2 examples)
        all_modularity = [r["modularity_score"] for r in results]
        avg_modularity = float(np.mean(all_modularity)) if all_modularity else 0.0
        
        # Average by depth
        depth_avgs = {
            d: float(np.mean(scores)) 
            for d, scores in modularity_by_depth.items()
        }
        
        # Average by type
        type_avgs = {
            t: float(np.mean(scores)) 
            for t, scores in modularity_by_type.items()
        }
        
        # Correctness stratification
        modularity_on_correct = float(np.mean(modularity_correct)) if modularity_correct else None
        modularity_on_incorrect = float(np.mean(modularity_incorrect)) if modularity_incorrect else None
        
        # Ablation metrics
        ablation_modularities = [
            r.get("ablation_modularity", 0.0) 
            for r in results 
            if "ablation" in r and r.get("ablation_modularity") is not None
        ]
        avg_ablation_modularity = float(np.mean(ablation_modularities)) if ablation_modularities else None
        
        logger.info(f"Analyzed {len(results)} compositional examples "
                   f"(skipped {skipped_simple} simple examples with depth < 2)")
        
        return ProcessingAnalysisResult(
            modularity_score=avg_modularity,
            holistic_score=1 - avg_modularity,
            modularity_by_depth=depth_avgs,
            modularity_by_type=type_avgs,
            modularity_on_correct=modularity_on_correct,
            modularity_on_incorrect=modularity_on_incorrect,
            ablation_modularity=avg_ablation_modularity,
            num_examples=len(results),
            num_skipped_simple=skipped_simple,
        )


def compare_model_modularity(
    models: Dict[str, Any],
    tokenizer,
    examples: List[Dict[str, str]],
    dataset_type: str = "scan",
    output_path: Optional[Path] = None,
) -> Dict[str, ProcessingAnalysisResult]:
    """
    Compare modularity across multiple models.
    
    Args:
        models: Dict mapping model name to model instance
        tokenizer: Tokenizer
        examples: Test examples
        dataset_type: Dataset type
        output_path: Path to save comparison report
        
    Returns:
        Dict mapping model name to ProcessingAnalysisResult
    """
    analyzer = ModularityAnalyzer(dataset_type=dataset_type)
    results = {}
    
    for name, model in models.items():
        logger.info(f"Analyzing {name}...")
        result = analyzer.analyze_batch(model, tokenizer, examples)
        results[name] = result
    
    # Save comparison
    if output_path:
        comparison = {
            name: result.to_dict() 
            for name, result in results.items()
        }
        with open(output_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        logger.info(f"Saved comparison to {output_path}")
    
    return results
