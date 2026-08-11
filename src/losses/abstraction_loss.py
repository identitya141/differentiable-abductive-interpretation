"""
Abstraction Loss Functions for DAI

This module provides the core loss functions that enforce abstract interpretation
constraints on neural representations. The main loss penalizes violations of
abstract domain properties.

Key Loss Components:
1. Concretization Loss: Penalizes hidden states that violate their abstract constraints
2. Composition Loss: Penalizes when composed representations don't match composed abstractions
3. Consistency Loss: Enforces consistent abstractions for semantically similar inputs
4. Type Checking Loss: Penalizes invalid type transitions (input→output)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.abstract_domains import AbstractDomain, AbstractElement


@dataclass
class AbstractionLossOutput:
    """
    Output from abstraction loss computation.
    """
    total_loss: torch.Tensor
    concretization_loss: torch.Tensor
    composition_loss: Optional[torch.Tensor] = None
    consistency_loss: Optional[torch.Tensor] = None
    type_checking_loss: Optional[torch.Tensor] = None
    
    # Diagnostics
    loss_components: Optional[Dict[str, torch.Tensor]] = None


class AbstractionLoss(nn.Module):
    """
    Main abstraction loss class.
    
    This loss enforces that neural hidden states respect abstract domain constraints.
    It is the core training signal that biases representations toward program-like structure.
    
    Mathematical Formulation:
    
    L_abstraction = Σ_l λ_l * L_concretize(h_l, α(h_l))
                  + μ * L_compose
                  + ν * L_consistency
    
    where:
    - L_concretize: Penalizes hidden states outside their abstract regions
    - L_compose: Penalizes when f(α(h1), α(h2)) ≠ α(f(h1, h2))
    - L_consistency: Penalizes inconsistent abstractions for similar inputs
    """
    
    def __init__(
        self,
        abstract_domain: AbstractDomain,
        concretization_weight: float = 1.0,
        composition_weight: float = 0.5,
        consistency_weight: float = 0.1,
        entropy_regularization: float = 0.1,
        contrastive_weight: float = 0.0,
        structural_contrastive_weight: float = 0.0,
        composition_objective: str = "domain",
        contrastive_temperature: float = 0.1,
        temperature: float = 1.0,
    ):
        """
        Initialize abstraction loss.
        
        Args:
            abstract_domain: The abstract domain to enforce
            concretization_weight: Weight for concretization loss
            composition_weight: Weight for composition loss
            consistency_weight: Weight for consistency loss
            entropy_regularization: Regularize abstraction entropy
            temperature: Temperature for soft abstractions
        """
        super().__init__()
        
        self.abstract_domain = abstract_domain
        self.concretization_weight = concretization_weight
        self.composition_weight = composition_weight
        self.consistency_weight = consistency_weight
        self.entropy_regularization = entropy_regularization
        self.contrastive_weight = contrastive_weight
        self.structural_contrastive_weight = structural_contrastive_weight
        if composition_objective not in {"domain", "mse", "cosine"}:
            raise ValueError(
                "composition_objective must be one of: domain, mse, cosine"
            )
        self.composition_objective = composition_objective
        if contrastive_temperature <= 0:
            raise ValueError("contrastive_temperature must be positive")
        self.contrastive_temperature = contrastive_temperature
        self.temperature = temperature
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        attention_weights: Optional[torch.Tensor] = None,  # For compatibility with CompositionAwareAbstractionLoss
        composition_pairs: Optional[List[Tuple[int, int, int]]] = None,
        composition_specs: Optional[List[List[Any]]] = None,
        similar_pairs: Optional[List[Tuple[int, int]]] = None,
    ) -> AbstractionLossOutput:
        """
        Compute abstraction loss.
        
        Args:
            hidden_states: Hidden states [batch, seq, hidden_dim]
            attention_mask: Mask for padding [batch, seq]
            attention_weights: Attention weights (unused in base class, for subclass compatibility)
            composition_pairs: List of (i, j, k) where position k is composition of i, j
            composition_specs: Per-example child and parent token spans
            similar_pairs: List of (i, j) where positions i, j should have similar abstractions
            
        Returns:
            AbstractionLossOutput with all loss components
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Compute abstractions
        abstraction = self.abstract_domain.abstract(hidden_states)
        
        # 1. Concretization loss
        concretization_loss = self.abstract_domain.concretize_loss(
            hidden_states, abstraction, attention_mask=attention_mask
        )
        
        # 2. Composition loss (if composition pairs provided)
        composition_loss = None
        composition_count = 0
        composition_per_example = hidden_states.new_zeros(hidden_states.size(0))
        composition_count_per_example = hidden_states.new_zeros(hidden_states.size(0))
        if composition_specs is not None and any(composition_specs):
            (
                composition_loss,
                composition_count,
                composition_per_example,
                composition_count_per_example,
            ) = self._compute_span_composition_loss(
                hidden_states, attention_mask, composition_specs
            )
        elif composition_pairs is not None and len(composition_pairs) > 0:
            composition_loss = self._compute_composition_loss(
                hidden_states, abstraction, composition_pairs
            )
            composition_count = len(composition_pairs) * batch_size

        contrastive_loss = None
        contrastive_count = 0
        if (
            self.contrastive_weight > 0
            and composition_specs is not None
            and any(composition_specs)
        ):
            contrastive_loss, contrastive_count = self._compute_contrastive_loss(
                hidden_states, attention_mask, composition_specs
            )

        structural_contrastive_loss = None
        structural_contrastive_count = 0
        if (
            self.structural_contrastive_weight > 0
            and composition_specs is not None
            and any(composition_specs)
        ):
            (
                structural_contrastive_loss,
                structural_contrastive_count,
            ) = self._compute_structural_contrastive_loss(
                hidden_states, attention_mask, composition_specs
            )
        
        # 3. Consistency loss (if similar pairs provided)
        consistency_loss = None
        if similar_pairs is not None and len(similar_pairs) > 0:
            consistency_loss = self._compute_consistency_loss(
                hidden_states, abstraction, similar_pairs
            )
        
        # 4. Entropy regularization
        entropy_loss = self._compute_entropy_regularization(
            abstraction, attention_mask=attention_mask
        )
        
        # Combine losses
        total_loss = self.concretization_weight * concretization_loss
        total_loss = total_loss + self.entropy_regularization * entropy_loss
        
        if composition_loss is not None:
            total_loss = total_loss + self.composition_weight * composition_loss

        if contrastive_loss is not None:
            total_loss = total_loss + self.contrastive_weight * contrastive_loss

        if structural_contrastive_loss is not None:
            total_loss = (
                total_loss
                + self.structural_contrastive_weight * structural_contrastive_loss
            )
        
        if consistency_loss is not None:
            total_loss = total_loss + self.consistency_weight * consistency_loss
        
        return AbstractionLossOutput(
            total_loss=total_loss,
            concretization_loss=concretization_loss,
            composition_loss=composition_loss,
            consistency_loss=consistency_loss,
            loss_components={
                'concretization': concretization_loss,
                'entropy': entropy_loss,
                'composition': composition_loss if composition_loss is not None else torch.tensor(0.0),
                'composition_count': hidden_states.new_tensor(float(composition_count)),
                'composition_per_example': composition_per_example,
                'composition_count_per_example': composition_count_per_example,
                'contrastive': contrastive_loss if contrastive_loss is not None else hidden_states.new_zeros(()),
                'contrastive_count': hidden_states.new_tensor(float(contrastive_count)),
                'structural_contrastive': structural_contrastive_loss if structural_contrastive_loss is not None else hidden_states.new_zeros(()),
                'structural_contrastive_count': hidden_states.new_tensor(float(structural_contrastive_count)),
                'consistency': consistency_loss if consistency_loss is not None else torch.tensor(0.0),
            }
        )

    def _compute_structural_contrastive_loss(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        composition_specs: List[List[Any]],
    ) -> Tuple[torch.Tensor, int]:
        """Match composed children to their parent against in-batch parents."""
        composed_features = []
        parent_features = []
        operators = []
        sequence_length = hidden_states.size(1)

        for batch_index, example_specs in enumerate(composition_specs):
            valid_length = (
                int(attention_mask[batch_index].sum().item())
                if attention_mask is not None
                else sequence_length
            )
            for spec in example_specs:
                spans = (spec.left_span, spec.right_span, spec.parent_span)
                if any(
                    start < 0 or start >= end or end > valid_length
                    for start, end in spans
                ):
                    raise ValueError(
                        f"Invalid composition spans {spans} for valid length {valid_length}"
                    )
                left_hidden = self._pool_span(
                    hidden_states, batch_index, spec.left_span
                )
                right_hidden = self._pool_span(
                    hidden_states, batch_index, spec.right_span
                )
                parent_hidden = self._pool_span(
                    hidden_states, batch_index, spec.parent_span
                )
                composed_abstract = self.abstract_domain.compose(
                    self.abstract_domain.abstract(left_hidden),
                    self.abstract_domain.abstract(right_hidden),
                    spec.operator,
                )
                parent_abstract = self.abstract_domain.abstract(parent_hidden)
                composed_features.append(
                    self._contrastive_features(
                        composed_abstract, (left_hidden + right_hidden) / 2
                    )
                )
                parent_features.append(
                    self._contrastive_features(parent_abstract, parent_hidden)
                )
                operators.append(spec.operator)

        if len(composed_features) < 2:
            return hidden_states.new_zeros(()), 0

        composed_tensor = F.normalize(torch.cat(composed_features, dim=0), dim=-1)
        parent_tensor = F.normalize(torch.cat(parent_features, dim=0), dim=-1)
        logits = (
            composed_tensor @ parent_tensor.transpose(0, 1)
        ) / self.contrastive_temperature
        targets = torch.arange(logits.size(0), device=logits.device)
        # Equivalent operators are not negatives; masking them avoids pushing
        # reusable instances of the same composition rule apart.
        same_operator = torch.tensor(
            [[left == right for right in operators] for left in operators],
            device=logits.device,
            dtype=torch.bool,
        )
        same_operator.fill_diagonal_(False)
        logits = logits.masked_fill(same_operator, float("-inf"))
        return F.cross_entropy(logits, targets), logits.size(0)

    def _compute_contrastive_loss(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        composition_specs: List[List[Any]],
    ) -> Tuple[torch.Tensor, int]:
        """Supervised contrastive loss over parent abstractions by operator."""
        features = []
        operators = []
        sequence_length = hidden_states.size(1)

        for batch_index, example_specs in enumerate(composition_specs):
            valid_length = (
                int(attention_mask[batch_index].sum().item())
                if attention_mask is not None
                else sequence_length
            )
            for spec in example_specs:
                start, end = spec.parent_span
                if start < 0 or start >= end or end > valid_length:
                    raise ValueError(
                        f"Invalid parent span {spec.parent_span} for valid length {valid_length}"
                    )
                parent_hidden = self._pool_span(
                    hidden_states, batch_index, spec.parent_span
                )
                parent_abstract = self.abstract_domain.abstract(parent_hidden)
                features.append(self._contrastive_features(parent_abstract, parent_hidden))
                operators.append(spec.operator)

        if len(features) < 2:
            return hidden_states.new_zeros(()), 0

        feature_tensor = F.normalize(torch.cat(features, dim=0), dim=-1)
        similarities = feature_tensor @ feature_tensor.transpose(0, 1)
        similarities = similarities / self.contrastive_temperature
        sample_count = similarities.size(0)
        identity_mask = torch.eye(
            sample_count, device=similarities.device, dtype=torch.bool
        )
        positive_mask = torch.tensor(
            [
                [left == right for right in operators]
                for left in operators
            ],
            device=similarities.device,
            dtype=torch.bool,
        ) & ~identity_mask
        valid_anchors = positive_mask.any(dim=1)
        if not valid_anchors.any():
            return hidden_states.new_zeros(()), 0

        denominator_logits = similarities.masked_fill(identity_mask, float('-inf'))
        log_probabilities = similarities - torch.logsumexp(
            denominator_logits, dim=1, keepdim=True
        )
        positive_counts = positive_mask.sum(dim=1).clamp(min=1)
        anchor_losses = -(
            log_probabilities.masked_fill(~positive_mask, 0.0).sum(dim=1)
            / positive_counts
        )
        return anchor_losses[valid_anchors].mean(), int(valid_anchors.sum().item())

    @staticmethod
    def _contrastive_features(
        abstraction: AbstractElement,
        parent_hidden: torch.Tensor,
    ) -> torch.Tensor:
        type_element = getattr(abstraction, 'type_component', abstraction)
        if hasattr(type_element, 'type_logits'):
            # Type logits are non-identifiable up to an additive constant, and
            # composed type elements are represented as log-probabilities.
            # Compare both sides in the same canonical probability space and
            # keep the contrastive similarity calculation stable under AMP.
            return F.softmax(type_element.type_logits[:, 0, :].float(), dim=-1)
        return parent_hidden[:, 0, :].float()

    def _compute_span_composition_loss(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        composition_specs: List[List[Any]],
    ) -> Tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]:
        """Compare parent abstractions with composed pooled child abstractions."""
        if len(composition_specs) != hidden_states.size(0):
            raise ValueError(
                "composition_specs batch size must match hidden_states: "
                f"{len(composition_specs)} != {hidden_states.size(0)}"
            )

        total_loss = hidden_states.new_zeros(())
        valid_count = 0
        example_losses = []
        example_counts = []
        sequence_length = hidden_states.size(1)

        for batch_index, example_specs in enumerate(composition_specs):
            example_loss = hidden_states.new_zeros(())
            example_count = 0
            valid_length = (
                int(attention_mask[batch_index].sum().item())
                if attention_mask is not None
                else sequence_length
            )
            for spec in example_specs:
                spans = (spec.left_span, spec.right_span, spec.parent_span)
                if any(start < 0 or start >= end or end > valid_length for start, end in spans):
                    raise ValueError(
                        f"Invalid composition spans {spans} for valid length {valid_length}"
                    )

                left_hidden = self._pool_span(hidden_states, batch_index, spec.left_span)
                right_hidden = self._pool_span(hidden_states, batch_index, spec.right_span)
                parent_hidden = self._pool_span(hidden_states, batch_index, spec.parent_span)

                left_abstract = self.abstract_domain.abstract(left_hidden)
                right_abstract = self.abstract_domain.abstract(right_hidden)
                parent_abstract = self.abstract_domain.abstract(parent_hidden)
                composed_abstract = self.abstract_domain.compose(
                    left_abstract, right_abstract, spec.operator
                )
                pair_loss = self._composition_pair_loss(
                    parent_abstract, composed_abstract
                )
                total_loss = total_loss + pair_loss
                example_loss = example_loss + pair_loss
                valid_count += 1
                example_count += 1

            example_losses.append(
                example_loss / example_count if example_count else example_loss
            )
            example_counts.append(float(example_count))

        if valid_count == 0:
            return (
                total_loss,
                0,
                torch.stack(example_losses),
                hidden_states.new_tensor(example_counts),
            )
        return (
            total_loss / valid_count,
            valid_count,
            torch.stack(example_losses),
            hidden_states.new_tensor(example_counts),
        )

    def _composition_pair_loss(
        self,
        parent_abstract: AbstractElement,
        composed_abstract: AbstractElement,
    ) -> torch.Tensor:
        if self.composition_objective == "domain":
            return self.abstract_domain.consistency_loss(
                parent_abstract, composed_abstract
            )

        parent_features = self._abstract_features(parent_abstract)
        composed_features = self._abstract_features(composed_abstract)
        if self.composition_objective == "mse":
            return F.mse_loss(parent_features, composed_features)
        return (1.0 - F.cosine_similarity(
            parent_features, composed_features, dim=-1
        )).mean()

    @classmethod
    def _abstract_features(cls, abstraction: AbstractElement) -> torch.Tensor:
        if hasattr(abstraction, "type_component"):
            return torch.cat(
                [
                    cls._abstract_features(abstraction.type_component),
                    cls._abstract_features(abstraction.monotonicity_component),
                ],
                dim=-1,
            )
        if hasattr(abstraction, "type_probs"):
            return abstraction.type_probs.flatten(start_dim=1)
        if hasattr(abstraction, "monotonicity_probs"):
            return abstraction.monotonicity_probs.flatten(start_dim=1)
        if hasattr(abstraction, "lower") and hasattr(abstraction, "upper"):
            return torch.cat(
                [
                    abstraction.lower.flatten(start_dim=1),
                    abstraction.upper.flatten(start_dim=1),
                ],
                dim=-1,
            )
        raise TypeError(
            f"Unsupported abstract element for {cls.__name__}: "
            f"{type(abstraction).__name__}"
        )

    @staticmethod
    def _pool_span(
        hidden_states: torch.Tensor,
        batch_index: int,
        span: Tuple[int, int],
    ) -> torch.Tensor:
        """Mean-pool one half-open token span while preserving batch/sequence axes."""
        start, end = span
        return hidden_states[batch_index : batch_index + 1, start:end].mean(
            dim=1, keepdim=True
        )
    
    def _compute_composition_loss(
        self,
        hidden_states: torch.Tensor,
        abstraction: AbstractElement,
        composition_pairs: List[Tuple[int, int, int]],
    ) -> torch.Tensor:
        """
        Compute loss for composition constraint violations.
        
        Enforces approximate equality: γ(h_k) ≈ α(γ(h_i), γ(h_j))
        
        Note: We enforce *similarity* between the abstraction of the composed
        representation and the composition of abstractions. For true lattice
        subsumption (⊑), use `subsumption_loss` if available in the domain.
        
        This encourages the model to learn representations where composing
        in hidden space corresponds to composing in abstract space.
        """
        total_loss = 0.0
        num_pairs = 0
        
        # Reuse pre-computed abstraction when possible
        has_slicing = hasattr(abstraction, 'select_positions') or hasattr(abstraction, '__getitem__')
        
        for i, j, k in composition_pairs:
            # Get abstractions - reuse if possible, else recompute
            if has_slicing:
                try:
                    a_i = abstraction.select_positions(i) if hasattr(abstraction, 'select_positions') else abstraction[:, i:i+1]
                    a_j = abstraction.select_positions(j) if hasattr(abstraction, 'select_positions') else abstraction[:, j:j+1]
                    a_k = abstraction.select_positions(k) if hasattr(abstraction, 'select_positions') else abstraction[:, k:k+1]
                except (AttributeError, TypeError):
                    has_slicing = False
            
            if not has_slicing:
                # Fallback: recompute from hidden states
                h_i = hidden_states[:, i:i+1, :]
                h_j = hidden_states[:, j:j+1, :]
                h_k = hidden_states[:, k:k+1, :]
                a_i = self.abstract_domain.abstract(h_i)
                a_j = self.abstract_domain.abstract(h_j)
                a_k = self.abstract_domain.abstract(h_k)
            
            # Compute abstract composition
            a_composed = self.abstract_domain.compose(a_i, a_j, "sequential")
            
            # Use subsumption_loss if available, else consistency_loss
            if hasattr(self.abstract_domain, 'subsumption_loss'):
                pair_loss = self.abstract_domain.subsumption_loss(a_k, a_composed)
            else:
                pair_loss = self.abstract_domain.consistency_loss(a_k, a_composed)
            
            total_loss = total_loss + pair_loss
            num_pairs += 1
        
        if num_pairs > 0:
            return total_loss / num_pairs
        return torch.tensor(0.0, device=hidden_states.device)
    
    def _compute_consistency_loss(
        self,
        hidden_states: torch.Tensor,
        abstraction: AbstractElement,
        similar_pairs: List[Tuple[int, int]],
    ) -> torch.Tensor:
        """
        Compute loss for consistency constraint violations.
        
        Enforces that semantically similar positions have similar abstractions.
        """
        total_loss = 0.0
        num_pairs = 0
        
        for i, j in similar_pairs:
            h_i = hidden_states[:, i:i+1, :]
            h_j = hidden_states[:, j:j+1, :]
            
            a_i = self.abstract_domain.abstract(h_i)
            a_j = self.abstract_domain.abstract(h_j)
            
            pair_loss = self.abstract_domain.consistency_loss(a_i, a_j)
            total_loss = total_loss + pair_loss
            num_pairs += 1
        
        if num_pairs > 0:
            return total_loss / num_pairs
        return torch.tensor(0.0, device=hidden_states.device)
    
    def _compute_entropy_regularization(
        self,
        abstraction: AbstractElement,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute entropy regularization to encourage confident abstractions.
        
        Low entropy → confident type/monotonicity assignments
        High entropy → uncertain assignments (penalized)
        
        Note: Temperature is applied here to control sharpness of the
        probability distribution before computing entropy.
        """
        if hasattr(abstraction, 'type_component'):
            if hasattr(abstraction.type_component, 'type_logits'):
                type_log_probs = F.log_softmax(
                    abstraction.type_component.type_logits.float()
                    / self.temperature,
                    dim=-1,
                )
                type_probs = type_log_probs.exp()
            else:
                type_probs = abstraction.type_component.type_probs
                type_log_probs = type_probs.float().clamp_min(
                    torch.finfo(torch.float32).tiny
                ).log()
            entropy = -(type_probs.float() * type_log_probs).sum(dim=-1)
            if attention_mask is not None:
                mask = attention_mask.to(entropy.dtype)
                return (entropy * mask).sum() / mask.sum().clamp_min(1.0)
            return entropy.mean()
        elif hasattr(abstraction, 'type_logits'):
            type_log_probs = F.log_softmax(
                abstraction.type_logits.float() / self.temperature, dim=-1
            )
            type_probs = type_log_probs.exp()
            entropy = -(type_probs * type_log_probs).sum(dim=-1)
            if attention_mask is not None:
                mask = attention_mask.to(entropy.dtype)
                return (entropy * mask).sum() / mask.sum().clamp_min(1.0)
            return entropy.mean()
        else:
            return torch.tensor(0.0)


class CompositionAwareAbstractionLoss(AbstractionLoss):
    """
    Extended abstraction loss that automatically detects compositional structure.
    
    This loss analyzes input sequences to identify likely composition relationships
    based on attention patterns or positional heuristics.
    """
    
    def __init__(
        self,
        abstract_domain: AbstractDomain,
        use_attention_for_composition: bool = True,
        attention_threshold: float = 0.3,
        **kwargs
    ):
        super().__init__(abstract_domain, **kwargs)
        self.use_attention_for_composition = use_attention_for_composition
        self.attention_threshold = attention_threshold
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        attention_weights: Optional[torch.Tensor] = None,
        **kwargs
    ) -> AbstractionLossOutput:
        """
        Compute abstraction loss with automatic composition detection.
        """
        # Detect composition pairs from attention (if provided)
        composition_pairs = None
        if self.use_attention_for_composition and attention_weights is not None:
            composition_pairs = self._detect_compositions_from_attention(
                attention_weights, attention_mask
            )
        
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            composition_pairs=composition_pairs,
            **kwargs
        )
    
    def _detect_compositions_from_attention(
        self,
        attention_weights: torch.Tensor,
        attention_mask: Optional[torch.Tensor]
    ) -> List[Tuple[int, int, int]]:
        """
        Detect likely composition relationships from attention patterns.
        
        Heuristic: If position k attends strongly to positions i and j,
        then k might be a composition of i and j.
        
        Filtering applied:
        - Respects attention_mask to ignore padding
        - Requires both top-1 and top-2 to exceed threshold
        - Limits to local window to reduce noise
        - Skips very early positions (likely special tokens)
        """
        batch_size = attention_weights.size(0)
        seq_len = attention_weights.size(-1)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Create mask for valid positions [batch, seq]
            mask_2d = attention_mask.float()
            # Expand for attention: [batch, 1, 1, seq] for broadcasting
            mask_expanded = mask_2d.unsqueeze(1).unsqueeze(2)
            # Mask out padding positions in attention
            masked_attention = attention_weights * mask_expanded
            # Average only over valid batch elements
            valid_count = mask_2d.sum(dim=0, keepdim=True).clamp(min=1)
            avg_attention = (masked_attention.sum(dim=0) / batch_size).mean(dim=0)  # [seq, seq]
        else:
            avg_attention = attention_weights.mean(dim=(0, 1))  # [seq, seq]
        
        composition_pairs = []
        local_window = 8  # Only look within this window for composition
        min_position = 2  # Skip first 2 positions (likely special tokens)
        
        for k in range(min_position, seq_len):
            # Only look at local window to reduce noise
            window_start = max(0, k - local_window)
            attention_to_prev = avg_attention[k, window_start:k]
            
            if attention_to_prev.numel() >= 2:
                values, indices = attention_to_prev.topk(min(2, attention_to_prev.numel()))
                
                # Require BOTH top-1 and top-2 to exceed threshold
                if len(indices) >= 2 and values[0] > self.attention_threshold and values[1] > self.attention_threshold * 0.5:
                    # Adjust indices back to global positions
                    i = (window_start + indices[0]).item()
                    j = (window_start + indices[1]).item()
                    
                    # Skip if either position is at boundary (likely special tokens)
                    if i >= min_position and j >= min_position:
                        composition_pairs.append((i, j, k))
        
        # Limit number of pairs to avoid excessive computation
        max_pairs = 50
        if len(composition_pairs) > max_pairs:
            # Sample uniformly
            step = len(composition_pairs) // max_pairs
            composition_pairs = composition_pairs[::step][:max_pairs]
        
        return composition_pairs


class HierarchicalAbstractionLoss(nn.Module):
    """
    Abstraction loss that enforces hierarchical structure across layers.
    
    Different layers should capture different levels of abstraction:
    - Early layers: Fine-grained type distinctions
    - Middle layers: Compositional structure
    - Later layers: High-level semantic abstraction
    
    This loss enforces that abstractions refine as we go deeper.
    """
    
    def __init__(
        self,
        abstract_domains: List[AbstractDomain],
        layer_weights: Optional[List[float]] = None,
        refinement_weight: float = 0.2,
    ):
        """
        Initialize hierarchical abstraction loss.
        
        Args:
            abstract_domains: List of domains, one per constrained layer
            layer_weights: Weight for each layer's loss
            refinement_weight: Weight for abstraction refinement constraint
        """
        super().__init__()
        
        # Store domains in regular list (may not be nn.Module subclasses)
        self.abstract_domains = list(abstract_domains)
        
        if layer_weights is None:
            layer_weights = [1.0] * len(abstract_domains)
        self.layer_weights = layer_weights
        self.refinement_weight = refinement_weight
        
        # Individual layer losses (these ARE nn.Modules)
        self.layer_losses = nn.ModuleList([
            AbstractionLoss(domain) for domain in abstract_domains
        ])
    
    def forward(
        self,
        layer_hidden_states: List[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> AbstractionLossOutput:
        """
        Compute hierarchical abstraction loss.
        
        Args:
            layer_hidden_states: List of hidden states, one per constrained layer
            attention_mask: Attention mask
            
        Returns:
            Combined loss output
        """
        total_loss = 0.0
        all_concretization_losses = []
        
        prev_abstraction = None
        
        for i, (hidden_states, weight) in enumerate(zip(layer_hidden_states, self.layer_weights)):
            # Compute layer loss
            layer_output = self.layer_losses[i](
                hidden_states=hidden_states,
                attention_mask=attention_mask,
            )
            
            total_loss = total_loss + weight * layer_output.total_loss
            all_concretization_losses.append(layer_output.concretization_loss)
            
            # Refinement constraint: later layers should refine earlier abstractions
            current_abstraction = self.abstract_domains[i].abstract(hidden_states)
            
            if prev_abstraction is not None and self.refinement_weight > 0:
                refinement_loss = self._compute_refinement_loss(
                    prev_abstraction, current_abstraction
                )
                total_loss = total_loss + self.refinement_weight * refinement_loss
            
            prev_abstraction = current_abstraction
        
        return AbstractionLossOutput(
            total_loss=total_loss,
            concretization_loss=torch.stack(all_concretization_losses).sum(),
            loss_components={
                f'layer_{i}': loss for i, loss in enumerate(all_concretization_losses)
            }
        )
    
    def _compute_refinement_loss(
        self,
        prev_abstraction: AbstractElement,
        curr_abstraction: AbstractElement,
    ) -> torch.Tensor:
        """
        Enforce that current abstraction refines previous.
        
        Refinement: The current layer's abstraction should be at least
        as specific as the previous layer's (lower entropy, more confident).
        """
        # Compare entropies
        if hasattr(prev_abstraction, 'type_component'):
            prev_probs = prev_abstraction.type_component.type_probs
            curr_probs = curr_abstraction.type_component.type_probs
        elif hasattr(prev_abstraction, 'type_logits'):
            prev_probs = F.softmax(prev_abstraction.type_logits, dim=-1)
            curr_probs = F.softmax(curr_abstraction.type_logits, dim=-1)
        else:
            return torch.tensor(0.0)
        
        prev_entropy = -(prev_probs * (prev_probs + 1e-10).log()).sum(dim=-1)
        curr_entropy = -(curr_probs * (curr_probs + 1e-10).log()).sum(dim=-1)
        
        # Current entropy should be <= previous entropy (more refined)
        refinement_violation = F.relu(curr_entropy - prev_entropy)
        
        return refinement_violation.mean()


class OverConstraintDetector(nn.Module):
    """
    Detects when abstraction constraints are too strong.
    
    Over-constraint occurs when:
    1. Task loss increases due to abstraction constraints
    2. Gradients from abstraction loss dominate task gradients
    3. Representation capacity is overly restricted
    
    This module monitors for these issues and can adjust constraint strength.
    """
    
    def __init__(
        self,
        task_loss_window: int = 100,
        gradient_ratio_threshold: float = 10.0,
        task_loss_increase_threshold: float = 0.1,
    ):
        super().__init__()
        
        self.task_loss_window = task_loss_window
        self.gradient_ratio_threshold = gradient_ratio_threshold
        self.task_loss_increase_threshold = task_loss_increase_threshold
        
        # Tracking buffers
        self.register_buffer('task_loss_history', torch.zeros(task_loss_window))
        self.register_buffer('history_pointer', torch.tensor(0))
        self.register_buffer('num_history_entries', torch.tensor(0))
        
        self._over_constraint_detected = False
    
    def update(
        self,
        task_loss: torch.Tensor,
        abstraction_loss: torch.Tensor,
        task_gradients: Optional[torch.Tensor] = None,
        abstraction_gradients: Optional[torch.Tensor] = None,
    ) -> Dict[str, bool]:
        """
        Update detector with current losses and optionally gradients.
        
        Returns:
            Dict with detection flags
        """
        # Update task loss history (circular buffer)
        ptr = self.history_pointer.item()
        self.task_loss_history[ptr] = task_loss.detach()
        new_ptr = (ptr + 1) % self.task_loss_window
        self.history_pointer.fill_(new_ptr)
        self.num_history_entries.fill_(
            min(int(self.num_history_entries.item()) + 1, self.task_loss_window)
        )
        
        detections = {}
        
        entries_filled = int(self.num_history_entries.item())
        
        # Check 1: Task loss increasing (only after buffer is sufficiently filled)
        if entries_filled >= self.task_loss_window // 2:
            # Reconstruct chronological order from circular buffer
            if entries_filled >= self.task_loss_window:
                # Buffer has wrapped - reconstruct order
                chronological = torch.cat([
                    self.task_loss_history[new_ptr:],
                    self.task_loss_history[:new_ptr]
                ])
            else:
                # Buffer hasn't wrapped yet
                chronological = self.task_loss_history[:entries_filled]
            
            half = len(chronological) // 2
            first_half = chronological[:half].mean()
            second_half = chronological[half:].mean()
            
            if second_half > first_half * (1 + self.task_loss_increase_threshold):
                detections['task_loss_increasing'] = True
                self._over_constraint_detected = True
        
        # Check 2: Gradient ratio
        if task_gradients is not None and abstraction_gradients is not None:
            task_grad_norm = task_gradients.norm()
            abs_grad_norm = abstraction_gradients.norm()
            
            if task_grad_norm > 0:
                ratio = abs_grad_norm / task_grad_norm
                if ratio > self.gradient_ratio_threshold:
                    detections['gradient_ratio_high'] = True
                    self._over_constraint_detected = True
        
        return detections
    
    @property
    def is_over_constrained(self) -> bool:
        return self._over_constraint_detected
    
    def reset(self):
        self._over_constraint_detected = False
        self.task_loss_history.zero_()
        self.history_pointer.zero_()
        self.num_history_entries.zero_()


class SubConstituentLoss(nn.Module):
    """
    Sub-Constituent Abstraction Loss from the proposal.
    
    This implements the exact formula from the research proposal:
    
        L_compose = ||γ(h_x) - α(γ(h_{x_1}), γ(h_{x_2}))||²
    
    where:
    - x is a compositional input (e.g., "jump twice and walk left")
    - x_1, x_2 are sub-constituents (e.g., "jump twice", "walk left")
    - h_x is the hidden representation of the full input
    - γ is the abstraction function (mapping hidden states to abstract elements)
    - α is the abstract composition operator (composing abstractions in the lattice)
    
    This loss enforces that the model's abstraction of a complex expression
    matches the composition of its sub-part abstractions. This is the key
    training signal for compositional generalization.
    
    Usage:
        loss_fn = SubConstituentLoss(abstract_domain, encoder, tokenizer)
        loss = loss_fn(full_input="jump twice and walk", 
                       sub_inputs=["jump twice", "walk"])
    """
    
    def __init__(
        self,
        abstract_domain: AbstractDomain,
        encoder: nn.Module,
        tokenizer,
        composition_type: str = "join",
        pool_strategy: str = "mean",
        weight: float = 1.0,
    ):
        """
        Initialize sub-constituent loss.
        
        Args:
            abstract_domain: The abstract domain (TypeMonotonicity, etc.)
            encoder: The encoder model (T5 encoder or DAI encoder)
            tokenizer: Tokenizer for encoding strings
            composition_type: How to compose abstractions ("join", "sequence", "parallel")
            pool_strategy: How to pool sequence representations ("mean", "cls", "last")
            weight: Weight for this loss component
        """
        super().__init__()
        
        self.abstract_domain = abstract_domain
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.composition_type = composition_type
        self.pool_strategy = pool_strategy
        self.weight = weight
    
    def forward(
        self,
        full_input: str,
        sub_inputs: List[str],
        attention_mask_full: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute sub-constituent composition loss for a single example.
        
        Args:
            full_input: The full compositional input string
            sub_inputs: List of sub-constituent input strings
            attention_mask_full: Optional attention mask for full input
            
        Returns:
            Composition loss: ||γ(h_x) - α(γ(h_{x_1}), γ(h_{x_2}))||²
        """
        if len(sub_inputs) < 2:
            # Need at least 2 sub-constituents to compose
            device = next(self.encoder.parameters()).device
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        device = next(self.encoder.parameters()).device
        
        # Encode full input → h_x
        h_full = self._encode_and_pool(full_input, device)
        
        # Encode sub-inputs → h_{x_i}
        h_subs = [self._encode_and_pool(sub, device) for sub in sub_inputs]
        
        # Abstract full representation: γ(h_x)
        gamma_full = self._abstract(h_full)
        
        # Abstract sub-representations: γ(h_{x_i})
        gamma_subs = [self._abstract(h) for h in h_subs]
        
        # Compose abstractions: α(γ(h_{x_1}), γ(h_{x_2}), ...)
        composed_abstraction = self._compose_abstractions(gamma_subs)
        
        # Compute loss: ||γ(h_x) - α(γ(h_{x_1}), γ(h_{x_2}))||²
        loss = self._abstraction_distance(gamma_full, composed_abstraction)
        
        return self.weight * loss
    
    def forward_batch(
        self,
        full_inputs: List[str],
        sub_inputs_list: List[List[str]],
    ) -> torch.Tensor:
        """
        Compute sub-constituent loss for a batch of examples.
        
        This batched version tokenizes and encodes all strings at once for
        significant speedup (5-20x) compared to per-example processing.
        
        Args:
            full_inputs: List of full compositional input strings
            sub_inputs_list: List of lists of sub-constituent strings
            
        Returns:
            Mean composition loss over the batch
        """
        device = next(self.encoder.parameters()).device
        
        # Filter to valid examples (at least 2 sub-constituents)
        valid_indices = [i for i, subs in enumerate(sub_inputs_list) if len(subs) >= 2]
        if len(valid_indices) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Collect all strings to encode in one batch
        all_strings = []
        string_mapping = []  # (example_idx, 'full' | sub_idx)
        
        for i in valid_indices:
            # Add full input
            all_strings.append(full_inputs[i])
            string_mapping.append((i, 'full'))
            # Add sub-inputs
            for j, sub in enumerate(sub_inputs_list[i]):
                all_strings.append(sub)
                string_mapping.append((i, j))
        
        # Batch tokenize all strings at once
        inputs = self.tokenizer(
            all_strings,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Batch encode all strings at once
        with torch.set_grad_enabled(self.training):
            outputs = self.encoder(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                return_dict=True,
            )
        
        hidden = outputs.last_hidden_state  # [total_strings, seq_len, hidden_dim]
        
        # Pool all representations
        if self.pool_strategy == "mean":
            mask = inputs["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        elif self.pool_strategy == "cls":
            pooled = hidden[:, 0, :]
        elif self.pool_strategy == "last":
            lengths = inputs["attention_mask"].sum(dim=1) - 1
            pooled = hidden[torch.arange(hidden.size(0), device=device), lengths, :]
        else:
            pooled = hidden.mean(dim=1)
        
        # Regroup pooled vectors by example
        example_vectors = {i: {'full': None, 'subs': []} for i in valid_indices}
        for vec_idx, (ex_idx, marker) in enumerate(string_mapping):
            vec = pooled[vec_idx:vec_idx+1]  # Keep batch dim [1, hidden]
            if marker == 'full':
                example_vectors[ex_idx]['full'] = vec
            else:
                example_vectors[ex_idx]['subs'].append(vec)
        
        # Compute loss for each example
        losses = []
        for ex_idx in valid_indices:
            h_full = example_vectors[ex_idx]['full']
            h_subs = example_vectors[ex_idx]['subs']
            
            if h_full is None or len(h_subs) < 2:
                continue
            
            # Abstract full representation: γ(h_x)
            gamma_full = self._abstract(h_full)
            
            # Abstract sub-representations: γ(h_{x_i})
            gamma_subs = [self._abstract(h) for h in h_subs]
            
            # Compose abstractions: α(γ(h_{x_1}), γ(h_{x_2}), ...)
            composed_abstraction = self._compose_abstractions(gamma_subs)
            
            # Compute loss: ||γ(h_x) - α(γ(h_{x_1}), γ(h_{x_2}))||²
            loss = self._abstraction_distance(gamma_full, composed_abstraction)
            losses.append(loss)
        
        if len(losses) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        return self.weight * torch.stack(losses).mean()
    
    def _encode_and_pool(self, text: str, device: torch.device) -> torch.Tensor:
        """Encode text and pool to single vector."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.set_grad_enabled(self.training):
            outputs = self.encoder(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                return_dict=True,
            )
        
        hidden = outputs.last_hidden_state  # [1, seq_len, hidden_dim]
        
        # Pool to single vector
        if self.pool_strategy == "mean":
            mask = inputs["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        elif self.pool_strategy == "cls":
            pooled = hidden[:, 0, :]
        elif self.pool_strategy == "last":
            # Get last non-padded token
            lengths = inputs["attention_mask"].sum(dim=1) - 1
            pooled = hidden[torch.arange(hidden.size(0)), lengths, :]
        else:
            pooled = hidden.mean(dim=1)
        
        return pooled  # [1, hidden_dim]
    
    def _abstract(self, hidden: torch.Tensor) -> AbstractElement:
        """
        Apply abstraction function γ to hidden representation.
        
        Args:
            hidden: Hidden representation [1, hidden_dim]
            
        Returns:
            AbstractElement representing the abstraction
        """
        # Expand to [batch, seq=1, hidden] for abstract domain
        hidden_expanded = hidden.unsqueeze(1)
        return self.abstract_domain.abstract(hidden_expanded)
    
    def _compose_abstractions(
        self,
        abstractions: List[AbstractElement],
    ) -> AbstractElement:
        """
        Compose multiple abstractions using abstract operator α.
        
        For the DAI framework, this uses the abstract domain's join or 
        composition operation. The composition follows the lattice structure.
        
        Args:
            abstractions: List of AbstractElements to compose
            
        Returns:
            Composed AbstractElement
        """
        if len(abstractions) == 0:
            raise ValueError("No abstractions to compose")
        
        if len(abstractions) == 1:
            return abstractions[0]
        
        # Use the abstract domain's compose method if available
        result = abstractions[0]
        
        for next_abs in abstractions[1:]:
            if hasattr(self.abstract_domain, 'compose'):
                result = self.abstract_domain.compose(result, next_abs, self.composition_type)
            else:
                # Fallback: element-wise join (maximum in lattice)
                result = self._fallback_compose(result, next_abs)
        
        return result
    
    def _fallback_compose(
        self,
        abs1: AbstractElement,
        abs2: AbstractElement,
    ) -> AbstractElement:
        """
        Fallback composition when domain doesn't have compose method.
        
        Uses element-wise maximum on type logits (approximates join).
        """
        # Handle TypeMonotonicity elements
        if hasattr(abs1, 'type_component') and hasattr(abs2, 'type_component'):
            # Compose type components
            composed_type_logits = torch.maximum(
                abs1.type_component.type_logits,
                abs2.type_component.type_logits,
            )
            
            # Compose monotonicity if available
            composed_mono_logits = None
            if hasattr(abs1, 'monotonicity_component') and abs1.monotonicity_component is not None:
                composed_mono_logits = torch.maximum(
                    abs1.monotonicity_component.monotonicity_logits,
                    abs2.monotonicity_component.monotonicity_logits,
                )
            
            # Create composed element (simplified - reuse type component structure)
            from src.models.abstract_domains import TypeElement
            composed_type = TypeElement(
                type_logits=composed_type_logits,
                confidence=torch.minimum(abs1.type_component.confidence, abs2.type_component.confidence),
            )
            
            # Return as TypeMonotonicity element
            from src.models.abstract_domains import TypeMonotonicityElement, MonotonicityElement
            mono_comp = None
            if composed_mono_logits is not None:
                mono_comp = MonotonicityElement(
                    monotonicity_logits=composed_mono_logits,
                    confidence=torch.minimum(
                        abs1.monotonicity_component.confidence,
                        abs2.monotonicity_component.confidence,
                    ),
                )
            
            return TypeMonotonicityElement(
                type_component=composed_type,
                monotonicity_component=mono_comp,
            )
        
        # Handle simple type elements
        elif hasattr(abs1, 'type_logits'):
            composed_logits = torch.maximum(abs1.type_logits, abs2.type_logits)
            from src.models.abstract_domains import TypeElement
            return TypeElement(
                type_logits=composed_logits,
                confidence=torch.minimum(abs1.confidence, abs2.confidence),
            )
        
        else:
            raise ValueError(f"Unknown abstract element type: {type(abs1)}")
    
    def _abstraction_distance(
        self,
        abs1: AbstractElement,
        abs2: AbstractElement,
        use_kl: bool = True,
    ) -> torch.Tensor:
        """
        Compute distance between two abstractions.
        
        This is the ||γ(h_x) - α(γ(h_{x_1}), γ(h_{x_2}))||² term, but we use
        symmetric KL divergence for probability distributions (more principled
        than MSE on logits which is sensitive to arbitrary scaling).
        
        Args:
            abs1: First abstraction (γ(h_x))
            abs2: Second abstraction (α(γ(h_{x_1}), γ(h_{x_2})))
            use_kl: If True, use symmetric KL on probabilities; else MSE on logits
            
        Returns:
            Distance between abstractions
        """
        if use_kl and self._has_type_logits(abs1) and self._has_type_logits(abs2):
            # Use symmetric KL divergence on probability distributions
            logits1 = self._get_type_logits(abs1)
            logits2 = self._get_type_logits(abs2)
            
            # Convert to probabilities
            probs1 = F.softmax(logits1, dim=-1)
            probs2 = F.softmax(logits2, dim=-1)
            
            # Symmetric KL: KL(p||q) + KL(q||p)
            # Using log_softmax for numerical stability
            log_probs1 = F.log_softmax(logits1, dim=-1)
            log_probs2 = F.log_softmax(logits2, dim=-1)
            
            kl_pq = (probs1 * (log_probs1 - log_probs2)).sum(dim=-1)
            kl_qp = (probs2 * (log_probs2 - log_probs1)).sum(dim=-1)
            
            return (kl_pq + kl_qp).mean() / 2  # Average symmetric KL
        else:
            # Fallback to MSE for non-probabilistic representations
            rep1 = self._get_abstraction_representation(abs1)
            rep2 = self._get_abstraction_representation(abs2)
            return F.mse_loss(rep1, rep2)
    
    def _has_type_logits(self, abstract_element: AbstractElement) -> bool:
        """Check if abstract element has type logits."""
        if hasattr(abstract_element, 'type_component'):
            return hasattr(abstract_element.type_component, 'type_logits')
        return hasattr(abstract_element, 'type_logits')
    
    def _get_type_logits(self, abstract_element: AbstractElement) -> torch.Tensor:
        """Extract type logits from abstract element."""
        if hasattr(abstract_element, 'type_component'):
            return abstract_element.type_component.type_logits.flatten()
        return abstract_element.type_logits.flatten()
    
    def _get_abstraction_representation(
        self,
        abstract_element: AbstractElement,
    ) -> torch.Tensor:
        """Extract tensor representation from abstract element for comparison."""
        if hasattr(abstract_element, 'type_component'):
            # TypeMonotonicity element - use type logits
            rep = abstract_element.type_component.type_logits.flatten()
            
            # Optionally concatenate monotonicity
            if hasattr(abstract_element, 'monotonicity_component') and abstract_element.monotonicity_component is not None:
                mono = abstract_element.monotonicity_component.monotonicity_logits.flatten()
                rep = torch.cat([rep, mono], dim=-1)
            
            return rep
        
        elif hasattr(abstract_element, 'type_logits'):
            return abstract_element.type_logits.flatten()
        
        elif hasattr(abstract_element, 'lower') and hasattr(abstract_element, 'upper'):
            # Interval domain - concatenate lower and upper bounds
            return torch.cat([
                abstract_element.lower.flatten(),
                abstract_element.upper.flatten(),
            ], dim=-1)
        
        elif hasattr(abstract_element, 'bounds'):
            # Alternative interval representation with bounds tuple/tensor
            bounds = abstract_element.bounds
            if isinstance(bounds, (tuple, list)) and len(bounds) == 2:
                return torch.cat([bounds[0].flatten(), bounds[1].flatten()], dim=-1)
            else:
                return bounds.flatten()
        
        else:
            raise ValueError(f"Cannot extract representation from {type(abstract_element)}")
