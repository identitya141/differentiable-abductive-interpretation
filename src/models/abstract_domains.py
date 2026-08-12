"""
Abstract Domains for Differentiable Abstract Interpretation

This module defines differentiable abstract-domain-inspired representations.
Each domain specifies:
- Abstract elements and their structure
- Abstraction function α: Concrete → Abstract
- A representation-consistency penalty
- Learned abstract composition operations

Reference: Cousot & Cousot, "Abstract Interpretation: A Unified Lattice Model" (1977)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class AbstractDomainType(Enum):
    """Enumeration of supported abstract domains."""
    TYPE_DOMAIN = "type"
    INTERVAL_DOMAIN = "interval"
    MONOTONICITY_DOMAIN = "monotonicity"
    TYPE_MONOTONICITY_DOMAIN = "type_monotonicity"  # Combined domain (recommended)
    RELATIONAL_DOMAIN = "relational"


@dataclass
class AbstractElement:
    """Base class for abstract domain elements."""
    pass


@dataclass
class TypeElement(AbstractElement):
    """
    Type domain element representing semantic categories.
    
    In compositional tasks, types might be:
    - SCAN: {action, direction, modifier, quantifier}
    - COGS: {noun, verb, preposition, determiner, proper_noun}
    - CFQ: {entity, relation, variable, literal}
    - GSM8K: {quantity, operation, intermediate, final}
    """
    type_logits: torch.Tensor  # Shape: [batch, seq, num_types]
    
    @property
    def type_probs(self) -> torch.Tensor:
        return F.softmax(self.type_logits, dim=-1)
    
    @property
    def hard_types(self) -> torch.Tensor:
        return self.type_logits.argmax(dim=-1)


@dataclass
class IntervalElement(AbstractElement):
    """
    Interval domain element for numerical reasoning.
    
    Represents bounds [lower, upper] for each dimension of the representation.
    Used for arithmetic tasks like GSM8K.
    """
    lower: torch.Tensor  # Shape: [batch, seq, dim]
    upper: torch.Tensor  # Shape: [batch, seq, dim]
    
    def contains(self, x: torch.Tensor) -> torch.Tensor:
        """Check if concrete values fall within intervals."""
        return ((x >= self.lower) & (x <= self.upper)).all(dim=-1)
    
    @property
    def width(self) -> torch.Tensor:
        """Interval width (measure of abstraction precision)."""
        return self.upper - self.lower


@dataclass  
class MonotonicityElement(AbstractElement):
    """
    Monotonicity domain for capturing order-preserving relationships.
    
    For each representation dimension, tracks whether the mapping
    from input to that dimension is monotonically increasing (+1),
    decreasing (-1), or non-monotonic (0).
    """
    monotonicity_logits: torch.Tensor  # Shape: [batch, seq, dim, 3] for {dec, non, inc}
    
    @property
    def monotonicity_probs(self) -> torch.Tensor:
        return F.softmax(self.monotonicity_logits, dim=-1)
    
    @property
    def hard_monotonicity(self) -> torch.Tensor:
        """Returns -1, 0, or 1 for each dimension."""
        return self.monotonicity_logits.argmax(dim=-1) - 1


@dataclass
class TypeMonotonicityElement(AbstractElement):
    """
    Combined Type-Monotonicity domain (RECOMMENDED).
    
    This is our primary abstract domain, combining:
    1. Type information: semantic category of each token/position
    2. Monotonicity: order-preservation properties for arithmetic
    
    The combination is crucial because:
    - Types provide compositional structure (what can compose with what)
    - Monotonicity provides arithmetic invariants (order preservation)
    """
    type_component: TypeElement
    monotonicity_component: MonotonicityElement
    
    # Coupling tensor: how type affects monotonicity constraints
    # Shape: [num_types, num_monotonicity_classes]
    coupling_logits: Optional[torch.Tensor] = None


class AbstractDomain(nn.Module, ABC):
    """
    Base class for differentiable abstract domains.
    
    Each domain must implement:
    - abstract(): Convert concrete representations to abstract elements
    - concretize_loss(): Differentiable penalty for concrete violating abstract
    - compose(): How abstractions compose under neural operations
    """
    
    def __init__(self, hidden_dim: int, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
    
    @abstractmethod
    def abstract(self, h: torch.Tensor) -> AbstractElement:
        """
        Abstraction function α: H → A
        
        Maps concrete hidden representations to abstract domain elements.
        Must be differentiable for end-to-end training.
        
        Args:
            h: Hidden representations [batch, seq, hidden_dim]
            
        Returns:
            Abstract domain element
        """
        pass
    
    @abstractmethod
    def concretize_loss(
        self, 
        h: torch.Tensor, 
        a: AbstractElement,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute violation of abstract constraints.
        
        This is NOT the concretization function γ (which returns a set),
        but a differentiable loss measuring how much h violates a.
        
        Args:
            h: Concrete hidden representations
            a: Abstract element (constraint)
            
        Returns:
            Scalar loss tensor
        """
        pass
    
    @abstractmethod
    def compose(
        self,
        a1: AbstractElement,
        a2: AbstractElement,
        composition_type: str
    ) -> AbstractElement:
        """
        Abstract composition operator.
        
        Computes the abstraction of composing two abstract elements.
        Implementations are learned differentiable transfer functions; this
        interface does not require or imply sound over-approximation.
        
        Args:
            a1, a2: Abstract elements to compose
            composition_type: How elements are composed (e.g., "concat", "add")
            
        Returns:
            Abstract element representing composed abstraction
        """
        pass
    
    @abstractmethod
    def consistency_loss(
        self,
        a1: AbstractElement,
        a2: AbstractElement,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Measure consistency between two abstractions.
        
        Used to enforce that similar inputs have compatible abstractions.
        """
        pass


class TypeDomain(AbstractDomain):
    """
    Type abstract domain for semantic categories.
    
    This domain captures the semantic type of each position in the sequence,
    enabling type-based composition rules.
    
    Type System:
    - Each position is assigned a type from a fixed vocabulary
    - Types determine valid compositions (type checking)
    - Type errors indicate compositional violations
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_types: int = 16,  # Number of type categories
        type_embed_dim: int = 64,
        composition_rules_trainable: bool = True,
        operator_specific_composition: bool = True,
        **kwargs
    ):
        super().__init__(hidden_dim, **kwargs)
        self.num_types = num_types
        self.type_embed_dim = type_embed_dim
        self.operator_specific_composition = operator_specific_composition
        
        # Abstraction network: h → type logits
        self.abstraction_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, num_types)
        )
        
        # Type embeddings for reconstruction
        self.type_embeddings = nn.Embedding(num_types, type_embed_dim)
        
        # Reconstruction network: type_embed → hidden subspace
        self.reconstruction_net = nn.Linear(type_embed_dim, hidden_dim)
        
        # Type composition rules (learnable)
        # Shape: [num_types, num_types, num_types] 
        # Entry [i,j,k] = probability that types i,j compose to type k
        self.composition_rules = nn.Parameter(
            torch.zeros(num_types, num_types, num_types)
        )
        nn.init.xavier_uniform_(self.composition_rules)
        self.operator_composition_rules = nn.ParameterDict()
        for operator in (
            "direction",
            "opposite",
            "around",
            "twice",
            "thrice",
            "and",
            "after",
            "agent",
            "theme",
            "recipient",
            "ccomp",
            "xcomp",
            "nmod",
            "relation",
            "join",
        ):
            rules = nn.Parameter(torch.zeros(num_types, num_types, num_types))
            nn.init.xavier_uniform_(rules)
            self.operator_composition_rules[operator] = rules
        if not composition_rules_trainable:
            self.composition_rules.requires_grad_(False)
            self.operator_composition_rules.requires_grad_(False)
    
    def abstract(self, h: torch.Tensor) -> TypeElement:
        """Map hidden states to type distributions."""
        type_logits = self.abstraction_net(h)
        return TypeElement(type_logits=type_logits)
    
    def concretize_loss(
        self, 
        h: torch.Tensor, 
        a: TypeElement,
        temperature: float = 1.0,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute type consistency loss.
        
        Penalizes representations that don't align with their assigned types.
        Uses reconstruction through type embedding as proxy for type consistency.
        """
        with torch.autocast(device_type=h.device.type, enabled=False):
            type_log_probs = F.log_softmax(
                a.type_logits.float() / temperature, dim=-1
            )
            type_probs = type_log_probs.exp()
            type_embed = torch.einsum(
                'bst,te->bse', type_probs, self.type_embeddings.weight.float()
            )
            h_reconstructed = self.reconstruction_net(type_embed)
            reconstruction_error = (
                h_reconstructed - h.detach().float()
            ).pow(2).mean(dim=-1)
            if attention_mask is not None:
                mask = attention_mask.to(reconstruction_error.dtype)
                return (reconstruction_error * mask).sum() / mask.sum().clamp_min(1.0)
            return reconstruction_error.mean()
    
    def compose(
        self,
        a1: TypeElement,
        a2: TypeElement,
        composition_type: str = "sequential"
    ) -> TypeElement:
        """
        Compute abstract composition of types.
        
        Uses learned composition rules to determine output type distribution.
        """
        rules = (
            self.operator_composition_rules[composition_type]
            if (
                self.operator_specific_composition
                and composition_type in self.operator_composition_rules
            )
            else self.composition_rules
        )
        if composition_type == "sequential" or composition_type in self.operator_composition_rules:
            with torch.autocast(
                device_type=a1.type_logits.device.type, enabled=False
            ):
                p1 = F.softmax(a1.type_logits.float(), dim=-1)
                p2 = F.softmax(a2.type_logits.float(), dim=-1)
                composition_probs = F.softmax(rules.float(), dim=-1)
                joint = torch.einsum(
                    'bsi,bsj->bsij', p1[:, -1:, :], p2[:, :1, :]
                )
                result_probs = torch.einsum(
                    'bsij,ijk->bsk', joint, composition_probs
                )
                result_logits = result_probs.clamp_min(
                    torch.finfo(result_probs.dtype).tiny
                ).log()
            return TypeElement(type_logits=result_logits)
        else:
            raise NotImplementedError(f"Composition type {composition_type} not implemented")
    
    def consistency_loss(
        self,
        a1: TypeElement,
        a2: TypeElement,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Enforce that similar positions have similar type distributions.
        
        Uses KL divergence between type distributions.
        """
        log_p1 = F.log_softmax(a1.type_logits.float(), dim=-1)
        log_p2 = F.log_softmax(a2.type_logits.float(), dim=-1)
        kl_forward = F.kl_div(
            log_p1, log_p2, reduction='none', log_target=True
        ).sum(dim=-1)
        kl_backward = F.kl_div(
            log_p2, log_p1, reduction='none', log_target=True
        ).sum(dim=-1)

        symmetric = 0.5 * (kl_forward + kl_backward)
        if attention_mask is not None and symmetric.shape == attention_mask.shape:
            mask = attention_mask.to(symmetric.dtype)
            return (symmetric * mask).sum() / mask.sum().clamp_min(1.0)
        return symmetric.mean()
    
    def type_checking_loss(
        self,
        a_input: TypeElement,
        a_output: TypeElement,
        valid_mappings: torch.Tensor
    ) -> torch.Tensor:
        """
        Enforce valid input→output type mappings.
        
        Args:
            a_input: Type abstraction of input
            a_output: Type abstraction of output
            valid_mappings: [num_types, num_types] binary matrix of valid mappings
            
        Returns:
            Loss penalizing invalid type transitions
        """
        p_in = F.softmax(a_input.type_logits, dim=-1)
        p_out = F.softmax(a_output.type_logits, dim=-1)
        
        # Compute expected validity
        # [batch, seq, num_types] x [batch, seq, num_types] → [batch, seq, num_types, num_types]
        joint = torch.einsum('bsi,bso->bsio', p_in, p_out)
        
        # Mask by valid mappings
        validity = (joint * valid_mappings.unsqueeze(0).unsqueeze(0)).sum(dim=(-1, -2))
        
        # Loss: minimize probability of invalid mappings
        return -validity.mean()


class IntervalDomain(AbstractDomain):
    """
    Interval abstract domain for numerical bounds.
    
    Particularly useful for arithmetic tasks (GSM8K, CLUTRR counting).
    Predicts learned lower and upper bounds on projected hidden states.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        interval_dim: int = 32,  # Reduced dimensionality for interval tracking
        precision_weight: float = 0.01,
        **kwargs
    ):
        super().__init__(hidden_dim, **kwargs)
        self.interval_dim = interval_dim
        self.precision_weight = precision_weight
        
        # Project to interval space
        self.projection = nn.Linear(hidden_dim, interval_dim)
        
        # Predict interval bounds
        self.lower_net = nn.Sequential(
            nn.Linear(interval_dim, interval_dim),
            nn.Tanh(),
            nn.Linear(interval_dim, interval_dim)
        )
        self.upper_net = nn.Sequential(
            nn.Linear(interval_dim, interval_dim),
            nn.Tanh(),
            nn.Linear(interval_dim, interval_dim)
        )
    
    def abstract(self, h: torch.Tensor) -> IntervalElement:
        """Compute interval bounds from hidden states."""
        projected = self.projection(h)
        lower_raw = self.lower_net(projected)
        upper_raw = self.upper_net(projected)
        
        return IntervalElement(
            lower=torch.minimum(lower_raw, upper_raw),
            upper=torch.maximum(lower_raw, upper_raw),
        )
    
    def concretize_loss(
        self,
        h: torch.Tensor,
        a: IntervalElement,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Penalize hidden states outside their intervals.
        """
        projected = self.projection(h)
        
        # Hinge loss for lower bound violations
        lower_violation = F.relu(a.lower - projected)
        
        # Hinge loss for upper bound violations  
        upper_violation = F.relu(projected - a.upper)
        
        # Total violation
        # Prevent the trivial solution of expanding intervals indefinitely.
        violation = (
            lower_violation + upper_violation
            + self.precision_weight * (a.upper - a.lower)
        )
        
        violation = violation.mean(dim=-1)
        if attention_mask is not None:
            mask = attention_mask.to(violation.dtype)
            return (violation * mask).sum() / mask.sum().clamp_min(1.0)
        return violation.mean()
    
    def compose(
        self,
        a1: IntervalElement,
        a2: IntervalElement,
        composition_type: str = "add"
    ) -> IntervalElement:
        """
        Interval arithmetic for composition.
        """
        if composition_type == "add":
            return IntervalElement(
                lower=a1.lower + a2.lower,
                upper=a1.upper + a2.upper
            )
        elif composition_type == "multiply":
            # Conservative bounds for multiplication
            products = torch.stack([
                a1.lower * a2.lower,
                a1.lower * a2.upper,
                a1.upper * a2.lower,
                a1.upper * a2.upper
            ], dim=0)
            return IntervalElement(
                lower=products.min(dim=0).values,
                upper=products.max(dim=0).values
            )
        else:
            raise NotImplementedError
    
    def consistency_loss(
        self,
        a1: IntervalElement,
        a2: IntervalElement,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Measure interval overlap as consistency.
        """
        # Compute intersection
        lower = torch.maximum(a1.lower, a2.lower)
        upper = torch.minimum(a1.upper, a2.upper)
        
        # Penalize non-overlapping intervals
        non_overlap = F.relu(lower - upper)
        
        non_overlap = non_overlap.mean(dim=-1)
        if attention_mask is not None and non_overlap.shape == attention_mask.shape:
            mask = attention_mask.to(non_overlap.dtype)
            return (non_overlap * mask).sum() / mask.sum().clamp_min(1.0)
        return non_overlap.mean()


class MonotonicityDomain(AbstractDomain):
    """
    Monotonicity abstract domain.
    
    Tracks whether each representation dimension preserves, reverses,
    or does not preserve ordering from input to output.
    
    Critical for arithmetic reasoning where order matters.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        track_dim: int = 64,
        **kwargs
    ):
        super().__init__(hidden_dim, **kwargs)
        self.track_dim = track_dim
        
        # Project to tracking space
        self.projection = nn.Linear(hidden_dim, track_dim)
        
        # Classify monotonicity: 3 classes {decreasing, non-monotonic, increasing}
        self.monotonicity_classifier = nn.Sequential(
            nn.Linear(track_dim, track_dim),
            nn.GELU(),
            nn.Linear(track_dim, 3)  # Per dimension
        )
    
    def abstract(self, h: torch.Tensor) -> MonotonicityElement:
        """Classify monotonicity for each dimension."""
        projected = self.projection(h)  # [batch, seq, track_dim]
        
        # Compute finite differences (for sequential data)
        if projected.size(1) > 1:
            diffs = projected[:, 1:, :] - projected[:, :-1, :]
            # Classify based on sign of differences
            mono_logits = self.monotonicity_classifier(diffs)
        else:
            mono_logits = self.monotonicity_classifier(projected)
        
        return MonotonicityElement(monotonicity_logits=mono_logits)
    
    def concretize_loss(
        self,
        h: torch.Tensor,
        a: MonotonicityElement,
        input_order: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Penalize monotonicity violations.
        
        If input_order specifies ordering, penalize when representation
        doesn't preserve expected monotonicity.
        """
        projected = self.projection(h)
        mono_probs = F.softmax(a.monotonicity_logits, dim=-1)
        
        # Expected monotonicity: -1 (dec), 0 (non), 1 (inc)
        expected_mono = mono_probs @ torch.tensor([-1., 0., 1.], device=h.device)
        
        # Actual differences
        if projected.size(1) > 1:
            actual_diffs = projected[:, 1:, :] - projected[:, :-1, :]
            # The classifier emits one three-way label per transition, so the
            # supervision target must also summarize each transition.
            actual_sign = torch.sign(actual_diffs.detach().mean(dim=-1))
            
            # Penalize when expected monotonicity doesn't match actual
            violation = (expected_mono - actual_sign).pow(2)
            if attention_mask is not None:
                transition_mask = (
                    attention_mask[:, 1:] * attention_mask[:, :-1]
                ).to(violation.dtype)
                return (violation * transition_mask).sum() / transition_mask.sum().clamp_min(1.0)
            return violation.mean()
        
        return torch.tensor(0.0, device=h.device)
    
    def compose(
        self,
        a1: MonotonicityElement,
        a2: MonotonicityElement,
        composition_type: str = "chain"
    ) -> MonotonicityElement:
        """
        Compose monotonicity properties.
        
        For chain composition: mono(f∘g) = mono(f) * mono(g)
        """
        with torch.autocast(
            device_type=a1.monotonicity_logits.device.type, enabled=False
        ):
            p1 = F.softmax(a1.monotonicity_logits.float(), dim=-1)
            p2 = F.softmax(a2.monotonicity_logits.float(), dim=-1)

            if composition_type == "chain":
                transition = p1.new_zeros((3, 3, 3))
                monotonicity_values = (-1, 0, 1)
                for left_index, left_value in enumerate(monotonicity_values):
                    for right_index, right_value in enumerate(monotonicity_values):
                        result_index = (left_value * right_value) + 1
                        transition[left_index, right_index, result_index] = 1.0
                composed_probs = torch.einsum(
                    'bsi,bsj,ijk->bsk', p1, p2, transition
                )
            else:
                composed_probs = 0.5 * (p1 + p2)
            composed_logits = composed_probs.clamp_min(
                torch.finfo(composed_probs.dtype).tiny
            ).log()

        return MonotonicityElement(
            monotonicity_logits=composed_logits
        )
    
    def consistency_loss(
        self,
        a1: MonotonicityElement,
        a2: MonotonicityElement,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Monotonicity should be consistent for same operations.
        """
        p1 = F.softmax(a1.monotonicity_logits.float(), dim=-1)
        p2 = F.softmax(a2.monotonicity_logits.float(), dim=-1)

        error = (p1 - p2).pow(2).mean(dim=-1)
        if attention_mask is not None:
            mask = attention_mask
            if mask.size(1) == error.size(1) + 1:
                mask = mask[:, 1:] * mask[:, :-1]
            if mask.shape == error.shape:
                mask = mask.to(error.dtype)
                return (error * mask).sum() / mask.sum().clamp_min(1.0)
        return error.mean()


class TypeMonotonicityDomain(AbstractDomain):
    """
    Combined Type-Monotonicity Domain (RECOMMENDED PRIMARY DOMAIN)
    
    This is our main abstract domain that combines:
    1. Type abstraction for compositional structure
    2. Monotonicity abstraction for arithmetic invariants
    3. Coupling between types and monotonicity constraints
    
    Why this combination:
    - Types alone don't capture numerical properties
    - Monotonicity alone doesn't capture compositional categories
    - Together they provide comprehensive compositional structure
    
    This is a learned product representation. It does not by itself establish
    a Galois connection or classical abstract-interpretation soundness.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_types: int = 16,
        type_embed_dim: int = 64,
        monotonicity_dim: int = 64,
        coupling_strength: float = 0.5,
        composition_rules_trainable: bool = True,
        operator_specific_composition: bool = True,
        **kwargs
    ):
        super().__init__(hidden_dim, **kwargs)
        
        self.num_types = num_types
        self.type_embed_dim = type_embed_dim
        self.monotonicity_dim = monotonicity_dim
        self.coupling_strength = coupling_strength
        
        # Component domains
        self.type_domain = TypeDomain(
            hidden_dim=hidden_dim,
            num_types=num_types,
            type_embed_dim=type_embed_dim,
            composition_rules_trainable=composition_rules_trainable,
            operator_specific_composition=operator_specific_composition,
        )
        self.monotonicity_domain = MonotonicityDomain(
            hidden_dim=hidden_dim,
            track_dim=monotonicity_dim
        )
        
        # Type-monotonicity coupling
        # Which monotonicity patterns are valid for each type
        self.type_mono_coupling = nn.Parameter(
            torch.zeros(num_types, 3)  # 3 monotonicity classes
        )
        nn.init.normal_(self.type_mono_coupling, std=0.1)
    
    def abstract(self, h: torch.Tensor) -> TypeMonotonicityElement:
        """
        Joint abstraction to type-monotonicity domain.
        """
        type_elem = self.type_domain.abstract(h)
        mono_elem = self.monotonicity_domain.abstract(h)
        
        # Compute coupling logits
        type_probs = type_elem.type_probs
        if type_probs.size(1) > 1:
            type_probs = 0.5 * (type_probs[:, 1:] + type_probs[:, :-1])
        coupling = torch.einsum('bst,tm->bsm', type_probs, self.type_mono_coupling)
        
        return TypeMonotonicityElement(
            type_component=type_elem,
            monotonicity_component=mono_elem,
            coupling_logits=coupling
        )
    
    def concretize_loss(
        self,
        h: torch.Tensor,
        a: TypeMonotonicityElement,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Combined concretization loss from both domains plus coupling.
        """
        # Individual domain losses
        type_loss = self.type_domain.concretize_loss(
            h, a.type_component, attention_mask=attention_mask
        )
        mono_loss = self.monotonicity_domain.concretize_loss(
            h, a.monotonicity_component, attention_mask=attention_mask
        )
        
        # Coupling loss: monotonicity should match type expectations
        if a.coupling_logits is not None:
            expected_log_mono = F.log_softmax(
                a.coupling_logits.float(), dim=-1
            )
            actual_log_mono = F.log_softmax(
                a.monotonicity_component.monotonicity_logits.float(), dim=-1
            )
            
            # Align monotonicity with type-based expectations
            if expected_log_mono.shape == actual_log_mono.shape:
                coupling_per_transition = F.kl_div(
                    actual_log_mono,
                    expected_log_mono,
                    reduction='none',
                    log_target=True,
                ).sum(dim=-1)
                if attention_mask is not None and attention_mask.size(1) > 1:
                    transition_mask = (
                        attention_mask[:, 1:] * attention_mask[:, :-1]
                    ).to(coupling_per_transition.dtype)
                    coupling_loss = (
                        coupling_per_transition * transition_mask
                    ).sum() / transition_mask.sum().clamp_min(1.0)
                else:
                    coupling_loss = coupling_per_transition.mean()
            else:
                coupling_loss = torch.tensor(0.0, device=h.device)
        else:
            coupling_loss = torch.tensor(0.0, device=h.device)
        
        return type_loss + mono_loss + self.coupling_strength * coupling_loss
    
    def compose(
        self,
        a1: TypeMonotonicityElement,
        a2: TypeMonotonicityElement,
        composition_type: str = "sequential"
    ) -> TypeMonotonicityElement:
        """
        Compose both type and monotonicity abstractions.
        """
        type_composed = self.type_domain.compose(
            a1.type_component, a2.type_component, composition_type
        )
        monotonicity_composition = (
            "parallel" if composition_type in {"and", "after"} else "chain"
        )
        mono_composed = self.monotonicity_domain.compose(
            a1.monotonicity_component,
            a2.monotonicity_component,
            monotonicity_composition,
        )
        
        return TypeMonotonicityElement(
            type_component=type_composed,
            monotonicity_component=mono_composed,
            coupling_logits=None
        )
    
    def consistency_loss(
        self,
        a1: TypeMonotonicityElement,
        a2: TypeMonotonicityElement,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Combined consistency from both domains.
        """
        type_consist = self.type_domain.consistency_loss(
            a1.type_component, a2.type_component, attention_mask=attention_mask
        )
        mono_consist = self.monotonicity_domain.consistency_loss(
            a1.monotonicity_component,
            a2.monotonicity_component,
            attention_mask=attention_mask,
        )
        
        return type_consist + mono_consist


def get_abstract_domain(
    domain_type: Union[str, AbstractDomainType],
    hidden_dim: int,
    **kwargs
) -> AbstractDomain:
    """
    Factory function for creating abstract domains.
    
    Args:
        domain_type: Which abstract domain to create
        hidden_dim: Hidden dimension of the transformer
        **kwargs: Additional domain-specific arguments
        
    Returns:
        Instantiated abstract domain
    """
    if isinstance(domain_type, str):
        domain_type = AbstractDomainType(domain_type)
    
    domain_classes = {
        AbstractDomainType.TYPE_DOMAIN: TypeDomain,
        AbstractDomainType.INTERVAL_DOMAIN: IntervalDomain,
        AbstractDomainType.MONOTONICITY_DOMAIN: MonotonicityDomain,
        AbstractDomainType.TYPE_MONOTONICITY_DOMAIN: TypeMonotonicityDomain,
    }
    
    if domain_type not in domain_classes:
        raise ValueError(f"Unknown domain type: {domain_type}")
    
    return domain_classes[domain_type](hidden_dim=hidden_dim, **kwargs)
