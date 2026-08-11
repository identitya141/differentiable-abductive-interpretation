"""
Abstraction Layer for Differentiable Abstract Interpretation

This module implements the neural network layer that applies abstract interpretation
constraints to transformer hidden representations. It can be inserted at configurable
layers of a transformer to enforce compositional structure.

The layer:
1. Extracts abstract representations from concrete hidden states
2. Computes abstraction violation penalties
3. Optionally projects hidden states toward valid abstract regions
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .abstract_domains import (
    AbstractDomain,
    AbstractElement,
    TypeMonotonicityDomain,
    get_abstract_domain,
)

# Import full abstraction loss (composition/consistency/entropy)
from src.losses.abstraction_loss import AbstractionLoss, CompositionAwareAbstractionLoss


class AbstractionLayer(nn.Module):
    """
    Neural layer that applies abstract interpretation constraints.
    
    This layer is inserted after specified transformer layers and:
    1. Computes abstract representations of hidden states
    2. Measures abstraction violations (for loss computation)
    3. Optionally projects states toward valid abstract regions
    
    Architecture:
    
    Input (h) ──┬──────────────────────────────────────── Output
                │
                ▼
           AbstractDomain.abstract()
                │
                ▼
           AbstractElement (a)
                │
                ├──► concretize_loss(h, a) ──► abstraction_loss
                │
                └──► project_to_abstract(h, a) ──► Optional projection
    """
    
    def __init__(
        self,
        hidden_dim: int,
        domain_type: str = "type_monotonicity",
        domain_kwargs: Optional[Dict] = None,
        apply_projection: bool = False,
        projection_strength: float = 0.1,
        store_abstractions: bool = True,
        layer_idx: Optional[int] = None,
        # New: Full loss integration options
        use_full_abstraction_loss: bool = True,
        concretization_weight: float = 1.0,
        composition_weight: float = 0.5,
        consistency_weight: float = 0.1,
        entropy_regularization: float = 0.1,
        contrastive_weight: float = 0.0,
        structural_contrastive_weight: float = 0.0,
        composition_objective: str = "domain",
        contrastive_temperature: float = 0.1,
        abstraction_temperature: float = 1.0,
        use_attention_for_composition: bool = False,
        attention_threshold: float = 0.3,
    ):
        """
        Initialize abstraction layer.
        
        Args:
            hidden_dim: Hidden dimension of transformer
            domain_type: Type of abstract domain ("type", "interval", "monotonicity", "type_monotonicity")
            domain_kwargs: Additional kwargs for abstract domain
            apply_projection: Whether to project hidden states toward valid regions
            projection_strength: How much to move states toward valid regions (0-1)
            store_abstractions: Whether to store abstractions for later analysis
            layer_idx: Which transformer layer this is attached to (for logging)
            use_full_abstraction_loss: If True, use AbstractionLoss with composition/consistency/entropy
            concretization_weight: Weight for concretization loss component
            composition_weight: Weight for composition loss component
            consistency_weight: Weight for consistency loss component
            entropy_regularization: Weight for entropy regularization
            abstraction_temperature: Temperature for soft abstractions
            use_attention_for_composition: If True, auto-detect composition pairs from attention
            attention_threshold: Threshold for attention-based composition detection
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.domain_type = domain_type
        self.apply_projection = apply_projection
        self.projection_strength = projection_strength
        self.store_abstractions = store_abstractions
        self.layer_idx = layer_idx
        self.use_full_abstraction_loss = use_full_abstraction_loss
        
        # Initialize abstract domain
        domain_kwargs = domain_kwargs or {}
        self.abstract_domain = get_abstract_domain(
            domain_type=domain_type,
            hidden_dim=hidden_dim,
            **domain_kwargs
        )
        
        # Initialize full abstraction loss (composition + consistency + entropy)
        if use_full_abstraction_loss:
            if use_attention_for_composition:
                self.abstraction_loss_fn = CompositionAwareAbstractionLoss(
                    abstract_domain=self.abstract_domain,
                    concretization_weight=concretization_weight,
                    composition_weight=composition_weight,
                    consistency_weight=consistency_weight,
                    entropy_regularization=entropy_regularization,
                    contrastive_weight=contrastive_weight,
                    structural_contrastive_weight=structural_contrastive_weight,
                    composition_objective=composition_objective,
                    contrastive_temperature=contrastive_temperature,
                    temperature=abstraction_temperature,
                    use_attention_for_composition=True,
                    attention_threshold=attention_threshold,
                )
            else:
                self.abstraction_loss_fn = AbstractionLoss(
                    abstract_domain=self.abstract_domain,
                    concretization_weight=concretization_weight,
                    composition_weight=composition_weight,
                    consistency_weight=consistency_weight,
                    entropy_regularization=entropy_regularization,
                    contrastive_weight=contrastive_weight,
                    structural_contrastive_weight=structural_contrastive_weight,
                    composition_objective=composition_objective,
                    contrastive_temperature=contrastive_temperature,
                    temperature=abstraction_temperature,
                )
        else:
            self.abstraction_loss_fn = None
        
        # Projection network (if enabled)
        if apply_projection:
            self.projection_net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            # Initialize near identity
            nn.init.zeros_(self.projection_net[-1].weight)
            nn.init.zeros_(self.projection_net[-1].bias)
        
        # Storage for abstractions (for analysis)
        self._cached_abstraction: Optional[AbstractElement] = None
        self._cached_loss: Optional[torch.Tensor] = None
        self._cached_loss_components: Optional[Dict[str, torch.Tensor]] = None
        self._cached_attention_mask: Optional[torch.Tensor] = None
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        compute_loss: bool = True,
        attention_weights: Optional[torch.Tensor] = None,
        composition_pairs: Optional[List[Tuple[int, int, int]]] = None,
        composition_specs: Optional[List[List[Any]]] = None,
        similar_pairs: Optional[List[Tuple[int, int]]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Apply abstraction constraints to hidden states.
        
        Args:
            hidden_states: Transformer hidden states [batch, seq, hidden_dim]
            attention_mask: Mask for padding [batch, seq]
            compute_loss: Whether to compute abstraction loss
            attention_weights: Attention weights for composition detection [batch, heads, seq, seq]
            composition_pairs: Manual list of (i, j, k) composition triplets
            similar_pairs: Manual list of (i, j) pairs that should have similar abstractions
            
        Returns:
            Tuple of:
            - Modified hidden states (same shape)
            - Dict containing 'abstraction_loss', 'abstraction_loss_components', and diagnostics
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Compute abstract representation
        abstraction = self.abstract_domain.abstract(hidden_states)
        
        # Store for analysis
        if self.store_abstractions:
            self._cached_abstraction = abstraction
            self._cached_attention_mask = attention_mask
        
        # Compute abstraction loss
        outputs = {}
        if compute_loss:
            if self.use_full_abstraction_loss and self.abstraction_loss_fn is not None:
                # Use full AbstractionLoss with composition/consistency/entropy
                loss_output = self.abstraction_loss_fn(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    attention_weights=attention_weights,  # For CompositionAwareAbstractionLoss
                    composition_pairs=composition_pairs,
                    composition_specs=composition_specs,
                    similar_pairs=similar_pairs,
                )
                loss = loss_output.total_loss
                self._cached_loss = loss
                self._cached_loss_components = loss_output.loss_components
                outputs['abstraction_loss'] = loss
                outputs['abstraction_loss_components'] = loss_output.loss_components
            else:
                # Fallback: just concretize_loss (original behavior)
                loss = self.abstract_domain.concretize_loss(hidden_states, abstraction)
                self._cached_loss = loss
                self._cached_loss_components = {'concretization': loss}
                outputs['abstraction_loss'] = loss
                outputs['abstraction_loss_components'] = {'concretization': loss}
            
            # Add diagnostic info
            outputs['abstraction_diagnostics'] = self._compute_diagnostics(
                hidden_states, abstraction, attention_mask
            )
        
        # Optional: project hidden states toward valid abstract regions
        if self.apply_projection:
            hidden_states = self._project_toward_abstract(hidden_states, abstraction)
        
        return hidden_states, outputs
    
    def _project_toward_abstract(
        self,
        hidden_states: torch.Tensor,
        abstraction: AbstractElement
    ) -> torch.Tensor:
        """
        Project hidden states toward valid abstract regions.
        
        This is a soft projection that moves states slightly toward
        the centroid of their abstract class, encouraging representations
        that are more consistent with the abstract interpretation.
        """
        # Compute projection direction
        projection_delta = self.projection_net(hidden_states)
        
        # Apply soft projection
        projected = hidden_states + self.projection_strength * projection_delta
        
        return projected
    
    def _compute_diagnostics(
        self,
        hidden_states: torch.Tensor,
        abstraction: AbstractElement,
        attention_mask: Optional[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute diagnostic statistics about abstractions.
        """
        diagnostics = {}
        
        # Type domain diagnostics
        if hasattr(abstraction, 'type_component') or hasattr(abstraction, 'type_logits'):
            type_elem = getattr(abstraction, 'type_component', abstraction)
            if hasattr(type_elem, 'type_probs'):
                type_probs = type_elem.type_probs
                
                # Type entropy (lower = more confident)
                entropy = -(type_probs * (type_probs + 1e-10).log()).sum(dim=-1)
                if attention_mask is not None:
                    valid = attention_mask.bool()
                    entropy = entropy[valid]
                    valid_probs = type_probs[valid]
                else:
                    valid_probs = type_probs.reshape(-1, type_probs.size(-1))
                diagnostics['type_entropy_mean'] = entropy.mean()
                diagnostics['type_entropy_std'] = entropy.std()

                mean_type_distribution = valid_probs.mean(dim=0)
                diagnostics['type_usage_entropy'] = -(
                    mean_type_distribution
                    * mean_type_distribution.clamp_min(1e-10).log()
                ).sum()
                diagnostics['max_type_fraction'] = mean_type_distribution.max()
                diagnostics['type_frequencies'] = mean_type_distribution
                
                # Type diversity (number of types used)
                hard_types = valid_probs.argmax(dim=-1)
                num_unique_types = hard_types.unique().numel()
                diagnostics['num_unique_types'] = torch.tensor(num_unique_types, dtype=torch.float)
        
        # Monotonicity diagnostics
        if hasattr(abstraction, 'monotonicity_component'):
            mono_elem = abstraction.monotonicity_component
            if hasattr(mono_elem, 'monotonicity_probs'):
                mono_probs = mono_elem.monotonicity_probs
                
                # Monotonicity confidence
                max_prob = mono_probs.max(dim=-1).values
                diagnostics['monotonicity_confidence'] = max_prob.mean()
        
        return diagnostics
    
    def get_cached_abstraction(self) -> Optional[AbstractElement]:
        """Return the most recently computed abstraction."""
        return self._cached_abstraction
    
    def get_cached_loss(self) -> Optional[torch.Tensor]:
        """Return the most recently computed loss."""
        return self._cached_loss

    def get_cached_loss_components(self) -> Optional[Dict[str, torch.Tensor]]:
        """Return components from the most recently computed abstraction loss."""
        return self._cached_loss_components


class MultiLayerAbstractionModule(nn.Module):
    """
    Manages abstraction layers across multiple transformer layers.
    
    This module coordinates:
    1. Which transformer layers have abstraction constraints
    2. Hierarchical abstraction (different constraints at different depths)
    3. Cross-layer consistency constraints
    
    Layer Selection Strategy:
    - Early layers (1-3): Type constraints (categorical structure)
    - Middle layers (4-6): Monotonicity constraints (operational structure)
    - Later layers (7+): Combined constraints (compositional integration)
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        constrained_layers: List[int],
        domain_type: str = "type_monotonicity",
        domain_kwargs: Optional[Dict] = None,
        layer_specific_domains: Optional[Dict[int, str]] = None,
        apply_projection: bool = False,
        cross_layer_consistency: bool = True,
        consistency_weight: float = 0.1,
        projection_strength: float = 0.1,
        concretization_weight: float = 1.0,
        composition_weight: float = 0.5,
        entropy_regularization: float = 0.1,
        contrastive_weight: float = 0.0,
        structural_contrastive_weight: float = 0.0,
        composition_objective: str = "domain",
        contrastive_temperature: float = 0.1,
    ):
        """
        Initialize multi-layer abstraction module.
        
        Args:
            hidden_dim: Hidden dimension of transformer
            num_layers: Total number of transformer layers
            constrained_layers: Which layers to apply constraints to
            domain_type: Default domain type for all layers
            domain_kwargs: Additional kwargs for domains
            layer_specific_domains: Override domain type for specific layers
            apply_projection: Whether to project hidden states
            cross_layer_consistency: Whether to enforce consistency across layers
            consistency_weight: Weight for cross-layer consistency loss
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.constrained_layers = sorted(constrained_layers)
        self.cross_layer_consistency = cross_layer_consistency
        self.consistency_weight = consistency_weight
        
        # Create abstraction layer for each constrained layer
        self.abstraction_layers = nn.ModuleDict()
        
        layer_specific_domains = layer_specific_domains or {}
        
        for layer_idx in constrained_layers:
            layer_domain = layer_specific_domains.get(layer_idx, domain_type)
            
            self.abstraction_layers[str(layer_idx)] = AbstractionLayer(
                hidden_dim=hidden_dim,
                domain_type=layer_domain,
                domain_kwargs=domain_kwargs,
                apply_projection=apply_projection,
                projection_strength=projection_strength,
                layer_idx=layer_idx,
                concretization_weight=concretization_weight,
                composition_weight=composition_weight,
                consistency_weight=consistency_weight,
                entropy_regularization=entropy_regularization,
                contrastive_weight=contrastive_weight,
                structural_contrastive_weight=structural_contrastive_weight,
                composition_objective=composition_objective,
                contrastive_temperature=contrastive_temperature,
            )
    
    def forward(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        attention_weights: Optional[torch.Tensor] = None,
        composition_pairs: Optional[List[Tuple[int, int, int]]] = None,
        composition_specs: Optional[List[List[Any]]] = None,
        similar_pairs: Optional[List[Tuple[int, int]]] = None,
        compute_loss: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Apply abstraction constraint at specified layer.
        
        Args:
            layer_idx: Which transformer layer we're at
            hidden_states: Hidden states from that layer
            attention_mask: Attention mask
            attention_weights: Optional attention weights for composition detection
            composition_pairs: Optional manual composition pairs
            similar_pairs: Optional manual similar pairs
            
        Returns:
            Potentially modified hidden states and loss dict
        """
        if layer_idx not in self.constrained_layers:
            return hidden_states, {}
        
        layer_key = str(layer_idx)
        return self.abstraction_layers[layer_key](
            hidden_states,
            attention_mask=attention_mask,
            compute_loss=compute_loss,
            attention_weights=attention_weights,
            composition_pairs=composition_pairs,
            composition_specs=composition_specs,
            similar_pairs=similar_pairs,
        )
    
    def compute_cross_layer_consistency_loss(self) -> torch.Tensor:
        """
        Compute consistency loss across abstraction layers.
        
        Enforces that abstractions at different layers are compatible,
        creating a hierarchical abstraction structure.
        """
        if not self.cross_layer_consistency:
            return torch.tensor(0.0)
        
        if len(self.constrained_layers) < 2:
            return torch.tensor(0.0)
        
        total_consistency_loss = 0.0
        num_pairs = 0
        
        # Compare adjacent constrained layers
        for i in range(len(self.constrained_layers) - 1):
            layer1 = str(self.constrained_layers[i])
            layer2 = str(self.constrained_layers[i + 1])
            
            abs1 = self.abstraction_layers[layer1].get_cached_abstraction()
            abs2 = self.abstraction_layers[layer2].get_cached_abstraction()
            
            if abs1 is not None and abs2 is not None:
                domain = self.abstraction_layers[layer1].abstract_domain
                consistency_loss = domain.consistency_loss(
                    abs1,
                    abs2,
                    attention_mask=self.abstraction_layers[layer1]._cached_attention_mask,
                )
                total_consistency_loss = total_consistency_loss + consistency_loss
                num_pairs += 1
        
        if num_pairs > 0:
            return self.consistency_weight * total_consistency_loss / num_pairs
        
        return torch.tensor(0.0)
    
    def get_total_abstraction_loss(self) -> torch.Tensor:
        """
        Sum abstraction losses from all layers.
        """
        total_loss = torch.tensor(0.0)
        
        for layer_key in self.abstraction_layers:
            layer_loss = self.abstraction_layers[layer_key].get_cached_loss()
            if layer_loss is not None:
                total_loss = total_loss + layer_loss
        
        # Add cross-layer consistency
        total_loss = total_loss + self.compute_cross_layer_consistency_loss()
        
        return total_loss
    
    def get_all_diagnostics(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Gather diagnostics from all abstraction layers.
        """
        all_diagnostics = {}
        
        for layer_key in self.abstraction_layers:
            layer = self.abstraction_layers[layer_key]
            if hasattr(layer, '_cached_abstraction') and layer._cached_abstraction is not None:
                layer_diagnostics = layer._compute_diagnostics(
                    None,  # Hidden states not cached
                    layer._cached_abstraction,
                    layer._cached_attention_mask,
                )
                loss_components = layer.get_cached_loss_components() or {}
                layer_diagnostics.update({
                    f'loss_{name}': value
                    for name, value in loss_components.items()
                })
                all_diagnostics[f'layer_{layer_key}'] = layer_diagnostics
        
        return all_diagnostics


class AbstractionScheduler:
    """
    Schedules when and how strongly to apply abstraction constraints.
    
    Supports both STEP-BASED and EPOCH-BASED scheduling:
    
    STEP-BASED (recommended, finer control):
        Set use_step_schedule=True and provide warmup_steps, ramp_steps.
        - Steps 0 to warmup_steps: λ = 0 (let model learn task manifold)
        - Steps warmup_steps to warmup_steps+ramp_steps: Linear ramp to max_weight
        - Steps > warmup_steps+ramp_steps: λ = max_weight
        
        Rule of thumb: warmup over 10-30% of total training steps.
        Example for 10K steps: warmup_steps=1000, ramp_steps=2000
    
    EPOCH-BASED (legacy, coarser control):
        Set use_step_schedule=False (default for backward compatibility).
        - Epochs 0 to warmup_epochs: λ = 0
        - Epochs warmup_epochs to warmup_epochs+ramp_epochs: Linear ramp
        - Epochs > warmup_epochs+ramp_epochs: λ = max_weight
    
    BACKOFF MECHANISM (automatic safety valve):
        When task_loss_increasing warnings occur N times consecutively:
        - λ is set to 0 for K steps (backoff period)
        - After K steps, resume the normal schedule
        This turns warnings into automatic corrective action instead of spam.
    
    Step-based scheduling eliminates most 'task_loss_increasing' warnings
    by giving the model more time to learn the task before constraints.
    """
    
    def __init__(
        self,
        warmup_epochs: int = 3,
        ramp_epochs: int = 3,
        max_weight: float = 1.0,
        min_weight: float = 0.0,
        schedule_type: str = "linear",  # "linear", "cosine", "step"
        layer_schedule: Optional[Dict[int, Tuple[int, int]]] = None,  # layer -> (start_epoch, full_epoch)
        # Step-based scheduling (preferred)
        use_step_schedule: bool = False,
        warmup_steps: int = 1000,
        ramp_steps: int = 2000,
        # Backoff mechanism (automatic λ reduction on warnings)
        backoff_trigger_count: int = 5,     # Consecutive warnings before backoff
        backoff_steps: int = 100,           # Steps to keep λ=0 during backoff
        backoff_enabled: bool = True,       # Whether to use backoff mechanism
    ):
        """
        Initialize scheduler.
        
        Args:
            warmup_epochs: Epochs before abstraction starts (epoch-based mode)
            ramp_epochs: Epochs to ramp from min to max weight (epoch-based mode)
            max_weight: Maximum abstraction loss weight (λ_final)
            min_weight: Starting abstraction loss weight (usually 0)
            schedule_type: How to interpolate: "linear", "cosine", "step"
            layer_schedule: Per-layer schedule overrides (epoch-based)
            use_step_schedule: If True, use step-based scheduling (recommended)
            warmup_steps: Steps before abstraction starts (step-based mode)
            ramp_steps: Steps to ramp from min to max weight (step-based mode)
            backoff_trigger_count: Number of consecutive warnings before backoff
            backoff_steps: Number of steps to keep λ=0 during backoff
            backoff_enabled: Whether to enable automatic backoff on warnings
        """
        # Epoch-based settings (legacy)
        self.warmup_epochs = warmup_epochs
        self.ramp_epochs = ramp_epochs
        self.layer_schedule = layer_schedule or {}
        
        # Step-based settings (recommended)
        self.use_step_schedule = use_step_schedule
        self.warmup_steps = warmup_steps
        self.ramp_steps = ramp_steps
        
        # Common settings
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.schedule_type = schedule_type
        
        # Backoff mechanism state
        self.backoff_enabled = backoff_enabled
        self.backoff_trigger_count = backoff_trigger_count
        self.backoff_steps = backoff_steps
        self._consecutive_warnings = 0          # Current consecutive warning count
        self._backoff_active = False            # Currently in backoff period?
        self._backoff_start_step = 0            # Step when backoff started
        self._total_backoffs = 0                # Total backoffs triggered (diagnostic)

    def state_dict(self) -> Dict[str, Any]:
        """Serialize mutable scheduling/backoff state for exact resume."""
        return {
            "consecutive_warnings": self._consecutive_warnings,
            "backoff_active": self._backoff_active,
            "backoff_start_step": self._backoff_start_step,
            "total_backoffs": self._total_backoffs,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self._consecutive_warnings = int(state.get("consecutive_warnings", 0))
        self._backoff_active = bool(state.get("backoff_active", False))
        self._backoff_start_step = int(state.get("backoff_start_step", 0))
        self._total_backoffs = int(state.get("total_backoffs", 0))
    
    def get_weight(
        self, 
        epoch: int, 
        layer_idx: Optional[int] = None,
        global_step: Optional[int] = None,
    ) -> float:
        """
        Get abstraction loss weight for current training state.
        
        Args:
            epoch: Current epoch (used for epoch-based or layer schedule)
            layer_idx: Optional layer index for per-layer scheduling
            global_step: Current global step (used for step-based scheduling)
        
        Returns:
            Current λ weight for abstraction loss
        """
        # Check if we're in backoff period (λ=0 override)
        if self._is_in_backoff(global_step):
            return 0.0
        
        # Step-based scheduling takes priority if enabled and step is provided
        if self.use_step_schedule and global_step is not None:
            return self._get_weight_by_step(global_step)
        
        # Fall back to epoch-based scheduling
        return self._get_weight_by_epoch(epoch, layer_idx)
    
    def notify_warning(self, global_step: int, warning_type: str = "task_loss_increasing") -> bool:
        """
        Notify scheduler that an over-constraint warning occurred.
        
        If consecutive warnings reach backoff_trigger_count, triggers λ backoff.
        
        Args:
            global_step: Current global training step
            warning_type: Type of warning (currently only task_loss_increasing triggers backoff)
        
        Returns:
            True if backoff was triggered, False otherwise
        """
        if not self.backoff_enabled:
            return False
        
        # Only respond to task_loss_increasing for now
        if warning_type != "task_loss_increasing":
            return False
        
        # Don't accumulate warnings during warmup or backoff
        if self._is_in_backoff(global_step):
            return False
        if self.use_step_schedule and global_step < self.warmup_steps:
            return False
        
        self._consecutive_warnings += 1
        
        if self._consecutive_warnings >= self.backoff_trigger_count:
            # Trigger backoff!
            self._backoff_active = True
            self._backoff_start_step = global_step
            self._total_backoffs += 1
            self._consecutive_warnings = 0  # Reset counter
            return True
        
        return False
    
    def notify_no_warning(self):
        """
        Notify scheduler that a step completed without warning.
        
        Resets the consecutive warning counter.
        """
        self._consecutive_warnings = 0
    
    def _is_in_backoff(self, global_step: Optional[int]) -> bool:
        """
        Check if we're currently in a backoff period (λ=0).
        
        Args:
            global_step: Current global training step
        
        Returns:
            True if in backoff period
        """
        if not self._backoff_active or global_step is None:
            return False
        
        steps_in_backoff = global_step - self._backoff_start_step
        if steps_in_backoff >= self.backoff_steps:
            # Backoff period is over, resume normal schedule
            self._backoff_active = False
            return False
        
        return True
    
    def _get_weight_by_step(self, global_step: int) -> float:
        """
        Compute weight based on global training step.
        
        Schedule:
            - Steps [0, warmup_steps): λ = 0
            - Steps [warmup_steps, warmup_steps + ramp_steps): Linear/cosine ramp
            - Steps >= warmup_steps + ramp_steps: λ = max_weight
        """
        if global_step < self.warmup_steps:
            return 0.0
        
        # Compute progress through ramp phase
        steps_into_ramp = global_step - self.warmup_steps
        if self.ramp_steps <= 0:
            progress = 1.0
        else:
            progress = min(1.0, steps_into_ramp / self.ramp_steps)
        
        return self._apply_schedule_function(progress)
    
    def _get_weight_by_epoch(self, epoch: int, layer_idx: Optional[int] = None) -> float:
        """
        Compute weight based on epoch (legacy mode).
        """
        # Check layer-specific schedule
        if layer_idx is not None and layer_idx in self.layer_schedule:
            start_epoch, full_epoch = self.layer_schedule[layer_idx]
            if epoch < start_epoch:
                return 0.0
            effective_epoch = epoch - start_epoch
            effective_ramp = full_epoch - start_epoch
        else:
            if epoch < self.warmup_epochs:
                return 0.0
            effective_epoch = epoch - self.warmup_epochs
            effective_ramp = self.ramp_epochs
        
        # Compute schedule progress
        if effective_ramp <= 0:
            progress = 1.0
        else:
            progress = min(1.0, effective_epoch / effective_ramp)
        
        return self._apply_schedule_function(progress)
    
    def _apply_schedule_function(self, progress: float) -> float:
        """
        Apply the schedule function to interpolate between min and max weight.
        
        Args:
            progress: Value in [0, 1] indicating progress through ramp phase
        
        Returns:
            Interpolated weight value
        """
        if self.schedule_type == "linear":
            weight = self.min_weight + progress * (self.max_weight - self.min_weight)
        elif self.schedule_type == "cosine":
            # Smooth cosine annealing
            import math
            cosine_factor = 0.5 * (1 - math.cos(progress * math.pi))
            weight = self.min_weight + cosine_factor * (self.max_weight - self.min_weight)
        elif self.schedule_type == "step":
            # Binary: 0 until fully ramped, then max
            weight = self.max_weight if progress >= 1.0 else self.min_weight
        else:
            weight = self.max_weight
        
        return weight
    
    def is_layer_active(self, epoch: int, layer_idx: int, global_step: Optional[int] = None) -> bool:
        """
        Check if a layer should have abstraction constraints at current state.
        """
        if self.use_step_schedule and global_step is not None:
            return global_step >= self.warmup_steps
        
        if layer_idx in self.layer_schedule:
            start_epoch, _ = self.layer_schedule[layer_idx]
            return epoch >= start_epoch
        return epoch >= self.warmup_epochs
    
    def get_schedule_info(self, epoch: int, global_step: Optional[int] = None) -> dict:
        """
        Get diagnostic information about current schedule state.
        
        Useful for logging and debugging.
        """
        weight = self.get_weight(epoch, global_step=global_step)
        in_backoff = self._is_in_backoff(global_step)
        
        if in_backoff:
            phase = "backoff"
            backoff_remaining = self.backoff_steps - (global_step - self._backoff_start_step)
            progress = 1.0 - (backoff_remaining / self.backoff_steps)
        elif self.use_step_schedule and global_step is not None:
            if global_step < self.warmup_steps:
                phase = "warmup"
                progress = global_step / self.warmup_steps if self.warmup_steps > 0 else 1.0
            elif global_step < self.warmup_steps + self.ramp_steps:
                phase = "ramp"
                progress = (global_step - self.warmup_steps) / self.ramp_steps if self.ramp_steps > 0 else 1.0
            else:
                phase = "full"
                progress = 1.0
        else:
            if epoch < self.warmup_epochs:
                phase = "warmup"
                progress = epoch / self.warmup_epochs if self.warmup_epochs > 0 else 1.0
            elif epoch < self.warmup_epochs + self.ramp_epochs:
                phase = "ramp"
                progress = (epoch - self.warmup_epochs) / self.ramp_epochs if self.ramp_epochs > 0 else 1.0
            else:
                phase = "full"
                progress = 1.0
        
        return {
            "phase": phase,
            "progress": progress,
            "weight": weight,
            "max_weight": self.max_weight,
            "mode": "step" if (self.use_step_schedule and global_step is not None) else "epoch",
            "in_backoff": in_backoff,
            "consecutive_warnings": self._consecutive_warnings,
            "total_backoffs": self._total_backoffs,
        }
