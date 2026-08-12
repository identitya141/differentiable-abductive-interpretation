"""
DAI Transformer: Differentiable Abstract Interpretation Enhanced Transformer

This module implements the main DAI model architecture, which augments a
standard transformer (T5) with differentiable abstract interpretation constraints.

Key Design Decisions:
1. Base Model: T5-small/T5-base (encoder-decoder, well-studied for compositional tasks)
2. Abstraction Placement: After encoder layers 2, 4, 6 (configurable)
3. Training Mode: Fine-tuning with abstraction loss as auxiliary objective
4. Inference Mode: Standard T5 inference (abstraction learned, not enforced)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5Config, T5ForConditionalGeneration, T5EncoderModel
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput

from .abstraction_layer import (
    AbstractionLayer,
    MultiLayerAbstractionModule,
    AbstractionScheduler,
)
from .abstract_domains import get_abstract_domain


@dataclass
class DAIConfig:
    """
    Configuration for DAI model.
    """
    # Base model
    base_model_name: str = "t5-small"
    
    # Abstraction settings
    domain_type: str = "type"  # Frozen primary abstract domain
    constrained_layers: List[int] = None  # Which encoder layers to constrain
    
    # Domain-specific settings
    num_types: int = 16
    type_embed_dim: int = 64
    monotonicity_dim: int = 64
    
    # Abstraction application
    apply_projection: bool = False  # Project hidden states at train time
    projection_strength: float = 0.1
    cross_layer_consistency: bool = False
    consistency_weight: float = 0.1
    composition_rules_trainable: bool = True
    operator_specific_composition: bool = True
    concretization_weight: float = 1.0
    composition_weight: float = 0.5
    entropy_regularization: float = 0.1
    contrastive_weight: float = 0.0
    structural_contrastive_weight: float = 0.0
    composition_objective: str = "domain"
    contrastive_temperature: float = 0.1
    require_grounded_composition: bool = False
    
    # Loss settings - START CONSERVATIVE: 2e-6, increase later if stable
    abstraction_loss_weight: float = 0.000002  # λ_final = 2e-6 (conservative start)
    
    # Schedule settings (epoch-based, legacy)
    warmup_epochs: int = 3
    ramp_epochs: int = 3
    
    # Step-based scheduling (recommended, finer control)
    # Safe schedule for SCAN (~10,620 total steps):
    #   0–1500: λ = 0 (pure task learning)
    #   1500–5000: ramp 0 → 2e-6
    #   5000+: λ = 2e-6 (full constraint)
    use_step_schedule: bool = True
    warmup_steps: int = 1500  # Steps with λ=0 (pure task learning)
    ramp_steps: int = 3500    # Steps to linearly ramp to λ_final
    
    # λ Backoff mechanism (automatic safety valve)
    # When N consecutive task_loss_increasing warnings occur, set λ=0 for K steps
    # This turns warning spam into automatic corrective action
    backoff_enabled: bool = True
    backoff_trigger_count: int = 5     # Consecutive warnings before backoff
    backoff_steps: int = 100           # Steps to keep λ=0 during backoff
    
    def __post_init__(self):
        if self.constrained_layers is None:
            # Default: constrain middle encoder layers
            self.constrained_layers = [2, 4]  # For T5-small (6 encoder layers)


@dataclass
class DAIModelOutput:
    """
    Output from DAI model including abstraction losses.
    """
    # Standard seq2seq outputs
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    encoder_last_hidden_state: Optional[torch.Tensor] = None
    
    # DAI-specific outputs
    task_loss: Optional[torch.Tensor] = None
    abstraction_loss: Optional[torch.Tensor] = None  # weighted: lambda_t * raw_abstraction_loss
    consistency_loss: Optional[torch.Tensor] = None
    
    # Diagnostic tracking for abstraction loss tuning
    raw_abstraction_loss: Optional[torch.Tensor] = None  # Before weighting
    abstraction_weight: Optional[float] = None  # Current lambda_t value
    
    # Diagnostics
    abstraction_diagnostics: Optional[Dict] = None


class TaskOnlyT5(nn.Module):
    """Reference T5 task path adapted to the shared DAI training interface."""

    def __init__(self, base_model_name: str, pretrained: bool = True):
        super().__init__()
        if pretrained:
            self.t5 = T5ForConditionalGeneration.from_pretrained(base_model_name)
        else:
            t5_config = T5Config.from_pretrained(base_model_name)
            self.t5 = T5ForConditionalGeneration(t5_config)
        self.config = DAIConfig(
            base_model_name=base_model_name,
            concretization_weight=0.0,
            composition_weight=0.0,
            entropy_regularization=0.0,
            contrastive_weight=0.0,
            structural_contrastive_weight=0.0,
            abstraction_loss_weight=0.0,
        )

    def set_epoch(self, epoch: int) -> None:
        """Accept the shared trainer epoch hook; task-only training has no schedule."""

    def set_step(self, global_step: int) -> None:
        """Accept the shared trainer step hook; task-only training has no schedule."""

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        compute_abstraction_loss: bool = False,
        composition_specs: Optional[List[List[Any]]] = None,
        **kwargs,
    ) -> DAIModelOutput:
        del compute_abstraction_loss, composition_specs
        outputs = self.t5(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_input_ids=decoder_input_ids,
            return_dict=True,
            **kwargs,
        )
        return DAIModelOutput(
            loss=outputs.loss,
            logits=outputs.logits,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            task_loss=outputs.loss,
            abstraction_loss=None,
            raw_abstraction_loss=None,
            abstraction_weight=0.0,
            abstraction_diagnostics={},
        )

    def resize_token_embeddings(self, new_num_tokens: int):
        return self.t5.resize_token_embeddings(new_num_tokens)

    def generate(self, *args, **kwargs):
        return self.t5.generate(*args, **kwargs)


class DAIEncoderWrapper(nn.Module):
    """
    Wrapper around T5 encoder that applies abstraction constraints.
    
    This wrapper intercepts hidden states after specified layers
    and applies abstract interpretation constraints.
    """
    
    def __init__(
        self,
        t5_encoder: nn.Module,
        config: DAIConfig,
    ):
        super().__init__()
        self.t5_encoder = t5_encoder
        self.config = config
        
        # Get hidden dim from T5 config
        hidden_dim = t5_encoder.config.d_model
        num_layers = t5_encoder.config.num_layers
        
        # Initialize abstraction module
        domain_kwargs = {
            'num_types': config.num_types,
            'type_embed_dim': config.type_embed_dim,
            'monotonicity_dim': config.monotonicity_dim,
            'composition_rules_trainable': config.composition_rules_trainable,
            'operator_specific_composition': config.operator_specific_composition,
        }
        
        self.abstraction_module = MultiLayerAbstractionModule(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            constrained_layers=config.constrained_layers,
            domain_type=config.domain_type,
            domain_kwargs=domain_kwargs,
            apply_projection=config.apply_projection,
            projection_strength=config.projection_strength,
            cross_layer_consistency=config.cross_layer_consistency,
            consistency_weight=config.consistency_weight,
            concretization_weight=config.concretization_weight,
            composition_weight=config.composition_weight,
            entropy_regularization=config.entropy_regularization,
            contrastive_weight=config.contrastive_weight,
            structural_contrastive_weight=config.structural_contrastive_weight,
            composition_objective=config.composition_objective,
            contrastive_temperature=config.contrastive_temperature,
        )
        
        # Track abstraction losses
        self._layer_abstraction_losses: Dict[int, torch.Tensor] = {}
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,  # Need hidden states for abstraction
        return_dict: Optional[bool] = None,
        use_attention_for_composition: bool = False,  # Pass attention to abstraction layer
        composition_specs: Optional[List[List[Any]]] = None,
        compute_abstraction_loss: bool = True,
    ):
        """
        Forward pass with abstraction constraints.
        
        Args:
            use_attention_for_composition: If True, pass attention weights to abstraction
                                           layer for composition detection (slower but more accurate)
        """
        # Clear cached losses
        self._layer_abstraction_losses = {}
        self._layer_loss_components = {}  # Track components per layer

        if (
            self.training
            and self.config.require_grounded_composition
            and composition_specs is None
        ):
            raise ValueError(
                "Grounded composition is required, but this batch has no composition metadata"
            )
        
        # Get embeddings
        if inputs_embeds is None:
            inputs_embeds = self.t5_encoder.embed_tokens(input_ids)

        input_shape = inputs_embeds.size()[:-1]
        if attention_mask is not None:
            extended_attention_mask = self.t5_encoder.get_extended_attention_mask(
                attention_mask, input_shape
            )
        else:
            extended_attention_mask = None

        # Apply encoder blocks with abstraction
        hidden_states = self.t5_encoder.dropout(inputs_embeds)
        position_bias = None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions or use_attention_for_composition else None
        
        # Determine if we need attention weights for abstraction
        need_attentions = output_attentions or use_attention_for_composition
        
        # Iterate through encoder blocks
        for layer_idx, layer in enumerate(self.t5_encoder.block):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            # Apply encoder layer
            layer_outputs = layer(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                output_attentions=need_attentions,
            )
            hidden_states = layer_outputs[0]
            position_bias = layer_outputs[1]

            # Capture attention for this layer
            layer_attention = None
            if need_attentions and len(layer_outputs) > 2:
                layer_attention = layer_outputs[2]
                if output_attentions:
                    all_attentions = all_attentions + (layer_attention,)
            
            # Apply abstraction constraint (if this layer is constrained)
            if layer_idx in self.config.constrained_layers and compute_abstraction_loss:
                hidden_states, abs_outputs = self.abstraction_module(
                    layer_idx=layer_idx,
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    attention_weights=layer_attention if use_attention_for_composition else None,
                    composition_specs=composition_specs,
                    compute_loss=compute_abstraction_loss,
                )
                if 'abstraction_loss' in abs_outputs:
                    self._layer_abstraction_losses[layer_idx] = abs_outputs['abstraction_loss']
                if 'abstraction_loss_components' in abs_outputs:
                    self._layer_loss_components[layer_idx] = abs_outputs['abstraction_loss_components']
        
        # Final layer norm
        hidden_states = self.t5_encoder.final_layer_norm(hidden_states)
        hidden_states = self.t5_encoder.dropout(hidden_states)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions if output_attentions else None,
        )
    
    def get_total_abstraction_loss(self) -> torch.Tensor:
        """Get total abstraction loss from all constrained layers.
        
        Returns the MEAN abstraction loss across all layers for numerical stability.
        """
        if not self._layer_abstraction_losses:
            return torch.tensor(0.0)
        
        # Use MEAN instead of SUM to keep loss on same scale regardless of num layers
        num_layers = len(self._layer_abstraction_losses)
        total = sum(self._layer_abstraction_losses.values()) / num_layers
        
        # Add cross-layer consistency loss (already normalized)
        cross_layer_loss = self.abstraction_module.compute_cross_layer_consistency_loss()
        total = total + cross_layer_loss
        
        return total


class DAITransformer(nn.Module):
    """
    Main DAI Transformer model.
    
    This model wraps a T5 model and adds differentiable abstract interpretation
    constraints to bias representations toward compositional structure.
    
    Architecture:
    
    Input ─► T5 Encoder (with DAI constraints) ─► T5 Decoder ─► Output
                        │
                        ▼
              Abstraction Loss (training only)
    
    Training Objective:
        L = L_task + λ(t) * L_abstraction
    
    where:
    - L_task: Standard cross-entropy seq2seq loss
    - L_abstraction: Sum of abstraction violation losses
    - λ(t): Scheduled abstraction weight
    """
    
    def __init__(
        self,
        config: DAIConfig,
        pretrained: bool = True,
    ):
        super().__init__()
        self.config = config
        
        # Load base T5 model
        if pretrained:
            self.t5 = T5ForConditionalGeneration.from_pretrained(config.base_model_name)
        else:
            t5_config = T5Config.from_pretrained(config.base_model_name)
            self.t5 = T5ForConditionalGeneration(t5_config)
        
        # Wrap encoder with abstraction constraints
        self.dai_encoder = DAIEncoderWrapper(
            t5_encoder=self.t5.encoder,
            config=config,
        )
        
        # Scheduler for abstraction loss (supports step-based scheduling + backoff)
        self.abstraction_scheduler = AbstractionScheduler(
            warmup_epochs=config.warmup_epochs,
            ramp_epochs=config.ramp_epochs,
            max_weight=config.abstraction_loss_weight,
            use_step_schedule=config.use_step_schedule,
            warmup_steps=config.warmup_steps,
            ramp_steps=config.ramp_steps,
            backoff_trigger_count=getattr(config, 'backoff_trigger_count', 5),
            backoff_steps=getattr(config, 'backoff_steps', 100),
            backoff_enabled=getattr(config, 'backoff_enabled', True),
        )
        
        # Current epoch and step (updated by trainer)
        self._current_epoch = 0
        self._global_step = 0
    
    def set_epoch(self, epoch: int):
        """Set current training epoch for scheduling."""
        self._current_epoch = epoch
    
    def set_step(self, global_step: int):
        """Set current global training step for step-based scheduling."""
        self._global_step = global_step
    
    def notify_warning(self, warning_type: str = "task_loss_increasing") -> bool:
        """Notify scheduler of over-constraint warning. Returns True if backoff triggered."""
        return self.abstraction_scheduler.notify_warning(self._global_step, warning_type)
    
    def notify_no_warning(self):
        """Notify scheduler that step completed without warning."""
        self.abstraction_scheduler.notify_no_warning()
    
    def resize_token_embeddings(self, new_num_tokens: int):
        """Resize token embeddings to match tokenizer vocabulary size.
        
        Args:
            new_num_tokens: New vocabulary size
            
        Returns:
            The resized embedding layer
        """
        return self.t5.resize_token_embeddings(new_num_tokens)
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,
        return_dict: bool = True,
        compute_abstraction_loss: bool = True,
        compute_abstraction_diagnostics: bool = False,
        composition_specs: Optional[List[List[Any]]] = None,
    ) -> DAIModelOutput:
        """
        Forward pass with DAI constraints.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            decoder_input_ids: Decoder input IDs [batch, tgt_len]
            decoder_attention_mask: Decoder attention mask [batch, tgt_len]
            labels: Target token IDs for loss computation [batch, tgt_len]
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return hidden states
            return_dict: Whether to return dict or tuple
            compute_abstraction_loss: Whether to compute abstraction loss
            
        Returns:
            DAIModelOutput with task loss, abstraction loss, and outputs
        """
        abs_weight = None
        should_compute_abstraction = False
        if compute_abstraction_loss and self.training:
            abs_weight = self.abstraction_scheduler.get_weight(
                self._current_epoch,
                global_step=self._global_step,
            )
            should_compute_abstraction = abs_weight > 0.0
        if compute_abstraction_diagnostics:
            # Explicit, loss-free diagnostic pass. This refreshes per-example
            # caches in eval mode without enabling dropout or modifying task loss.
            should_compute_abstraction = True

        # During warmup/evaluation this is a genuine task-only encoder pass.
        encoder_outputs = self.dai_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            composition_specs=composition_specs,
            compute_abstraction_loss=should_compute_abstraction,
        )

        # Delegate all decoder/LM details to the installed Transformers version.
        task_outputs = self.t5(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        task_loss = task_outputs.loss
        lm_logits = task_outputs.logits
        
        # Compute abstraction loss
        abstraction_loss = None
        raw_abs_loss = None
        if should_compute_abstraction and self.training:
            raw_abs_loss = self.dai_encoder.get_total_abstraction_loss()
            abstraction_loss = abs_weight * raw_abs_loss
        elif compute_abstraction_loss and self.training:
            abstraction_loss = lm_logits.detach().new_zeros(())
        
        # Combine losses
        total_loss = task_loss
        if total_loss is not None and abstraction_loss is not None:
            total_loss = total_loss + abstraction_loss
        
        return DAIModelOutput(
            loss=total_loss,
            logits=lm_logits,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            task_loss=task_loss,
            abstraction_loss=abstraction_loss,
            raw_abstraction_loss=raw_abs_loss,
            abstraction_weight=abs_weight,
            abstraction_diagnostics=(
                self.dai_encoder.abstraction_module.get_all_diagnostics()
                if should_compute_abstraction else None
            ),
        )
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **generate_kwargs
    ) -> torch.Tensor:
        """
        Generate output sequences.
        
        At inference time, abstraction constraints are not enforced,
        but the model has learned representations that respect them.
        """
        # Encode with DAI encoder
        encoder_outputs = self.dai_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            compute_abstraction_loss=False,
        )
        
        return self.t5.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            **generate_kwargs
        )
    
    def get_abstraction_diagnostics(self) -> Dict:
        """Get diagnostic information about learned abstractions."""
        return self.dai_encoder.abstraction_module.get_all_diagnostics()
    
    def save_pretrained(self, save_directory: str):
        """
        Save the DAI model to a directory.
        
        Saves:
        - dai_config.json: DAI-specific configuration
        - pytorch_model.bin: Full model state dict
        - t5_config.json: Base T5 configuration (for reference)
        
        Args:
            save_directory: Path to save the model
        """
        import os
        import json
        from dataclasses import asdict
        
        os.makedirs(save_directory, exist_ok=True)
        
        # Save DAI config
        config_dict = asdict(self.config)
        with open(os.path.join(save_directory, "dai_config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)
        
        # Save T5 config for reference
        self.t5.config.save_pretrained(save_directory)
        
        # Save full model state dict
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        
        print(f"DAI model saved to {save_directory}")
    
    @classmethod
    def from_pretrained(cls, load_directory: str, **kwargs) -> "DAITransformer":
        """
        Load a DAI model from a directory.
        
        Args:
            load_directory: Path to load the model from
            **kwargs: Override config options
            
        Returns:
            Loaded DAITransformer model
        """
        import os
        import json
        
        # Load DAI config
        config_path = os.path.join(load_directory, "dai_config.json")
        if not os.path.exists(config_path):
            raise ValueError(f"No dai_config.json found in {load_directory}")
        
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        
        # Override with kwargs
        config_dict.update(kwargs)
        
        # Create config
        config = DAIConfig(**config_dict)
        
        # Create model (with pretrained=False to avoid downloading again)
        # We'll load weights from checkpoint
        model = cls(config=config, pretrained=False)
        
        # Load state dict
        state_dict_path = os.path.join(load_directory, "pytorch_model.bin")
        if os.path.exists(state_dict_path):
            state_dict = torch.load(state_dict_path, map_location="cpu")
            model.load_state_dict(state_dict)
            print(f"Loaded DAI model from {load_directory}")
        else:
            raise ValueError(f"No pytorch_model.bin found in {load_directory}")
        
        return model


class DAITransformerForSequenceClassification(nn.Module):
    """
    DAI Transformer for sequence classification tasks (e.g., CLUTRR).
    
    Uses encoder-only T5 with abstraction constraints and a classification head.
    """
    
    def __init__(
        self,
        config: DAIConfig,
        num_labels: int,
        pretrained: bool = True,
    ):
        super().__init__()
        self.config = config
        self.num_labels = num_labels
        
        # Load base T5 encoder
        if pretrained:
            t5 = T5ForConditionalGeneration.from_pretrained(config.base_model_name)
            self.t5_encoder = t5.encoder
        else:
            t5_config = T5Config.from_pretrained(config.base_model_name)
            t5 = T5ForConditionalGeneration(t5_config)
            self.t5_encoder = t5.encoder
        
        # Wrap with DAI
        self.dai_encoder = DAIEncoderWrapper(
            t5_encoder=self.t5_encoder,
            config=config,
        )
        
        # Classification head
        hidden_dim = self.t5_encoder.config.d_model
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_labels),
        )
        
        # Scheduler (supports step-based scheduling + backoff)
        self.abstraction_scheduler = AbstractionScheduler(
            warmup_epochs=config.warmup_epochs,
            ramp_epochs=config.ramp_epochs,
            max_weight=config.abstraction_loss_weight,
            use_step_schedule=config.use_step_schedule,
            warmup_steps=config.warmup_steps,
            ramp_steps=config.ramp_steps,
            backoff_trigger_count=getattr(config, 'backoff_trigger_count', 5),
            backoff_steps=getattr(config, 'backoff_steps', 100),
            backoff_enabled=getattr(config, 'backoff_enabled', True),
        )
        
        self._current_epoch = 0
        self._global_step = 0
    
    def set_epoch(self, epoch: int):
        self._current_epoch = epoch
    
    def set_step(self, global_step: int):
        """Set current global training step for step-based scheduling."""
        self._global_step = global_step
    
    def notify_warning(self, warning_type: str = "task_loss_increasing") -> bool:
        """Notify scheduler of over-constraint warning. Returns True if backoff triggered."""
        return self.abstraction_scheduler.notify_warning(self._global_step, warning_type)
    
    def notify_no_warning(self):
        """Notify scheduler that step completed without warning."""
        self.abstraction_scheduler.notify_no_warning()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        compute_abstraction_loss: bool = True,
    ) -> DAIModelOutput:
        """Forward pass for classification."""
        # Encode
        encoder_outputs = self.dai_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        # Pool (mean over sequence)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(encoder_outputs.last_hidden_state.size()).float()
            sum_hidden = torch.sum(encoder_outputs.last_hidden_state * mask_expanded, dim=1)
            sum_mask = mask_expanded.sum(dim=1)
            pooled = sum_hidden / sum_mask
        else:
            pooled = encoder_outputs.last_hidden_state.mean(dim=1)
        
        # Classify
        logits = self.classifier(pooled)
        
        # Task loss
        task_loss = None
        if labels is not None:
            task_loss = F.cross_entropy(logits, labels)
        
        # Abstraction loss
        abstraction_loss = None
        raw_abs_loss = None
        abs_weight = None
        if compute_abstraction_loss and self.training:
            raw_abs_loss = self.dai_encoder.get_total_abstraction_loss()
            abs_weight = self.abstraction_scheduler.get_weight(
                self._current_epoch, 
                global_step=self._global_step
            )
            abstraction_loss = (
                abs_weight * raw_abs_loss
                if abs_weight > 0.0
                else raw_abs_loss.detach().new_zeros(())
            )
        
        # Combine
        total_loss = task_loss
        if total_loss is not None and abstraction_loss is not None:
            total_loss = total_loss + abstraction_loss
        
        return DAIModelOutput(
            loss=total_loss,
            logits=logits,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            task_loss=task_loss,
            abstraction_loss=abstraction_loss,
        )


def create_dai_model(
    task_type: str = "seq2seq",
    base_model: str = "t5-small",
    domain_type: str = "type",
    constrained_layers: Optional[List[int]] = None,
    num_labels: Optional[int] = None,
    **kwargs
) -> nn.Module:
    """
    Factory function to create DAI model.
    
    Args:
        task_type: "seq2seq" or "classification"
        base_model: Base model name
        domain_type: Abstract domain to use
        constrained_layers: Which layers to constrain
        num_labels: Number of labels (for classification)
        **kwargs: Additional config options
        
    Returns:
        DAI model instance
    """
    config = DAIConfig(
        base_model_name=base_model,
        domain_type=domain_type,
        constrained_layers=constrained_layers,
        **kwargs
    )
    
    if task_type == "seq2seq":
        return DAITransformer(config=config, pretrained=True)
    elif task_type == "classification":
        if num_labels is None:
            raise ValueError("num_labels required for classification")
        return DAITransformerForSequenceClassification(
            config=config,
            num_labels=num_labels,
            pretrained=True
        )
    else:
        raise ValueError(f"Unknown task type: {task_type}")
