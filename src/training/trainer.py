"""
Training Pipeline for DAI Experiments

Implements the complete training protocol with:
- Deterministic seeding for reproducibility
- Scheduled abstraction loss application
- Comprehensive logging and checkpointing
- Gradient monitoring for over-constraint detection
"""

import os
import random
import time
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import numpy as np
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from transformers.optimization import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from tqdm import tqdm

from src.models.dai_transformer import DAITransformer, DAIModelOutput
from src.utils.reproducibility import set_seed, get_reproducibility_info
from src.losses.abstraction_loss import OverConstraintDetector, SubConstituentLoss
from src.utils.generation import generate_scan_optimized, get_generation_config


logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """
    Complete training configuration.
    
    All hyperparameters are explicitly specified for reproducibility.
    """
    # Experiment identification
    experiment_name: str = "dai_experiment"
    run_id: Optional[str] = None
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True
    
    # Optimization
    optimizer: str = "adamw"
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Learning rate schedule
    lr_scheduler: str = "cosine_with_warmup"
    warmup_ratio: float = 0.1
    warmup_steps: Optional[int] = None  # Overrides warmup_ratio if set
    
    # Training duration
    num_epochs: int = 20
    max_steps: Optional[int] = None  # Overrides num_epochs if set
    
    # Batching
    train_batch_size: int = 32
    eval_batch_size: int = 64
    gradient_accumulation_steps: int = 1
    
    # DAI-specific: Abstraction loss scheduling
    # λ (abstraction_loss_weight) controls how strongly to enforce abstract constraints
    # START CONSERVATIVE: 2e-6, increase later if stable + EM improving
    abstraction_loss_weight: float = 0.000002  # λ_final = 2e-6 (conservative start)
    
    # Epoch-based scheduling (legacy, coarser control)
    abstraction_warmup_epochs: int = 3
    abstraction_ramp_epochs: int = 3
    
    # Step-based scheduling (recommended, finer control)
    # Safe schedule for SCAN (~10,620 total steps):
    #   0–1500: λ = 0 (pure task learning)
    #   1500–5000: ramp 0 → 2e-6
    #   5000+: λ = 2e-6 (full constraint)
    abstraction_use_step_schedule: bool = True
    abstraction_warmup_steps: int = 1500   # Steps with λ=0 (pure task learning)
    abstraction_ramp_steps: int = 3500     # Steps to linearly ramp to λ_final
    
    # Cap weighted abstraction loss relative to task loss.
    # If set (e.g., 0.8), enforces: wt_abs <= max_ratio * task_loss per batch.
    # This prevents abstraction from dominating once task loss becomes very small.
    abstraction_max_abs_task_ratio: Optional[float] = 0.8
    
    # λ Backoff mechanism (automatic safety valve)
    # When consecutive task_loss_increasing warnings occur, temporarily set λ=0
    # This turns warning spam into automatic corrective action
    abstraction_backoff_enabled: bool = True
    abstraction_backoff_trigger_count: int = 5    # Consecutive warnings before backoff
    abstraction_backoff_steps: int = 100          # Steps to keep λ=0 during backoff
    
    # Sub-constituent loss (proposal-compliant compositional constraint)
    use_sub_constituent_loss: bool = False
    sub_constituent_weight: float = 0.1
    sub_constituent_sample_ratio: float = 0.25  # Sample this fraction of batch for speed
    sub_constituent_pool_strategy: str = "mean"
    
    # Over-constraint detection
    use_over_constraint_detection: bool = True
    over_constraint_window: int = 100
    over_constraint_gradient_ratio: float = 10.0
    over_constraint_loss_threshold: float = 0.1
    
    # Mixed precision
    fp16: bool = False
    bf16: bool = False
    fp16_initial_scale: float = 1024.0
    
    # Checkpointing
    save_strategy: str = "epoch"  # "epoch", "steps", "best"
    save_steps: int = 500
    save_total_limit: int = 3
    
    # Logging
    logging_steps: int = 100
    log_abstraction_diagnostics: bool = True
    
    # Evaluation
    eval_strategy: str = "epoch"  # "epoch", "steps", "no"
    eval_steps: int = 500
    
    # Output
    output_dir: str = "./experiments"
    overwrite_output_dir: bool = False
    
    # Early stopping
    early_stopping_patience: Optional[int] = None
    early_stopping_threshold: float = 0.0
    
    def __post_init__(self):
        if self.run_id is None:
            self.run_id = f"{self.experiment_name}_{int(time.time())}"
        
        self.output_path = Path(self.output_dir) / self.run_id
        

@dataclass
class TrainingState:
    """
    Mutable training state for checkpointing.
    """
    global_step: int = 0
    epoch: int = 0
    # Negative infinity guarantees that epoch zero is a selectable checkpoint,
    # even when validation exact match is initially zero.
    best_metric: float = float("-inf")
    best_epoch: int = 0
    epochs_without_improvement: int = 0
    
    # Loss tracking
    total_loss_sum: float = 0.0
    task_loss_sum: float = 0.0
    abstraction_loss_sum: float = 0.0
    raw_abstraction_loss_sum: float = 0.0  # Unweighted abstraction loss
    num_batches: int = 0
    examples_seen: int = 0
    
    # Latest abstraction weight (for logging)
    current_abstraction_weight: float = 0.0
    
    def reset_epoch_stats(self):
        self.total_loss_sum = 0.0
        self.task_loss_sum = 0.0
        self.abstraction_loss_sum = 0.0
        self.raw_abstraction_loss_sum = 0.0
        self.num_batches = 0
    
    @property
    def avg_loss(self) -> float:
        return self.total_loss_sum / max(1, self.num_batches)
    
    @property
    def avg_task_loss(self) -> float:
        return self.task_loss_sum / max(1, self.num_batches)
    
    @property
    def avg_abstraction_loss(self) -> float:
        return self.abstraction_loss_sum / max(1, self.num_batches)
    
    @property
    def avg_raw_abstraction_loss(self) -> float:
        return self.raw_abstraction_loss_sum / max(1, self.num_batches)


class DAITrainer:
    """
    Trainer for DAI models.
    
    Features:
    - Full reproducibility with deterministic operations
    - Scheduled abstraction loss application
    - Comprehensive logging
    - Checkpoint management
    - Over-constraint detection
    """
    
    def __init__(
        self,
        model: DAITransformer,
        config: TrainingConfig,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        compute_metrics: Optional[Callable] = None,
        callbacks: Optional[List[Callable]] = None,
        tokenizer = None,  # Required for sub-constituent loss
        composition_parser = None,  # Optional parser for sub-constituent extraction
    ):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.compute_metrics = compute_metrics
        self.callbacks = callbacks or []
        self.tokenizer = tokenizer
        self.composition_parser = composition_parser
        
        # Set up reproducibility
        set_seed(config.seed, deterministic=config.deterministic)
        
        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Calculate training steps
        self.steps_per_epoch = len(train_dataloader) // config.gradient_accumulation_steps
        if config.max_steps is not None:
            self.total_steps = config.max_steps
            self.num_epochs = (config.max_steps // self.steps_per_epoch) + 1
        else:
            self.num_epochs = config.num_epochs
            self.total_steps = self.steps_per_epoch * self.num_epochs
        
        # Set up optimizer
        self.optimizer = self._create_optimizer()
        
        # Set up learning rate scheduler
        self.lr_scheduler = self._create_scheduler()
        
        # Mixed precision (only enable if CUDA is available)
        cuda_available = torch.cuda.is_available()
        self.scaler = (
            GradScaler(init_scale=config.fp16_initial_scale)
            if config.fp16 and cuda_available
            else None
        )
        self.use_amp = (config.fp16 or config.bf16) and cuda_available
        self.amp_dtype = torch.bfloat16 if config.bf16 else torch.float16
        
        # Training state
        self.state = TrainingState()
        
        # Set up output directory
        self._setup_output_dir()
        
        # Logging
        self.train_log: List[Dict] = []
        self.eval_log: List[Dict] = []
        
        # Over-constraint detection
        self.over_constraint_detector = None
        if config.use_over_constraint_detection:
            self.over_constraint_detector = OverConstraintDetector(
                task_loss_window=config.over_constraint_window,
                gradient_ratio_threshold=config.over_constraint_gradient_ratio,
                task_loss_increase_threshold=config.over_constraint_loss_threshold,
            )
            self.over_constraint_detector.to(self.device)
        
        # Sub-constituent loss (proposal equation: ||γ(h_x) - α(γ(h_{x_1}), γ(h_{x_2}))||²)
        self.sub_constituent_loss_fn = None
        if config.use_sub_constituent_loss and tokenizer is not None:
            try:
                # Get abstract domain from model's abstraction module
                abstract_domain = self._get_abstract_domain_from_model()
                if abstract_domain is not None:
                    self.sub_constituent_loss_fn = SubConstituentLoss(
                        abstract_domain=abstract_domain,
                        encoder=model.t5.encoder,  # Use T5 encoder
                        tokenizer=tokenizer,
                        composition_type="join",
                        pool_strategy=config.sub_constituent_pool_strategy,
                        weight=1.0,  # Weighting handled in training step
                    )
                    logger.info("SubConstituentLoss initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize SubConstituentLoss: {e}")
    
    def _get_abstract_domain_from_model(self):
        """Extract abstract domain from model's abstraction module."""
        try:
            # Access through DAI encoder's abstraction module
            if hasattr(self.model, 'dai_encoder'):
                module = self.model.dai_encoder.abstraction_module
                # Get first layer's abstract domain
                for layer_key in module.abstraction_layers:
                    return module.abstraction_layers[layer_key].abstract_domain
        except Exception as e:
            logger.warning(f"Could not extract abstract domain: {e}")
        return None
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with proper weight decay handling."""
        # Separate parameters that should/shouldn't have weight decay
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters()
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters()
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        return AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
        )
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        # Calculate warmup steps
        if self.config.warmup_steps is not None:
            warmup_steps = self.config.warmup_steps
        else:
            warmup_steps = int(self.total_steps * self.config.warmup_ratio)
        
        if self.config.lr_scheduler == "cosine_with_warmup":
            return get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=self.total_steps,
            )
        elif self.config.lr_scheduler == "linear":
            return LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=self.total_steps,
            )
        elif self.config.lr_scheduler == "constant":
            return None
        raise ValueError(f"Unknown learning-rate scheduler: {self.config.lr_scheduler}")

    def _current_learning_rate(self) -> float:
        """Return the effective learning rate with or without a scheduler."""
        if self.lr_scheduler is not None:
            return float(self.lr_scheduler.get_last_lr()[0])
        return float(self.optimizer.param_groups[0]["lr"])
    
    def _setup_output_dir(self):
        """Set up output directory structure."""
        self.config.output_path.mkdir(parents=True, exist_ok=True)
        
        # Subdirectories
        (self.config.output_path / "checkpoints").mkdir(exist_ok=True)
        (self.config.output_path / "logs").mkdir(exist_ok=True)
        
        # Save config
        config_path = self.config.output_path / "training_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2, default=str)
        
        # Save reproducibility info
        repro_path = self.config.output_path / "reproducibility.json"
        with open(repro_path, 'w') as f:
            json.dump(get_reproducibility_info(), f, indent=2)
    
    def train(self):
        """
        Main training loop.
        
        Returns:
            Training results dictionary
        """
        logger.info(f"Starting training: {self.config.experiment_name}")
        logger.info(f"  Epochs: {self.num_epochs}")
        logger.info(f"  Steps per epoch: {self.steps_per_epoch}")
        logger.info(f"  Total steps: {self.total_steps}")
        logger.info(f"  Batch size: {self.config.train_batch_size}")
        logger.info(f"  Gradient accumulation: {self.config.gradient_accumulation_steps}")
        
        self.model.train()
        training_start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        for epoch in range(self.num_epochs):
            self.state.epoch = epoch
            self.state.reset_epoch_stats()
            
            # Update model epoch for abstraction scheduling
            self.model.set_epoch(epoch)
            
            epoch_start_time = time.time()
            
            # Training epoch
            self._train_epoch(epoch)
            
            epoch_time = time.time() - epoch_start_time
            
            # Log epoch summary
            epoch_summary = {
                "epoch": epoch,
                "train_loss": self.state.avg_loss,
                "task_loss": self.state.avg_task_loss,
                "abstraction_loss": self.state.avg_abstraction_loss,
                "learning_rate": self.lr_scheduler.get_last_lr()[0] if self.lr_scheduler else self.config.learning_rate,
                "epoch_time": epoch_time,
            }
            self.train_log.append(epoch_summary)
            
            logger.info(
                f"Epoch {epoch}: loss={self.state.avg_loss:.4f}, "
                f"task_loss={self.state.avg_task_loss:.4f}, "
                f"abs_loss={self.state.avg_abstraction_loss:.4f}"
            )
            
            # Evaluation
            if self.config.eval_strategy == "epoch" and self.eval_dataloader is not None:
                eval_results = self.evaluate()
                self.eval_log.append({"epoch": epoch, **eval_results})
                
                # Check for improvement - prefer exact_match, then accuracy, then loss
                # Use explicit key tracking for debugging
                if "exact_match" in eval_results:
                    metric = eval_results["exact_match"]
                    metric_key = "exact_match"
                elif "accuracy" in eval_results:
                    metric = eval_results["accuracy"]
                    metric_key = "accuracy"
                else:
                    metric = eval_results.get("loss", 0)
                    metric_key = "loss"
                
                if metric > self.state.best_metric:
                    logger.info(
                        f"New best model at epoch {epoch}! "
                        f"{metric_key}={metric:.4f} > prev_best={self.state.best_metric:.4f}"
                    )
                    self.state.best_metric = metric
                    self.state.best_epoch = epoch
                    self.state.epochs_without_improvement = 0
                    self._save_checkpoint("best")
                else:
                    self.state.epochs_without_improvement += 1
                    logger.debug(
                        f"No improvement at epoch {epoch}: "
                        f"{metric_key}={metric:.4f} <= best={self.state.best_metric:.4f}"
                    )
                
                # Early stopping
                if (self.config.early_stopping_patience is not None and
                    self.state.epochs_without_improvement >= self.config.early_stopping_patience):
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            # Save checkpoint
            if self.config.save_strategy == "epoch":
                self._save_checkpoint(f"epoch_{epoch}")
            
            # Run callbacks
            for callback in self.callbacks:
                callback(self, epoch, epoch_summary)

            if (
                self.config.max_steps is not None
                and self.state.global_step >= self.config.max_steps
            ):
                break

        training_wall_clock_seconds = time.time() - training_start_time
        peak_cuda_memory_bytes = (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        )

        # Save final checkpoint and logs
        self._save_checkpoint("final")
        self._save_logs()
        
        return {
            "best_metric": self.state.best_metric,
            "best_epoch": self.state.best_epoch,
            "final_loss": self.state.avg_loss,
            "optimizer_updates": self.state.global_step,
            "examples_seen": self.state.examples_seen,
            "training_wall_clock_seconds": training_wall_clock_seconds,
            "accelerator_hours": training_wall_clock_seconds / 3600.0,
            "peak_cuda_memory_bytes": peak_cuda_memory_bytes,
            "train_log": self.train_log,
            "eval_log": self.eval_log,
        }
    
    def _train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        self.optimizer.zero_grad()
        
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {epoch}",
            disable=not self._is_main_process(),
        )
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = self._prepare_batch(batch)
            
            # Update model's global step for step-based λ scheduling
            # This must happen BEFORE forward pass so the scheduler uses the current step
            if hasattr(self.model, 'set_step'):
                self.model.set_step(self.state.global_step)
            
            # Forward pass
            # NOTE: Don't pass decoder_input_ids - let model create them from labels
            # using T5's _shift_right to ensure consistency with generate()
            with autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    compute_abstraction_loss=True,
                    composition_specs=batch.get("composition_specs"),
                )

                # Use model-provided task/abstraction losses but optionally cap
                # abstraction so it cannot swamp the task loss late in training.
                task_loss = outputs.task_loss
                abs_loss = outputs.abstraction_loss
                raw_abs = outputs.raw_abstraction_loss
                abs_weight = outputs.abstraction_weight
                abstraction_diagnostics = outputs.abstraction_diagnostics or {}
                composition_counts = [
                    layer_diagnostics.get("loss_composition_count", 0.0)
                    for layer_diagnostics in abstraction_diagnostics.values()
                ]
                composition_count = max(
                    (
                        float(value.detach().item())
                        if isinstance(value, torch.Tensor)
                        else float(value)
                    )
                    for value in composition_counts
                ) if composition_counts else 0.0
                composition_losses = [
                    layer_diagnostics.get("loss_composition")
                    for layer_diagnostics in abstraction_diagnostics.values()
                    if layer_diagnostics.get("loss_composition") is not None
                ]
                composition_loss_value = (
                    sum(float(value.detach().item()) for value in composition_losses)
                    / len(composition_losses)
                    if composition_losses
                    else 0.0
                )
                type_usage_entropies = [
                    layer_diagnostics.get("type_usage_entropy")
                    for layer_diagnostics in abstraction_diagnostics.values()
                    if layer_diagnostics.get("type_usage_entropy") is not None
                ]
                max_type_fractions = [
                    layer_diagnostics.get("max_type_fraction")
                    for layer_diagnostics in abstraction_diagnostics.values()
                    if layer_diagnostics.get("max_type_fraction") is not None
                ]
                type_usage_entropy = (
                    sum(float(value.detach().item()) for value in type_usage_entropies)
                    / len(type_usage_entropies)
                    if type_usage_entropies else 0.0
                )
                max_type_fraction = (
                    max(float(value.detach().item()) for value in max_type_fractions)
                    if max_type_fractions else 0.0
                )

                effective_abs_loss = abs_loss
                effective_abs_weight = abs_weight
                if (
                    self.config.abstraction_max_abs_task_ratio is not None
                    and task_loss is not None
                    and abs_loss is not None
                ):
                    cap = self.config.abstraction_max_abs_task_ratio * task_loss.detach()
                    scale = torch.clamp(cap / (abs_loss.detach() + 1e-12), max=1.0)
                    effective_abs_loss = abs_loss * scale
                    if effective_abs_weight is not None:
                        effective_abs_weight = float(effective_abs_weight) * float(scale.item())

                # Compute effective total loss for backward.
                loss = task_loss
                if loss is not None and effective_abs_loss is not None:
                    loss = loss + effective_abs_loss
                elif loss is None:
                    # Fallback (shouldn't happen for supervised training)
                    loss = outputs.loss
                
                # Sub-constituent loss (proposal equation)
                sub_constituent_loss = None
                if (
                    self.sub_constituent_loss_fn is not None
                    and self.config.use_sub_constituent_loss
                    and "input_text" in batch
                    and self.composition_parser is not None
                ):
                    # Sample subset of batch for speed
                    import random
                    batch_size = len(batch["input_text"])
                    sample_size = max(1, int(batch_size * self.config.sub_constituent_sample_ratio))
                    sample_indices = random.sample(range(batch_size), sample_size)
                    
                    full_inputs = [batch["input_text"][i] for i in sample_indices]
                    sub_inputs_list = [
                        self.composition_parser.get_sub_constituents(inp)
                        for inp in full_inputs
                    ]
                    
                    # Only compute if we have valid pairs
                    valid_count = sum(1 for subs in sub_inputs_list if len(subs) >= 2)
                    if valid_count > 0:
                        sub_constituent_loss = self.sub_constituent_loss_fn.forward_batch(
                            full_inputs, sub_inputs_list
                        )
                        loss = loss + self.config.sub_constituent_weight * sub_constituent_loss

                if self.config.gradient_accumulation_steps > 1:
                    loss = loss / self.config.gradient_accumulation_steps

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite training loss at epoch={epoch}, step={step}: "
                    f"{loss.detach().float().item()}"
                )
            
            should_log_update = (
                (step + 1) % self.config.gradient_accumulation_steps == 0
                and (self.state.global_step + 1) % self.config.logging_steps == 0
            )
            task_gradient_norm = 0.0
            abstraction_gradient_norm = 0.0
            abstraction_task_gradient_ratio = 0.0
            if should_log_update and task_loss is not None:
                shared_parameters = [
                    parameter
                    for parameter in self.model.t5.encoder.parameters()
                    if parameter.requires_grad
                ]

                def gradient_norm(objective):
                    if objective is None or not objective.requires_grad:
                        return 0.0
                    gradients = torch.autograd.grad(
                        objective,
                        shared_parameters,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    squares = [
                        gradient.detach().float().norm().pow(2)
                        for gradient in gradients
                        if gradient is not None
                    ]
                    return float(torch.stack(squares).sum().sqrt().item()) if squares else 0.0

                task_gradient_norm = gradient_norm(task_loss)
                abstraction_gradient_norm = gradient_norm(effective_abs_loss)
                abstraction_task_gradient_ratio = (
                    abstraction_gradient_norm / (task_gradient_norm + 1e-12)
                )

            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            composition_grad_norm = 0.0
            if should_log_update:
                composition_gradient_squares = []
                for name, parameter in self.model.named_parameters():
                    if (
                        "operator_composition_rules" in name
                        and parameter.grad is not None
                    ):
                        composition_gradient_squares.append(
                            parameter.grad.detach().float().norm().pow(2)
                        )
                if composition_gradient_squares:
                    composition_grad_norm = float(
                        torch.stack(composition_gradient_squares).sum().sqrt().item()
                    )
            
            # Over-constraint detection with λ backoff support
            # IMPORTANT: Only run detection when λ > 0 (abstraction loss is active)
            # Otherwise normal training loss fluctuations trigger spurious warnings
            abs_is_active = (effective_abs_weight is not None and effective_abs_weight > 0)
            if self.over_constraint_detector is not None and task_loss is not None and abs_is_active:
                detections = self.over_constraint_detector.update(
                    task_loss=task_loss.detach(),
                    abstraction_loss=(effective_abs_loss.detach() if effective_abs_loss is not None 
                                      else torch.tensor(0.0, device=self.device)),
                    task_gradients=(
                        torch.tensor(task_gradient_norm, device=self.device)
                        if should_log_update else None
                    ),
                    abstraction_gradients=(
                        torch.tensor(abstraction_gradient_norm, device=self.device)
                        if should_log_update else None
                    ),
                )
                if detections:
                    # Notify model of warning (may trigger λ backoff)
                    backoff_triggered = False
                    if hasattr(self.model, 'notify_warning'):
                        for warning_type in detections:
                            if self.model.notify_warning(warning_type):
                                backoff_triggered = True
                    
                    if backoff_triggered:
                        logger.warning(f"⚠️ Over-constraint detected: {detections} → λ BACKOFF TRIGGERED (λ=0 for next {self.config.abstraction_backoff_steps} steps)")
                    else:
                        logger.warning(f"⚠️ Over-constraint detected: {detections}")
                elif should_log_update:
                    # A clean measured update resets the consecutive-warning
                    # counter. Unmeasured steps must not erase gradient-ratio
                    # warnings emitted at the configured diagnostic interval.
                    if hasattr(self.model, 'notify_no_warning'):
                        self.model.notify_no_warning()
            
            # Update tracking
            # Track effective (post-cap) losses to match optimization.
            self.state.total_loss_sum += (loss.item() * self.config.gradient_accumulation_steps)
            if task_loss is not None:
                self.state.task_loss_sum += task_loss.item()
            if effective_abs_loss is not None:
                self.state.abstraction_loss_sum += effective_abs_loss.item()
            if raw_abs is not None:
                self.state.raw_abstraction_loss_sum += raw_abs.item()
            if effective_abs_weight is not None:
                self.state.current_abstraction_weight = effective_abs_weight
            self.state.num_batches += 1
            self.state.examples_seen += batch["input_ids"].size(0)
            
            # Gradient update
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError(
                        f"Non-finite gradient norm at epoch={epoch}, step={step}: "
                        f"{grad_norm.detach().float().item()}"
                    )
                
                # Optimizer step
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                
                self.optimizer.zero_grad()
                self.state.global_step += 1
                
                # Logging
                if self.state.global_step % self.config.logging_steps == 0:
                    progress_bar.set_postfix({
                        "loss": f"{self.state.avg_loss:.4f}",
                        "grad_norm": f"{grad_norm:.4f}",
                    })
                    # Compute epoch-averaged abs/task ratio
                    avg_abs_task_ratio = self.state.avg_abstraction_loss / (self.state.avg_task_loss + 1e-8)
                    
                    # Compute per-batch ratio for current step (more accurate for debugging)
                    batch_task = task_loss.item() if task_loss is not None else 0.0
                    batch_abs = effective_abs_loss.item() if effective_abs_loss is not None else 0.0
                    batch_ratio = batch_abs / (batch_task + 1e-8)
                    weighted_composition_loss = (
                        composition_loss_value
                        * float(effective_abs_weight or 0.0)
                        * float(self.model.config.composition_weight)
                    )
                    
                    # Persistent log entry with full diagnostic info
                    # Now includes BOTH epoch-avg and per-batch metrics for better debugging
                    logger.info(
                        f"Step {self.state.global_step} | Epoch {epoch} [{step+1}/{len(self.train_dataloader)}] | "
                        f"loss={self.state.avg_loss:.4f} | task_loss={self.state.avg_task_loss:.4f} | "
                        f"raw_abs={self.state.avg_raw_abstraction_loss:.4f} | "
                        f"λ={self.state.current_abstraction_weight:.6f} | "
                        f"wt_abs={self.state.avg_abstraction_loss:.6f} | "
                        f"comp_loss={composition_loss_value:.6f} | "
                        f"wt_comp={weighted_composition_loss:.6f} | "
                        f"comp_count={composition_count:.0f} | "
                        f"comp_grad_norm={composition_grad_norm:.6f} | "
                        f"task_grad_norm={task_gradient_norm:.6f} | "
                        f"abs_grad_norm={abstraction_gradient_norm:.6f} | "
                        f"abs_task_grad_ratio={abstraction_task_gradient_ratio:.6f} | "
                        f"type_usage_entropy={type_usage_entropy:.6f} | "
                        f"max_type_fraction={max_type_fraction:.6f} | "
                        f"avg_ratio={avg_abs_task_ratio:.4f} | "
                        f"grad_norm={grad_norm:.4f} | "
                        f"lr={self._current_learning_rate():.2e}"
                    )
                    # Per-batch diagnostics (current step, not epoch-averaged)
                    logger.info(
                        f"  └─ batch: task={batch_task:.4f} | abs={batch_abs:.6f} | ratio={batch_ratio:.4f}"
                    )
                    
                    # Warn if per-batch ratio is too high (abstraction dominating task)
                    # Use batch ratio for warning since it reflects current step behavior
                    if batch_ratio > 1.0:
                        logger.warning(
                            f"⚠️ Abstraction loss dominating task loss (batch_ratio={batch_ratio:.2f})! "
                            f"Consider reducing abstraction_loss_weight."
                        )
                    
                    # Log healthy ratio status (use batch ratio)
                    if 0.3 <= batch_ratio <= 0.8:
                        logger.info(f"✓ Healthy batch abs/task ratio: {batch_ratio:.2f}")
                
                # Step-based saving
                if (self.config.save_strategy == "steps" and
                    self.state.global_step % self.config.save_steps == 0):
                    self._save_checkpoint(f"step_{self.state.global_step}")
                
                # Check max steps
                if (self.config.max_steps is not None and
                    self.state.global_step >= self.config.max_steps):
                    break
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate model on eval_dataloader.
        
        Returns:
            Dictionary of evaluation metrics
        """
        if self.eval_dataloader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                batch = self._prepare_batch(batch)
                
                # NOTE: Don't pass decoder_input_ids - let model create them from labels
                # using T5's _shift_right to ensure consistency with generate()
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    compute_abstraction_loss=False,
                )
                
                total_loss += outputs.loss.item() * batch["input_ids"].size(0)
                total_samples += batch["input_ids"].size(0)
                
                # Generate predictions for accuracy
                # Use optimized generation with EOS-ban for SCAN length split
                if hasattr(self.model, 'generate'):
                    # Get dataset-specific generation config (passes tokenizer to detect atomic mode)
                    gen_config = get_generation_config(
                        getattr(self.config, 'dataset_type', 'scan'),
                        tokenizer=self.tokenizer,
                    )
                    # Fast eval defaults: force greedy decoding (num_beams=1) for per-epoch eval
                    # This massively reduces eval time while still tracking learning progress
                    gen_config = dict(gen_config)  # Make a copy to avoid mutating shared config
                    gen_config["num_beams"] = 1
                    gen_config["num_return_sequences"] = 1
                    # For atomic token mode, use tighter length constraints for speed
                    dataset_type = getattr(self.config, 'dataset_type', 'scan')
                    if dataset_type.lower().startswith("scan"):
                        gen_config["min_new_tokens"] = min(gen_config.get("min_new_tokens", 0), 20)
                        gen_config["max_new_tokens"] = min(gen_config.get("max_new_tokens", 256), 60)
                    generated = generate_scan_optimized(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        **gen_config,
                    )
                    all_predictions.extend(generated.tolist())
                    all_labels.extend(batch["labels"].tolist())
        
        self.model.train()
        
        results = {
            "loss": total_loss / total_samples,
        }
        
        # Compute accuracy if we have predictions
        if all_predictions and self.compute_metrics is not None:
            metrics = self.compute_metrics(all_predictions, all_labels)
            results.update(metrics)
        
        logger.info(f"Evaluation: {results}")
        return results
    
    def _prepare_batch(self, batch) -> Dict[str, torch.Tensor]:
        """Move batch to device."""
        if hasattr(batch, '_asdict'):
            batch = batch._asdict()
        elif hasattr(batch, '__dataclass_fields__'):
            batch = {k: getattr(batch, k) for k in batch.__dataclass_fields__}
        
        prepared = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                prepared[k] = v.to(self.device)
            elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                prepared[k] = torch.stack(v).to(self.device)
            else:
                prepared[k] = v
        
        return prepared
    
    def _save_checkpoint(self, name: str):
        """Save a checkpoint."""
        checkpoint_dir = self.config.output_path / "checkpoints" / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        torch.save(self.model.state_dict(), checkpoint_dir / "model.pt")
        
        # Save optimizer state
        torch.save(self.optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
        
        # Save scheduler state
        if self.lr_scheduler is not None:
            torch.save(self.lr_scheduler.state_dict(), checkpoint_dir / "scheduler.pt")
        if self.scaler is not None:
            torch.save(self.scaler.state_dict(), checkpoint_dir / "grad_scaler.pt")

        rng_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        torch.save(rng_state, checkpoint_dir / "rng_state.pt")
        torch.save(
            self.model.abstraction_scheduler.state_dict(),
            checkpoint_dir / "abstraction_scheduler.pt",
        )
        
        # Save training state with additional diagnostic info
        state_data = {
            "global_step": self.state.global_step,
            "epoch": self.state.epoch,
            "best_metric": self.state.best_metric,
            "best_epoch": self.state.best_epoch,
            "checkpoint_name": name,
        }
        with open(checkpoint_dir / "training_state.json", 'w') as f:
            json.dump(state_data, f, indent=2)
        
        logger.info(
            f"Saved checkpoint: {name} (epoch={self.state.epoch}, "
            f"best_metric={self.state.best_metric:.4f}, best_epoch={self.state.best_epoch})"
        )
    
    def load_checkpoint(self, name: str):
        """Load a checkpoint."""
        checkpoint_dir = self.config.output_path / "checkpoints" / name
        
        # Load model
        self.model.load_state_dict(
            torch.load(checkpoint_dir / "model.pt", map_location=self.device)
        )
        
        # Load optimizer
        self.optimizer.load_state_dict(
            torch.load(checkpoint_dir / "optimizer.pt", map_location=self.device)
        )
        
        # Load scheduler
        if self.lr_scheduler is not None and (checkpoint_dir / "scheduler.pt").exists():
            self.lr_scheduler.load_state_dict(
                torch.load(checkpoint_dir / "scheduler.pt", map_location=self.device)
            )
        if self.scaler is not None and (checkpoint_dir / "grad_scaler.pt").exists():
            self.scaler.load_state_dict(
                torch.load(checkpoint_dir / "grad_scaler.pt", map_location=self.device)
            )
        abstraction_scheduler_path = checkpoint_dir / "abstraction_scheduler.pt"
        if abstraction_scheduler_path.exists():
            self.model.abstraction_scheduler.load_state_dict(
                torch.load(abstraction_scheduler_path, map_location="cpu")
            )
        rng_path = checkpoint_dir / "rng_state.pt"
        if rng_path.exists():
            rng_state = torch.load(rng_path, map_location="cpu")
            random.setstate(rng_state["python"])
            np.random.set_state(rng_state["numpy"])
            torch.set_rng_state(rng_state["torch_cpu"])
            if torch.cuda.is_available() and rng_state["torch_cuda"] is not None:
                torch.cuda.set_rng_state_all(rng_state["torch_cuda"])
        
        # Load training state
        with open(checkpoint_dir / "training_state.json", 'r') as f:
            state_dict = json.load(f)
            self.state.global_step = state_dict["global_step"]
            self.state.epoch = state_dict["epoch"]
            self.state.best_metric = state_dict["best_metric"]
            self.state.best_epoch = state_dict["best_epoch"]
        
        logger.info(f"Loaded checkpoint: {name}")
    
    def _save_logs(self):
        """Save training and evaluation logs."""
        log_dir = self.config.output_path / "logs"
        
        with open(log_dir / "train_log.json", 'w') as f:
            json.dump(self.train_log, f, indent=2)
        
        with open(log_dir / "eval_log.json", 'w') as f:
            json.dump(self.eval_log, f, indent=2)
    
    def _is_main_process(self) -> bool:
        """Check if this is the main process (for distributed training)."""
        return True  # Single-GPU for now


def train_dai_model(
    model: DAITransformer,
    train_dataloader: DataLoader,
    eval_dataloader: Optional[DataLoader] = None,
    config: Optional[TrainingConfig] = None,
    **kwargs
) -> Dict:
    """
    Convenience function to train a DAI model.
    
    Args:
        model: DAI model to train
        train_dataloader: Training data
        eval_dataloader: Evaluation data
        config: Training configuration
        **kwargs: Override config parameters
        
    Returns:
        Training results
    """
    if config is None:
        config = TrainingConfig(**kwargs)
    else:
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)
    
    trainer = DAITrainer(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
    )
    
    return trainer.train()
