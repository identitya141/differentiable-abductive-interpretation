#!/usr/bin/env python3
"""
Main Training Script for DAI Experiments

Usage:
    # Train with config file
    python scripts/train.py --config configs/experiments/main_results.yaml

    # Train with overrides
    python scripts/train.py --config configs/experiments/main_results.yaml \
        --override "training.learning_rate=1e-4,model.num_types=32"
    
    # Train specific dataset
    python scripts/train.py --dataset scan --split length
    
    # Quick test
    python scripts/train.py --dataset scan --quick
"""

import argparse
from dataclasses import asdict
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
from transformers import T5Tokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.dai_transformer import DAITransformer, DAIConfig, TaskOnlyT5, create_dai_model
from src.training.trainer import DAITrainer, TrainingConfig
from src.evaluation.metrics import evaluate_model, CompositionalMetrics, normalize_for_eval
from src.evaluation.compositional_metrics import CompositionParser
from src.utils.config import load_config, parse_override, ExperimentConfig
from src.utils.reproducibility import set_seed, get_reproducibility_info, ReproducibilityManager
from src.utils.tokenizer_utils import (
    extend_tokenizer_for_dataset,
    resize_with_deterministic_added_token_init,
)
from src.utils.benchmark_contract import apply_benchmark_contract

# Dataset imports
from src.data.scan_dataset import SCANDataModule
from src.data.cogs_dataset import COGSDataModule
from src.data.slog_dataset import SLOGDataModule
from src.data.cfq_dataset import CFQDataModule
from src.data.clutrr_dataset import CLUTRRDataModule
from src.data.gsm8k_dataset import GSM8KDataModule

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_file_logging(output_dir: Path) -> logging.FileHandler:
    """Set up file handler for logging to .txt file for easy copying."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "training_log.txt"
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    # Add to root logger so all loggers write to this file
    logging.getLogger().addHandler(file_handler)
    logger.info(f"Logging to file: {log_file}")
    return file_handler


def log_config_verification(config, dataset_type: str):
    """Log comprehensive config verification at startup for debugging."""
    logger.info("=" * 60)
    logger.info("CONFIG VERIFICATION (for debugging reproducibility)")
    logger.info("=" * 60)
    
    # Data config
    logger.info("[DATA CONFIG]")
    logger.info(f"  dataset (raw):       {config.data.dataset}")
    logger.info(f"  dataset_type (used): {dataset_type}")
    logger.info(f"  dataset_split:       {config.data.dataset_split}")
    logger.info(f"  max_source_length:   {config.data.max_source_length}")
    logger.info(f"  max_target_length:   {config.data.max_target_length}")
    
    # Model config
    logger.info("[MODEL CONFIG]")
    logger.info(f"  base_model:          {config.model.base_model}")
    logger.info(f"  domain_type:         {config.model.domain_type}")
    logger.info(f"  num_types:           {config.model.num_types}")
    logger.info(f"  constrained_layers:  {config.model.constrained_layers}")
    
    # Training config
    logger.info("[TRAINING CONFIG]")
    logger.info(f"  num_epochs:          {config.training.num_epochs}")
    logger.info(f"  learning_rate:       {config.training.learning_rate}")
    logger.info(f"  train_batch_size:    {config.training.train_batch_size}")
    logger.info(f"  eval_batch_size:     {config.training.eval_batch_size}")
    logger.info(f"  warmup_ratio:        {config.training.warmup_ratio}")
    logger.info(f"  lr_scheduler:        {config.training.lr_scheduler}")
    logger.info(f"  seed:                {config.training.seed}")
    
    # Abstraction config (critical for DAI)
    logger.info("[ABSTRACTION CONFIG] (λ schedule)")
    logger.info(f"  abstraction_loss_weight: {config.abstraction.abstraction_loss_weight}")
    logger.info(f"  use_step_schedule:   {getattr(config.abstraction, 'use_step_schedule', 'NOT SET')}")
    logger.info(f"  warmup_steps:        {getattr(config.abstraction, 'warmup_steps', 'NOT SET')}")
    logger.info(f"  ramp_steps:          {getattr(config.abstraction, 'ramp_steps', 'NOT SET')}")
    logger.info(f"  warmup_epochs:       {config.abstraction.warmup_epochs}")
    logger.info(f"  ramp_epochs:         {config.abstraction.ramp_epochs}")
    logger.info(f"  max_abs_task_ratio:  {config.abstraction.max_abs_task_ratio}")
    logger.info(f"  backoff_enabled:     {getattr(config.abstraction, 'backoff_enabled', 'NOT SET')}")
    
    logger.info("=" * 60)


DATASET_MODULES = {
    'scan': SCANDataModule,
    'cogs': COGSDataModule,
    'slog': SLOGDataModule,
    'cfq': CFQDataModule,
    'clutrr': CLUTRRDataModule,
    'gsm8k': GSM8KDataModule,
}

def get_data_module(dataset: str, tokenizer, config: ExperimentConfig):
    """Create data module for specified dataset."""
    data_config = config.data
    
    if dataset == 'scan':
        return SCANDataModule(
            tokenizer=tokenizer,
            scan_split=data_config.dataset_split,
            batch_size=config.training.train_batch_size,
            max_source_length=data_config.max_source_length,
            max_target_length=data_config.max_target_length,
            num_workers=data_config.num_workers,
            eval_batch_size=config.training.eval_batch_size,
            eval_num_workers=data_config.eval_num_workers,
            cache_dir=data_config.cache_dir,
            data_dir=data_config.data_dir,
            validation_fraction=data_config.validation_fraction,
            composition_structure_mode=data_config.composition_structure_mode,
            structure_corruption_probability=data_config.structure_corruption_probability,
            input_representation=data_config.input_representation,
            nonce_primitives=data_config.nonce_primitives,
            seed=config.training.seed,
            split_seed=data_config.split_seed,
        )
    elif dataset == 'cogs':
        return COGSDataModule(
            tokenizer=tokenizer,
            batch_size=config.training.train_batch_size,
            max_source_length=data_config.max_source_length,
            max_target_length=data_config.max_target_length,
            num_workers=data_config.num_workers,
            eval_batch_size=config.training.eval_batch_size,
            eval_num_workers=data_config.eval_num_workers,
            cache_dir=data_config.cache_dir,
            data_dir=data_config.data_dir,
            composition_structure_mode=data_config.composition_structure_mode,
            seed=config.training.seed,
        )
    elif dataset == 'slog':
        return SLOGDataModule(
            tokenizer=tokenizer,
            batch_size=config.training.train_batch_size,
            max_source_length=data_config.max_source_length,
            max_target_length=data_config.max_target_length,
            num_workers=data_config.num_workers,
            eval_batch_size=config.training.eval_batch_size,
            eval_num_workers=data_config.eval_num_workers,
            cache_dir=data_config.cache_dir,
            data_dir=data_config.data_dir,
            composition_structure_mode=data_config.composition_structure_mode,
            seed=config.training.seed,
        )
    elif dataset == 'cfq':
        return CFQDataModule(
            tokenizer=tokenizer,
            cfq_split=data_config.dataset_split,
            batch_size=config.training.train_batch_size,
            max_source_length=data_config.max_source_length,
            max_target_length=data_config.max_target_length,
            num_workers=data_config.num_workers,
            eval_batch_size=config.training.eval_batch_size,
            eval_num_workers=data_config.eval_num_workers,
            cache_dir=data_config.cache_dir,
            data_dir=data_config.data_dir,
            validation_fraction=data_config.validation_fraction,
            seed=config.training.seed,
            split_seed=data_config.split_seed,
            composition_structure_mode=data_config.composition_structure_mode,
        )
    elif dataset == 'clutrr':
        return CLUTRRDataModule(
            tokenizer=tokenizer,
            batch_size=config.training.train_batch_size,
            max_source_length=data_config.max_source_length,
            max_target_length=data_config.max_target_length,
            num_workers=data_config.num_workers,
            cache_dir=data_config.cache_dir,
        )
    elif dataset == 'gsm8k':
        return GSM8KDataModule(
            tokenizer=tokenizer,
            batch_size=config.training.train_batch_size,
            max_source_length=data_config.max_source_length,
            max_target_length=data_config.max_target_length,
            num_workers=data_config.num_workers,
            cache_dir=data_config.cache_dir,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def create_model(config: ExperimentConfig) -> torch.nn.Module:
    """Create DAI model from config."""
    model_config = config.model
    abstraction_config = config.abstraction

    if model_config.architecture == "reference_t5":
        return TaskOnlyT5(
            base_model_name=model_config.base_model,
            pretrained=model_config.pretrained,
        )
    if model_config.architecture != "dai":
        raise ValueError(f"Unknown model architecture: {model_config.architecture}")
    
    dai_config = DAIConfig(
        base_model_name=model_config.base_model,
        domain_type=model_config.domain_type,
        constrained_layers=model_config.constrained_layers,
        num_types=model_config.num_types,
        type_embed_dim=model_config.type_embed_dim,
        monotonicity_dim=model_config.monotonicity_dim,
        apply_projection=model_config.apply_projection,
        projection_strength=model_config.projection_strength,
        cross_layer_consistency=model_config.cross_layer_consistency,
        consistency_weight=model_config.consistency_weight,
        composition_rules_trainable=model_config.composition_rules_trainable,
        operator_specific_composition=model_config.operator_specific_composition,
        concretization_weight=abstraction_config.concretization_weight,
        composition_weight=abstraction_config.composition_weight,
        entropy_regularization=abstraction_config.entropy_regularization,
        contrastive_weight=abstraction_config.contrastive_weight,
        structural_contrastive_weight=abstraction_config.structural_contrastive_weight,
        composition_objective=abstraction_config.composition_objective,
        contrastive_temperature=abstraction_config.contrastive_temperature,
        require_grounded_composition=(
            model_config.require_grounded_composition
            if model_config.require_grounded_composition is not None
            else (
                config.data.dataset.split("_")[0] in {"scan", "cogs", "slog", "cfq"}
                and (
                    abstraction_config.composition_weight > 0
                    or abstraction_config.structural_contrastive_weight > 0
                )
            )
        ),
        abstraction_loss_weight=abstraction_config.abstraction_loss_weight,
        # Epoch-based scheduling (legacy)
        warmup_epochs=abstraction_config.warmup_epochs,
        ramp_epochs=abstraction_config.ramp_epochs,
        # Step-based scheduling (recommended)
        use_step_schedule=getattr(abstraction_config, 'use_step_schedule', True),
        warmup_steps=getattr(abstraction_config, 'warmup_steps', 1000),
        ramp_steps=getattr(abstraction_config, 'ramp_steps', 2000),
        # λ Backoff mechanism (automatic safety valve)
        backoff_enabled=getattr(abstraction_config, 'backoff_enabled', True),
        backoff_trigger_count=getattr(abstraction_config, 'backoff_trigger_count', 5),
        backoff_steps=getattr(abstraction_config, 'backoff_steps', 100),
    )
    
    return DAITransformer(config=dai_config, pretrained=model_config.pretrained)


def train(
    config: ExperimentConfig,
    seed: Optional[int] = None,
    resume_from_checkpoint: Optional[str] = None,
    evaluation_only: bool = False,
) -> Dict:
    """
    Run training with given configuration.
    
    Args:
        config: Experiment configuration
        seed: Random seed (overrides config if provided)
        
    Returns:
        Training results dictionary
    """
    # Set seed
    seed = config.training.seed if seed is None else seed
    config.training.seed = seed
    contract = apply_benchmark_contract(config)
    run_output_dir = Path(config.output_dir) / f"seed_{seed}"
    repro_manager = ReproducibilityManager(seed, str(run_output_dir))
    repro_manager.save_info()
    resolved_config_path = run_output_dir / "resolved_config.json"
    with resolved_config_path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2, sort_keys=True)
    
    # Set up file logging for easy copying of job outputs
    file_handler = setup_file_logging(run_output_dir)
    
    # Derive dataset_type early for routing
    dataset_type = config.data.dataset.split("_")[0] if "_" in config.data.dataset else config.data.dataset
    
    # Log comprehensive config verification (critical for debugging)
    log_config_verification(config, dataset_type)
    
    logger.info(f"Starting experiment: {config.experiment_name}")
    logger.info(f"Seed: {seed}")
    logger.info(f"Output dir: {run_output_dir}")
    
    # Create tokenizer
    tokenizer = T5Tokenizer.from_pretrained(config.model.base_model)
    
    # Extend tokenizer with dataset-specific special tokens
    # For SCAN, this adds I_WALK, I_RUN, etc. as single tokens (not subwords)
    # NOTE: dataset_type already derived above for routing
    tokenizer, num_tokens_added = extend_tokenizer_for_dataset(
        tokenizer, dataset_type, verbose=True
    )
    if num_tokens_added > 0:
        logger.info(f"Extended tokenizer with {num_tokens_added} special tokens for {dataset_type}")
    
    # Create data module
    # FIX: Use dataset_type (e.g., "scan") not config.data.dataset (e.g., "scan_length")
    # get_data_module() expects: "scan", "cogs", "cfq", "clutrr", "gsm8k"
    data_module = get_data_module(dataset_type, tokenizer, config)
    data_module.setup()
    
    logger.info(f"Dataset: {config.data.dataset}")
    logger.info(f"Train examples: {len(data_module.train_dataset)}")
    logger.info(f"Test examples: {len(data_module.test_dataset)}")
    
    # Create model
    model = create_model(config)
    
    # Resize token embeddings to match extended tokenizer vocabulary
    # This is critical when special tokens (e.g., SCAN actions) are added
    resize_with_deterministic_added_token_init(
        model, len(tokenizer), seed=config.data.split_seed
    )
    logger.info(f"Model embeddings resized to {len(tokenizer)} tokens")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Create training config
    training_config = TrainingConfig(
        experiment_name=config.experiment_name,
        dataset_type=dataset_type,
        seed=seed,
        num_epochs=config.training.num_epochs,
        max_steps=config.training.max_steps,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        max_grad_norm=config.training.max_grad_norm,
        train_batch_size=config.training.train_batch_size,
        eval_batch_size=config.training.eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        warmup_ratio=config.training.warmup_ratio,
        lr_scheduler=config.training.lr_scheduler,
        fp16=config.training.fp16,
        fp16_initial_scale=config.training.fp16_initial_scale,
        abstraction_loss_weight=config.abstraction.abstraction_loss_weight,
        abstraction_warmup_epochs=config.abstraction.warmup_epochs,
        abstraction_ramp_epochs=config.abstraction.ramp_epochs,
        abstraction_use_step_schedule=config.abstraction.use_step_schedule,
        abstraction_warmup_steps=config.abstraction.warmup_steps,
        abstraction_ramp_steps=config.abstraction.ramp_steps,
        abstraction_max_abs_task_ratio=config.abstraction.max_abs_task_ratio,
        abstraction_backoff_enabled=config.abstraction.backoff_enabled,
        abstraction_backoff_trigger_count=config.abstraction.backoff_trigger_count,
        abstraction_backoff_steps=config.abstraction.backoff_steps,
        eval_strategy=config.eval_strategy,
        save_strategy=config.save_strategy,
        logging_steps=config.logging_steps,
        output_dir=str(run_output_dir),
    )
    
    # Create metric computation function with full diagnostics
    # NOTE: dataset_type already derived at top of function
    
    def compute_metrics(predictions, targets):
        metrics_computer = CompositionalMetrics(tokenizer=tokenizer, dataset_type=dataset_type)
        result = metrics_computer.compute(predictions, targets)
        metrics = {
            'accuracy': result.accuracy,
            'exact_match': result.exact_match,
            'ood_accuracy': result.out_of_distribution_accuracy,
            'gen_gap': result.generalization_gap,
        }
        # Add length diagnostics (critical for debugging EM issues)
        if result.avg_pred_length is not None:
            metrics['avg_pred_len'] = result.avg_pred_length
            metrics['avg_target_len'] = result.avg_target_length
            metrics['len_ratio'] = result.length_ratio
        if result.token_accuracy is not None:
            metrics['token_acc'] = result.token_accuracy
        return metrics
    
    # Create trainer
    trainer = DAITrainer(
        model=model,
        config=training_config,
        train_dataloader=data_module.train_dataloader(),
        eval_dataloader=data_module.validation_dataloader(),
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,  # Required for generation during evaluation
    )
    
    # Train
    if evaluation_only:
        logger.info("Evaluation-only replay from validation-selected best checkpoint")
        trainer.load_checkpoint("best")
        prior_paths = sorted(
            run_output_dir.glob(f"results_seed{seed}*.pre_replay.json")
        )
        results = (
            json.loads(prior_paths[-1].read_text(encoding="utf-8"))
            if prior_paths else {}
        )
        results.update({
            'evaluation_only_replay': True,
            'optimizer_updates': trainer.state.global_step,
            'examples_seen': trainer.state.examples_seen,
        })
    else:
        results = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    results['experiment_name'] = config.experiment_name
    results['dataset'] = dataset_type
    results['split'] = config.data.dataset_split
    results['method'] = os.environ.get('PUBLICATION_METHOD', config.experiment_name)
    results['seed'] = seed
    results['resolved_config'] = str(resolved_config_path)
    results['source_revision'] = repro_manager.info.get('git_hash')
    results['source_tree_dirty'] = repro_manager.info.get(
        'has_uncommitted_changes'
    )
    results['source_snapshot_id'] = os.environ.get('SOURCE_SNAPSHOT_ID')
    results['source_git_revision'] = os.environ.get('SOURCE_GIT_REVISION')
    results['config_sha256'] = os.environ.get('CONFIG_SHA256')
    results['data_sha256'] = os.environ.get('DATA_SHA256')
    results['model_parameters'] = {
        'total': total_params,
        'trainable': trainable_params,
    }
    results['benchmark_contract'] = asdict(contract)
    train_rows = getattr(data_module.train_dataset, "_tokenized_examples", [])
    train_relation_counts = [
        len(row.get("composition_specs", [])) for row in train_rows
    ]
    annotated_relation_counts = [count for count in train_relation_counts if count]
    results['structural_supervision'] = {
        'training_examples': len(train_relation_counts),
        'annotated_training_examples': len(annotated_relation_counts),
        'annotated_training_fraction': (
            len(annotated_relation_counts) / len(train_relation_counts)
            if train_relation_counts else 0.0
        ),
        'mean_relations_per_annotated_example': (
            sum(annotated_relation_counts) / len(annotated_relation_counts)
            if annotated_relation_counts else 0.0
        ),
        'structurally_supervised_batch_fraction': results.get(
            'structurally_supervised_batch_fraction', 0.0
        ),
    }

    # Final metrics must use the validation-selected checkpoint, never the
    # in-memory final epoch.
    if not evaluation_only:
        trainer.load_checkpoint("best")
    results['selected_checkpoint'] = {
        'name': 'best',
        'metric': 'validation_exact_match',
        'best_epoch': trainer.state.best_epoch,
        'best_metric': trainer.state.best_metric,
    }

    # Final held-out IID and OOD evaluation
    logger.info("Running final held-out IID evaluation...")
    iid_dataloader = (
        data_module.iid_test_dataloader()
        if hasattr(data_module, "iid_test_dataloader")
        else data_module.validation_dataloader()
    )
    iid_eval = evaluate_model(
        model=model,
        dataloader=iid_dataloader,
        tokenizer=tokenizer,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        dataset_type=dataset_type,
    )
    logger.info("Running final OOD evaluation...")
    final_eval = evaluate_model(
        model=model,
        dataloader=data_module.test_dataloader(),
        tokenizer=tokenizer,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        dataset_type=dataset_type,
    )
    
    iid_ood_gap = iid_eval.exact_match - final_eval.exact_match
    results['iid_evaluation'] = {
        'accuracy': iid_eval.accuracy,
        'exact_match': iid_eval.exact_match,
        'accuracy_by_depth': iid_eval.accuracy_by_depth,
        'accuracy_by_category': iid_eval.accuracy_by_category,
        'num_examples': iid_eval.num_examples,
        'num_correct': iid_eval.num_correct,
    }
    results['final_evaluation'] = {
        'accuracy': final_eval.accuracy,
        'exact_match': final_eval.exact_match,
        'ood_accuracy': final_eval.out_of_distribution_accuracy,
        'generalization_gap': iid_ood_gap,
        'accuracy_by_depth': final_eval.accuracy_by_depth,
        'accuracy_by_category': final_eval.accuracy_by_category,
        'num_examples': final_eval.num_examples,
        'num_correct': final_eval.num_correct,
    }

    predictions_path = run_output_dir / f"predictions_seed{seed}.jsonl"
    composition_parser = CompositionParser(dataset_type)
    test_rows = getattr(data_module.test_dataset, "_tokenized_examples", [])
    with predictions_path.open('w', encoding='utf-8') as handle:
        for index, (input_text, prediction, target) in enumerate(zip(
            final_eval.inputs or [],
            final_eval.predictions or [],
            final_eval.targets or [],
        )):
            normalized_prediction = normalize_for_eval(prediction, dataset_type)
            normalized_target = normalize_for_eval(target, dataset_type)
            artifact = {
                'example_index': index,
                'experiment_name': config.experiment_name,
                'method': os.environ.get('PUBLICATION_METHOD', config.experiment_name),
                'dataset': dataset_type,
                'split': config.data.dataset_split,
                'seed': seed,
                'input': input_text,
                'composition_depth': composition_parser.get_depth(input_text),
                'generalization_category': (
                    final_eval.categories[index]
                    if final_eval.categories is not None
                    else None
                ),
                'prediction': prediction,
                'target': target,
                'normalized_prediction': normalized_prediction,
                'normalized_target': normalized_target,
                'correct': normalized_prediction == normalized_target,
                'composition_violation': (
                    final_eval.composition_violations[index]
                    if final_eval.composition_violations is not None
                    else None
                ),
                'structural_relation_count': (
                    len(test_rows[index].get('composition_specs', []))
                    if index < len(test_rows) else None
                ),
                'structurally_annotated': (
                    bool(test_rows[index].get('composition_specs', []))
                    if index < len(test_rows) else None
                ),
            }
            handle.write(json.dumps(artifact, sort_keys=True) + '\n')
    results['prediction_artifact'] = str(predictions_path)
    
    # Save results
    results_path = run_output_dir / f"results_seed{seed}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Results saved to {results_path}")
    logger.info(f"Final accuracy: {final_eval.accuracy:.4f}")
    logger.info(f"OOD accuracy: {final_eval.out_of_distribution_accuracy:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train DAI model")
    
    # Config options
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to experiment config YAML file"
    )
    parser.add_argument(
        "--override",
        type=str,
        default=None,
        help="Config overrides in format 'key1=value1,key2=value2'"
    )
    
    # Quick options
    parser.add_argument(
        "--dataset",
        type=str,
        choices=['scan', 'cogs', 'slog', 'cfq', 'clutrr', 'gsm8k'],
        default='scan',
        help="Dataset to train on"
    )
    parser.add_argument(
        "--split",
        type=str,
        default='length',
        help="Dataset split (dataset-specific)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed override (defaults to the config seed)"
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Resume from a named checkpoint directory (for example epoch_7)",
    )
    parser.add_argument(
        "--evaluation-only",
        action="store_true",
        help="Recompute artifacts from the best checkpoint without training",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick training run (fewer epochs, smaller batch)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./experiments",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config is not None:
        overrides = parse_override(args.override) if args.override else None
        config = load_config(args.config, overrides)
    else:
        # Create default config from arguments
        from src.utils.config import (
            ExperimentConfig, ModelConfig, TrainingConfig as TConfig,
            DataConfig, AbstractionConfig, get_dataset_config
        )
        
        data_config = get_dataset_config(args.dataset)
        data_config.dataset_split = args.split
        
        training_config = TConfig(seed=args.seed if args.seed is not None else 42)
        if args.quick:
            training_config.num_epochs = 3
            training_config.train_batch_size = 8
        
        config = ExperimentConfig(
            experiment_name=f"dai_{args.dataset}_{args.split}",
            output_dir=args.output_dir,
            model=ModelConfig(),
            training=training_config,
            data=data_config,
            abstraction=AbstractionConfig(),
        )

    # Apply overrides if provided (supports nested keys like training.* / data.* / abstraction.*)
    if args.override:
        from src.utils.config import dataclass_to_dict, merge_configs, dict_to_config
        overrides = parse_override(args.override)
        config = dict_to_config(merge_configs(dataclass_to_dict(config), overrides))
    
    # Run training
    results = train(
        config,
        seed=args.seed,
        resume_from_checkpoint=args.resume_from_checkpoint,
        evaluation_only=args.evaluation_only,
    )
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    print(f"Final Accuracy: {results['final_evaluation']['accuracy']:.4f}")
    print(f"OOD Accuracy: {results['final_evaluation']['ood_accuracy']:.4f}")
    print(f"Generalization Gap: {results['final_evaluation']['generalization_gap']:.4f}")
    print("="*50)


if __name__ == "__main__":
    main()
