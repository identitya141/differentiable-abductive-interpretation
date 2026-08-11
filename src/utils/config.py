"""
Configuration Management for DAI Experiments

Provides a hierarchical JSON/YAML configuration system with:
- Base configurations
- Dataset-specific overrides
- Experiment-specific overrides
- Command-line overrides
"""

import json
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

@dataclass
class ModelConfig:
    """Model configuration."""
    base_model: str = "t5-small"
    architecture: str = "dai"
    pretrained: bool = True
    domain_type: str = "type_monotonicity"
    constrained_layers: List[int] = field(default_factory=lambda: [2, 4])
    num_types: int = 16
    type_embed_dim: int = 64
    monotonicity_dim: int = 64
    apply_projection: bool = False
    projection_strength: float = 0.1
    cross_layer_consistency: bool = True
    consistency_weight: float = 0.1
    composition_rules_trainable: bool = True
    operator_specific_composition: bool = True
    require_grounded_composition: Optional[bool] = None


@dataclass
class TrainingConfig:
    """Training configuration."""
    seed: int = 42
    num_epochs: int = 20
    max_steps: Optional[int] = None
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    train_batch_size: int = 32
    eval_batch_size: int = 64
    gradient_accumulation_steps: int = 1
    warmup_ratio: float = 0.1
    lr_scheduler: str = "cosine_with_warmup"
    fp16: bool = False
    fp16_initial_scale: float = 1024.0


@dataclass
class DataConfig:
    """Data configuration."""
    dataset: str = "scan"
    dataset_split: str = "length"
    max_source_length: int = 128
    max_target_length: int = 128
    num_workers: int = 4
    eval_num_workers: int = 0
    split_seed: int = 42
    cache_dir: Optional[str] = None
    data_dir: Optional[str] = None
    validation_fraction: float = 0.1
    composition_structure_mode: str = "grounded"
    structure_corruption_probability: float = 0.0
    input_representation: str = "plain"
    nonce_primitives: bool = False


@dataclass
class AbstractionConfig:
    """Abstraction-specific configuration."""
    # START CONSERVATIVE: 2e-6, increase later if stable + EM improving
    abstraction_loss_weight: float = 0.000002  # λ_final = 2e-6
    
    # Epoch-based scheduling (legacy, coarser control)
    warmup_epochs: int = 3
    ramp_epochs: int = 3
    
    # Step-based scheduling (recommended, finer control)
    # Safe schedule for SCAN (~10,620 total steps):
    #   0–1500: λ = 0, 1500–5000: ramp, 5000+: λ = 2e-6
    use_step_schedule: bool = True
    warmup_steps: int = 1500   # Steps with λ=0 (pure task learning)
    ramp_steps: int = 3500     # Steps to linearly ramp λ from 0 → λ_final
    
    # λ Backoff mechanism (automatic safety valve)
    # When N consecutive task_loss_increasing warnings occur, set λ=0 for K steps
    # This turns warning spam into automatic corrective action
    backoff_enabled: bool = True
    backoff_trigger_count: int = 5     # Consecutive warnings before backoff
    backoff_steps: int = 100           # Steps to keep λ=0 during backoff
    
    # Cap weighted abstraction loss relative to task loss.
    # If set (e.g., 0.8), enforces: wt_abs <= max_ratio * task_loss per batch.
    max_abs_task_ratio: Optional[float] = 0.8
    
    concretization_weight: float = 1.0
    composition_weight: float = 0.5
    consistency_weight: float = 0.1
    entropy_regularization: float = 0.1
    contrastive_weight: float = 0.0
    structural_contrastive_weight: float = 0.0
    composition_objective: str = "domain"
    contrastive_temperature: float = 0.1


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    experiment_name: str = "dai_experiment"
    output_dir: str = "./experiments"
    
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    abstraction: AbstractionConfig = field(default_factory=AbstractionConfig)
    
    # Evaluation
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    
    # Logging
    logging_steps: int = 100
    log_abstraction_diagnostics: bool = True


def load_config(
    config_path: Union[str, Path],
    overrides: Optional[Dict[str, Any]] = None,
) -> ExperimentConfig:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        overrides: Dictionary of values to override
        
    Returns:
        ExperimentConfig
    """
    config_path = Path(config_path)
    
    with open(config_path, 'r') as f:
        if config_path.suffix.lower() == '.json':
            config_dict = json.load(f)
        else:
            import yaml

            config_dict = yaml.safe_load(f)
    
    # Handle inheritance
    if 'base' in config_dict:
        base_path = config_path.parent / config_dict['base']
        base_config = load_config(base_path)
        config_dict = merge_configs(dataclass_to_dict(base_config), config_dict)
        del config_dict['base']
    
    # Apply overrides
    if overrides:
        config_dict = merge_configs(config_dict, overrides)
    
    # Convert to dataclasses
    return dict_to_config(config_dict)


def save_config(config: ExperimentConfig, path: Union[str, Path]):
    """Save configuration to YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    config_dict = dataclass_to_dict(config)
    
    with open(path, 'w') as f:
        if path.suffix.lower() == '.json':
            json.dump(config_dict, f, indent=2)
            f.write('\n')
        else:
            import yaml

            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


def dataclass_to_dict(obj: Any) -> Dict[str, Any]:
    """Convert nested dataclass to dictionary."""
    if hasattr(obj, '__dataclass_fields__'):
        result = {}
        for f in fields(obj):
            value = getattr(obj, f.name)
            result[f.name] = dataclass_to_dict(value)
        return result
    elif isinstance(obj, list):
        return [dataclass_to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: dataclass_to_dict(v) for k, v in obj.items()}
    else:
        return obj


def dict_to_config(config_dict: Dict[str, Any]) -> ExperimentConfig:
    """Convert dictionary to ExperimentConfig."""
    model_dict = config_dict.get('model', {})
    training_dict = config_dict.get('training', {})
    data_dict = config_dict.get('data', {})
    abstraction_dict = config_dict.get('abstraction', {})
    
    return ExperimentConfig(
        experiment_name=config_dict.get('experiment_name', 'dai_experiment'),
        output_dir=config_dict.get('output_dir', './experiments'),
        model=ModelConfig(**model_dict),
        training=TrainingConfig(**training_dict),
        data=DataConfig(**data_dict),
        abstraction=AbstractionConfig(**abstraction_dict),
        eval_strategy=config_dict.get('eval_strategy', 'epoch'),
        save_strategy=config_dict.get('save_strategy', 'epoch'),
        logging_steps=config_dict.get('logging_steps', 100),
        log_abstraction_diagnostics=config_dict.get('log_abstraction_diagnostics', True),
    )


def merge_configs(base: Dict, override: Dict) -> Dict:
    """Recursively merge override into base."""
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def parse_override(override_str: str) -> Dict[str, Any]:
    """
    Parse command-line override string.
    
    Format: "model.num_types=32,training.learning_rate=1e-4"
    """
    overrides = {}
    
    for item in override_str.split(','):
        if '=' not in item:
            continue
        
        key, value = item.split('=', 1)
        
        # Parse value type
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                if value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
        
        # Handle nested keys
        parts = key.split('.')
        current = overrides
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    
    return overrides


# Pre-defined configurations for each dataset
DATASET_CONFIGS = {
    'scan': DataConfig(
        dataset='scan',
        dataset_split='length',
        max_source_length=64,
        max_target_length=128,
    ),
    'cogs': DataConfig(
        dataset='cogs',
        max_source_length=128,
        max_target_length=256,
    ),
    'slog': DataConfig(
        dataset='slog',
        max_source_length=256,
        max_target_length=512,
    ),
    'cfq': DataConfig(
        dataset='cfq',
        dataset_split='mcd1',
        max_source_length=128,
        max_target_length=256,
    ),
    'clutrr': DataConfig(
        dataset='clutrr',
        max_source_length=256,
        max_target_length=32,
    ),
    'gsm8k': DataConfig(
        dataset='gsm8k',
        max_source_length=256,
        max_target_length=512,
    ),
}


def get_dataset_config(dataset: str) -> DataConfig:
    """Get default configuration for a dataset."""
    if dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset}")
    return DATASET_CONFIGS[dataset]
