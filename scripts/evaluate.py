#!/usr/bin/env python3
"""
Evaluation Script for DAI Models

Evaluates trained models on test sets and computes all metrics
required for the paper.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/dai/best_model --output results/dai.json
    python scripts/evaluate.py --checkpoint checkpoints/dai/best_model --quick  # Fast evaluation
"""

import argparse
from dataclasses import asdict
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import (
    CompositionalMetrics,
    EvaluationResult,
    OverConstraintMetrics,
)


def load_model_and_tokenizer(checkpoint_path: Path):
    """Load trained model and tokenizer from checkpoint."""
    from transformers import AutoTokenizer
    from src.models.dai_transformer import DAITransformer
    
    print(f"Loading model from {checkpoint_path}")
    
    # Load config
    config_path = checkpoint_path / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {}
    
    # Load model
    model = DAITransformer.from_pretrained(str(checkpoint_path))
    model.eval()
    
    # Load tokenizer
    tokenizer_path = checkpoint_path / "tokenizer"
    if tokenizer_path.exists():
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    else:
        # Fall back to base model tokenizer
        base_model = config.get("base_model", "t5-small")
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    return model, tokenizer, config


def load_dataset(dataset_name: str, split: str, data_dir: Path, tokenizer):
    """Load evaluation dataset."""
    from src.data import (
        SCANDataset,
        COGSDataset,
        CFQDataset,
        CLUTRRDataset,
        GSM8KDataset,
        SLOGDataset,
    )
    
    dataset_classes = {
        "scan": SCANDataset,
        "cogs": COGSDataset,
        "cfq": CFQDataset,
        "clutrr": CLUTRRDataset,
        "gsm8k": GSM8KDataset,
        "slog": SLOGDataset,
    }
    
    dataset_cls = dataset_classes.get(dataset_name)
    if dataset_cls is None:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return dataset_cls(
        tokenizer=tokenizer,
        data_dir=str(data_dir / dataset_name),
        split=split,
    )


def evaluate_model(
    model,
    tokenizer,
    dataset,
    device: torch.device,
    max_samples: Optional[int] = None,
    batch_size: int = 32,
    dataset_type: str = "scan",
    baseline_type: Optional[str] = None,
    num_workers: int = 0,
) -> EvaluationResult:
    """
    Evaluate model on dataset.
    
    Args:
        model: The model to evaluate
        tokenizer: Tokenizer for decoding
        dataset: Dataset to evaluate on
        device: Device for computation
        max_samples: Optional limit on samples
        batch_size: Batch size for evaluation
        dataset_type: Dataset type for normalization ("scan", "cogs", "cfq", etc.)
        baseline_type: Baseline type for output normalization ("cot", "scratchpad", etc.)
        num_workers: DataLoader worker processes. Zero is the portable default;
            callers may increase it for large cluster evaluations.
    
    Returns:
        EvaluationResult with all metrics
    """
    from src.evaluation.metrics import normalize_for_eval
    
    model = model.to(device)
    model.eval()
    
    # Initialize metrics
    metrics = CompositionalMetrics(dataset_type=dataset_type)
    
    predictions = []
    targets = []
    inputs = []
    is_ood = []
    categories = []
    depths = []
    
    # Create dataloader
    from torch.utils.data import DataLoader
    
    collate_fn = getattr(dataset, "collate_fn", None)
    if max_samples:
        indices = list(range(min(max_samples, len(dataset))))
        from torch.utils.data import Subset
        dataset = Subset(dataset, indices)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move to device
            input_ids = batch.input_ids.to(device)
            attention_mask = batch.attention_mask.to(device)
            
            # Generate predictions
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=128,
                num_beams=4,
                early_stopping=True,
            )
            
            # Decode predictions and targets
            raw_preds = tokenizer.batch_decode(generated, skip_special_tokens=True)
            labels = batch.labels.clone()
            labels[labels == -100] = tokenizer.pad_token_id
            raw_targets = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            # Normalize for fair cross-baseline evaluation
            batch_preds = [
                normalize_for_eval(p, dataset_type=dataset_type, baseline_type=baseline_type)
                for p in raw_preds
            ]
            batch_targets = [
                normalize_for_eval(t, dataset_type=dataset_type, baseline_type=None)
                for t in raw_targets
            ]
            
            predictions.extend(batch_preds)
            targets.extend(batch_targets)
            
            inputs.extend(batch.original_input_texts or batch.input_texts or [])
            if batch.is_ood is not None:
                is_ood.extend(batch.is_ood.tolist())
            categories.extend(batch.generalization_categories or [])
            depths.extend(batch.composition_depths or [])
    
    # Compute metrics
    result = metrics.compute(
        predictions=predictions,
        targets=targets,
        inputs=inputs or None,
        is_ood=is_ood or None,
        categories=categories or None,
        depths=depths if any(depth is not None for depth in depths) else None,
    )
    
    return result


def evaluate_ood_splits(
    model,
    tokenizer,
    dataset_name: str,
    data_dir: Path,
    device: torch.device,
    max_samples: Optional[int] = None,
    baseline_type: Optional[str] = None,
) -> Dict[str, EvaluationResult]:
    """Evaluate on multiple OOD splits for compositional generalization."""
    results = {}
    
    # Define OOD splits for each dataset
    ood_splits = {
        "scan": ["length", "addprim_jump", "addprim_turn_left"],
        "cogs": ["gen"],
        "cfq": ["mcd1", "mcd2", "mcd3"],
        "clutrr": ["k=4", "k=5", "k=6", "k=7", "k=8", "k=9", "k=10"],
        "gsm8k": ["hard"],  # Custom split of harder problems
        "slog": ["iid_test", "gen"],
    }
    
    splits = ood_splits.get(dataset_name, [])
    
    for split in splits:
        print(f"\nEvaluating {dataset_name} - {split}")
        try:
            dataset = load_dataset(dataset_name, split, data_dir, tokenizer)
            result = evaluate_model(
                model, tokenizer, dataset, device, max_samples,
                dataset_type=dataset_name, baseline_type=baseline_type
            )
            results[split] = result
        except Exception as e:
            print(f"  Warning: Could not evaluate {split}: {e}")
    
    return results


def save_results(results: Dict, output_path: Path):
    """Save evaluation results to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    def serialize(value):
        if isinstance(value, EvaluationResult):
            return asdict(value)
        if isinstance(value, dict):
            return {key: serialize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [serialize(item) for item in value]
        return value

    serializable = serialize(results)
    
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


def print_results_table(results: Dict):
    """Print results in a formatted table."""
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    for dataset, dataset_results in results.items():
        print(f"\n{dataset.upper()}")
        print("-"*60)
        
        if isinstance(dataset_results, EvaluationResult):
            print(f"  Exact Match: {dataset_results.exact_match:.2%}")
            print(f"  Token F1:    {dataset_results.token_f1:.2%}")
        elif isinstance(dataset_results, dict):
            for split, result in dataset_results.items():
                if isinstance(result, EvaluationResult):
                    print(f"  {split:20s} EM: {result.exact_match:.2%}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate DAI models")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="results/evaluation.json",
                        help="Output path for results")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Directory containing datasets")
    parser.add_argument("--datasets", type=str, nargs="+",
                        default=["scan", "cogs", "cfq", "clutrr", "slog"],
                        help="Datasets to evaluate on")
    parser.add_argument("--baseline-type", type=str, default=None,
                        choices=["vanilla", "cot", "scratchpad", "tinyllama_lora", "llama", None],
                        help="Baseline type for output normalization")
    parser.add_argument("--quick", action="store_true",
                        help="Quick evaluation with reduced samples")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")
    
    args = parser.parse_args()
    
    # Setup
    checkpoint_path = Path(args.checkpoint)
    data_dir = Path(args.data_dir)
    device = torch.device(args.device)
    max_samples = 100 if args.quick else None
    
    print("="*60)
    print("DAI Evaluation")
    print("="*60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Quick mode: {args.quick}")
    print(f"Baseline type: {args.baseline_type or 'auto-detect'}")
    
    # Load model
    model, tokenizer, config = load_model_and_tokenizer(checkpoint_path)
    
    # Evaluate on each dataset
    all_results = {"config": config, "datasets": {}}
    
    for dataset_name in args.datasets:
        print(f"\n{'='*60}")
        print(f"Evaluating on {dataset_name.upper()}")
        print("="*60)
        
        try:
            # Evaluate on standard test split
            test_dataset = load_dataset(dataset_name, "test", data_dir, tokenizer)
            test_result = evaluate_model(
                model, tokenizer, test_dataset, device,
                max_samples=max_samples,
                batch_size=args.batch_size,
                dataset_type=dataset_name,
                baseline_type=args.baseline_type,
            )
            
            # Evaluate on OOD splits
            ood_results = evaluate_ood_splits(
                model, tokenizer, dataset_name, data_dir, device,
                max_samples=max_samples,
                baseline_type=args.baseline_type,
            )
            
            all_results["datasets"][dataset_name] = {
                "test": test_result,
                "ood_splits": ood_results,
            }
            
        except Exception as e:
            print(f"Error evaluating {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save and display results
    save_results(all_results, Path(args.output))
    print_results_table(all_results["datasets"])


if __name__ == "__main__":
    main()
