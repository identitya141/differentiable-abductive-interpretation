"""
SCAN Dataset for Compositional Generalization

SCAN (Simplified Commands for Auditory Navigation) is a compositional
command-to-action translation benchmark.

Reference: Lake & Baroni (2018) "Generalization without Systematicity"

Key Splits:
- Simple: Random train/test split (sanity check)
- Length: Train on short commands, test on long commands
- Add primitive jump: Train without "jump", test with "jump"
- Add primitive turn_left: Similar to above
- Template splits: Systematic template-based splits

Compositional Structure:
- Commands: Combinations of {walk, look, run, jump} × {left, right, around} × {twice, thrice}
- Actions: Sequences of primitive actions {WALK, LOOK, RUN, JUMP, LTURN, RTURN}

Why Relevant for DAI:
- Clear compositional structure amenable to abstract interpretation
- Types: {action, direction, modifier}
- Composition rules are deterministic and learnable
"""

import os
import re
from typing import Dict, List, Optional

from datasets import load_dataset
from torch.utils.data import DataLoader, Subset
from src.utils.benchmark_contract import paired_holdout_indices

from .base_dataset import BaseCompositionalDataset, CompositionalExample
from .scan_composition import (
    align_composition_specs_to_tokens,
    extract_composition_specs,
    linearize_scan_command,
    parse_scan_command,
    replace_scan_primitives_with_nonce_words,
    transform_composition_specs,
)


class SCANDataset(BaseCompositionalDataset):
    """
    SCAN dataset loader.
    
    Supports multiple compositional splits for thorough evaluation.
    """
    
    # SCAN type system (for abstract interpretation)
    TOKEN_TYPES = {
        # Actions
        "walk": "ACTION",
        "look": "ACTION",
        "run": "ACTION",
        "jump": "ACTION",
        # Directions
        "left": "DIRECTION",
        "right": "DIRECTION",
        "around": "MODIFIER",  # Actually a modifier
        "opposite": "MODIFIER",
        # Quantifiers
        "twice": "QUANTIFIER",
        "thrice": "QUANTIFIER",
        # Connectives
        "and": "CONNECTIVE",
        "after": "CONNECTIVE",
        # Turn is special
        "turn": "TURN_ACTION",
    }
    
    # Output action types
    OUTPUT_TYPES = {
        "WALK": "OUT_ACTION",
        "LOOK": "OUT_ACTION",
        "RUN": "OUT_ACTION",
        "JUMP": "OUT_ACTION",
        "LTURN": "OUT_TURN",
        "RTURN": "OUT_TURN",
    }
    
    # Valid composition rules (type1, type2) -> result_type
    COMPOSITION_RULES = {
        ("TURN_ACTION", "DIRECTION"): "TURN_PHRASE",
        ("ACTION", "MODIFIER"): "MODIFIED_ACTION",
        ("MODIFIED_ACTION", "DIRECTION"): "DIRECTIONAL_ACTION",
        ("ACTION", "QUANTIFIER"): "REPEATED_ACTION",
    }
    
    # Available splits in SCAN
    AVAILABLE_SPLITS = {
        "simple": "simple",
        "length": "length",
        "addprim_jump": "addprim_jump",
        "addprim_turn_left": "addprim_turn_left",
        "template_around_right": "template_around_right",
        "template_jump_around_right": "template_jump_around_right",
        "template_right": "template_right",
        "template_opposite_right": "template_opposite_right",
    }
    
    def __init__(
        self,
        tokenizer,
        split: str = "train",
        scan_split: str = "length",  # Which SCAN compositional split
        max_source_length: int = 64,
        max_target_length: int = 128,
        data_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        composition_structure_mode: str = "grounded",
        structure_corruption_probability: float = 0.0,
        input_representation: str = "plain",
        nonce_primitives: bool = False,
        seed: int = 42,
    ):
        """
        Initialize SCAN dataset.
        
        Args:
            tokenizer: Tokenizer
            split: "train" or "test"
            scan_split: Which SCAN split to use (length, addprim_jump, etc.)
            max_source_length: Max input length
            max_target_length: Max output length
            data_dir: Data directory
            cache_dir: Cache directory
        """
        self.scan_split = scan_split
        self.composition_structure_mode = composition_structure_mode
        self.structure_corruption_probability = structure_corruption_probability
        self.input_representation = input_representation
        self.nonce_primitives = nonce_primitives
        self.seed = seed
        if scan_split not in self.AVAILABLE_SPLITS:
            raise ValueError(f"Unknown SCAN split: {scan_split}. Available: {list(self.AVAILABLE_SPLITS.keys())}")
        if input_representation not in {"plain", "tree_linearized"}:
            raise ValueError(f"Unknown SCAN input representation: {input_representation}")
        if not 0.0 <= structure_corruption_probability <= 1.0:
            raise ValueError("structure_corruption_probability must be between 0 and 1")
        if input_representation == "tree_linearized" and composition_structure_mode != "grounded":
            raise ValueError("tree-linearized inputs cannot use span-structure controls")
        
        super().__init__(
            tokenizer=tokenizer,
            split=split,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            data_dir=data_dir,
            cache_dir=cache_dir,
        )
    
    @property
    def dataset_name(self) -> str:
        return f"SCAN-{self.scan_split}"
    
    @property
    def task_description(self) -> str:
        return "Translate natural language navigation commands to action sequences"
    
    def _load_data(self) -> List[CompositionalExample]:
        """Load SCAN data from HuggingFace datasets."""
        if self.data_dir is not None:
            local_path = os.path.join(
                self.data_dir,
                self.scan_split,
                f"tasks_{self.split}_{self.scan_split}.txt",
            )
            if os.path.isfile(local_path):
                return self._load_from_local()

        try:
            # Load from HuggingFace
            dataset = load_dataset(
                "scan",
                self.scan_split,
                cache_dir=self.cache_dir,
            data_dir=self.data_dir,
            )
        except Exception as e:
            print(f"Warning: Could not load SCAN from HuggingFace: {e}")
            print("Attempting to load from local files...")
            return self._load_from_local()
        
        examples = []
        split_data = dataset[self.split]
        
        for item in split_data:
            command = item["commands"]
            actions = item["actions"]
            
            # Extract primitives and structure
            primitives = self._extract_primitives(command)
            structure = self._parse_command(command)
            composition_specs = list(
                extract_composition_specs(parse_scan_command(command))
            )
            
            examples.append(CompositionalExample(
                input_text=self._prepare_input_text(command),
                target_text=actions,
                original_input_text=command,
                split=self.split,
                compositional_structure=structure,
                primitives=primitives,
                novel_compositions=self._find_novel_compositions(command) if self.split == "test" else None,
                composition_specs=(
                    [] if self.input_representation == "tree_linearized"
                    else composition_specs
                ),
            ))
        
        return examples
    
    def _load_from_local(self) -> List[CompositionalExample]:
        """Load from local TSV files."""
        if self.data_dir is None:
            raise ValueError("data_dir required for local loading")
        
        file_path = os.path.join(
            self.data_dir,
            self.scan_split,
            f"tasks_{self.split}_{self.scan_split}.txt"
        )
        
        examples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith("IN:"):
                    parts = line.split("OUT:")
                    command = parts[0].replace("IN:", "").strip()
                    actions = parts[1].strip() if len(parts) > 1 else ""
                    composition_specs = list(
                        extract_composition_specs(parse_scan_command(command))
                    )
                    
                    examples.append(CompositionalExample(
                        input_text=self._prepare_input_text(command),
                        target_text=actions,
                        original_input_text=command,
                        split=self.split,
                        primitives=self._extract_primitives(command),
                        compositional_structure=self._parse_command(command),
                        composition_specs=(
                            [] if self.input_representation == "tree_linearized"
                            else composition_specs
                        ),
                    ))
        
        return examples
    
    def _extract_primitives(self, command: str) -> List[str]:
        """Extract primitive tokens from command."""
        tokens = command.lower().split()
        return [t for t in tokens if t in self.TOKEN_TYPES]

    def _prepare_input_text(self, command: str) -> str:
        if self.input_representation == "tree_linearized":
            return linearize_scan_command(command)
        if self.nonce_primitives:
            return replace_scan_primitives_with_nonce_words(command)
        return command

    def _tokenize_all(self) -> List[Dict]:
        """Tokenize examples and align grammar word spans to encoder tokens."""
        tokenized = super()._tokenize_all()
        for example, encoded in zip(self.examples, tokenized):
            if not example.composition_specs:
                encoded["composition_specs"] = []
                continue
            encoded["composition_specs"] = list(
                transform_composition_specs(
                    align_composition_specs_to_tokens(
                        example.input_text,
                        example.composition_specs or [],
                        self.tokenizer,
                    ),
                    token_count=int(encoded["attention_mask"].sum().item()),
                    mode=self.composition_structure_mode,
                    seed=f"{self.seed}:{example.input_text}",
                    corruption_probability=self.structure_corruption_probability,
                )
            )
        return tokenized
    
    def _parse_command(self, command: str) -> str:
        """
        Parse command into compositional structure.
        
        Returns a string representation of the parse tree.
        """
        tokens = command.lower().split()
        types = [self.TOKEN_TYPES.get(t, "UNKNOWN") for t in tokens]
        return " ".join(f"{t}:{typ}" for t, typ in zip(tokens, types))
    
    def _find_novel_compositions(self, command: str) -> List[str]:
        """
        Identify which compositions in this command are novel.
        
        For length split: Commands with more primitives
        For addprim: Commands containing the held-out primitive
        """
        novel = []
        tokens = command.lower().split()
        
        if self.scan_split == "addprim_jump" and "jump" in tokens:
            novel.append("jump_composition")
        elif self.scan_split == "length" and len(tokens) > 6:
            novel.append("long_sequence")
        elif "around" in tokens and "right" in tokens:
            novel.append("around_right")
        
        return novel
    
    def _get_compositional_structure(self, example: CompositionalExample) -> Dict:
        """Get compositional structure for DAI analysis."""
        command = example.input_text.lower()
        tokens = command.split()
        
        structure = {
            "tokens": tokens,
            "types": [self.TOKEN_TYPES.get(t, "UNKNOWN") for t in tokens],
            "length": len(tokens),
            "has_modifier": any(t in ["around", "opposite"] for t in tokens),
            "has_quantifier": any(t in ["twice", "thrice"] for t in tokens),
            "has_connective": any(t in ["and", "after"] for t in tokens),
        }
        
        # Count composition depth
        depth = 1
        if structure["has_modifier"]:
            depth += 1
        if structure["has_quantifier"]:
            depth += 1
        if structure["has_connective"]:
            depth += 1
        structure["composition_depth"] = depth
        
        return structure
    
    def get_type_vocabulary(self) -> Dict[str, int]:
        """Get mapping from types to indices for abstract interpretation."""
        all_types = set(self.TOKEN_TYPES.values()) | set(self.OUTPUT_TYPES.values())
        return {t: i for i, t in enumerate(sorted(all_types))}


class SCANDataModule:
    """
    Data module for SCAN experiments.
    
    Manages multiple SCAN splits for comprehensive evaluation.
    """
    
    def __init__(
        self,
        tokenizer,
        scan_split: str = "length",
        batch_size: int = 32,
        max_source_length: int = 64,
        max_target_length: int = 128,
        num_workers: int = 4,
        eval_batch_size: Optional[int] = None,
        eval_num_workers: int = 0,
        cache_dir: Optional[str] = None,
        data_dir: Optional[str] = None,
        validation_fraction: float = 0.1,
        composition_structure_mode: str = "grounded",
        structure_corruption_probability: float = 0.0,
        input_representation: str = "plain",
        nonce_primitives: bool = False,
        seed: int = 42,
        split_seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.scan_split = scan_split
        self.batch_size = batch_size
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.num_workers = num_workers
        self.eval_batch_size = eval_batch_size or batch_size
        self.eval_num_workers = eval_num_workers
        self.cache_dir = cache_dir
        self.data_dir = data_dir
        self.composition_structure_mode = composition_structure_mode
        self.structure_corruption_probability = structure_corruption_probability
        self.input_representation = input_representation
        self.nonce_primitives = nonce_primitives
        if not 0.0 < validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1")
        self.validation_fraction = validation_fraction
        self.seed = seed
        self.split_seed = split_seed
        
        self.train_dataset = None
        self.validation_dataset = None
        self.test_dataset = None
    
    def setup(self):
        """Initialize datasets."""
        full_train_dataset = SCANDataset(
            tokenizer=self.tokenizer,
            split="train",
            scan_split=self.scan_split,
            max_source_length=self.max_source_length,
            max_target_length=self.max_target_length,
            cache_dir=self.cache_dir,
            data_dir=self.data_dir,
            composition_structure_mode=self.composition_structure_mode,
            structure_corruption_probability=self.structure_corruption_probability,
            input_representation=self.input_representation,
            nonce_primitives=self.nonce_primitives,
            seed=self.seed,
        )

        training_indices, validation_indices = paired_holdout_indices(
            len(full_train_dataset), self.validation_fraction, self.split_seed
        )

        self.train_dataset = Subset(full_train_dataset, training_indices)
        self.validation_dataset = Subset(full_train_dataset, validation_indices)
        
        self.test_dataset = SCANDataset(
            tokenizer=self.tokenizer,
            split="test",
            scan_split=self.scan_split,
            max_source_length=self.max_source_length,
            max_target_length=self.max_target_length,
            cache_dir=self.cache_dir,
            data_dir=self.data_dir,
            composition_structure_mode=self.composition_structure_mode,
            structure_corruption_probability=self.structure_corruption_probability,
            input_representation=self.input_representation,
            nonce_primitives=self.nonce_primitives,
            seed=self.seed,
        )
    
    def train_dataloader(self):
        return self._subset_dataloader(self.train_dataset, shuffle=True)

    def validation_dataloader(self):
        return self._subset_dataloader(
            self.validation_dataset, shuffle=False, evaluation=True
        )
    
    def test_dataloader(self):
        return self.test_dataset.get_dataloader(
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.eval_num_workers,
        )

    def _subset_dataloader(
        self, dataset: Subset, shuffle: bool, evaluation: bool = False
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.eval_batch_size if evaluation else self.batch_size,
            shuffle=shuffle,
            num_workers=self.eval_num_workers if evaluation else self.num_workers,
            collate_fn=dataset.dataset.collate_fn,
        )
    
    @property
    def num_train_examples(self) -> int:
        return len(self.train_dataset) if self.train_dataset else 0
    
    @property
    def num_test_examples(self) -> int:
        return len(self.test_dataset) if self.test_dataset else 0
