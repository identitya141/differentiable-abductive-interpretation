"""
COGS Dataset for Compositional Generalization

COGS (Compositional Generalization Challenge based on Semantic Interpretation)
tests systematic generalization in semantic parsing.

Reference: Kim & Linzen (2020) "COGS: A Compositional Generalization Challenge Based on Semantic Interpretation"

Key Features:
- Semantic parsing from English to logical forms
- Systematic train/test split based on linguistic constructions
- Tests novel combinations of known primitives

Compositional Structure:
- Nouns, verbs, prepositions, determiners
- Novel combinations: new noun-verb pairs, new PP-attachments
- Requires understanding of semantic roles (agent, theme, recipient)

Why Relevant for DAI:
- Rich linguistic type system (NP, VP, PP, etc.)
- Clear semantic composition rules
- Tests systematic recombination of learned primitives
"""

import os
from typing import Dict, List, Optional

from datasets import load_dataset

from .base_dataset import BaseCompositionalDataset, CompositionalExample
from .cogs_composition import (
    align_cogs_composition_specs_to_tokens,
    extract_cogs_composition_specs,
)
from .composition_controls import transform_aligned_specs


class COGSDataset(BaseCompositionalDataset):
    """
    COGS dataset loader.
    
    Provides semantic parsing examples with compositional splits.
    """
    
    # COGS type system
    TOKEN_TYPES = {
        # Determiners
        "a": "DET",
        "the": "DET",
        "every": "DET",
        "no": "DET",
        # Common nouns
        "cat": "NOUN",
        "dog": "NOUN",
        "girl": "NOUN",
        "boy": "NOUN",
        "cake": "NOUN",
        "box": "NOUN",
        # Verbs
        "saw": "VERB",
        "helped": "VERB",
        "liked": "VERB",
        "gave": "VERB",
        "put": "VERB",
        "wanted": "VERB",
        # Prepositions
        "on": "PREP",
        "in": "PREP",
        "beside": "PREP",
        "to": "PREP",
        # Proper nouns
        "Emma": "PROPN",
        "Liam": "PROPN",
        "Olivia": "PROPN",
    }
    
    # Logical form types
    LF_TYPES = {
        "*": "STAR",
        ".": "DOT",
        ";": "SEMICOLON",
        "AND": "AND",
        "agent": "ROLE",
        "theme": "ROLE",
        "recipient": "ROLE",
        "ccomp": "ROLE",
        "xcomp": "ROLE",
        "nmod": "MODIFIER",
    }
    
    def __init__(
        self,
        tokenizer,
        split: str = "train",
        max_source_length: int = 128,
        max_target_length: int = 256,
        data_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        composition_structure_mode: str = "grounded",
        seed: int = 42,
    ):
        self.composition_structure_mode = composition_structure_mode
        self.seed = seed
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
        return "COGS"
    
    @property
    def task_description(self) -> str:
        return "Parse English sentences to logical forms"
    
    def _load_data(self) -> List[CompositionalExample]:
        """Load COGS data from HuggingFace."""
        if self.data_dir is not None:
            return self._load_from_local()

        try:
            dataset = load_dataset(
                "cogs",
                cache_dir=self.cache_dir,
            )
        except Exception as e:
            print(f"Warning: Could not load COGS from HuggingFace: {e}")
            return self._load_from_local()
        
        # COGS has train, dev, test, gen (generalization) splits
        if self.split == "test":
            split_data = dataset["gen"]  # Use generalization split for OOD
        elif self.split in {"dev", "iid_test"}:
            split_data = dataset["test" if self.split == "iid_test" else "dev"]
        else:
            split_data = dataset["train"]
        
        examples = []
        for item in split_data:
            sentence = item["sentence"]
            logical_form = item["logical_form"]
            category = item.get("type", item.get("category"))
            
            examples.append(CompositionalExample(
                input_text=sentence,
                target_text=logical_form,
                split=self.split,
                is_ood=self.split == "test",
                generalization_category=category,
                primitives=self._extract_primitives(sentence),
                compositional_structure=self._parse_structure(sentence, logical_form),
                composition_specs=list(
                    extract_cogs_composition_specs(sentence, logical_form)
                ),
            ))
        
        return examples
    
    def _load_from_local(self) -> List[CompositionalExample]:
        """Load from local TSV files."""
        if self.data_dir is None:
            raise ValueError("data_dir required for local loading")
        
        file_map = {
            "train": "train.tsv",
            "dev": "dev.tsv",
            "iid_test": "test.tsv",
            "test": "gen.tsv",  # Generalization split
        }
        
        file_path = os.path.join(self.data_dir, file_map[self.split])
        examples = []
        
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    sentence = parts[0]
                    logical_form = parts[1]
                    category = parts[2] if len(parts) >= 3 else None
                    
                    examples.append(CompositionalExample(
                        input_text=sentence,
                        target_text=logical_form,
                        split=self.split,
                        is_ood=self.split == "test",
                        generalization_category=category,
                        primitives=self._extract_primitives(sentence),
                        compositional_structure=self._parse_structure(
                            sentence, logical_form
                        ),
                        composition_specs=list(
                            extract_cogs_composition_specs(sentence, logical_form)
                        ),
                    ))
        
        return examples

    def _tokenize_all(self) -> List[Dict]:
        """Tokenize examples and align logical-form roles to encoder tokens."""
        tokenized = super()._tokenize_all()
        for example, encoded in zip(self.examples, tokenized):
            aligned = align_cogs_composition_specs_to_tokens(
                    example.input_text,
                    example.composition_specs or [],
                    self.tokenizer,
                )
            encoded["composition_specs"] = list(
                transform_aligned_specs(
                    aligned,
                    token_count=int(encoded["attention_mask"].sum().item()),
                    mode=self.composition_structure_mode,
                    seed=f"{self.seed}:{example.input_text}",
                )
            )
        return tokenized
    
    def _extract_primitives(self, sentence: str) -> List[str]:
        """Extract primitive tokens."""
        tokens = sentence.lower().split()
        return [t for t in tokens if t in self.TOKEN_TYPES]
    
    def _parse_structure(self, sentence: str, logical_form: str) -> str:
        """Parse compositional structure."""
        tokens = sentence.lower().split()
        types = [self.TOKEN_TYPES.get(t, "UNK") for t in tokens]
        return " ".join(f"{t}:{typ}" for t, typ in zip(tokens, types))
    
    def _get_compositional_structure(self, example: CompositionalExample) -> Dict:
        """Get compositional structure for analysis."""
        sentence = example.input_text.lower()
        lf = example.target_text
        
        tokens = sentence.split()
        
        structure = {
            "tokens": tokens,
            "types": [self.TOKEN_TYPES.get(t, "UNK") for t in tokens],
            "num_nouns": sum(1 for t in tokens if self.TOKEN_TYPES.get(t) == "NOUN"),
            "num_verbs": sum(1 for t in tokens if self.TOKEN_TYPES.get(t) == "VERB"),
            "num_preps": sum(1 for t in tokens if self.TOKEN_TYPES.get(t) == "PREP"),
            "has_embedded_clause": "ccomp" in lf or "xcomp" in lf,
            "num_entities": lf.count("*"),
        }
        
        return structure
    
    def get_type_vocabulary(self) -> Dict[str, int]:
        """Get type vocabulary for abstract interpretation."""
        all_types = set(self.TOKEN_TYPES.values()) | set(self.LF_TYPES.values())
        return {t: i for i, t in enumerate(sorted(all_types))}


class COGSDataModule:
    """Data module for COGS experiments."""
    
    def __init__(
        self,
        tokenizer,
        batch_size: int = 32,
        max_source_length: int = 128,
        max_target_length: int = 256,
        num_workers: int = 4,
        cache_dir: Optional[str] = None,
        data_dir: Optional[str] = None,
        composition_structure_mode: str = "grounded",
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.num_workers = num_workers
        self.cache_dir = cache_dir
        self.data_dir = data_dir
        self.composition_structure_mode = composition_structure_mode
        self.seed = seed
        
        self.train_dataset = None
        self.dev_dataset = None
        self.iid_test_dataset = None
        self.test_dataset = None
    
    def setup(self):
        """Initialize datasets."""
        self.train_dataset = COGSDataset(
            tokenizer=self.tokenizer,
            split="train",
            max_source_length=self.max_source_length,
            max_target_length=self.max_target_length,
            cache_dir=self.cache_dir,
            data_dir=self.data_dir,
            composition_structure_mode=self.composition_structure_mode,
            seed=self.seed,
        )
        
        self.dev_dataset = COGSDataset(
            tokenizer=self.tokenizer,
            split="dev",
            max_source_length=self.max_source_length,
            max_target_length=self.max_target_length,
            cache_dir=self.cache_dir,
            data_dir=self.data_dir,
            composition_structure_mode=self.composition_structure_mode,
            seed=self.seed,
        )

        self.iid_test_dataset = COGSDataset(
            tokenizer=self.tokenizer,
            split="iid_test",
            max_source_length=self.max_source_length,
            max_target_length=self.max_target_length,
            cache_dir=self.cache_dir,
            data_dir=self.data_dir,
            composition_structure_mode=self.composition_structure_mode,
            seed=self.seed,
        )
        
        self.test_dataset = COGSDataset(
            tokenizer=self.tokenizer,
            split="test",
            max_source_length=self.max_source_length,
            max_target_length=self.max_target_length,
            cache_dir=self.cache_dir,
            data_dir=self.data_dir,
            composition_structure_mode=self.composition_structure_mode,
            seed=self.seed,
        )
    
    def train_dataloader(self):
        return self.train_dataset.get_dataloader(
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
    
    def val_dataloader(self):
        return self.dev_dataset.get_dataloader(
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def validation_dataloader(self):
        return self.val_dataloader()

    def iid_test_dataloader(self):
        return self.iid_test_dataset.get_dataloader(
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
    
    def test_dataloader(self):
        return self.test_dataset.get_dataloader(
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
