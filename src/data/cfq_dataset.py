"""
CFQ Dataset for Compositional Generalization

CFQ (Compositional Freebase Questions) tests compositional generalization
in semantic parsing for question answering.

Reference: Keysers et al. (2020) "Measuring Compositional Generalization: A Comprehensive Method on Realistic Data"

Key Features:
- Question answering with SPARQL queries
- Maximum Compound Divergence (MCD) splits for controlled difficulty
- Based on Freebase knowledge graph

MCD Splits:
- MCD1: Low divergence (easier)
- MCD2: Medium divergence
- MCD3: High divergence (hardest)

Compositional Structure:
- Entities, relations, variables
- Complex query patterns with joins, filters
- Systematic recombination of query fragments

Why Relevant for DAI:
- Query structure maps to abstract types
- Relational composition (joins) is central
- Clear monotonicity in aggregation operations
"""

import os
import random
from typing import Dict, List, Optional

from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader, Subset

from .base_dataset import BaseCompositionalDataset, CompositionalExample
from .cfq_composition import (
    align_cfq_composition_specs_to_tokens,
    extract_cfq_composition_specs,
)
from .composition_controls import transform_aligned_specs


class CFQDataset(BaseCompositionalDataset):
    """
    CFQ dataset loader.
    
    Supports MCD1, MCD2, MCD3 splits for varying compositional difficulty.
    """
    
    # CFQ type system (simplified)
    TOKEN_TYPES = {
        # Question words
        "who": "WH",
        "what": "WH",
        "which": "WH",
        "where": "WH",
        "when": "WH",
        "how": "WH",
        # Verbs
        "did": "AUX",
        "was": "AUX",
        "were": "AUX",
        "is": "AUX",
        "are": "AUX",
        "directed": "VERB",
        "wrote": "VERB",
        "produced": "VERB",
        "starred": "VERB",
        "appeared": "VERB",
        # Prepositions
        "in": "PREP",
        "by": "PREP",
        "with": "PREP",
        "that": "REL",
        "and": "CONJ",
    }
    
    # SPARQL types
    SPARQL_TYPES = {
        "SELECT": "SELECT",
        "WHERE": "WHERE",
        "FILTER": "FILTER",
        "?x": "VAR",
        "ns:": "NAMESPACE",
        ".": "TRIPLE_END",
        "{": "BLOCK_START",
        "}": "BLOCK_END",
    }
    
    AVAILABLE_SPLITS = ["mcd1", "mcd2", "mcd3"]
    
    def __init__(
        self,
        tokenizer,
        split: str = "train",
        cfq_split: str = "mcd1",  # MCD difficulty level
        max_source_length: int = 128,
        max_target_length: int = 256,
        data_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        composition_structure_mode: str = "grounded",
        seed: int = 42,
    ):
        self.cfq_split = cfq_split
        self.composition_structure_mode = composition_structure_mode
        self.seed = seed
        if cfq_split not in self.AVAILABLE_SPLITS:
            raise ValueError(f"Unknown CFQ split: {cfq_split}")
        
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
        return f"CFQ-{self.cfq_split.upper()}"
    
    @property
    def task_description(self) -> str:
        return "Parse natural language questions to SPARQL queries"
    
    def _load_data(self) -> List[CompositionalExample]:
        """Load CFQ data from HuggingFace."""
        if self.data_dir is not None:
            dataset_path = os.path.join(self.data_dir, self.cfq_split)
            if os.path.isfile(os.path.join(dataset_path, "dataset_dict.json")):
                dataset = load_from_disk(dataset_path)
            else:
                return self._load_from_local()
        else:
            dataset = load_dataset(
                "cfq",
                self.cfq_split,
                cache_dir=self.cache_dir,
            )
        
        split_data = dataset[self.split]
        examples = []
        
        for item in split_data:
            question = item["question"]
            query = item["query"]
            
            examples.append(CompositionalExample(
                input_text=question,
                target_text=query,
                split=self.split,
                primitives=self._extract_primitives(question),
                compositional_structure=self._analyze_query(query),
                composition_specs=list(extract_cfq_composition_specs(question, query)),
            ))
        
        return examples
    
    def _load_from_local(self) -> List[CompositionalExample]:
        """Load from local files."""
        if self.data_dir is None:
            raise ValueError("data_dir required for local loading")
        
        # CFQ uses JSON format locally
        import json
        
        file_path = os.path.join(
            self.data_dir,
            self.cfq_split,
            f"{self.split}.json"
        )
        
        examples = []
        with open(file_path, 'r') as f:
            data = json.load(f)
            for item in data:
                examples.append(CompositionalExample(
                    input_text=item["question"],
                    target_text=item["query"],
                    split=self.split,
                    primitives=self._extract_primitives(item["question"]),
                    compositional_structure=self._analyze_query(item["query"]),
                    composition_specs=list(
                        extract_cfq_composition_specs(
                            item["question"], item["query"]
                        )
                    ),
                ))
        
        return examples

    def _tokenize_all(self) -> List[Dict]:
        """Tokenize examples and align grounded query relations."""
        tokenized = super()._tokenize_all()
        for example, encoded in zip(self.examples, tokenized):
            aligned = align_cfq_composition_specs_to_tokens(
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
    
    def _extract_primitives(self, question: str) -> List[str]:
        """Extract primitive tokens."""
        tokens = question.lower().split()
        return [t for t in tokens if t in self.TOKEN_TYPES]
    
    def _analyze_query(self, query: str) -> str:
        """Analyze SPARQL query structure."""
        # Count query components
        num_vars = query.count("?x")
        num_triples = query.count(" .")
        has_filter = "FILTER" in query
        has_union = "UNION" in query
        
        return f"vars:{num_vars},triples:{num_triples},filter:{has_filter},union:{has_union}"
    
    def _get_compositional_structure(self, example: CompositionalExample) -> Dict:
        """Get compositional structure for analysis."""
        question = example.input_text.lower()
        query = example.target_text
        
        structure = {
            "question_tokens": question.split(),
            "question_types": [self.TOKEN_TYPES.get(t, "UNK") for t in question.split()],
            "num_clauses": question.count(" and ") + question.count(" that ") + 1,
            "query_vars": query.count("?x"),
            "query_triples": query.count(" ."),
            "has_filter": "FILTER" in query,
            "has_union": "UNION" in query,
            "has_optional": "OPTIONAL" in query,
            "complexity": "simple" if query.count(" .") <= 2 else "complex",
        }
        
        return structure
    
    def get_type_vocabulary(self) -> Dict[str, int]:
        """Get type vocabulary."""
        all_types = set(self.TOKEN_TYPES.values()) | set(self.SPARQL_TYPES.values())
        return {t: i for i, t in enumerate(sorted(all_types))}


class CFQDataModule:
    """Data module for CFQ experiments."""
    
    def __init__(
        self,
        tokenizer,
        cfq_split: str = "mcd1",
        batch_size: int = 32,
        max_source_length: int = 128,
        max_target_length: int = 256,
        num_workers: int = 4,
        cache_dir: Optional[str] = None,
        data_dir: Optional[str] = None,
        validation_fraction: float = 0.1,
        seed: int = 42,
        composition_structure_mode: str = "grounded",
    ):
        self.tokenizer = tokenizer
        self.cfq_split = cfq_split
        self.batch_size = batch_size
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.num_workers = num_workers
        self.cache_dir = cache_dir
        self.data_dir = data_dir
        if not 0.0 < validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1")
        self.validation_fraction = validation_fraction
        self.seed = seed
        self.composition_structure_mode = composition_structure_mode
        
        self.train_dataset = None
        self.validation_dataset = None
        self.test_dataset = None
    
    def setup(self):
        """Initialize datasets."""
        self.train_dataset = CFQDataset(
            tokenizer=self.tokenizer,
            split="train",
            cfq_split=self.cfq_split,
            max_source_length=self.max_source_length,
            max_target_length=self.max_target_length,
            cache_dir=self.cache_dir,
            data_dir=self.data_dir,
            composition_structure_mode=self.composition_structure_mode,
            seed=self.seed,
        )

        indices = list(range(len(self.train_dataset)))
        random.Random(self.seed).shuffle(indices)
        validation_size = max(1, round(len(indices) * self.validation_fraction))
        validation_indices = indices[:validation_size]
        training_indices = indices[validation_size:]
        if not training_indices:
            raise ValueError("CFQ training split is too small for validation holdout")
        full_train_dataset = self.train_dataset
        self.train_dataset = Subset(full_train_dataset, training_indices)
        self.validation_dataset = Subset(full_train_dataset, validation_indices)
        
        self.test_dataset = CFQDataset(
            tokenizer=self.tokenizer,
            split="test",
            cfq_split=self.cfq_split,
            max_source_length=self.max_source_length,
            max_target_length=self.max_target_length,
            cache_dir=self.cache_dir,
            data_dir=self.data_dir,
            composition_structure_mode=self.composition_structure_mode,
            seed=self.seed,
        )
    
    def train_dataloader(self):
        return self._subset_dataloader(self.train_dataset, shuffle=True)

    def validation_dataloader(self):
        return self._subset_dataloader(self.validation_dataset, shuffle=False)
    
    def test_dataloader(self):
        return self.test_dataset.get_dataloader(
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def _subset_dataloader(self, dataset: Subset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=dataset.dataset.collate_fn,
        )
