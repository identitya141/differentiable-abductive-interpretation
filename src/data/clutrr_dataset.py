"""
CLUTRR Dataset for Relational Reasoning

CLUTRR (Compositional Language Understanding with Text-based Relational Reasoning)
tests multi-hop relational reasoning with compositional generalization.

Reference: Sinha et al. (2019) "CLUTRR: A Diagnostic Benchmark for Inductive Reasoning from Text"

Key Features:
- Family relationship inference from text stories
- Train on k-hop chains, test on k+n-hop chains
- Requires inductive reasoning over relations

Compositional Structure:
- Entities: People with family relationships
- Relations: parent, child, sibling, spouse, etc.
- Composition: Transitive closure over relations

Why Relevant for DAI:
- Clear relational type system
- Monotonic chain reasoning (more hops = more complex)
- Type composition: daughter(parent) = self/sibling
"""

import csv
import glob
import os
from typing import Dict, List, Optional

from .base_dataset import BaseCompositionalDataset, CompositionalExample
from .clutrr_composition import (
    align_clutrr_composition_specs_to_tokens,
    extract_clutrr_composition_specs,
)


class CLUTRRDataset(BaseCompositionalDataset):
    """
    CLUTRR dataset loader.
    
    Tests compositional generalization in relational reasoning.
    """
    
    # Relation type system
    RELATION_TYPES = {
        "mother": "PARENT",
        "father": "PARENT",
        "son": "CHILD",
        "daughter": "CHILD",
        "brother": "SIBLING",
        "sister": "SIBLING",
        "husband": "SPOUSE",
        "wife": "SPOUSE",
        "grandfather": "GRANDPARENT",
        "grandmother": "GRANDPARENT",
        "grandson": "GRANDCHILD",
        "granddaughter": "GRANDCHILD",
        "uncle": "UNCLE_AUNT",
        "aunt": "UNCLE_AUNT",
        "nephew": "NIECE_NEPHEW",
        "niece": "NIECE_NEPHEW",
        "cousin": "COUSIN",
    }
    
    # Relation composition rules (simplified)
    COMPOSITION_RULES = {
        ("PARENT", "PARENT"): "GRANDPARENT",
        ("CHILD", "CHILD"): "GRANDCHILD",
        ("PARENT", "SIBLING"): "UNCLE_AUNT",
        ("CHILD", "SIBLING"): "NIECE_NEPHEW",
        ("SIBLING", "CHILD"): "NIECE_NEPHEW",
        ("PARENT", "CHILD"): "SIBLING",  # or self
    }
    
    def __init__(
        self,
        tokenizer,
        split: str = "train",
        train_hops: List[int] = None,  # Which hop counts for training
        test_hops: List[int] = None,   # Which hop counts for testing
        max_source_length: int = 256,
        max_target_length: int = 32,
        data_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        allow_synthetic: bool = False,
    ):
        self.train_hops = train_hops or [2, 3, 4]
        self.test_hops = test_hops or [5, 6, 7, 8, 9, 10]
        self.allow_synthetic = allow_synthetic
        
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
        return "CLUTRR"
    
    @property
    def task_description(self) -> str:
        return "Infer family relationships from text stories"
    
    def _load_data(self) -> List[CompositionalExample]:
        """Load CLUTRR data."""
        try:
            return self._load_from_local()
        except (FileNotFoundError, ValueError):
            if self.allow_synthetic:
                return self._generate_synthetic_examples()
            raise
    
    def _load_from_local(self) -> List[CompositionalExample]:
        """Load from local files."""
        if self.data_dir is None:
            raise ValueError("data_dir required")
        
        examples: List[CompositionalExample] = []
        hops = self.train_hops if self.split == "train" else self.test_hops
        candidates = sorted(
            glob.glob(os.path.join(self.data_dir, "**", "*.csv"), recursive=True)
        )
        split_files = [
            path for path in candidates if path.endswith(f"_{self.split}.csv")
        ]
        for file_path in split_files:
            with open(file_path, "r", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    task_name = row.get("task_name", "")
                    try:
                        hop_count = int(task_name.rsplit(".", 1)[1])
                    except (IndexError, ValueError):
                        continue
                    if hop_count not in hops:
                        continue
                    required = (
                        "story",
                        "text_query",
                        "text_target",
                        "story_edges",
                        "edge_types",
                        "query_edge",
                        "genders",
                    )
                    missing = [field for field in required if not row.get(field)]
                    if missing:
                        raise ValueError(
                            f"Official CLUTRR row is missing fields: {', '.join(missing)}"
                        )
                    story = row["story"]
                    input_text = f"{story} Question: {row['text_query']}"
                    examples.append(
                        CompositionalExample(
                            input_text=input_text,
                            original_input_text=story,
                            target_text=row["text_target"],
                            split=self.split,
                            generalization_category=f"k={hop_count}",
                            compositional_structure=f"hops:{hop_count}",
                            primitives=self._extract_relations(story),
                            composition_specs=list(
                                extract_clutrr_composition_specs(
                                    story,
                                    row["story_edges"],
                                    row["edge_types"],
                                    row["query_edge"],
                                    row["genders"],
                                )
                            ),
                        )
                    )

        if not examples:
            raise FileNotFoundError(
                f"No official CLUTRR {self.split!r} rows for hops {hops} under "
                f"{self.data_dir!r}"
            )
        return examples

    def _tokenize_all(self) -> List[Dict]:
        """Tokenize examples and align grounded relation chains."""
        tokenized = super()._tokenize_all()
        for example, encoded in zip(self.examples, tokenized):
            encoded["composition_specs"] = list(
                align_clutrr_composition_specs_to_tokens(
                    example.input_text,
                    example.composition_specs or [],
                    self.tokenizer,
                )
            )
        return tokenized
    
    def _generate_synthetic_examples(self) -> List[CompositionalExample]:
        """Generate synthetic examples for testing."""
        import random
        
        names = ["Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry"]
        relations = list(self.RELATION_TYPES.keys())
        
        examples = []
        hops = self.train_hops if self.split == "train" else self.test_hops
        
        num_examples = 1000 if self.split == "train" else 200
        
        for _ in range(num_examples):
            k = random.choice(hops)
            
            # Generate chain
            chain_names = random.sample(names, min(k + 1, len(names)))
            chain_rels = [random.choice(relations[:8]) for _ in range(k)]  # Basic relations
            
            # Build story
            story_parts = []
            for i in range(k):
                story_parts.append(f"{chain_names[i]} is the {chain_rels[i]} of {chain_names[i+1]}.")
            story = " ".join(story_parts)
            
            query = f"What is the relationship between {chain_names[0]} and {chain_names[-1]}?"
            target = self._compose_relations(chain_rels)
            
            examples.append(CompositionalExample(
                input_text=f"{story} {query}",
                target_text=target,
                split=self.split,
                compositional_structure=f"hops:{k}",
                primitives=chain_rels,
            ))
        
        return examples
    
    def _extract_relations(self, story: str) -> List[str]:
        """Extract relation mentions from story."""
        relations = []
        story_lower = story.lower()
        for rel in self.RELATION_TYPES:
            if rel in story_lower:
                relations.append(rel)
        return relations
    
    def _compose_relations(self, relations: List[str]) -> str:
        """Compose a chain of relations to get final relation."""
        if len(relations) == 0:
            return "unknown"
        if len(relations) == 1:
            return relations[0]
        
        # Simplified composition (in reality, need proper reasoning)
        current_type = self.RELATION_TYPES.get(relations[0], "UNKNOWN")
        
        for rel in relations[1:]:
            next_type = self.RELATION_TYPES.get(rel, "UNKNOWN")
            composed = self.COMPOSITION_RULES.get((current_type, next_type))
            if composed:
                current_type = composed
            else:
                current_type = "RELATIVE"  # Fallback
        
        # Map back to relation
        type_to_rel = {
            "GRANDPARENT": "grandparent",
            "GRANDCHILD": "grandchild",
            "UNCLE_AUNT": "uncle/aunt",
            "NIECE_NEPHEW": "niece/nephew",
            "SIBLING": "sibling",
            "COUSIN": "cousin",
            "RELATIVE": "relative",
        }
        
        return type_to_rel.get(current_type, "relative")
    
    def _get_compositional_structure(self, example: CompositionalExample) -> Dict:
        """Get compositional structure for analysis."""
        story = example.input_text
        
        structure = {
            "num_sentences": story.count("."),
            "relations": self._extract_relations(story),
            "relation_types": [self.RELATION_TYPES.get(r, "UNK") for r in self._extract_relations(story)],
            "hop_count": int(example.compositional_structure.split(":")[1]) if example.compositional_structure else 0,
        }
        
        return structure
    
    def get_type_vocabulary(self) -> Dict[str, int]:
        """Get type vocabulary."""
        all_types = set(self.RELATION_TYPES.values())
        return {t: i for i, t in enumerate(sorted(all_types))}
    
    def get_label_vocabulary(self) -> Dict[str, int]:
        """Get label vocabulary for classification."""
        all_relations = list(self.RELATION_TYPES.keys()) + ["relative", "unknown"]
        return {r: i for i, r in enumerate(sorted(all_relations))}


class CLUTRRDataModule:
    """Data module for CLUTRR experiments."""
    
    def __init__(
        self,
        tokenizer,
        train_hops: List[int] = None,
        test_hops: List[int] = None,
        batch_size: int = 32,
        max_source_length: int = 256,
        max_target_length: int = 32,
        num_workers: int = 4,
        data_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        allow_synthetic: bool = False,
    ):
        self.tokenizer = tokenizer
        self.train_hops = train_hops or [2, 3, 4]
        self.test_hops = test_hops or [5, 6, 7, 8, 9, 10]
        self.batch_size = batch_size
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.allow_synthetic = allow_synthetic
        
        self.train_dataset = None
        self.test_datasets = {}  # Separate dataset per test hop count
    
    def setup(self):
        """Initialize datasets."""
        self.train_dataset = CLUTRRDataset(
            tokenizer=self.tokenizer,
            split="train",
            train_hops=self.train_hops,
            max_source_length=self.max_source_length,
            max_target_length=self.max_target_length,
            data_dir=self.data_dir,
            cache_dir=self.cache_dir,
            allow_synthetic=self.allow_synthetic,
        )
        
        # Create separate test set for each hop count
        for k in self.test_hops:
            self.test_datasets[f"k={k}"] = CLUTRRDataset(
                tokenizer=self.tokenizer,
                split="test",
                test_hops=[k],
                max_source_length=self.max_source_length,
                max_target_length=self.max_target_length,
                data_dir=self.data_dir,
                cache_dir=self.cache_dir,
                allow_synthetic=self.allow_synthetic,
            )
    
    def train_dataloader(self):
        return self.train_dataset.get_dataloader(
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
    
    def test_dataloader(self, hop_count: Optional[int] = None):
        """Get test dataloader for specific hop count or all."""
        if hop_count is not None:
            key = f"k={hop_count}"
            if key in self.test_datasets:
                return self.test_datasets[key].get_dataloader(
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                )
        
        # Return all combined
        from torch.utils.data import ConcatDataset
        all_test = ConcatDataset(list(self.test_datasets.values()))
        return DataLoader(
            all_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
