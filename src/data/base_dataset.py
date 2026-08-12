"""
Base Dataset Classes for DAI Experiments

Provides common functionality for all compositional generalization benchmarks.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer


@dataclass
class CompositionalExample:
    """
    Single example from a compositional generalization dataset.
    
    Attributes:
        input_text: Source sequence (e.g., "jump around right twice")
        target_text: Target sequence (e.g., "RTURN RTURN RTURN RTURN JUMP ...")
        split: Which split this example belongs to
        compositional_structure: Optional parse/structure information
        primitives: List of primitives in this example
        novel_compositions: Which compositions are novel (for OOD examples)
    """
    input_text: str
    target_text: str
    original_input_text: Optional[str] = None
    split: str = "train"
    is_ood: Optional[bool] = None
    generalization_category: Optional[str] = None
    compositional_structure: Optional[str] = None
    primitives: Optional[List[str]] = None
    novel_compositions: Optional[List[str]] = None
    composition_specs: Optional[List[Any]] = None


@dataclass
class CompositionalBatch:
    """
    Batched data for compositional generalization tasks.
    """
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    decoder_input_ids: Optional[torch.Tensor] = None
    decoder_attention_mask: Optional[torch.Tensor] = None
    
    # Metadata
    input_texts: Optional[List[str]] = None
    original_input_texts: Optional[List[str]] = None
    target_texts: Optional[List[str]] = None
    is_ood: Optional[torch.Tensor] = None  # 1 if out-of-distribution
    generalization_categories: Optional[List[Optional[str]]] = None
    composition_depths: Optional[List[Optional[int]]] = None
    composition_specs: Optional[List[List[Any]]] = None


class BaseCompositionalDataset(Dataset, ABC):
    """
    Abstract base class for compositional generalization datasets.
    
    Subclasses must implement:
    - _load_data(): Load and parse the dataset
    - _get_compositional_structure(): Extract compositional info from examples
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        split: str = "train",
        max_source_length: int = 128,
        max_target_length: int = 128,
        data_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize dataset.
        
        Args:
            tokenizer: HuggingFace tokenizer
            split: Which split to load ("train", "test", etc.)
            max_source_length: Maximum source sequence length
            max_target_length: Maximum target sequence length
            data_dir: Directory containing data files
            cache_dir: Directory for caching processed data
        """
        self.tokenizer = tokenizer
        self.split = split
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        
        # Load data
        self.examples: List[CompositionalExample] = self._load_data()
        
        # Tokenize examples
        self._tokenized_examples = self._tokenize_all()
    
    @abstractmethod
    def _load_data(self) -> List[CompositionalExample]:
        """Load and return examples for this split."""
        pass
    
    @abstractmethod
    def _get_compositional_structure(self, example: CompositionalExample) -> Dict:
        """Extract compositional structure information from example."""
        pass
    
    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """Return dataset name."""
        pass
    
    @property
    @abstractmethod
    def task_description(self) -> str:
        """Return human-readable task description."""
        pass
    
    def _tokenize_all(self) -> List[Dict[str, torch.Tensor]]:
        """Tokenize all examples."""
        tokenized = []
        
        for example in self.examples:
            composition_structure = self._get_compositional_structure(example)
            source_ids = self.tokenizer.encode(
                example.input_text, add_special_tokens=True
            )
            if len(source_ids) > self.max_source_length:
                raise ValueError(
                    f"Source requires {len(source_ids)} tokens but the benchmark "
                    f"contract permits {self.max_source_length}; refusing truncation"
                )
            target_ids = self.tokenizer.encode(
                example.target_text, add_special_tokens=True
            )
            if len(target_ids) > self.max_target_length:
                raise ValueError(
                    f"Gold target requires {len(target_ids)} tokens but the benchmark "
                    f"contract permits {self.max_target_length}; refusing truncation"
                )
            # Tokenize input
            input_encoding = self.tokenizer(
                example.input_text,
                max_length=self.max_source_length,
                padding="max_length",
                truncation=False,
                return_tensors="pt",
            )
            
            # Tokenize target (using text_target instead of deprecated as_target_tokenizer)
            target_encoding = self.tokenizer(
                text_target=example.target_text,
                max_length=self.max_target_length,
                padding="max_length",
                truncation=False,
                return_tensors="pt",
            )
            
            # Prepare labels (replace padding with -100)
            labels = target_encoding.input_ids.squeeze()
            labels[labels == self.tokenizer.pad_token_id] = -100
            
            tokenized.append({
                "input_ids": input_encoding.input_ids.squeeze(),
                "attention_mask": input_encoding.attention_mask.squeeze(),
                "labels": labels,
                "input_text": example.input_text,
                "original_input_text": example.original_input_text or example.input_text,
                "target_text": example.target_text,
                "is_ood": (
                    example.is_ood
                    if example.is_ood is not None
                    else example.split != "train"
                ),
                "generalization_category": example.generalization_category,
                "composition_depth": composition_structure.get(
                    "depth", composition_structure.get("composition_depth")
                ),
                "composition_specs": example.composition_specs or [],
            })
        
        return tokenized
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self._tokenized_examples[idx]
    
    def get_example(self, idx: int) -> CompositionalExample:
        """Get raw example by index."""
        return self.examples[idx]
    
    def collate_fn(self, batch: List[Dict]) -> CompositionalBatch:
        """Custom collate function for DataLoader."""
        input_ids = torch.stack([b["input_ids"] for b in batch])
        attention_mask = torch.stack([b["attention_mask"] for b in batch])
        labels = torch.stack([b["labels"] for b in batch])
        
        # NOTE: decoder_input_ids are now created by the model using T5's _shift_right
        # to ensure consistency between teacher-forced loss and generation
        
        return CompositionalBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_input_ids=None,  # Model will create from labels
            input_texts=[b["input_text"] for b in batch],
            original_input_texts=[b["original_input_text"] for b in batch],
            target_texts=[b["target_text"] for b in batch],
            is_ood=torch.tensor([b["is_ood"] for b in batch]),
            generalization_categories=[
                b["generalization_category"] for b in batch
            ],
            composition_depths=[b["composition_depth"] for b in batch],
            composition_specs=[b["composition_specs"] for b in batch],
        )
    
    def get_dataloader(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
    ) -> DataLoader:
        """Create DataLoader for this dataset."""
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            pin_memory=torch.cuda.is_available(),
        )


class CompositionalDataModule:
    """
    Data module that manages train/val/test splits for a compositional dataset.
    """
    
    def __init__(
        self,
        dataset_class: type,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = 32,
        num_workers: int = 4,
        **dataset_kwargs
    ):
        self.dataset_class = dataset_class
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_kwargs = dataset_kwargs
        
        self._train_dataset = None
        self._val_dataset = None
        self._test_datasets = {}
    
    def setup(self):
        """Initialize all datasets."""
        self._train_dataset = self.dataset_class(
            tokenizer=self.tokenizer,
            split="train",
            **self.dataset_kwargs
        )
        
        # Some datasets have specific test splits
        test_splits = self._get_test_splits()
        for split_name in test_splits:
            self._test_datasets[split_name] = self.dataset_class(
                tokenizer=self.tokenizer,
                split=split_name,
                **self.dataset_kwargs
            )
    
    def _get_test_splits(self) -> List[str]:
        """Return list of test split names. Override in subclasses."""
        return ["test"]
    
    def train_dataloader(self) -> DataLoader:
        return self._train_dataset.get_dataloader(
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
    
    def test_dataloader(self, split: str = "test") -> DataLoader:
        return self._test_datasets[split].get_dataloader(
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
    
    @property
    def train_dataset(self):
        return self._train_dataset
    
    @property
    def test_datasets(self) -> Dict[str, BaseCompositionalDataset]:
        return self._test_datasets
