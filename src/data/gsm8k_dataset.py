"""
GSM8K Dataset for Mathematical Reasoning

GSM8K (Grade School Math 8K) tests multi-step arithmetic reasoning.

Reference: Cobbe et al. (2021) "Training Verifiers to Solve Math Word Problems"

Key Features:
- Grade school math word problems
- Multi-step arithmetic reasoning
- Chain-of-thought solutions

Compositional Structure:
- Quantities and operations
- Multi-step derivations
- Intermediate results

Why Relevant for DAI:
- Clear numerical type system (quantities, operations, results)
- Monotonicity in arithmetic (order preservation)
- Step-by-step composition of operations
"""

import os
import re
import json
from typing import Dict, List, Optional, Tuple

from datasets import load_dataset

from .base_dataset import BaseCompositionalDataset, CompositionalExample


class GSM8KDataset(BaseCompositionalDataset):
    """
    GSM8K dataset loader.
    
    Tests compositional mathematical reasoning.
    
    Split options:
    - "train": Standard training split
    - "test": Standard test split
    - "hard": Custom hard split (≥5 reasoning steps)
    - "easy": Custom easy split (≤3 reasoning steps)
    """
    
    # Type system for math expressions
    MATH_TYPES = {
        # Operations
        "+": "ADD",
        "-": "SUB",
        "*": "MUL",
        "/": "DIV",
        "=": "EQUALS",
        # Quantity indicators
        "total": "AGGREGATE",
        "sum": "AGGREGATE",
        "each": "DISTRIBUTE",
        "per": "RATE",
        "remaining": "DIFFERENCE",
        "left": "DIFFERENCE",
        # Comparisons
        "more": "COMPARE_MORE",
        "less": "COMPARE_LESS",
        "times": "MULTIPLY_IND",
        "twice": "MULTIPLY_IND",
    }
    
    # Hard split threshold (problems with >= this many steps)
    HARD_STEP_THRESHOLD = 5
    # Easy split threshold (problems with <= this many steps)
    EASY_STEP_THRESHOLD = 3
    
    def __init__(
        self,
        tokenizer,
        split: str = "train",
        include_chain_of_thought: bool = True,
        max_source_length: int = 256,
        max_target_length: int = 512,
        max_steps: Optional[int] = None,  # Filter by max reasoning steps
        min_steps: Optional[int] = None,  # Filter by min reasoning steps
        data_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        self.include_chain_of_thought = include_chain_of_thought
        self.max_steps = max_steps
        self.min_steps = min_steps
        
        # Handle custom "hard" and "easy" splits
        self._custom_split = None
        if split in ("hard", "easy"):
            self._custom_split = split
            split = "test"  # Load from test split and filter
        
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
        return "GSM8K"
    
    @property
    def task_description(self) -> str:
        return "Solve grade school math word problems"
    
    def _load_data(self) -> List[CompositionalExample]:
        """Load GSM8K data from HuggingFace."""
        try:
            dataset = load_dataset(
                "gsm8k",
                "main",
                cache_dir=self.cache_dir,
            )
        except Exception as e:
            print(f"Warning: Could not load GSM8K from HuggingFace: {e}")
            return self._load_from_local()
        
        split_data = dataset[self.split]
        examples = []
        
        for item in split_data:
            question = item["question"]
            answer = item["answer"]
            
            # Parse answer to extract steps and final answer
            steps, final_answer = self._parse_answer(answer)
            num_steps = len(steps)
            
            # Apply step-based filtering
            # 1. Explicit max_steps filter
            if self.max_steps is not None and num_steps > self.max_steps:
                continue
            
            # 2. Explicit min_steps filter
            if self.min_steps is not None and num_steps < self.min_steps:
                continue
            
            # 3. Custom split filtering (hard/easy)
            if self._custom_split == "hard" and num_steps < self.HARD_STEP_THRESHOLD:
                continue
            if self._custom_split == "easy" and num_steps > self.EASY_STEP_THRESHOLD:
                continue
            
            # Create target based on CoT preference
            if self.include_chain_of_thought:
                target = answer
            else:
                target = final_answer
            
            examples.append(CompositionalExample(
                input_text=question,
                target_text=target,
                split=self._custom_split or self.split,  # Use custom split name if set
                compositional_structure=self._analyze_problem(question, steps),
                primitives=self._extract_primitives(question),
            ))
        
        return examples
    
    def _load_from_local(self) -> List[CompositionalExample]:
        """Load from local JSONL files."""
        if self.data_dir is None:
            raise ValueError("data_dir required")
        
        file_path = os.path.join(self.data_dir, f"{self.split}.jsonl")
        examples = []
        
        with open(file_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                question = item["question"]
                answer = item["answer"]
                
                steps, final_answer = self._parse_answer(answer)
                num_steps = len(steps)
                
                # Apply step-based filtering
                if self.max_steps is not None and num_steps > self.max_steps:
                    continue
                if self.min_steps is not None and num_steps < self.min_steps:
                    continue
                if self._custom_split == "hard" and num_steps < self.HARD_STEP_THRESHOLD:
                    continue
                if self._custom_split == "easy" and num_steps > self.EASY_STEP_THRESHOLD:
                    continue
                
                target = answer if self.include_chain_of_thought else final_answer
                
                examples.append(CompositionalExample(
                    input_text=question,
                    target_text=target,
                    split=self._custom_split or self.split,
                    compositional_structure=self._analyze_problem(question, steps),
                    primitives=self._extract_primitives(question),
                ))
        
        return examples
    
    def _parse_answer(self, answer: str) -> Tuple[List[str], str]:
        """
        Parse GSM8K answer format.
        
        Format: Step by step reasoning ending with #### <final_answer>
        """
        # Split by ####
        parts = answer.split("####")
        
        if len(parts) == 2:
            reasoning = parts[0].strip()
            final_answer = parts[1].strip()
        else:
            reasoning = answer
            final_answer = ""
        
        # Extract individual steps
        steps = [s.strip() for s in reasoning.split("\n") if s.strip()]
        
        return steps, final_answer
    
    def _extract_primitives(self, question: str) -> List[str]:
        """Extract math primitives from question."""
        primitives = []
        question_lower = question.lower()
        
        for keyword, type_ in self.MATH_TYPES.items():
            if keyword in question_lower:
                primitives.append(keyword)
        
        # Extract numbers
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', question)
        primitives.extend([f"num:{n}" for n in numbers[:5]])  # Limit to 5 numbers
        
        return primitives
    
    def _analyze_problem(self, question: str, steps: List[str]) -> str:
        """Analyze problem structure."""
        num_numbers = len(re.findall(r'\b\d+(?:\.\d+)?\b', question))
        num_steps = len(steps)
        
        # Detect operation types
        operations = []
        combined = question + " " + " ".join(steps)
        if "+" in combined or "add" in combined.lower():
            operations.append("ADD")
        if "-" in combined or "subtract" in combined.lower():
            operations.append("SUB")
        if "*" in combined or "multiply" in combined.lower() or "times" in combined.lower():
            operations.append("MUL")
        if "/" in combined or "divide" in combined.lower():
            operations.append("DIV")
        
        return f"nums:{num_numbers},steps:{num_steps},ops:{'+'.join(operations)}"
    
    def _get_compositional_structure(self, example: CompositionalExample) -> Dict:
        """Get compositional structure for analysis."""
        question = example.input_text.lower()
        
        # Extract numbers
        numbers = [float(n) for n in re.findall(r'\b\d+(?:\.\d+)?\b', question)]
        
        structure = {
            "num_numbers": len(numbers),
            "numbers": numbers[:10],  # First 10
            "has_addition": any(w in question for w in ["add", "total", "sum", "plus"]),
            "has_subtraction": any(w in question for w in ["subtract", "left", "remaining", "minus"]),
            "has_multiplication": any(w in question for w in ["times", "multiply", "each", "per"]),
            "has_division": any(w in question for w in ["divide", "split", "share"]),
            "question_length": len(question.split()),
        }
        
        # Count operations
        ops = sum([
            structure["has_addition"],
            structure["has_subtraction"],
            structure["has_multiplication"],
            structure["has_division"],
        ])
        structure["operation_count"] = ops
        
        # Estimate reasoning steps from structure
        if example.compositional_structure:
            parts = example.compositional_structure.split(",")
            for part in parts:
                if part.startswith("steps:"):
                    structure["reasoning_steps"] = int(part.split(":")[1])
        
        return structure
    
    def get_type_vocabulary(self) -> Dict[str, int]:
        """Get type vocabulary."""
        all_types = set(self.MATH_TYPES.values())
        all_types.update(["NUMBER", "INTERMEDIATE", "FINAL"])
        return {t: i for i, t in enumerate(sorted(all_types))}


class GSM8KDataModule:
    """Data module for GSM8K experiments."""
    
    def __init__(
        self,
        tokenizer,
        include_chain_of_thought: bool = True,
        batch_size: int = 16,
        max_source_length: int = 256,
        max_target_length: int = 512,
        num_workers: int = 4,
        cache_dir: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.include_chain_of_thought = include_chain_of_thought
        self.batch_size = batch_size
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.num_workers = num_workers
        self.cache_dir = cache_dir
        
        self.train_dataset = None
        self.test_dataset = None
    
    def setup(self):
        """Initialize datasets."""
        self.train_dataset = GSM8KDataset(
            tokenizer=self.tokenizer,
            split="train",
            include_chain_of_thought=self.include_chain_of_thought,
            max_source_length=self.max_source_length,
            max_target_length=self.max_target_length,
            cache_dir=self.cache_dir,
        )
        
        self.test_dataset = GSM8KDataset(
            tokenizer=self.tokenizer,
            split="test",
            include_chain_of_thought=self.include_chain_of_thought,
            max_source_length=self.max_source_length,
            max_target_length=self.max_target_length,
            cache_dir=self.cache_dir,
        )
    
    def train_dataloader(self):
        return self.train_dataset.get_dataloader(
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
    
    def test_dataloader(self):
        return self.test_dataset.get_dataloader(
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class GSM8KByDifficultyModule:
    """
    Data module that stratifies GSM8K by difficulty (number of reasoning steps).
    
    Useful for analyzing compositional generalization: train on easy, test on hard.
    
    Now uses the built-in "hard" and "easy" split support in GSM8KDataset.
    """
    
    def __init__(
        self,
        tokenizer,
        train_max_steps: int = 3,  # Train on problems with ≤3 steps
        test_min_steps: int = 5,   # Test on problems with ≥5 steps (matches HARD_STEP_THRESHOLD)
        batch_size: int = 16,
        **kwargs
    ):
        self.tokenizer = tokenizer
        self.train_max_steps = train_max_steps
        self.test_min_steps = test_min_steps
        self.batch_size = batch_size
        self.kwargs = kwargs
        
        self.train_dataset = None
        self.test_easy_dataset = None
        self.test_hard_dataset = None
    
    def setup(self):
        """Initialize stratified datasets."""
        # Full training set filtered by max steps
        self.train_dataset = GSM8KDataset(
            tokenizer=self.tokenizer,
            split="train",
            max_steps=self.train_max_steps,
            **self.kwargs
        )
        
        # Easy test (in-distribution) - use the built-in "easy" split
        self.test_easy_dataset = GSM8KDataset(
            tokenizer=self.tokenizer,
            split="easy",  # Uses EASY_STEP_THRESHOLD
            **self.kwargs
        )
        
        # Hard test (out-of-distribution) - use the built-in "hard" split
        self.test_hard_dataset = GSM8KDataset(
            tokenizer=self.tokenizer,
            split="hard",  # Uses HARD_STEP_THRESHOLD
            **self.kwargs
        )
        
        # Print statistics
        print(f"GSM8K stratified split:")
        print(f"  Train (≤{self.train_max_steps} steps): {len(self.train_dataset)} examples")
        print(f"  Test Easy (≤{GSM8KDataset.EASY_STEP_THRESHOLD} steps): {len(self.test_easy_dataset)} examples")
        print(f"  Test Hard (≥{GSM8KDataset.HARD_STEP_THRESHOLD} steps): {len(self.test_hard_dataset)} examples")
    
    def train_dataloader(self):
        return self.train_dataset.get_dataloader(
            batch_size=self.batch_size,
            shuffle=True,
        )
    
    def test_easy_dataloader(self):
        """In-distribution test."""
        return self.test_easy_dataset.get_dataloader(
            batch_size=self.batch_size,
            shuffle=False,
        )
    
    def test_hard_dataloader(self):
        """Out-of-distribution test (compositional generalization)."""
        return self.test_hard_dataset.get_dataloader(
            batch_size=self.batch_size,
            shuffle=False,
        )
