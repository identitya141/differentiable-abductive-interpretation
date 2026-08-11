"""
Baseline Models for DAI Experiments

Implements all required baselines for fair comparison:
1. Vanilla T5 (fine-tuned without any modification)
2. Chain-of-Thought (CoT) prompting
3. Scratchpad training
4. Neuro-Symbolic baseline (Neural Module Networks inspired)

Each baseline is carefully implemented to be a strong competitor.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Type

import re
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Config, T5Tokenizer, PreTrainedTokenizer
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput


def linearize_source_only_tree(text: str) -> str:
    """Create a deterministic balanced tree using source tokens only."""
    words = text.split()

    def build(items):
        if len(items) <= 1:
            return items[0] if items else ""
        middle = len(items) // 2
        return f"( SEQ {build(items[:middle])} {build(items[middle:])} )"

    return f"{text} <TREE> {build(words)}"

# Optional LLaMA imports
try:
    from transformers import (
        LlamaForCausalLM,
        LlamaTokenizer,
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    HAS_LLAMA = True
except ImportError:
    HAS_LLAMA = False

# Optional PEFT imports for LoRA
try:
    from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False


@dataclass
class BaselineConfig:
    """Configuration for baseline models."""
    base_model: str = "t5-small"
    max_source_length: int = 128
    max_target_length: int = 128
    # Tokenizer will be set after model creation for special token handling
    tokenizer: Optional[PreTrainedTokenizer] = None


class BaselineModel(nn.Module, ABC):
    """Abstract base class for baseline models."""

    @property
    def main_input_name(self) -> str:
        return "input_ids"

    @property
    def generation_config(self):
        """Expose the wrapped HF model's generation settings to Trainer."""
        return self._hf_model.generation_config

    @property
    def config(self):
        """Expose the wrapped HF config while retaining our baseline settings."""
        return self._hf_model.config

    def can_generate(self) -> bool:
        return True

    @property
    def _hf_model(self):
        return getattr(self, "t5", getattr(self, "model", None))
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return baseline name."""
        pass
    
    @abstractmethod
    def forward(self, **kwargs):
        """Forward pass."""
        pass
    
    @abstractmethod
    def generate(self, **kwargs):
        """Generate outputs."""
        pass


class VanillaT5(BaselineModel):
    """
    Vanilla T5 Baseline
    
    Standard T5 fine-tuned on the task without any modifications.
    This is the most basic baseline and represents what a standard
    seq2seq model achieves without compositional inductive biases.
    
    Expected Behavior:
    - Good in-distribution performance
    - Poor compositional generalization
    - Provides lower bound for improvement
    """
    
    def __init__(self, config: BaselineConfig):
        super().__init__()
        self.baseline_config = config
        self.t5 = T5ForConditionalGeneration.from_pretrained(config.base_model)
    
    @property
    def name(self) -> str:
        return "Vanilla T5"
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ):
        return self.t5(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
        )
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ):
        return self.t5.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )


class RandomInitT5(VanillaT5):
    """Architecture-matched T5 trained without pretrained weights."""

    def __init__(self, config: BaselineConfig):
        nn.Module.__init__(self)
        self.baseline_config = config
        t5_config = T5Config.from_pretrained(config.base_model)
        self.t5 = T5ForConditionalGeneration(t5_config)
        self.allowed_output_token_ids: Optional[Tuple[int, ...]] = None

    def set_allowed_output_token_ids(self, token_ids) -> None:
        """Restrict decoding to tokens observed on the training target side."""
        allowed = {int(token_id) for token_id in token_ids if int(token_id) >= 0}
        if self.t5.config.eos_token_id is not None:
            allowed.add(int(self.t5.config.eos_token_id))
        if not allowed:
            raise ValueError("The random-init output vocabulary cannot be empty")
        self.allowed_output_token_ids = tuple(sorted(allowed))

    def generate(self, input_ids, attention_mask=None, **kwargs):
        if self.allowed_output_token_ids is not None:
            allowed = list(self.allowed_output_token_ids)
            kwargs.setdefault(
                "prefix_allowed_tokens_fn",
                lambda _batch_id, _input_ids: allowed,
            )
        return self.t5.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

    @property
    def name(self) -> str:
        return "Random-initialized T5"


class TreeLinearizedT5(VanillaT5):
    """Pretrained T5 control whose inputs use a linearized parse tree."""

    input_representation = "tree_linearized"

    @property
    def name(self) -> str:
        return "Tree-linearized T5"


class ChainOfThoughtT5(BaselineModel):
    """
    Chain-of-Thought (CoT) Baseline
    
    T5 with Chain-of-Thought prompting/training. The model is trained
    to produce intermediate reasoning steps before the final answer.
    
    Reference: Wei et al. (2022) "Chain-of-Thought Prompting Elicits 
               Reasoning in Large Language Models"
    
    Implementation:
    - Training: Prepend "Let's think step by step." to inputs
    - Targets include intermediate reasoning (if available)
    - For datasets without CoT annotations, we use template-based CoT
    
    Expected Behavior:
    - Better on multi-step reasoning (GSM8K)
    - Moderate improvement on compositional tasks
    - Benefits from explicit reasoning chain
    
    Usage:
        model = ChainOfThoughtT5(config, tokenizer)
        # For training: use preprocess_batch() to prepare inputs
        inputs = model.preprocess_batch(texts, tokenizer)
        # For inference: use generate_from_text() for automatic preprocessing
        answers = model.generate_from_text(texts)
    """
    
    COT_PREFIX = "Let's think step by step. "
    COT_ANSWER_MARKER = "Therefore, the answer is: "
    
    def __init__(self, config: BaselineConfig, tokenizer: Optional[PreTrainedTokenizer] = None):
        super().__init__()
        self.baseline_config = config
        self.t5 = T5ForConditionalGeneration.from_pretrained(config.base_model)
        self.tokenizer = tokenizer
    
    @property
    def name(self) -> str:
        return "Chain-of-Thought T5"
    
    def preprocess_input(self, input_text: str) -> str:
        """Add CoT prompt to input."""
        return f"{self.COT_PREFIX}{input_text}"
    
    def preprocess_batch(
        self,
        input_texts: List[str],
        tokenizer: Optional[PreTrainedTokenizer] = None,
        max_length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess a batch of inputs with CoT prefix.
        
        Args:
            input_texts: List of input strings
            tokenizer: Tokenizer to use (uses self.tokenizer if None)
            max_length: Maximum sequence length
            
        Returns:
            Tokenized inputs with CoT prefix
        """
        tokenizer = tokenizer or self.tokenizer
        if tokenizer is None:
            raise ValueError("Tokenizer not set. Pass tokenizer to __init__ or this method.")
        
        max_length = max_length or self.baseline_config.max_source_length
        processed = [self.preprocess_input(text) for text in input_texts]
        
        return tokenizer(
            processed,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
    
    def format_target_with_cot(self, reasoning: str, answer: str) -> str:
        """
        Format target with reasoning chain and final answer.
        
        Args:
            reasoning: The reasoning/explanation steps
            answer: The final answer
            
        Returns:
            Formatted target string
        """
        return f"{reasoning} {self.COT_ANSWER_MARKER}{answer}"
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ):
        # Standard forward (CoT is in the data preprocessing)
        return self.t5(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
        )
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 256,
        **kwargs
    ):
        # Generate with longer max_length to accommodate CoT
        return self.t5.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            **kwargs
        )
    
    def generate_from_text(
        self,
        input_texts: List[str],
        tokenizer: Optional[PreTrainedTokenizer] = None,
        max_length: int = 256,
        extract_answer: bool = True,
        **kwargs
    ) -> List[str]:
        """
        Generate from text inputs, handling preprocessing automatically.
        
        Args:
            input_texts: Raw input texts (CoT prefix will be added)
            tokenizer: Tokenizer to use
            max_length: Maximum generation length
            extract_answer: If True, extract just the final answer
            
        Returns:
            List of generated texts (or extracted answers if extract_answer=True)
        """
        tokenizer = tokenizer or self.tokenizer
        if tokenizer is None:
            raise ValueError("Tokenizer not set. Pass tokenizer to __init__ or this method.")
        
        inputs = self.preprocess_batch(input_texts, tokenizer)
        inputs = {k: v.to(self.t5.device) for k, v in inputs.items()}
        
        outputs = self.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            **kwargs
        )
        
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        if extract_answer:
            return [self.extract_final_answer(text) for text in decoded]
        return decoded
    
    @staticmethod
    def extract_final_answer(generated_text: str) -> str:
        """Extract final answer from CoT output."""
        # Look for common answer markers
        markers = ["Therefore, the answer is:", "Therefore,", "So,", "The answer is", "####"]
        
        for marker in markers:
            if marker in generated_text:
                parts = generated_text.split(marker)
                if len(parts) > 1:
                    return parts[-1].strip()
        
        # If no marker, return last sentence
        sentences = generated_text.split(".")
        return sentences[-1].strip() if sentences else generated_text


class ScratchpadT5(BaselineModel):
    """
    Scratchpad Training Baseline
    
    T5 trained with a scratchpad: the model writes intermediate
    computations in a designated scratchpad area before producing output.
    
    Reference: Nye et al. (2021) "Show Your Work: Scratchpads for 
               Intermediate Computation with Language Models"
    
    Implementation:
    - Input format: "<input> [SCRATCH]"
    - Output format: "<scratchpad contents> [/SCRATCH] <answer>"
    - Model learns to use scratchpad for intermediate steps
    
    Expected Behavior:
    - Better on multi-step arithmetic
    - Can learn to decompose problems
    - More structured than CoT
    """
    
    SCRATCH_START = "[SCRATCH]"
    SCRATCH_END = "[/SCRATCH]"
    SPECIAL_TOKENS = [SCRATCH_START, SCRATCH_END]
    
    def __init__(self, config: BaselineConfig, tokenizer: Optional[PreTrainedTokenizer] = None):
        super().__init__()
        self.baseline_config = config
        self.t5 = T5ForConditionalGeneration.from_pretrained(config.base_model)
        self.tokenizer = tokenizer
        
        # Add special tokens to tokenizer and resize embeddings
        if tokenizer is not None:
            self._add_special_tokens(tokenizer)
    
    def _add_special_tokens(self, tokenizer: PreTrainedTokenizer) -> int:
        """
        Add scratchpad special tokens to tokenizer and resize model embeddings.
        
        Returns:
            Number of tokens added
        """
        num_added = tokenizer.add_special_tokens({
            "additional_special_tokens": self.SPECIAL_TOKENS
        })
        if num_added > 0:
            self.t5.resize_token_embeddings(len(tokenizer))
        self.tokenizer = tokenizer
        return num_added
    
    def setup_tokenizer(self, tokenizer: PreTrainedTokenizer) -> PreTrainedTokenizer:
        """
        Setup tokenizer with special tokens. Call this before using the model.
        
        Args:
            tokenizer: The tokenizer to modify
            
        Returns:
            Modified tokenizer with special tokens added
        """
        self._add_special_tokens(tokenizer)
        return tokenizer
    
    @property
    def name(self) -> str:
        return "Scratchpad T5"
    
    def preprocess_input(self, input_text: str) -> str:
        """Add scratchpad marker to input."""
        return f"{input_text} {self.SCRATCH_START}"
    
    def preprocess_batch(
        self,
        input_texts: List[str],
        tokenizer: Optional[PreTrainedTokenizer] = None,
        max_length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess a batch of inputs with scratchpad formatting.
        
        Args:
            input_texts: List of input strings
            tokenizer: Tokenizer to use (uses self.tokenizer if None)
            max_length: Maximum sequence length
            
        Returns:
            Tokenized inputs with scratchpad markers
        """
        tokenizer = tokenizer or self.tokenizer
        if tokenizer is None:
            raise ValueError("Tokenizer not set. Call setup_tokenizer() first.")
        
        max_length = max_length or self.baseline_config.max_source_length
        processed = [self.preprocess_input(text) for text in input_texts]
        
        return tokenizer(
            processed,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
    
    def format_target(self, scratchpad: str, answer: str) -> str:
        """Format target with scratchpad."""
        return f"{scratchpad} {self.SCRATCH_END} {answer}"
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ):
        return self.t5(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
        )
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 512,
        **kwargs
    ):
        return self.t5.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            **kwargs
        )
    
    def generate_from_text(
        self,
        input_texts: List[str],
        tokenizer: Optional[PreTrainedTokenizer] = None,
        max_length: int = 512,
        **kwargs
    ) -> List[str]:
        """
        Generate from text inputs, handling preprocessing automatically.
        
        Args:
            input_texts: Raw input texts (scratchpad marker will be added)
            tokenizer: Tokenizer to use
            max_length: Maximum generation length
            
        Returns:
            List of extracted answers (scratchpad contents removed)
        """
        tokenizer = tokenizer or self.tokenizer
        if tokenizer is None:
            raise ValueError("Tokenizer not set. Call setup_tokenizer() first.")
        
        inputs = self.preprocess_batch(input_texts, tokenizer)
        inputs = {k: v.to(self.t5.device) for k, v in inputs.items()}
        
        outputs = self.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            **kwargs
        )
        
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=False)
        return [self.extract_answer(text) for text in decoded]
    
    @staticmethod
    def extract_answer(generated_text: str) -> str:
        """Extract answer from scratchpad output."""
        if "[/SCRATCH]" in generated_text:
            return generated_text.split("[/SCRATCH]")[-1].strip()
        return generated_text


class ModularNeuralNetwork(BaselineModel):
    """
    Neuro-Symbolic / Neural Module Network Baseline
    
    Inspired by Neural Module Networks (NMN), this baseline uses
    learned modules for different operations that can be composed.
    
    Reference: Andreas et al. (2016) "Neural Module Networks"
    
    Implementation:
    - Fixed set of operation modules (e.g., FIND, RELATE, COUNT)
    - Controller network selects and composes modules
    - End-to-end differentiable
    
    Simplified Version:
    - Modules are implemented as small MLPs
    - Module selection via attention
    - Sequential composition
    
    Expected Behavior:
    - Strong on tasks with clear modular structure
    - Interpretable intermediate representations
    - May struggle with very long compositions
    """
    
    def __init__(
        self,
        config: BaselineConfig,
        num_modules: int = 8,
        module_dim: int = 256,
        num_composition_steps: int = 4,
    ):
        super().__init__()
        self.baseline_config = config
        self.num_modules = num_modules
        self.module_dim = module_dim
        self.num_composition_steps = num_composition_steps
        
        # Base T5 for encoding
        self.t5 = T5ForConditionalGeneration.from_pretrained(config.base_model)
        hidden_dim = self.t5.config.d_model
        
        # Module bank: each module is a small MLP
        self.module_bank = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, module_dim),
                nn.ReLU(),
                nn.Linear(module_dim, hidden_dim),
            )
            for _ in range(num_modules)
        ])
        
        # Controller: selects which module to apply
        self.controller = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_modules),
        )
        
        # Composition gate
        self.composition_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
    
    @property
    def name(self) -> str:
        return "Neural Module Network"
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ):
        # Encode input
        encoder_outputs = self.t5.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden = encoder_outputs.last_hidden_state
        
        # Apply modular composition
        composed = self._modular_composition(hidden, attention_mask)
        
        # Handle decoder_input_ids: if None but labels exist, use T5's shift_right
        if decoder_input_ids is None and labels is not None:
            decoder_input_ids = self.t5._shift_right(labels)
        elif decoder_input_ids is None:
            # If neither provided, create a start token
            batch_size = input_ids.shape[0]
            decoder_input_ids = torch.full(
                (batch_size, 1),
                self.t5.config.decoder_start_token_id,
                dtype=torch.long,
                device=input_ids.device,
            )
        
        # Decode from composed representation
        decoder_outputs = self.t5.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=composed,
            encoder_attention_mask=attention_mask,
        )
        
        sequence_output = decoder_outputs.last_hidden_state
        if self.t5.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.t5.model_dim ** -0.5)
        lm_logits = self.t5.lm_head(sequence_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        
        return Seq2SeqLMOutput(loss=loss, logits=lm_logits)
    
    def _modular_composition(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply learned modules in sequence.
        
        At each step:
        1. Controller selects module probabilities
        2. Weighted sum of module outputs
        3. Compose with previous state
        """
        batch_size, seq_len, hidden_dim = hidden.shape
        state = hidden
        
        for step in range(self.num_composition_steps):
            # Pool state for controller input
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (state * mask).sum(dim=1) / mask.sum(dim=1)
            else:
                pooled = state.mean(dim=1)
            
            # Get module selection probabilities
            module_probs = F.softmax(self.controller(pooled), dim=-1)  # [batch, num_modules]
            
            # Apply each module and weight by probability
            module_outputs = []
            for i, module in enumerate(self.module_bank):
                output = module(state)  # [batch, seq, hidden]
                module_outputs.append(output)
            
            # Weighted combination
            module_outputs = torch.stack(module_outputs, dim=1)  # [batch, num_modules, seq, hidden]
            weights = module_probs.unsqueeze(-1).unsqueeze(-1)  # [batch, num_modules, 1, 1]
            composed_output = (module_outputs * weights).sum(dim=1)  # [batch, seq, hidden]
            
            # Gated composition with previous state
            gate_input = torch.cat([state, composed_output], dim=-1)
            gate = self.composition_gate(gate_input)
            state = gate * composed_output + (1 - gate) * state
        
        return state
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ):
        # Encode and compose
        encoder_outputs = self.t5.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        composed = self._modular_composition(
            encoder_outputs.last_hidden_state,
            attention_mask
        )
        
        # Use proper BaseModelOutput instead of fragile wrapper
        encoder_outputs_wrapped = BaseModelOutput(last_hidden_state=composed)
        
        return self.t5.generate(
            encoder_outputs=encoder_outputs_wrapped,
            attention_mask=attention_mask,
            **kwargs
        )


class StructuralControlT5(BaselineModel):
    """Shared adapter for publication controls implemented with the DAI path.

    These controls intentionally reuse the proposed model's encoder plumbing so
    that only the structural signal changes.  They remain baselines because the
    registry fixes their objective/data corruption independently of the proposed
    full structural-contrastive method.
    """

    composition_structure_mode = "grounded"
    structural_contrastive_weight = 1.0
    composition_weight = 0.5

    def __init__(self, config: BaselineConfig):
        super().__init__()
        from src.models.dai_transformer import DAIConfig, DAITransformer

        dai_config = DAIConfig(
            base_model_name=config.base_model,
            concretization_weight=0.0,
            composition_weight=self.composition_weight,
            entropy_regularization=0.0,
            contrastive_weight=0.0,
            structural_contrastive_weight=self.structural_contrastive_weight,
            cross_layer_consistency=False,
            consistency_weight=0.0,
        )
        self.baseline_config = config
        self.model = DAITransformer(dai_config)
        self.t5 = self.model.t5

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def generate(self, **kwargs):
        return self.model.generate(**kwargs)


class RandomStructureT5(StructuralControlT5):
    """DAI control trained with freshly randomized composition structures."""

    composition_structure_mode = "random"

    @property
    def name(self) -> str:
        return "Random-structure T5"


class ShuffledStructureT5(StructuralControlT5):
    """DAI control trained with structures shuffled across examples."""

    composition_structure_mode = "shuffled"

    @property
    def name(self) -> str:
        return "Shuffled-structure T5"


class SimpleConsistencyT5(StructuralControlT5):
    """Composition-consistency control without contrastive auxiliary loss."""

    structural_contrastive_weight = 0.0
    composition_weight = 1.0

    @property
    def name(self) -> str:
        return "Simple structural-consistency T5"


class SymbolicRuleAugmentedT5(BaselineModel):
    """
    Symbolic Rule-Augmented Baseline
    
    T5 augmented with explicit symbolic rules that are applied
    during inference. Rules are specified as regex-like patterns.
    
    This represents a strong neuro-symbolic baseline where symbolic
    knowledge is explicitly programmed. Implements full SCAN grammar.
    
    SCAN Grammar (fully implemented):
    - Primitives: jump, walk, run, look → JUMP, WALK, RUN, LOOK
    - Directions: turn left, turn right → LTURN, RTURN
    - Modifiers: twice, thrice → repeat action 2x, 3x
    - Conjunctions: and → sequential composition
    - Temporal: after → reverse order (B after A → A B)
    - Spatial: around left/right → LTURN + action * 4
    - Directional: left/right (standalone) → LTURN/RTURN + action
    - Opposite: opposite left/right → 2x LTURN/RTURN
    
    Expected Behavior:
    - Very strong when rules match the task (near 100% on SCAN)
    - Poor generalization beyond specified rules
    - Not end-to-end trainable (rules are fixed)
    """
    
    # Primitive action mappings
    PRIMITIVES = {
        "jump": "JUMP",
        "walk": "WALK",
        "run": "RUN",
        "look": "LOOK",
    }
    
    # Direction mappings  
    DIRECTIONS = {
        "left": "LTURN",
        "right": "RTURN",
    }
    
    def __init__(
        self,
        config: BaselineConfig,
        rules: Optional[Dict[str, str]] = None,
        dataset_type: str = "scan",  # "scan", "cogs", "cfq", "clutrr", "gsm8k"
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ):
        super().__init__()
        self.baseline_config = config
        self.tokenizer = tokenizer
        self.dataset_type = dataset_type.lower()
        self.t5 = T5ForConditionalGeneration.from_pretrained(config.base_model)
        
        # Only SCAN has full rule coverage; others fall back to neural
        self._supports_rules = self.dataset_type == "scan"
        if not self._supports_rules:
            import logging
            logging.getLogger(__name__).info(
                f"SymbolicRuleAugmentedT5: No rules for {dataset_type}, using neural fallback"
            )
    
    @property
    def name(self) -> str:
        return "Symbolic Rule-Augmented T5"
    
    def forward(self, **kwargs):
        return self.t5(**kwargs)
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        tokenizer=None,
        **kwargs
    ):
        # Only try symbolic rules for SCAN dataset
        tokenizer = tokenizer or self.tokenizer
        if self._supports_rules and tokenizer is not None:
            inputs = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            rule_outputs = [self._apply_rules(inp) for inp in inputs]
            
            # If rules apply to ALL inputs, use them
            if all(out is not None for out in rule_outputs):
                return tokenizer(
                    rule_outputs,
                    return_tensors="pt",
                    padding=True,
                ).input_ids.to(input_ids.device)
        
        # Fall back to neural generation (always for non-SCAN, or when rules fail)
        return self.t5.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
    
    def _apply_rules(self, text: str) -> Optional[str]:
        """
        Apply SCAN symbolic rules to input text.
        
        Handles the full SCAN grammar including:
        - Primitives (jump, walk, run, look)
        - Directions (turn left, turn right)
        - Modifiers (twice, thrice)
        - Conjunctions (and)
        - Temporal (after)
        - Spatial (around left/right)
        - Directional prefix (left/right before action)
        - Opposite (opposite left/right)
        """
        text = text.lower().strip()
        
        try:
            result = self._parse_command(text)
            return result if result else None
        except Exception:
            return None
    
    def _parse_command(self, text: str) -> Optional[str]:
        """Parse a SCAN command recursively."""
        text = text.strip()
        
        if not text:
            return ""
        
        # Handle "after" - B after A means execute A then B
        if " after " in text:
            parts = text.split(" after ", 1)
            if len(parts) == 2:
                b_part = self._parse_command(parts[0].strip())
                a_part = self._parse_command(parts[1].strip())
                if a_part is not None and b_part is not None:
                    return f"{a_part} {b_part}".strip()
                return None
        
        # Handle "and" - A and B means execute A then B
        if " and " in text:
            parts = text.split(" and ", 1)
            if len(parts) == 2:
                a_part = self._parse_command(parts[0].strip())
                b_part = self._parse_command(parts[1].strip())
                if a_part is not None and b_part is not None:
                    return f"{a_part} {b_part}".strip()
                return None
        
        # Handle modifiers: twice/thrice at the end
        for modifier, count in [("thrice", 3), ("twice", 2)]:
            if text.endswith(f" {modifier}"):
                base = text[:-len(modifier)-1].strip()
                base_result = self._parse_command(base)
                if base_result is not None:
                    return " ".join([base_result] * count)
        
        # Handle "around left/right" - turn + action 4 times
        for direction, turn in [("left", "LTURN"), ("right", "RTURN")]:
            if f" around {direction}" in text:
                # Extract the action before "around"
                action_text = text.replace(f" around {direction}", "").strip()
                if action_text == "turn":
                    return " ".join([turn] * 4)
                action_result = self._parse_primitive_or_turn(action_text)
                if action_result is not None:
                    # around X = (TURN + action) * 4
                    single = f"{turn} {action_result}"
                    return " ".join([single] * 4)
        
        # Handle "opposite left/right" - turn twice
        for direction, turn in [("left", "LTURN"), ("right", "RTURN")]:
            if text == f"turn opposite {direction}":
                return f"{turn} {turn}"
            if text.endswith(f" opposite {direction}"):
                action_text = text[:-len(f" opposite {direction}")].strip()
                action_result = self._parse_primitive_or_turn(action_text)
                if action_result is not None:
                    return f"{turn} {turn} {action_result}"
            if text.startswith(f"opposite {direction} "):
                # opposite left/right + action = TURN TURN + action
                action_text = text[len(f"opposite {direction} "):].strip()
                action_result = self._parse_command(action_text)
                if action_result is not None:
                    return f"{turn} {turn} {action_result}"
        
        # Handle directional prefix: "left/right" before action
        for direction, turn in [("left", "LTURN"), ("right", "RTURN")]:
            if text.endswith(f" {direction}"):
                action_text = text[:-len(f" {direction}")].strip()
                action_result = self._parse_primitive_or_turn(action_text)
                if action_result is not None:
                    return f"{turn} {action_result}"
            if text.startswith(f"{direction} "):
                action_text = text[len(direction)+1:].strip()
                action_result = self._parse_primitive_or_turn(action_text)
                if action_result is not None:
                    return f"{turn} {action_result}"
        
        # Base case: primitive or turn
        return self._parse_primitive_or_turn(text)
    
    def _parse_primitive_or_turn(self, text: str) -> Optional[str]:
        """Parse a primitive action or turn command."""
        text = text.strip()
        
        # Check for "turn left/right"
        for direction, turn in [("left", "LTURN"), ("right", "RTURN")]:
            if text == f"turn {direction}":
                return turn
        
        # Check for primitives
        if text in self.PRIMITIVES:
            return self.PRIMITIVES[text]
        
        return None
    
    def test_rules(self) -> Dict[str, Tuple[str, str, bool]]:
        """
        Test the rule implementation against known SCAN examples.
        
        Returns:
            Dict mapping input to (expected, actual, passed)
        """
        test_cases = {
            "jump": "JUMP",
            "walk": "WALK",
            "run": "RUN",
            "look": "LOOK",
            "turn left": "LTURN",
            "turn right": "RTURN",
            "jump twice": "JUMP JUMP",
            "run thrice": "RUN RUN RUN",
            "jump and walk": "JUMP WALK",
            "jump after walk": "WALK JUMP",
            "turn left twice": "LTURN LTURN",
            "jump around left": "LTURN JUMP LTURN JUMP LTURN JUMP LTURN JUMP",
            "turn around right": "RTURN RTURN RTURN RTURN",
            "jump left": "LTURN JUMP",
            "walk opposite right": "RTURN RTURN WALK",
            "left jump": "LTURN JUMP",
            "right walk": "RTURN WALK",
            "turn opposite left": "LTURN LTURN",
            "jump twice and walk": "JUMP JUMP WALK",
            "jump after walk twice": "WALK WALK JUMP",
        }
        
        results = {}
        for input_text, expected in test_cases.items():
            actual = self._apply_rules(input_text)
            passed = actual == expected
            results[input_text] = (expected, actual, passed)
        
        return results


class TinyLlamaBaseline(BaselineModel):
    """
    Fine-tuned TinyLlama 1.1B decoder-only baseline.
    
    TinyLlama fine-tuned on the task using LoRA for efficient adaptation.
    This represents a modern decoder-only LLM baseline which excels at
    in-context learning and instruction following.
    
    Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
    
    Implementation:
    - Uses LoRA (Low-Rank Adaptation) for efficient fine-tuning
    - Supports 4-bit quantization for memory efficiency
    - Formatted as instruction-following task
    - Dataset-aware instructions for different tasks
    
    Expected Behavior:
    - Strong few-shot and in-distribution performance
    - Better than T5 on longer outputs
    - May still struggle with compositional generalization
    """
    
    INSTRUCTION_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""
    
    # Dataset-specific instructions for proper task formatting
    DATASET_INSTRUCTIONS = {
        "scan": "Translate the following command to actions:",
        "cogs": "Parse the following sentence into a logical form:",
        "slog": "Parse the following structurally novel sentence into a logical form:",
        "cfq": "Translate the following question into a SPARQL query:",
        "clutrr": "Given the family relationships described, determine the relationship between the specified people:",
        "gsm8k": "Solve the following math word problem step by step:",
    }
    
    def __init__(
        self,
        config: BaselineConfig,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        use_lora: bool = True,
        use_4bit: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        instruction: Optional[str] = None,
        dataset_type: str = "scan",
    ):
        """
        Initialize the TinyLlama baseline.
        
        Args:
            config: Baseline configuration
            model_name: HuggingFace model name or path
            use_lora: Whether to use LoRA for efficient fine-tuning
            use_4bit: Whether to use 4-bit quantization
            lora_r: LoRA rank
            lora_alpha: LoRA alpha scaling
            lora_dropout: LoRA dropout
            instruction: Task instruction for prompting (overrides dataset default)
            dataset_type: Dataset type for automatic instruction selection
        """
        super().__init__()
        
        if not HAS_LLAMA:
            raise ImportError(
                "TinyLlama baseline requires transformers>=4.31.0. "
                "Install with: pip install transformers>=4.31.0"
            )
        
        self.baseline_config = config
        self.model_name = model_name
        self.dataset_type = dataset_type.lower()
        # Use provided instruction or fall back to dataset-specific default
        self.instruction = instruction or self.DATASET_INSTRUCTIONS.get(
            self.dataset_type, 
            "Complete the following task:"  # Generic fallback
        )
        self.use_lora = use_lora
        self.use_4bit = use_4bit
        
        # Quantization config
        bnb_config = None
        if use_4bit:
            try:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
            except Exception:
                print("Warning: 4-bit quantization not available, loading in fp16")
                bnb_config = None
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id
        
        # Apply LoRA if requested
        if use_lora and HAS_PEFT:
            if bnb_config is not None:
                self.model = prepare_model_for_kbit_training(self.model)
            
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        elif use_lora and not HAS_PEFT:
            print("Warning: LoRA requested but PEFT not installed. Using full fine-tuning.")
    
    @property
    def name(self) -> str:
        return f"TinyLlama ({self.model_name.split('/')[-1]})"
    
    def format_prompt(self, input_text: str) -> str:
        """Format input as instruction prompt."""
        return self.INSTRUCTION_TEMPLATE.format(
            instruction=self.instruction,
            input=input_text,
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        Forward pass for training.
        
        Note: LLaMA is a causal LM, so we use the standard language modeling objective.
        Labels should be the full sequence (prompt + response) with prompt masked out.
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 128,
        temperature: float = 0.1,
        do_sample: bool = False,
        **kwargs
    ):
        """Generate responses."""
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs
        )
    
    def generate_from_text(
        self,
        input_texts: List[str],
        max_new_tokens: int = 128,
        **kwargs
    ) -> List[str]:
        """Generate responses from text inputs."""
        # Format prompts
        prompts = [self.format_prompt(text) for text in input_texts]
        
        # Tokenize
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.baseline_config.max_source_length,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        outputs = self.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            **kwargs
        )
        
        # Decode (only the new tokens)
        # Use attention_mask to get actual prompt length (handles padding correctly)
        responses = []
        for i, output in enumerate(outputs):
            # Count actual tokens (non-padding) in the prompt
            prompt_len = int(inputs["attention_mask"][i].sum().item())
            response = self.tokenizer.decode(
                output[prompt_len:],
                skip_special_tokens=True,
            )
            responses.append(response.strip())
        
        return responses
    
    def get_trainable_parameters(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


@dataclass(frozen=True)
class BaselineSpec:
    """Canonical identity and execution metadata for one publication baseline."""

    key: str
    display_name: str
    model_class: Type[BaselineModel]
    runner: str = "baseline"
    config_name: Optional[str] = None
    supported_datasets: Tuple[str, ...] = ("scan", "cogs", "slog", "cfq", "clutrr", "gsm8k")


# This is the sole authoritative list of the eleven publication baselines.
# Aliases are deliberately kept in a separate map so they cannot inflate counts.
BASELINE_REGISTRY: Dict[str, BaselineSpec] = {
    "reference_t5": BaselineSpec("reference_t5", "Vanilla / reference T5", VanillaT5, config_name="vanilla_t5.yaml"),
    "random_init_t5": BaselineSpec("random_init_t5", "Random-initialized T5", RandomInitT5, config_name="random_init_t5.yaml"),
    "tree_linearized_t5": BaselineSpec("tree_linearized_t5", "Tree-linearized T5", TreeLinearizedT5, config_name="tree_linearized_t5.yaml"),
    "random_structure": BaselineSpec("random_structure", "Random structure", RandomStructureT5, runner="dai_control", config_name="random_structure.yaml"),
    "shuffled_structure": BaselineSpec("shuffled_structure", "Shuffled structure", ShuffledStructureT5, runner="dai_control", config_name="shuffled_structure.yaml"),
    "simple_consistency": BaselineSpec("simple_consistency", "Simple structural consistency", SimpleConsistencyT5, runner="dai_control", config_name="simple_consistency.yaml"),
    "cot": BaselineSpec("cot", "Chain-of-Thought T5", ChainOfThoughtT5, config_name="cot_t5.yaml"),
    "scratchpad": BaselineSpec("scratchpad", "Scratchpad T5", ScratchpadT5, config_name="scratchpad_t5.yaml"),
    "modular": BaselineSpec("modular", "Neural Module Network", ModularNeuralNetwork, config_name="modular_nn.yaml"),
    "symbolic": BaselineSpec("symbolic", "Symbolic rule-augmented T5", SymbolicRuleAugmentedT5, config_name="symbolic_rules.yaml"),
    "tinyllama_lora": BaselineSpec("tinyllama_lora", "TinyLlama 1.1B + LoRA", TinyLlamaBaseline, config_name="tinyllama_lora.yaml"),
}

BASELINE_ALIASES = {
    "vanilla": "reference_t5",
    "reference": "reference_t5",
    "random_init": "random_init_t5",
    "tree_linearized": "tree_linearized_t5",
    "nesy": "modular",
    "nmn": "modular",
    "llama": "tinyllama_lora",
    "llama_lora": "tinyllama_lora",
    "tinyllama": "tinyllama_lora",
}


def canonical_baseline_name(baseline_type: str) -> str:
    """Resolve a CLI/backward-compatible name to one registry key."""
    key = BASELINE_ALIASES.get(baseline_type, baseline_type)
    if key not in BASELINE_REGISTRY:
        available = ", ".join(BASELINE_REGISTRY)
        raise ValueError(f"Unknown baseline: {baseline_type}. Available: {available}")
    return key


def create_baseline(
    baseline_type: str,
    config: Optional[BaselineConfig] = None,
    dataset_type: str = "scan",
    **kwargs
) -> BaselineModel:
    """
    Factory function for creating baseline models.
    
    Args:
        baseline_type: One of "vanilla", "cot", "scratchpad", "nesy"/"modular",
            "symbolic", or "llama"
        config: Baseline configuration
        dataset_type: Dataset type for models that need dataset-specific behavior
                     (e.g., "scan", "cogs", "cfq", "clutrr", "gsm8k")
        **kwargs: Additional arguments for specific baselines
        
    Returns:
        Baseline model instance
    """
    config_fields = {
        key: kwargs.pop(key)
        for key in ("base_model", "max_source_length", "max_target_length")
        if key in kwargs
    }
    if config is None:
        config = BaselineConfig(**config_fields)
    elif config_fields:
        config = BaselineConfig(
            base_model=config_fields.get("base_model", config.base_model),
            max_source_length=config_fields.get("max_source_length", config.max_source_length),
            max_target_length=config_fields.get("max_target_length", config.max_target_length),
            tokenizer=config.tokenizer,
        )
    
    baseline_type = canonical_baseline_name(baseline_type)
    spec = BASELINE_REGISTRY[baseline_type]
    if dataset_type not in spec.supported_datasets:
        raise ValueError(f"{baseline_type} does not support dataset {dataset_type}")

    baseline_cls = spec.model_class
    if baseline_type in ("symbolic", "tinyllama_lora"):
        return baseline_cls(config, dataset_type=dataset_type, **kwargs)
    return baseline_cls(config, **kwargs)
