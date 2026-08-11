"""
Attention Pattern Visualization and Analysis

This module provides tools for visualizing and analyzing attention patterns
in the DAI transformer, which is critical for failure analysis and understanding
how the model processes compositional structures.

Key Features:
1. Attention heatmaps for individual examples
2. Aggregate attention pattern analysis
3. Compositional attention detection
4. Modular vs. holistic processing metrics
5. Failure pattern identification through attention

Reference: Clark et al. (2019) "What Does BERT Look At?"
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import json
import logging

import numpy as np
import torch
import torch.nn.functional as F

# Optional visualization imports
try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


logger = logging.getLogger(__name__)


# Token normalization helper - use everywhere for consistency
def normalize_token(token: str) -> str:
    """Normalize a token for comparison (handles sentencepiece, casing, etc.)."""
    return token.lower().replace("▁", "").replace("Ġ", "").strip()


# SCAN-style compositional operators for structure-aware attention
COMPOSITIONAL_OPERATORS = {
    "modifiers": ["twice", "thrice"],
    "conjunctions": ["and"],
    "temporal": ["after"],
    "spatial": ["around", "opposite"],
    "directional": ["left", "right"],
}

ALL_OPERATORS = set()
for ops in COMPOSITIONAL_OPERATORS.values():
    ALL_OPERATORS.update(ops)


@dataclass
class AttentionPattern:
    """Represents attention patterns for a single example."""
    
    # Raw attention weights: [num_layers, num_heads, seq_len, seq_len]
    encoder_attention: Optional[torch.Tensor] = None
    decoder_attention: Optional[torch.Tensor] = None
    cross_attention: Optional[torch.Tensor] = None
    
    # Input/output tokens
    input_tokens: List[str] = field(default_factory=list)
    output_tokens: List[str] = field(default_factory=list)
    
    # Metadata
    input_text: str = ""
    output_text: str = ""
    prediction: str = ""
    is_correct: bool = False
    
    # Computed metrics
    compositionality_score: float = 0.0
    modularity_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "input_text": self.input_text,
            "output_text": self.output_text,
            "prediction": self.prediction,
            "is_correct": self.is_correct,
            "compositionality_score": self.compositionality_score,
            "modularity_score": self.modularity_score,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


@dataclass 
class AttentionAnalysisResult:
    """Aggregated attention analysis results."""
    
    # Pattern statistics
    avg_compositionality_score: float = 0.0
    avg_modularity_score: float = 0.0
    
    # Correct vs incorrect comparison
    correct_compositionality: float = 0.0
    incorrect_compositionality: float = 0.0
    correct_modularity: float = 0.0
    incorrect_modularity: float = 0.0
    
    # Head specialization
    head_specializations: Dict[str, List[Tuple[int, int]]] = field(default_factory=dict)
    
    # Layer-wise statistics
    layer_compositionality: Dict[int, float] = field(default_factory=dict)
    layer_modularity: Dict[int, float] = field(default_factory=dict)
    
    # Attention entropy by layer
    layer_entropy: Dict[int, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "avg_compositionality_score": self.avg_compositionality_score,
            "avg_modularity_score": self.avg_modularity_score,
            "compositionality_gap": self.correct_compositionality - self.incorrect_compositionality,
            "modularity_gap": self.correct_modularity - self.incorrect_modularity,
            "layer_compositionality": self.layer_compositionality,
            "layer_entropy": self.layer_entropy,
        }


class AttentionVisualizer:
    """
    Visualize and analyze attention patterns.
    
    This class provides methods for:
    1. Generating attention heatmaps
    2. Computing compositionality metrics from attention
    3. Identifying modular vs holistic processing
    4. Comparing attention patterns between correct/incorrect predictions
    """
    
    def __init__(
        self,
        output_dir: Optional[Path] = None,
        figsize: Tuple[int, int] = (12, 8),
        cmap: str = "Blues",
    ):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save figures
            figsize: Default figure size
            cmap: Colormap for heatmaps
        """
        self.output_dir = Path(output_dir) if output_dir else Path("./attention_viz")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figsize = figsize
        self.cmap = cmap
    
    def extract_attention(
        self,
        model,
        tokenizer,
        input_text: str,
        target_text: Optional[str] = None,
        device: Optional[torch.device] = None,
    ) -> AttentionPattern:
        """
        Extract attention patterns from model for a single example.
        
        Args:
            model: The DAI transformer model
            tokenizer: Tokenizer for encoding
            input_text: Input text
            target_text: Target text (for teacher forcing to get decoder/cross attention)
            device: Device for computation
            
        Returns:
            AttentionPattern with encoder, decoder, and cross attention weights
        """
        device = device or next(model.parameters()).device
        
        # Tokenize input
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get actual sequence length (excluding padding)
        input_seq_len = int(inputs["attention_mask"][0].sum().item())
        
        # Get input tokens and slice to actual length
        all_input_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        input_tokens = all_input_tokens[:input_seq_len]
        
        # Tokenize target if provided (for decoder/cross attention)
        labels = None
        output_tokens = []
        output_seq_len = 0
        if target_text is not None:
            target_inputs = tokenizer(
                target_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            )
            labels = target_inputs["input_ids"].to(device)
            output_seq_len = int(target_inputs["attention_mask"][0].sum().item())
            all_output_tokens = tokenizer.convert_ids_to_tokens(labels[0])
            output_tokens = all_output_tokens[:output_seq_len]
        
        encoder_attention = None
        decoder_attention = None
        cross_attention = None
        
        # Forward with attention outputs
        with torch.no_grad():
            if hasattr(model, 't5'):
                # T5-based model - pass labels for teacher forcing
                outputs = model.t5(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    labels=labels,
                    output_attentions=True,
                    return_dict=True,
                )
                
                # Extract encoder attention and slice to actual seq len
                if outputs.encoder_attentions is not None:
                    encoder_attention = torch.stack(outputs.encoder_attentions)
                    # Slice to actual sequence length: [layers, heads, seq, seq]
                    encoder_attention = encoder_attention[:, :, :input_seq_len, :input_seq_len]
                
                # Extract decoder attention if available
                if hasattr(outputs, 'decoder_attentions') and outputs.decoder_attentions is not None:
                    decoder_attention = torch.stack(outputs.decoder_attentions)
                    if output_seq_len > 0:
                        decoder_attention = decoder_attention[:, :, :output_seq_len, :output_seq_len]
                
                # Extract cross attention if available
                if hasattr(outputs, 'cross_attentions') and outputs.cross_attentions is not None:
                    cross_attention = torch.stack(outputs.cross_attentions)
                    if output_seq_len > 0:
                        cross_attention = cross_attention[:, :, :output_seq_len, :input_seq_len]
            else:
                # Generic model - try to get attention
                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    labels=labels,
                    output_attentions=True,
                    return_dict=True,
                )
                
                if getattr(outputs, 'encoder_attentions', None) is not None:
                    encoder_attention = torch.stack(outputs.encoder_attentions)
                    encoder_attention = encoder_attention[:, :, :input_seq_len, :input_seq_len]
                
                if getattr(outputs, 'decoder_attentions', None) is not None:
                    decoder_attention = torch.stack(outputs.decoder_attentions)
                    if output_seq_len > 0:
                        decoder_attention = decoder_attention[:, :, :output_seq_len, :output_seq_len]
                
                if getattr(outputs, 'cross_attentions', None) is not None:
                    cross_attention = torch.stack(outputs.cross_attentions)
                    if output_seq_len > 0:
                        cross_attention = cross_attention[:, :, :output_seq_len, :input_seq_len]
        
        return AttentionPattern(
            encoder_attention=encoder_attention,
            decoder_attention=decoder_attention,
            cross_attention=cross_attention,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_text=input_text,
            output_text=target_text or "",
        )
    
    def plot_attention_heatmap(
        self,
        attention: torch.Tensor,
        tokens: List[str],
        layer: int = -1,
        head: Optional[int] = None,
        title: str = "Attention Pattern",
        save_path: Optional[Path] = None,
        show: bool = True,
    ) -> Optional[Figure]:
        """
        Plot attention heatmap for a specific layer/head.
        
        Args:
            attention: Attention tensor [num_layers, num_heads, seq, seq] or [seq, seq]
            tokens: Token labels
            layer: Which layer to visualize (-1 for last)
            head: Which head to visualize (None for average)
            title: Plot title
            save_path: Path to save figure
            show: Whether to display figure
            
        Returns:
            matplotlib Figure if available
        """
        if not HAS_MATPLOTLIB:
            logger.warning("matplotlib not available for visualization")
            return None
        
        # Extract specific layer/head
        if attention.dim() == 4:
            attn = attention[layer]  # [num_heads, seq, seq]
            if head is not None:
                attn = attn[head]  # [seq, seq]
                head_str = f" (head {head})"
            else:
                attn = attn.mean(dim=0)  # Average over heads
                head_str = " (avg)"
        elif attention.dim() == 3:
            if head is not None:
                attn = attention[head]
                head_str = f" (head {head})"
            else:
                attn = attention.mean(dim=0)
                head_str = " (avg)"
        else:
            attn = attention
            head_str = ""
        
        attn = attn.cpu().numpy()
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if HAS_SEABORN:
            sns.heatmap(
                attn,
                xticklabels=tokens,
                yticklabels=tokens,
                cmap=self.cmap,
                ax=ax,
                square=True,
                cbar_kws={"shrink": 0.8},
            )
        else:
            im = ax.imshow(attn, cmap=self.cmap, aspect='auto')
            ax.set_xticks(range(len(tokens)))
            ax.set_yticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=45, ha='right')
            ax.set_yticklabels(tokens)
            plt.colorbar(im, ax=ax, shrink=0.8)
        
        ax.set_title(f"{title} - Layer {layer}{head_str}")
        ax.set_xlabel("Key")
        ax.set_ylabel("Query")
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved attention heatmap to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def plot_multi_head_attention(
        self,
        attention: torch.Tensor,
        tokens: List[str],
        layer: int = -1,
        max_heads: int = 8,
        title: str = "Multi-Head Attention",
        save_path: Optional[Path] = None,
        show: bool = True,
    ) -> Optional[Figure]:
        """
        Plot attention patterns for multiple heads in a grid.
        
        Args:
            attention: Attention tensor [num_layers, num_heads, seq, seq]
            tokens: Token labels
            layer: Which layer to visualize
            max_heads: Maximum number of heads to show
            title: Plot title
            save_path: Path to save figure
            show: Whether to display
            
        Returns:
            matplotlib Figure
        """
        if not HAS_MATPLOTLIB:
            return None
        
        if attention.dim() < 4:
            logger.warning("Multi-head plot requires 4D attention tensor")
            return None
        
        attn = attention[layer].cpu().numpy()  # [num_heads, seq, seq]
        num_heads = min(attn.shape[0], max_heads)
        
        # Create grid
        cols = min(4, num_heads)
        rows = (num_heads + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        axes = np.atleast_2d(axes)
        
        for i in range(num_heads):
            row, col = i // cols, i % cols
            ax = axes[row, col]
            
            if HAS_SEABORN:
                sns.heatmap(
                    attn[i],
                    xticklabels=tokens if i >= (rows-1)*cols else False,
                    yticklabels=tokens if col == 0 else False,
                    cmap=self.cmap,
                    ax=ax,
                    cbar=False,
                    square=True,
                )
            else:
                ax.imshow(attn[i], cmap=self.cmap, aspect='auto')
            
            ax.set_title(f"Head {i}")
        
        # Hide empty subplots
        for i in range(num_heads, rows * cols):
            row, col = i // cols, i % cols
            axes[row, col].axis('off')
        
        fig.suptitle(f"{title} - Layer {layer}", fontsize=14)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def compute_compositionality_score(
        self,
        attention: torch.Tensor,
        tokens: List[str],
        primitive_positions: Optional[List[int]] = None,
    ) -> float:
        """
        Compute compositionality score from attention patterns.
        
        High compositionality: Attention focuses on compositional relationships
        (e.g., modifier attending to operand, conjunction attending to both sides)
        
        Low compositionality: Attention is diffuse, purely positional, or Markov-like
        
        Uses structure-aware metrics instead of simple locality:
        1. Operator-operand attention: Do modifiers (twice/thrice) attend to actions?
        2. Conjunction symmetry: Do conjunctions (and) attend to both sides?
        3. Temporal binding: Does "after" properly attend to both clauses?
        4. Sparsity: Is attention focused rather than uniform?
        
        Args:
            attention: Attention tensor [num_layers, num_heads, seq, seq]
            tokens: Token list (already sliced to actual length, no padding)
            primitive_positions: Known positions of primitive tokens
            
        Returns:
            Compositionality score [0, 1]
        """
        if attention is None:
            return 0.0
        
        # Use last layer average
        if attention.dim() == 4:
            attn = attention[-1].mean(dim=0)  # [seq, seq]
        else:
            attn = attention.mean(dim=0) if attention.dim() == 3 else attention
        
        attn = attn.cpu().numpy()
        seq_len = attn.shape[0]
        
        # Normalize tokens for matching
        norm_tokens = [normalize_token(t) for t in tokens]
        
        # Find operator positions
        modifier_positions = []  # twice, thrice
        conjunction_positions = []  # and
        temporal_positions = []  # after
        action_positions = []  # jump, walk, run, look, turn
        
        action_words = {"jump", "walk", "run", "look", "turn"}
        
        for i, tok in enumerate(norm_tokens):
            if tok in COMPOSITIONAL_OPERATORS.get("modifiers", []):
                modifier_positions.append(i)
            elif tok in COMPOSITIONAL_OPERATORS.get("conjunctions", []):
                conjunction_positions.append(i)
            elif tok in COMPOSITIONAL_OPERATORS.get("temporal", []):
                temporal_positions.append(i)
            elif tok in action_words:
                action_positions.append(i)
        
        scores = []
        
        # Metric 1: Modifier-operand attention
        # Modifiers (twice/thrice) should attend to their operand (preceding action/phrase)
        if modifier_positions and action_positions:
            modifier_operand_attention = 0.0
            count = 0
            for mod_pos in modifier_positions:
                # Modifier should attend to tokens before it (its scope)
                if mod_pos > 0:
                    # Find nearest action before modifier
                    preceding_actions = [a for a in action_positions if a < mod_pos]
                    if preceding_actions:
                        # Attention from modifier to its operand span
                        operand_end = mod_pos
                        operand_start = max(0, preceding_actions[-1])
                        if operand_start < operand_end:
                            modifier_operand_attention += attn[mod_pos, operand_start:operand_end].sum()
                            modifier_operand_attention += attn[operand_start:operand_end, mod_pos].sum()
                            count += 1
            
            if count > 0:
                # Normalize by total attention from/to modifier positions
                total_modifier_attn = sum(attn[m, :].sum() + attn[:, m].sum() for m in modifier_positions)
                modifier_score = modifier_operand_attention / (total_modifier_attn + 1e-10)
                scores.append(min(1.0, modifier_score * 2))  # Scale up since we want ~0.5 to be good
        
        # Metric 2: Conjunction symmetry
        # "and" should attend to tokens on both left and right
        if conjunction_positions:
            symmetry_scores = []
            for conj_pos in conjunction_positions:
                if 0 < conj_pos < seq_len - 1:
                    left_attention = attn[conj_pos, :conj_pos].sum()
                    right_attention = attn[conj_pos, conj_pos+1:].sum()
                    
                    total = left_attention + right_attention
                    if total > 1e-10:
                        # Perfect symmetry = 0.5/0.5, measure deviation from symmetry
                        balance = min(left_attention, right_attention) / (max(left_attention, right_attention) + 1e-10)
                        symmetry_scores.append(balance)
            
            if symmetry_scores:
                scores.append(np.mean(symmetry_scores))
        
        # Metric 3: Temporal binding for "after"
        # "X after Y" means Y happens first, X second
        # "after" should attend to both X (before it) and Y (after it)
        if temporal_positions:
            temporal_binding_scores = []
            for temp_pos in temporal_positions:
                if 0 < temp_pos < seq_len - 1:
                    left_attention = attn[temp_pos, :temp_pos].sum()
                    right_attention = attn[temp_pos, temp_pos+1:].sum()
                    
                    total = left_attention + right_attention
                    if total > 1e-10:
                        # Should attend to both sides
                        min_side = min(left_attention, right_attention)
                        binding_score = min_side / (total / 2 + 1e-10)  # 1.0 if equal attention both sides
                        temporal_binding_scores.append(min(1.0, binding_score))
            
            if temporal_binding_scores:
                scores.append(np.mean(temporal_binding_scores))
        
        # Metric 4: Attention sparsity (compositional = focused, not uniform)
        attention_flat = attn.flatten()
        attention_flat = attention_flat / (attention_flat.sum() + 1e-10)
        entropy = -np.sum(attention_flat * np.log(attention_flat + 1e-10))
        max_entropy = np.log(seq_len * seq_len)
        sparsity_score = 1 - (entropy / max_entropy)
        scores.append(sparsity_score)
        
        # Metric 5: Off-diagonal attention (compositional = looks at related tokens, not self)
        diagonal_attention = np.trace(attn) / seq_len
        off_diagonal_ratio = 1 - diagonal_attention
        scores.append(off_diagonal_ratio)
        
        # If no operators found, fall back to just sparsity + off-diagonal
        if len(scores) == 2:
            return float(np.clip(np.mean(scores), 0, 1))
        
        # Combine all scores
        compositionality = np.mean(scores)
        
        return float(np.clip(compositionality, 0, 1))
    
    def compute_modularity_score(
        self,
        attention: torch.Tensor,
        tokens: List[str],
        conjunction_tokens: List[str] = ["and", "after", ","],
    ) -> float:
        """
        Compute modularity score from attention patterns.
        
        High modularity: Sub-expressions are processed independently
        (attention is block-diagonal around conjunctions)
        
        Low modularity: Holistic processing (all tokens attend to all)
        
        Args:
            attention: Attention tensor
            tokens: Token list (already sliced, no padding)
            conjunction_tokens: Tokens that separate modules
            
        Returns:
            Modularity score [0, 1]
        """
        if attention is None:
            return 0.0
        
        # Use last layer average
        if attention.dim() == 4:
            attn = attention[-1].mean(dim=0)
        else:
            attn = attention.mean(dim=0) if attention.dim() == 3 else attention
        
        attn = attn.cpu().numpy()
        seq_len = attn.shape[0]
        
        # Normalize tokens for matching
        norm_tokens = [normalize_token(t) for t in tokens]
        
        # Find conjunction positions using normalized tokens
        conj_positions = []
        for i, tok in enumerate(norm_tokens):
            if tok in conjunction_tokens:
                conj_positions.append(i)
        
        if len(conj_positions) == 0:
            # No conjunctions - check if attention is still modular
            # High modularity = attention concentrated on local context
            local_window = max(1, seq_len // 4)
            local_attention = 0
            for i in range(seq_len):
                for j in range(max(0, i-local_window), min(seq_len, i+local_window+1)):
                    local_attention += attn[i, j]
            return float(local_attention / (attn.sum() + 1e-10))
        
        # Create blocks based on conjunction positions
        boundaries = [0] + conj_positions + [seq_len]
        
        # Compute within-block vs between-block attention
        within_block = 0
        between_block = 0
        
        for b in range(len(boundaries) - 1):
            start, end = boundaries[b], boundaries[b + 1]
            
            # Within block
            within_block += attn[start:end, start:end].sum()
            
            # Between blocks
            for b2 in range(len(boundaries) - 1):
                if b2 != b:
                    start2, end2 = boundaries[b2], boundaries[b2 + 1]
                    between_block += attn[start:end, start2:end2].sum()
        
        total = within_block + between_block
        if total < 1e-10:
            return 0.0
        
        modularity = within_block / total
        
        return float(np.clip(modularity, 0, 1))
    
    def analyze_head_specialization(
        self,
        attention: torch.Tensor,
        tokens: List[str],
    ) -> Dict[str, List[Tuple[int, int]]]:
        """
        Identify specialized attention heads using percentile-based thresholds.
        
        Uses adaptive thresholds based on the distribution of attention patterns
        across all heads, rather than fixed thresholds that don't scale with
        sequence length.
        
        Categories:
        - positional: Attends to fixed positions (e.g., first/last token)
        - local: Attends to nearby tokens (diagonal-heavy)
        - separator: Attends to separators (punctuation, conjunctions)
        - broad: Diffuse attention over many tokens
        
        Args:
            attention: Attention tensor [num_layers, num_heads, seq, seq]
            tokens: Token list (already sliced, no padding)
            
        Returns:
            Dict mapping specialization type to list of (layer, head) tuples
        """
        if attention is None or attention.dim() != 4:
            return {}
        
        num_layers, num_heads, seq_len, _ = attention.shape
        attn = attention.cpu().numpy()
        
        # Normalize tokens
        norm_tokens = [normalize_token(t) for t in tokens]
        
        # First pass: compute metrics for all heads to establish thresholds
        all_positional_scores = []
        all_local_scores = []
        all_separator_scores = []
        
        separator_words = {"and", "after", ",", ".", ";"}
        sep_positions = [i for i, t in enumerate(norm_tokens) if t in separator_words]
        
        head_metrics = []
        
        for layer in range(num_layers):
            for head in range(num_heads):
                head_attn = attn[layer, head]
                
                # Positional: max column mass (how concentrated on single positions)
                position_attention = head_attn.sum(axis=0)
                total_attn = head_attn.sum() + 1e-10
                max_pos_ratio = position_attention.max() / total_attn
                all_positional_scores.append(max_pos_ratio)
                
                # Local: diagonal mass
                local_window = min(2, seq_len // 4) if seq_len > 4 else 1
                local_attention = 0
                for i in range(seq_len):
                    for j in range(max(0, i-local_window), min(seq_len, i+local_window+1)):
                        local_attention += head_attn[i, j]
                local_ratio = local_attention / total_attn
                all_local_scores.append(local_ratio)
                
                # Separator: attention to separator positions
                if sep_positions:
                    sep_attention = head_attn[:, sep_positions].sum()
                    sep_ratio = sep_attention / total_attn
                else:
                    sep_ratio = 0.0
                all_separator_scores.append(sep_ratio)
                
                head_metrics.append({
                    "layer": layer,
                    "head": head,
                    "positional": max_pos_ratio,
                    "local": local_ratio,
                    "separator": sep_ratio,
                })
        
        # Compute percentile-based thresholds
        # Use 75th percentile as threshold - heads above this are "specialized"
        positional_threshold = np.percentile(all_positional_scores, 75)
        local_threshold = np.percentile(all_local_scores, 75)
        separator_threshold = np.percentile(all_separator_scores, 75) if sep_positions else 1.0
        
        # Ensure minimum thresholds to avoid classifying everything
        positional_threshold = max(positional_threshold, 0.15)
        local_threshold = max(local_threshold, 0.4)
        separator_threshold = max(separator_threshold, 0.1)
        
        specializations = {
            "positional": [],
            "local": [],
            "separator": [],
            "broad": [],
        }
        
        # Second pass: classify using adaptive thresholds
        for metrics in head_metrics:
            layer, head = metrics["layer"], metrics["head"]
            
            # Classify by strongest specialization (if above threshold)
            if metrics["positional"] > positional_threshold and metrics["positional"] > metrics["local"]:
                specializations["positional"].append((layer, head))
            elif metrics["local"] > local_threshold:
                specializations["local"].append((layer, head))
            elif sep_positions and metrics["separator"] > separator_threshold:
                specializations["separator"].append((layer, head))
            else:
                specializations["broad"].append((layer, head))
        
        return specializations
    
    def _classify_head(
        self,
        head_attn: np.ndarray,
        tokens: List[str],
        thresholds: Optional[Dict[str, float]] = None,
    ) -> str:
        """
        Classify a single attention head.
        
        Uses provided thresholds or sensible defaults. For batch classification
        with adaptive thresholds, use analyze_head_specialization() instead.
        
        Args:
            head_attn: Attention matrix [seq, seq]
            tokens: Token list
            thresholds: Optional dict with 'positional', 'local', 'separator' thresholds
            
        Returns:
            Classification string
        """
        seq_len = head_attn.shape[0]
        thresholds = thresholds or {"positional": 0.2, "local": 0.5, "separator": 0.15}
        
        total_attn = head_attn.sum() + 1e-10
        
        # Normalize tokens
        norm_tokens = [normalize_token(t) for t in tokens]
        
        # Check for positional pattern (high attention to specific positions)
        position_attention = head_attn.sum(axis=0)
        max_pos_ratio = position_attention.max() / total_attn
        if max_pos_ratio > thresholds["positional"]:
            return "positional"
        
        # Check for local pattern (diagonal-heavy)
        local_window = min(2, seq_len // 4) if seq_len > 4 else 1
        local_attention = 0
        for i in range(seq_len):
            for j in range(max(0, i-local_window), min(seq_len, i+local_window+1)):
                local_attention += head_attn[i, j]
        
        if local_attention / total_attn > thresholds["local"]:
            return "local"
        
        # Check for separator pattern
        separator_words = {"and", "after", ",", ".", ";"}
        sep_positions = [i for i, t in enumerate(norm_tokens) if t in separator_words]
        
        if sep_positions:
            sep_attention = head_attn[:, sep_positions].sum()
            if sep_attention / total_attn > thresholds["separator"]:
                return "separator"
        
        # Default: broad attention
        return "broad"
    
    def compare_correct_vs_incorrect(
        self,
        patterns: List[AttentionPattern],
    ) -> Dict[str, float]:
        """
        Compare attention patterns between correct and incorrect predictions.
        
        Args:
            patterns: List of AttentionPattern objects with is_correct set
            
        Returns:
            Dict with comparison statistics
        """
        correct_patterns = [p for p in patterns if p.is_correct]
        incorrect_patterns = [p for p in patterns if not p.is_correct]
        
        if not correct_patterns or not incorrect_patterns:
            return {"error": "Need both correct and incorrect examples"}
        
        # Compute average scores
        correct_comp = np.mean([p.compositionality_score for p in correct_patterns])
        incorrect_comp = np.mean([p.compositionality_score for p in incorrect_patterns])
        
        correct_mod = np.mean([p.modularity_score for p in correct_patterns])
        incorrect_mod = np.mean([p.modularity_score for p in incorrect_patterns])
        
        return {
            "correct_compositionality": float(correct_comp),
            "incorrect_compositionality": float(incorrect_comp),
            "compositionality_gap": float(correct_comp - incorrect_comp),
            "correct_modularity": float(correct_mod),
            "incorrect_modularity": float(incorrect_mod),
            "modularity_gap": float(correct_mod - incorrect_mod),
            "num_correct": len(correct_patterns),
            "num_incorrect": len(incorrect_patterns),
        }
    
    def generate_attention_report(
        self,
        patterns: List[AttentionPattern],
        output_path: Optional[Path] = None,
    ) -> AttentionAnalysisResult:
        """
        Generate comprehensive attention analysis report.
        
        Args:
            patterns: List of AttentionPattern objects
            output_path: Path to save JSON report
            
        Returns:
            AttentionAnalysisResult with all statistics
        """
        if not patterns:
            return AttentionAnalysisResult()
        
        # Compute average scores
        comp_scores = [p.compositionality_score for p in patterns]
        mod_scores = [p.modularity_score for p in patterns]
        
        result = AttentionAnalysisResult(
            avg_compositionality_score=float(np.mean(comp_scores)),
            avg_modularity_score=float(np.mean(mod_scores)),
        )
        
        # Correct vs incorrect
        comparison = self.compare_correct_vs_incorrect(patterns)
        result.correct_compositionality = comparison.get("correct_compositionality", 0)
        result.incorrect_compositionality = comparison.get("incorrect_compositionality", 0)
        result.correct_modularity = comparison.get("correct_modularity", 0)
        result.incorrect_modularity = comparison.get("incorrect_modularity", 0)
        
        # Head specialization (from first pattern with attention)
        for p in patterns:
            if p.encoder_attention is not None:
                result.head_specializations = self.analyze_head_specialization(
                    p.encoder_attention, p.input_tokens
                )
                break
        
        # Save report
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
            logger.info(f"Saved attention report to {output_path}")
        
        return result


def visualize_compositional_attention(
    model,
    tokenizer,
    examples: List[Dict[str, str]],
    output_dir: Path,
    device: Optional[torch.device] = None,
) -> AttentionAnalysisResult:
    """
    Convenience function to analyze attention for a list of examples.
    
    Extracts encoder, decoder, and cross-attention when target is provided.
    Computes structure-aware compositionality and modularity metrics.
    
    Args:
        model: DAI transformer model
        tokenizer: Tokenizer
        examples: List of {"input": ..., "target": ..., "prediction": ..., "correct": bool}
        output_dir: Directory to save visualizations
        device: Computation device
        
    Returns:
        AttentionAnalysisResult
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    visualizer = AttentionVisualizer(output_dir=output_dir)
    patterns = []
    
    for i, example in enumerate(examples):
        pattern = visualizer.extract_attention(
            model=model,
            tokenizer=tokenizer,
            input_text=example["input"],
            target_text=example.get("target"),  # Pass target for decoder/cross attention
            device=device,
        )
        
        pattern.prediction = example.get("prediction", "")
        pattern.is_correct = example.get("correct", False)
        
        # Compute metrics using encoder attention
        if pattern.encoder_attention is not None:
            pattern.compositionality_score = visualizer.compute_compositionality_score(
                pattern.encoder_attention, pattern.input_tokens
            )
            pattern.modularity_score = visualizer.compute_modularity_score(
                pattern.encoder_attention, pattern.input_tokens
            )
            
            # Save encoder attention visualization for first few examples
            if i < 5:
                visualizer.plot_attention_heatmap(
                    pattern.encoder_attention,
                    pattern.input_tokens,
                    title=f"Encoder Attn - Example {i}: {'✓' if pattern.is_correct else '✗'}",
                    save_path=output_dir / f"encoder_attention_example_{i}.png",
                    show=False,
                )
        
        # Also visualize cross-attention if available (shows input-output alignment)
        if pattern.cross_attention is not None and i < 5:
            # Cross attention: [layers, heads, output_seq, input_seq]
            cross_attn = pattern.cross_attention[-1].mean(dim=0).cpu().numpy()
            
            if HAS_MATPLOTLIB:
                fig, ax = plt.subplots(figsize=(12, 8))
                if HAS_SEABORN:
                    sns.heatmap(
                        cross_attn,
                        xticklabels=pattern.input_tokens,
                        yticklabels=pattern.output_tokens,
                        cmap="Blues",
                        ax=ax,
                        square=False,
                    )
                else:
                    im = ax.imshow(cross_attn, cmap="Blues", aspect='auto')
                    ax.set_xticks(range(len(pattern.input_tokens)))
                    ax.set_yticks(range(len(pattern.output_tokens)))
                    ax.set_xticklabels(pattern.input_tokens, rotation=45, ha='right')
                    ax.set_yticklabels(pattern.output_tokens)
                    plt.colorbar(im, ax=ax)
                
                ax.set_title(f"Cross Attn - Example {i}: {'✓' if pattern.is_correct else '✗'}")
                ax.set_xlabel("Input (Key)")
                ax.set_ylabel("Output (Query)")
                plt.tight_layout()
                fig.savefig(output_dir / f"cross_attention_example_{i}.png", dpi=150, bbox_inches='tight')
                plt.close()
        
        patterns.append(pattern)
    
    # Generate report
    result = visualizer.generate_attention_report(
        patterns,
        output_path=output_dir / "attention_analysis.json",
    )
    
    return result
