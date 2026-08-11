"""
Compositional Metrics for Evaluating Generalization

This module provides specialized metrics for measuring compositional generalization:
1. Per-composition-depth accuracy (how well does the model handle deeper compositions?)
2. Per-primitive accuracy (which primitives are handled well/poorly?)
3. Novel composition detection (identify truly novel test compositions)
4. Compositional distance metrics (how "far" is a test example from training?)

These metrics are essential for understanding WHERE compositional generalization fails,
not just WHETHER it fails.

Reference: Keysers et al. (2020) "Measuring Compositional Generalization"
"""

from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import re

import numpy as np
import torch

# Optional SciPy import for statistical analysis
try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@dataclass
class CompositionalAnalysis:
    """
    Comprehensive compositional analysis results.
    """
    # Overall metrics
    overall_accuracy: float = 0.0
    overall_exact_match: float = 0.0
    
    # By depth
    accuracy_by_depth: Dict[int, float] = field(default_factory=dict)
    count_by_depth: Dict[int, int] = field(default_factory=dict)
    
    # By primitive
    accuracy_by_primitive: Dict[str, float] = field(default_factory=dict)
    count_by_primitive: Dict[str, int] = field(default_factory=dict)
    
    # Conditional primitive stats: accuracy by (primitive, context)
    # Context = "simple" | "conjunction" | "modified"
    accuracy_by_primitive_context: Dict[Tuple[str, str], float] = field(default_factory=dict)
    count_by_primitive_context: Dict[Tuple[str, str], int] = field(default_factory=dict)
    
    # By composition type
    accuracy_by_composition: Dict[str, float] = field(default_factory=dict)
    count_by_composition: Dict[str, int] = field(default_factory=dict)
    
    # Novelty analysis
    novel_composition_accuracy: float = 0.0
    seen_composition_accuracy: float = 0.0
    novel_count: int = 0
    seen_count: int = 0
    
    # Compositional distance (real distance, not just novelty)
    accuracy_by_distance: Dict[int, float] = field(default_factory=dict)
    count_by_distance: Dict[int, int] = field(default_factory=dict)
    
    # Error categorization
    error_types: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        result = {
            "overall_accuracy": self.overall_accuracy,
            "overall_exact_match": self.overall_exact_match,
            "accuracy_by_depth": self.accuracy_by_depth,
            "accuracy_by_primitive": self.accuracy_by_primitive,
            "accuracy_by_composition": self.accuracy_by_composition,
            "novel_composition_accuracy": self.novel_composition_accuracy,
            "seen_composition_accuracy": self.seen_composition_accuracy,
            "generalization_gap": self.seen_composition_accuracy - self.novel_composition_accuracy,
            "accuracy_by_distance": self.accuracy_by_distance,
            "error_types": self.error_types,
        }
        # Add conditional primitive stats (convert tuple keys to strings)
        if self.accuracy_by_primitive_context:
            result["accuracy_by_primitive_context"] = {
                f"{prim}:{ctx}": acc
                for (prim, ctx), acc in self.accuracy_by_primitive_context.items()
            }
        return result


class CompositionParser:
    """
    Parse input sequences to extract compositional structure.
    
    Supports multiple dataset formats:
    - SCAN: "jump around left twice"
    - COGS: Logical forms with nested structure
    - CFQ: SPARQL queries
    """
    
    # SCAN primitives and modifiers
    SCAN_PRIMITIVES = {"jump", "walk", "run", "look", "turn"}
    SCAN_DIRECTIONS = {"left", "right"}
    # Note: 'and' and 'after' are conjunctions, not modifiers (avoid double-counting)
    SCAN_MODIFIERS = {"twice", "thrice", "around", "opposite"}
    SCAN_CONJUNCTIONS = {"and", "after"}
    
    # COGS primitives
    COGS_PRIMITIVES = {"agent", "theme", "recipient", "goal", "source", "location"}
    
    def __init__(self, dataset_type: str = "scan"):
        self.dataset_type = dataset_type.lower()
    
    def get_depth(self, text: str) -> int:
        """
        Compute compositional depth of an input.
        
        Depth = maximum nesting level of compositions.
        """
        if self.dataset_type == "scan":
            return self._scan_depth(text)
        elif self.dataset_type == "cogs":
            return self._cogs_depth(text)
        elif self.dataset_type == "cfq":
            return self._cfq_depth(text)
        else:
            return self._generic_depth(text)
    
    def _scan_depth(self, text: str) -> int:
        """
        SCAN depth based on TRUE operator nesting (recursive parsing).
        
        This measures how deeply nested the compositional structure is,
        not just a count of modifiers/tokens.
        
        Examples:
        - "jump" → depth 1 (base primitive)
        - "jump twice" → depth 2 (unary op on primitive)
        - "jump around left" → depth 2 (unary op on primitive)
        - "jump around left twice" → depth 3 (unary op on unary op)
        - "jump twice and walk" → depth 3 (binary op over depth-2 and depth-1)
        - "jump twice and walk twice" → depth 3 (binary op over two depth-2)
        """
        return self._scan_depth_recursive(text.lower())
    
    def _scan_depth_recursive(self, text: str) -> int:
        """
        Recursively compute true nesting depth for SCAN.
        
        Operator precedence (lowest to highest):
        1. Binary ops: and, after (split first)
        2. Unary repetition: twice, thrice (at end)
        3. Unary directional: around, opposite (in middle)
        4. Base primitive + optional direction
        """
        tokens = text.strip().split()
        if not tokens:
            return 0
        
        # Priority 1: Split on binary operators (lowest precedence)
        for op in ["and", "after"]:
            if op in tokens:
                i = tokens.index(op)
                left = " ".join(tokens[:i])
                right = " ".join(tokens[i+1:])
                left_depth = self._scan_depth_recursive(left) if left else 0
                right_depth = self._scan_depth_recursive(right) if right else 0
                # Binary op adds 1 depth over the max of its children
                return 1 + max(left_depth, right_depth)
        
        # Priority 2: Handle trailing unary repetition (twice/thrice)
        if tokens and tokens[-1] in {"twice", "thrice"}:
            base = " ".join(tokens[:-1])
            return 1 + self._scan_depth_recursive(base)
        
        # Priority 3: Handle unary around/opposite (higher precedence than repetition)
        if "around" in tokens:
            i = tokens.index("around")
            base = " ".join(tokens[:i])  # e.g., "jump" in "jump around left"
            # The "around left" is an operator phrase, base is what it operates on
            return 1 + self._scan_depth_recursive(base) if base else 1
        
        if "opposite" in tokens:
            i = tokens.index("opposite")
            base = " ".join(tokens[:i])  # e.g., "turn" in "turn opposite left"
            return 1 + self._scan_depth_recursive(base) if base else 1
        
        # Priority 4: Base case - primitive with optional direction
        # "jump", "jump left", "turn right" are all depth 1
        return 1
    
    def _cogs_depth(self, text: str) -> int:
        """COGS depth based on nesting of semantic roles."""
        # Count nested parentheses
        max_depth = 0
        current_depth = 0
        for char in text:
            if char == '(':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == ')':
                current_depth -= 1
        return max(1, max_depth)
    
    def _cfq_depth(self, text: str) -> int:
        """CFQ depth based on SPARQL structure."""
        # Count nested clauses
        depth = 1
        depth += text.count("FILTER")
        depth += text.count("OPTIONAL")
        depth += text.count("{")
        return depth
    
    def _generic_depth(self, text: str) -> int:
        """Generic depth based on token count heuristic."""
        return len(text.split()) // 3 + 1
    
    def get_primitives(self, text: str) -> List[str]:
        """Extract primitive elements from input."""
        if self.dataset_type == "scan":
            return self._scan_primitives(text)
        elif self.dataset_type == "cogs":
            return self._cogs_primitives(text)
        else:
            return self._generic_primitives(text)
    
    def _scan_primitives(self, text: str) -> List[str]:
        """Extract SCAN primitives."""
        tokens = text.lower().split()
        primitives = []
        for token in tokens:
            if token in self.SCAN_PRIMITIVES:
                primitives.append(token)
        return primitives
    
    def _cogs_primitives(self, text: str) -> List[str]:
        """Extract COGS semantic roles."""
        primitives = []
        for prim in self.COGS_PRIMITIVES:
            if prim in text.lower():
                primitives.append(prim)
        return primitives
    
    def _generic_primitives(self, text: str) -> List[str]:
        """Generic primitive extraction."""
        # Return content words (simple heuristic)
        stop_words = {"the", "a", "an", "is", "are", "to", "of", "and", "or"}
        tokens = text.lower().split()
        return [t for t in tokens if t not in stop_words and len(t) > 2]
    
    def get_composition_type(self, text: str) -> str:
        """
        Identify the type of composition.
        
        Returns a string describing the composition pattern.
        """
        if self.dataset_type == "scan":
            return self._scan_composition_type(text)
        else:
            return "generic"
    
    def _scan_composition_type(self, text: str) -> str:
        """Classify SCAN composition type."""
        tokens = text.lower().split()
        
        if "and" in tokens:
            return "conjunction"
        elif "after" in tokens:
            return "sequence"
        elif "around" in tokens:
            return "around_modifier"
        elif "opposite" in tokens:
            return "opposite_modifier"
        elif "twice" in tokens or "thrice" in tokens:
            return "repetition"
        elif any(d in tokens for d in self.SCAN_DIRECTIONS):
            return "directional"
        else:
            return "simple"
    
    def get_sub_constituents(self, text: str) -> List[str]:
        """
        Extract sub-constituents of a compositional expression.
        
        For the loss term: γ(h_x) vs α(γ(h_x1), γ(h_x2))
        We need to identify x1, x2 given x.
        """
        if self.dataset_type == "scan":
            return self._scan_sub_constituents(text)
        else:
            return []
    
    def _scan_sub_constituents(self, text: str) -> List[str]:
        """
        Extract SCAN sub-constituents.
        
        Examples:
        - "jump twice" → ["jump"]
        - "jump around left" → ["jump", "around left"]
        - "jump and walk" → ["jump", "walk"]
        - "jump twice and walk" → ["jump twice", "walk"]
        - "jump opposite left" → ["jump", "opposite left"]
        - "jump around left twice" → ["jump around left"]
        """
        tokens = text.lower().split()
        constituents = []
        
        # Priority 1: Handle conjunctions (highest level split)
        if "and" in tokens:
            idx = tokens.index("and")
            left = " ".join(tokens[:idx])
            right = " ".join(tokens[idx+1:])
            if left:
                constituents.append(left)
            if right:
                constituents.append(right)
            return constituents
        
        if "after" in tokens:
            idx = tokens.index("after")
            left = " ".join(tokens[:idx])
            right = " ".join(tokens[idx+1:])
            if left:
                constituents.append(left)
            if right:
                constituents.append(right)
            return constituents
        
        # Priority 2: Handle "around ... twice" or "opposite ... twice"
        # e.g., "jump around left twice" → ["jump around left"]
        if ("twice" in tokens or "thrice" in tokens):
            modifier_idx = None
            for i, t in enumerate(tokens):
                if t in ["twice", "thrice"]:
                    modifier_idx = i
                    break
            if modifier_idx and modifier_idx > 0:
                base = " ".join(tokens[:modifier_idx])
                if base:
                    constituents.append(base)
                    return constituents
        
        # Priority 3: Handle "around" modifier
        # e.g., "jump around left" → ["jump", "around left"]
        if "around" in tokens:
            idx = tokens.index("around")
            base = " ".join(tokens[:idx])
            direction_phrase = " ".join(tokens[idx:])
            if base:
                constituents.append(base)
            if direction_phrase:
                constituents.append(direction_phrase)
            return constituents
        
        # Priority 4: Handle "opposite" modifier  
        # e.g., "jump opposite left" → ["jump", "opposite left"]
        if "opposite" in tokens:
            idx = tokens.index("opposite")
            base = " ".join(tokens[:idx])
            direction_phrase = " ".join(tokens[idx:])
            if base:
                constituents.append(base)
            if direction_phrase:
                constituents.append(direction_phrase)
            return constituents
        
        return constituents


class CompositionalMetrics:
    """
    Compute comprehensive compositional generalization metrics.
    """
    
    def __init__(
        self,
        dataset_type: str = "scan",
        training_examples: Optional[List[str]] = None,
    ):
        """
        Initialize metrics calculator.
        
        Args:
            dataset_type: Type of dataset (scan, cogs, cfq, clutrr, gsm8k)
            training_examples: List of training inputs for novelty detection
        """
        self.dataset_type = dataset_type
        self.parser = CompositionParser(dataset_type)
        
        # Store training examples for distance computation
        self.training_inputs: List[str] = list(training_examples) if training_examples else []
        
        # Build training composition SIGNATURE index for novelty detection
        # Using tree-structured signatures (per Keysers et al.)
        self.training_signatures: Set[str] = set()
        
        # Precompute training feature vectors for fast distance computation
        self._training_features: List[Tuple[Set[str], str, Set[str]]] = []
        
        if training_examples:
            for ex in training_examples:
                sig = self._get_composition_signature(ex)
                self.training_signatures.add(sig)
                
                # Cache features: (primitives_set, composition_type, modifiers_set)
                prims = set(self.parser.get_primitives(ex))
                comp_type = self.parser.get_composition_type(ex)
                tokens = ex.lower().split()
                mods = set(t for t in tokens if t in CompositionParser.SCAN_MODIFIERS)
                self._training_features.append((prims, comp_type, mods))
    
    def _normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        # Collapse whitespace, lowercase, strip
        return " ".join(text.lower().split())
    
    def _normalize_output(self, text: str) -> str:
        """Normalize output for comparison (may need dataset-specific canonicalization)."""
        return " ".join(text.strip().split())
    
    def _get_composition_signature(self, text: str) -> str:
        """
        Get tree-structured compositional signature for novelty detection.
        
        A signature captures the compositional STRUCTURE including bracketing.
        This follows the spirit of Keysers et al. (2020) - novelty is about
        novel *structural* combinations, not just bag-of-features.
        
        Examples:
        - "jump twice" → "twice(jump)"
        - "jump around left" → "around(jump,left)"
        - "jump twice and walk" → "and(twice(jump),walk)"
        - "jump and walk twice" → "and(jump,twice(walk))"
        
        This distinguishes "jump twice and walk" from "jump and walk twice"!
        """
        if self.dataset_type == "scan":
            return self._scan_structural_signature(text.lower())
        else:
            # Fallback to type + primitives for non-SCAN datasets
            comp_type = self.parser.get_composition_type(text)
            primitives = sorted(self.parser.get_primitives(text))
            return f"{comp_type}|{','.join(primitives)}"
    
    def _scan_structural_signature(self, text: str) -> str:
        """
        Build tree-structured signature for SCAN using recursive parsing.
        
        Same operator precedence as depth calculation.
        """
        tokens = text.strip().split()
        if not tokens:
            return ""
        
        # Priority 1: Split on binary operators
        for op in ["and", "after"]:
            if op in tokens:
                i = tokens.index(op)
                left = " ".join(tokens[:i])
                right = " ".join(tokens[i+1:])
                left_sig = self._scan_structural_signature(left) if left else ""
                right_sig = self._scan_structural_signature(right) if right else ""
                return f"{op}({left_sig},{right_sig})"
        
        # Priority 2: Trailing unary repetition
        if tokens and tokens[-1] in {"twice", "thrice"}:
            modifier = tokens[-1]
            base = " ".join(tokens[:-1])
            base_sig = self._scan_structural_signature(base) if base else ""
            return f"{modifier}({base_sig})"
        
        # Priority 3: Unary around/opposite
        if "around" in tokens:
            i = tokens.index("around")
            base = " ".join(tokens[:i])
            direction = tokens[i+1] if i+1 < len(tokens) else "?"
            base_sig = self._scan_structural_signature(base) if base else ""
            return f"around({base_sig},{direction})"
        
        if "opposite" in tokens:
            i = tokens.index("opposite")
            base = " ".join(tokens[:i])
            direction = tokens[i+1] if i+1 < len(tokens) else "?"
            base_sig = self._scan_structural_signature(base) if base else ""
            return f"opposite({base_sig},{direction})"
        
        # Priority 4: Base - primitive + optional direction
        # Normalize: sorted to collapse "jump left" == "left jump" (shouldn't happen in SCAN)
        # Keep direction as part of signature for accuracy
        if len(tokens) == 2 and tokens[1] in CompositionParser.SCAN_DIRECTIONS:
            return f"{tokens[0]}_{tokens[1]}"
        elif len(tokens) == 1:
            return tokens[0]
        else:
            # Unknown pattern - use sorted tokens
            return "_".join(sorted(tokens))
    
    def is_novel_composition(self, text: str) -> bool:
        """
        Check if a composition signature was seen during training.
        
        Uses compositional signatures rather than exact string match,
        which is the correct notion of 'novel composition' per Keysers et al.
        """
        if not self.training_signatures:
            # No training data provided - can't determine novelty
            return True  # Assume novel if we don't know
        return self._get_composition_signature(text) not in self.training_signatures
    
    def _compute_fast_distance(self, test_input: str) -> int:
        """
        Compute compositional distance to training set using cached features.
        
        Distance = minimum feature difference to any training example.
        Features: primitives (symmetric diff) + composition type match + modifiers diff.
        
        This is a fast approximation of Keysers et al. MCD.
        """
        if not self._training_features:
            return 0
        
        # Extract test features
        test_prims = set(self.parser.get_primitives(test_input))
        test_comp_type = self.parser.get_composition_type(test_input)
        tokens = test_input.lower().split()
        test_mods = set(t for t in tokens if t in CompositionParser.SCAN_MODIFIERS)
        
        min_distance = float('inf')
        
        for train_prims, train_comp_type, train_mods in self._training_features:
            # Primitive difference (symmetric difference size)
            prim_diff = len(test_prims.symmetric_difference(train_prims))
            
            # Composition type difference
            comp_diff = 0 if test_comp_type == train_comp_type else 1
            
            # Modifier difference
            mod_diff = len(test_mods.symmetric_difference(train_mods))
            
            distance = prim_diff + comp_diff + mod_diff
            min_distance = min(min_distance, distance)
            
            # Early exit if we find exact match
            if min_distance == 0:
                return 0
        
        return min_distance if min_distance != float('inf') else 0
    
    def compute_metrics(
        self,
        inputs: List[str],
        predictions: List[str],
        targets: List[str],
    ) -> CompositionalAnalysis:
        """
        Compute comprehensive compositional metrics.
        
        Args:
            inputs: List of input sequences
            predictions: List of model predictions
            targets: List of target sequences
            
        Returns:
            CompositionalAnalysis with all metrics
        """
        analysis = CompositionalAnalysis()
        
        # Accumulators
        depth_correct = defaultdict(int)
        depth_total = defaultdict(int)
        
        primitive_correct = defaultdict(int)
        primitive_total = defaultdict(int)
        
        # Conditional primitive stats: primitive accuracy by composition context
        # Key = (primitive, context_type) where context = "simple" | "conjunction" | "sequence" | etc.
        primitive_by_context_correct = defaultdict(int)
        primitive_by_context_total = defaultdict(int)
        
        composition_correct = defaultdict(int)
        composition_total = defaultdict(int)
        
        # Use REAL compositional distance (not just 0/1 novelty)
        distance_correct = defaultdict(int)
        distance_total = defaultdict(int)
        
        novel_correct = 0
        novel_total = 0
        seen_correct = 0
        seen_total = 0
        
        error_types = defaultdict(int)
        
        total_correct = 0
        
        for inp, pred, target in zip(inputs, predictions, targets):
            # Normalize outputs for fair comparison
            pred_norm = self._normalize_output(pred)
            target_norm = self._normalize_output(target)
            is_correct = pred_norm == target_norm
            if is_correct:
                total_correct += 1
            
            # By depth
            depth = self.parser.get_depth(inp)
            depth_total[depth] += 1
            if is_correct:
                depth_correct[depth] += 1
            
            # By primitives (with conditional context tracking)
            primitives = self.parser.get_primitives(inp)
            comp_type = self.parser.get_composition_type(inp)
            context = "simple" if comp_type == "simple" else (
                "conjunction" if comp_type in ("conjunction", "sequence") else "modified"
            )
            for prim in primitives:
                primitive_total[prim] += 1
                primitive_by_context_total[(prim, context)] += 1
                if is_correct:
                    primitive_correct[prim] += 1
                    primitive_by_context_correct[(prim, context)] += 1
            
            # By composition type (comp_type already computed above)
            composition_total[comp_type] += 1
            if is_correct:
                composition_correct[comp_type] += 1
            
            # Novelty
            if self.is_novel_composition(inp):
                novel_total += 1
                if is_correct:
                    novel_correct += 1
            else:
                seen_total += 1
                if is_correct:
                    seen_correct += 1
            
            # Error categorization
            if not is_correct:
                error_type = self._categorize_error(inp, pred_norm, target_norm)
                error_types[error_type] += 1
            
            # By compositional distance (real distance, not just novelty)
            # Uses feature-based distance to nearest training example
            if self._training_features:
                distance = self._compute_fast_distance(inp)
                distance_bin = min(distance, 3)  # Bucket: 0, 1, 2, 3+
                distance_total[distance_bin] += 1
                if is_correct:
                    distance_correct[distance_bin] += 1
        
        # Compute final metrics
        n = len(inputs)
        analysis.overall_accuracy = total_correct / n if n > 0 else 0
        analysis.overall_exact_match = analysis.overall_accuracy
        
        # By depth
        for depth in depth_total:
            analysis.accuracy_by_depth[depth] = (
                depth_correct[depth] / depth_total[depth]
                if depth_total[depth] > 0 else 0
            )
            analysis.count_by_depth[depth] = depth_total[depth]
        
        # By primitive
        for prim in primitive_total:
            analysis.accuracy_by_primitive[prim] = (
                primitive_correct[prim] / primitive_total[prim]
                if primitive_total[prim] > 0 else 0
            )
            analysis.count_by_primitive[prim] = primitive_total[prim]
        
        # Conditional primitive stats by context
        for (prim, context) in primitive_by_context_total:
            key = (prim, context)
            total = primitive_by_context_total[key]
            correct = primitive_by_context_correct[key]
            analysis.accuracy_by_primitive_context[key] = (
                correct / total if total > 0 else 0
            )
            analysis.count_by_primitive_context[key] = total
        
        # By composition type
        for comp in composition_total:
            analysis.accuracy_by_composition[comp] = (
                composition_correct[comp] / composition_total[comp]
                if composition_total[comp] > 0 else 0
            )
            analysis.count_by_composition[comp] = composition_total[comp]
        
        # Novelty
        analysis.novel_composition_accuracy = (
            novel_correct / novel_total if novel_total > 0 else 0
        )
        analysis.seen_composition_accuracy = (
            seen_correct / seen_total if seen_total > 0 else 0
        )
        analysis.novel_count = novel_total
        analysis.seen_count = seen_total
        
        # Errors
        analysis.error_types = dict(error_types)
        
        # Distance (real compositional distance)
        for dist in distance_total:
            analysis.accuracy_by_distance[dist] = (
                distance_correct[dist] / distance_total[dist]
                if distance_total[dist] > 0 else 0
            )
            analysis.count_by_distance[dist] = distance_total[dist]
        
        return analysis
    
    def _categorize_error(
        self,
        input_text: str,
        prediction: str,
        target: str,
    ) -> str:
        """
        Categorize the type of error using multiset analysis and edit distance.
        
        Categories:
        - truncation: Output significantly shorter
        - over_generation: Output significantly longer
        - completely_wrong: No token overlap
        - order_error: Same tokens with same counts, wrong order
        - repetition_error: Token type overlap is high, but counts differ
        - partial_error: Some tokens correct, some wrong
        
        Uses token-level Levenshtein for more nuanced analysis.
        """
        pred_tokens = prediction.strip().split()
        target_tokens = target.strip().split()
        
        # Use Counter for multiset comparison
        pred_counter = Counter(pred_tokens)
        target_counter = Counter(target_tokens)
        
        # Compute overlap (shared tokens with min counts)
        overlap = sum((pred_counter & target_counter).values())
        
        # Check for completely wrong first
        if overlap == 0:
            return "completely_wrong"
        
        # Check if same multiset but wrong order (true order error)
        if pred_counter == target_counter:
            return "order_error"
        
        # Length difference analysis
        len_diff = len(pred_tokens) - len(target_tokens)
        if len_diff <= -3:  # Significantly shorter
            return "truncation"
        elif len_diff >= 3:  # Significantly longer
            return "over_generation"
        
        # Repetition error detection:
        # Same token TYPES appear, but with wrong COUNTS
        # This catches errors from twice/thrice/around/opposite expansions
        pred_types = set(pred_counter.keys())
        target_types = set(target_counter.keys())
        type_overlap = len(pred_types & target_types)
        type_union = len(pred_types | target_types)
        jaccard = type_overlap / type_union if type_union > 0 else 0
        
        if jaccard >= 0.7:  # High type overlap
            # Check if counts differ (repetition issue)
            count_mismatch = False
            for token in pred_types & target_types:
                if pred_counter[token] != target_counter[token]:
                    count_mismatch = True
                    break
            if count_mismatch:
                return "repetition_error"
        
        # Use edit distance for partial vs other errors
        edit_dist = self._token_levenshtein(pred_tokens, target_tokens)
        max_len = max(len(pred_tokens), len(target_tokens), 1)
        normalized_dist = edit_dist / max_len
        
        if normalized_dist > 0.7:
            return "completely_wrong"
        
        return "partial_error"
    
    def _token_levenshtein(self, seq1: List[str], seq2: List[str]) -> int:
        """
        Compute token-level Levenshtein (edit) distance.
        
        Uses dynamic programming. No external dependencies.
        """
        m, n = len(seq1), len(seq2)
        
        # Create DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Base cases
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i - 1][j],      # deletion
                        dp[i][j - 1],      # insertion
                        dp[i - 1][j - 1]   # substitution
                    )
        
        return dp[m][n]


def compute_compositional_distance(
    test_input: str,
    training_inputs: List[str],
    parser: CompositionParser,
) -> int:
    """
    Compute compositional distance from training set.
    
    Distance = minimum number of novel compositions in test input
    that weren't seen in training.
    
    Based on: Keysers et al. (2020) MCD (Maximum Compound Divergence)
    """
    test_primitives = set(parser.get_primitives(test_input))
    test_composition = parser.get_composition_type(test_input)
    
    min_distance = float('inf')
    
    for train_input in training_inputs:
        train_primitives = set(parser.get_primitives(train_input))
        train_composition = parser.get_composition_type(train_input)
        
        # Primitive difference
        prim_diff = len(test_primitives.symmetric_difference(train_primitives))
        
        # Composition difference
        comp_diff = 0 if test_composition == train_composition else 1
        
        distance = prim_diff + comp_diff
        min_distance = min(min_distance, distance)
    
    return min_distance if min_distance != float('inf') else 0


def analyze_by_depth(
    analysis: CompositionalAnalysis,
) -> Dict[str, Any]:
    """
    Analyze accuracy degradation by composition depth.
    
    Returns trend analysis.
    """
    depths = sorted(analysis.accuracy_by_depth.keys())
    accuracies = [analysis.accuracy_by_depth[d] for d in depths]
    
    if len(depths) < 2:
        return {"trend": "insufficient_data"}
    
    # Compute trend (negative = degradation with depth)
    # Fix issue #1: Make SciPy optional - use numpy fallback
    if HAS_SCIPY:
        slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(depths, accuracies)
        r_squared = r_value ** 2
    else:
        # Pure numpy fallback (no p-value, but slope/r² are the important bits)
        x = np.array(depths, dtype=float)
        y = np.array(accuracies, dtype=float)
        n = len(x)
        
        # Linear regression via least squares
        x_mean, y_mean = x.mean(), y.mean()
        slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
        intercept = y_mean - slope * x_mean
        
        # R-squared
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        p_value = None  # Not computed without SciPy
    
    result = {
        "depths": depths,
        "accuracies": accuracies,
        "slope": float(slope),
        "r_squared": float(r_squared),
        "trend": "degrading" if slope < -0.05 else "stable" if abs(slope) < 0.05 else "improving",
        "degradation_per_depth": float(-slope) if slope < 0 else 0,
    }
    
    if p_value is not None:
        result["p_value"] = float(p_value)
    
    return result


def format_compositional_report(analysis: CompositionalAnalysis) -> str:
    """Format analysis as human-readable report."""
    lines = [
        "=" * 60,
        "COMPOSITIONAL GENERALIZATION ANALYSIS",
        "=" * 60,
        "",
        f"Overall Accuracy: {analysis.overall_accuracy:.2%}",
        "",
        "--- Accuracy by Composition Depth ---",
    ]
    
    for depth in sorted(analysis.accuracy_by_depth.keys()):
        acc = analysis.accuracy_by_depth[depth]
        count = analysis.count_by_depth[depth]
        lines.append(f"  Depth {depth}: {acc:.2%} (n={count})")
    
    lines.extend([
        "",
        "--- Accuracy by Primitive ---",
    ])
    
    for prim, acc in sorted(analysis.accuracy_by_primitive.items(), key=lambda x: -x[1]):
        count = analysis.count_by_primitive[prim]
        lines.append(f"  {prim}: {acc:.2%} (n={count})")
    
    lines.extend([
        "",
        "--- Accuracy by Composition Type ---",
    ])
    
    for comp, acc in sorted(analysis.accuracy_by_composition.items(), key=lambda x: -x[1]):
        count = analysis.count_by_composition[comp]
        lines.append(f"  {comp}: {acc:.2%} (n={count})")
    
    lines.extend([
        "",
        "--- Novel vs. Seen Compositions ---",
        f"  Novel compositions: {analysis.novel_composition_accuracy:.2%} (n={analysis.novel_count})",
        f"  Seen compositions:  {analysis.seen_composition_accuracy:.2%} (n={analysis.seen_count})",
        f"  Generalization gap: {analysis.seen_composition_accuracy - analysis.novel_composition_accuracy:.2%}",
    ])
    
    # Add distance breakdown if available
    if analysis.accuracy_by_distance:
        lines.extend([
            "",
            "--- Accuracy by Compositional Distance ---",
        ])
        for dist in sorted(analysis.accuracy_by_distance.keys()):
            acc = analysis.accuracy_by_distance[dist]
            label = f"Distance {dist}" if dist < 3 else "Distance 3+"
            lines.append(f"  {label}: {acc:.2%}")
    
    lines.extend([
        "",
        "--- Error Types ---",
    ])
    
    for error_type, count in sorted(analysis.error_types.items(), key=lambda x: -x[1]):
        lines.append(f"  {error_type}: {count}")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)


# =============================================================================
# Multi-Seed Statistical Significance Testing
# =============================================================================

@dataclass
class MultiSeedResult:
    """Results from multi-seed experiments with statistical significance."""
    
    # Metric means and stds
    metric_means: Dict[str, float] = field(default_factory=dict)
    metric_stds: Dict[str, float] = field(default_factory=dict)
    
    # Per-seed results
    per_seed_results: List[Dict[str, float]] = field(default_factory=list)
    
    # Statistical tests
    significance_tests: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Confidence intervals
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    
    seeds: List[int] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric_means": self.metric_means,
            "metric_stds": self.metric_stds,
            "confidence_intervals": {
                k: list(v) for k, v in self.confidence_intervals.items()
            },
            "significance_tests": self.significance_tests,
            "seeds": self.seeds,
            "num_seeds": len(self.seeds),
        }


def compute_multi_seed_statistics(
    seed_results: List[Dict[str, float]],
    baseline_results: Optional[List[Dict[str, float]]] = None,
    confidence_level: float = 0.95,
) -> MultiSeedResult:
    """
    Compute statistics across multiple random seeds with significance testing.
    
    Args:
        seed_results: List of per-seed metric dictionaries
        baseline_results: Optional baseline results for comparison testing
        confidence_level: Confidence level for intervals (default 0.95 = 95%)
        
    Returns:
        MultiSeedResult with means, stds, and significance tests
    """
    if not seed_results:
        return MultiSeedResult()
    
    result = MultiSeedResult(per_seed_results=seed_results)
    
    # Get all metric names
    metric_names = set()
    for r in seed_results:
        metric_names.update(r.keys())
    
    # Compute means and stds
    for metric in metric_names:
        values = [r.get(metric, 0.0) for r in seed_results if metric in r]
        if values:
            result.metric_means[metric] = float(np.mean(values))
            result.metric_stds[metric] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            
            # Confidence interval
            n = len(values)
            if n > 1:
                if HAS_SCIPY:
                    # Use t-distribution for small samples
                    t_value = scipy_stats.t.ppf((1 + confidence_level) / 2, df=n-1)
                else:
                    # Approximate with z-score
                    z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
                    t_value = z_scores.get(confidence_level, 1.96)
                
                margin = t_value * result.metric_stds[metric] / np.sqrt(n)
                ci_low = result.metric_means[metric] - margin
                ci_high = result.metric_means[metric] + margin
                result.confidence_intervals[metric] = (ci_low, ci_high)
    
    # Significance tests against baseline
    if baseline_results:
        result.significance_tests = run_significance_tests(
            seed_results, baseline_results
        )
    
    return result


def run_significance_tests(
    treatment_results: List[Dict[str, float]],
    control_results: List[Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """
    Run statistical significance tests comparing treatment vs control.
    
    Uses:
    - Independent t-test (parametric)
    - Mann-Whitney U test (non-parametric)
    - Effect size (Cohen's d)
    
    Args:
        treatment_results: List of per-seed results for treatment (e.g., DAI)
        control_results: List of per-seed results for control (e.g., baseline)
        
    Returns:
        Dict mapping metric name to test results
    """
    tests = {}
    
    # Get common metrics
    treatment_metrics = set()
    control_metrics = set()
    for r in treatment_results:
        treatment_metrics.update(r.keys())
    for r in control_results:
        control_metrics.update(r.keys())
    common_metrics = treatment_metrics & control_metrics
    
    for metric in common_metrics:
        treatment_values = np.array([r.get(metric, 0.0) for r in treatment_results if metric in r])
        control_values = np.array([r.get(metric, 0.0) for r in control_results if metric in r])
        
        if len(treatment_values) < 2 or len(control_values) < 2:
            continue
        
        test_result = {
            "treatment_mean": float(np.mean(treatment_values)),
            "control_mean": float(np.mean(control_values)),
            "treatment_std": float(np.std(treatment_values, ddof=1)),
            "control_std": float(np.std(control_values, ddof=1)),
            "improvement": float(np.mean(treatment_values) - np.mean(control_values)),
        }
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            ((len(treatment_values) - 1) * np.var(treatment_values, ddof=1) +
             (len(control_values) - 1) * np.var(control_values, ddof=1)) /
            (len(treatment_values) + len(control_values) - 2)
        )
        if pooled_std > 0:
            cohens_d = (np.mean(treatment_values) - np.mean(control_values)) / pooled_std
            test_result["cohens_d"] = float(cohens_d)
            test_result["effect_size"] = (
                "large" if abs(cohens_d) > 0.8 else
                "medium" if abs(cohens_d) > 0.5 else
                "small" if abs(cohens_d) > 0.2 else
                "negligible"
            )
        
        # Statistical tests (require SciPy)
        if HAS_SCIPY:
            # Independent t-test
            t_stat, t_pvalue = scipy_stats.ttest_ind(treatment_values, control_values)
            test_result["t_statistic"] = float(t_stat)
            test_result["t_pvalue"] = float(t_pvalue)
            test_result["t_significant_005"] = t_pvalue < 0.05
            test_result["t_significant_001"] = t_pvalue < 0.01
            
            # Mann-Whitney U test (non-parametric)
            try:
                u_stat, u_pvalue = scipy_stats.mannwhitneyu(
                    treatment_values, control_values, alternative='two-sided'
                )
                test_result["u_statistic"] = float(u_stat)
                test_result["u_pvalue"] = float(u_pvalue)
                test_result["u_significant_005"] = u_pvalue < 0.05
            except Exception:
                pass
        
        tests[metric] = test_result
    
    return tests


def format_multi_seed_report(
    result: MultiSeedResult,
    model_name: str = "Model",
    baseline_name: str = "Baseline",
) -> str:
    """Format multi-seed results as human-readable report."""
    lines = [
        "=" * 70,
        f"MULTI-SEED STATISTICAL ANALYSIS: {model_name}",
        "=" * 70,
        f"Number of seeds: {len(result.seeds) if result.seeds else len(result.per_seed_results)}",
        "",
        "--- Metric Summary (Mean ± Std) ---",
    ]
    
    for metric in sorted(result.metric_means.keys()):
        mean = result.metric_means[metric]
        std = result.metric_stds.get(metric, 0)
        ci = result.confidence_intervals.get(metric)
        
        if ci:
            lines.append(f"  {metric}: {mean:.4f} ± {std:.4f}  [95% CI: {ci[0]:.4f}, {ci[1]:.4f}]")
        else:
            lines.append(f"  {metric}: {mean:.4f} ± {std:.4f}")
    
    if result.significance_tests:
        lines.extend([
            "",
            f"--- Significance Tests vs. {baseline_name} ---",
        ])
        
        for metric, tests in sorted(result.significance_tests.items()):
            improvement = tests.get("improvement", 0)
            sign = "+" if improvement > 0 else ""
            effect = tests.get("effect_size", "unknown")
            
            line = f"  {metric}: {sign}{improvement:.4f}"
            
            if "t_pvalue" in tests:
                p = tests["t_pvalue"]
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                line += f" (p={p:.4f}{sig}, d={tests.get('cohens_d', 0):.2f} [{effect}])"
            
            lines.append(line)
        
        lines.extend([
            "",
            "Significance: * p<0.05, ** p<0.01, *** p<0.001",
        ])
    
    lines.append("=" * 70)
    
    return "\n".join(lines)


# =============================================================================
# Detailed Per-Depth and Per-Primitive Breakdown
# =============================================================================

def generate_detailed_breakdown(
    analysis: CompositionalAnalysis,
    output_format: str = "markdown",
) -> str:
    """
    Generate detailed breakdown tables for per-depth and per-primitive accuracy.
    
    Args:
        analysis: CompositionalAnalysis results
        output_format: "markdown", "latex", or "text"
        
    Returns:
        Formatted breakdown string
    """
    if output_format == "markdown":
        return _markdown_breakdown(analysis)
    elif output_format == "latex":
        return _latex_breakdown(analysis)
    else:
        return format_compositional_report(analysis)


def _markdown_breakdown(analysis: CompositionalAnalysis) -> str:
    """Generate markdown tables."""
    lines = [
        "# Compositional Generalization Breakdown",
        "",
        f"**Overall Accuracy:** {analysis.overall_accuracy:.2%}",
        "",
        "## Accuracy by Composition Depth",
        "",
        "| Depth | Accuracy | Count |",
        "|-------|----------|-------|",
    ]
    
    for depth in sorted(analysis.accuracy_by_depth.keys()):
        acc = analysis.accuracy_by_depth[depth]
        count = analysis.count_by_depth[depth]
        lines.append(f"| {depth} | {acc:.2%} | {count} |")
    
    lines.extend([
        "",
        "## Accuracy by Primitive",
        "",
        "| Primitive | Accuracy | Count |",
        "|-----------|----------|-------|",
    ])
    
    for prim, acc in sorted(analysis.accuracy_by_primitive.items(), key=lambda x: -x[1]):
        count = analysis.count_by_primitive[prim]
        lines.append(f"| {prim} | {acc:.2%} | {count} |")
    
    lines.extend([
        "",
        "## Accuracy by Composition Type",
        "",
        "| Type | Accuracy | Count |",
        "|------|----------|-------|",
    ])
    
    for comp, acc in sorted(analysis.accuracy_by_composition.items(), key=lambda x: -x[1]):
        count = analysis.count_by_composition[comp]
        lines.append(f"| {comp} | {acc:.2%} | {count} |")
    
    lines.extend([
        "",
        "## Novel vs. Seen Compositions",
        "",
        "| Category | Accuracy | Count |",
        "|----------|----------|-------|",
        f"| Novel | {analysis.novel_composition_accuracy:.2%} | {analysis.novel_count} |",
        f"| Seen | {analysis.seen_composition_accuracy:.2%} | {analysis.seen_count} |",
        "",
        f"**Generalization Gap:** {(analysis.seen_composition_accuracy - analysis.novel_composition_accuracy):.2%}",
    ])
    
    return "\n".join(lines)


def _latex_breakdown(analysis: CompositionalAnalysis) -> str:
    """Generate LaTeX tables."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Compositional Generalization Breakdown}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Depth & Accuracy & Count \\",
        r"\midrule",
    ]
    
    for depth in sorted(analysis.accuracy_by_depth.keys()):
        acc = analysis.accuracy_by_depth[depth] * 100
        count = analysis.count_by_depth[depth]
        lines.append(f"{depth} & {acc:.1f}\\% & {count} \\\\")
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Accuracy by Primitive}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Primitive & Accuracy & Count \\",
        r"\midrule",
    ])
    
    for prim, acc in sorted(analysis.accuracy_by_primitive.items(), key=lambda x: -x[1]):
        acc_pct = acc * 100
        count = analysis.count_by_primitive[prim]
        lines.append(f"{prim} & {acc_pct:.1f}\\% & {count} \\\\")
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    return "\n".join(lines)
