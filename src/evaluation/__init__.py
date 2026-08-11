"""Evaluation package - Metrics and evaluation utilities."""

from .metrics import (
    EvaluationResult,
    CompositionalMetrics,
    OverConstraintMetrics,
    TrainingStabilityMetrics,
    evaluate_model,
    run_comprehensive_evaluation,
    compute_exact_match_accuracy,
    compute_token_accuracy_teacher_forced,
    compute_prefix_accuracy,
    compute_length_diagnostics,
    # Output normalization for fair cross-baseline evaluation
    normalize_for_eval,
    normalize_batch_for_eval,
)

from .compositional_metrics import (
    CompositionalMetrics as DetailedCompositionalMetrics,
    CompositionParser,
    CompositionalAnalysis,
    format_compositional_report,
    analyze_by_depth,
    compute_compositional_distance,
    # Multi-seed statistics
    MultiSeedResult,
    compute_multi_seed_statistics,
    run_significance_tests,
    format_multi_seed_report,
    generate_detailed_breakdown,
)

from .attention_analysis import (
    AttentionPattern,
    AttentionAnalysisResult,
    AttentionVisualizer,
    visualize_compositional_attention,
)

from .modularity_analysis import (
    ProcessingAnalysisResult,
    ModularityAnalyzer,
    compare_model_modularity,
)

__all__ = [
    # Core metrics
    "EvaluationResult",
    "CompositionalMetrics",
    "OverConstraintMetrics",
    "TrainingStabilityMetrics",
    # Evaluation functions
    "evaluate_model",
    "run_comprehensive_evaluation",
    "compute_exact_match_accuracy",
    "compute_token_accuracy_teacher_forced",
    "compute_prefix_accuracy",
    "compute_length_diagnostics",
    # Output normalization for fair cross-baseline evaluation
    "normalize_for_eval",
    "normalize_batch_for_eval",
    # Detailed compositional analysis
    "DetailedCompositionalMetrics",
    "CompositionParser",
    "CompositionalAnalysis",
    "format_compositional_report",
    "analyze_by_depth",
    "compute_compositional_distance",
    # Multi-seed statistics
    "MultiSeedResult",
    "compute_multi_seed_statistics",
    "run_significance_tests",
    "format_multi_seed_report",
    "generate_detailed_breakdown",
    # Attention analysis
    "AttentionPattern",
    "AttentionAnalysisResult",
    "AttentionVisualizer",
    "visualize_compositional_attention",
    # Modularity analysis
    "ProcessingAnalysisResult",
    "ModularityAnalyzer",
    "compare_model_modularity",
]
