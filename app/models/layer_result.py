"""
Structured layer result contract.
Every scoring layer returns a dict following this schema.
"""

from typing import Any


def make_layer_result(
    score: float,
    confidence: float = 0.0,
    feature_scores: dict[str, float] | None = None,
    reasoning: list[str] | None = None,
    warnings: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict:
    """Create a standardized LayerResult dict.

    Args:
        score: Overall layer score (0-100).
        confidence: Confidence in the score (0-100).
        feature_scores: Sub-feature breakdown.
        reasoning: Human-readable reasons for this score.
        warnings: Flags or concerns detected.
        metadata: Additional structured data.

    Returns:
        A LayerResult dict.
    """
    return {
        "score": round(max(0.0, min(100.0, score)), 2),
        "confidence": round(max(0.0, min(100.0, confidence)), 2),
        "feature_scores": feature_scores or {},
        "reasoning": reasoning or [],
        "warnings": warnings or [],
        "metadata": metadata or {},
    }
