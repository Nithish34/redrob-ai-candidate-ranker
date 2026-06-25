"""
Feature Extractor
=================
Prepares candidate profiles for scoring by validating and normalizing them.
"""

from __future__ import annotations

from app.preprocess.normalizer import normalize_candidate
from app.preprocess.validator import validate_candidate


def prepare_candidate(candidate: dict) -> dict | None:
    """Validate, normalize, and prepare candidate data for pipeline layers.

    Parameters
    ----------
    candidate : dict
        Raw candidate dictionary.

    Returns
    -------
    dict | None
        Normalized candidate dictionary, or None if validation fails.
    """
    if not validate_candidate(candidate):
        return None
        
    return normalize_candidate(candidate)
