"""
Validator Engine
================
Validates candidate record structure and integrity.
"""

from __future__ import annotations

from app.utils.logger import get_logger

log = get_logger("validator")


def validate_candidate(candidate: dict) -> bool:
    """Validate a candidate dictionary structure.

    Parameters
    ----------
    candidate : dict
        Candidate record dictionary.

    Returns
    -------
    bool
        True if structurally valid, False if critical data is missing or corrupted.
    """
    if not isinstance(candidate, dict):
        log.warning("Candidate record is not a dictionary.")
        return False

    candidate_id = candidate.get("candidate_id")
    if not candidate_id or not isinstance(candidate_id, str):
        log.warning("Candidate record is missing candidate_id or candidate_id is not a string.")
        return False

    # Check for critical sections
    required_sections = ["profile", "career_history", "skills", "redrob_signals"]
    for section in required_sections:
        if section not in candidate:
            log.warning("Candidate %s is missing critical section: '%s'", candidate_id, section)
            return False

    # Verify profile structure
    profile = candidate.get("profile")
    if not isinstance(profile, dict):
        log.warning("Candidate %s: 'profile' section is not a dictionary.", candidate_id)
        return False

    # Verify career history structure
    history = candidate.get("career_history")
    if not isinstance(history, list):
        log.warning("Candidate %s: 'career_history' section is not a list.", candidate_id)
        return False

    # Verify skills structure
    skills = candidate.get("skills")
    if not isinstance(skills, list):
        log.warning("Candidate %s: 'skills' section is not a list.", candidate_id)
        return False

    # Verify redrob_signals structure
    signals = candidate.get("redrob_signals")
    if not isinstance(signals, dict):
        log.warning("Candidate %s: 'redrob_signals' section is not a dictionary.", candidate_id)
        return False

    return True
