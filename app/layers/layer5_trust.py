"""
Layer 5 — Trust Engine
======================
Measures profile CREDIBILITY: is what the candidate claims actually supported
by evidence across career history, assessments, verification signals, GitHub
activity, and internal consistency?

Config keys (from config['layer5']):
    evidence_claim_weight : 0.40
    assessment_weight     : 0.20
    verification_weight   : 0.15
    github_weight         : 0.15
    consistency_weight    : 0.10
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_WEIGHTS: dict[str, float] = {
    "evidence_claim_weight": 0.40,
    "assessment_weight": 0.20,
    "verification_weight": 0.15,
    "github_weight": 0.15,
    "consistency_weight": 0.10,
}

# Skills considered relevant for AI/ML assessment validation
_AIML_KEYWORDS: set[str] = {
    "python", "ml", "machine learning", "nlp", "natural language processing",
    "data", "data science", "deep learning", "tensorflow", "pytorch",
    "scikit-learn", "sklearn", "keras", "pandas", "numpy", "ai",
    "artificial intelligence", "computer vision", "cv", "llm",
    "large language model", "transformers", "statistics", "neural network",
    "neural networks", "data engineering", "data analysis", "spark", "sql",
}

_NEUTRAL_GITHUB_SCORE: float = 40.0
_NEUTRAL_ASSESSMENT_SCORE: float = 50.0
_EXPERIENCE_TOLERANCE: float = 0.20  # 20 %


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _safe_lower(value: Any) -> str:
    """Return lowered string, or empty string for non-string / None values."""
    if isinstance(value, str):
        return value.lower()
    return ""


def _get_config(config: dict, key: str) -> float:
    """Retrieve a weight from config['layer5'], falling back to defaults."""
    layer_cfg = config.get("layer5", {})
    return float(layer_cfg.get(key, _DEFAULT_WEIGHTS.get(key, 0.0)))


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    """Clamp *value* to [lo, hi]."""
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# Feature 1 — Evidence-to-Claim Ratio  (40 %)
# ---------------------------------------------------------------------------

def _score_evidence_claim(candidate: dict) -> dict[str, Any]:
    """
    For every claimed skill, check whether it appears (case-insensitive) in
    ANY career_history description.

    * Expert/advanced skills found in descriptions → high trust.
    * Expert skills with NO evidence → double penalty.
    """
    skills: list[dict] = candidate.get("skills", [])
    career_history: list[dict] = candidate.get("career_history", [])

    if not skills:
        return {
            "score": 50.0,
            "supported": 0,
            "unsupported": 0,
            "expert_unsupported": 0,
            "total": 0,
            "detail": "No skills listed — neutral score.",
        }

    # Build a single blob of all career descriptions for matching
    description_blob = " ".join(
        _safe_lower(entry.get("description", ""))
        for entry in career_history
    ).strip()

    total_skills = len(skills)
    supported_count = 0
    unsupported_skills: list[str] = []
    expert_unsupported: list[str] = []
    high_trust_skills: list[str] = []

    for skill in skills:
        skill_name = _safe_lower(skill.get("name", ""))
        proficiency = _safe_lower(skill.get("proficiency", ""))
        if not skill_name:
            continue

        found_in_career = skill_name in description_blob

        if found_in_career:
            supported_count += 1
            if proficiency in ("expert", "advanced"):
                high_trust_skills.append(skill_name)
        else:
            unsupported_skills.append(skill_name)
            if proficiency == "expert":
                expert_unsupported.append(skill_name)

    # Base ratio (0-100)
    supported_ratio = (supported_count / total_skills) * 100.0

    # Apply double penalty for expert-but-unsupported skills
    # Each such skill reduces the score by an extra penalty proportional
    # to its share of total skills.
    expert_penalty = (len(expert_unsupported) / total_skills) * 100.0
    adjusted_score = _clamp(supported_ratio - expert_penalty)

    detail_parts: list[str] = [
        f"{supported_count}/{total_skills} skills evidenced in career history.",
    ]
    if high_trust_skills:
        detail_parts.append(
            f"High-trust (expert/advanced + evidenced): {', '.join(high_trust_skills[:5])}."
        )
    if expert_unsupported:
        detail_parts.append(
            f"Expert claims with NO evidence: {', '.join(expert_unsupported[:5])} — double penalty applied."
        )

    return {
        "score": round(adjusted_score, 2),
        "supported": supported_count,
        "unsupported": len(unsupported_skills),
        "expert_unsupported": len(expert_unsupported),
        "high_trust": len(high_trust_skills),
        "total": total_skills,
        "detail": " ".join(detail_parts),
    }


# ---------------------------------------------------------------------------
# Feature 2 — Assessment Validation  (20 %)
# ---------------------------------------------------------------------------

def _score_assessment(candidate: dict) -> dict[str, Any]:
    """
    Average skill_assessment_scores for AI/ML-relevant keys.
    If no assessment data exists, return neutral 50.
    """
    signals: dict = candidate.get("redrob_signals", {})
    assessment_scores: dict = signals.get("skill_assessment_scores", {})

    if not assessment_scores:
        return {
            "score": _NEUTRAL_ASSESSMENT_SCORE,
            "matched_skills": 0,
            "detail": "No assessment data available — neutral score assigned.",
        }

    relevant_scores: list[float] = []
    matched_keys: list[str] = []

    for skill_key, score_val in assessment_scores.items():
        if _safe_lower(skill_key) in _AIML_KEYWORDS:
            try:
                relevant_scores.append(float(score_val))
                matched_keys.append(skill_key)
            except (TypeError, ValueError):
                continue

    if not relevant_scores:
        # Assessments exist but none match AI/ML keywords — use overall avg
        all_scores: list[float] = []
        for score_val in assessment_scores.values():
            try:
                all_scores.append(float(score_val))
            except (TypeError, ValueError):
                continue
        if all_scores:
            avg = sum(all_scores) / len(all_scores)
            return {
                "score": round(_clamp(avg), 2),
                "matched_skills": 0,
                "detail": f"No AI/ML-specific assessments; used overall avg of {len(all_scores)} assessments.",
            }
        return {
            "score": _NEUTRAL_ASSESSMENT_SCORE,
            "matched_skills": 0,
            "detail": "Assessment data present but unparseable — neutral score.",
        }

    avg = sum(relevant_scores) / len(relevant_scores)
    return {
        "score": round(_clamp(avg), 2),
        "matched_skills": len(relevant_scores),
        "detail": f"Averaged {len(relevant_scores)} AI/ML assessments ({', '.join(matched_keys[:5])}).",
    }


# ---------------------------------------------------------------------------
# Feature 3 — Verification Score  (15 %)
# ---------------------------------------------------------------------------

def _score_verification(candidate: dict) -> dict[str, Any]:
    """verified_email + verified_phone + linkedin_connected → score out of 100."""
    signals: dict = candidate.get("redrob_signals", {})

    checks = {
        "verified_email": bool(signals.get("verified_email", False)),
        "verified_phone": bool(signals.get("verified_phone", False)),
        "linkedin_connected": bool(signals.get("linkedin_connected", False)),
    }

    verified_count = sum(checks.values())
    score = (verified_count / 3) * 100.0

    return {
        "score": round(score, 2),
        "checks": checks,
        "verified_count": verified_count,
        "detail": f"{verified_count}/3 verification signals confirmed.",
    }


# ---------------------------------------------------------------------------
# Feature 4 — GitHub Credibility  (15 %)
# ---------------------------------------------------------------------------

def _score_github(candidate: dict) -> dict[str, Any]:
    """
    github_activity_score is already 0-100.
    If -1 (no GitHub), assign neutral 40.
    """
    signals: dict = candidate.get("redrob_signals", {})
    raw_score = signals.get("github_activity_score", -1)

    try:
        raw_score = float(raw_score)
    except (TypeError, ValueError):
        raw_score = -1.0

    if raw_score < 0:
        return {
            "score": _NEUTRAL_GITHUB_SCORE,
            "raw": raw_score,
            "detail": "No GitHub data — neutral score assigned.",
        }

    return {
        "score": round(_clamp(raw_score), 2),
        "raw": raw_score,
        "detail": f"GitHub activity score: {raw_score}.",
    }


# ---------------------------------------------------------------------------
# Feature 5 — Internal Consistency  (10 %)
# ---------------------------------------------------------------------------

def _score_consistency(candidate: dict) -> dict[str, Any]:
    """
    Perform four consistency checks:
      1. headline vs career titles alignment
      2. summary themes vs career descriptions
      3. current_title matches latest career_history title
      4. years_of_experience ≈ sum of career durations (within 20 %)
    """
    profile: dict = candidate.get("profile", {})
    career_history: list[dict] = candidate.get("career_history", [])

    checks_passed = 0
    total_checks = 4
    check_details: dict[str, bool] = {}

    headline = _safe_lower(profile.get("headline", ""))
    summary = _safe_lower(profile.get("summary", ""))
    current_title = _safe_lower(profile.get("current_title", ""))
    years_exp = profile.get("years_of_experience", None)

    career_titles: list[str] = [
        _safe_lower(entry.get("title", "")) for entry in career_history
    ]
    career_descriptions_blob = " ".join(
        _safe_lower(entry.get("description", "")) for entry in career_history
    )

    # ---- Check 1: headline vs career titles ----
    headline_aligned = False
    if headline and career_titles:
        # Extract meaningful words from headline (> 2 chars)
        headline_words = {w for w in headline.split() if len(w) > 2}
        for title in career_titles:
            title_words = {w for w in title.split() if len(w) > 2}
            if headline_words & title_words:
                headline_aligned = True
                break
    elif not headline:
        # No headline → skip (count as pass to avoid unfair penalty)
        headline_aligned = True
    check_details["headline_vs_titles"] = headline_aligned
    if headline_aligned:
        checks_passed += 1

    # ---- Check 2: summary themes vs career descriptions ----
    summary_aligned = False
    if summary and career_descriptions_blob:
        summary_words = {w for w in summary.split() if len(w) > 3}
        career_words = set(career_descriptions_blob.split())
        if summary_words:
            overlap = len(summary_words & career_words)
            overlap_ratio = overlap / len(summary_words)
            summary_aligned = overlap_ratio >= 0.15  # at least 15 % overlap
    elif not summary:
        summary_aligned = True  # no summary → don't penalize
    check_details["summary_vs_career"] = summary_aligned
    if summary_aligned:
        checks_passed += 1

    # ---- Check 3: current_title matches latest career_history title ----
    title_match = False
    if current_title and career_history:
        # Find the latest entry (is_current=True or first in list)
        latest_entry = None
        for entry in career_history:
            if entry.get("is_current", False):
                latest_entry = entry
                break
        if latest_entry is None:
            latest_entry = career_history[0]

        latest_title = _safe_lower(latest_entry.get("title", ""))
        if latest_title and current_title:
            # Fuzzy: check if one contains the other or significant word overlap
            ct_words = {w for w in current_title.split() if len(w) > 2}
            lt_words = {w for w in latest_title.split() if len(w) > 2}
            if ct_words and lt_words:
                overlap = len(ct_words & lt_words)
                title_match = overlap >= min(len(ct_words), len(lt_words)) * 0.5
            elif current_title in latest_title or latest_title in current_title:
                title_match = True
    elif not current_title:
        title_match = True  # no current title → don't penalize
    check_details["current_title_match"] = title_match
    if title_match:
        checks_passed += 1

    # ---- Check 4: years_of_experience vs sum of career durations ----
    experience_consistent = False
    if years_exp is not None and career_history:
        try:
            claimed_years = float(years_exp)
        except (TypeError, ValueError):
            claimed_years = -1.0

        total_months = sum(
            entry.get("duration_months", 0) or 0 for entry in career_history
        )
        computed_years = total_months / 12.0

        if claimed_years > 0 and computed_years > 0:
            diff_ratio = abs(claimed_years - computed_years) / claimed_years
            experience_consistent = diff_ratio <= _EXPERIENCE_TOLERANCE
        elif claimed_years == 0 and computed_years == 0:
            experience_consistent = True
    elif years_exp is None:
        experience_consistent = True  # missing data → don't penalize
    check_details["experience_duration_match"] = experience_consistent
    if experience_consistent:
        checks_passed += 1

    score = (checks_passed / total_checks) * 100.0

    return {
        "score": round(score, 2),
        "checks_passed": checks_passed,
        "total_checks": total_checks,
        "check_details": check_details,
        "detail": f"{checks_passed}/{total_checks} consistency checks passed.",
    }


# ---------------------------------------------------------------------------
# Build reasoning string
# ---------------------------------------------------------------------------

def _build_reasoning(
    evidence: dict,
    assessment: dict,
    verification: dict,
    github: dict,
    consistency: dict,
    final_score: float,
) -> str:
    """Compose a human-readable reasoning paragraph."""
    parts: list[str] = [
        f"Trust Score: {final_score:.1f}/100.",
        f"Evidence-to-Claim: {evidence['detail']}",
        f"Assessments: {assessment['detail']}",
        f"Verification: {verification['detail']}",
        f"GitHub: {github['detail']}",
        f"Consistency: {consistency['detail']}",
    ]
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def execute(candidate: dict, parsed_jd: dict, config: dict) -> dict:
    """
    Layer 5 — Trust Engine.

    Measures profile credibility by checking whether candidate claims are
    supported by evidence, assessments, verification signals, GitHub activity,
    and internal consistency.

    Parameters
    ----------
    candidate : dict
        Candidate profile following the standard schema.
    parsed_jd : dict
        Parsed job description (not heavily used by this layer, but kept
        for interface consistency).
    config : dict
        Configuration dict; this layer reads ``config['layer5']``.

    Returns
    -------
    dict
        LayerResult with keys: score, confidence, feature_scores, reasoning,
        warnings, metadata.
    """
    # --- Weights ---
    w_evidence = _get_config(config, "evidence_claim_weight")
    w_assessment = _get_config(config, "assessment_weight")
    w_verification = _get_config(config, "verification_weight")
    w_github = _get_config(config, "github_weight")
    w_consistency = _get_config(config, "consistency_weight")

    # --- Feature scoring ---
    evidence_result = _score_evidence_claim(candidate)
    assessment_result = _score_assessment(candidate)
    verification_result = _score_verification(candidate)
    github_result = _score_github(candidate)
    consistency_result = _score_consistency(candidate)

    # --- Weighted combination ---
    raw_score = (
        evidence_result["score"] * w_evidence
        + assessment_result["score"] * w_assessment
        + verification_result["score"] * w_verification
        + github_result["score"] * w_github
        + consistency_result["score"] * w_consistency
    )
    final_score = round(_clamp(raw_score), 2)

    # --- Confidence ---
    # Higher confidence when we have more data points
    data_signals = 0
    total_signals = 5
    if candidate.get("skills"):
        data_signals += 1
    if candidate.get("redrob_signals", {}).get("skill_assessment_scores"):
        data_signals += 1
    signals = candidate.get("redrob_signals", {})
    if any(
        signals.get(k) is not None
        for k in ("verified_email", "verified_phone", "linkedin_connected")
    ):
        data_signals += 1
    if signals.get("github_activity_score", -1) != -1:
        data_signals += 1
    if candidate.get("career_history"):
        data_signals += 1
    confidence = round(_clamp((data_signals / total_signals) * 100.0), 2)

    # --- Warnings ---
    warnings: list[str] = []
    if evidence_result.get("expert_unsupported", 0) > 0:
        warnings.append(
            f"{evidence_result['expert_unsupported']} expert-level skill(s) have NO career evidence."
        )
    if evidence_result.get("total", 0) == 0:
        warnings.append("No skills listed — evidence-to-claim check skipped.")
    if verification_result.get("verified_count", 0) == 0:
        warnings.append("Zero verification signals confirmed.")
    if not consistency_result.get("check_details", {}).get("experience_duration_match", True):
        warnings.append("Claimed years of experience does not match career history durations (>20% gap).")
    if not consistency_result.get("check_details", {}).get("current_title_match", True):
        warnings.append("Profile current_title does not match latest career history title.")

    # --- Feature scores ---
    feature_scores = {
        "evidence_claim_ratio": evidence_result["score"],
        "assessment_validation": assessment_result["score"],
        "verification_score": verification_result["score"],
        "github_credibility": github_result["score"],
        "internal_consistency": consistency_result["score"],
    }

    # --- Reasoning ---
    reasoning = _build_reasoning(
        evidence_result,
        assessment_result,
        verification_result,
        github_result,
        consistency_result,
        final_score,
    )

    # --- Metadata ---
    metadata = {
        "evidence_detail": {
            "supported": evidence_result.get("supported", 0),
            "unsupported": evidence_result.get("unsupported", 0),
            "expert_unsupported": evidence_result.get("expert_unsupported", 0),
            "high_trust": evidence_result.get("high_trust", 0),
            "total_skills": evidence_result.get("total", 0),
        },
        "assessment_detail": {
            "matched_skills": assessment_result.get("matched_skills", 0),
        },
        "verification_detail": verification_result.get("checks", {}),
        "github_detail": {
            "raw_score": github_result.get("raw", -1),
        },
        "consistency_detail": consistency_result.get("check_details", {}),
        "weights_used": {
            "evidence_claim_weight": w_evidence,
            "assessment_weight": w_assessment,
            "verification_weight": w_verification,
            "github_weight": w_github,
            "consistency_weight": w_consistency,
        },
    }

    return {
        "score": final_score,
        "confidence": confidence,
        "feature_scores": feature_scores,
        "reasoning": reasoning,
        "warnings": warnings,
        "metadata": metadata,
    }
