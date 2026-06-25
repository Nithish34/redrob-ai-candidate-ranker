"""
Layer 1 — Honeypot Filter
=========================
This is a quality gate that detects fake or impossible candidate profiles.
It does NOT score for JD fit — only profile integrity.

Four features, weighted via config['layer1']:
  1. Role Coherence    (role_coherence_weight, default 0.35)
  2. Timeline Consistency (timeline_weight,     default 0.30)
  3. Skill Evidence    (skill_evidence_weight, default 0.25)
  4. Education Match   (education_weight,      default 0.10)

If Honeypot Score < reject_threshold (default 60), the candidate is rejected (passed = False).
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def execute(candidate: dict, parsed_jd: dict, config: dict) -> dict:
    """Evaluate candidate profile integrity.

    Parameters
    ----------
    candidate : dict
        Candidate profile with keys: profile, career_history, skills, education, etc.
    parsed_jd : dict
        Parsed job description requirements.
    config : dict
        Full pipeline config; layer-specific keys live under config['layer1'].

    Returns
    -------
    dict
        LayerResult with score, confidence, feature_scores, reasoning,
        warnings, metadata, and passed (bool).
    """
    layer_cfg = config.get("layer1", {})
    warnings: list[str] = []

    # --- Extract weights and thresholds from config ---
    reject_threshold = layer_cfg.get("reject_threshold", 60)
    w_role = layer_cfg.get("role_coherence_weight", 0.35)
    w_timeline = layer_cfg.get("timeline_weight", 0.30)
    w_skill = layer_cfg.get("skill_evidence_weight", 0.25)
    w_edu = layer_cfg.get("education_weight", 0.10)

    # --- Compute each feature ---
    role_result = _score_role_coherence(candidate, warnings)
    timeline_result = _score_timeline_consistency(candidate, layer_cfg, warnings)
    skill_result = _score_skill_evidence(candidate, warnings)
    edu_result = _score_education_consistency(candidate, layer_cfg, warnings)

    # --- Weighted combination ---
    final_score = (
        role_result["score"] * w_role
        + timeline_result["score"] * w_timeline
        + skill_result["score"] * w_skill
        + edu_result["score"] * w_edu
    )
    final_score = max(0.0, min(100.0, final_score))

    passed = final_score >= reject_threshold

    # --- Confidence ---
    confidence = _compute_confidence(candidate, warnings)

    # --- Reasoning ---
    reasoning_parts = [
        f"Role Coherence ({w_role:.0%}): {role_result['score']:.1f}/100 — {role_result['reasoning']}",
        f"Timeline ({w_timeline:.0%}): {timeline_result['score']:.1f}/100 — {timeline_result['reasoning']}",
        f"Skill Evidence ({w_skill:.0%}): {skill_result['score']:.1f}/100 — {skill_result['reasoning']}",
        f"Education ({w_edu:.0%}): {edu_result['score']:.1f}/100 — {edu_result['reasoning']}",
    ]

    return {
        "score": round(final_score, 2),
        "confidence": round(confidence, 2),
        "feature_scores": {
            "role_coherence": round(role_result["score"], 2),
            "timeline_consistency": round(timeline_result["score"], 2),
            "skill_evidence_consistency": round(skill_result["score"], 2),
            "education_consistency": round(edu_result["score"], 2),
        },
        "reasoning": " | ".join(reasoning_parts),
        "warnings": warnings,
        "passed": passed,
        "metadata": {
            "reject_threshold": reject_threshold,
            "role_detail": role_result.get("metadata", {}),
            "timeline_detail": timeline_result.get("metadata", {}),
            "skill_detail": skill_result.get("metadata", {}),
            "education_detail": edu_result.get("metadata", {}),
        },
    }


# ---------------------------------------------------------------------------
# Feature 1 — Role Coherence (35%)
# ---------------------------------------------------------------------------

def _score_role_coherence(candidate: dict, warnings: list[str]) -> dict[str, Any]:
    """Compare candidate headline/current_title with career history titles."""
    profile = candidate.get("profile", {})
    headline = (profile.get("headline") or "").lower()
    current_title = (profile.get("current_title") or "").lower()
    
    profile_text = f"{headline} {current_title}"
    
    tech_keywords = {
        'engineer', 'developer', 'scientist', 'analyst', 'programmer', 
        'technical', 'architect', 'coder', 'lead', 'tech', 'ai', 'ml', 
        'software', 'data', 'nlp'
    }
    non_tech_keywords = {
        'manager', 'executive', 'writer', 'designer', 'accountant', 
        'support', 'sales', 'marketing', 'hr', 'recruiter', 'recruiting', 
        'customer', 'business', 'admin', 'operations'
    }
    
    has_tech = any(kw in profile_text for kw in tech_keywords)
    has_non_tech = any(kw in profile_text for kw in non_tech_keywords)
    
    # Classify candidate profile
    profile_type = "neutral"
    if has_tech and not has_non_tech:
        profile_type = "tech"
    elif has_non_tech and not has_tech:
        profile_type = "non-tech"
    elif has_tech and has_non_tech:
        # Both present, check which has more keyword hits
        tech_hits = sum(1 for kw in tech_keywords if kw in profile_text)
        non_tech_hits = sum(1 for kw in non_tech_keywords if kw in profile_text)
        profile_type = "tech" if tech_hits >= non_tech_hits else "non-tech"

    career_history = candidate.get("career_history", [])
    if not career_history:
        return {
            "score": 100.0,
            "reasoning": "No career history to evaluate coherence.",
            "metadata": {"profile_type": profile_type, "total_roles": 0, "matched_roles": 0}
        }
        
    matched_roles = 0
    total_roles = len(career_history)
    role_classifications = {}
    
    for role in career_history:
        title = (role.get("title") or "").lower()
        
        # Classify the role title
        r_tech = any(kw in title for kw in tech_keywords)
        r_non_tech = any(kw in title for kw in non_tech_keywords)
        role_type = "neutral"
        if r_tech and not r_non_tech:
            role_type = "tech"
        elif r_non_tech and not r_tech:
            role_type = "non-tech"
        elif r_tech and r_non_tech:
            t_hits = sum(1 for kw in tech_keywords if kw in title)
            nt_hits = sum(1 for kw in non_tech_keywords if kw in title)
            role_type = "tech" if t_hits >= nt_hits else "non-tech"
            
        role_classifications[role.get("title", "Unknown")] = role_type
        
        if profile_type == "tech":
            # Profile is tech, so career roles should be tech or neutral (avoid non-tech)
            if role_type in ("tech", "neutral"):
                matched_roles += 1
        elif profile_type == "non-tech":
            # Profile is non-tech, career roles should be non-tech or neutral
            if role_type in ("non-tech", "neutral"):
                matched_roles += 1
        else:
            # Neutral profile, everything matches
            matched_roles += 1

    score = (matched_roles / total_roles) * 100.0
    
    reasoning = f"Profile type classified as {profile_type}. Matched {matched_roles}/{total_roles} history titles."
    if score < 50:
        warnings.append(f"Significant role family mismatch: Profile is {profile_type} but career titles do not align.")
        
    return {
        "score": score,
        "reasoning": reasoning,
        "metadata": {
            "profile_type": profile_type,
            "total_roles": total_roles,
            "matched_roles": matched_roles,
            "role_classifications": role_classifications
        }
    }


# ---------------------------------------------------------------------------
# Feature 2 — Timeline Consistency (30%)
# ---------------------------------------------------------------------------

def _score_timeline_consistency(candidate: dict, layer_cfg: dict, warnings: list[str]) -> dict[str, Any]:
    """Check for chronological inconsistencies, overlaps, and duration mismatches."""
    penalty_per_violation = layer_cfg.get("timeline_penalty_per_violation", 25)
    violations = 0
    violation_reasons = []

    profile = candidate.get("profile", {})
    career_history = candidate.get("career_history", [])
    education = candidate.get("education", [])
    
    # Check 1: Education end_year vs first job start_date year (allow same year)
    # Parse start years of jobs
    job_start_years = []
    for role in career_history:
        start_date = role.get("start_date")
        if start_date and isinstance(start_date, str):
            try:
                year = int(start_date[:4])
                job_start_years.append(year)
            except ValueError:
                continue
                
    edu_end_years = []
    for edu in education:
        end_year = edu.get("end_year")
        if end_year is not None:
            try:
                edu_end_years.append(int(end_year))
            except ValueError:
                continue
                
    if job_start_years and edu_end_years:
        first_job_year = min(job_start_years)
        max_edu_end_year = max(edu_end_years)
        # If first job started more than 1 year before highest degree ended
        if first_job_year < max_edu_end_year - 1:
            violations += 1
            violation_reasons.append(f"First job start year ({first_job_year}) is before education end year ({max_edu_end_year})")

    # Check 2: Overlapping roles (>2 at the same time)
    # Let's count overlapping intervals
    # We will map each role into a list of active year-month intervals
    reference_today_months = 2026 * 12 + 6
    active_months_counts = {}
    
    def _parse_to_months(date_str: str) -> int | None:
        if not date_str or not isinstance(date_str, str):
            return None
        try:
            parts = date_str.split("-")
            return int(parts[0]) * 12 + int(parts[1])
        except (IndexError, ValueError):
            return None

    for role in career_history:
        start_str = role.get("start_date")
        end_str = role.get("end_date")
        is_current = role.get("is_current", False)
        
        if not start_str:
            continue
            
        start_m = _parse_to_months(start_str)
        if start_m is None:
            continue
            
        if is_current or not end_str:
            end_m = reference_today_months
        else:
            end_m = _parse_to_months(end_str)
            if end_m is None:
                end_m = reference_today_months
                
        # Limit checking range to prevent infinite loops on corrupted data
        if start_m > end_m:
            violations += 1
            violation_reasons.append("Role start date is after end date")
            continue
            
        for m in range(start_m, end_m + 1):
            active_months_counts[m] = active_months_counts.get(m, 0) + 1

    # Check if any month has more than 2 roles
    max_overlaps = max(active_months_counts.values()) if active_months_counts else 0
    if max_overlaps > 2:
        violations += 1
        violation_reasons.append(f"Candidate has {max_overlaps} overlapping roles at the same time (>2)")

    # Check 3: years_of_experience vs actual career duration (>50% mismatch)
    claimed_exp = profile.get("years_of_experience")
    if claimed_exp is not None:
        try:
            claimed_exp = float(claimed_exp)
        except ValueError:
            claimed_exp = -1.0
            
        if claimed_exp >= 0:
            total_months = sum(role.get("duration_months") or 0 for role in career_history)
            computed_years = total_months / 12.0
            
            # Allow small minimum experiences, otherwise calculate percentage mismatch
            mismatch_ratio = 0.0
            if claimed_exp > 0:
                mismatch_ratio = abs(claimed_exp - computed_years) / claimed_exp
            elif computed_years > 0:
                mismatch_ratio = computed_years
                
            if mismatch_ratio > 0.50 and abs(claimed_exp - computed_years) > 1.5:
                violations += 1
                violation_reasons.append(f"Claimed experience ({claimed_exp} yrs) and career duration ({computed_years:.1f} yrs) differ by >50%")

    # Check 4: Impossibly large role durations or negative durations
    for role in career_history:
        duration = role.get("duration_months")
        if duration is not None:
            try:
                duration_val = float(duration)
                if duration_val < 0:
                    violations += 1
                    violation_reasons.append(f"Role has negative duration: {duration_val} months")
                    break
                if duration_val > 600: # 50 years
                    violations += 1
                    violation_reasons.append(f"Role has impossibly long duration: {duration_val:.0f} months")
                    break
            except ValueError:
                continue

    score = max(0.0, 100.0 - (violations * penalty_per_violation))
    
    reasoning = "Timeline looks consistent."
    if violations > 0:
        reasoning = f"Detected {violations} timeline violation(s): " + "; ".join(violation_reasons)
        warnings.extend(violation_reasons)
        
    return {
        "score": score,
        "reasoning": reasoning,
        "metadata": {
            "violations": violations,
            "reasons": violation_reasons,
            "max_overlaps": max_overlaps
        }
    }


# ---------------------------------------------------------------------------
# Feature 3 — Skill Evidence Consistency (25%)
# ---------------------------------------------------------------------------

def _score_skill_evidence(candidate: dict, warnings: list[str]) -> dict[str, Any]:
    """Count how many claimed skills appear in career descriptions or are consistent."""
    skills = candidate.get("skills", [])
    career_history = candidate.get("career_history", [])
    
    if not skills:
        return {
            "score": 100.0,
            "reasoning": "No skills claimed, nothing to verify.",
            "metadata": {"total_skills": 0, "supported_skills": 0}
        }
        
    # Concatenate career descriptions
    descriptions = " ".join((role.get("description") or "").lower() for role in career_history)
    
    total_claims = len(skills)
    supported_claims = 0
    suspicious_skills = []
    
    for skill in skills:
        name = (skill.get("name") or "").lower()
        proficiency = (skill.get("proficiency") or "").lower()
        endorsements = skill.get("endorsements", 0)
        duration_months = skill.get("duration_months", 0)
        
        if not name:
            total_claims -= 1
            continue
            
        has_evidence = name in descriptions
        
        # Check suspicious rules
        # Rule 1: Expert proficiency but duration < 12 months
        is_expert_short = (proficiency == "expert" and duration_months < 12)
        # Rule 2: High endorsements (>20) but no evidence in career text
        is_high_endorsement_no_evidence = (endorsements > 20 and not has_evidence)
        
        is_suspicious = is_expert_short or is_high_endorsement_no_evidence
        
        if has_evidence and not is_suspicious:
            supported_claims += 1
        else:
            if is_suspicious:
                suspicious_skills.append(skill.get("name"))
                
    if total_claims <= 0:
        return {
            "score": 100.0,
            "reasoning": "No valid skills claimed.",
            "metadata": {"total_skills": 0, "supported_skills": 0}
        }
        
    score = (supported_claims / total_claims) * 100.0
    
    reasoning = f"Evidenced {supported_claims}/{total_claims} claimed skills in career history descriptions."
    if suspicious_skills:
        reasoning += f" Suspicious skills detected: {', '.join(suspicious_skills[:5])}"
        warnings.append(f"Suspicious skill claims (e.g. expert but short duration, or high endorsements without evidence): {', '.join(suspicious_skills)}")
        
    return {
        "score": score,
        "reasoning": reasoning,
        "metadata": {
            "total_skills": total_claims,
            "supported_skills": supported_claims,
            "suspicious_skills": suspicious_skills
        }
    }


# ---------------------------------------------------------------------------
# Feature 4 — Education Consistency (10%)
# ---------------------------------------------------------------------------

def _score_education_consistency(candidate: dict, layer_cfg: dict, warnings: list[str]) -> dict[str, Any]:
    """Check for logical degree ordering, future dates, and duration consistency."""
    penalty_per_violation = layer_cfg.get("education_penalty_per_violation", 20)
    violations = 0
    violation_reasons = []
    
    education = candidate.get("education", [])
    if not education:
        return {
            "score": 100.0,
            "reasoning": "No education history listed.",
            "metadata": {"violations": 0}
        }
        
    # Check degree order: Masters (or PhD) should finish after Bachelors
    bachelors_end = None
    masters_start = None
    masters_end = None
    phd_start = None
    
    # Standardize degrees
    for edu in education:
        degree = (edu.get("degree") or "").lower()
        end_year = edu.get("end_year")
        start_year = edu.get("start_year")
        
        try:
            end_yr = int(end_year) if end_year else None
            start_yr = int(start_year) if start_year else None
        except ValueError:
            continue
            
        # Check future dates (relative to current year 2026)
        if (start_yr and start_yr > 2027) or (end_yr and end_yr > 2030):
            violations += 1
            violation_reasons.append(f"Education has invalid future date (start: {start_yr}, end: {end_yr})")
            
        # Check impossibly short/inverted degree durations
        if start_yr and end_yr:
            if end_yr < start_yr:
                violations += 1
                violation_reasons.append(f"Degree ends ({end_yr}) before it starts ({start_yr})")
            elif end_yr - start_yr > 8:
                violations += 1
                violation_reasons.append(f"Degree duration is abnormally long ({end_yr - start_yr} yrs)")

        # Classify degree
        is_bach = any(kw in degree for kw in ("bachelor", "b.tech", "b.e.", "b.s.", "bba", "bca", "b.sc", "bs"))
        is_mast = any(kw in degree for kw in ("master", "m.tech", "m.e.", "m.s.", "mba", "mca", "m.sc", "ms"))
        is_phd = any(kw in degree for kw in ("phd", "ph.d", "doctor", "doctorate"))
        
        if is_bach and end_yr:
            bachelors_end = min(bachelors_end, end_yr) if bachelors_end else end_yr
        if is_mast:
            if start_yr:
                masters_start = min(masters_start, start_yr) if masters_start else start_yr
            if end_yr:
                masters_end = max(masters_end, end_yr) if masters_end else end_yr
        if is_phd and start_yr:
            phd_start = min(phd_start, start_yr) if phd_start else start_yr

    # Rule checks
    if bachelors_end and masters_end and masters_end < bachelors_end:
        violations += 1
        violation_reasons.append(f"Masters end year ({masters_end}) is before Bachelors end year ({bachelors_end})")
    if bachelors_end and phd_start and phd_start < bachelors_end:
        violations += 1
        violation_reasons.append(f"PhD start year ({phd_start}) is before Bachelors end year ({bachelors_end})")

    score = max(0.0, 100.0 - (violations * penalty_per_violation))
    
    reasoning = "Education timeline appears consistent."
    if violations > 0:
        reasoning = f"Detected {violations} education consistency issues: " + "; ".join(violation_reasons)
        warnings.extend(violation_reasons)
        
    return {
        "score": score,
        "reasoning": reasoning,
        "metadata": {
            "violations": violations,
            "reasons": violation_reasons
        }
    }


# ---------------------------------------------------------------------------
# Confidence estimation
# ---------------------------------------------------------------------------

def _compute_confidence(candidate: dict, warnings: list[str]) -> float:
    """Confidence in Honeypot assessment based on completeness of crucial data."""
    profile = candidate.get("profile", {})
    career_history = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    education = candidate.get("education", [])
    
    score = 0.0
    if profile.get("headline") or profile.get("current_title"):
        score += 25
    if career_history:
        score += 35
    if skills:
        score += 20
    if education:
        score += 20
        
    return score
