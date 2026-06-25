"""
Normalizer Engine
=================
Standardizes candidate data fields (skills, company names, dates, etc.)
before any scoring occurs.
"""

from __future__ import annotations

import re


def normalize_candidate(candidate: dict) -> dict:
    """Normalize a candidate dictionary in place, return the normalized dictionary.

    Parameters
    ----------
    candidate : dict
        Raw candidate dictionary.

    Returns
    -------
    dict
        Normalized candidate dictionary.
    """
    # 1. Normalize profile
    profile = candidate.setdefault("profile", {})
    
    # Normalize title and industry
    if "current_title" in profile and profile["current_title"]:
        profile["current_title"] = str(profile["current_title"]).strip()
    else:
        profile["current_title"] = ""
        
    if "headline" in profile and profile["headline"]:
        profile["headline"] = str(profile["headline"]).strip()
    else:
        profile["headline"] = ""
        
    if "location" in profile and profile["location"]:
        profile["location"] = str(profile["location"]).strip()
    else:
        profile["location"] = ""
        
    # Standardize years of experience to float
    exp = profile.get("years_of_experience", 0.0)
    try:
        profile["years_of_experience"] = float(exp)
    except (ValueError, TypeError):
        profile["years_of_experience"] = 0.0

    # 2. Normalize career history
    history = candidate.setdefault("career_history", [])
    for entry in history:
        # Standardize company name
        company = entry.get("company", "")
        if company:
            entry["company"] = _normalize_company_name(str(company))
        else:
            entry["company"] = ""
            
        # Standardize title
        title = entry.get("title", "")
        if title:
            entry["title"] = str(title).strip()
        else:
            entry["title"] = ""
            
        # Ensure duration_months is integer
        dur = entry.get("duration_months")
        try:
            entry["duration_months"] = int(dur) if dur is not None else 0
        except (ValueError, TypeError):
            entry["duration_months"] = 0

    # 3. Normalize skills
    skills = candidate.setdefault("skills", [])
    for skill in skills:
        name = skill.get("name", "")
        if name:
            skill["name"] = _normalize_skill_name(str(name))
        else:
            skill["name"] = ""
            
        proficiency = skill.get("proficiency", "")
        if proficiency:
            skill["proficiency"] = str(proficiency).strip().lower()
        else:
            skill["proficiency"] = "intermediate" # sensible default
            
        endorsements = skill.get("endorsements", 0)
        try:
            skill["endorsements"] = int(endorsements)
        except (ValueError, TypeError):
            skill["endorsements"] = 0

    # 4. Normalize education
    education = candidate.setdefault("education", [])
    for edu in education:
        degree = edu.get("degree", "")
        if degree:
            edu["degree"] = str(degree).strip().lower()
        else:
            edu["degree"] = ""
            
        try:
            edu["start_year"] = int(edu["start_year"]) if edu.get("start_year") else None
            edu["end_year"] = int(edu["end_year"]) if edu.get("end_year") else None
        except (ValueError, TypeError):
            edu["start_year"] = None
            edu["end_year"] = None

    return candidate


def _normalize_skill_name(name: str) -> str:
    """Normalize skill names to match keyword searches (e.g. 'Python 3' -> 'python')."""
    name = name.strip().lower()
    
    # Simple aliases
    if name in ("python3", "python 3", "python programming"):
        return "python"
    if name in ("javascript", "js", "ecmascript"):
        return "javascript"
    if name in ("typescript", "ts"):
        return "typescript"
    if name in ("machine learning", "ml", "machinelearning"):
        return "machine learning"
    if name in ("deep learning", "dl", "deeplearning"):
        return "deep learning"
    if name in ("natural language processing", "nlp"):
        return "nlp"
    if name in ("vector databases", "vector database", "vector db"):
        return "vector database"
    if name in ("semantic search", "semanticsearch"):
        return "semantic search"
        
    return name


def _normalize_company_name(name: str) -> str:
    """Clean company names by stripping common suffixes like LLC, Inc, Ltd."""
    name = name.strip().lower()
    
    # Suffixes pattern
    pattern = r"\b(ltd|limited|llc|inc|incorporated|pvt|private|corp|corporation|co|technologies|solutions|services|systems|gmbh)\b"
    name = re.sub(pattern, "", name)
    name = re.sub(r"\s+", " ", name).strip()
    
    return name
