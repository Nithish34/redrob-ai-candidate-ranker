"""
Layer 6 — Career Intelligence Engine
======================================
Evaluates the candidate's long-term value, career growth, learning velocity,
adaptability, optionality, and hiring risk.

Six features, weighted via config['layer6']:
  1. Career Trajectory   (trajectory_weight, default 0.30)
  2. Learning Velocity   (learning_velocity_weight, default 0.25)
  3. Capability Expansion (expansion_weight, default 0.15)
  4. Career Optionality  (optionality_weight, default 0.10)
  5. Hidden Gem Score    (hidden_gem_weight, default 0.10)
  6. Hiring Risk (inverted) (risk_weight, default 0.10)
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def execute(candidate: dict, parsed_jd: dict, config: dict) -> dict:
    """Predict future growth, versatility, and potential hiring risks for a candidate.

    Parameters
    ----------
    candidate : dict
        Candidate profile with career_history, skills, redrob_signals, etc.
    parsed_jd : dict
        Parsed job description requirements.
    config : dict
        Full pipeline config; layer-specific keys live under config['layer6'].

    Returns
    -------
    dict
        LayerResult with score, confidence, feature_scores, reasoning,
        warnings, and metadata (including future_fit, hiring_risk level).
    """
    layer_cfg = config.get("layer6", {})
    warnings: list[str] = []

    # --- Extract weights ---
    w_traj = layer_cfg.get("trajectory_weight", 0.30)
    w_learn = layer_cfg.get("learning_velocity_weight", 0.25)
    w_expand = layer_cfg.get("expansion_weight", 0.15)
    w_option = layer_cfg.get("optionality_weight", 0.10)
    w_gem = layer_cfg.get("hidden_gem_weight", 0.10)
    w_risk = layer_cfg.get("risk_weight", 0.10)

    # --- Score features ---
    traj_res = _score_trajectory(candidate, warnings)
    learn_res = _score_learning_velocity(candidate, warnings)
    expand_res = _score_capability_expansion(candidate, parsed_jd, warnings)
    option_res = _score_career_optionality(candidate, warnings)
    gem_res = _score_hidden_gem(candidate, warnings)
    risk_res = _score_hiring_risk(candidate, warnings)

    # Inverted risk score (higher score = lower risk = better)
    inverted_risk_score = 100.0 - risk_res["raw_risk"]

    # --- Weighted sum ---
    final_score = (
        traj_res["score"] * w_traj
        + learn_res["score"] * w_learn
        + expand_res["score"] * w_expand
        + option_res["score"] * w_option
        + gem_res["score"] * w_gem
        + inverted_risk_score * w_risk
    )
    final_score = max(0.0, min(100.0, final_score))

    # --- Additional computations ---
    # future_fit = 0.30×learning + 0.25×expansion + 0.20×trajectory + 0.15×evidence_proxy + 0.10×(100-risk)
    future_fit = (
        0.30 * learn_res["score"]
        + 0.25 * expand_res["score"]
        + 0.20 * traj_res["score"]
        + 0.15 * gem_res["metadata"]["evidence_proxy"]
        + 0.10 * inverted_risk_score
    )
    
    # risk level: Low (<30 raw risk), Medium (30-60), High (>60)
    raw_risk = risk_res["raw_risk"]
    if raw_risk < 30:
        risk_level = "Low"
    elif raw_risk <= 60:
        risk_level = "Medium"
    else:
        risk_level = "High"
        warnings.append(f"Candidate classified as HIGH hiring risk (raw risk: {raw_risk:.0f})")

    # --- Confidence ---
    confidence = _compute_confidence(candidate)

    # --- Reasoning ---
    reasoning_parts = [
        f"Trajectory ({w_traj:.0%}): {traj_res['score']:.1f}/100 — {traj_res['reasoning']}",
        f"Learning Velocity ({w_learn:.0%}): {learn_res['score']:.1f}/100 — {learn_res['reasoning']}",
        f"Capability Expansion ({w_expand:.0%}): {expand_res['score']:.1f}/100 — {expand_res['reasoning']}",
        f"Optionality ({w_option:.0%}): {option_res['score']:.1f}/100 — {option_res['reasoning']}",
        f"Hidden Gem ({w_gem:.0%}): {gem_res['score']:.1f}/100 — {gem_res['reasoning']}",
        f"Hiring Risk ({w_risk:.0%}): {inverted_risk_score:.1f}/100 (Risk: {risk_level} — {risk_res['reasoning']})",
    ]

    return {
        "score": round(final_score, 2),
        "confidence": round(confidence, 2),
        "feature_scores": {
            "career_trajectory": round(traj_res["score"], 2),
            "learning_velocity": round(learn_res["score"], 2),
            "capability_expansion": round(expand_res["score"], 2),
            "career_optionality": round(option_res["score"], 2),
            "hidden_gem_score": round(gem_res["score"], 2),
            "hiring_risk_score": round(inverted_risk_score, 2),
        },
        "reasoning": " | ".join(reasoning_parts),
        "warnings": warnings,
        "metadata": {
            "future_fit": round(future_fit, 2),
            "hiring_risk_level": risk_level,
            "raw_hiring_risk": raw_risk,
            "trajectory_detail": traj_res.get("metadata", {}),
            "learning_detail": learn_res.get("metadata", {}),
            "expansion_detail": expand_res.get("metadata", {}),
            "optionality_detail": option_res.get("metadata", {}),
            "gem_detail": gem_res.get("metadata", {}),
            "risk_detail": risk_res.get("metadata", {}),
        },
    }


# ---------------------------------------------------------------------------
# Feature 1 — Career Trajectory (30%)
# ---------------------------------------------------------------------------

def _score_trajectory(candidate: dict, warnings: list[str]) -> dict[str, Any]:
    """Score title progression from earliest to latest role."""
    career_history = candidate.get("career_history", [])
    if not career_history:
        return {
            "score": 50.0,
            "reasoning": "No career history to determine trajectory.",
            "metadata": {"roles_count": 0, "progression_delta": 0}
        }

    # Seniority level mapper
    levels = {
        "intern": 1, "trainee": 1, "apprentice": 1,
        "junior": 2, "jr": 2, "entry": 2,
        "associate": 3, "assoc": 3,
        "engineer": 4, "analyst": 4, "scientist": 4, "developer": 4, "programmer": 4,
        "senior": 5, "sr": 5,
        "lead": 6, "staff": 6, "manager": 6, "team lead": 6,
        "principal": 7, "director": 7, "architect": 7,
        "vp": 8, "head": 8, "chief": 8, "founder": 8, "co-founder": 8
    }

    def get_level(title: str) -> int:
        title_lower = title.lower()
        # Check from highest to lowest keyword to prevent wrong overlaps
        for kw in ["founder", "co-founder", "chief", "head", "vp", "director", "principal", "architect", "lead", "staff", "manager", "senior", "sr", "associate", "junior", "jr", "intern", "trainee"]:
            if kw in title_lower:
                return levels[kw]
        return 4  # Default mid-level engineer/analyst

    # Sort roles by start_date ascending (earliest to latest)
    sorted_roles = sorted(
        [r for r in career_history if r.get("start_date")],
        key=lambda r: r.get("start_date", "")
    )
    if not sorted_roles:
        # Fallback to order in list if start_dates are missing
        sorted_roles = list(career_history)

    role_levels = [get_level(r.get("title", "")) for r in sorted_roles]
    n_roles = len(role_levels)

    if n_roles <= 1:
        # Single role
        current_lvl = role_levels[0] if role_levels else 4
        score = 70.0 if current_lvl >= 5 else 50.0
        reasoning = f"Single role held at level {current_lvl}."
        return {
            "score": score,
            "reasoning": reasoning,
            "metadata": {"role_levels": role_levels, "progression_delta": 0}
        }

    # Calculate progression delta
    first_lvl = role_levels[0]
    last_lvl = role_levels[-1]
    trajectory_delta = last_lvl - first_lvl

    base_score = 60.0
    growth_points = trajectory_delta * 12.0
    
    # Calculate transitions
    upward_transitions = 0
    downward_transitions = 0
    for i in range(n_roles - 1):
        diff = role_levels[i+1] - role_levels[i]
        if diff > 0:
            upward_transitions += 1
        elif diff < 0:
            downward_transitions += 1

    score = base_score + growth_points + (upward_transitions * 5) - (downward_transitions * 10)
    
    # Bonus for senior standing
    if last_lvl >= 5:
        score += 10.0
    if last_lvl >= 7:
        score += 10.0

    score = max(0.0, min(100.0, score))
    
    reasoning = f"Progression: Level {first_lvl} to {last_lvl} over {n_roles} roles. Upward transitions: {upward_transitions}."
    if downward_transitions > 0:
        reasoning += f" Warning: {downward_transitions} demotion/downward move(s) detected."
        warnings.append("Demotion/downward move detected in career trajectory.")

    return {
        "score": score,
        "reasoning": reasoning,
        "metadata": {
            "role_levels": role_levels,
            "progression_delta": trajectory_delta,
            "upward_transitions": upward_transitions,
            "downward_transitions": downward_transitions
        }
    }


# ---------------------------------------------------------------------------
# Feature 2 — Learning Velocity (25%)
# ---------------------------------------------------------------------------

def _score_learning_velocity(candidate: dict, warnings: list[str]) -> dict[str, Any]:
    """Score the speed of technology domain expansion over the candidate's career."""
    career_history = candidate.get("career_history", [])
    profile = candidate.get("profile", {})
    
    if not career_history:
        return {
            "score": 50.0,
            "reasoning": "No career history to determine learning velocity.",
            "metadata": {"unique_domains_count": 0, "years_worked": 0}
        }

    domains = {
        "backend": [
            "python", "django", "flask", "fastapi", "java", "spring", "golang", 
            "node", "ruby", "php", "c#", ".net", "microservices", "sql", 
            "postgresql", "mysql", "redis", "rabbitmq", "kafka"
        ],
        "frontend": [
            "react", "angular", "vue", "javascript", "typescript", "html", 
            "css", "next.js", "bootstrap", "tailwind"
        ],
        "data": [
            "spark", "hadoop", "flink", "databricks", "redshift", "bigquery", 
            "snowflake", "hive", "etl", "data warehouse"
        ],
        "ml_ai": [
            "pytorch", "tensorflow", "keras", "scikit-learn", "sklearn", "pandas", 
            "numpy", "scipy", "transformers", "huggingface", "llm", "langchain", 
            "llama", "openai", "deep learning", "machine learning", "neural networks", 
            "computer vision", "nlp", "spacy", "nltk", "embeddings"
        ],
        "cloud": [
            "aws", "azure", "gcp", "docker", "kubernetes", "terraform", 
            "ansible", "jenkins", "ci/cd", "devops"
        ],
        "mobile": [
            "android", "ios", "flutter", "react native", "swift", "kotlin"
        ],
        "security": [
            "oauth", "jwt", "encryption", "ssl", "firewall", "pentest", "cybersecurity"
        ]
    }

    # Concatenate all career descriptions
    career_text = " ".join((role.get("description") or "").lower() for role in career_history)
    career_text += " " + " ".join((role.get("title") or "").lower() for role in career_history)
    # Also skills
    career_text += " " + " ".join((s.get("name") or "").lower() for s in candidate.get("skills", []))

    matched_domains = []
    for dom_name, keywords in domains.items():
        if any(kw in career_text for kw in keywords):
            matched_domains.append(dom_name)

    total_months = sum(role.get("duration_months") or 0 for role in career_history)
    years_worked = max(0.5, total_months / 12.0)
    
    unique_domains = len(matched_domains)
    
    # Formula: unique_domains / sqrt(years) * 25
    raw_velocity = (unique_domains / (years_worked ** 0.5)) * 25.0
    score = max(10.0, min(100.0, raw_velocity))

    reasoning = f"Acquired {unique_domains} distinct tech domains over {years_worked:.1f} working years."
    
    return {
        "score": score,
        "reasoning": reasoning,
        "metadata": {
            "matched_domains": matched_domains,
            "unique_domains_count": unique_domains,
            "years_worked": years_worked
        }
    }


# ---------------------------------------------------------------------------
# Feature 3 — Capability Expansion (15%)
# ---------------------------------------------------------------------------

def _score_capability_expansion(candidate: dict, parsed_jd: dict, warnings: list[str]) -> dict[str, Any]:
    """Score alignment with required domains and technical versatility."""
    required_domains = parsed_jd.get("required_domains", [])
    if not required_domains:
        return {
            "score": 100.0,
            "reasoning": "No required domains in JD.",
            "metadata": {"matched_domains": 0}
        }

    # Re-use domain keyword check
    # Let's check which required domains are supported in the candidate profile
    from app.layers.layer2_jdfit import _build_candidate_corpus, _DEFAULT_DOMAIN_KEYWORDS
    
    corpus = _build_candidate_corpus(candidate)
    
    matched_count = 0
    matched_list = []
    for domain in required_domains:
        # Standardize domain names
        norm_domain = domain.lower().replace(" ", "_")
        keywords = _DEFAULT_DOMAIN_KEYWORDS.get(norm_domain, [domain.lower()])
        
        if any(kw in corpus for kw in keywords):
            matched_count += 1
            matched_list.append(domain)

    # Base score on fraction of domains matched
    n_required = len(required_domains)
    base_score = (matched_count / n_required) * 100.0
    
    # Adjacent domains boost (backend, cloud, data)
    adjacent_keywords = {
        "backend": ["backend", "microservices", "django", "flask", "fastapi", "golang", "java"],
        "cloud": ["cloud", "aws", "gcp", "azure", "docker", "kubernetes", "mlops"],
        "data": ["data engineering", "spark", "hadoop", "databricks", "bigquery"]
    }
    
    adj_matched = 0
    for adj, kws in adjacent_keywords.items():
        if any(kw in corpus for kw in kws):
            adj_matched += 1
            
    score = base_score + (adj_matched * 5.0)
    score = max(0.0, min(100.0, score))

    reasoning = f"Matched {matched_count}/{n_required} JD domains: {', '.join(matched_list)}. Adjacent domain boost: +{adj_matched * 5:.0f}."

    return {
        "score": score,
        "reasoning": reasoning,
        "metadata": {
            "matched_jd_domains": matched_list,
            "matched_count": matched_count,
            "adjacent_matched_count": adj_matched
        }
    }


# ---------------------------------------------------------------------------
# Feature 4 — Career Optionality (10%)
# ---------------------------------------------------------------------------

def _score_career_optionality(candidate: dict, warnings: list[str]) -> dict[str, Any]:
    """Score versatility based on the number of distinct roles held."""
    career_history = candidate.get("career_history", [])
    if not career_history:
        return {
            "score": 50.0,
            "reasoning": "No career history to determine optionality.",
            "metadata": {"role_types_count": 0}
        }

    role_categories = {
        "ml_engineer": ["machine learning", "ml", "deep learning", "cv", "computer vision", "nlp", "ai", "artificial intelligence"],
        "data_scientist": ["data scientist", "data science", "scientist", "statistician"],
        "data_engineer": ["data engineer", "big data", "data warehouse", "database engineer", "etl"],
        "backend_engineer": ["backend", "software engineer", "software developer", "system engineer", "java engineer", "python engineer"],
        "frontend_engineer": ["frontend", "ui", "web developer", "react developer", "angular developer"],
        "fullstack_engineer": ["fullstack", "full stack", "web developer"],
        "devops_engineer": ["devops", "cloud engineer", "sre", "site reliability", "infrastructure"],
        "manager_lead": ["manager", "lead", "director", "head", "vp", "architect", "founder"]
    }

    matched_roles = set()
    for role in career_history:
        title = (role.get("title") or "").lower()
        for role_cat, keywords in role_categories.items():
            if any(kw in title for kw in keywords):
                matched_roles.add(role_cat)

    role_count = len(matched_roles)
    if role_count >= 3:
        score = 100.0
    elif role_count == 2:
        score = 80.0
    elif role_count == 1:
        score = 60.0
    else:
        score = 20.0

    reasoning = f"Held {role_count} distinct role families ({', '.join(matched_roles) if matched_roles else 'none'})."

    return {
        "score": score,
        "reasoning": reasoning,
        "metadata": {
            "role_types_held": list(matched_roles),
            "role_types_count": role_count
        }
    }


# ---------------------------------------------------------------------------
# Feature 5 — Hidden Gem Score (10%)
# ---------------------------------------------------------------------------

def _score_hidden_gem(candidate: dict, warnings: list[str]) -> dict[str, Any]:
    """Score is higher for candidates with rich profiles/skills but low popularity."""
    career_history = candidate.get("career_history", [])
    signals = candidate.get("redrob_signals", {})

    # 1. Evidence proxy (description lengths and keyword richness)
    total_desc_len = sum(len(role.get("description") or "") for role in career_history)
    len_score = min(100.0, total_desc_len / 15.0)
    
    # Keyword richness
    skills = candidate.get("skills", [])
    skills_count = len(skills)
    skills_score = min(100.0, skills_count * 5.0) # 20+ skills -> 100
    
    evidence_proxy = (len_score + skills_score) / 2.0

    # 2. Popularity proxy (views & recruiter saves)
    views = signals.get("profile_views_received_30d", 0)
    saves = signals.get("saved_by_recruiters_30d", 0)
    
    try: views_val = float(views)
    except (ValueError, TypeError): views_val = 0.0
    try: saves_val = float(saves)
    except (ValueError, TypeError): saves_val = 0.0

    views_score = min(100.0, views_val * 2.0)
    saves_score = min(100.0, saves_val * 10.0)
    
    popularity_proxy = (views_score + saves_score) / 2.0

    # Hidden Gem Score
    score = max(10.0, evidence_proxy - (popularity_proxy * 0.4))
    score = min(100.0, score)

    reasoning = f"Evidence strength: {evidence_proxy:.1f} vs Popularity: {popularity_proxy:.1f}."

    return {
        "score": score,
        "reasoning": reasoning,
        "metadata": {
            "evidence_proxy": evidence_proxy,
            "popularity_proxy": popularity_proxy,
            "description_length": total_desc_len,
            "skills_count": skills_count
        }
    }


# ---------------------------------------------------------------------------
# Feature 6 — Hiring Risk (10%)
# ---------------------------------------------------------------------------

def _score_hiring_risk(candidate: dict, warnings: list[str]) -> dict[str, Any]:
    """Score potential hire risks (job hopping, short stints). Score is raw risk (0-100)."""
    career_history = candidate.get("career_history", [])
    
    if not career_history:
        return {
            "score": 50.0,
            "raw_risk": 50.0,
            "reasoning": "No career history to evaluate hiring risk.",
            "metadata": {"average_tenure_months": 0, "short_stints_count": 0}
        }

    total_months = sum(role.get("duration_months") or 0 for role in career_history)
    n_roles = len(career_history)
    
    # 1. Average tenure
    average_tenure = total_months / n_roles if n_roles > 0 else 0.0
    
    if average_tenure < 12.0:
        raw_risk = 80.0
    elif average_tenure < 18.0:
        raw_risk = 50.0
    elif average_tenure < 24.0:
        raw_risk = 30.0
    else:
        raw_risk = 10.0

    # 2. Too many short stints (<12 months)
    short_stints = 0
    for role in career_history:
        dur = role.get("duration_months") or 0
        if dur > 0 and dur < 12:
            short_stints += 1

    if short_stints >= 3:
        raw_risk += 30.0
    elif short_stints == 2:
        raw_risk += 15.0
    elif short_stints == 1:
        raw_risk += 5.0

    raw_risk = min(100.0, raw_risk)
    
    reasoning = f"Avg tenure: {average_tenure:.1f} mos. Short stints (<12 mos): {short_stints}."
    if raw_risk > 50:
        warnings.append(f"Hiring Risk: short average tenure ({average_tenure:.1f} mos) and/or {short_stints} short stints.")

    return {
        "score": 100.0 - raw_risk, # Keep interface score consistent
        "raw_risk": raw_risk,
        "reasoning": reasoning,
        "metadata": {
            "average_tenure_months": average_tenure,
            "short_stints_count": short_stints
        }
    }


# ---------------------------------------------------------------------------
# Confidence estimation
# ---------------------------------------------------------------------------

def _compute_confidence(candidate: dict) -> float:
    """Confidence in Career Intelligence assessment based on career history length."""
    career_history = candidate.get("career_history", [])
    if not career_history:
        return 10.0
    
    # More roles -> higher confidence
    roles_count = len(career_history)
    if roles_count >= 5:
        return 100.0
    elif roles_count >= 3:
        return 80.0
    elif roles_count >= 2:
        return 60.0
    else:
        return 40.0
