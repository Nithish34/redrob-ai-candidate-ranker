"""
Layer 2 — JD Fit Engine
========================
Measures how well a candidate matches THIS specific job description
(Senior AI Engineer at Redrob AI).

Four scored features, weighted via config['layer2']:
  1. Experience Fit  (experience_weight, default 0.30)
  2. Domain Fit      (domain_weight,     default 0.40)
  3. Product Exp     (product_weight,    default 0.20)
  4. Startup Fit     (startup_weight,    default 0.10)

Every score lives in 0–100.  All text matching is case-insensitive.
Missing fields are handled gracefully with .get() and sensible defaults.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def execute(candidate: dict, parsed_jd: dict, config: dict) -> dict:
    """Score a candidate's fit against the parsed job description.

    Parameters
    ----------
    candidate : dict
        Candidate profile with keys: profile, career_history, skills, etc.
    parsed_jd : dict
        Parsed job description with required_domains, role_keywords, etc.
    config : dict
        Full pipeline config; layer-specific keys live under config['layer2'].

    Returns
    -------
    dict
        LayerResult with score, confidence, feature_scores, reasoning,
        warnings, and metadata.
    """
    layer_cfg = config.get("layer2", {})
    warnings: list[str] = []

    # --- extract weights from config (never hardcoded) ---
    w_experience = layer_cfg.get("experience_weight", 0.30)
    w_domain = layer_cfg.get("domain_weight", 0.40)
    w_product = layer_cfg.get("product_weight", 0.20)
    w_startup = layer_cfg.get("startup_weight", 0.10)

    # --- compute each feature ---
    exp_result = _score_experience_fit(candidate, layer_cfg, warnings)
    dom_result = _score_domain_fit(candidate, parsed_jd, layer_cfg, warnings)
    prod_result = _score_product_experience(candidate, layer_cfg, warnings)
    startup_result = _score_startup_fit(candidate, layer_cfg, warnings)

    # --- weighted combination ---
    final_score = _clamp(
        exp_result["score"] * w_experience
        + dom_result["score"] * w_domain
        + prod_result["score"] * w_product
        + startup_result["score"] * w_startup
    )

    # --- confidence ---
    confidence = _compute_confidence(candidate, warnings)

    # --- reasoning ---
    reasoning_parts: list[str] = []
    reasoning_parts.append(
        f"Experience Fit ({w_experience:.0%}): {exp_result['score']:.1f}/100 — {exp_result['reasoning']}"
    )
    reasoning_parts.append(
        f"Domain Fit ({w_domain:.0%}): {dom_result['score']:.1f}/100 — {dom_result['reasoning']}"
    )
    reasoning_parts.append(
        f"Product Exp ({w_product:.0%}): {prod_result['score']:.1f}/100 — {prod_result['reasoning']}"
    )
    reasoning_parts.append(
        f"Startup Fit ({w_startup:.0%}): {startup_result['score']:.1f}/100 — {startup_result['reasoning']}"
    )

    return {
        "score": round(final_score, 2),
        "confidence": round(confidence, 2),
        "feature_scores": {
            "experience_fit": round(exp_result["score"], 2),
            "domain_fit": round(dom_result["score"], 2),
            "product_experience": round(prod_result["score"], 2),
            "startup_fit": round(startup_result["score"], 2),
        },
        "reasoning": " | ".join(reasoning_parts),
        "warnings": warnings,
        "metadata": {
            "weights_used": {
                "experience": w_experience,
                "domain": w_domain,
                "product": w_product,
                "startup": w_startup,
            },
            "experience_detail": exp_result.get("metadata", {}),
            "domain_detail": dom_result.get("metadata", {}),
            "product_detail": prod_result.get("metadata", {}),
            "startup_detail": startup_result.get("metadata", {}),
        },
    }


# ---------------------------------------------------------------------------
# Feature 1 — Experience Fit (default 30%)
# ---------------------------------------------------------------------------

def _score_experience_fit(
    candidate: dict,
    layer_cfg: dict,
    warnings: list[str],
) -> dict[str, Any]:
    """Score = max(0, 100 - gap × penalty).  Sweet spot is 5-9 years."""

    ideal_years: float = layer_cfg.get("ideal_experience_years", 7)
    penalty_per_year: float = layer_cfg.get("experience_penalty_per_year", 10)

    profile = candidate.get("profile", {})
    candidate_years = profile.get("years_of_experience")

    if candidate_years is None:
        warnings.append("years_of_experience missing; defaulting experience fit to 0.")
        return {
            "score": 0.0,
            "reasoning": "Experience data unavailable.",
            "metadata": {"candidate_years": None, "ideal_years": ideal_years},
        }

    try:
        candidate_years = float(candidate_years)
    except (TypeError, ValueError):
        warnings.append(
            f"years_of_experience is non-numeric ('{candidate_years}'); defaulting to 0."
        )
        return {
            "score": 0.0,
            "reasoning": "Experience data not parsable.",
            "metadata": {"candidate_years": candidate_years, "ideal_years": ideal_years},
        }

    gap = abs(candidate_years - ideal_years)
    score = _clamp(100.0 - gap * penalty_per_year)

    # Build a human-readable note about the sweet spot
    if 5 <= candidate_years <= 9:
        note = f"{candidate_years:.1f} yrs falls in the 5-9 yr sweet spot"
    elif candidate_years < 5:
        note = f"{candidate_years:.1f} yrs is below the 5-9 yr sweet spot"
    else:
        note = f"{candidate_years:.1f} yrs is above the 5-9 yr sweet spot"

    return {
        "score": score,
        "reasoning": note,
        "metadata": {
            "candidate_years": candidate_years,
            "ideal_years": ideal_years,
            "gap": gap,
        },
    }


# ---------------------------------------------------------------------------
# Feature 2 — Domain Fit (default 40%, highest weight)
# ---------------------------------------------------------------------------

_DEFAULT_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "retrieval": [
        "retrieval", "information retrieval", "rag", "retrieval-augmented",
        "dense retrieval", "sparse retrieval", "bm25", "vector search",
        "semantic search", "embedding retrieval",
    ],
    "ranking": [
        "ranking", "learning to rank", "ltr", "re-ranking", "reranking",
        "candidate ranking", "relevance ranking", "search ranking",
    ],
    "recommendation": [
        "recommendation", "recommender", "collaborative filtering",
        "content-based filtering", "personalization", "rec sys",
        "recommendation engine", "recommendation system",
    ],
    "search": [
        "search", "search engine", "elasticsearch", "solr", "opensearch",
        "full-text search", "query understanding", "query rewriting",
        "search relevance", "search quality",
    ],
    "evaluation": [
        "evaluation", "a/b test", "ab test", "ndcg", "mrr", "precision",
        "recall", "map@k", "offline evaluation", "online evaluation",
        "metrics", "model evaluation",
    ],
    "production_ml": [
        "production ml", "mlops", "ml pipeline", "model serving",
        "feature store", "ml infrastructure", "model deployment",
        "ml system", "ml platform", "kubeflow", "mlflow", "sagemaker",
    ],
}


def _score_domain_fit(
    candidate: dict,
    parsed_jd: dict,
    layer_cfg: dict,
    warnings: list[str],
) -> dict[str, Any]:
    """Check which required domains the candidate's text covers."""

    domain_keywords: dict[str, list[str]] = layer_cfg.get(
        "domain_keywords", _DEFAULT_DOMAIN_KEYWORDS
    )

    if not domain_keywords:
        warnings.append("domain_keywords config is empty; domain score set to 0.")
        return {
            "score": 0.0,
            "reasoning": "No domain keywords configured.",
            "metadata": {},
        }

    # Build a single searchable corpus from the candidate's text signals
    corpus = _build_candidate_corpus(candidate)

    matched_domains: list[str] = []
    unmatched_domains: list[str] = []

    for domain, keywords in domain_keywords.items():
        if _any_keyword_in_text(keywords, corpus):
            matched_domains.append(domain)
        else:
            unmatched_domains.append(domain)

    total = len(domain_keywords)
    score = (len(matched_domains) / total * 100.0) if total > 0 else 0.0

    if not matched_domains:
        reasoning = "No required domains matched in candidate text."
    elif not unmatched_domains:
        reasoning = "All required domains matched."
    else:
        reasoning = (
            f"Matched {len(matched_domains)}/{total} domains: "
            f"{', '.join(matched_domains)}. "
            f"Missing: {', '.join(unmatched_domains)}."
        )

    return {
        "score": _clamp(score),
        "reasoning": reasoning,
        "metadata": {
            "matched_domains": matched_domains,
            "unmatched_domains": unmatched_domains,
            "total_domains": total,
        },
    }


# ---------------------------------------------------------------------------
# Feature 3 — Product Experience (default 20%)
# ---------------------------------------------------------------------------

def _score_product_experience(
    candidate: dict,
    layer_cfg: dict,
    warnings: list[str],
) -> dict[str, Any]:
    """Ratio of months at product companies vs consulting companies.

    Unknown companies default to 'product' (benefit of the doubt).
    """

    product_companies: list[str] = layer_cfg.get("product_companies", [])
    consulting_companies: list[str] = layer_cfg.get("consulting_companies", [])

    # Pre-lower for comparison
    product_lower = [c.lower() for c in product_companies]
    consulting_lower = [c.lower() for c in consulting_companies]

    career_history: list[dict] = candidate.get("career_history", [])

    if not career_history:
        warnings.append("career_history is empty; product experience score defaulting to 50.")
        return {
            "score": 50.0,
            "reasoning": "No career history available; neutral default.",
            "metadata": {"product_months": 0, "consulting_months": 0, "unknown_months": 0},
        }

    product_months = 0.0
    consulting_months = 0.0
    unknown_months = 0.0
    company_classifications: dict[str, str] = {}

    for role in career_history:
        company_name: str = role.get("company", "") or ""
        duration: float = _safe_float(role.get("duration_months"), 0.0)
        classification = _classify_company(
            company_name, product_lower, consulting_lower
        )
        company_classifications[company_name] = classification

        if classification == "consulting":
            consulting_months += duration
        elif classification == "product":
            product_months += duration
        else:
            # Unknown → benefit of the doubt → treat as product
            unknown_months += duration
            product_months += duration

    total_months = product_months + consulting_months
    if total_months <= 0:
        warnings.append("Total career months is 0; product experience defaulting to 50.")
        return {
            "score": 50.0,
            "reasoning": "No duration data; neutral default.",
            "metadata": {
                "product_months": product_months,
                "consulting_months": consulting_months,
                "unknown_months": unknown_months,
            },
        }

    score = _clamp(product_months / total_months * 100.0)

    reasoning = (
        f"{product_months:.0f} product months vs {consulting_months:.0f} consulting months "
        f"({unknown_months:.0f} unclassified → credited as product)."
    )

    return {
        "score": score,
        "reasoning": reasoning,
        "metadata": {
            "product_months": product_months,
            "consulting_months": consulting_months,
            "unknown_months": unknown_months,
            "total_months": total_months,
            "company_classifications": company_classifications,
        },
    }


# ---------------------------------------------------------------------------
# Feature 4 — Startup Fit (default 10%)
# ---------------------------------------------------------------------------

_DEFAULT_STARTUP_SIGNALS: list[str] = [
    "startup", "early stage", "0 to 1", "zero to one", "founding",
    "bootstrapped", "seed stage", "series a", "series b",
    "scrappy", "wear many hats", "full-stack ownership",
    "built from scratch", "greenfield", "mvp", "product-market fit",
    "fast-paced", "hypergrowth", "scale-up",
]


def _score_startup_fit(
    candidate: dict,
    layer_cfg: dict,
    warnings: list[str],
) -> dict[str, Any]:
    """Scan career_history descriptions for startup-mindset signals."""

    startup_signals: list[str] = layer_cfg.get(
        "startup_signals", _DEFAULT_STARTUP_SIGNALS
    )

    if not startup_signals:
        warnings.append("startup_signals config is empty; startup score set to 0.")
        return {
            "score": 0.0,
            "reasoning": "No startup signals configured.",
            "metadata": {},
        }

    # Build corpus from career descriptions (and title/company for extra signal)
    corpus = _build_career_corpus(candidate)

    matched_signals: list[str] = []
    for signal in startup_signals:
        if signal.lower() in corpus:
            matched_signals.append(signal)

    total = len(startup_signals)
    score = _clamp(len(matched_signals) / total * 100.0) if total > 0 else 0.0

    if matched_signals:
        reasoning = f"Matched {len(matched_signals)}/{total} startup signals: {', '.join(matched_signals[:5])}"
        if len(matched_signals) > 5:
            reasoning += f" (+{len(matched_signals) - 5} more)"
    else:
        reasoning = "No startup signals found in career history."

    # Also consider company_size as a supplementary signal
    company_size_bonus, size_note = _company_size_startup_bonus(candidate)
    if company_size_bonus > 0:
        score = _clamp(score + company_size_bonus)
        reasoning += f" | {size_note}"

    return {
        "score": score,
        "reasoning": reasoning,
        "metadata": {
            "matched_signals": matched_signals,
            "total_signals": total,
            "company_size_bonus": company_size_bonus,
        },
    }


# ---------------------------------------------------------------------------
# Helpers — corpus building
# ---------------------------------------------------------------------------

def _build_candidate_corpus(candidate: dict) -> str:
    """Concatenate all searchable text from the candidate into one
    lower-cased string for keyword matching."""

    parts: list[str] = []

    profile = candidate.get("profile", {})
    parts.append(profile.get("headline", "") or "")
    parts.append(profile.get("summary", "") or "")

    for role in candidate.get("career_history", []):
        parts.append(role.get("description", "") or "")
        parts.append(role.get("title", "") or "")

    # Also pull in skill names (someone listing 'mlops' as a skill is a signal)
    for skill in candidate.get("skills", []):
        parts.append(skill.get("name", "") or "")

    return " ".join(parts).lower()


def _build_career_corpus(candidate: dict) -> str:
    """Build a lower-cased text blob from career history entries only."""

    parts: list[str] = []
    for role in candidate.get("career_history", []):
        parts.append(role.get("description", "") or "")
        parts.append(role.get("title", "") or "")
        parts.append(role.get("company", "") or "")

    # Include company_size values (e.g., "startup", "small")
    profile = candidate.get("profile", {})
    parts.append(profile.get("current_company_size", "") or "")

    return " ".join(parts).lower()


# ---------------------------------------------------------------------------
# Helpers — matching & classification
# ---------------------------------------------------------------------------

def _any_keyword_in_text(keywords: list[str], text: str) -> bool:
    """Return True if any keyword appears in text (both already lower-cased or
    lowered here for safety)."""
    for kw in keywords:
        if kw.lower() in text:
            return True
    return False


def _classify_company(
    company_name: str,
    product_lower: list[str],
    consulting_lower: list[str],
) -> str:
    """Classify a company as 'product', 'consulting', or 'unknown'.

    Uses partial (substring) matching so that 'flipkart' matches
    'Flipkart Technologies'.
    """
    name_lower = company_name.lower().strip()
    if not name_lower:
        return "unknown"

    for c in consulting_lower:
        if c in name_lower or name_lower in c:
            return "consulting"

    for p in product_lower:
        if p in name_lower or name_lower in p:
            return "product"

    # Unknown → defaults to product at the caller level
    return "unknown"


def _company_size_startup_bonus(candidate: dict) -> tuple[float, str]:
    """Give a small bonus if the candidate has worked at small / startup-sized
    companies (based on company_size field)."""

    startup_sizes = {"startup", "small", "1-10", "11-50", "51-200"}
    career_history: list[dict] = candidate.get("career_history", [])

    startup_roles = 0
    total_roles = len(career_history)

    for role in career_history:
        size = (role.get("company_size", "") or "").lower().strip()
        if size in startup_sizes:
            startup_roles += 1

    # Also check current company size from profile
    profile = candidate.get("profile", {})
    current_size = (profile.get("current_company_size", "") or "").lower().strip()
    if current_size in startup_sizes:
        startup_roles += 1
        total_roles += 1

    if total_roles == 0:
        return 0.0, ""

    ratio = startup_roles / total_roles
    # Cap bonus at 15 points
    bonus = min(ratio * 15.0, 15.0)
    note = f"company_size bonus: {startup_roles}/{total_roles} roles at startup-sized cos (+{bonus:.1f}pts)"
    return bonus, note


# ---------------------------------------------------------------------------
# Helpers — confidence
# ---------------------------------------------------------------------------

def _compute_confidence(candidate: dict, warnings: list[str]) -> float:
    """Estimate how confident we are in the final score based on data
    completeness."""

    confidence = 1.0

    # Penalise for each warning (data-quality issue)
    confidence -= len(warnings) * 0.10

    profile = candidate.get("profile", {})
    if not profile.get("years_of_experience"):
        confidence -= 0.10
    if not profile.get("headline") and not profile.get("summary"):
        confidence -= 0.10

    career_history = candidate.get("career_history", [])
    if not career_history:
        confidence -= 0.15
    else:
        # Penalise if most descriptions are empty
        described = sum(
            1 for r in career_history if (r.get("description") or "").strip()
        )
        if described < len(career_history) * 0.5:
            confidence -= 0.10

    return max(round(confidence, 2), 0.0)


# ---------------------------------------------------------------------------
# Helpers — numeric utilities
# ---------------------------------------------------------------------------

def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    """Clamp a value to [lo, hi]."""
    return max(lo, min(hi, value))


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert value to float, returning *default* on failure."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
