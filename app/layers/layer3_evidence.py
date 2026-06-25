"""
Layer 3 — Evidence Engine (35% of final score).

Evaluates what the candidate has ACTUALLY BUILT by mining career_history
descriptions for concrete evidence across five capability areas:
  1. Retrieval   (semantic search, vector DBs, embeddings, …)
  2. Ranking     (learning-to-rank, NDCG, re-ranking, …)
  3. Recommendation (collaborative filtering, matching, …)
  4. Production ML (deployment, serving, MLOps, latency, …)
  5. Evaluation  (A/B testing, offline metrics, precision/recall, …)

Evidence is weighted by *action-verb strength* — "built" scores higher
than "learned" — so the layer distinguishes doers from observers.
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# Default keyword banks (used when parsed_jd doesn't override)
# ---------------------------------------------------------------------------

_DEFAULT_KEYWORDS: dict[str, list[str]] = {
    "retrieval": [
        "semantic search", "vector search", "hybrid search",
        "dense retrieval", "sparse retrieval",
        "faiss", "milvus", "pinecone", "qdrant", "weaviate",
        "elasticsearch", "opensearch",
        "embedding", "embeddings", "sentence-transformers", "bge", "e5",
        "information retrieval", "vector database",
        "search index", "inverted index",
    ],
    "ranking": [
        "ranking", "ranker", "learning to rank", "relevance",
        "ndcg", "mrr", "map", "search ranking",
        "re-ranking", "reranking", "candidate ranking",
        "sort", "scoring", "xgboost ranking",
    ],
    "recommendation": [
        "recommendation", "recommender", "matching",
        "personalization", "collaborative filtering",
        "content filtering", "feed ranking", "marketplace",
        "matching engine", "candidate matching",
        "similar items", "user preferences",
    ],
    "production": [
        "production", "deployment", "deploy", "inference",
        "serving", "model serving", "mlops", "ci/cd",
        "monitoring", "scalable", "scale", "millions",
        "api", "microservice", "latency", "throughput",
        "pipeline", "real-time",
    ],
    "evaluation": [
        "evaluation", "a/b testing", "ab testing",
        "offline evaluation", "online evaluation",
        "benchmark", "metrics", "precision", "recall",
        "f1", "ndcg", "mrr", "experiment",
        "statistical significance",
    ],
}

# Default evidence-strength verb map (verb → 0-100 strength score)
_DEFAULT_EVIDENCE_STRENGTH: dict[str, int] = {
    "built":       100,
    "designed":    100,
    "architected": 100,
    "led":         100,
    "implemented": 90,
    "developed":   90,
    "created":     90,
    "engineered":  90,
    "optimized":   85,
    "improved":    85,
    "enhanced":    85,
    "scaled":      85,
    "maintained":  65,
    "managed":     65,
    "supported":   65,
    "contributed": 60,
    "assisted":    45,
    "helped":      45,
    "participated": 40,
    "collaborated": 55,
    "learned":     20,
    "studied":     20,
    "explored":    15,
    "interested":   5,
    "familiar":     5,
}

# Source-type confidence multipliers
_SOURCE_WEIGHTS: dict[str, float] = {
    "career_description": 1.0,
    "profile_summary":    0.75,
    "profile_headline":   0.50,
    "skill":              0.30,
}

# ---------------------------------------------------------------------------
# Helper: extract text blocks from the candidate
# ---------------------------------------------------------------------------

def _gather_text_blocks(candidate: dict) -> list[dict[str, Any]]:
    """Return a list of ``{text, source_type}`` dicts from every relevant
    field in the candidate record.

    Order: career descriptions (primary) → profile summary → headline → skills.
    """
    blocks: list[dict[str, Any]] = []

    # 1. Career-history descriptions (PRIMARY source)
    for role in candidate.get("career_history", []):
        desc = role.get("description", "")
        if desc and isinstance(desc, str) and desc.strip():
            blocks.append({"text": desc, "source_type": "career_description"})

    # 2. Profile summary (secondary)
    summary = candidate.get("profile", {}).get("summary", "")
    if summary and isinstance(summary, str) and summary.strip():
        blocks.append({"text": summary, "source_type": "profile_summary"})

    # 3. Profile headline (tertiary)
    headline = candidate.get("profile", {}).get("headline", "")
    if headline and isinstance(headline, str) and headline.strip():
        blocks.append({"text": headline, "source_type": "profile_headline"})

    # 4. Skills (supporting only, low confidence)
    for skill in candidate.get("skills", []):
        name = skill.get("name", "")
        if name and isinstance(name, str) and name.strip():
            blocks.append({"text": name, "source_type": "skill"})

    return blocks


# ---------------------------------------------------------------------------
# Helper: split text into sentences (lightweight)
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?;])\s+|[\n\r]+')


def _split_sentences(text: str) -> list[str]:
    """Split *text* into rough sentence-level chunks."""
    parts = _SENTENCE_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Helper: compute Evidence Strength Score for a sentence
# ---------------------------------------------------------------------------

def _sentence_strength(sentence_lower: str, strength_map: dict[str, int]) -> int:
    """Return the *highest* evidence-strength score found in *sentence_lower*.

    If no recognised verb is found we still give a baseline of 50 (the keyword
    is present, but we can't assess how deeply the candidate was involved).
    """
    best = 0
    for verb, score in strength_map.items():
        # Match the verb as a whole word to avoid false positives
        # (e.g. "scale" inside "scalable" is fine, but we want word-boundary
        #  matching for short verbs like "led").
        if re.search(r'\b' + re.escape(verb) + r'\b', sentence_lower):
            best = max(best, score)
    # Baseline: keyword exists but no strength verb detected
    return best if best > 0 else 50


# ---------------------------------------------------------------------------
# Helper: score ONE capability area
# ---------------------------------------------------------------------------

_REGEX_CACHE: dict[tuple[str, ...], re.Pattern] = {}

def _get_compiled_pattern(keywords: list[str]) -> re.Pattern:
    key = tuple(keywords)
    if key not in _REGEX_CACHE:
        # Sort keywords by length descending so that longer matches (like "semantic search")
        # are matched before shorter matches (like "search")
        sorted_kws = sorted(keywords, key=len, reverse=True)
        pattern_str = r'\b(' + '|'.join(re.escape(k.lower()) for k in sorted_kws) + r')\b'
        _REGEX_CACHE[key] = re.compile(pattern_str)
    return _REGEX_CACHE[key]


def _score_capability(
    keywords: list[str],
    text_blocks: list[dict[str, Any]],
    strength_map: dict[str, int],
) -> dict[str, Any]:
    """Score a single capability area across all text blocks.

    Returns a dict with:
      - ``score``       : 0-100 normalised score
      - ``match_count`` : total weighted match count
      - ``matches``     : list of individual match records (for transparency)
      - ``sources``     : set of source_type strings that contributed
    """
    matches: list[dict[str, Any]] = []
    total_weighted = 0.0

    if not keywords:
        return {
            "score": 0.0,
            "match_count": 0,
            "total_weighted": 0.0,
            "matches": [],
            "sources": [],
        }

    pattern = _get_compiled_pattern(keywords)

    for block in text_blocks:
        text_lower = block["text"].lower()
        source_type: str = block["source_type"]
        source_mult: float = _SOURCE_WEIGHTS.get(source_type, 0.5)

        sentences = _split_sentences(text_lower)
        # For very short blocks (e.g. a skill name), treat the whole thing
        # as one "sentence".
        if not sentences:
            sentences = [text_lower]

        for sentence in sentences:
            found_words = pattern.findall(sentence)
            if found_words:
                for matched_kw_lower in set(found_words):
                    orig_kw = next((k for k in keywords if k.lower() == matched_kw_lower), matched_kw_lower)
                    ess = _sentence_strength(sentence, strength_map)
                    weighted = ess * source_mult
                    total_weighted += weighted
                    matches.append({
                        "keyword": orig_kw,
                        "strength": ess,
                        "source_type": source_type,
                        "weighted_score": round(weighted, 2),
                    })

    # Normalise to 0-100.
    # Heuristic: a candidate with ~5 strong career-description matches
    # (500 weighted points) should score ~100.
    raw = min(total_weighted / 5.0, 100.0)
    score = round(raw, 2)

    sources_used = list({m["source_type"] for m in matches})

    return {
        "score": score,
        "match_count": len(matches),
        "total_weighted": round(total_weighted, 2),
        "matches": matches,
        "sources": sources_used,
    }


# ---------------------------------------------------------------------------
# Helper: overall confidence estimation
# ---------------------------------------------------------------------------

def _compute_confidence(
    candidate: dict,
    text_blocks: list[dict[str, Any]],
    capability_results: dict[str, dict],
) -> float:
    """Estimate how *confident* we should be in the evidence score.

    Factors:
      - Amount of career text available (more text → higher confidence)
      - Number of career roles with descriptions
      - Whether evidence comes from career descriptions vs only skills
    """
    career_blocks = [b for b in text_blocks if b["source_type"] == "career_description"]
    total_career_chars = sum(len(b["text"]) for b in career_blocks)
    num_roles_with_desc = len(career_blocks)

    # Base confidence from text volume
    # 2000+ chars → full volume credit; <200 chars → low credit
    volume_confidence = min(total_career_chars / 2000.0, 1.0)

    # Breadth: how many roles have descriptions (cap at 5)
    breadth_confidence = min(num_roles_with_desc / 5.0, 1.0)

    # Source quality: fraction of total matches that come from career descriptions
    total_matches = sum(
        r.get("match_count", 0) for r in capability_results.values()
    )
    career_matches = sum(
        sum(1 for m in r.get("matches", []) if m["source_type"] == "career_description")
        for r in capability_results.values()
    )
    source_quality = (career_matches / total_matches) if total_matches > 0 else 0.3

    confidence = (
        0.40 * volume_confidence
        + 0.30 * breadth_confidence
        + 0.30 * source_quality
    )
    return round(min(max(confidence, 0.0), 1.0), 4)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def execute(candidate: dict, parsed_jd: dict, config: dict) -> dict:
    """Layer 3 — Evidence Engine.

    Scans career-history descriptions (and supporting sources) for concrete
    evidence of work across five capability areas, weighted by action-verb
    strength.

    Parameters
    ----------
    candidate : dict
        Candidate record with ``career_history``, ``profile``, ``skills``, etc.
    parsed_jd : dict
        Parsed job description (unused by this layer directly, but available
        for future keyword overrides).
    config : dict
        Configuration dict; this layer reads ``config['layer3']``.

    Returns
    -------
    dict
        ``LayerResult`` with *score*, *confidence*, *feature_scores*,
        *reasoning*, *warnings*, and *metadata* (including
        ``capability_matrix``).
    """
    layer_cfg = config.get("layer3", {})

    # ---- Weights (from config, with defaults) ----------------------------
    retrieval_weight:      float = layer_cfg.get("retrieval_weight", 0.25)
    ranking_weight:        float = layer_cfg.get("ranking_weight", 0.25)
    recommendation_weight: float = layer_cfg.get("recommendation_weight", 0.20)
    production_weight:     float = layer_cfg.get("production_weight", 0.15)
    evaluation_weight:     float = layer_cfg.get("evaluation_weight", 0.15)

    weights: dict[str, float] = {
        "retrieval":      retrieval_weight,
        "ranking":        ranking_weight,
        "recommendation": recommendation_weight,
        "production":     production_weight,
        "evaluation":     evaluation_weight,
    }

    # Evidence-strength verb map (allow config override / merge)
    strength_map: dict[str, int] = dict(_DEFAULT_EVIDENCE_STRENGTH)
    cfg_strength = layer_cfg.get("evidence_strength", {})
    if isinstance(cfg_strength, dict):
        for verb, score in cfg_strength.items():
            strength_map[verb.lower()] = int(score)

    # ---- Gather text blocks from candidate -------------------------------
    text_blocks = _gather_text_blocks(candidate)

    warnings: list[str] = []
    if not text_blocks:
        warnings.append("No text content found in candidate record.")
    career_blocks = [b for b in text_blocks if b["source_type"] == "career_description"]
    if not career_blocks:
        warnings.append(
            "No career-history descriptions available; evidence is based "
            "on profile/skills only (low confidence)."
        )

    # ---- Score each capability area --------------------------------------
    keywords = dict(_DEFAULT_KEYWORDS)
    # Allow parsed_jd to supply additional keywords per capability
    jd_keywords = parsed_jd.get("evidence_keywords", {})
    if isinstance(jd_keywords, dict):
        for cap, extra_kws in jd_keywords.items():
            cap_lower = cap.lower()
            if cap_lower in keywords and isinstance(extra_kws, list):
                keywords[cap_lower] = keywords[cap_lower] + [
                    k for k in extra_kws if k not in keywords[cap_lower]
                ]

    capability_results: dict[str, dict] = {}
    for cap_name, kw_list in keywords.items():
        capability_results[cap_name] = _score_capability(
            kw_list, text_blocks, strength_map,
        )

    # ---- Build feature_scores & capability_matrix ------------------------
    feature_scores: dict[str, float] = {}
    capability_matrix: dict[str, dict[str, Any]] = {}
    reasoning_parts: list[str] = []

    for cap_name in keywords:
        result = capability_results[cap_name]
        cap_score = result["score"]
        w = weights.get(cap_name, 0.0)

        feature_key = f"{cap_name}_evidence"
        feature_scores[feature_key] = cap_score

        capability_matrix[cap_name] = {
            "score": cap_score,
            "weight": w,
            "match_count": result["match_count"],
            "total_weighted": result["total_weighted"],
            "sources": result["sources"],
        }

        if cap_score > 0:
            reasoning_parts.append(
                f"{cap_name.title()}: {cap_score:.1f}/100 "
                f"({result['match_count']} matches from "
                f"{', '.join(result['sources']) if result['sources'] else 'none'})"
            )
        else:
            reasoning_parts.append(f"{cap_name.title()}: no evidence found")

    # ---- Weighted composite score ----------------------------------------
    total_weight = sum(weights.values()) or 1.0
    composite = sum(
        capability_results[cap]["score"] * weights.get(cap, 0.0)
        for cap in keywords
    ) / total_weight
    composite = round(min(max(composite, 0.0), 100.0), 2)

    # ---- Confidence -------------------------------------------------------
    confidence = _compute_confidence(candidate, text_blocks, capability_results)

    # ---- Reasoning string -------------------------------------------------
    reasoning = (
        f"Evidence Engine composite: {composite}/100 "
        f"(confidence {confidence:.0%}). "
        + "; ".join(reasoning_parts)
        + "."
    )

    # ---- Metadata ---------------------------------------------------------
    total_matches = sum(r["match_count"] for r in capability_results.values())
    strongest_cap = max(capability_results, key=lambda c: capability_results[c]["score"])
    weakest_cap = min(capability_results, key=lambda c: capability_results[c]["score"])

    metadata: dict[str, Any] = {
        "capability_matrix": capability_matrix,
        "total_evidence_matches": total_matches,
        "strongest_capability": strongest_cap,
        "weakest_capability": weakest_cap,
        "text_blocks_analyzed": len(text_blocks),
        "career_descriptions_analyzed": len(career_blocks),
    }

    return {
        "score": composite,
        "confidence": confidence,
        "feature_scores": feature_scores,
        "reasoning": reasoning,
        "warnings": warnings,
        "metadata": metadata,
    }
