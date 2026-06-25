"""
Final Ranking Engine — Combines all layer outputs into a single ranked list.

Handles:
- Score normalization (min-max per layer)
- Weighted combination
- Deterministic tie-breaking
- Tier assignment
- Top-100 selection
"""

from app.config.settings import load_config
from app.utils.logger import get_logger

log = get_logger("ranker")


def rank_candidates(candidates_with_scores: list[dict]) -> list[dict]:
    """Rank candidates by combining all layer scores.

    Args:
        candidates_with_scores: List of dicts, each with:
            - candidate_id: str
            - candidate: original candidate dict
            - layer_results: dict mapping layer name to LayerResult dict
              Keys: layer1, layer2, layer3, layer4, layer5, layer6

    Returns:
        Top-100 candidates ranked by final score, each dict includes:
        candidate_id, rank, score, tier, confidence, layer_results, candidate.
    """
    config = load_config()
    weights = config.get("ranking", {})
    tier_thresholds = config.get("tiers", {})

    w_evidence = weights.get("evidence_weight", 0.35)
    w_jdfit = weights.get("jd_fit_weight", 0.25)
    w_behavior = weights.get("behavior_weight", 0.15)
    w_trust = weights.get("trust_weight", 0.15)
    w_career = weights.get("career_weight", 0.10)

    # ── Collect raw layer scores ──────────────────────────────────────
    scored = []
    for entry in candidates_with_scores:
        lr = entry.get("layer_results", {})

        # Layer 1 is a gate — only passed candidates should be here
        l2 = lr.get("layer2", {}).get("score", 0)
        l3 = lr.get("layer3", {}).get("score", 0)
        l4 = lr.get("layer4", {}).get("score", 0)
        l5 = lr.get("layer5", {}).get("score", 0)
        l6 = lr.get("layer6", {}).get("score", 0)

        scored.append({
            **entry,
            "raw_scores": {
                "evidence": l3,
                "jd_fit": l2,
                "behavior": l4,
                "trust": l5,
                "career": l6,
            },
        })

    # ── Min-Max normalization per layer ───────────────────────────────
    if not scored:
        return []

    layer_names = ["evidence", "jd_fit", "behavior", "trust", "career"]
    mins = {}
    maxs = {}
    for ln in layer_names:
        values = [s["raw_scores"][ln] for s in scored]
        mins[ln] = min(values)
        maxs[ln] = max(values)

    for s in scored:
        norm = {}
        for ln in layer_names:
            range_val = maxs[ln] - mins[ln]
            if range_val > 0:
                norm[ln] = (s["raw_scores"][ln] - mins[ln]) / range_val * 100
            else:
                norm[ln] = 50.0  # All same score → neutral
        s["normalized_scores"] = norm

    # ── Weighted final score ──────────────────────────────────────────
    for s in scored:
        n = s["normalized_scores"]
        final = (
            w_evidence * n["evidence"]
            + w_jdfit * n["jd_fit"]
            + w_behavior * n["behavior"]
            + w_trust * n["trust"]
            + w_career * n["career"]
        )
        s["final_score"] = round(final, 4)

        # Confidence = average of layer confidences
        lr = s.get("layer_results", {})
        confidences = [
            lr.get(f"layer{i}", {}).get("confidence", 50)
            for i in range(2, 7)
        ]
        s["confidence"] = round(sum(confidences) / len(confidences), 2)

    # ── Sort with deterministic tie-breaking ──────────────────────────
    scored.sort(key=lambda s: (
        -round(s["final_score"] / 100.0, 4),
        s.get("candidate_id", ""),
    ))

    # ── Top-100 selection + tier assignment ───────────────────────────
    top_100 = scored[:100]
    for i, s in enumerate(top_100, 1):
        s["rank"] = i
        s["tier"] = _assign_tier(s["final_score"], tier_thresholds)

    log.info(
        "Ranked %d candidates. Top score: %.2f, Bottom score: %.2f",
        len(top_100),
        top_100[0]["final_score"] if top_100 else 0,
        top_100[-1]["final_score"] if top_100 else 0,
    )

    return top_100


def _assign_tier(score: float, thresholds: dict) -> str:
    """Assign a hiring tier based on final score."""
    if score >= thresholds.get("A", 95):
        return "A"
    elif score >= thresholds.get("B", 90):
        return "B"
    elif score >= thresholds.get("C", 80):
        return "C"
    elif score >= thresholds.get("D", 70):
        return "D"
    else:
        return "E"
