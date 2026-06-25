"""
Layer 4 — Behavior Engine
==========================
Measures RECRUITABILITY: can we actually hire this candidate today?
This includes:
  1. Availability      (availability_weight, default 0.30)
  2. Responsiveness    (responsiveness_weight, default 0.25)
  3. Reliability       (reliability_weight, default 0.20)
  4. Hiring Logistics  (logistics_weight, default 0.15)
  5. Recruiter Engagement (engagement_weight, default 0.10)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def execute(candidate: dict, parsed_jd: dict, config: dict) -> dict:
    """Evaluate candidate recruitability using platform activity and engagement signals.

    Parameters
    ----------
    candidate : dict
        Candidate profile, specifically candidate['redrob_signals'].
    parsed_jd : dict
        Parsed job description requirements.
    config : dict
        Full pipeline config; layer-specific keys live under config['layer4'].

    Returns
    -------
    dict
        LayerResult with score, confidence, feature_scores, reasoning,
        warnings, and metadata.
    """
    layer_cfg = config.get("layer4", {})
    warnings: list[str] = []

    # --- Extract weights ---
    w_avail = layer_cfg.get("availability_weight", 0.30)
    w_resp = layer_cfg.get("responsiveness_weight", 0.25)
    w_rel = layer_cfg.get("reliability_weight", 0.20)
    w_log = layer_cfg.get("logistics_weight", 0.15)
    w_eng = layer_cfg.get("engagement_weight", 0.10)

    # --- Score features ---
    avail_res = _score_availability(candidate, layer_cfg, warnings)
    resp_res = _score_responsiveness(candidate, layer_cfg, warnings)
    rel_res = _score_reliability(candidate, layer_cfg, warnings)
    log_res = _score_logistics(candidate, parsed_jd, layer_cfg, warnings)
    eng_res = _score_engagement(candidate, warnings)

    # --- Weighted sum ---
    final_score = (
        avail_res["score"] * w_avail
        + resp_res["score"] * w_resp
        + rel_res["score"] * w_rel
        + log_res["score"] * w_log
        + eng_res["score"] * w_eng
    )
    final_score = max(0.0, min(100.0, final_score))

    # --- Confidence ---
    confidence = _compute_confidence(candidate)

    # --- Reasoning ---
    reasoning_parts = [
        f"Availability ({w_avail:.0%}): {avail_res['score']:.1f}/100 — {avail_res['reasoning']}",
        f"Responsiveness ({w_resp:.0%}): {resp_res['score']:.1f}/100 — {resp_res['reasoning']}",
        f"Reliability ({w_rel:.0%}): {rel_res['score']:.1f}/100 — {rel_res['reasoning']}",
        f"Logistics ({w_log:.0%}): {log_res['score']:.1f}/100 — {log_res['reasoning']}",
        f"Engagement ({w_eng:.0%}): {eng_res['score']:.1f}/100 — {eng_res['reasoning']}",
    ]

    return {
        "score": round(final_score, 2),
        "confidence": round(confidence, 2),
        "feature_scores": {
            "availability": round(avail_res["score"], 2),
            "responsiveness": round(resp_res["score"], 2),
            "reliability": round(rel_res["score"], 2),
            "hiring_logistics": round(log_res["score"], 2),
            "recruiter_engagement": round(eng_res["score"], 2),
        },
        "reasoning": " | ".join(reasoning_parts),
        "warnings": warnings,
        "metadata": {
            "availability_detail": avail_res.get("metadata", {}),
            "responsiveness_detail": resp_res.get("metadata", {}),
            "reliability_detail": rel_res.get("metadata", {}),
            "logistics_detail": log_res.get("metadata", {}),
            "engagement_detail": eng_res.get("metadata", {}),
        },
    }


# ---------------------------------------------------------------------------
# Feature 1 — Availability (30%)
# ---------------------------------------------------------------------------

def _score_availability(candidate: dict, layer_cfg: dict, warnings: list[str]) -> dict[str, Any]:
    """Score availability based on open to work flag, last active date, and applications submitted."""
    signals = candidate.get("redrob_signals", {})
    
    # open_to_work_flag: True=100, False=20
    open_to_work = signals.get("open_to_work_flag", False)
    otw_score = 100.0 if open_to_work else 20.0
    
    # last_active_date: Within 7 days=100, 30 days=80, 90 days=50, 180 days=20, >180=5
    # Reference date is 2026-06-01
    today_dt = datetime(2026, 6, 1)
    last_active_str = signals.get("last_active_date")
    
    active_score = 5.0
    days_since_active = 999.0
    if last_active_str:
        try:
            active_dt = datetime.strptime(last_active_str, "%Y-%m-%d")
            days_since_active = (today_dt - active_dt).days
            if days_since_active <= 7:
                active_score = 100.0
            elif days_since_active <= 30:
                active_score = 80.0
            elif days_since_active <= 90:
                active_score = 50.0
            elif days_since_active <= 180:
                active_score = 20.0
            else:
                active_score = 5.0
        except ValueError:
            warnings.append(f"Could not parse last_active_date '{last_active_str}'")
            active_score = 20.0 # Neutral default
            
    # applications_submitted_30d: >5=100, 3-5=80, 1-2=50, 0=20
    apps = signals.get("applications_submitted_30d", 0)
    try:
        apps = int(apps)
    except (ValueError, TypeError):
        apps = 0
        
    if apps > 5:
        app_score = 100.0
    elif apps >= 3:
        app_score = 80.0
    elif apps >= 1:
        app_score = 50.0
    else:
        app_score = 20.0
        
    # Weights for sub-features
    f_otw = layer_cfg.get("open_to_work_factor", 0.45)
    f_active = layer_cfg.get("last_active_factor", 0.35)
    f_app = layer_cfg.get("application_factor", 0.20)
    
    score = (otw_score * f_otw) + (active_score * f_active) + (app_score * f_app)
    
    reasoning = (
        f"Open to work: {'Yes' if open_to_work else 'No'} ({otw_score:.0f}); "
        f"Last active: {days_since_active:.0f} days ago ({active_score:.0f}); "
        f"Apps submitted: {apps} ({app_score:.0f})"
    )
    
    return {
        "score": score,
        "reasoning": reasoning,
        "metadata": {
            "open_to_work_flag": open_to_work,
            "days_since_active": days_since_active,
            "applications_submitted_30d": apps,
            "sub_scores": {"otw": otw_score, "active": active_score, "apps": app_score}
        }
    }


# ---------------------------------------------------------------------------
# Feature 2 — Responsiveness (25%)
# ---------------------------------------------------------------------------

def _score_responsiveness(candidate: dict, layer_cfg: dict, warnings: list[str]) -> dict[str, Any]:
    """Score recruiter response rate and speed."""
    signals = candidate.get("redrob_signals", {})
    
    # recruiter_response_rate × 100 (0-1 range)
    rate = signals.get("recruiter_response_rate", 0.0)
    try:
        rate = float(rate)
    except (ValueError, TypeError):
        rate = 0.0
    rate_score = rate * 100.0
    
    # ResponseSpeed = max(0, 100 - (avg_response_time_hours / 72 × 100))
    time_cap = layer_cfg.get("response_time_cap_hours", 72)
    resp_time = signals.get("avg_response_time_hours", time_cap)
    try:
        resp_time = float(resp_time)
    except (ValueError, TypeError):
        resp_time = float(time_cap)
        
    speed_score = max(0.0, 100.0 - (resp_time / time_cap * 100.0))
    
    f_rate = layer_cfg.get("response_rate_factor", 0.70)
    f_speed = layer_cfg.get("response_speed_factor", 0.30)
    
    score = (rate_score * f_rate) + (speed_score * f_speed)
    
    reasoning = f"Response rate: {rate:.0%} ({rate_score:.0f}); Avg response time: {resp_time:.1f} hrs ({speed_score:.0f})"
    
    return {
        "score": score,
        "reasoning": reasoning,
        "metadata": {
            "response_rate": rate,
            "avg_response_time_hours": resp_time,
            "sub_scores": {"rate": rate_score, "speed": speed_score}
        }
    }


# ---------------------------------------------------------------------------
# Feature 3 — Reliability (20%)
# ---------------------------------------------------------------------------

def _score_reliability(candidate: dict, layer_cfg: dict, warnings: list[str]) -> dict[str, Any]:
    """Score interview attendance and offer acceptance history."""
    signals = candidate.get("redrob_signals", {})
    
    # interview_completion_rate (0-1) × 100
    interview_comp = signals.get("interview_completion_rate", 0.0)
    try:
        interview_comp = float(interview_comp)
    except (ValueError, TypeError):
        interview_comp = 0.0
    interview_score = interview_comp * 100.0
    
    # offer_acceptance_rate: if -1 (no data), ignore. If 0-1, multiply by 100.
    offer_acc = signals.get("offer_acceptance_rate", -1.0)
    try:
        offer_acc = float(offer_acc)
    except (ValueError, TypeError):
        offer_acc = -1.0
        
    f_interview = layer_cfg.get("interview_completion_factor", 0.70)
    f_offer = layer_cfg.get("offer_acceptance_factor", 0.30)
    
    if offer_acc < 0:
        # No offer history → score is solely based on interview completion
        score = interview_score
        reasoning = f"Interview attendance: {interview_comp:.0%} ({interview_score:.0f}); No offer history."
    else:
        offer_score = offer_acc * 100.0
        score = (interview_score * f_interview) + (offer_score * f_offer)
        reasoning = f"Interview attendance: {interview_comp:.0%} ({interview_score:.0f}); Offer acceptance: {offer_acc:.0%} ({offer_score:.0f})"
        
    return {
        "score": score,
        "reasoning": reasoning,
        "metadata": {
            "interview_completion_rate": interview_comp,
            "offer_acceptance_rate": offer_acc,
            "sub_scores": {"interview": interview_score, "offer": offer_acc * 100.0 if offer_acc >= 0 else None}
        }
    }


# ---------------------------------------------------------------------------
# Feature 4 — Hiring Logistics (15%)
# ---------------------------------------------------------------------------

def _score_logistics(candidate: dict, parsed_jd: dict, layer_cfg: dict, warnings: list[str]) -> dict[str, Any]:
    """Score notice period, relocation preferences, and work mode compatibility."""
    signals = candidate.get("redrob_signals", {})
    profile = candidate.get("profile", {})
    
    # 1. NoticeScore = max(0, 100 - (notice_period_days / 180 × 100))
    notice_days = signals.get("notice_period_days", 0)
    try:
        notice_days = float(notice_days)
    except (ValueError, TypeError):
        notice_days = 0.0
    notice_score = max(0.0, 100.0 - (notice_days / 180.0 * 100.0))
    
    # 2. RelocationScore: willing_to_relocate=True → 100, False but location matches → 80, False and no match → 40
    relocate = signals.get("willing_to_relocate", False)
    
    # Preferred locations
    jd_locs = layer_cfg.get("jd_preferred_locations", [])
    candidate_loc = (profile.get("location") or "").lower()
    
    loc_match = any(loc.lower() in candidate_loc for loc in jd_locs)
    
    if relocate:
        relocation_score = 100.0
    elif loc_match:
        relocation_score = 80.0
    else:
        relocation_score = 40.0
        
    # 3. WorkModeScore: hybrid/flexible → 100, onsite → 90, remote → 60
    workmode = (signals.get("preferred_work_mode") or "").lower()
    if workmode in ("hybrid", "flexible"):
        workmode_score = 100.0
    elif workmode == "onsite":
        workmode_score = 90.0
    elif workmode == "remote":
        workmode_score = 60.0
    else:
        workmode_score = 80.0 # Neutral fallback
        
    f_notice = layer_cfg.get("notice_factor", 0.50)
    f_reloc = layer_cfg.get("relocation_factor", 0.30)
    f_mode = layer_cfg.get("workmode_factor", 0.20)
    
    score = (notice_score * f_notice) + (relocation_score * f_reloc) + (workmode_score * f_mode)
    
    reasoning = (
        f"Notice: {notice_days:.0f} days ({notice_score:.0f}); "
        f"Relocate: {'Yes' if relocate else 'No'} (matched loc: {'Yes' if loc_match else 'No'}, {relocation_score:.0f}); "
        f"WorkMode: {workmode or 'Unk'} ({workmode_score:.0f})"
    )
    
    return {
        "score": score,
        "reasoning": reasoning,
        "metadata": {
            "notice_period_days": notice_days,
            "willing_to_relocate": relocate,
            "location_matches_jd": loc_match,
            "preferred_work_mode": workmode,
            "sub_scores": {"notice": notice_score, "relocation": relocation_score, "workmode": workmode_score}
        }
    }


# ---------------------------------------------------------------------------
# Feature 5 — Recruiter Engagement (10%)
# ---------------------------------------------------------------------------

def _score_engagement(candidate: dict, warnings: list[str]) -> dict[str, Any]:
    """Score profile popularity and network size."""
    signals = candidate.get("redrob_signals", {})
    
    # 1. profile_views_received_30d: >50=100, >20=80, >10=60, >5=40, else 20
    views = signals.get("profile_views_received_30d", 0)
    try: views = int(views)
    except (ValueError, TypeError): views = 0
    if views > 50: view_score = 100.0
    elif views > 20: view_score = 80.0
    elif views > 10: view_score = 60.0
    elif views > 5: view_score = 40.0
    else: view_score = 20.0
    
    # 2. saved_by_recruiters_30d: >10=100, >5=80, >2=60, >0=40, else 20
    saves = signals.get("saved_by_recruiters_30d", 0)
    try: saves = int(saves)
    except (ValueError, TypeError): saves = 0
    if saves > 10: save_score = 100.0
    elif saves > 5: save_score = 80.0
    elif saves > 2: save_score = 60.0
    elif saves > 0: save_score = 40.0
    else: save_score = 20.0
    
    # 3. search_appearance_30d: >100=100, >50=80, >20=60, >5=40, else 20
    searches = signals.get("search_appearance_30d", 0)
    try: searches = int(searches)
    except (ValueError, TypeError): searches = 0
    if searches > 100: search_score = 100.0
    elif searches > 50: search_score = 80.0
    elif searches > 20: search_score = 60.0
    elif searches > 5: search_score = 40.0
    else: search_score = 20.0
    
    # 4. connection_count: >500=100, >200=80, >100=60, >50=40, else 20
    conns = signals.get("connection_count", 0)
    try: conns = int(conns)
    except (ValueError, TypeError): conns = 0
    if conns > 500: conn_score = 100.0
    elif conns > 200: conn_score = 80.0
    elif conns > 100: conn_score = 60.0
    elif conns > 50: conn_score = 40.0
    else: conn_score = 20.0
    
    score = (view_score + save_score + search_score + conn_score) / 4.0
    
    reasoning = f"Views: {views} ({view_score:.0f}); Saves: {saves} ({save_score:.0f}); Search App: {searches} ({search_score:.0f}); Conns: {conns} ({conn_score:.0f})"
    
    return {
        "score": score,
        "reasoning": reasoning,
        "metadata": {
            "profile_views_received_30d": views,
            "saved_by_recruiters_30d": saves,
            "search_appearance_30d": searches,
            "connection_count": conns,
            "sub_scores": {"views": view_score, "saves": save_score, "searches": search_score, "connections": conn_score}
        }
    }


# ---------------------------------------------------------------------------
# Confidence estimation
# ---------------------------------------------------------------------------

def _compute_confidence(candidate: dict) -> float:
    """Confidence score in behavioral signals, usually high if the dictionary has these keys."""
    signals = candidate.get("redrob_signals", {})
    if not signals:
        return 10.0
        
    # Check what fraction of keys is present
    required_keys = [
        "open_to_work_flag", "last_active_date", "applications_submitted_30d",
        "recruiter_response_rate", "avg_response_time_hours",
        "interview_completion_rate", "notice_period_days", "willing_to_relocate",
        "preferred_work_mode", "profile_views_received_30d", "connection_count"
    ]
    
    present = sum(1 for k in required_keys if signals.get(k) is not None)
    return (present / len(required_keys)) * 100.0
