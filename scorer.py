"""
Redrob AI Candidate Ranker — Scoring Engine
============================================
Pure Python (stdlib only). No external dependencies.
Shared by both rank.py (CLI) and app.py (Gradio HF Space).

Scoring Components
------------------
1. AI/ML Core Skills Match   — 40%
2. Title & Career Trajectory — 25%
3. Years of Experience       — 15%
4. Availability & Logistics  — 10%
5. Education                 — 10%
+ Behavioral Modifier        — multiplicative (×0.80 – ×1.20)
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

# ---------------------------------------------------------------------------
# 1. AI/ML Skill Taxonomy  (name → importance weight 0-1)
# ---------------------------------------------------------------------------

AI_SKILL_TAXONOMY: dict[str, float] = {
    # ── Core ML frameworks ──────────────────────────────────────────────────
    "python": 1.0,
    "pytorch": 1.0,
    "tensorflow": 1.0,
    "keras": 0.90,
    "scikit-learn": 0.90,
    "scikit learn": 0.90,
    "sklearn": 0.90,
    "xgboost": 0.85,
    "lightgbm": 0.85,
    "catboost": 0.80,
    "jax": 0.85,
    # ── LLMs & GenAI ────────────────────────────────────────────────────────
    "llm": 1.00,
    "large language model": 1.00,
    "fine-tuning llms": 1.00,
    "fine tuning llms": 1.00,
    "finetuning": 0.90,
    "lora": 0.90,
    "qlora": 0.90,
    "rlhf": 0.95,
    "prompt engineering": 0.85,
    "langchain": 0.85,
    "llamaindex": 0.85,
    "llama index": 0.85,
    "rag": 0.90,
    "retrieval augmented generation": 0.90,
    "huggingface": 0.90,
    "hugging face": 0.90,
    "transformers": 1.00,
    "bert": 0.90,
    "gpt": 0.90,
    "llama": 0.90,
    "mistral": 0.85,
    "stable diffusion": 0.85,
    "diffusion models": 0.85,
    "generative ai": 1.00,
    "generative models": 0.90,
    # ── NLP ─────────────────────────────────────────────────────────────────
    "nlp": 1.00,
    "natural language processing": 1.00,
    "text classification": 0.85,
    "sentiment analysis": 0.80,
    "named entity recognition": 0.80,
    "ner": 0.80,
    "text generation": 0.85,
    "machine translation": 0.80,
    "question answering": 0.85,
    "summarization": 0.80,
    # ── Computer Vision ─────────────────────────────────────────────────────
    "computer vision": 1.00,
    "image classification": 0.90,
    "object detection": 0.90,
    "image segmentation": 0.85,
    "gans": 0.85,
    "opencv": 0.80,
    "yolo": 0.85,
    # ── Deep Learning ───────────────────────────────────────────────────────
    "deep learning": 1.00,
    "neural networks": 0.90,
    "neural network": 0.90,
    "cnn": 0.85,
    "lstm": 0.85,
    "rnn": 0.85,
    "attention mechanism": 0.90,
    "self-supervised learning": 0.90,
    "reinforcement learning": 0.90,
    "transfer learning": 0.85,
    # ── MLOps & Infra ───────────────────────────────────────────────────────
    "mlops": 1.00,
    "mlflow": 0.90,
    "weights & biases": 0.85,
    "wandb": 0.85,
    "kubeflow": 0.85,
    "bentoml": 0.80,
    "ray": 0.80,
    "ray serve": 0.80,
    "feature engineering": 0.85,
    "feature store": 0.85,
    "model deployment": 0.85,
    "model serving": 0.85,
    # ── Speech ──────────────────────────────────────────────────────────────
    "speech recognition": 0.90,
    "tts": 0.85,
    "text to speech": 0.85,
    "asr": 0.85,
    # ── Vector DBs ──────────────────────────────────────────────────────────
    "milvus": 0.85,
    "pinecone": 0.85,
    "weaviate": 0.80,
    "chroma": 0.80,
    "vector database": 0.85,
    "vector db": 0.85,
    # ── Cloud ML ────────────────────────────────────────────────────────────
    "sagemaker": 0.90,
    "vertex ai": 0.90,
    "azure ml": 0.85,
    # ── Data Science ────────────────────────────────────────────────────────
    "data science": 0.90,
    "machine learning": 1.00,
    "statistical modeling": 0.80,
    "time series": 0.80,
    "recommendation systems": 0.85,
    "anomaly detection": 0.80,
    "a/b testing": 0.70,
    # ── Secondary / supporting ──────────────────────────────────────────────
    "spark": 0.55,
    "airflow": 0.50,
    "docker": 0.50,
    "kubernetes": 0.50,
    "sql": 0.45,
    "git": 0.30,
}

# Short abbreviations that must EXACTLY match the skill name (to avoid false positives)
_EXACT_ONLY: frozenset[str] = frozenset({"ml", "ai", "nlp", "rnn", "cnn", "ner", "rag",
                                          "tts", "asr", "gpt", "sql", "git", "ray", "jax",
                                          "gans"})

# ---------------------------------------------------------------------------
# 2. Title taxonomy  (lower-cased title fragment → score 0-1)
# ---------------------------------------------------------------------------

_TITLE_TIERS: list[tuple[list[str], float]] = [
    # Tier 1 — directly AI/ML (score 0.90–1.00)
    (["machine learning engineer", "ml engineer", "senior machine learning engineer",
      "staff machine learning engineer", "principal ml engineer"], 1.00),
    (["ai engineer", "artificial intelligence engineer", "ai/ml engineer",
      "ai ml engineer"], 1.00),
    (["research scientist", "applied scientist", "research engineer",
      "ai researcher", "ml researcher"], 1.00),
    (["nlp engineer", "natural language processing engineer"], 1.00),
    (["computer vision engineer", "cv engineer"], 1.00),
    (["deep learning engineer", "neural network engineer"], 1.00),
    (["mlops engineer", "ml platform engineer", "ml infrastructure engineer"], 1.00),
    (["data scientist", "senior data scientist", "staff data scientist",
      "junior data scientist"], 0.95),
    (["junior ml engineer", "associate ml engineer", "ml engineer intern"], 0.85),
    # Tier 2 — adjacent (0.40–0.70)
    (["data engineer", "senior data engineer"], 0.65),
    (["analytics engineer"], 0.60),
    (["quantitative analyst", "quant analyst"], 0.60),
    (["data analyst", "senior data analyst", "business intelligence"], 0.50),
    (["software engineer", "backend engineer", "full stack engineer",
      "full-stack engineer", "frontend engineer"], 0.35),
    (["platform engineer", "devops engineer", "cloud engineer"], 0.30),
    (["business analyst"], 0.15),
    (["project manager", "product manager"], 0.12),
    # Tier 3 — unrelated (≤0.10)
    (["civil engineer", "mechanical engineer", "structural engineer",
      "chemical engineer"], 0.05),
    (["hr manager", "human resources", "talent acquisition",
      "recruiter"], 0.05),
    (["marketing manager", "digital marketing", "content writer",
      "content strategist", "seo"], 0.05),
    (["accountant", "finance manager", "ca ", "cfo", "chartered accountant"], 0.03),
    (["sales executive", "sales manager", "account executive",
      "business development"], 0.05),
    (["graphic designer", "ui designer", "ux designer",
      "visual designer"], 0.05),
    (["operations manager", "operations executive", "supply chain"], 0.06),
    (["customer support", "customer service", "support engineer"], 0.04),
]

# Build flat lookup
_TITLE_SCORE_MAP: list[tuple[str, float]] = []
for _phrases, _score in _TITLE_TIERS:
    for _phrase in _phrases:
        _TITLE_SCORE_MAP.append((_phrase.lower(), _score))

# Keywords that bump an otherwise-unknown title to 0.80
_AI_TITLE_KEYWORDS = frozenset([
    "machine learning", "deep learning", "computer vision",
    "natural language", "neural", "mlops", "llm", "genai",
    "generative", "data science",
])

# ---------------------------------------------------------------------------
# 3. Education lookups
# ---------------------------------------------------------------------------

_TIER_SCORE = {"tier_1": 1.00, "tier_2": 0.82, "tier_3": 0.62,
               "tier_4": 0.42, "unknown": 0.52}

_DEGREE_SCORE: list[tuple[str, float]] = [
    ("ph.d", 1.00), ("phd", 1.00), ("doctorate", 1.00),
    ("m.tech", 0.92), ("m.e.", 0.92), ("m.s.", 0.88), ("ms", 0.88),
    ("m.sc", 0.85), ("msc", 0.85), ("m.eng", 0.88),
    ("mba", 0.62),
    ("b.tech", 0.78), ("b.e.", 0.78), ("b.e", 0.78), ("be", 0.78),
    ("b.sc", 0.68), ("bsc", 0.68), ("b.s.", 0.70), ("bs", 0.70),
]

_RELEVANT_FIELDS = frozenset([
    "computer science", "information technology", "software engineering",
    "electronics", "electrical", "mathematics", "statistics",
    "data science", "artificial intelligence", "machine learning",
    "computational", "physics", "math", "engineering",
])

# ---------------------------------------------------------------------------
# Proficiency multiplier for skills
# ---------------------------------------------------------------------------

_PROF_MULT = {"beginner": 0.50, "intermediate": 0.75,
              "advanced": 1.00, "expert": 1.05}

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _today() -> date:
    return date.today()


def _parse_date(ds: str | None) -> date | None:
    if not ds:
        return None
    try:
        return date.fromisoformat(str(ds)[:10])
    except (ValueError, TypeError):
        return None


def _days_since(ds: str | None) -> int:
    """Days since date string. Returns 9999 if missing/unparseable."""
    d = _parse_date(ds)
    return 9999 if d is None else (_today() - d).days


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _match_skill(name_lower: str) -> float:
    """Return taxonomy importance weight for a skill name, or 0.0 if not matched."""
    # 1. Exact match
    if name_lower in AI_SKILL_TAXONOMY:
        return AI_SKILL_TAXONOMY[name_lower]
    # 2. Substring match — only for terms longer than 4 chars and not in exact-only set
    best = 0.0
    for key, weight in AI_SKILL_TAXONOMY.items():
        if key in _EXACT_ONLY:
            continue
        if len(key) > 4 and (key in name_lower or (len(name_lower) > 4 and name_lower in key)):
            best = max(best, weight)
    return best


# ===========================================================================
# Component scorers
# ===========================================================================

def _score_skills(candidate: dict) -> tuple[float, int]:
    """
    Returns (normalized_score 0-1, count_of_matched_ai_skills).
    """
    skills: list[dict] = candidate.get("skills", [])
    assessment_scores: dict[str, float] = (
        candidate.get("redrob_signals", {}).get("skill_assessment_scores", {})
    )

    total_weight = 0.0
    matched_count = 0

    for sk in skills:
        name = sk.get("name", "").lower().strip()
        if not name:
            continue

        importance = _match_skill(name)
        if importance == 0.0:
            continue

        matched_count += 1
        prof = _PROF_MULT.get(sk.get("proficiency", "beginner"), 0.50)
        endorsements = sk.get("endorsements", 0)
        duration_months = sk.get("duration_months", 0)

        # Quality gate: penalise likely keyword stuffers
        if endorsements < 2 and duration_months < 4:
            quality = 0.45
        elif endorsements >= 30:
            quality = 1.12
        elif endorsements >= 10:
            quality = 1.05
        else:
            quality = 1.00

        # Assessment score bonus (platform-verified skill)
        assess_bonus = 1.0
        for aname, ascore in assessment_scores.items():
            if aname.lower() in name or name in aname.lower():
                assess_bonus = 1.0 + (ascore / 100.0) * 0.15
                break

        total_weight += importance * prof * quality * assess_bonus

    # Saturating normalisation: 12+ rich, diverse AI skills → 1.0
    score = _clamp(total_weight / 12.0)
    return score, matched_count


def _score_title(candidate: dict) -> float:
    current_title = candidate.get("profile", {}).get("current_title", "").lower()

    # Direct lookup in title map
    best = 0.0
    for phrase, score in _TITLE_SCORE_MAP:
        if phrase in current_title:
            best = max(best, score)

    # Keyword bump for unlisted titles
    for kw in _AI_TITLE_KEYWORDS:
        if kw in current_title:
            best = max(best, 0.80)

    # Career history: past AI/ML roles contribute 30 %
    career_bonus = 0.0
    for job in candidate.get("career_history", [])[:5]:
        jtitle = job.get("title", "").lower()
        for phrase, score in _TITLE_SCORE_MAP:
            if phrase in jtitle:
                career_bonus = max(career_bonus, score * 0.30)

        # ML keywords in role description
        desc = job.get("description", "").lower()
        ml_desc_kws = ["machine learning", "deep learning", "neural network",
                       "model training", "llm", "nlp", "computer vision",
                       "pytorch", "tensorflow", "transformer", "fine-tun"]
        if any(kw in desc for kw in ml_desc_kws):
            career_bonus = max(career_bonus, 0.25)

    return _clamp(best + career_bonus)


def _score_experience(candidate: dict) -> float:
    yoe = float(candidate.get("profile", {}).get("years_of_experience", 0))
    if 3.0 <= yoe <= 8.0:
        return 1.00
    elif 2.0 <= yoe < 3.0 or 8.0 < yoe <= 12.0:
        return 0.85
    elif 1.0 <= yoe < 2.0 or 12.0 < yoe <= 15.0:
        return 0.65
    elif yoe < 1.0:
        return 0.35
    else:  # > 15
        return 0.50


def _score_availability(candidate: dict) -> float:
    sig = candidate.get("redrob_signals", {})
    score = 0.0

    if sig.get("open_to_work_flag", False):
        score += 0.40

    notice = sig.get("notice_period_days", 90)
    if notice <= 15:
        score += 0.35
    elif notice <= 30:
        score += 0.28
    elif notice <= 60:
        score += 0.18
    elif notice <= 90:
        score += 0.08
    # else 0

    if sig.get("willing_to_relocate", False):
        score += 0.15

    mode = sig.get("preferred_work_mode", "")
    if mode in ("remote", "flexible", "hybrid"):
        score += 0.10

    return _clamp(score)


def _score_education(candidate: dict) -> float:
    education: list[dict] = candidate.get("education", [])
    if not education:
        return 0.30

    best = 0.0
    for edu in education:
        tier_s = _TIER_SCORE.get(edu.get("tier", "unknown"), 0.52)

        degree = edu.get("degree", "").lower()
        deg_s = 0.60
        for key, val in _DEGREE_SCORE:
            if key in degree:
                deg_s = max(deg_s, val)

        field = edu.get("field_of_study", "").lower()
        field_ok = any(f in field for f in _RELEVANT_FIELDS)
        field_s = 0.20 if field_ok else 0.0

        combined = tier_s * 0.50 + deg_s * 0.30 + field_s
        best = max(best, combined)

    return _clamp(best)


def _behavioral_modifier(candidate: dict) -> float:
    sig = candidate.get("redrob_signals", {})
    mod = 1.0

    # Profile completeness  (-0.05 to +0.05)
    completeness = sig.get("profile_completeness_score", 50.0) / 100.0
    mod += (completeness - 0.5) * 0.10

    # Recruiter response rate  (-0.04 to +0.04)
    rr = sig.get("recruiter_response_rate", 0.5)
    mod += (rr - 0.5) * 0.08

    # GitHub activity (-1 = no GitHub)  (0 to +0.04)
    gh = sig.get("github_activity_score", -1)
    if gh >= 0:
        mod += (gh / 100.0) * 0.04

    # Interview completion rate  (-0.02 to +0.02)
    icr = sig.get("interview_completion_rate", 0.5)
    mod += (icr - 0.5) * 0.04

    # Recency of activity
    days_inactive = _days_since(sig.get("last_active_date"))
    if days_inactive < 30:
        mod += 0.02
    elif days_inactive < 60:
        mod += 0.01
    elif days_inactive > 365:
        mod -= 0.05
    elif days_inactive > 180:
        mod -= 0.03

    # Trust signals  (up to +0.02)
    if sig.get("verified_email", False):
        mod += 0.007
    if sig.get("verified_phone", False):
        mod += 0.007
    if sig.get("linkedin_connected", False):
        mod += 0.006

    return _clamp(mod, 0.85, 1.15)


# ===========================================================================
# Weights
# ===========================================================================

WEIGHTS = {
    "skills":       0.40,
    "title":        0.25,
    "experience":   0.15,
    "availability": 0.10,
    "education":    0.10,
}

# ===========================================================================
# Public API
# ===========================================================================

def score_candidate(candidate: dict) -> dict:
    """
    Score a single candidate profile dict.

    Returns
    -------
    dict with keys:
        candidate_id  – str
        score         – float [0, 1]  (final composite score)
        components    – dict of individual component scores
        ai_skill_count– int
        reasoning     – str (one-line human-readable justification)
    """
    skills_s, ai_skill_count = _score_skills(candidate)
    title_s     = _score_title(candidate)
    exp_s       = _score_experience(candidate)
    avail_s     = _score_availability(candidate)
    edu_s       = _score_education(candidate)
    beh_mod     = _behavioral_modifier(candidate)

    weighted = (
        skills_s   * WEIGHTS["skills"]
        + title_s  * WEIGHTS["title"]
        + exp_s    * WEIGHTS["experience"]
        + avail_s  * WEIGHTS["availability"]
        + edu_s    * WEIGHTS["education"]
    )
    final_score = _clamp(weighted * beh_mod)

    profile = candidate.get("profile", {})
    title   = profile.get("current_title", "Unknown")
    yoe     = float(profile.get("years_of_experience", 0))
    sig     = candidate.get("redrob_signals", {})
    rr      = sig.get("recruiter_response_rate", 0.0)

    reasoning = (
        f"{title} with {yoe:.1f} yrs; "
        f"{ai_skill_count} AI core skills; "
        f"response rate {rr:.2f}."
    )

    return {
        "candidate_id":   candidate.get("candidate_id", ""),
        "score":          round(final_score, 6),
        "components": {
            "skills":              round(skills_s, 4),
            "title":               round(title_s, 4),
            "experience":          round(exp_s, 4),
            "availability":        round(avail_s, 4),
            "education":           round(edu_s, 4),
            "behavioral_modifier": round(beh_mod, 4),
        },
        "ai_skill_count": ai_skill_count,
        "reasoning":      reasoning,
        # extra fields used by app.py display
        "current_title":  profile.get("current_title", ""),
        "years_of_experience": yoe,
        "location":       profile.get("location", ""),
        "open_to_work":   sig.get("open_to_work_flag", False),
    }


def score_from_json_bytes(raw: bytes | str) -> list[dict]:
    """
    Parse JSON or JSONL bytes/str and return scored results sorted by score desc.
    Accepts:
      - A JSON array  ([{...}, {...}])
      - JSONL         (one JSON object per line)
    """
    if isinstance(raw, bytes):
        text = raw.decode("utf-8", errors="replace")
    else:
        text = raw

    text = text.strip()
    candidates: list[dict] = []

    if text.startswith("["):
        # JSON array
        try:
            candidates = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON array: {e}") from e
    else:
        # JSONL
        for line in text.splitlines():
            line = line.strip()
            if line:
                try:
                    candidates.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # skip bad lines

    if not candidates:
        raise ValueError("No valid candidate records found in uploaded file.")

    results = [score_candidate(c) for c in candidates]
    results.sort(key=lambda r: (-r["score"], r["candidate_id"]))
    return results
