"""
Core logic for the Resume Retrieval System.

Given a list of candidate profiles and a job description, this module:
1. Builds a searchable text representation for each candidate.
2. Optionally applies hard filters (e.g. minimum experience, required skills).
3. Ranks candidates against the job description using TF-IDF + cosine similarity.
4. Returns the top-N ranked candidates as a list of dicts (ready for a table).
"""

import json
import io
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# 1. Loading candidate data (JSON / JSONL / CSV)
# ---------------------------------------------------------------------------

def load_candidates(file_path: str) -> pd.DataFrame:
    """Load candidate profiles from a JSON, JSONL, or CSV file into a DataFrame."""
    if file_path.endswith(".jsonl"):
        records = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        df = pd.DataFrame(records)

    elif file_path.endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Support either a top-level list, or {"candidates": [...]}
        if isinstance(data, dict):
            data = data.get("candidates", list(data.values()))
        df = pd.DataFrame(data)

    elif file_path.endswith(".csv"):
        df = pd.read_csv(file_path)

    else:
        raise ValueError("Unsupported file type. Please upload .json, .jsonl, or .csv")

    return _normalize_columns(df)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Make sure expected columns exist, filling sensible defaults if missing."""
    expected = ["name", "title", "skills", "experience_years", "summary", "location"]
    for col in expected:
        if col not in df.columns:
            df[col] = ""
    # 'skills' may arrive as a list, a comma-separated string, or missing
    df["skills"] = df["skills"].apply(_skills_to_str)
    return df


def _skills_to_str(value) -> str:
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    if pd.isna(value):
        return ""
    return str(value)


# ---------------------------------------------------------------------------
# 2. Building the searchable text blob per candidate
# ---------------------------------------------------------------------------

def build_candidate_text(row: pd.Series) -> str:
    parts = [
        str(row.get("title", "")),
        str(row.get("skills", "")),
        str(row.get("summary", "")),
        str(row.get("experience_years", "")),
    ]
    return " ".join(p for p in parts if p and p != "nan")


# ---------------------------------------------------------------------------
# 3. Optional hard filters
# ---------------------------------------------------------------------------

def apply_filters(df: pd.DataFrame, min_experience: float = 0, required_skills: str = "") -> pd.DataFrame:
    filtered = df.copy()

    if min_experience and min_experience > 0:
        exp = pd.to_numeric(filtered["experience_years"], errors="coerce").fillna(0)
        filtered = filtered[exp >= min_experience]

    if required_skills.strip():
        required = [s.strip().lower() for s in re.split(r"[,;]", required_skills) if s.strip()]
        if required:
            mask = filtered["skills"].str.lower().apply(
                lambda s: all(req in s for req in required)
            )
            filtered = filtered[mask]

    return filtered


# ---------------------------------------------------------------------------
# 4. Ranking via TF-IDF + cosine similarity
# ---------------------------------------------------------------------------

def rank_candidates(df: pd.DataFrame, job_description: str, top_k: int = 10) -> pd.DataFrame:
    if df.empty:
        return df.assign(match_score=[])

    corpus_texts = df.apply(build_candidate_text, axis=1).tolist()
    documents = corpus_texts + [job_description]

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(documents)

    job_vec = tfidf_matrix[-1]
    candidate_vecs = tfidf_matrix[:-1]

    scores = cosine_similarity(candidate_vecs, job_vec).flatten()

    result = df.copy()
    result["match_score"] = (scores * 100).round(1)
    result = result.sort_values("match_score", ascending=False).head(top_k)
    return result


# ---------------------------------------------------------------------------
# 5. End-to-end pipeline
# ---------------------------------------------------------------------------

def retrieve(file_path: str, job_description: str, min_experience: float = 0,
             required_skills: str = "", top_k: int = 10) -> pd.DataFrame:
    # Check if the file matches our rich candidate JSONL structure
    import json
    try:
        # Check if first line of jsonl or top-level element contains our schema key
        is_rich_schema = False
        if file_path.endswith(".jsonl"):
            with open(file_path, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
                if first_line.startswith("{"):
                    data = json.loads(first_line)
                    if "candidate_id" in data and "redrob_signals" in data:
                        is_rich_schema = True
        elif file_path.endswith(".json"):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    first = data[0]
                    if isinstance(first, dict) and "candidate_id" in first and "redrob_signals" in first:
                        is_rich_schema = True
                elif isinstance(data, dict):
                    # check candidates list
                    cand_list = data.get("candidates")
                    if isinstance(cand_list, list) and len(cand_list) > 0:
                        first = cand_list[0]
                        if isinstance(first, dict) and "candidate_id" in first and "redrob_signals" in first:
                            is_rich_schema = True

        if is_rich_schema:
            from app.main import process_single_candidate
            from app.config.settings import load_config
            from app.parsers.jd_parser import parse_jd
            from app.ranking.ranker import rank_candidates
            from app.parsers.candidate_parser import stream_candidates

            config = load_config()
            jd = parse_jd()

            # Optional: Override JD with custom JD if provided in UI (simple integration)
            if job_description and job_description.strip():
                # We can update Nice-To-Haves or required domains if found in text
                pass

            passed_candidates = []
            
            # Load candidates
            if file_path.endswith(".jsonl"):
                cands = stream_candidates(file_path)
            else:
                with open(file_path, "r", encoding="utf-8") as f:
                    cands_raw = json.load(f)
                    if isinstance(cands_raw, dict):
                        cands_raw = cands_raw.get("candidates", list(cands_raw.values()))
                    cands = cands_raw

            for cand_raw in cands:
                scored = process_single_candidate(cand_raw, jd, config)
                if scored:
                    passed_candidates.append(scored)

            top_results = rank_candidates(passed_candidates)
            top_results = top_results[:top_k]

            records = []
            for entry in top_results:
                cand = entry["candidate"]
                profile = cand.get("profile", {})
                skills_list = [s.get("name", "") for s in cand.get("skills", [])]
                skills_str = ", ".join(filter(None, skills_list))
                scaled_score = round(entry["final_score"] / 100.0, 4)

                records.append({
                    "name": profile.get("anonymized_name", ""),
                    "title": profile.get("current_title", ""),
                    "experience_years": profile.get("years_of_experience", 0.0),
                    "skills": skills_str,
                    "location": profile.get("location", ""),
                    "match_score": scaled_score
                })
            return pd.DataFrame(records)

    except Exception as e:
        # Fallback on any error
        import logging
        logging.getLogger("retrieval").warning("Error running modular pipeline, falling back: %s", e)
        pass

    # Simple TF-IDF fallback for legacy files
    df = load_candidates(file_path)
    df = apply_filters(df, min_experience, required_skills)
    ranked = rank_candidates(df, job_description, top_k)

    display_cols = ["name", "title", "experience_years", "skills", "location", "match_score"]
    display_cols = [c for c in display_cols if c in ranked.columns]
    return ranked[display_cols].reset_index(drop=True)

