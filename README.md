---
title: Resume Retrieval System
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
python_version: "3.10"
app_file: app.py
pinned: false
---

# Resume Retrieval System

Upload a list of candidate profiles (JSON, JSONL, or CSV) and paste a job
description. The app filters, retrieves, and ranks the candidates, showing
the top results in a table.

## How it works

1. **Load** — candidate profiles are parsed from JSON / JSONL / CSV into a
   table with fields like `name`, `title`, `skills`, `experience_years`,
   `summary`, `location`.
2. **Filter** (optional) — hard filters such as minimum years of experience
   or required skills are applied first.
3. **Rank** — each candidate's text (title + skills + summary) is vectorized
   with TF-IDF and compared against the job description using cosine
   similarity to produce a `match_score`.
4. **Display** — the top-N candidates are shown in a sortable table.

## File format

Each candidate record should look like:

```json
{
  "name": "Asha Patel",
  "title": "Backend Engineer",
  "skills": ["Python", "Django", "PostgreSQL", "AWS"],
  "experience_years": 5,
  "summary": "Built scalable REST APIs and microservices for fintech products.",
  "location": "Mumbai"
}
```

- **JSON**: a list of such objects (or `{"candidates": [...]}`)
- **JSONL**: one such object per line
- **CSV**: same fields as columns (skills as a comma-separated string)

## Running locally

```bash
pip install -r requirements.txt
python app.py
```

## Swapping in semantic embeddings (optional upgrade)

The default ranker uses TF-IDF, which is fast and needs no model download.
For better semantic matching, swap `rank_candidates` in `retrieval.py` to use
`sentence-transformers` (e.g. `all-MiniLM-L6-v2`) embeddings + cosine
similarity instead of TF-IDF — add `sentence-transformers` to
`requirements.txt` if you do this.
