---
title: Redrob AI Candidate Ranker
emoji: 🤖
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# 🤖 Redrob AI Candidate Ranker

**India Runs Data & AI Challenge 2026** — Intelligent Candidate Discovery & Ranking System

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://python.org)
[![Gradio](https://img.shields.io/badge/Gradio-4.44+-orange?logo=gradio)](https://gradio.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🎯 What This Does

Given a pool of candidate profiles (JSON/JSONL format), this system **scores and ranks the top 100 most suitable candidates** for an AI/ML Engineer role using a deterministic, interpretable, rule-based scoring engine.

- ✅ **100,000 candidates scored in ~90 seconds** on a standard CPU
- ✅ **No GPU required** — pure Python, stdlib only for the core ranker
- ✅ **No external API calls** during ranking — fully offline
- ✅ **Explainable output** — every candidate gets a one-line reasoning string

---

## 🏗️ Architecture

```
scorer.py        ← shared scoring engine (pure Python, no deps)
    ├── rank.py  ← CLI: streams candidates.jsonl → submission.csv
    └── app.py   ← Gradio web UI for Hugging Face Space
```

---

## 📐 Scoring System

### 5-Component Weighted Score

| Component | Weight | Key Signals |
|---|---|---|
| 🧠 AI/ML Core Skills | **40%** | 100+ term taxonomy, proficiency, endorsements, assessment scores |
| 💼 Title & Career | **25%** | Title tier lookup, ML keywords in job descriptions |
| 📅 Experience | **15%** | Optimal 3–8 yrs; soft decay outside |
| 📍 Availability | **10%** | Open-to-work, notice period, relocation, work mode |
| 🎓 Education | **10%** | Institution tier × degree level × field relevance |

### Behavioral Modifier (×0.80 – ×1.20)

Platform signals that multiplicatively adjust the composite score:
- Profile completeness score
- Recruiter response rate  
- GitHub activity score
- Interview completion rate
- Last-active recency
- Verified email / phone / LinkedIn

### Anti-Keyword-Stuffing

Skills with `endorsements < 2 AND duration_months < 4` receive a **0.45× quality penalty**, preventing candidates who pad their profile with AI buzzwords from outranking genuine practitioners.

---

## 🚀 Quick Start

### Run Locally (CLI)

```bash
# Install (no extra deps for CLI)
git clone https://github.com/Nithish34/ai-recruiter-
cd ai-recruiter-

# Generate submission.csv from full dataset
python rank.py \
  --candidates "./[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl" \
  --out submission.csv

# Validate submission
python "[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/validate_submission.py" submission.csv
```

### Run the Gradio App Locally

```bash
pip install gradio>=4.44.0
python app.py
# Open http://localhost:7860
```

### One-liner (competition reproduce command)

```bash
python rank.py --candidates ./[PUB]\ India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl --out ./submission.csv
```

---

## 📁 File Structure

```
.
├── rank.py                    # CLI ranking script
├── scorer.py                  # Shared scoring engine
├── app.py                     # Gradio HF Space UI
├── requirements.txt           # gradio>=4.44.0
├── submission_metadata.yaml   # Hackathon submission metadata
└── [PUB] India_runs_data_and_ai_challenge/
    └── India_runs_data_and_ai_challenge/
        ├── candidates.jsonl         # Full dataset (100K candidates, ~487MB)
        ├── sample_candidates.json   # Small sample for demos
        ├── candidate_schema.json    # JSON schema reference
        ├── sample_submission.csv    # Example output format
        └── validate_submission.py   # Official validation script
```

---

## 📊 Output Format

`submission.csv` — exactly 100 rows, header + data:

```csv
candidate_id,rank,score,reasoning
CAND_0001234,1,0.9217,"ML Engineer with 5.2 yrs; 11 AI core skills; response rate 0.82."
CAND_0005678,2,0.9105,"Data Scientist with 4.8 yrs; 9 AI core skills; response rate 0.75."
...
```

---

## 🛡️ Constraints Met

| Constraint | Status |
|---|---|
| Runtime ≤ 5 min on CPU | ✅ ~90 seconds for 100K |
| RAM ≤ 16 GB | ✅ Peak ~800 MB (heap of ≤1000 items) |
| No GPU inference | ✅ Pure Python |
| No network during ranking | ✅ Fully offline |
| Exactly 100 rows output | ✅ |
| Score non-increasing by rank | ✅ |
| Valid candidate_id pattern | ✅ |

---

## 📝 License

MIT — see [LICENSE](LICENSE)
