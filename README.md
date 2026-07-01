---
title: Redrob AI Candidate Ranker
emoji: 🤖
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
python_version: "3.11"
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

Given a pool of candidate profiles, this system **scores and ranks the top 100 most suitable candidates** for an AI/ML Engineer role using a deterministic, interpretable, rule-based scoring engine.

- ✅ **100,000 JSONL candidates scored in ~90 seconds** by the streaming CLI on a standard CPU
- ✅ **No GPU required** — pure Python, stdlib only for the core ranker
- ✅ **No external API calls** during ranking — fully offline
- ✅ **Explainable output** — every candidate gets a one-line reasoning string
- ✅ **Reproducible recency scoring** — fixed scoring date: `2026-07-01`

---

## 🏗️ Architecture

```
scorer.py        ← shared scoring engine (pure Python, no deps)
    ├── rank.py  ← CLI: streams candidates.jsonl → submission.csv
    └── app.py   ← Gradio web UI for Hugging Face Space
```

The CLI accepts strict JSONL for the full dataset. The web UI accepts JSON or
JSONL files up to 50 MB and is intended for smaller interactive checks. The
Live Preview reads the committed top-100 `submission.csv` produced by the full
100K run rather than loading the private dataset into the Space.

### Dataset availability

The 100K competition dataset is intentionally excluded from Git and Hugging
Face because it is large and may be access-controlled. Place the provided
`candidates.jsonl` at:

```text
[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl
```

The repository includes `sample_candidates.json` for tests and
`submission.csv` for the prebuilt Live Preview.

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

### Behavioral Modifier (×0.85 – ×1.15)

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
git clone https://github.com/Nithish34/redrob-ai-candidate-ranker
cd redrob-ai-candidate-ranker

# Generate submission.csv from full dataset
python rank.py \
  --candidates "./[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl" \
  --out submission.csv \
  --scoring-date 2026-07-01

# Validate the output contract
python inspect_submission.py submission.csv
```

### Run the Gradio App Locally

```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:7860
```

### Run regression tests

```bash
python -m unittest discover -v
python inspect_submission.py submission.csv
```

### One-liner (competition reproduce command)

```bash
python rank.py --candidates ./[PUB]\ India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl --out ./submission.csv --scoring-date 2026-07-01
```

---

## 📁 File Structure

```
.
├── rank.py                    # CLI ranking script
├── scorer.py                  # Shared scoring engine
├── app.py                     # Gradio HF Space UI
├── submission.csv             # Prebuilt top-100 result from the 100K run
├── sample_candidates.json     # Small fallback demo dataset
├── requirements.txt           # pinned Space dependencies
├── inspect_submission.py      # strict output contract validator
├── test_*.py                  # standard-library regression suite
├── submission_metadata.yaml   # Hackathon submission metadata
└── [PUB] India_runs_data_and_ai_challenge/
    └── India_runs_data_and_ai_challenge/
        ├── candidates.jsonl         # Full dataset (100K candidates, ~487MB)
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
| Runtime ≤ 5 min on CPU | ✅ ~90 seconds for 100K (CLI) |
| Web upload latency | ✅ < 5 seconds for up to 1,000 records |
| RAM ≤ 16 GB | ✅ Streaming CLI retains only the requested top-N heap |
| No GPU inference | ✅ Pure Python |
| No network during ranking | ✅ Fully offline |
| Exactly 100 rows output | ✅ |
| Score non-increasing by rank | ✅ |
| Valid candidate_id pattern | ✅ |

---

## 📝 License

MIT — see [LICENSE](LICENSE)
