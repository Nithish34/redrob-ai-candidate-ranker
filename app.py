"""
Redrob AI Candidate Ranker — Gradio Web UI
==========================================
Deployed on Hugging Face Spaces.
Accepts a JSON / JSONL candidate file, ranks candidates,
and returns a downloadable submission.csv.

Three tabs:
  1. Upload & Rank   – upload any sample file, get ranked CSV
  2. Live Preview    – inspect the prebuilt top-100 result from the 100K run
  3. Methodology     – scoring formula documentation
"""

from __future__ import annotations

import csv
import io
import json
import os
import tempfile
import time
from pathlib import Path

import gradio as gr

from rank import rank_candidates
from scorer import DEFAULT_SCORING_DATE, score_from_json_bytes, WEIGHTS

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
SAMPLE_FILE = _HERE / "sample_candidates.json"
PREBUILT_SUBMISSION = _HERE / "submission.csv"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_JSON_RECORDS = 10_000
GENERATED_FILE_TTL_SECONDS = 60 * 60
GENERATED_FILE_DIR = Path(tempfile.gettempdir()) / "redrob-candidate-ranker"

# ---------------------------------------------------------------------------
# Helper: build results table + CSV bytes
# ---------------------------------------------------------------------------

def _prune_generated_files() -> None:
    """Remove expired generated downloads so a long-running Space stays bounded."""
    GENERATED_FILE_DIR.mkdir(parents=True, exist_ok=True)
    cutoff = time.time() - GENERATED_FILE_TTL_SECONDS
    for path in GENERATED_FILE_DIR.glob("submission-*.csv"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            pass


def _build_outputs(results: list[dict], top_n: int = 100):
    """
    Given scored & sorted results list, return:
      - table_data  : list[list] for gr.Dataframe
      - csv_path    : str  (temp file path for gr.File download)
      - summary_md  : str  (markdown summary)
    """
    top = results[:top_n]

    # ── Table ────────────────────────────────────────────────────────────────
    headers = ["Rank", "Candidate ID", "Score", "Title", "Exp (yrs)",
               "Skills", "Open to Work", "Reasoning"]
    rows = []
    for rank, r in enumerate(top, 1):
        rows.append([
            rank,
            r["candidate_id"],
            f"{r['score']:.4f}",
            r.get("current_title", ""),
            f"{r.get('years_of_experience', 0):.1f}",
            r.get("ai_skill_count", 0),
            "Yes" if r.get("open_to_work") else "No",
            r["reasoning"],
        ])

    # ── CSV ──────────────────────────────────────────────────────────────────
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["candidate_id", "rank", "score", "reasoning"])
    for rank, r in enumerate(top, 1):
        writer.writerow([r["candidate_id"], rank, f"{r['score']:.4f}", r["reasoning"]])

    _prune_generated_files()
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        prefix="submission-",
        suffix=".csv",
        delete=False,
        encoding="utf-8",
        newline="",
        dir=GENERATED_FILE_DIR,
    )
    tmp.write(buf.getvalue())
    tmp.close()

    # ── Summary ──────────────────────────────────────────────────────────────
    total = len(results)
    top1 = top[0] if top else {}
    summary = (
        f"### ✅ Ranked {total} candidates — showing top {len(top)}\n\n"
        f"**🥇 #1:** `{top1.get('candidate_id','—')}` — "
        f"score **{top1.get('score', 0):.4f}** — {top1.get('reasoning','')}\n\n"
        f"Score range: **{top[-1]['score']:.4f}** – **{top[0]['score']:.4f}**\n\n"
        f"Scoring date: **{DEFAULT_SCORING_DATE.isoformat()}**"
        if top else "No results."
    )

    return rows, headers, tmp.name, summary


# ---------------------------------------------------------------------------
# Tab 1 & 2 processing functions
# ---------------------------------------------------------------------------

def process_upload(file_obj, top_n_slider):
    if file_obj is None:
        return (
            gr.update(value=[]),
            None,
            "Please upload a `.json` or `.jsonl` file first.",
        )
    try:
        top_n = int(top_n_slider)
        if top_n <= 0:
            raise ValueError("Top-N must be greater than zero.")

        if isinstance(file_obj, (str, Path)):
            uploaded_path = Path(file_obj)
        elif hasattr(file_obj, "name"):
            uploaded_path = Path(file_obj.name)
        elif isinstance(file_obj, bytes):
            if len(file_obj) > MAX_UPLOAD_BYTES:
                raise ValueError("Upload exceeds the 50 MB web limit.")
            results = score_from_json_bytes(
                file_obj,
                scoring_date=DEFAULT_SCORING_DATE,
                max_records=MAX_JSON_RECORDS,
            )
            rows, headers, csv_path, summary = _build_outputs(results, top_n)
            return gr.update(value=rows), csv_path, summary
        else:
            raise ValueError("Unsupported uploaded file value.")

        if not uploaded_path.exists():
            raise ValueError("Uploaded file is no longer available.")
        if uploaded_path.stat().st_size > MAX_UPLOAD_BYTES:
            raise ValueError(
                "Upload exceeds the 50 MB web limit. "
                "Use rank.py for the full 100K JSONL dataset."
            )

        suffix = uploaded_path.suffix.lower()
        if suffix == ".jsonl":
            results = rank_candidates(
                uploaded_path,
                top_n=top_n,
                verbose=False,
                scoring_date=DEFAULT_SCORING_DATE.isoformat(),
                require_exact_top=False,
            )
        elif suffix == ".json":
            results = score_from_json_bytes(
                uploaded_path.read_bytes(),
                scoring_date=DEFAULT_SCORING_DATE,
                max_records=MAX_JSON_RECORDS,
            )
        else:
            raise ValueError("Only .json and .jsonl files are supported.")

        rows, headers, csv_path, summary = _build_outputs(results, top_n)
        return gr.update(value=rows), csv_path, summary
    except Exception as exc:
        return (
            gr.update(value=[]),
            None,
            f"Error: {exc}",
        )


def process_prebuilt_preview(top_n_slider):
    """Load the precomputed top-100 ranking produced from the 100K dataset."""
    try:
        top_n = int(top_n_slider)

        if PREBUILT_SUBMISSION.exists():
            with PREBUILT_SUBMISSION.open("r", encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                required = {"candidate_id", "rank", "score", "reasoning"}
                missing = required.difference(reader.fieldnames or [])
                if missing:
                    raise ValueError(
                        "Prebuilt submission is missing columns: "
                        + ", ".join(sorted(missing))
                    )

                results = [
                    {
                        "candidate_id": row["candidate_id"],
                        "rank": int(row["rank"]),
                        "score": float(row["score"]),
                        "reasoning": row["reasoning"],
                    }
                    for row in reader
                ]

            results.sort(key=lambda row: row["rank"])
            top = results[:top_n]
            rows = [
                [
                    row["rank"],
                    row["candidate_id"],
                    f"{row['score']:.6f}",
                    row["reasoning"],
                ]
                for row in top
            ]
            top1 = top[0] if top else {}
            summary = (
                "### ✅ Loaded prebuilt 100K-candidate ranking\n\n"
                f"Showing **{len(top)} of {len(results)}** retained finalists from "
                "the full 100,000-candidate run.\n\n"
                f"**🥇 #1:** `{top1.get('candidate_id', '—')}` — "
                f"score **{top1.get('score', 0):.6f}**"
                if top
                else "No prebuilt ranking rows were found."
            )
            return gr.update(value=rows), str(PREBUILT_SUBMISSION), summary

        if SAMPLE_FILE.exists():
            raw = SAMPLE_FILE.read_bytes()
            results = score_from_json_bytes(
                raw,
                scoring_date=DEFAULT_SCORING_DATE,
                max_records=MAX_JSON_RECORDS,
            )
            top = results[:top_n]
            rows = [
                [
                    rank,
                    row["candidate_id"],
                    f"{row['score']:.6f}",
                    row["reasoning"],
                ]
                for rank, row in enumerate(top, 1)
            ]
            _, _, csv_path, _ = _build_outputs(results, top_n)
            summary = (
                "### ⚠️ Loaded fallback demo data\n\n"
                f"`submission.csv` was not found, so this preview ranked "
                f"the bundled sample and is showing **{len(top)}** candidates."
            )
            return gr.update(value=rows), csv_path, summary

        return (
            gr.update(value=[]),
            None,
            "Prebuilt `submission.csv` and fallback sample data were not found.",
        )
    except Exception as exc:
        return (
            gr.update(value=[]),
            None,
            f"Error: {exc}",
        )


# ---------------------------------------------------------------------------
# Methodology markdown
# ---------------------------------------------------------------------------

METHODOLOGY_MD = f"""
## 🤖 Scoring Methodology

This ranker uses a **5-component weighted scoring system** with a multiplicative behavioral modifier.
All logic is deterministic — no ML inference, no network calls.
Activity recency is evaluated against the fixed scoring date
**{DEFAULT_SCORING_DATE.isoformat()}**.

---

### Component Weights

| Component | Weight | Description |
|---|---|---|
| 🧠 AI/ML Core Skills | **{WEIGHTS['skills']*100:.0f}%** | Matches 100+ term AI/ML taxonomy; rewards advanced proficiency & endorsements; penalises keyword stuffers |
| 💼 Title & Career | **{WEIGHTS['title']*100:.0f}%** | Tiered lookup of current + past titles; ML keywords in role descriptions |
| 📅 Experience | **{WEIGHTS['experience']*100:.0f}%** | Soft optimum 3–8 years; smooth decay outside |
| 📍 Availability | **{WEIGHTS['availability']*100:.0f}%** | Open-to-work flag, notice period, relocation, work mode |
| 🎓 Education | **{WEIGHTS['education']*100:.0f}%** | Institution tier × degree level × field relevance |

---

### Behavioral Modifier  ×(0.85 – 1.15)

Multiplicative adjustment using platform-specific signals:

- **Profile completeness score** — more complete = higher modifier
- **Recruiter response rate** — responsive candidates score higher
- **GitHub activity score** — active open-source contributors get a bonus
- **Interview completion rate** — shows reliability
- **Last-active recency** — recently active candidates get a bonus; dormant (>180 days) get a penalty
- **Verified email / phone / LinkedIn** — small trust bonuses

---

### Keyword-Stuffer Detection

Skills with **< 2 endorsements AND < 4 months duration** receive a 0.45× quality penalty,
preventing candidates who pad their profile with AI buzzwords from ranking unfairly high.

---

### Score Formula

```
weighted = Σ (component_score × weight)
final    = clamp(weighted × behavioral_modifier, 0, 1)
```

Scores are **non-increasing by rank** (enforced by sort). Tie-break: ascending `candidate_id`.

---

### Runtime

~90 seconds for **100,000 candidates** on a standard CPU with 16 GB RAM.
The streaming CLI retains only the requested top-N heap. Interactive web
uploads are capped at 50 MB; the full dataset should be processed with `rank.py`.
"""

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
/* ── Global ── */
body { font-family: 'Inter', 'Segoe UI', sans-serif; }

/* ── Header ── */
.header-block {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 40%, #4c1d95 100%);
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 8px;
    box-shadow: 0 8px 32px rgba(79, 70, 229, 0.35);
}
.header-block h1 {
    color: #ffffff !important;
    font-size: 2.2rem;
    font-weight: 800;
    margin: 0 0 6px 0;
    letter-spacing: -0.5px;
}
.header-block p {
    color: #c7d2fe;
    font-size: 1.05rem;
    margin: 0;
}

/* ── Badge row ── */
.badge-row { display: flex; gap: 10px; flex-wrap: wrap; margin: 12px 0 4px; }
.badge {
    background: rgba(99, 102, 241, 0.18);
    border: 1px solid rgba(99, 102, 241, 0.4);
    border-radius: 20px;
    padding: 4px 14px;
    color: #a5b4fc;
    font-size: 0.82rem;
    font-weight: 600;
}

/* ── Upload card ── */
.upload-card {
    border: 2px dashed rgba(99, 102, 241, 0.5);
    border-radius: 12px;
    background: rgba(30, 27, 75, 0.08);
    transition: border-color 0.3s ease;
}
.upload-card:hover { border-color: rgba(99, 102, 241, 0.9); }

/* ── Buttons ── */
.run-btn {
    background: linear-gradient(90deg, #4f46e5, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 12px 28px !important;
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.4) !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}
.run-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(79, 70, 229, 0.55) !important;
}

/* ── Summary markdown ── */
.summary-md {
    background: linear-gradient(135deg, rgba(16,185,129,0.08), rgba(79,70,229,0.08));
    border-left: 4px solid #10b981;
    border-radius: 10px;
    padding: 16px 20px;
}

/* ── Dataframe ── */
.gr-dataframe table thead tr th {
    background: linear-gradient(90deg, #1e1b4b, #312e81) !important;
    color: #e0e7ff !important;
    font-weight: 700 !important;
}

/* ── Tabs ── */
.tab-nav button.selected {
    background: linear-gradient(90deg, #4f46e5, #7c3aed) !important;
    color: white !important;
    border-radius: 8px !important;
}
"""

# ---------------------------------------------------------------------------
# Build Gradio app
# ---------------------------------------------------------------------------

with gr.Blocks(
    title="Redrob AI Candidate Ranker",
    theme=gr.themes.Soft(
        primary_hue=gr.themes.colors.indigo,
        secondary_hue=gr.themes.colors.purple,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"],
    ),
    css=CUSTOM_CSS,
) as demo:

    # ── Header ───────────────────────────────────────────────────────────────
    gr.HTML("""
    <div class="header-block">
        <h1>🤖 Redrob AI Candidate Ranker</h1>
        <p>Intelligent candidate discovery ranking system</p>
        <div class="badge-row">
            <span class="badge">⚡ 100K candidates in ~90s</span>
            <span class="badge">🧠 5-component scoring</span>
            <span class="badge">🔒 No ML inference</span>
            <span class="badge">📦 Pure Python</span>
            <span class="badge">🏆 Redrob Hackathon 2026</span>
        </div>
    </div>
    """)

    # ── Tabs ─────────────────────────────────────────────────────────────────
    with gr.Tabs() as tabs:

        # ── Tab 1: Upload & Rank ─────────────────────────────────────────────
        with gr.TabItem("📤 Upload & Rank", id="upload_tab"):
            gr.Markdown(
                "Upload your own **`.json`** (array) or **`.jsonl`** (one candidate per line) file. "
                "The ranker will score every candidate and return a competition-ready "
                "`submission.csv`. Web uploads are limited to **50 MB**; use `rank.py` "
                "for the full 100K dataset."
            )

            with gr.Row():
                with gr.Column(scale=2):
                    file_input = gr.File(
                        label="📁 Drop candidates.json / candidates.jsonl here",
                        file_types=[".json", ".jsonl"],
                        elem_classes=["upload-card"],
                    )
                    top_n_slider_1 = gr.Slider(
                        minimum=10, maximum=100, value=100, step=10,
                        label="Number of top candidates to return",
                    )
                    run_btn_1 = gr.Button(
                        "🚀 Run Ranking", variant="primary",
                        elem_classes=["run-btn"],
                    )

                with gr.Column(scale=3):
                    summary_1 = gr.Markdown(
                        value="*Upload a file and click **Run Ranking** to see results.*",
                        elem_classes=["summary-md"],
                    )
                    download_btn_1 = gr.File(
                        label="⬇️ Download submission.csv",
                        visible=True,
                        interactive=False,
                    )

            results_table_1 = gr.Dataframe(
                headers=["Rank", "Candidate ID", "Score", "Title",
                         "Exp (yrs)", "Skills", "Open to Work", "Reasoning"],
                datatype=["number", "str", "str", "str", "str", "number", "str", "str"],
                label="📊 Ranked Results",
                wrap=True,
            )

            run_btn_1.click(
                fn=process_upload,
                inputs=[file_input, top_n_slider_1],
                outputs=[results_table_1, download_btn_1, summary_1],
            )

        # ── Tab 2: Live Preview ──────────────────────────────────────────────
        with gr.TabItem("🔍 Live Preview", id="preview_tab"):
            gr.Markdown(
                "Preview the **precomputed top-100 ranking** generated from the full "
                "**100,000-candidate dataset**. The large private dataset is not loaded "
                "into the Space; this tab reads the bundled `submission.csv` result."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    top_n_slider_2 = gr.Slider(
                        minimum=5, maximum=100, value=20, step=5,
                        label="Finalists to show",
                    )
                    run_btn_2 = gr.Button(
                        "▶️ Load Prebuilt 100K Ranking", variant="primary",
                        elem_classes=["run-btn"],
                    )
                    download_btn_2 = gr.File(
                        label="⬇️ Download full top-100 submission.csv",
                        interactive=False,
                    )

                with gr.Column(scale=2):
                    summary_2 = gr.Markdown(
                        value=(
                            "*Click **Load Prebuilt 100K Ranking** to preview the "
                            "competition-ready result without rerunning the full dataset.*"
                        ),
                        elem_classes=["summary-md"],
                    )

            results_table_2 = gr.Dataframe(
                headers=["Rank", "Candidate ID", "Score", "Reasoning"],
                datatype=["number", "str", "str", "str"],
                label="📊 Prebuilt Top-100 Results",
                wrap=True,
            )

            run_btn_2.click(
                fn=process_prebuilt_preview,
                inputs=[top_n_slider_2],
                outputs=[results_table_2, download_btn_2, summary_2],
            )

        # ── Tab 3: Methodology ───────────────────────────────────────────────
        with gr.TabItem("ℹ️ Methodology", id="method_tab"):
            gr.Markdown(METHODOLOGY_MD)

            gr.HTML("""
            <div style="
                margin-top: 24px;
                background: linear-gradient(135deg, #1e1b4b, #312e81);
                border-radius: 12px;
                padding: 24px 28px;
                color: #e0e7ff;
            ">
                <h3 style="margin:0 0 12px; color:#a5b4fc;">📐 Weight Breakdown</h3>
                <div style="display:flex; flex-direction:column; gap:10px;">
                    <div>
                        <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                            <span>🧠 AI/ML Core Skills</span><span style="color:#818cf8;font-weight:700;">40%</span>
                        </div>
                        <div style="background:#1e1b4b;border-radius:6px;height:10px;">
                            <div style="background:linear-gradient(90deg,#4f46e5,#7c3aed);width:40%;height:10px;border-radius:6px;"></div>
                        </div>
                    </div>
                    <div>
                        <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                            <span>💼 Title &amp; Career</span><span style="color:#818cf8;font-weight:700;">25%</span>
                        </div>
                        <div style="background:#1e1b4b;border-radius:6px;height:10px;">
                            <div style="background:linear-gradient(90deg,#4f46e5,#7c3aed);width:25%;height:10px;border-radius:6px;"></div>
                        </div>
                    </div>
                    <div>
                        <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                            <span>📅 Experience</span><span style="color:#818cf8;font-weight:700;">15%</span>
                        </div>
                        <div style="background:#1e1b4b;border-radius:6px;height:10px;">
                            <div style="background:linear-gradient(90deg,#4f46e5,#7c3aed);width:15%;height:10px;border-radius:6px;"></div>
                        </div>
                    </div>
                    <div>
                        <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                            <span>📍 Availability</span><span style="color:#818cf8;font-weight:700;">10%</span>
                        </div>
                        <div style="background:#1e1b4b;border-radius:6px;height:10px;">
                            <div style="background:linear-gradient(90deg,#4f46e5,#7c3aed);width:10%;height:10px;border-radius:6px;"></div>
                        </div>
                    </div>
                    <div>
                        <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                            <span>🎓 Education</span><span style="color:#818cf8;font-weight:700;">10%</span>
                        </div>
                        <div style="background:#1e1b4b;border-radius:6px;height:10px;">
                            <div style="background:linear-gradient(90deg,#4f46e5,#7c3aed);width:10%;height:10px;border-radius:6px;"></div>
                        </div>
                    </div>
                </div>
            </div>
            """)

    # ── Footer ───────────────────────────────────────────────────────────────
    gr.HTML("""
    <div style="text-align:center; margin-top:20px; padding: 12px;
                color:#6b7280; font-size:0.85rem; border-top: 1px solid #e5e7eb;">
        Built for the <strong>Redrob India Runs Data &amp; AI Challenge 2026</strong> ·
        Pure Python · No GPU · No external APIs
    </div>
    """)


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
