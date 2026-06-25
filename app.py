"""
Resume Retrieval System — Gradio app.

Upload a list of candidate profiles (JSON, JSONL, or CSV) and paste a job
description. The app filters, retrieves, and ranks candidates by relevance,
showing the top results in a table.
"""

import gradio as gr
import pandas as pd

from retrieval import retrieve


def run_search(file, job_description, min_experience, required_skills, top_k):
    if file is None:
        return pd.DataFrame(), "Please upload a candidate file (.json, .jsonl, or .csv)."
    if not job_description or not job_description.strip():
        return pd.DataFrame(), "Please paste a job description."

    try:
        results = retrieve(
            file_path=file.name,
            job_description=job_description,
            min_experience=min_experience or 0,
            required_skills=required_skills or "",
            top_k=int(top_k),
        )
    except Exception as e:
        return pd.DataFrame(), f"Error: {e}"

    if results.empty:
        return results, "No candidates matched your filters. Try loosening them."

    return results, f"Found {len(results)} matching candidate(s)."


with gr.Blocks(title="Resume Retrieval System") as demo:
    gr.Markdown(
        """
        # 📄 Resume Retrieval System
        Upload candidate profiles and paste a job description.
        The system filters, retrieves, and ranks the best-matching candidates.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                label="Candidate profiles (.json, .jsonl, or .csv)",
                file_types=[".json", ".jsonl", ".csv"],
            )
            job_description = gr.Textbox(
                label="Job Description",
                placeholder="Paste the job description here...",
                lines=8,
            )
            with gr.Accordion("Optional filters", open=False):
                min_experience = gr.Number(label="Minimum years of experience", value=0)
                required_skills = gr.Textbox(
                    label="Required skills (comma-separated)",
                    placeholder="e.g. Python, AWS",
                )
                top_k = gr.Slider(label="Number of results", minimum=1, maximum=50, value=10, step=1)

            search_btn = gr.Button("🔍 Search Candidates", variant="primary")

        with gr.Column(scale=2):
            status = gr.Markdown()
            results_table = gr.Dataframe(label="Top Matching Candidates", wrap=True)

    search_btn.click(
        fn=run_search,
        inputs=[file_input, job_description, min_experience, required_skills, top_k],
        outputs=[results_table, status],
    )

    gr.Examples(
        examples=[["sample_candidates.json",
                    "Looking for a Machine Learning Engineer with Python, NLP, and "
                    "experience deploying retrieval systems on AWS.", 2, "Python", 5]],
        inputs=[file_input, job_description, min_experience, required_skills, top_k],
    )

if __name__ == "__main__":
    demo.launch()
