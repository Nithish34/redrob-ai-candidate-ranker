"""
Redrob Intelligent Candidate Discovery & Ranking System — Main Entry Point
========================================================================
Runs the end-to-end pipeline:
1. Loads configuration & job description.
2. Streams candidate profiles from JSONL.
3. Preprocesses & filters honeypots in parallel.
4. Scores candidates through L2-L6.
5. Ranks and generates reasoning for Top 100.
6. Outputs submission.csv.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config.settings import load_config
from app.parsers.candidate_parser import stream_candidates
from app.parsers.jd_parser import parse_jd
from app.preprocess.extractor import prepare_candidate
from app.layers import (
    layer1_honeypot,
    layer2_jdfit,
    layer3_evidence,
    layer4_behavior,
    layer5_trust,
    layer6_career,
)
from app.ranking.ranker import rank_candidates
from app.reasoning.reasoning_engine import generate_reasoning
from app.utils.logger import get_logger

log = get_logger("main_pipeline")


def process_single_candidate(candidate_raw: dict, jd: dict, config: dict) -> dict | None:
    """Worker function to process and score a single candidate profile.

    Parameters
    ----------
    candidate_raw : dict
        Raw candidate dictionary from JSONL.
    jd : dict
        Parsed Job Description dictionary.
    config : dict
        Configuration weights.

    Returns
    -------
    dict | None
        Scored candidate dictionary with all layer results, or None if filtered/invalid.
    """
    try:
        # 1. Preprocess (Validate & Normalize)
        cand = prepare_candidate(candidate_raw)
        if not cand:
            return None

        # 2. Layer 1: Honeypot Filter (Gate)
        l1_res = layer1_honeypot.execute(cand, jd, config)
        if not l1_res.get("passed", False):
            return None

        # 3. Layer 2: JD Fit Engine
        l2_res = layer2_jdfit.execute(cand, jd, config)

        # 4. Layer 3: Evidence Engine
        l3_res = layer3_evidence.execute(cand, jd, config)

        # 5. Layer 4: Behavior Engine
        l4_res = layer4_behavior.execute(cand, jd, config)

        # 6. Layer 5: Trust Engine
        l5_res = layer5_trust.execute(cand, jd, config)

        # 7. Layer 6: Career Intelligence
        l6_res = layer6_career.execute(cand, jd, config)

        return {
            "candidate_id": cand["candidate_id"],
            "candidate": cand,
            "layer_results": {
                "layer1": l1_res,
                "layer2": l2_res,
                "layer3": l3_res,
                "layer4": l4_res,
                "layer5": l5_res,
                "layer6": l6_res,
            },
        }
    except Exception as e:
        # Gracefully handle single candidate errors to prevent pipeline crash
        log.warning("Error processing candidate %s: %s", candidate_raw.get("candidate_id", "Unknown"), e)
        return None


def main():
    parser = argparse.ArgumentParser(description="Redrob Candidate Discovery & Ranking Pipeline")
    parser.add_argument(
        "--candidates",
        type=str,
        required=True,
        help="Path to candidates.jsonl file",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Path to output submission.csv",
    )
    args = parser.parse_args()

    start_time = time.time()
    log.info("Starting candidate ranking pipeline...")

    # Load configuration and job description
    config = load_config()
    jd = parse_jd()
    log.info("Loaded weights config and job description details.")

    # Check candidates file
    candidates_path = args.candidates
    if not os.path.exists(candidates_path):
        log.error("Candidates file not found at: %s", candidates_path)
        sys.exit(1)

    # Process candidates in parallel batches using ProcessPoolExecutor
    num_workers = config.get("processing", {}).get("num_workers", 4)
    batch_size = config.get("processing", {}).get("batch_size", 5000)
    log.info("Processing using %d parallel workers, batch size: %d", num_workers, batch_size)

    passed_candidates = []
    rejected_count = 0
    total_count = 0

    # Stream candidates and process in batches
    batch = []
    worker_fn = partial(process_single_candidate, jd=jd, config=config)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for candidate_raw in stream_candidates(candidates_path):
            total_count += 1
            batch.append(candidate_raw)

            if len(batch) >= batch_size:
                # Process batch in parallel
                results = executor.map(worker_fn, batch)
                for res in results:
                    if res:
                        passed_candidates.append(res)
                    else:
                        rejected_count += 1
                batch = []
                log.info("Processed %d candidates...", total_count)

        # Process any remaining candidates in the last batch
        if batch:
            results = executor.map(worker_fn, batch)
            for res in results:
                if res:
                    passed_candidates.append(res)
                else:
                    rejected_count += 1
            log.info("Processed final batch. Total: %d candidates.", total_count)

    log.info(
        "Candidate filtering complete. Passed: %d, Rejected/Honeypot: %d, Total: %d",
        len(passed_candidates),
        rejected_count,
        total_count,
    )

    # 4. Final Ranking Engine
    log.info("Running ranking engine on passed candidates...")
    top_100 = rank_candidates(passed_candidates)

    if not top_100:
        log.error("No candidates passed the pipeline. Cannot generate submission.")
        sys.exit(1)

    # 5. Reasoning Engine
    log.info("Generating recruiter explanations for Top 100...")
    for entry in top_100:
        entry["reasoning"] = generate_reasoning(entry, entry["rank"])

    # 6. Write final CSV
    output_path = args.out
    log.info("Writing submission results to: %s", output_path)

    # Ensure output directory exists
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        
        for entry in top_100:
            # Scale score to 0-1 range to match sample_submission format
            scaled_score = round(entry["final_score"] / 100.0, 4)
            writer.writerow([
                entry["candidate_id"],
                entry["rank"],
                f"{scaled_score:.4f}",
                entry["reasoning"]
            ])

    duration = time.time() - start_time
    log.info("Pipeline complete! Ranked Top 100 candidates in %.2f seconds.", duration)


if __name__ == "__main__":
    main()
