#!/usr/bin/env python3
"""
Redrob AI Candidate Ranker — CLI Entry Point
============================================
Streams candidates.jsonl line-by-line (memory-efficient heap),
scores every candidate with scorer.py, and writes top-100 to submission.csv.

Usage
-----
    python rank.py
    python rank.py --candidates path/to/candidates.jsonl --out submission.csv
    python rank.py --top 100 --quiet
"""

import argparse
import csv
import heapq
import json
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Import shared scoring engine
# ---------------------------------------------------------------------------
from scorer import (
    DEFAULT_SCORING_DATE,
    parse_scoring_date,
    score_candidate,
    validate_candidate,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
DEFAULT_CANDIDATES = (
    _HERE
    / "[PUB] India_runs_data_and_ai_challenge"
    / "India_runs_data_and_ai_challenge"
    / "candidates.jsonl"
)
DEFAULT_OUTPUT = _HERE / "submission.csv"
DEFAULT_TOP = 100


# ---------------------------------------------------------------------------
# Core ranking function
# ---------------------------------------------------------------------------

def rank_candidates(
    candidates_path: Path,
    top_n: int = 100,
    verbose: bool = True,
    scoring_date: str | None = None,
    require_exact_top: bool = True,
) -> list[dict]:
    """
    Stream candidates.jsonl one line at a time, score every candidate,
    and return the top_n results sorted by score descending.

    Uses a min-heap of size top_n so RAM stays well under 1 GB even for
    the full 100 K-candidate file.
    """
    if top_n <= 0:
        raise ValueError("top_n must be greater than zero.")
    if candidates_path.suffix.lower() != ".jsonl":
        raise ValueError("CLI ranking accepts JSONL input only.")

    effective_date = parse_scoring_date(scoring_date)
    heap: list[tuple] = []   # (score, negative numeric ID, result_dict)
    seen_ids: set[str] = set()
    count = 0
    t0 = time.time()

    with open(candidates_path, "r", encoding="utf-8") as fh:
        for line_number, raw_line in enumerate(fh, 1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                candidate = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {exc.msg}"
                ) from exc

            candidate = validate_candidate(candidate, f"line {line_number}")
            candidate_id = candidate["candidate_id"]
            if candidate_id in seen_ids:
                raise ValueError(
                    f"Duplicate candidate_id {candidate_id} on line {line_number}."
                )
            seen_ids.add(candidate_id)

            try:
                result = score_candidate(candidate, scoring_date=effective_date)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Candidate on line {line_number} cannot be scored: {exc}"
                ) from exc
            count += 1

            numeric_id = int(result["candidate_id"].removeprefix("CAND_"))
            entry = (result["score"], -numeric_id, result)

            if len(heap) < top_n:
                heapq.heappush(heap, entry)
            elif entry[:2] > heap[0][:2]:
                heapq.heapreplace(heap, entry)

            if verbose and count % 10_000 == 0:
                elapsed = time.time() - t0
                rate = count / elapsed if elapsed else 0
                print(
                    f"  [{elapsed:5.1f}s]  Scored {count:,} candidates  "
                    f"({rate:,.0f}/s)  heap_min={heap[0][0]:.4f}",
                    flush=True,
                )

    elapsed = time.time() - t0
    if verbose:
        print(
            f"\n  [OK] Finished: {count:,} candidates scored in {elapsed:.1f}s "
            f"(0 lines skipped)",
            flush=True,
        )

    if count == 0:
        raise ValueError("No candidate records were found in the JSONL file.")
    if require_exact_top and count < top_n:
        raise ValueError(
            f"Input contains {count} candidates, fewer than the requested top {top_n}."
        )

    # Sort: primary = score descending, tie-break = candidate_id ascending
    top = sorted(heap, key=lambda e: (-e[0], e[2]["candidate_id"]))
    return [e[2] for e in top[:top_n]]


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def write_submission(results: list[dict], output_path: Path) -> None:
    """Write ranked results to a competition-valid CSV file."""
    with open(output_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, result in enumerate(results, start=1):
            writer.writerow([
                result["candidate_id"],
                rank,
                f"{result['score']:.6f}",
                result["reasoning"],
            ])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Redrob AI Candidate Ranker — rank top-N candidates from candidates.jsonl",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--candidates", "-c",
        type=Path,
        default=DEFAULT_CANDIDATES,
        help="Path to candidates.jsonl",
    )
    parser.add_argument(
        "--out", "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output CSV path",
    )
    parser.add_argument(
        "--top", "-n",
        type=int,
        default=DEFAULT_TOP,
        help="Number of top candidates to include in output",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--scoring-date",
        default=DEFAULT_SCORING_DATE.isoformat(),
        help="Fixed date used for activity-recency scoring (YYYY-MM-DD)",
    )
    args = parser.parse_args()

    # Validate inputs
    if not args.candidates.exists():
        print(f"ERROR: candidates file not found: {args.candidates}", file=sys.stderr)
        print(
            "  Hint: run from the repo root, or pass --candidates <path>",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.candidates.suffix.lower() != ".jsonl":
        parser.error("--candidates must point to a .jsonl file")
    if args.top <= 0:
        parser.error("--top must be greater than zero")

    print("=" * 60)
    print("  Redrob AI Candidate Ranker")
    print("=" * 60)
    print(f"  Candidates : {args.candidates}")
    print(f"  Output     : {args.out}")
    print(f"  Top-N      : {args.top}")
    print(f"  Score date : {args.scoring_date}")
    print()

    try:
        results = rank_candidates(
            args.candidates,
            top_n=args.top,
            verbose=not args.quiet,
            scoring_date=args.scoring_date,
            require_exact_top=True,
        )
        write_submission(results, args.out)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    # ── Pretty-print top 5 ──────────────────────────────────────────────────
    print()
    print("  Top 5 candidates:")
    print("  " + "-" * 56)
    for i, r in enumerate(results[:5], 1):
        print(f"  {i:2d}. {r['candidate_id']}  score={r['score']:.4f}")
        print(f"       {r['reasoning']}")
    print()
    print(f"  [DONE] Submission written to: {args.out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
