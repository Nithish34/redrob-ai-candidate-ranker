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
from scorer import score_candidate

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
) -> list[dict]:
    """
    Stream candidates.jsonl one line at a time, score every candidate,
    and return the top_n results sorted by score descending.

    Uses a min-heap of size top_n so RAM stays well under 1 GB even for
    the full 100 K-candidate file.
    """
    heap: list[tuple] = []   # (score, candidate_id, result_dict)
    count = 0
    skipped = 0
    t0 = time.time()

    with open(candidates_path, "r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue

            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            result = score_candidate(candidate)
            count += 1

            entry = (result["score"], result["candidate_id"], result)

            if len(heap) < top_n:
                heapq.heappush(heap, entry)
            elif entry[0] > heap[0][0]:
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
            f"({skipped} lines skipped)",
            flush=True,
        )

    # Sort: primary = score descending, tie-break = candidate_id ascending
    top = sorted(heap, key=lambda e: (-e[0], e[1]))
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
    args = parser.parse_args()

    # Validate inputs
    if not args.candidates.exists():
        print(f"ERROR: candidates file not found: {args.candidates}", file=sys.stderr)
        print(
            "  Hint: run from the repo root, or pass --candidates <path>",
            file=sys.stderr,
        )
        sys.exit(1)

    print("=" * 60)
    print("  Redrob AI Candidate Ranker")
    print("=" * 60)
    print(f"  Candidates : {args.candidates}")
    print(f"  Output     : {args.out}")
    print(f"  Top-N      : {args.top}")
    print()

    results = rank_candidates(args.candidates, top_n=args.top, verbose=not args.quiet)

    write_submission(results, args.out)

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
