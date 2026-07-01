"""
build_sample.py
===============
Extracts 100 diverse candidate records from the full 100K JSONL dataset
and writes them to sample_candidates.json.

Selection strategy
------------------
Scores all 100K candidates with the same scorer used in production, then
picks 100 records at evenly-spaced positions across the sorted ranking
(positions 0, ~1000, ~2000 ... ~99000). This guarantees the sample covers
the full score range from the strongest AI/ML candidates to the weakest.

Usage
-----
    python build_sample.py              # uses default paths
    python build_sample.py --input path/to/candidates.jsonl --count 100
"""

import argparse
import json
import time
from pathlib import Path

HERE = Path(__file__).parent

DEFAULT_JSONL = (
    HERE
    / "[PUB] India_runs_data_and_ai_challenge"
    / "India_runs_data_and_ai_challenge"
    / "candidates.jsonl"
)
DEFAULT_OUT = HERE / "sample_candidates.json"
DEFAULT_COUNT = 100


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input",  "-i", type=Path, default=DEFAULT_JSONL,
                        help="Path to candidates.jsonl (default: competition dataset)")
    parser.add_argument("--out",    "-o", type=Path, default=DEFAULT_OUT,
                        help="Output JSON file (default: sample_candidates.json)")
    parser.add_argument("--count",  "-n", type=int,  default=DEFAULT_COUNT,
                        help="Number of candidates to include (default: 100)")
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"ERROR: Input file not found: {args.input}")
    if args.count < 1:
        raise SystemExit("ERROR: --count must be at least 1")

    # ── Step 1: Stream-read all records ────────────────────────────────────
    print(f"Reading {args.input} ...")
    t0 = time.time()
    all_records: list[dict] = []
    with args.input.open("r", encoding="utf-8") as fh:
        for lineno, raw_line in enumerate(fh, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                all_records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"  WARNING: skipping malformed line {lineno}: {exc}")
    print(f"  {len(all_records):,} records loaded in {time.time() - t0:.1f}s")

    # ── Step 2: Score every record ─────────────────────────────────────────
    print("Scoring all candidates (this may take ~90 s for 100K) ...")
    from scorer import score_from_json_bytes, DEFAULT_SCORING_DATE
    t1 = time.time()
    raw_bytes = json.dumps(all_records).encode("utf-8")
    results = score_from_json_bytes(raw_bytes, scoring_date=DEFAULT_SCORING_DATE)
    print(f"  {len(results):,} candidates scored in {time.time() - t1:.1f}s")
    print(f"  Score range: {results[-1]['score']:.4f} - {results[0]['score']:.4f}")

    # ── Step 3: Pick evenly-spaced positions across the full ranking ───────
    n = min(args.count, len(results))
    if n < args.count:
        print(f"  WARNING: only {len(results)} candidates available, using all")
    if n == 1:
        indices = [0]
    else:
        indices = [int(i * (len(results) - 1) / (n - 1)) for i in range(n)]

    selected_ids = {results[i]["candidate_id"] for i in indices}
    print(f"\nSelected {len(selected_ids)} candidates at evenly-spaced score positions")
    print(f"  Top-3 IDs   : {[results[i]['candidate_id'] for i in indices[:3]]}")
    print(f"  Bottom-3 IDs: {[results[i]['candidate_id'] for i in indices[-3:]]}")

    # ── Step 4: Build output (preserve original full record detail) ────────
    id_to_record = {r["candidate_id"]: r for r in all_records}
    selected = [id_to_record[cid] for cid in selected_ids if cid in id_to_record]

    # ── Step 5: Write ──────────────────────────────────────────────────────
    args.out.write_text(
        json.dumps(selected, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    size_kb = args.out.stat().st_size / 1024
    print(f"\nWrote {len(selected)} records to {args.out}  ({size_kb:.1f} KB)")

    # ── Step 6: Round-trip verification ───────────────────────────────────
    print("Verifying ...")
    check = score_from_json_bytes(args.out.read_bytes(), scoring_date=DEFAULT_SCORING_DATE)
    assert len(check) == len(selected), \
        f"Round-trip count mismatch: expected {len(selected)}, got {len(check)}"
    scores = [r["score"] for r in check]
    assert all(a >= b for a, b in zip(scores, scores[1:])), \
        "Scores not sorted descending after round-trip"

    print(f"\n[PASS] {len(check)} candidates verified, scores sorted descending")
    print(f"       Score range: {scores[-1]:.4f} - {scores[0]:.4f}")
    print()
    print("  Rank  Candidate ID    Score   Role")
    print("  ----  ------------    -----   ----")
    for rank, r in enumerate(check, 1):
        if rank <= 5 or rank > len(check) - 5:
            print(f"  {rank:4d}  {r['candidate_id']}  {r['score']:.4f}  {r['reasoning'][:55]}")
        elif rank == 6:
            print(f"  ...   ({len(check) - 10} more candidates)")


if __name__ == "__main__":
    main()
