"""
Build sample_candidates.json by selecting 10 diverse candidates from the
official 50-record sample that ships with the competition dataset.

Selection strategy:
- Top 3 scorers  (high quality, definitely AI/ML)
- Middle 4       (varied roles, medium scores)
- Bottom 3       (lower scores, different backgrounds)

This gives the test suite a realistic spread across the full score range.
"""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from scorer import score_from_json_bytes, DEFAULT_SCORING_DATE

HERE = Path(__file__).parent

# ── Load official sample ────────────────────────────────────────────────────
OFFICIAL = (
    HERE
    / "[PUB] India_runs_data_and_ai_challenge"
    / "India_runs_data_and_ai_challenge"
    / "sample_candidates.json"
)

print(f"Loading {OFFICIAL} ...")
raw = OFFICIAL.read_bytes()
all_candidates = json.loads(raw)
print(f"  {len(all_candidates)} records found in official sample")

# ── Score all 50 ────────────────────────────────────────────────────────────
print("Scoring ...")
results = score_from_json_bytes(raw, scoring_date=DEFAULT_SCORING_DATE)
print(f"  {len(results)} scored")

# ── Print full distribution for visibility ──────────────────────────────────
print("\nFull score distribution:")
for i, r in enumerate(results, 1):
    print(f"  {i:2d}. {r['candidate_id']}  score={r['score']:.4f}  {r['reasoning'][:70]}")

# ── Pick 10 diverse candidates ──────────────────────────────────────────────
# Top 3 + evenly spaced 4 from middle + bottom 3
top3    = [r["candidate_id"] for r in results[:3]]
middle4 = [r["candidate_id"] for r in results[6:46:10]]   # positions 6,16,26,36
bottom3 = [r["candidate_id"] for r in results[-3:]]

selected_ids = top3 + middle4 + bottom3
print(f"\nSelected IDs ({len(selected_ids)}): {selected_ids}")

# ── Build output preserving original record order ───────────────────────────
id_to_record = {c["candidate_id"]: c for c in all_candidates}
selected = [id_to_record[cid] for cid in selected_ids if cid in id_to_record]
print(f"Records matched: {len(selected)}")

# ── Write sample_candidates.json ────────────────────────────────────────────
OUT = HERE / "sample_candidates.json"
OUT.write_text(json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nWrote {len(selected)} records to {OUT}")
print(f"File size: {OUT.stat().st_size:,} bytes")

# ── Verify it round-trips through the scorer ─────────────────────────────────
check = score_from_json_bytes(OUT.read_bytes(), scoring_date=DEFAULT_SCORING_DATE)
assert len(check) == 10, f"Expected 10 results, got {len(check)}"
scores = [r["score"] for r in check]
assert all(a >= b for a, b in zip(scores, scores[1:])), "Scores not sorted descending"
print("\n[PASS] Verification passed: 10 candidates, sorted descending")
print(f"   Score range: {scores[-1]:.4f} - {scores[0]:.4f}")
for r in check:
    print(f"   {r['candidate_id']}  {r['score']:.4f}  {r['reasoning'][:60]}")
