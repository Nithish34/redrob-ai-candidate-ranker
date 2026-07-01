import copy
import json
import types
import unittest
from pathlib import Path

from scorer import (
    DEFAULT_SCORING_DATE,
    score_candidate,
    score_from_json_bytes,
)


SAMPLE_FILE = Path(__file__).with_name("sample_candidates.json")


class ScorerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample_candidates = json.loads(SAMPLE_FILE.read_text(encoding="utf-8"))

    def test_sample_scores_are_sorted_and_repeatable(self):
        raw = SAMPLE_FILE.read_bytes()
        first = score_from_json_bytes(raw, scoring_date=DEFAULT_SCORING_DATE)
        second = score_from_json_bytes(raw, scoring_date=DEFAULT_SCORING_DATE)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 10)
        self.assertTrue(
            all(
                left["score"] >= right["score"]
                for left, right in zip(first, first[1:])
            )
        )

    def test_malformed_jsonl_reports_its_line(self):
        valid = json.dumps(self.sample_candidates[0])
        raw = f"{valid}\nnot-json\n".encode()

        with self.assertRaisesRegex(ValueError, "line 2"):
            score_from_json_bytes(raw)

    def test_duplicate_candidate_ids_are_rejected(self):
        duplicate = [
            self.sample_candidates[0],
            copy.deepcopy(self.sample_candidates[0]),
        ]

        with self.assertRaisesRegex(ValueError, "Duplicate candidate_id"):
            score_from_json_bytes(json.dumps(duplicate))

    def test_json_array_must_contain_candidate_objects(self):
        with self.assertRaisesRegex(ValueError, "non-object items found at position"):
            score_from_json_bytes('["not a candidate"]')

    def test_invalid_candidate_id_is_rejected(self):
        candidate = copy.deepcopy(self.sample_candidates[0])
        candidate["candidate_id"] = ""

        with self.assertRaisesRegex(ValueError, "invalid candidate_id"):
            score_candidate(candidate)

    def test_future_activity_date_does_not_receive_recency_bonus(self):
        future = copy.deepcopy(self.sample_candidates[0])
        neutral = copy.deepcopy(self.sample_candidates[0])
        future["redrob_signals"]["last_active_date"] = "2030-01-01"
        neutral["redrob_signals"]["last_active_date"] = "2025-07-01"

        future_score = score_candidate(future, scoring_date="2026-07-01")
        neutral_score = score_candidate(neutral, scoring_date="2026-07-01")

        self.assertEqual(
            future_score["components"]["behavioral_modifier"],
            neutral_score["components"]["behavioral_modifier"],
        )


    # ------------------------------------------------------------------
    # Phase 5 — sample file availability
    # ------------------------------------------------------------------

    def test_sample_file_exists_with_10_records(self):
        """sample_candidates.json must be committed and contain exactly 10 records."""
        self.assertTrue(
            SAMPLE_FILE.exists(),
            "sample_candidates.json must be committed to the repository",
        )
        candidates = json.loads(SAMPLE_FILE.read_text(encoding="utf-8"))
        self.assertEqual(
            len(candidates),
            10,
            f"sample_candidates.json must contain exactly 10 records, got {len(candidates)}",
        )

    # ------------------------------------------------------------------
    # Phase 5 — upload handler path-type compatibility
    # ------------------------------------------------------------------

    def test_upload_handler_accepts_string_path(self):
        """process_upload must work when Gradio passes the path as a plain str."""
        from app import process_upload  # imported here to keep scorer tests isolated

        table_update, csv_path, summary = process_upload(str(SAMPLE_FILE), 10)
        self.assertIsNotNone(csv_path, "CSV path must not be None on success")
        self.assertIn("Ranked", summary, "Summary must confirm ranking succeeded")

    def test_upload_handler_accepts_file_object(self):
        """process_upload must work when Gradio passes an object with a .name attribute."""
        from app import process_upload

        fake_file = types.SimpleNamespace(name=str(SAMPLE_FILE))
        table_update, csv_path, summary = process_upload(fake_file, 10)
        self.assertIsNotNone(csv_path, "CSV path must not be None on success")
        self.assertIn("Ranked", summary, "Summary must confirm ranking succeeded")


if __name__ == "__main__":
    unittest.main()
