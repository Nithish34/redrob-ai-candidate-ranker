import csv
import unittest
from pathlib import Path


SUBMISSION_FILE = Path(__file__).with_name("submission.csv")


class LivePreviewDataTests(unittest.TestCase):
    def test_prebuilt_submission_is_ready_for_live_preview(self):
        self.assertTrue(
            SUBMISSION_FILE.exists(),
            "submission.csv must be committed for the Live Preview tab",
        )

        with SUBMISSION_FILE.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            self.assertEqual(
                reader.fieldnames,
                ["candidate_id", "rank", "score", "reasoning"],
            )
            rows = list(reader)

        self.assertEqual(len(rows), 100)
        self.assertEqual([int(row["rank"]) for row in rows], list(range(1, 101)))
        self.assertEqual(len({row["candidate_id"] for row in rows}), 100)
        self.assertTrue(all(row["candidate_id"] for row in rows))
        self.assertTrue(all(row["reasoning"] for row in rows))

        scores = [float(row["score"]) for row in rows]
        self.assertTrue(
            all(left >= right for left, right in zip(scores, scores[1:])),
            "submission.csv scores must be non-increasing",
        )


if __name__ == "__main__":
    unittest.main()
