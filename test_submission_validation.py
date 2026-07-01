import csv
import tempfile
import unittest
from pathlib import Path

from inspect_submission import validate_submission


class SubmissionValidationTests(unittest.TestCase):
    def write_submission(self, directory: str, rows) -> Path:
        path = Path(directory) / "submission.csv"
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["candidate_id", "rank", "score", "reasoning"],
            )
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_valid_submission_passes(self):
        rows = [
            {
                "candidate_id": "CAND_0000001",
                "rank": 1,
                "score": "0.900000",
                "reasoning": "Strong match.",
            },
            {
                "candidate_id": "CAND_0000002",
                "rank": 2,
                "score": "0.800000",
                "reasoning": "Good match.",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_submission(directory, rows)
            self.assertEqual(validate_submission(path, expected_rows=2), [])

    def test_invalid_submission_reports_contract_errors(self):
        rows = [
            {
                "candidate_id": "bad-id",
                "rank": 2,
                "score": "1.5",
                "reasoning": "",
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_submission(directory, rows)
            errors = validate_submission(path, expected_rows=1)

        self.assertTrue(any("invalid candidate_id" in error for error in errors))
        self.assertTrue(any("Ranks must be sequential" in error for error in errors))
        self.assertTrue(any("between 0 and 1" in error for error in errors))
        self.assertTrue(any("reasoning is required" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
