import copy
import json
import tempfile
import unittest
from pathlib import Path

from rank import rank_candidates


SAMPLE_FILE = Path(__file__).with_name("sample_candidates.json")


class RankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample = json.loads(SAMPLE_FILE.read_text(encoding="utf-8"))[0]

    def write_jsonl(self, directory: str, records) -> Path:
        path = Path(directory) / "candidates.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for record in records:
                if isinstance(record, str):
                    fh.write(record + "\n")
                else:
                    fh.write(json.dumps(record) + "\n")
        return path

    def test_equal_scores_use_candidate_id_tie_break_at_cutoff(self):
        larger_id = copy.deepcopy(self.sample)
        smaller_id = copy.deepcopy(self.sample)
        larger_id["candidate_id"] = "CAND_0000002"
        smaller_id["candidate_id"] = "CAND_0000001"

        with tempfile.TemporaryDirectory() as directory:
            path = self.write_jsonl(directory, [larger_id, smaller_id])
            results = rank_candidates(path, top_n=1, verbose=False)

        self.assertEqual(results[0]["candidate_id"], "CAND_0000001")

    def test_malformed_jsonl_fails_with_line_number(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_jsonl(directory, [self.sample, "not-json"])
            with self.assertRaisesRegex(ValueError, "line 2"):
                rank_candidates(path, top_n=1, verbose=False)

    def test_duplicate_candidate_ids_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_jsonl(directory, [self.sample, self.sample])
            with self.assertRaisesRegex(ValueError, "Duplicate candidate_id"):
                rank_candidates(path, top_n=1, verbose=False)

    def test_non_jsonl_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidates.json"
            path.write_text(json.dumps([self.sample]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "JSONL input only"):
                rank_candidates(path, top_n=1, verbose=False)

    def test_invalid_top_n_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_jsonl(directory, [self.sample])
            with self.assertRaisesRegex(ValueError, "greater than zero"):
                rank_candidates(path, top_n=0, verbose=False)

    def test_exact_top_requirement_is_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_jsonl(directory, [self.sample])
            with self.assertRaisesRegex(ValueError, "fewer than"):
                rank_candidates(path, top_n=2, verbose=False)


if __name__ == "__main__":
    unittest.main()
