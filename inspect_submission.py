#!/usr/bin/env python3
"""Validate a generated competition submission and fail on any contract error."""

import argparse
import csv
import re
import sys
from pathlib import Path


EXPECTED_COLUMNS = ["candidate_id", "rank", "score", "reasoning"]
CANDIDATE_ID_PATTERN = re.compile(r"^CAND_\d{7}$")


def validate_submission(path: Path, expected_rows: int = 100) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        return [f"File not found: {path}"]

    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames != EXPECTED_COLUMNS:
                errors.append(
                    f"Expected columns {EXPECTED_COLUMNS}, got {reader.fieldnames or []}."
                )
            rows = list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        return [f"Could not read submission: {exc}"]

    if len(rows) != expected_rows:
        errors.append(f"Expected {expected_rows} rows, got {len(rows)}.")

    candidate_ids: list[str] = []
    ranks: list[int] = []
    scores: list[float] = []

    for line_number, row in enumerate(rows, 2):
        candidate_id = row.get("candidate_id", "")
        candidate_ids.append(candidate_id)
        if not CANDIDATE_ID_PATTERN.fullmatch(candidate_id):
            errors.append(f"Line {line_number}: invalid candidate_id {candidate_id!r}.")

        try:
            ranks.append(int(row.get("rank", "")))
        except (TypeError, ValueError):
            errors.append(f"Line {line_number}: rank must be an integer.")

        try:
            score = float(row.get("score", ""))
            scores.append(score)
            if not 0.0 <= score <= 1.0:
                errors.append(f"Line {line_number}: score must be between 0 and 1.")
        except (TypeError, ValueError):
            errors.append(f"Line {line_number}: score must be numeric.")

        if not row.get("reasoning", "").strip():
            errors.append(f"Line {line_number}: reasoning is required.")

    if len(candidate_ids) != len(set(candidate_ids)):
        errors.append("candidate_id values must be unique.")
    if ranks != list(range(1, len(rows) + 1)):
        errors.append("Ranks must be sequential and ordered from 1.")
    if len(scores) == len(rows) and any(
        left < right for left, right in zip(scores, scores[1:])
    ):
        errors.append("Scores must be non-increasing.")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", nargs="?", type=Path, default=Path("submission.csv"))
    parser.add_argument("--expected-rows", type=int, default=100)
    args = parser.parse_args()

    errors = validate_submission(args.submission, args.expected_rows)
    if errors:
        print("Submission validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        sys.exit(1)

    print(
        f"Submission is valid: {args.expected_rows} ranked candidates in "
        f"{args.submission}"
    )


if __name__ == "__main__":
    main()
