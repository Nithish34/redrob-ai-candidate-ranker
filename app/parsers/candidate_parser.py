"""
Candidate Parser — Stream-parse candidates.jsonl into dicts.

Handles large files efficiently by reading one line at a time.
"""

import json
from typing import Iterator

from app.utils.logger import get_logger

log = get_logger("candidate_parser")


def stream_candidates(file_path: str) -> Iterator[dict]:
    """Yield one candidate dict at a time from a JSONL file.

    Args:
        file_path: Path to candidates.jsonl.

    Yields:
        Parsed candidate dicts.
    """
    count = 0
    errors = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                candidate = json.loads(line)
                count += 1
                yield candidate
            except json.JSONDecodeError as e:
                errors += 1
                log.warning("Line %d: JSON parse error — %s", line_no, e)
                continue

    log.info("Parsed %d candidates (%d errors)", count, errors)


def load_all_candidates(file_path: str) -> list[dict]:
    """Load all candidates into memory. Use only when needed."""
    return list(stream_candidates(file_path))
