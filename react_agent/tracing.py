"""Append-only JSONL storage used for experiment traces and resume state."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any


_REQUIRED_FIELDS = {
    "sample_id",
    "run_id",
    "status",
    "answer",
    "evidence",
    "num_llm_calls",
    "num_tool_calls",
    "elapsed_seconds",
    "config",
    "trace",
}


def write_jsonl_record(path: Any, record: dict) -> None:
    """Append one UTF-8 JSON record without modifying the input object."""

    line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    with Path(path).open("a", encoding="utf-8") as file:
        file.write(line + "\n")


def load_completed_ids(path: Any, run_id: str) -> set[str]:
    """Return sample IDs already recorded for ``run_id``.

    Malformed, incomplete, and duplicate records are diagnosed and skipped so
    a partially written final line does not hide earlier completed work.
    """

    trace_path = Path(path)
    if not trace_path.exists():
        return set()

    completed: set[str] = set()
    seen: set[tuple[str, str]] = set()

    with trace_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                warnings.warn(
                    f"Skipping malformed JSONL record on line {line_number}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue

            if not isinstance(record, dict) or not _REQUIRED_FIELDS <= record.keys():
                warnings.warn(
                    f"Skipping incomplete JSONL record on line {line_number}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue

            pair = (record["run_id"], record["sample_id"])
            if pair in seen:
                warnings.warn(
                    f"Skipping duplicate JSONL record on line {line_number}: {pair!r}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue

            seen.add(pair)
            if record["run_id"] == run_id:
                completed.add(record["sample_id"])

    return completed
