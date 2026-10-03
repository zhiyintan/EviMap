"""Shared JSONL helpers for catalogue modules.

Every stage module used to carry its own copy of load_jsonl/write_jsonl; they
are consolidated here. The two historical variants are one switch: strict (the
default, a missing file raises) and missing_ok=True (summary steps tolerate
optional inputs).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List


def load_jsonl(path: Path, *, missing_ok: bool = False) -> List[dict]:
    """Read a JSONL file into a list of dicts, skipping blank lines."""
    if missing_ok and not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8")
            if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
