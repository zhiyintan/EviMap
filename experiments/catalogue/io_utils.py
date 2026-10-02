"""Shared JSONL helpers for catalogue modules.

以前每个 stage 模块各自复制一份 load_jsonl/write_jsonl（共 20 处），
2026-06-09 统一收到这里。两个历史变体合并为一个开关：
strict（默认，文件不存在就抛错）和 missing_ok=True（汇总类步骤容忍可选输入）。
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
