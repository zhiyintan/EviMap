#!/usr/bin/env python3
"""Run scripts/check_highlight_promise.js on a data directory.

    python scripts/check_highlight_promise.py <data_dir> [--max-unlocated 0.05] [--sample N]

Exits non-zero when a document listed under a topic cannot show that topic's
phrase. Needs node (PATH or ~/.local/node-*-linux-x64/bin).
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def node_binary() -> str:
    found = shutil.which("node")
    if found:
        return found
    for d in sorted(glob.glob(os.path.expanduser("~/.local/node-*-linux-x64/bin"))):
        if (Path(d) / "node").exists():
            return str(Path(d) / "node")
    raise SystemExit("node not found: install it or put ~/.local/node-*-linux-x64/bin on PATH")


def main(argv: list[str]) -> int:
    if not argv:
        raise SystemExit(__doc__)
    cmd = [node_binary(), str(ROOT / "scripts" / "check_highlight_promise.js"), *argv]
    return subprocess.run(cmd, cwd=str(ROOT)).returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
