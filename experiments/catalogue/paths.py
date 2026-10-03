"""Shared filesystem roots for catalogue modules."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def repo_relative(path: Path) -> str:
    """Path relative to the repository root; outside paths are unchanged."""
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)
