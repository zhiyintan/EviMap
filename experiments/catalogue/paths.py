"""Shared filesystem roots for catalogue modules."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def repo_relative(path: Path) -> str:
    """Render a path relative to the repository root so that recorded outputs
    do not embed machine-specific prefixes. Paths outside the repository are
    returned unchanged."""
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)
