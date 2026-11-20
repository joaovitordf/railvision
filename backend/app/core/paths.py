"""Repository root and filesystem path resolution helpers."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"


def resolve_repo_path(raw: str | Path) -> Path:
    """Resolve a path relative to the repository root, supporting ../ legacy conventions."""
    raw_str = str(raw).strip()
    candidate = Path(raw_str)
    if candidate.is_absolute():
        return candidate.resolve()
    if raw_str.startswith("../") or raw_str.startswith("..\\"):
        return (BACKEND_ROOT / candidate).resolve()
    return (REPO_ROOT / candidate).resolve()
