"""Helpers for resolving repository-root-relative paths."""

from pathlib import Path
from typing import Union


PathLike = Union[str, Path]

# This file lives in <repo>/utils/paths.py, so its parent.parent is the repo root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def repo_path(*parts: PathLike) -> Path:
    """Build an absolute path rooted at the repository root."""
    return PROJECT_ROOT.joinpath(*parts)


def repo_str(*parts: PathLike) -> str:
    """Return a repository-root-relative path as an absolute string."""
    return str(repo_path(*parts))


def resolve_repo_path(path_value: PathLike) -> str:
    """Resolve an absolute path as-is, otherwise anchor it to the repo root."""
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str(repo_path(path))
