from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

PROJECT_ROOT = Path(__file__).parent.parent
GITIGNORE_PATH = PROJECT_ROOT / ".gitignore"

ADDITIONAL_TREE_EXCLUDES: frozenset[str] = frozenset(
    {
        ".git",
        ".cache",
        ".ruff_cache",
        ".mypy_cache",
        ".pytest_cache",
        ".hypothesis",
        ".nox",
        ".tox",
    }
)


def read_gitignore_patterns() -> Iterator[str]:
    if not GITIGNORE_PATH.exists():
        return

    content = GITIGNORE_PATH.read_text(encoding="utf-8")

    for raw_line in content.splitlines():
        processed_line = raw_line.strip()

        if not processed_line or processed_line.startswith("#"):
            continue

        if processed_line.startswith("!"):
            continue

        processed_line = processed_line.removeprefix("/")
        processed_line = processed_line.removesuffix("/")

        yield processed_line


def expand_pattern(pattern: str) -> list[str]:
    if "[" in pattern and "]" in pattern and "*.py[cod]" in pattern:
        return ["*.pyc", "*.pyo", "*.pyd"]

    return [pattern]


def get_tree_ignore_patterns() -> str:
    patterns: set[str] = set(ADDITIONAL_TREE_EXCLUDES)

    for raw_pattern in read_gitignore_patterns():
        for expanded in expand_pattern(raw_pattern):
            patterns.add(expanded)

    return "|".join(sorted(patterns))


def get_cache_patterns() -> set[str]:
    cache_specific = {
        "__pycache__",
        "*.pyc",
        "*.pyo",
        "*.pyd",
        ".cache",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".nox",
        ".tox",
        "htmlcov",
        "coverage.xml",
        ".coverage",
        ".dmypy.json",
        "*.egg-info",
    }

    return cache_specific


if __name__ == "__main__":
    from rich.console import Console

    console = Console()

    console.print("\n[yellow]Tree Ignore Patterns:[/yellow]")
    console.print(get_tree_ignore_patterns())

    console.print("\n[yellow]Cache Patterns:[/yellow]")
    console.print(get_cache_patterns())
    console.print()
