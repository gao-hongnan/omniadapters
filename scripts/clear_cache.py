from __future__ import annotations

import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from fnmatch import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

try:
    from .patterns import get_cache_patterns
except ImportError:
    from scripts.patterns import get_cache_patterns

if TYPE_CHECKING:
    from collections.abc import Iterator

console = Console()

BYTES_PER_KB = 1024.0

type PathSize = tuple[Path, int]


def get_dir_size(path: Path) -> int:
    total = 0
    with suppress(OSError):
        for item in path.rglob("*"):
            if item.is_file():
                with suppress(OSError):
                    total += item.stat().st_size
    return total


def matches_pattern(path: Path, patterns: set[str]) -> bool:
    name = path.name
    return any(fnmatch(name, pattern) for pattern in patterns)


def find_cache_items(root: Path, patterns: set[str]) -> Iterator[PathSize]:
    visited_dirs: set[Path] = set()

    for path in root.rglob("*"):
        if any(parent in visited_dirs for parent in path.parents):
            continue

        if matches_pattern(path, patterns):
            if path.is_dir():
                visited_dirs.add(path)
                size = get_dir_size(path)
                yield (path, size)
            elif path.is_file():
                try:
                    size = path.stat().st_size
                    yield (path, size)
                except OSError:
                    yield (path, 0)


def delete_path(item: PathSize) -> PathSize | None:
    path, size = item
    try:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
    except OSError:
        return None
    else:
        return (path, size)


def format_size(size_bytes: int) -> str:
    size = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < BYTES_PER_KB:
            return f"{size:.1f} {unit}"
        size /= BYTES_PER_KB
    return f"{size:.1f} TB"


def main() -> None:
    start_time = time.perf_counter()
    root = Path.cwd()

    console.print("\n[yellow]🧹 Pruning Python cache files and directories...[/yellow]\n")

    console.print("[dim]Scanning for cache items...[/dim]")
    cache_patterns = get_cache_patterns()
    cache_items = list(find_cache_items(root, cache_patterns))

    if not cache_items:
        console.print("[green]✅ No cache items found - already clean![/green]\n")
        return

    total_items = len(cache_items)
    total_size = sum(size for _, size in cache_items)

    console.print(f"[yellow]Found {total_items:,} items ({format_size(total_size)}) to delete[/yellow]\n")

    deleted_count = 0
    deleted_size = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Deleting...", total=total_items)

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(delete_path, item): item for item in cache_items}

            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    _, size = result
                    deleted_count += 1
                    deleted_size += size

                progress.update(task, advance=1)

    elapsed = time.perf_counter() - start_time

    console.print(
        f"\n[green]✅ Deleted {deleted_count:,} items ({format_size(deleted_size)}) in {elapsed:.2f}s[/green]\n"
    )


if __name__ == "__main__":
    main()
