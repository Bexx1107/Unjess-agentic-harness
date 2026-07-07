"""File scanner — walks the workspace respecting ignore patterns."""

import logging
from pathlib import Path
from typing import Generator

from unjess.workspace import load_ignore_patterns, should_ignore

logger = logging.getLogger(__name__)

# File extensions we consider "code"
_CODE_EXTENSIONS: set[str] = {
    # Python
    ".py", ".pyx", ".pyi",
    # JavaScript / TypeScript
    ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
    # Web
    ".html", ".css", ".scss", ".sass", ".less", ".vue", ".svelte",
    # Systems
    ".c", ".h", ".cpp", ".hpp", ".cc", ".hh", ".rs", ".go",
    # JVM
    ".java", ".kt", ".kts", ".scala", ".groovy",
    # .NET
    ".cs", ".fs", ".vb",
    # Ruby / PHP / Perl
    ".rb", ".php", ".pl", ".pm",
    # Shell / Config
    ".sh", ".bash", ".zsh", ".fish", ".ps1",
    # Data / Config
    ".json", ".yaml", ".yml", ".toml", ".xml", ".ini", ".cfg",
    # Docs
    ".md", ".rst", ".txt",
    # Other
    ".sql", ".graphql", ".proto", ".swift", ".dart", ".lua",
    ".r", ".R", ".jl", ".ex", ".exs", ".erl", ".hrl", ".zig",
}

# Max file size to index (skip very large files)
_MAX_FILE_SIZE = 500_000  # 500KB


class FileScanner:
    """Walks the workspace and yields code files, respecting ignore patterns.

    Args:
        workspace: Project root directory.
    """

    def __init__(self, workspace: Path) -> None:
        self._workspace = workspace
        self._ignore_patterns = load_ignore_patterns(workspace)

    def scan(self) -> Generator[Path, None, None]:
        """Yield all code files in the workspace.

        Skips hidden files/dirs, ignored paths, binary files,
        and files over the size limit.
        """
        yield from self._walk(self._workspace)

    def _walk(self, directory: Path) -> Generator[Path, None, None]:
        """Recursively walk a directory."""
        try:
            children = sorted(directory.iterdir())
        except PermissionError:
            return

        for child in children:
            # Skip hidden
            if child.name.startswith("."):
                continue

            # Check ignore patterns
            try:
                rel = child.relative_to(self._workspace)
            except ValueError:
                continue

            if should_ignore(rel, self._ignore_patterns):
                continue

            if child.is_dir():
                yield from self._walk(child)

            elif child.is_file():
                # Check extension
                if child.suffix.lower() not in _CODE_EXTENSIONS:
                    continue

                # Check size
                try:
                    if child.stat().st_size > _MAX_FILE_SIZE:
                        continue
                except OSError:
                    continue

                yield child

    def count_files(self) -> int:
        """Count the total number of indexable files."""
        return sum(1 for _ in self.scan())

    def get_file_stats(self) -> dict[str, int]:
        """Get file counts grouped by extension."""
        stats: dict[str, int] = {}
        for path in self.scan():
            ext = path.suffix.lower()
            stats[ext] = stats.get(ext, 0) + 1
        return dict(sorted(stats.items(), key=lambda x: x[1], reverse=True))
