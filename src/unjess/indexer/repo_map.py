"""Repo map — compressed structural map of the codebase for context injection."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from unjess.context_manager import count_tokens
from unjess.indexer.scanner import FileScanner
from unjess.indexer.symbols import FileSymbols, Symbol, SymbolExtractor

logger = logging.getLogger(__name__)


@dataclass
class RepoMapEntry:
    """A single file entry in the repo map."""

    relative_path: str
    language: str
    line_count: int
    classes: list[str] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)
    methods: dict[str, list[str]] = field(default_factory=dict)  # class -> methods


class RepoMap:
    """Builds and formats a structural map of the codebase.

    The repo map is a compressed representation of every code file's
    structure — classes, functions, methods — designed to fit within
    a token budget for context injection.

    Args:
        workspace: Project root directory.
    """

    def __init__(self, workspace: Path) -> None:
        self._workspace = workspace
        self._scanner = FileScanner(workspace)
        self._extractor = SymbolExtractor(workspace)
        self._entries: dict[str, RepoMapEntry] = {}
        self._indexed = False

    @property
    def is_indexed(self) -> bool:
        """Whether the codebase has been indexed."""
        return self._indexed

    @property
    def file_count(self) -> int:
        """Number of indexed files."""
        return len(self._entries)

    # ----- Indexing -----

    def index(self) -> int:
        """Scan the workspace and build the repo map.

        Returns:
            Number of files indexed.
        """
        self._entries.clear()
        count = 0

        for file_path in self._scanner.scan():
            file_symbols = self._extractor.extract(file_path)

            if not file_symbols.symbols:
                # Still add files without symbols for structure
                entry = RepoMapEntry(
                    relative_path=file_symbols.relative_path or file_path.name,
                    language=file_symbols.language,
                    line_count=file_symbols.line_count,
                )
            else:
                entry = self._build_entry(file_symbols)

            self._entries[entry.relative_path] = entry
            count += 1

        self._indexed = True
        logger.info("Indexed %d files", count)
        return count

    def refresh(self, changed_files: list[Path]) -> int:
        """Incrementally re-index only changed files.

        Args:
            changed_files: List of files that changed.

        Returns:
            Number of files re-indexed.
        """
        count = 0
        for file_path in changed_files:
            if not file_path.exists():
                # File was deleted
                try:
                    rel = str(file_path.relative_to(self._workspace)).replace("\\", "/")
                    self._entries.pop(rel, None)
                except ValueError:
                    pass
                continue

            file_symbols = self._extractor.extract(file_path)
            entry = self._build_entry(file_symbols)
            self._entries[entry.relative_path] = entry
            count += 1

        return count

    # ----- Formatting -----

    def format(self, max_tokens: int = 2000, model: str = "") -> str:
        """Format the repo map to fit within a token budget.

        Args:
            max_tokens: Maximum tokens for the output.
            model: Model name (for accurate token counting).

        Returns:
            Formatted repo map string.
        """
        if not self._entries:
            return ""

        # Group by directory
        dirs: dict[str, list[RepoMapEntry]] = {}
        for entry in sorted(self._entries.values(), key=lambda e: e.relative_path):
            dir_name = str(Path(entry.relative_path).parent)
            if dir_name == ".":
                dir_name = "/"
            if dir_name not in dirs:
                dirs[dir_name] = []
            dirs[dir_name].append(entry)

        # Build the map, checking token budget
        lines: list[str] = ["# Repo Map", ""]
        current_tokens = count_tokens("\n".join(lines), model)

        for dir_name, entries in sorted(dirs.items()):
            dir_section = self._format_directory(dir_name, entries)
            section_tokens = count_tokens(dir_section, model)

            if current_tokens + section_tokens > max_tokens:
                # Try a compressed version
                compressed = self._format_directory_compressed(dir_name, entries)
                compressed_tokens = count_tokens(compressed, model)

                if current_tokens + compressed_tokens > max_tokens:
                    lines.append(f"\n... ({len(dirs) - len(lines) + 2} more directories)")
                    break

                lines.append(compressed)
                current_tokens += compressed_tokens
            else:
                lines.append(dir_section)
                current_tokens += section_tokens

        return "\n".join(lines)

    def _format_directory(self, dir_name: str, entries: list[RepoMapEntry]) -> str:
        """Format a full directory section."""
        lines = [f"## {dir_name}/"]

        for entry in entries:
            filename = Path(entry.relative_path).name
            lines.append(f"- {filename} ({entry.line_count}L)")

            for cls in entry.classes:
                lines.append(f"  - class {cls}")
                methods = entry.methods.get(cls, [])
                for method in methods[:10]:  # cap methods per class
                    lines.append(f"    - {method}")

            for func in entry.functions[:15]:  # cap functions per file
                lines.append(f"  - {func}")

        return "\n".join(lines)

    def _format_directory_compressed(self, dir_name: str, entries: list[RepoMapEntry]) -> str:
        """Format a compressed directory section (files only, no symbols)."""
        files = [Path(e.relative_path).name for e in entries]
        return f"## {dir_name}/ — {', '.join(files[:10])}"

    # ----- Relevance search -----

    def find_relevant(self, query: str, top_n: int = 10) -> list[str]:
        """Find files most relevant to a query by keyword matching.

        Args:
            query: Search query (user message or symbol name).
            top_n: Max results.

        Returns:
            List of relative file paths, ordered by relevance.
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())

        scored: list[tuple[int, str]] = []
        for rel_path, entry in self._entries.items():
            score = 0

            # Path match
            path_lower = rel_path.lower()
            for word in query_words:
                if word in path_lower:
                    score += 20

            # Symbol match
            all_names = entry.classes + entry.functions
            for methods in entry.methods.values():
                all_names.extend(methods)

            for name in all_names:
                name_lower = name.lower()
                for word in query_words:
                    if word in name_lower:
                        score += 10

            if score > 0:
                scored.append((score, rel_path))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [path for _, path in scored[:top_n]]

    # ----- Internal -----

    def _build_entry(self, file_symbols: FileSymbols) -> RepoMapEntry:
        """Build a RepoMapEntry from extracted FileSymbols."""
        entry = RepoMapEntry(
            relative_path=file_symbols.relative_path or str(file_symbols.path.name),
            language=file_symbols.language,
            line_count=file_symbols.line_count,
        )

        for sym in file_symbols.symbols:
            if sym.kind == "class":
                entry.classes.append(sym.name)
            elif sym.kind == "function":
                sig = sym.signature or sym.name
                entry.functions.append(sig)
            elif sym.kind == "method" and sym.parent:
                if sym.parent not in entry.methods:
                    entry.methods[sym.parent] = []
                sig = sym.signature or sym.name
                entry.methods[sym.parent].append(sig)

        return entry
