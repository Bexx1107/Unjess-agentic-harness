"""@ mention system — parse and resolve @file, @web, @git, @url syntax.

Mentions let users inject context into their messages:
  @file.py       → injects file contents
  @src/           → injects directory listing
  @web "query"   → searches the web (placeholder)
  @git diff      → injects git diff output
  @url https://  → fetches URL content (placeholder)
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from unjess.git_integration import GitIntegration

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Mention:
    """A parsed @ mention."""

    raw: str  # original text matched
    kind: str  # "file", "dir", "git", "web", "url"
    target: str  # the argument after @
    resolved_content: str = ""  # resolved content to inject

    @property
    def is_resolved(self) -> bool:
        """Whether this mention was successfully resolved."""
        return bool(self.resolved_content)


# ---------------------------------------------------------------------------
# Mention patterns
# ---------------------------------------------------------------------------

# Matches: @filename.ext, @path/to/file.ext, @directory/
_FILE_PATTERN = re.compile(r'@([\w./-]+\.[\w]+)')
_DIR_PATTERN = re.compile(r'@([\w./-]+/)')
_GIT_PATTERN = re.compile(r'@git\s+(diff|status|log|branch)')
_WEB_PATTERN = re.compile(r'@web\s+"([^"]+)"')
_URL_PATTERN = re.compile(r'@url\s+(https?://\S+)')


# ---------------------------------------------------------------------------
# Mention resolver
# ---------------------------------------------------------------------------

class MentionResolver:
    """Parses and resolves @ mentions in user messages.

    Args:
        workspace: Project root directory.
        git: Git integration instance.
    """

    def __init__(self, workspace: Path, git: Optional[GitIntegration] = None) -> None:
        self._workspace = workspace
        self._git = git

    def parse(self, message: str) -> list[Mention]:
        """Parse all @ mentions from a user message.

        Args:
            message: The user's message text.

        Returns:
            List of parsed Mention objects (not yet resolved).
        """
        mentions: list[Mention] = []

        # @git mentions
        for match in _GIT_PATTERN.finditer(message):
            mentions.append(Mention(
                raw=match.group(0),
                kind="git",
                target=match.group(1),
            ))

        # @web mentions
        for match in _WEB_PATTERN.finditer(message):
            mentions.append(Mention(
                raw=match.group(0),
                kind="web",
                target=match.group(1),
            ))

        # @url mentions
        for match in _URL_PATTERN.finditer(message):
            mentions.append(Mention(
                raw=match.group(0),
                kind="url",
                target=match.group(1),
            ))

        # @directory/ mentions (must check before file to avoid overlap)
        for match in _DIR_PATTERN.finditer(message):
            # Skip if already matched by git/web/url
            if any(m.raw in match.group(0) or match.group(0) in m.raw for m in mentions):
                continue
            mentions.append(Mention(
                raw=match.group(0),
                kind="dir",
                target=match.group(1),
            ))

        # @file mentions
        for match in _FILE_PATTERN.finditer(message):
            # Skip if already matched by another pattern
            if any(m.raw in match.group(0) or match.group(0) in m.raw for m in mentions):
                continue
            mentions.append(Mention(
                raw=match.group(0),
                kind="file",
                target=match.group(1),
            ))

        return mentions

    def resolve(self, mentions: list[Mention]) -> list[Mention]:
        """Resolve all mentions by loading their content.

        Args:
            mentions: List of parsed mentions.

        Returns:
            The same list with ``resolved_content`` populated.
        """
        for mention in mentions:
            try:
                if mention.kind == "file":
                    mention.resolved_content = self._resolve_file(mention.target)
                elif mention.kind == "dir":
                    mention.resolved_content = self._resolve_dir(mention.target)
                elif mention.kind == "git":
                    mention.resolved_content = self._resolve_git(mention.target)
                elif mention.kind == "web":
                    mention.resolved_content = self._resolve_web(mention.target)
                elif mention.kind == "url":
                    mention.resolved_content = self._resolve_url(mention.target)
            except Exception as exc:
                logger.warning("Failed to resolve mention %s: %s", mention.raw, exc)
                mention.resolved_content = f"[Error resolving {mention.raw}: {exc}]"

        return mentions

    def process_message(self, message: str) -> tuple[str, list[Mention]]:
        """Parse, resolve, and inject mention content into a message.

        Returns a modified message with mention content appended,
        and the list of resolved mentions.

        Args:
            message: Original user message.

        Returns:
            (enriched_message, resolved_mentions)
        """
        mentions = self.parse(message)
        if not mentions:
            return message, []

        self.resolve(mentions)

        # Build context injection
        context_parts: list[str] = []
        for mention in mentions:
            if mention.is_resolved:
                context_parts.append(
                    f"<mention source=\"{mention.raw}\">\n"
                    f"{mention.resolved_content}\n"
                    f"</mention>"
                )

        if context_parts:
            enriched = message + "\n\n" + "\n\n".join(context_parts)
            return enriched, mentions

        return message, mentions

    # ----- Resolvers -----

    def _resolve_file(self, target: str) -> str:
        """Resolve a file mention by reading file contents."""
        path = (self._workspace / target).resolve()

        # Security: ensure within workspace
        try:
            path.relative_to(self._workspace.resolve())
        except ValueError:
            return f"[Error: Path '{target}' is outside the workspace]"

        if not path.exists():
            return f"[File not found: {target}]"

        if not path.is_file():
            return f"[Not a file: {target}]"

        # Size check
        size = path.stat().st_size
        if size > 100_000:  # 100KB
            return f"[File too large: {target} ({size:,} bytes). Use grep_search instead.]"

        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return f"[Error reading {target}: {exc}]"

        return f"# {target}\n```\n{content}\n```"

    def _resolve_dir(self, target: str) -> str:
        """Resolve a directory mention by listing contents."""
        path = (self._workspace / target).resolve()

        try:
            path.relative_to(self._workspace.resolve())
        except ValueError:
            return f"[Error: Path '{target}' is outside the workspace]"

        if not path.exists():
            return f"[Directory not found: {target}]"

        if not path.is_dir():
            return f"[Not a directory: {target}]"

        entries: list[str] = []
        try:
            for child in sorted(path.iterdir()):
                if child.name.startswith("."):
                    continue
                indicator = "/" if child.is_dir() else ""
                size = ""
                if child.is_file():
                    size = f" ({child.stat().st_size:,} bytes)"
                entries.append(f"  {child.name}{indicator}{size}")
        except OSError as exc:
            return f"[Error listing {target}: {exc}]"

        return f"# {target}\n" + "\n".join(entries[:50])

    def _resolve_git(self, target: str) -> str:
        """Resolve a git mention."""
        if not self._git or not self._git.is_git_repo:
            return "[Not a git repository]"

        if target == "diff":
            diff = self._git.diff()
            return diff if diff else "[No uncommitted changes]"

        elif target == "status":
            status = self._git.status()
            return status.summary()

        elif target == "log":
            commits = self._git.log(n=10)
            if not commits:
                return "[No commits]"
            lines = []
            for c in commits:
                lines.append(f"{c.short_hash} {c.message} ({c.author}, {c.date})")
            return "\n".join(lines)

        elif target == "branch":
            return self._git.current_branch() or "[Unknown branch]"

        return f"[Unknown git command: {target}]"

    def _resolve_web(self, query: str) -> str:
        """Resolve a web search mention."""
        from unjess.tools.web_tools import _search_web
        return _search_web(query, max_results=5)

    def _resolve_url(self, url: str) -> str:
        """Resolve a URL mention."""
        from unjess.tools.web_tools import _read_url
        return _read_url(url, max_length=10000)
