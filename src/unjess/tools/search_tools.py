"""Search tools — grep / ripgrep integration."""

import re
import subprocess
import shutil
from pathlib import Path
from typing import Optional

from unjess.tools import ToolRegistry
from unjess.tools.file_tools import _resolve_safe


_MAX_RESULTS = 50


def _grep_search(
    workspace: Path,
    query: str,
    path: str = ".",
    case_insensitive: bool = False,
    regex: bool = False,
) -> str:
    """Search for a pattern in files.

    Uses ripgrep (rg) if available, falls back to a pure-Python search.

    Args:
        query: Search term or regex pattern.
        path: Directory or file to search (relative to workspace).
        case_insensitive: If True, ignore case.
        regex: If True, treat query as regex.

    Returns:
        Formatted search results (capped at 50 matches).
    """
    resolved = _resolve_safe(path, workspace)

    if not resolved.exists():
        return f"Error: Path not found: {path}"

    rg = shutil.which("rg")
    if rg:
        return _search_with_ripgrep(rg, query, resolved, case_insensitive, regex)
    return _search_with_python(query, resolved, workspace, case_insensitive, regex)


def _search_with_ripgrep(
    rg_path: str,
    query: str,
    search_path: Path,
    case_insensitive: bool,
    regex: bool,
) -> str:
    """Run ripgrep and format results."""
    cmd = [rg_path, "--line-number", "--no-heading", "--color=never", f"--max-count={_MAX_RESULTS}"]

    if case_insensitive:
        cmd.append("--ignore-case")
    if not regex:
        cmd.append("--fixed-strings")

    cmd.append(query)
    cmd.append(str(search_path))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=15,
            cwd=str(search_path) if search_path.is_dir() else str(search_path.parent),
        )
    except subprocess.TimeoutExpired:
        return "Error: Search timed out after 15 seconds."
    except Exception as exc:
        return f"Error running ripgrep: {exc}"

    if result.returncode == 1:  # no matches
        return f"No matches found for: {query}"
    if result.returncode != 0 and result.returncode != 1:
        return f"Error: ripgrep returned code {result.returncode}: {result.stderr.strip()}"

    lines = result.stdout.strip().splitlines()
    if len(lines) > _MAX_RESULTS:
        lines = lines[:_MAX_RESULTS]
        count = len(lines)
        lines.append(f"\n... (results capped at {_MAX_RESULTS})")
    else:
        count = len(lines)

    return f"Found {count} match(es) for '{query}':\n\n" + "\n".join(lines)


def _search_with_python(
    query: str,
    search_path: Path,
    workspace: Path,
    case_insensitive: bool,
    regex: bool,
) -> str:
    """Pure Python fallback search (no ripgrep)."""
    flags = re.IGNORECASE if case_insensitive else 0

    if regex:
        try:
            pattern = re.compile(query, flags)
        except re.error as exc:
            return f"Error: Invalid regex pattern: {exc}"
    else:
        escaped = re.escape(query)
        pattern = re.compile(escaped, flags)

    results: list[str] = []

    if search_path.is_file():
        files = [search_path]
    else:
        files = [
            f for f in search_path.rglob("*")
            if f.is_file()
            and not any(part.startswith(".") for part in f.parts)
            and f.suffix not in (".pyc", ".pyo", ".so", ".dll", ".exe", ".bin")
        ]

    for fpath in files:
        if len(results) >= _MAX_RESULTS:
            break

        try:
            text = fpath.read_text(encoding="utf-8", errors="ignore")
        except (PermissionError, OSError):
            continue

        for i, line in enumerate(text.splitlines(), 1):
            if pattern.search(line):
                try:
                    rel = fpath.relative_to(workspace.resolve())
                except ValueError:
                    rel = fpath
                results.append(f"{rel}:{i}: {line.rstrip()}")
                if len(results) >= _MAX_RESULTS:
                    break

    if not results:
        return f"No matches found for: {query}"

    header = f"Found {len(results)} match(es) for '{query}'"
    if len(results) >= _MAX_RESULTS:
        header += f" (capped at {_MAX_RESULTS})"
    header += ":"

    return header + "\n\n" + "\n".join(results)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_search_tools(registry: ToolRegistry, workspace: Path) -> None:
    """Register search tools on the given registry.

    Uses ``get_current_workspace()`` so the search path always reflects
    the active workspace, even after conversation switches.
    """
    from unjess.tools.file_tools import get_current_workspace

    registry.register(
        name="grep_search",
        description=(
            "Search for a text pattern or regex across files. "
            "Uses ripgrep if available, otherwise falls back to Python. "
            "Results capped at 50 matches."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search term or regex pattern."},
                "path": {"type": "string", "description": "Directory or file to search. Default: '.' (workspace root)."},
                "case_insensitive": {"type": "boolean", "description": "If true, ignore case. Default: false."},
                "regex": {"type": "boolean", "description": "If true, treat query as a regex. Default: false."},
            },
            "required": ["query"],
        },
        handler=lambda **kw: _grep_search(get_current_workspace(), **kw),
    )
