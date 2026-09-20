"""File tools — read, write, edit, list_dir.

Paths inside the workspace are auto-approved. Paths outside trigger a
permission prompt so the user can approve on a case-by-case basis.
"""

import difflib
import fnmatch
import os
import unicodedata
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from unjess.config import IGNORE_PATTERNS
from unjess.tools import ToolRegistry

if TYPE_CHECKING:
    from unjess.permissions import PermissionManager


# ---------------------------------------------------------------------------
# Path security
# ---------------------------------------------------------------------------

# Module-level reference set by register_file_tools()
_permissions: Optional["PermissionManager"] = None
_approved_external_paths: set[str] = set()

# Mutable workspace reference — updated dynamically when conversations switch.
# Using a dict instead of a bare Path so closures always read the latest value.
_workspace_ref: dict[str, Optional[Path]] = {"current": None}


def update_workspace(new_workspace: Path) -> None:
    """Update the active workspace for all file tools.

    Called when the user switches conversations or projects so that
    tool handlers always operate in the correct directory.

    Args:
        new_workspace: The new workspace root path.
    """
    _workspace_ref["current"] = new_workspace.resolve()
    # Clear previously-approved external paths since we changed workspace
    _approved_external_paths.clear()


def get_current_workspace() -> Path:
    """Return the current active workspace path.

    Returns:
        The resolved workspace path.

    Raises:
        RuntimeError: If no workspace has been set.
    """
    ws = _workspace_ref["current"]
    if ws is None:
        raise RuntimeError("No workspace has been set. Call register_file_tools() first.")
    return ws


def _fuzzy_resolve_path(candidate: Path) -> Path:
    """Fuzzily resolve candidate path if exact file path does not exist.

    Handles Unicode punctuation mismatches (e.g. '…' vs '.'), quote variations,
    and slight string variations using difflib.
    """
    if candidate.exists():
        return candidate
    parent = candidate.parent
    if not parent.exists() or not parent.is_dir():
        return candidate

    target_name = candidate.name
    # 1. Try normalized Unicode matching ('…' -> '.', etc.)
    norm_target = (
        unicodedata.normalize("NFKD", target_name)
        .replace("…", ".")
        .replace("–", "-")
        .replace("—", "-")
    )
    try:
        for child in parent.iterdir():
            child_norm = (
                unicodedata.normalize("NFKD", child.name)
                .replace("…", ".")
                .replace("–", "-")
                .replace("—", "-")
            )
            if child_norm.lower() == norm_target.lower():
                return child

        # 2. Try close fuzzy matching
        children = {c.name: c for c in parent.iterdir()}
        matches = difflib.get_close_matches(target_name, children.keys(), n=1, cutoff=0.85)
        if matches:
            return children[matches[0]]
    except Exception:
        pass

    return candidate


def _resolve_safe(path_str: str, workspace: Path, operation: str = "access") -> Path:
    """Resolve a path, asking permission if it's outside the workspace.

    Paths inside the workspace are always allowed. Paths outside require
    user approval via the permission system, with a prominent warning
    for write operations.

    Args:
        path_str: Raw path string from the LLM.
        workspace: The workspace root.
        operation: Description of what we're doing (for the prompt).

    Raises:
        PermissionError: If the user denies access.
    """
    # Handle relative and absolute paths
    candidate = Path(path_str)
    if not candidate.is_absolute():
        candidate = workspace / candidate

    candidate = _fuzzy_resolve_path(candidate)

    resolved = candidate.resolve()
    ws_resolved = workspace.resolve()

    # Inside workspace — always allowed
    if resolved == ws_resolved or resolved.is_relative_to(ws_resolved):
        return resolved

    # Outside workspace — always ask permission (reads AND writes)
    resolved_str = str(resolved)

    # Check if already approved this session
    if resolved_str in _approved_external_paths:
        return resolved

    # Check if parent directory was approved
    for approved_path in _approved_external_paths:
        if resolved_str.startswith(approved_path + os.sep):
            return resolved

    if _permissions is not None:
        # Build a clear warning message for outside-workspace access
        if operation in ("write", "edit", "create"):
            reason = (
                f"⚠️  OUTSIDE WORKSPACE — The AI is trying to {operation} to:\n"
                f"  {resolved}\n"
                f"This is OUTSIDE your project folder ({ws_resolved}).\n"
                f"This is likely a mistake. The AI should use relative paths."
            )
        else:
            reason = (
                f"Outside workspace — The AI wants to {operation}:\n"
                f"  {resolved}\n"
                f"Your project folder is: {ws_resolved}"
            )

        approved = _permissions.request_approval(
            tool_name=f"file_{operation}",
            args={"path": str(resolved), "reason": reason},
        )
        if approved:
            _approved_external_paths.add(resolved_str)
            return resolved

    raise PermissionError(
        f"Access denied: '{path_str}' is outside the workspace ({ws_resolved}). "
        f"The user declined permission."
    )


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _read_file(
    workspace: Path,
    path: str,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
) -> str:
    """Read a file, optionally returning a line range.

    Args:
        path: File path (relative to workspace or absolute within workspace).
        start_line: 1-indexed start line (inclusive).
        end_line: 1-indexed end line (inclusive).

    Returns:
        File contents with line numbers.
    """
    resolved = _resolve_safe(path, workspace)

    if not resolved.exists():
        return f"Error: File not found: {path}"
    if not resolved.is_file():
        return f"Error: Not a file: {path}"

    try:
        content = resolved.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"Error reading file: {exc}"

    lines = content.splitlines()
    total = len(lines)

    # Apply line range
    s = (start_line - 1) if start_line and start_line >= 1 else 0
    e = end_line if end_line and end_line <= total else total

    if s >= total:
        return f"Error: start_line {start_line} exceeds file length ({total} lines)"

    # Cap at 1000 lines
    max_lines = 1000
    if (e - s) > max_lines:
        e = s + max_lines

    selected = lines[s:e]
    numbered = [f"{i + s + 1}: {line}" for i, line in enumerate(selected)]

    header = f"File: {path} ({total} lines total, showing {s + 1}-{e})"
    return header + "\n" + "\n".join(numbered)


def _write_file(
    workspace: Path,
    path: str,
    content: str,
    overwrite: bool = False,
) -> str:
    """Create or overwrite a file.

    Args:
        path: File path (relative to workspace or absolute within workspace).
        content: File content to write.
        overwrite: If False, refuse to overwrite existing files.

    Returns:
        Success or error message.
    """
    resolved = _resolve_safe(path, workspace)

    if resolved.exists() and not overwrite:
        return (
            f"Error: File already exists: {path}. "
            f"Set overwrite=true to replace it."
        )

    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
    except Exception as exc:
        return f"Error writing file: {exc}"

    lines = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
    return f"Successfully wrote {lines} lines to {path}"


def _edit_file(
    workspace: Path,
    path: str,
    target: str,
    replacement: str,
) -> str:
    """Edit a file by replacing a target string match.

    Supports exact substring matching, newline-normalized matching (\\r\\n vs \\n),
    and whitespace-trimmed line block matching for resilience with local models.

    Args:
        path: File path.
        target: Exact string to find in the file.
        replacement: String to replace it with.

    Returns:
        A unified diff of the change, or an error message.
    """
    resolved = _resolve_safe(path, workspace)

    if not resolved.exists():
        return f"Error: File not found: {path}"

    try:
        old_content = resolved.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"Error reading file: {exc}"

    new_content: Optional[str] = None

    # Strategy 1: Direct exact substring match
    if target in old_content:
        count = old_content.count(target)
        if count > 1:
            return (
                f"Error: Target string found {count} times in {path}. "
                f"Please provide a more specific target that matches exactly once."
            )
        new_content = old_content.replace(target, replacement, 1)

    # Strategy 2: Newline-normalized match (\r\n / \r\r\n / \r vs \n)
    if new_content is None:
        norm_target = target.replace("\r\r\n", "\n").replace("\r\n", "\n").replace("\r", "\n")
        norm_old = old_content.replace("\r\r\n", "\n").replace("\r\n", "\n").replace("\r", "\n")
        if norm_target in norm_old:
            count = norm_old.count(norm_target)
            if count == 1:
                norm_rep = replacement.replace("\r\r\n", "\n").replace("\r\n", "\n").replace("\r", "\n")
                norm_new = norm_old.replace(norm_target, norm_rep, 1)
                new_content = norm_new.replace("\n", "\r\n") if "\r\n" in old_content else norm_new
            elif count > 1:
                return (
                    f"Error: Target string found {count} times in {path}. "
                    f"Please provide a more specific target that matches exactly once."
                )

    # Strategy 3: Whitespace-tolerant line block match
    if new_content is None:
        norm_target = target.replace("\r\r\n", "\n").replace("\r\n", "\n").replace("\r", "\n")
        norm_old = old_content.replace("\r\r\n", "\n").replace("\r\n", "\n").replace("\r", "\n")
        target_lines = [l.strip() for l in norm_target.splitlines() if l.strip()]
        old_lines = norm_old.splitlines()

        non_empty_indices = [idx for idx, line in enumerate(old_lines) if line.strip()]
        non_empty_lines = [old_lines[idx].strip() for idx in non_empty_indices]

        if target_lines and len(non_empty_lines) >= len(target_lines):
            matches: list[tuple[int, int]] = []
            t_len = len(target_lines)
            for i in range(len(non_empty_lines) - t_len + 1):
                if non_empty_lines[i : i + t_len] == target_lines:
                    start_old_idx = non_empty_indices[i]
                    end_old_idx = non_empty_indices[i + t_len - 1] + 1
                    matches.append((start_old_idx, end_old_idx))

            if len(matches) == 1:
                start_i, end_i = matches[0]
                rep_lines = replacement.replace("\r\r\n", "\n").replace("\r\n", "\n").replace("\r", "\n").splitlines()
                replaced_lines = old_lines[:start_i] + rep_lines + old_lines[end_i:]
                norm_new = "\n".join(replaced_lines)
                if norm_old.endswith("\n"):
                    norm_new += "\n"
                new_content = norm_new.replace("\n", "\r\n") if "\r\n" in old_content else norm_new
            elif len(matches) > 1:
                return (
                    f"Error: Target string matched {len(matches)} locations in {path} with fuzzy matching. "
                    f"Please provide a more specific target that matches exactly once."
                )

    if new_content is None:
        lines = old_content.splitlines()
        preview = "\n".join(lines[:20])
        return (
            f"Error: Target string not found in {path}.\n"
            f"File starts with:\n{preview}"
        )

    try:
        resolved.write_text(new_content, encoding="utf-8")
    except Exception as exc:
        return f"Error writing file: {exc}"

    # Generate diff
    diff = difflib.unified_diff(
        old_content.splitlines(keepends=True),
        new_content.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
    )
    diff_text = "".join(diff)

    return f"Successfully edited {path}\n\n{diff_text}"


def _list_dir(workspace: Path, path: str = ".") -> str:
    """List directory contents with file sizes.

    Args:
        path: Directory path (relative to workspace).

    Returns:
        Formatted directory listing.
    """
    resolved = _resolve_safe(path, workspace)

    if not resolved.exists():
        return f"Error: Directory not found: {path}"
    if not resolved.is_dir():
        return f"Error: Not a directory: {path}"

    # Gather entries
    entries: list[str] = []
    # Separate exact-match names from glob patterns for efficient matching
    _exact_ignores: set[str] = set()
    _glob_ignores: list[str] = []
    for pat in IGNORE_PATTERNS:
        if any(ch in pat for ch in "*?["):
            _glob_ignores.append(pat)
        else:
            _exact_ignores.add(pat)

    try:
        items = sorted(resolved.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except PermissionError:
        return f"Error: Permission denied reading: {path}"

    ws_root = workspace.resolve()
    for item in items:
        name = item.name
        if name.startswith("."):
            continue
        if name in _exact_ignores:
            continue
        if any(fnmatch.fnmatch(name, pat) for pat in _glob_ignores):
            continue

        try:
            if item.is_relative_to(ws_root):
                rel = item.relative_to(ws_root)
            else:
                rel = item.relative_to(resolved)
        except ValueError:
            rel = Path(item.name)

        if item.is_dir():
            # Fast empty check instead of counting all children
            try:
                is_empty = not any(item.iterdir())
                suffix = " (empty)" if is_empty else ""
            except (PermissionError, OSError):
                suffix = " (access denied)"
            entries.append(f"  📁 {rel}/{suffix}")
        else:
            try:
                size = item.stat().st_size
            except OSError:
                size = 0
            entries.append(f"  📄 {rel} ({_format_size(size)})")

    header = f"Directory: {path} ({len(entries)} items)"
    return header + "\n" + "\n".join(entries) if entries else header + "\n  (empty)"


def _format_size(size_bytes: int) -> str:
    """Format a byte count into a human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_file_tools(
    registry: ToolRegistry,
    workspace: Path,
    permissions: Optional["PermissionManager"] = None,
) -> None:
    """Register all file tools on the given registry.

    Tool handlers use ``get_current_workspace()`` instead of capturing
    *workspace* in a closure, so the active workspace can be changed at
    runtime via :func:`update_workspace`.

    Args:
        registry: The tool registry to register on.
        workspace: The initial workspace root for path sandboxing.
        permissions: Optional permission manager for outside-workspace access.
    """
    global _permissions
    _permissions = permissions

    # Store initial workspace in the mutable reference
    _workspace_ref["current"] = workspace.resolve()

    registry.register(
        name="read_file",
        description="Read a file's contents, optionally specifying a line range.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to the workspace root."},
                "start_line": {"type": "integer", "description": "Start line (1-indexed, inclusive). Optional."},
                "end_line": {"type": "integer", "description": "End line (1-indexed, inclusive). Optional."},
            },
            "required": ["path"],
        },
        handler=lambda **kw: _read_file(get_current_workspace(), **kw),
    )

    registry.register(
        name="write_file",
        description=(
            "Create a new file or overwrite an existing one. "
            "Parent directories are created automatically."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to the workspace root."},
                "content": {"type": "string", "description": "The full content to write to the file."},
                "overwrite": {"type": "boolean", "description": "Set true to overwrite an existing file. Default: false."},
            },
            "required": ["path", "content"],
        },
        handler=lambda **kw: _write_file(get_current_workspace(), **kw),
    )

    registry.register(
        name="edit_file",
        description=(
            "Edit a file by finding an exact string match and replacing it. "
            "The target must match exactly once in the file. "
            "Returns a unified diff of the change."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to the workspace root."},
                "target": {"type": "string", "description": "The exact string to find and replace. Must match exactly once."},
                "replacement": {"type": "string", "description": "The replacement string."},
            },
            "required": ["path", "target", "replacement"],
        },
        handler=lambda **kw: _edit_file(get_current_workspace(), **kw),
    )

    registry.register(
        name="list_dir",
        description="List the files and subdirectories in a directory.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path relative to the workspace root. Default: '.'"},
            },
            "required": [],
        },
        handler=lambda **kw: _list_dir(get_current_workspace(), **kw),
    )
