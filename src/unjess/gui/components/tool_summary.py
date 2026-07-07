"""Human-readable tool call summaries for the AG-style trace view.

Generates concise, descriptive summaries like "Read config.py" or
"Edited agent.py +5 -2" instead of raw tool names and argument dumps.
"""

from __future__ import annotations

from pathlib import PurePosixPath, PureWindowsPath


def _basename(path: str) -> str:
    """Extract the filename from a path, handling both Unix and Windows."""
    if not path:
        return "file"
    try:
        p = PureWindowsPath(path) if "\\" in path else PurePosixPath(path)
        return p.name or str(p)
    except Exception:
        return path.rsplit("/", 1)[-1] if "/" in path else path


def _count_diff_changes(diff: str) -> tuple[int, int]:
    """Count additions and deletions from a unified diff string.

    Args:
        diff: Unified diff text.

    Returns:
        Tuple of (additions, deletions).
    """
    adds = 0
    dels = 0
    for line in diff.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            adds += 1
        elif line.startswith("-") and not line.startswith("---"):
            dels += 1
    return adds, dels


def summarize_tool_call(
    name: str,
    args: dict,
    result: str = "",
    diff: str = "",
) -> str:
    """Generate a human-readable summary of a tool call.

    Produces AG-style descriptions like "Read config.py", "Edited agent.py +3 -1",
    or "Searched for 'pattern'" instead of raw tool names.

    Args:
        name: Tool function name.
        args: Tool arguments.
        result: Tool result text (optional, used for richer summaries).
        diff: Unified diff string (optional, used for +/- counts).

    Returns:
        Short human-readable summary string.
    """
    if name == "read_file":
        path = args.get("path", args.get("file_path", ""))
        return f"Read {_basename(path)}"

    elif name == "write_file" or name == "create_file":
        path = args.get("path", args.get("file_path", ""))
        return f"Created {_basename(path)}"

    elif name == "edit_file":
        path = args.get("path", args.get("file_path", ""))
        if diff:
            adds, dels = _count_diff_changes(diff)
            return f"Edited {_basename(path)} +{adds} -{dels}"
        return f"Edited {_basename(path)}"

    elif name == "list_dir":
        path = args.get("path", args.get("directory", ""))
        if result:
            # Count items in result
            item_count = len([
                line for line in result.splitlines()
                if line.strip() and not line.startswith("Directory listing")
            ])
            if item_count > 0:
                return f"Explored {_basename(path)} ({item_count} items)"
        return f"Explored {_basename(path)}"

    elif name == "grep_search":
        query = args.get("query", args.get("pattern", ""))
        path = args.get("path", args.get("directory", ""))
        if len(query) > 30:
            query = query[:27] + "…"
        summary = f'Searched for "{query}"'
        if path:
            summary += f" in {_basename(path)}"
        return summary

    elif name == "run_command":
        cmd = args.get("command", args.get("cmd", ""))
        if len(cmd) > 50:
            cmd = cmd[:47] + "…"
        return f"Ran `{cmd}`"

    elif name == "search_web":
        query = args.get("query", "")
        if len(query) > 40:
            query = query[:37] + "…"
        return f'Searched the web for "{query}"'

    elif name == "read_url":
        url = args.get("url", "")
        if len(url) > 50:
            url = url[:47] + "…"
        return f"Read {url}"

    elif name == "browser_get_text":
        url = args.get("url", "")
        if len(url) > 50:
            url = url[:47] + "…"
        return f"Browsed {url}"

    elif name == "browser_screenshot":
        return "Took a screenshot"

    elif name == "spawn_agent":
        task = args.get("task", args.get("prompt", ""))
        if len(task) > 40:
            task = task[:37] + "…"
        return f'Spawned agent: "{task}"'

    elif name == "ask_question":
        question = args.get("question", "")
        if len(question) > 40:
            question = question[:37] + "…"
        return f'Asked: "{question}"'

    # Fallback: humanize the tool name
    human_name = name.replace("_", " ").title()
    return f"Used {human_name}"


def tool_icon(name: str) -> str:
    """Return a Material icon name for common tool categories.

    Args:
        name: Tool function name.

    Returns:
        Material icon name string.
    """
    if "file" in name or "write" in name or "edit" in name or "create" in name:
        return "edit_note"
    if "read" in name or "view" in name:
        return "description"
    if "search" in name or "grep" in name:
        return "search"
    if "run" in name or "command" in name or "exec" in name:
        return "terminal"
    if "list" in name or "dir" in name:
        return "folder_open"
    if "web" in name or "url" in name or "browser" in name:
        return "language"
    if "agent" in name or "spawn" in name:
        return "smart_toy"
    if "question" in name or "ask" in name:
        return "help_outline"
    return "build"
