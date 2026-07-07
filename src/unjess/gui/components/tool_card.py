"""Collapsible tool-call cards rendered inside chat messages.

Each card shows the tool name, abbreviated arguments, and an expandable
body with the result text.  File-editing tools with a ``diff`` field
render a coloured diff view instead of plain text.
"""

from __future__ import annotations

import html
from typing import TYPE_CHECKING

from nicegui import ui

from unjess.gui.components.diff_view import render_diff

if TYPE_CHECKING:
    from unjess.gui.state import ToolCallDisplay


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MAX_RESULT_PREVIEW = 2000  # characters before truncation in expanded view
_MAX_ARG_DISPLAY = 60       # characters per argument value in the header


def _abbreviate_args(args: dict) -> str:
    """Build a compact one-line summary of tool arguments.

    Args:
        args: Raw argument dict.

    Returns:
        Abbreviated string like ``path=src/foo.py, query=hello…``.
    """
    parts: list[str] = []
    for key, value in args.items():
        val = str(value)
        if len(val) > _MAX_ARG_DISPLAY:
            val = val[: _MAX_ARG_DISPLAY - 1] + "…"
        parts.append(f"{key}={val}")
    return ", ".join(parts)


def _tool_icon(name: str) -> str:
    """Return an emoji for common tool categories.

    Args:
        name: Tool name.

    Returns:
        Emoji string.
    """
    if "file" in name or "write" in name or "edit" in name:
        return "📝"
    if "read" in name or "view" in name:
        return "📖"
    if "search" in name or "grep" in name:
        return "🔍"
    if "run" in name or "command" in name or "exec" in name:
        return "▶️"
    if "list" in name or "dir" in name:
        return "📁"
    if "web" in name or "url" in name or "browser" in name:
        return "🌐"
    return "🔧"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_tool_card(tc: ToolCallDisplay) -> None:
    """Render a single tool-call card with collapsible details.

    The card header shows the tool icon, name, and abbreviated args.
    Expanding reveals the full result or a coloured diff view.

    Args:
        tc: Tool call display data.
    """
    icon = _tool_icon(tc.name)
    args_summary = _abbreviate_args(tc.args)
    header_text = f"{icon} {tc.name}"

    with ui.expansion(
        text=header_text,
        value=not tc.collapsed,
    ).classes("w-full tool-card q-my-xs").props("dense header-class='text-caption'"):
        # Argument summary
        if args_summary:
            ui.label(args_summary).classes(
                "text-xs text-grey-6 q-px-sm q-pb-xs"
            ).style("font-family:monospace;word-break:break-all")

        # Diff view takes priority
        if tc.diff:
            render_diff(tc.diff, tc.args.get("path", tc.args.get("TargetFile", "")))
        elif tc.result:
            result_text = tc.result
            truncated = False
            if len(result_text) > _MAX_RESULT_PREVIEW:
                result_text = result_text[:_MAX_RESULT_PREVIEW]
                truncated = True

            ui.html(
                f'<pre style="margin:0;padding:8px;overflow-x:auto;'
                f'font-family:monospace;font-size:12px;line-height:1.4;'
                f'color:#c9d1d9;background:rgba(255,255,255,0.03);'
                f'border-radius:4px;white-space:pre-wrap;word-break:break-word">'
                f"{html.escape(result_text)}</pre>"
            ).classes("w-full q-px-sm")

            if truncated:
                remaining = len(tc.result) - _MAX_RESULT_PREVIEW
                ui.label(
                    f"… {remaining:,} more characters"
                ).classes("text-xs text-grey-7 q-px-sm q-pt-xs")
        else:
            # Still running or no result yet
            ui.label("Executing…").classes(
                "text-xs text-grey-7 italic q-px-sm"
            )
