"""Syntax-highlighted unified diff viewer.

Renders unified-diff text with coloured lines (green for additions,
red for deletions, grey for context) inside a ``ui.html`` block.
"""

from __future__ import annotations

import html
import re

from nicegui import ui


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DIFF_HEADER_RE = re.compile(r"^(---|\+\+\+|@@)")


def _count_changes(diff_text: str) -> tuple[int, int]:
    """Count added / removed lines in a unified diff.

    Args:
        diff_text: Full unified-diff string.

    Returns:
        ``(additions, deletions)`` counts, excluding diff headers.
    """
    additions = 0
    deletions = 0
    for line in diff_text.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            additions += 1
        elif line.startswith("-") and not line.startswith("---"):
            deletions += 1
    return additions, deletions


def _colorize_line(raw_line: str) -> str:
    """Return an HTML ``<span>`` for a single diff line.

    Args:
        raw_line: A single line of unified diff output.

    Returns:
        An HTML string with appropriate colour styling.
    """
    escaped = html.escape(raw_line)

    if raw_line.startswith("+++") or raw_line.startswith("---"):
        return f'<span style="color:#8b949e;font-weight:bold">{escaped}</span>'
    if raw_line.startswith("@@"):
        return f'<span style="color:#79c0ff">{escaped}</span>'
    if raw_line.startswith("+"):
        return f'<span style="color:#3fb950;background:rgba(63,185,80,0.10)">{escaped}</span>'
    if raw_line.startswith("-"):
        return f'<span style="color:#f85149;background:rgba(248,81,73,0.10)">{escaped}</span>'
    # Context line
    return f'<span style="color:#8b949e">{escaped}</span>'


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_diff(diff_text: str, path: str) -> None:
    """Render a unified diff with syntax-coloured lines.

    Creates a collapsible card showing the file path, ``+/-`` counts,
    and the diff body with per-line colour highlighting.

    Args:
        diff_text: The full unified-diff string.
        path: File path to display in the header.
    """
    if not diff_text or not diff_text.strip():
        return

    additions, deletions = _count_changes(diff_text)

    # Build coloured HTML
    coloured_lines = [_colorize_line(line) for line in diff_text.splitlines()]
    body_html = "\n".join(coloured_lines)

    with ui.card().classes("w-full diff-card q-pa-none q-my-xs"):
        # Header row: filename + counts
        with ui.row().classes(
            "w-full items-center q-px-sm q-py-xs"
        ).style("background:rgba(255,255,255,0.04);border-bottom:1px solid rgba(255,255,255,0.08)"):
            ui.icon("description").classes("text-sm text-grey-6")
            ui.label(path).classes("text-xs text-grey-5 q-ml-xs")
            ui.space()
            if additions:
                ui.label(f"+{additions}").classes("text-xs text-green-5")
            if deletions:
                ui.label(f"−{deletions}").classes("text-xs text-red-5 q-ml-xs")

        # Diff body — pre-formatted, monospace
        ui.html(
            f'<pre style="margin:0;padding:8px 12px;overflow-x:auto;'
            f'font-family:\'JetBrains Mono\',\'Fira Code\',monospace;'
            f'font-size:12px;line-height:1.5;background:transparent">'
            f"{body_html}</pre>"
        ).classes("w-full")
