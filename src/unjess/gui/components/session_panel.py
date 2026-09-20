"""Session panel component for the right sidebar.

Renders collapsible sections for subagents, files changed,
artifacts, and background tasks with status indicators and
relative timestamps.
"""
from __future__ import annotations

from datetime import datetime
import time
from typing import TYPE_CHECKING

from nicegui import ui

if TYPE_CHECKING:
    from unjess.gui.state import AppState


def _relative_time(dt: "datetime | float") -> str:
    """Return a human-friendly relative time string like '3m' or '2h'."""
    if isinstance(dt, (int, float)):
        seconds = time.time() - dt
    else:
        delta = datetime.now() - dt
        seconds = delta.total_seconds()
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds / 60)}m"
    if seconds < 86400:
        return f"{int(seconds / 3600)}h"
    return f"{int(seconds / 86400)}d"


_STATUS_ICONS: dict[str, tuple[str, str]] = {
    "COMPLETED": ("check_circle", "#4ade80"),
    "RUNNING": ("hourglass_top", "#facc15"),
    "PENDING": ("schedule", "#94a3b8"),
    "IDLE": ("pause_circle", "#94a3b8"),
    "FAILED": ("cancel", "#f87171"),
    "KILLED": ("stop_circle", "#f87171"),
}


def _section_header(label: str, count: int) -> None:
    """Render a section header row with a count badge."""
    with ui.row().classes("w-full items-center gap-1 px-2 pt-3 pb-1"):
        ui.label(label).classes("text-xs font-semibold tracking-wider text-gray-600")
        ui.label(str(count)).classes("text-xs text-gray-500 ml-2")


def _render_subagents_section(state: "AppState") -> None:
    """Render the subagents collapsible section."""
    if state.subagent_manager is None:
        agents: list = []
    else:
        all_agents = state.subagent_manager.list_all()
        conv_id = state.current_conversation_id
        agents = [a for a in all_agents if getattr(a, "parent_conversation_id", "") == conv_id]

    _section_header("SUBAGENTS", len(agents))

    with ui.expansion("", value=True).classes("w-full"):
        if not agents:
            ui.label("No subagents").classes("text-xs text-gray-500 px-2 py-1")
            return

        items = list(agents)
        visible = items[:5]

        for agent in visible:
            with ui.row().classes("w-full items-center justify-between px-2 py-1"):
                with ui.column().classes("gap-0"):
                    ui.label(agent.role).classes("text-sm font-semibold text-gray-300")
                    if agent.status.name == "RUNNING":
                        ui.label("Running...").classes("text-xs text-gray-500")
                    elif agent.finished_at and agent.created_at:
                        elapsed = agent.finished_at - agent.created_at
                        minutes = int(elapsed / 60)
                        ui.label(f"Worked for {minutes}m").classes("text-xs text-gray-500")
                    elif agent.created_at:
                        ui.label(f"Started {_relative_time(agent.created_at)} ago").classes(
                            "text-xs text-gray-500"
                        )

                icon_name, icon_color = _STATUS_ICONS.get(
                    agent.status.name, ("help", "#94a3b8")
                )
                ui.icon(icon_name).style(f"color: {icon_color}; font-size: 18px")

        if len(items) > 5:
            ui.label(f"See all ({len(items)})").classes(
                "text-xs text-blue-400 cursor-pointer px-2 py-1"
            )


def _render_files_changed_section(state: "AppState") -> None:
    """Render the files changed collapsible section."""
    files: list[dict] = getattr(state, "files_changed", []) or []

    _section_header("FILES CHANGED", len(files))

    with ui.expansion("", value=True).classes("w-full"):
        if not files:
            ui.label("No files changed").classes("text-xs text-gray-500 px-2 py-1")
            return

        visible = files[:5]

        for f in visible:
            with ui.row().classes("w-full items-center justify-between px-2 py-1"):
                with ui.column().classes("gap-0"):
                    ui.label(f.get("filename", "unknown")).classes(
                        "text-sm font-semibold text-gray-300"
                    )
                    ui.label(f.get("path", "")).classes("text-xs text-gray-500")

                with ui.row().classes("items-center gap-2"):
                    additions = f.get("additions", 0)
                    deletions = f.get("deletions", 0)
                    if additions:
                        ui.label(f"+{additions}").style("color: #4ade80; font-size: 12px")
                    if deletions:
                        ui.label(f"-{deletions}").style("color: #f87171; font-size: 12px")

        if len(files) > 5:
            ui.label(f"See all ({len(files)})").classes(
                "text-xs text-blue-400 cursor-pointer px-2 py-1"
            )


def _render_artifact_content(full_path: str, ext: str) -> None:
    """Read and render artifact file contents inline.

    Markdown files are rendered as rich HTML.
    Other text files are shown in a scrollable code block.
    """
    from pathlib import Path

    if not full_path:
        ui.label("(no path)").classes("text-xs text-gray-500")
        return

    path = Path(full_path)
    if not path.exists():
        ui.label("(file not found)").classes("text-xs text-red-400")
        return

    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        ui.label(f"(read error: {exc})").classes("text-xs text-red-400")
        return

    # Limit to 200 lines
    lines = content.splitlines()
    truncated = len(lines) > 200
    if truncated:
        content = "\n".join(lines[:200])

    # Wrapper to constrain width within sidebar
    with ui.column().classes("w-full").style(
        "max-width: 100%; min-width: 0; overflow: hidden; "
        "box-sizing: border-box;"
    ):
        if ext == ".md":
            # Render markdown with compact heading & text styles
            ui.markdown(content).classes("w-full text-xs sidebar-artifact-preview").style(
                "max-height: 400px; overflow-y: auto; overflow-x: auto; "
                "padding: 8px; border-radius: 6px; "
                "background: rgba(30, 30, 40, 0.6); "
                "word-wrap: break-word; max-width: 100%; min-width: 0; "
                "box-sizing: border-box;"
            )
        else:
            # Code block — contained within sidebar
            lang = {
                ".py": "python", ".js": "javascript", ".ts": "typescript",
                ".html": "html", ".css": "css", ".json": "json",
                ".yaml": "yaml", ".yml": "yaml", ".bat": "batch",
                ".sh": "bash", ".toml": "toml",
            }.get(ext, "text")
            code_el = ui.code(content, language=lang).classes("w-full")
            code_el.style(
                "max-height: 400px; overflow-y: auto; overflow-x: auto; "
                "font-size: 0.7rem; max-width: 100%; min-width: 0; "
                "box-sizing: border-box;"
            )
            # Force the inner pre/code to not overflow
            ui.run_javascript(f'''
                setTimeout(() => {{
                    const el = document.querySelector('[id="{code_el.id}"]');
                    if (el) {{
                        el.style.maxWidth = '100%';
                        el.style.minWidth = '0';
                        el.style.overflow = 'auto';
                        const pre = el.querySelector('pre');
                        if (pre) {{
                            pre.style.whiteSpace = 'pre-wrap';
                            pre.style.wordBreak = 'break-all';
                            pre.style.maxWidth = '100%';
                            pre.style.overflow = 'auto';
                        }}
                    }}
                }}, 100);
            ''')

    if truncated:
        ui.label(f"... {len(lines) - 200} more lines").classes(
            "text-xs text-gray-500 px-1"
        )


def _render_artifacts_section(state: "AppState") -> None:
    """Render the artifacts collapsible section with inline file preview."""
    artifacts: list[dict] = getattr(state, "artifacts", []) or []

    _section_header("ARTIFACTS", len(artifacts))

    with ui.expansion("", value=True).classes("w-full").style(
        "max-width: 100%; min-width: 0; overflow: hidden;"
    ):
        if not artifacts:
            ui.label("No artifacts yet").classes("text-xs text-gray-500 px-2 py-1")
            return

        # File type icons
        _icons = {
            ".md": "📝", ".html": "🌐", ".py": "🐍", ".js": "📜",
            ".ts": "📜", ".json": "📋", ".yaml": "⚙️", ".yml": "⚙️",
            ".css": "🎨", ".bat": "⚡", ".sh": "⚡", ".toml": "⚙️",
        }

        for artifact in artifacts:
            name = artifact.get("name", "untitled")
            ext = artifact.get("ext", "")
            full_path = artifact.get("full_path", "")
            icon = _icons.get(ext, "📄")
            created = artifact.get("created_at")

            with ui.expansion(f"{icon} {name}", value=False).classes(
                "w-full artifact-item"
            ).style(
                "font-size: 0.85rem; font-weight: 600; color: #d1d5db; "
                "max-width: 100%; min-width: 0; overflow: hidden;"
            ):
                # Show metadata row
                with ui.row().classes("w-full items-center gap-2 px-1 pb-1"):
                    if created and isinstance(created, datetime):
                        ui.label(_relative_time(created)).classes(
                            "text-xs text-gray-500"
                        )
                    ui.label(ext or "file").classes(
                        "text-xs px-1 rounded"
                    ).style(
                        "background: rgba(96, 165, 250, 0.15); color: #93c5fd;"
                    )

                # Render file contents
                _render_artifact_content(full_path, ext)


def _render_tasks_section(state: "AppState") -> None:
    """Render the background tasks collapsible section."""
    if state.task_manager is None:
        tasks: list = []
    else:
        all_tasks = state.task_manager.list_tasks()
        conv_id = state.current_conversation_id
        tasks = [t for t in all_tasks if getattr(t, "parent_conversation_id", "") == conv_id]

    _section_header("BACKGROUND TASKS", len(tasks))

    with ui.expansion("", value=True).classes("w-full"):
        if not tasks:
            ui.label("No background tasks").classes("text-xs text-gray-500 px-2 py-1")
            return

        visible = tasks[:5]

        for task in visible:
            with ui.row().classes("w-full items-center justify-between px-2 py-1"):
                with ui.column().classes("gap-0"):
                    ui.label(task.name).classes("text-sm font-semibold text-gray-300")
                    if task.started_at:
                        ui.label(f"{_relative_time(task.started_at)} ago").classes(
                            "text-xs text-gray-500"
                        )

                icon_name, icon_color = _STATUS_ICONS.get(
                    task.status.name, ("help", "#94a3b8")
                )
                with ui.row().classes("items-center gap-1"):
                    ui.icon(icon_name).style(f"color: {icon_color}; font-size: 16px")
                    ui.label(task.status.name).style(
                        f"color: {icon_color}; font-size: 11px; font-weight: 600"
                    )

        if len(tasks) > 5:
            ui.label(f"See all ({len(tasks)})").classes(
                "text-xs text-blue-400 cursor-pointer px-2 py-1"
            )


def render_session_panel(state: "AppState") -> None:
    """Render the right sidebar session panel.

    Displays session stats, collapsible sections for subagents, files changed,
    artifacts, and background tasks with status indicators.

    Args:
        state: The application state containing subagent_manager,
               files_changed, task_manager, and artifacts data.
    """
    with ui.column().classes("w-full gap-0"):
        # ── Session Stats ────────────────────────────────────────────
        _render_session_stats(state)
        ui.separator().classes("my-1")
        _render_subagents_section(state)
        ui.separator().classes("my-1")
        _render_files_changed_section(state)
        ui.separator().classes("my-1")
        _render_artifacts_section(state)
        ui.separator().classes("my-1")
        _render_tasks_section(state)


def _render_session_stats(state: "AppState") -> None:
    """Render cumulative session statistics at the top of the sidebar."""
    _section_header("SESSION", 0)

    with ui.column().classes("w-full px-3 py-1 gap-1"):
        # Tokens
        with ui.row().classes("items-center gap-2 w-full"):
            ui.icon("arrow_downward", size="14px").style("color: #4caf50")
            ui.label(f"{state.tokens_in:,}").style(
                "font-size: 0.75rem; color: #4caf50; font-family: monospace;"
            )
            ui.icon("arrow_upward", size="14px").style("color: #2196f3")
            ui.label(f"{state.tokens_out:,}").style(
                "font-size: 0.75rem; color: #2196f3; font-family: monospace;"
            )
            total = state.tokens_in + state.tokens_out
            ui.label(f"({total:,} total)").style(
                "font-size: 0.65rem; color: #616161; margin-left: auto;"
            )

        # Cost
        with ui.row().classes("items-center gap-2 w-full"):
            ui.icon("payments", size="14px").style("color: #e0e0e0")
            if state.session_cost > 0:
                ui.label(f"${state.session_cost:.4f}").style(
                    "font-size: 0.75rem; color: #e0e0e0; font-family: monospace; font-weight: 600;"
                )
                if state.is_free:
                    ui.label("(Free model active)").style(
                        "font-size: 0.65rem; color: #4caf50;"
                    )
            elif state.is_free:
                ui.label("FREE TIER").style(
                    "font-size: 0.75rem; color: #4caf50; font-weight: 600;"
                )
                if state.free_quota_pct is not None:
                    pct = int(state.free_quota_pct * 100)
                    ui.label(f"{pct}% remaining").style(
                        "font-size: 0.65rem; color: #616161;"
                    )
            else:
                ui.label("$0.0000").style(
                    "font-size: 0.75rem; color: #616161; font-family: monospace;"
                )

        # Model/Provider
        with ui.row().classes("items-center gap-2 w-full"):
            ui.icon("smart_toy", size="14px").style("color: #9e9e9e")
            model_text = state.model or "no model"
            ui.label(model_text).style(
                "font-size: 0.75rem; color: #bdbdbd;"
            )
            if state.provider:
                ui.label(f"({state.provider})").style(
                    "font-size: 0.65rem; color: #616161;"
                )

    # --- Historical usage (persisted across restarts) ---
    _render_usage_history()


def _format_cost(cost: float) -> str:
    """Format a cost value for display."""
    if cost == 0:
        return "FREE"
    if cost < 0.01:
        return f"${cost:.4f}"
    return f"${cost:.2f}"


def _render_usage_history() -> None:
    """Render persisted usage stats: today / this month / all-time."""
    from unjess.conversation_logger import CostTracker

    try:
        stats = CostTracker.get_usage_summary()
    except Exception:
        return

    _section_header("USAGE", 0)

    with ui.column().classes("w-full px-3 py-1 gap-2"):
        for label, key in [("Today", "today"), ("This Month", "month"), ("All Time", "all_time")]:
            bucket = stats[key]
            if bucket["calls"] == 0 and key != "today":
                continue  # Skip empty periods (but always show today)

            with ui.column().classes("w-full gap-0"):
                ui.label(label).style(
                    "font-size: 0.65rem; color: #757575; font-weight: 600; "
                    "text-transform: uppercase; letter-spacing: 0.05em;"
                )
                with ui.row().classes("items-center gap-3 w-full"):
                    cost_text = _format_cost(bucket["cost"])
                    color = "#4caf50" if cost_text == "FREE" else "#e0e0e0"
                    ui.label(cost_text).style(
                        f"font-size: 0.8rem; color: {color}; font-family: monospace; font-weight: 600;"
                    )
                    tokens = bucket["tokens_in"] + bucket["tokens_out"]
                    if tokens > 0:
                        if tokens > 1_000_000:
                            tok_str = f"{tokens / 1_000_000:.1f}M tok"
                        elif tokens > 1_000:
                            tok_str = f"{tokens / 1_000:.1f}K tok"
                        else:
                            tok_str = f"{tokens:,} tok"
                        ui.label(tok_str).style(
                            "font-size: 0.65rem; color: #616161; font-family: monospace;"
                        )
                    ui.label(f"{bucket['calls']} calls").style(
                        "font-size: 0.65rem; color: #616161; margin-left: auto;"
                    )
