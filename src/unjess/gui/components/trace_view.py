"""AG-style ordered trace view for agent execution steps.

Renders a chronological list of thinking, tool calls, and results
as an interactive timeline with collapsible details and smart summaries.
"""

from __future__ import annotations

import html
from typing import TYPE_CHECKING

from nicegui import ui

from unjess.gui.components.diff_view import render_diff
from unjess.gui.components.tool_summary import summarize_tool_call, tool_icon

if TYPE_CHECKING:
    from unjess.gui.state import TraceStep


_MAX_RESULT_PREVIEW = 2000  # characters before truncation
_MAX_ARG_DISPLAY = 80       # characters per argument value


def _abbreviate_args(args: dict) -> str:
    """Build a compact one-line summary of tool arguments."""
    parts: list[str] = []
    for key, value in args.items():
        val = str(value)
        if len(val) > _MAX_ARG_DISPLAY:
            val = val[: _MAX_ARG_DISPLAY - 1] + "…"
        parts.append(f"{key}={val}")
    return ", ".join(parts)


def _format_duration(ms: int) -> str:
    """Format milliseconds into a human-readable duration."""
    if ms < 1000:
        return f"{ms}ms"
    elif ms < 60_000:
        return f"{ms / 1000:.1f}s"
    else:
        minutes = ms // 60_000
        seconds = (ms % 60_000) / 1000
        return f"{minutes}m {seconds:.0f}s"


def _render_thinking_step(step: "TraceStep") -> None:
    """Render a thinking/reasoning trace step."""
    # Duration label
    if step.duration_ms >= 1000:
        label = f"Thought for {_format_duration(step.duration_ms)}"
    elif step.duration_ms > 0:
        label = "Thought briefly"
    else:
        label = "Thinking"

    with ui.expansion(label).classes(
        "w-full trace-step trace-thinking"
    ).props("dense").style(
        "margin: 0; padding: 0;"
    ) as expansion:
        expansion._props["header-class"] = (
            "text-xs text-gray-500 px-0 py-0"
        )
        expansion._props["expand-icon-class"] = (
            "text-gray-600 text-xs"
        )
        if step.content:
            ui.markdown(step.content).classes(
                "text-xs text-gray-500 leading-relaxed"
            ).style(
                "max-height: 300px; overflow-y: auto; "
                "padding: 8px 12px; border-radius: 6px; "
                "background: rgba(255,255,255,0.03);"
            )


def _render_tool_call_step(
    call_step: "TraceStep",
    result_step: "TraceStep | None",
) -> None:
    """Render a paired tool_call + tool_result as one expandable card."""
    icon_name = tool_icon(call_step.name)
    is_complete = result_step is not None
    is_error = is_complete and result_step.content.startswith("Error")

    # Build smart summary header
    summary = summarize_tool_call(
        call_step.name,
        call_step.args,
        result=result_step.content if result_step else "",
        diff=result_step.diff if result_step else "",
    )

    # Duration badge
    duration_text = ""
    if result_step and result_step.duration_ms > 0:
        duration_text = f" ({_format_duration(result_step.duration_ms)})"

    # Status styling
    if is_error:
        status_class = "trace-tool-error"
        status_icon = "error_outline"
    elif is_complete:
        status_class = "trace-tool-complete"
        status_icon = "check_circle"
    else:
        status_class = "trace-tool-running"
        status_icon = ""  # Will use spinner instead

    with ui.element("div").classes("w-full trace-step trace-tool"):
        # Header row — always visible
        with ui.expansion(
            text="",
            value=False,
        ).classes(f"w-full tool-card q-my-xs {status_class}").props(
            "dense"
        ) as expansion:
            expansion._props["header-class"] = "text-caption px-1 py-0"
            expansion._props["expand-icon-class"] = "text-gray-600 text-xs"

            # Custom header content via slot
            with expansion.add_slot("header"):
                with ui.row().classes("items-center gap-2 w-full no-wrap"):
                    # Status icon or spinner
                    if is_complete:
                        ui.icon(status_icon, size="14px").classes(
                            "text-green-400" if not is_error else "text-red-400"
                        )
                    else:
                        ui.spinner("dots", size="14px").classes("text-blue-400")

                    # Tool icon
                    ui.icon(icon_name, size="14px").classes("text-gray-500")

                    # Smart summary
                    ui.label(summary).classes(
                        "text-xs text-gray-300"
                    ).style("flex: 1; overflow: hidden; text-overflow: ellipsis;")

                    # Duration
                    if duration_text:
                        ui.label(duration_text).classes(
                            "text-xs text-gray-600"
                        ).style("font-family: monospace; white-space: nowrap;")

            # Expandable body
            # Arguments
            args_summary = _abbreviate_args(call_step.args)
            if args_summary:
                ui.label(args_summary).classes(
                    "text-xs text-grey-6 q-px-sm q-pb-xs"
                ).style("font-family: monospace; word-break: break-all")

            if result_step:
                # Diff view takes priority
                if result_step.diff:
                    path = call_step.args.get(
                        "path", call_step.args.get("file_path",
                        call_step.args.get("TargetFile", ""))
                    )
                    render_diff(result_step.diff, path)
                elif result_step.content:
                    result_text = result_step.content
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
                        remaining = len(result_step.content) - _MAX_RESULT_PREVIEW
                        ui.label(
                            f"… {remaining:,} more characters"
                        ).classes("text-xs text-grey-7 q-px-sm q-pt-xs")
            else:
                # Still running
                ui.label("Executing…").classes(
                    "text-xs text-grey-7 italic q-px-sm"
                )


def _render_error_step(step: "TraceStep") -> None:
    """Render an error trace step."""
    with ui.row().classes("items-center gap-2 w-full trace-step py-1"):
        ui.icon("error_outline", size="14px").classes("text-red-400")
        ui.label(step.content).classes(
            "text-xs text-red-300"
        ).style("word-break: break-word;")


def _render_info_step(step: "TraceStep") -> None:
    """Render an info/warning trace step."""
    icon = "warning" if step.name == "warning" else "info_outline"
    color = "text-yellow-400" if step.name == "warning" else "text-blue-400"
    text_color = "text-yellow-300" if step.name == "warning" else "text-blue-300"

    with ui.row().classes("items-center gap-2 w-full trace-step py-1"):
        ui.icon(icon, size="14px").classes(color)
        ui.label(step.content).classes(
            f"text-xs {text_color}"
        ).style("word-break: break-word;")


def render_trace(trace: list["TraceStep"], is_streaming: bool = False) -> None:
    """Render an ordered list of trace steps in AG-style.

    Pairs ``tool_call`` steps with their corresponding ``tool_result``
    steps and renders them as unified expandable cards. Thinking steps
    render as collapsible reasoning blocks. Errors and info steps render
    as simple inline messages.

    Args:
        trace: Ordered list of :class:`TraceStep` objects.
        is_streaming: Whether the message is still being generated.
    """
    if not trace:
        return

    with ui.column().classes("gap-0 mb-2 w-full trace-container"):
        # Build a lookup of tool_call -> tool_result pairs
        # For each tool_call, find the next matching tool_result
        result_map: dict[int, "TraceStep"] = {}  # call_index -> result_step
        used_results: set[int] = set()

        for i, step in enumerate(trace):
            if step.step_type == "tool_call":
                # Find the next tool_result with the same name
                for j in range(i + 1, len(trace)):
                    if (
                        j not in used_results
                        and trace[j].step_type == "tool_result"
                        and trace[j].name == step.name
                    ):
                        result_map[i] = trace[j]
                        used_results.add(j)
                        break

        # Render each step in order, skipping tool_results (they're paired)
        for i, step in enumerate(trace):
            if step.step_type == "thinking":
                _render_thinking_step(step)

            elif step.step_type == "tool_call":
                result = result_map.get(i)
                _render_tool_call_step(step, result)

            elif step.step_type == "tool_result":
                # Skip — already rendered as part of the tool_call pair
                if i in used_results:
                    continue
                # Orphaned result (shouldn't happen, but handle gracefully)
                _render_tool_call_step(
                    step,  # Use result as both call and result
                    step,
                )

            elif step.step_type == "error":
                _render_error_step(step)

            elif step.step_type == "info":
                _render_info_step(step)

            # "text" steps are accumulated into msg.content, rendered separately
