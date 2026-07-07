"""Chat message card renderer using Tailwind dark mode classes.

Tool calls render ABOVE the response text, collapsed by default.
Stats (tokens, cost) render at the bottom of each assistant message.
A rewind button on each assistant message allows undoing to that point.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional
from nicegui import ui

from unjess.gui.components.tool_card import render_tool_card
from unjess.gui.components.trace_view import render_trace

if TYPE_CHECKING:
    from unjess.gui.state import ChatMessage

# Global rewind callback — set by chat page
_rewind_callback: Optional[Callable[[int, str], None]] = None


def set_rewind_callback(callback: Callable[[int, str], None]) -> None:
    """Register the rewind callback from the chat page.

    Args:
        callback: Function taking (msg_index, checkpoint_id) to rewind to.
    """
    global _rewind_callback
    _rewind_callback = callback


def render_message(msg: "ChatMessage") -> None:
    """Render a single chat message as a dark card."""

    with ui.element("div").classes(
        "w-full rounded-lg p-4"
    ).style("background: #191919; border: 1px solid #222"):
        # Meta row
        with ui.row().classes("no-wrap items-center gap-2 mb-2 w-full"):
            icon = "person" if msg.role == "user" else "smart_toy"
            ui.icon(icon, size="16px").classes("text-gray-600")
            ui.label(msg.role.upper()).classes(
                "text-[11px] font-semibold tracking-wider text-gray-500"
            )

            if msg.is_streaming:
                ui.spinner("dots", size="14px").classes("text-purple-400")
                ui.label("Generating...").classes("text-[11px] text-gray-600")

            # Spacer to push rewind button to the right
            ui.element("div").classes("flex-grow")

            # Rewind button — on user messages that have a checkpoint
            if (
                msg.role == "user"
                and msg.checkpoint_id
                and not msg.is_streaming
                and _rewind_callback is not None
            ):
                _idx = msg.msg_index
                _ckpt = msg.checkpoint_id

                def _do_rewind(_e: object, idx: int = _idx, ckpt: str = _ckpt) -> None:
                    _rewind_callback(idx, ckpt)

                ui.button(icon="history", on_click=_do_rewind).props(
                    "round dense flat size=xs"
                ).classes("text-gray-600").style(
                    "min-width: 24px; width: 24px; height: 24px;"
                ).tooltip("Rewind to this point (undoes all changes after)")

        # Trace view (AG-style) or legacy flat layout
        if msg.trace:
            # AG-style: ordered chronological trace
            render_trace(msg.trace, msg.is_streaming)
        else:
            # Legacy: flat tool calls above text, then thinking
            if msg.tool_calls:
                with ui.column().classes("gap-0 mb-2 w-full"):
                    for tc in msg.tool_calls:
                        render_tool_card(tc)

            # Thinking — collapsible section above the response text
            if msg.thinking and not msg.is_streaming:
                duration = msg.thinking_duration
                if duration >= 1:
                    label = f"Thought for {int(duration)}s"
                elif duration > 0:
                    label = "Thought briefly"
                else:
                    label = "Thinking"

                with ui.expansion(label).classes(
                    "w-full thinking-block"
                ).props("dense").style(
                    "margin-bottom: 8px;"
                ) as expansion:
                    expansion._props["header-class"] = (
                        "text-xs text-gray-500 px-0 py-0"
                    )
                    expansion._props["expand-icon-class"] = (
                        "text-gray-600 text-xs"
                    )
                    ui.markdown(msg.thinking).classes(
                        "text-xs text-gray-500 leading-relaxed"
                    ).style(
                        "max-height: 300px; overflow-y: auto; "
                        "padding: 8px 12px; border-radius: 6px; "
                        "background: rgba(255,255,255,0.03);"
                    )

        # Attached images — shown above text content
        if msg.images:
            with ui.row().classes("w-full flex-wrap gap-2 py-1"):
                for img in msg.images:
                    data_uri = f"data:{img['mime_type']};base64,{img['data']}"
                    if img["mime_type"] == "application/pdf":
                        # PDF: show as a file chip
                        with ui.row().classes(
                            "items-center gap-1 px-2 py-1 rounded"
                        ).style(
                            "background: rgba(255,255,255,0.05); "
                            "border: 1px solid #333;"
                        ):
                            ui.icon("description", size="16px").classes(
                                "text-red-400"
                            )
                            ui.label(img.get("name", "file.pdf")).classes(
                                "text-xs text-gray-300"
                            )
                    else:
                        ui.image(data_uri).style(
                            "max-height: 200px; max-width: 300px; "
                            "border-radius: 8px; border: 1px solid #333; "
                            "object-fit: contain;"
                        )

        # Content — below trace/tool calls and thinking
        if msg.content:
            ui.markdown(msg.content).classes(
                "text-sm text-gray-200 leading-relaxed"
            ).style("user-select: text; -webkit-user-select: text; cursor: text;")

        # Stats footer — tokens in/out, cost, model
        if msg.stats and not msg.is_streaming:
            _render_stats(msg.stats)


def _render_stats(stats: dict) -> None:
    """Render a compact token/cost stats bar at the bottom of a message."""
    tokens_in = stats.get("tokens_in", 0)
    tokens_out = stats.get("tokens_out", 0)
    cost = stats.get("cost", 0)
    model = stats.get("model", "")
    is_free = stats.get("is_free", False)
    thinking = stats.get("thinking_tokens", 0)
    cache_read = stats.get("cache_read_tokens", 0)

    with ui.row().classes(
        "items-center gap-3 mt-3 pt-2 w-full flex-wrap"
    ).style("border-top: 1px solid #222"):
        # Token counts
        if tokens_in or tokens_out:
            with ui.row().classes("items-center gap-1"):
                ui.icon("arrow_downward", size="12px").style("color: #4caf50")
                ui.label(f"{tokens_in:,}").style(
                    "font-size: 0.7rem; color: #4caf50; font-family: monospace;"
                )
                ui.icon("arrow_upward", size="12px").style("color: #2196f3")
                ui.label(f"{tokens_out:,}").style(
                    "font-size: 0.7rem; color: #2196f3; font-family: monospace;"
                )

        # Thinking tokens
        if thinking:
            with ui.row().classes("items-center gap-1"):
                ui.icon("psychology", size="12px").style("color: #9c27b0")
                ui.label(f"{thinking:,}").style(
                    "font-size: 0.7rem; color: #9c27b0; font-family: monospace;"
                )

        # Cache
        if cache_read:
            with ui.row().classes("items-center gap-1"):
                ui.icon("cached", size="12px").style("color: #ff9800")
                ui.label(f"{cache_read:,}").style(
                    "font-size: 0.7rem; color: #ff9800; font-family: monospace;"
                )

        # Cost
        if cost and cost > 0:
            ui.label(f"${cost:.4f}").style(
                "font-size: 0.7rem; color: #e0e0e0; font-family: monospace; font-weight: 600;"
            )
        elif is_free:
            ui.label("FREE").style(
                "font-size: 0.65rem; color: #4caf50; font-weight: 700; "
                "background: rgba(76,175,80,0.15); padding: 1px 6px; border-radius: 4px;"
            )

        # Model name
        if model:
            ui.label(model).style(
                "font-size: 0.65rem; color: #616161; margin-left: auto;"
            )
