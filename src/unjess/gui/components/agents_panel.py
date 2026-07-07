"""Subagent status panel — shows active child agents and their state.

Renders a vertical list of agent cards, each showing role, status,
elapsed time, and a kill button.  Reads from the ``SubagentManager``
on the agent instance.
"""

from __future__ import annotations

import time
from typing import Any, TYPE_CHECKING

from nicegui import ui

if TYPE_CHECKING:
    from unjess.agent import Agent
    from unjess.subagents.manager import AgentInfo, AgentStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STATUS_COLOURS: dict[str, str] = {
    "pending": "amber-6",
    "running": "blue-5",
    "idle": "grey-6",
    "completed": "green-5",
    "failed": "red-5",
    "killed": "grey-7",
}

_STATUS_ICONS: dict[str, str] = {
    "pending": "hourglass_empty",
    "running": "sync",
    "idle": "pause_circle",
    "completed": "check_circle",
    "failed": "error",
    "killed": "cancel",
}


def _format_elapsed(seconds: float) -> str:
    """Format elapsed time in a human-friendly way.

    Args:
        seconds: Elapsed seconds.

    Returns:
        String like ``"12s"``, ``"2m 30s"``, or ``"1h 5m"``.
    """
    if seconds < 60:
        return f"{int(seconds)}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h {mins}m"


def _get_agents(agent: Agent) -> list[Any]:
    """Safely retrieve agent info list from the SubagentManager.

    Args:
        agent: The main Agent instance.

    Returns:
        List of ``AgentInfo`` dataclass instances, or empty list.
    """
    manager = getattr(agent, "_subagent_manager", None)
    if manager is None:
        return []
    agents_dict = getattr(manager, "_agents", {})
    return list(agents_dict.values())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_agents_panel(agent: Agent) -> None:
    """Render the subagent status panel.

    Shows a card for each active or recently-completed subagent with
    its role, type, status badge, elapsed time, and a kill button for
    active agents.

    If no subagents exist, nothing is rendered.

    Args:
        agent: The main Agent instance (reads ``_subagent_manager``).
    """
    agents = _get_agents(agent)

    if not agents:
        with ui.column().classes("w-full items-center q-py-lg"):
            ui.icon("groups").classes("text-grey-8").style("font-size:36px")
            ui.label("No active subagents").classes("text-sm text-grey-7")
        return

    # Separate active from completed
    active = [a for a in agents if a.is_active]
    finished = [a for a in agents if not a.is_active]

    with ui.column().classes("w-full gap-sm"):
        # Active agents header
        if active:
            ui.label(f"Active ({len(active)})").classes(
                "text-xs text-grey-6 text-weight-medium q-px-sm"
            )
            for info in active:
                _render_agent_card(info, agent, is_active=True)

        # Completed / failed agents (collapsed by default)
        if finished:
            with ui.expansion(
                f"Completed ({len(finished)})",
            ).classes("w-full text-xs text-grey-7").props("dense"):
                for info in finished:
                    _render_agent_card(info, agent, is_active=False)


def _render_agent_card(
    info: Any,
    agent: Agent,
    is_active: bool,
) -> None:
    """Render a single subagent status card.

    Args:
        info: ``AgentInfo`` dataclass.
        agent: Parent agent (for the kill action).
        is_active: Whether this agent is still running.
    """
    status_val = info.status.value if hasattr(info.status, "value") else str(info.status)
    colour = _STATUS_COLOURS.get(status_val, "grey-6")
    icon = _STATUS_ICONS.get(status_val, "help")
    elapsed = _format_elapsed(info.runtime_seconds)

    with ui.card().classes(
        "w-full q-pa-sm agent-card cursor-pointer"
    ).style(
        "background:rgba(255,255,255,0.03);"
        "border:1px solid rgba(255,255,255,0.06);"
        "border-radius:8px"
    ):
        with ui.row().classes("w-full items-center no-wrap"):
            # Status icon
            ui.icon(icon).classes(f"text-{colour}").style("font-size:18px")

            # Role + type
            with ui.column().classes("q-ml-sm gap-none"):
                ui.label(info.role or info.type_name).classes(
                    "text-sm text-white text-weight-medium"
                )
                ui.label(
                    f"{info.type_name} · {info.conversation_id[:8]}"
                ).classes("text-xs text-grey-6").style("font-family:monospace")

            ui.space()

            # Elapsed time
            ui.label(elapsed).classes(
                "text-xs text-grey-6"
            ).style("font-family:monospace")

            # Status badge
            ui.badge(
                status_val.upper(),
                color=colour,
            ).props("dense outline").classes("text-xs q-ml-sm")

            # Kill button (active agents only)
            if is_active:
                ui.button(
                    icon="stop",
                    color="red-8",
                    on_click=lambda _info=info: _kill_agent(agent, _info),
                ).props("flat dense round size=sm").tooltip("Kill agent")

        # Expandable detail section — click to see what the agent is doing
        with ui.expansion("Details").classes(
            "w-full text-xs"
        ).props("dense").style(
            "margin-top: 4px;"
        ):
            # Prompt
            ui.label("Prompt:").classes("text-xs text-grey-5 text-weight-bold")
            prompt_text = info.prompt[:500] + ("..." if len(info.prompt) > 500 else "")
            ui.label(prompt_text).classes("text-xs text-grey-6").style(
                "white-space: pre-wrap; word-break: break-word;"
            )

            # Result (if any)
            if info.result:
                ui.separator().classes("q-my-xs")
                ui.label("Result:").classes("text-xs text-green-5 text-weight-bold")
                result_text = info.result[:800] + ("..." if len(info.result) > 800 else "")
                ui.label(result_text).classes("text-xs text-grey-6").style(
                    "white-space: pre-wrap; word-break: break-word;"
                )

            # Error (if any)
            if info.error:
                ui.separator().classes("q-my-xs")
                ui.label("Error:").classes("text-xs text-red-5 text-weight-bold")
                ui.label(info.error).classes("text-xs text-red-4").style(
                    "white-space: pre-wrap; word-break: break-word;"
                )


def _kill_agent(agent: Agent, info: Any) -> None:
    """Kill a running subagent.

    Args:
        agent: The parent agent.
        info: ``AgentInfo`` of the subagent to kill.
    """
    manager = getattr(agent, "_subagent_manager", None)
    if manager is None:
        ui.notify("No subagent manager available", type="warning")
        return

    try:
        manager.kill(info.conversation_id)
        ui.notify(
            f"Killed agent {info.role or info.conversation_id[:8]}",
            type="info",
        )
    except Exception as exc:
        ui.notify(f"Failed to kill agent: {exc}", type="negative")
