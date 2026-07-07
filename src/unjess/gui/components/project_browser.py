"""Project browser sidebar component.

Renders a grouped list of conversations organised by workspace folder,
with the current workspace visually highlighted.  Quick chats (no
workspace) are shown under a separate "Quick Chats" heading.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Callable

from nicegui import ui

if TYPE_CHECKING:
    from unjess.gui.state import AppState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _relative_time(created_at: float | str) -> str:
    """Return a compact human-readable relative time (e.g. '1m', '5h', '2d').

    Accepts either a Unix timestamp (float) or an ISO date string.
    """
    try:
        if isinstance(created_at, (int, float)):
            seconds = int(time.time() - created_at)
        else:
            from datetime import datetime, timezone
            dt = datetime.fromisoformat(str(created_at))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            seconds = int((datetime.now(tz=timezone.utc) - dt).total_seconds())
    except (ValueError, TypeError, OSError):
        return ""

    if seconds < 0:
        return "now"
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h"
    days = hours // 24
    if days < 30:
        return f"{days}d"
    months = days // 30
    if months < 12:
        return f"{months}mo"
    years = days // 365
    return f"{years}y"


def _sort_key(created_at: float | str) -> float:
    """Normalise created_at to a float for sorting."""
    if isinstance(created_at, (int, float)):
        return float(created_at)
    try:
        from datetime import datetime, timezone
        dt = datetime.fromisoformat(str(created_at))
        return dt.timestamp()
    except (ValueError, TypeError):
        return 0.0


def _truncate(text: str, max_len: int = 28) -> str:
    """Truncate *text* to *max_len* characters, appending '…' when trimmed."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _workspace_basename(workspace: str) -> str:
    """Return the last path component of a workspace string."""
    name = workspace.rstrip("/\\").replace("\\", "/").rsplit("/", 1)[-1]
    return name or workspace


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_project_browser(
    state: "AppState",
    on_select_conversation: Callable[[str], None],
    on_refresh: Callable[[], None] | None = None,
) -> None:
    """Render the left-sidebar project browser.

    Parameters
    ----------
    state:
        Application state containing ``memory_store`` and ``workspace``.
    on_select_conversation:
        Callback invoked with the ``conversation_id`` when a conversation
        item is clicked.
    on_refresh:
        Callback to re-render the browser after a mutation (rename/delete).
    """

    # -- fetch summaries ------------------------------------------------------
    # Gather from memory store
    stored_summaries: list = []
    if state.memory_store is not None:
        try:
            stored_summaries = state.memory_store.get_recent_summaries(count=50)
        except Exception:  # noqa: BLE001
            stored_summaries = []

    # Merge with local (in-session) conversations
    # Use a simple namespace so both sources have the same attribute access.
    class _E:
        """Lightweight entry wrapper."""
        def __init__(self, cid: str, title: str, workspace: str, created_at: float) -> None:
            self.conversation_id = cid
            self.title = title
            self.workspace = workspace
            self.created_at = created_at

    seen_ids: set[str] = set()
    entries: list[_E] = []

    for s in stored_summaries:
        if s.conversation_id not in seen_ids:
            seen_ids.add(s.conversation_id)
            entries.append(_E(s.conversation_id, s.title, s.workspace, s.created_at))

    for lc in getattr(state, "local_conversations", []):
        cid = lc.get("conversation_id", "")
        if cid and cid not in seen_ids:
            seen_ids.add(cid)
            entries.append(_E(cid, lc.get("title", cid), lc.get("workspace", ""), lc.get("created_at", 0.0)))

    if not entries:
        ui.label("No conversations yet").style(
            "color: #757575; font-size: 0.8rem; padding: 8px 16px;"
        )
        return

    # -- separate quick chats from workspace conversations --------------------
    workspace_groups: dict[str, list] = {}
    quick_chats: list = []

    for s in entries:
        ws = s.workspace or ""
        if not ws:
            quick_chats.append(s)
        else:
            workspace_groups.setdefault(ws, []).append(s)

    # Sort conversations inside each group by created_at descending.
    for convs in workspace_groups.values():
        convs.sort(key=lambda c: _sort_key(c.created_at), reverse=True)
    quick_chats.sort(key=lambda c: _sort_key(c.created_at), reverse=True)

    # Sort workspace groups by most-recent conversation.
    sorted_workspaces = sorted(
        workspace_groups.keys(),
        key=lambda ws: _sort_key(workspace_groups[ws][0].created_at),
        reverse=True,
    )

    current_ws = state.workspace or ""

    # -- section header -------------------------------------------------------
    ui.label("PROJECTS").style(
        "font-size: 0.65rem; font-variant: small-caps; letter-spacing: 0.1em; "
        "color: #9e9e9e; padding: 12px 12px 4px 12px;"
    )

    # -- render each workspace group ------------------------------------------
    for ws in sorted_workspaces:
        convs = workspace_groups[ws]
        basename = _workspace_basename(ws)
        is_current = (ws.replace("\\", "/").rstrip("/").lower()
                      == current_ws.replace("\\", "/").rstrip("/").lower())

        # Workspace folder row
        with ui.row().classes("items-center gap-1 px-3 pt-3 pb-1"):
            ui.icon("folder").style(
                f"font-size: 1rem; color: {'#90caf9' if is_current else '#9e9e9e'};"
            )
            label = ui.label(basename).style(
                "font-size: 0.78rem; font-weight: 600; "
                f"color: {'#e0e0e0' if is_current else '#9e9e9e'};"
            )
            if is_current:
                label.tooltip("Current workspace")

        # Conversation items under this workspace
        _render_conversation_list(convs, on_select_conversation, state, on_refresh)

    # -- render quick chats section -------------------------------------------
    if quick_chats:
        with ui.row().classes("items-center gap-1 px-3 pt-3 pb-1"):
            ui.icon("chat").style("font-size: 1rem; color: #9e9e9e;")
            ui.label("Quick Chats").style(
                "font-size: 0.78rem; font-weight: 600; color: #9e9e9e;"
            )

        _render_conversation_list(quick_chats, on_select_conversation, state, on_refresh)


def _render_conversation_list(
    convs: list,
    on_select: Callable[[str], None],
    state: "AppState",
    on_refresh: Callable[[], None] | None = None,
) -> None:
    """Render a flat list of conversation items with context actions."""

    def _do_delete(cid: str) -> None:
        """Show confirmation dialog before deleting a conversation."""
        with ui.dialog() as dlg, ui.card().style(
            "background: #1a1a1a; min-width: 320px;"
        ):
            ui.label("Delete Conversation").classes(
                "text-sm font-semibold text-gray-300 mb-2"
            )
            ui.label(
                "Are you sure you want to delete this conversation? "
                "This cannot be undone."
            ).classes("text-sm text-gray-400")

            with ui.row().classes("w-full justify-end gap-2 mt-3"):
                ui.button("Cancel", on_click=dlg.close).props(
                    "flat dense no-caps"
                ).classes("text-gray-500")

                def _confirm(_e: object, _cid: str = cid, _dlg: object = dlg) -> None:
                    if state.memory_store:
                        state.memory_store.delete_summary(_cid)
                    state.local_conversations = [
                        lc for lc in state.local_conversations
                        if lc.get("conversation_id") != _cid
                    ]
                    ui.notify("Conversation deleted", type="info", position="top")
                    state.save_conversations()
                    _dlg.close()
                    if on_refresh:
                        on_refresh()

                ui.button("Delete", on_click=_confirm).props(
                    "unelevated dense no-caps"
                ).classes("text-white").style(
                    "background: #c62828;"
                )

        dlg.open()

    def _do_rename(cid: str, current_title: str) -> None:
        """Show a rename dialog for a conversation."""
        with ui.dialog() as dlg, ui.card().style(
            "background: #1a1a1a; min-width: 320px;"
        ):
            ui.label("Rename Conversation").classes(
                "text-sm font-semibold text-gray-300 mb-2"
            )
            inp = ui.input(value=current_title).props(
                'outlined dense dark'
            ).classes("w-full")

            with ui.row().classes("w-full justify-end gap-2 mt-3"):
                ui.button("Cancel", on_click=dlg.close).props(
                    "flat dense no-caps"
                ).classes("text-gray-500")

                def _save(_e, _cid=cid, _inp=inp, _dlg=dlg) -> None:
                    new_title = _inp.value.strip()
                    if not new_title:
                        return
                    if state.memory_store:
                        state.memory_store.rename_summary(_cid, new_title)
                    for lc in state.local_conversations:
                        if lc.get("conversation_id") == _cid:
                            lc["title"] = new_title
                    ui.notify("Renamed", type="info", position="top")
                    _dlg.close()
                    # Persist to disk
                    state.save_conversations()
                    if on_refresh:
                        on_refresh()

                ui.button("Save", on_click=_save).props(
                    "unelevated dense no-caps"
                ).classes("text-white").style("background: var(--accent);")

        dlg.open()

    for idx, conv in enumerate(convs):
        cid = conv.conversation_id
        full_title = conv.title or cid
        title = _truncate(full_title)
        rel = _relative_time(conv.created_at)
        _wrapper_id = f"conv-item-{idx}-{id(convs)}"

        with ui.element("div").classes(
            "w-full"
        ).style(
            "position: relative;"
        ) as wrapper:
            wrapper.props(f'id="{_wrapper_id}"')

            # On hover: hide chat icon, show action icons
            ui.html(f'''<style>
                #{_wrapper_id} .conv-actions {{ display: none; }}
                #{_wrapper_id} .conv-icon {{ display: inline-flex; }}
                #{_wrapper_id}:hover .conv-actions {{ display: inline-flex; align-items: center; gap: 2px; }}
                #{_wrapper_id}:hover .conv-icon {{ display: none; }}
            </style>''')

            btn = (
                ui.button(on_click=lambda _e, _cid=cid: on_select(_cid))
                .props("flat dense no-caps align=left")
                .classes("w-full pl-6 pr-2")
                .style("text-transform: none; justify-content: flex-start;")
            )

            with btn:
                with ui.row().classes("items-center gap-1 w-full"):
                    # Default icon (visible when not hovering)
                    ui.icon("chat_bubble_outline").classes("conv-icon").style(
                        "font-size: 14px; color: #757575;"
                    )
                    # Action icons (visible on hover, replace chat icon)
                    with ui.element("span").classes("conv-actions"):
                        ui.icon("edit").style(
                            "font-size: 13px; color: #666; cursor: pointer;"
                        ).on("click.stop", lambda _e, _cid=cid, _t=full_title: _do_rename(_cid, _t))
                        ui.icon("delete_outline").style(
                            "font-size: 13px; color: #666; cursor: pointer;"
                        ).on("click.stop", lambda _e, _cid=cid: _do_delete(_cid))

                    ui.label(title).style(
                        "font-size: 0.78rem; color: #bdbdbd; text-align: left; "
                        "white-space: nowrap; overflow: hidden; text-overflow: ellipsis; flex: 1;"
                    )
                    if rel:
                        ui.label(rel).style(
                            "font-size: 0.65rem; color: #616161; min-width: 24px; text-align: right;"
                        )


