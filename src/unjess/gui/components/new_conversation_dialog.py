"""Dialog for starting a new conversation.

Presents three options: continue in current workspace, open a specific
folder, or start a quick chat with no workspace attached.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from nicegui import ui

if TYPE_CHECKING:
    from unjess.gui.state import AppState


def create_new_conversation_dialog(
    state: "AppState",
    on_new_conversation: Callable[[str | None], None],
) -> ui.dialog:
    """Create and return a dialog for starting a new conversation.

    Args:
        state: Current application state, used to read the active workspace.
        on_new_conversation: Callback invoked with the chosen workspace path
            (a string) or ``None`` for a quick chat.

    Returns:
        The constructed :class:`ui.dialog` instance (not yet opened).
    """

    dialog = ui.dialog()
    folder_input_value: dict[str, str] = {"path": ""}

    with dialog, ui.card().style("width: 500px; background: #0a0a0a; border: 1px solid #333"):
        # ── Title ───────────────────────────────────────────────
        ui.label("New Conversation").style(
            "font-size: 1.4rem; font-weight: 600; color: #e0e0e0; "
            "margin-bottom: 8px"
        )

        # ── Option 1: Continue in current workspace (only if one is set) ──
        if state.workspace:
            workspace_basename = state.workspace.rstrip("/\\").rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
            with ui.card().style(
                "background: #1a1a1a; border: 1px solid #333; cursor: pointer; "
                "width: 100%"
            ).on(
                "click",
                lambda: (_select(dialog, on_new_conversation, state.workspace)),
            ):
                ui.label("Continue in current workspace").style(
                    "font-size: 1rem; font-weight: 500; color: #e0e0e0"
                )
                ui.label(workspace_basename).style(
                    "font-size: 0.85rem; color: #888; margin-top: 2px"
                )

        # ── Option 2: Open folder ───────────────────────────────
        with ui.card().style(
            "background: #1a1a1a; border: 1px solid #333; cursor: pointer; "
            "width: 100%"
        ).on(
            "click",
            lambda: (_select(dialog, on_new_conversation, folder_input_value["path"])
                     if folder_input_value["path"] else None),
        ):
            ui.label("Open folder").style(
                "font-size: 1rem; font-weight: 500; color: #e0e0e0"
            )
            ui.label("Type or paste a folder path").style(
                "font-size: 0.85rem; color: #888; margin-top: 2px"
            )
            folder_input = ui.input(
                placeholder="/path/to/project",
                on_change=lambda e: folder_input_value.update(path=e.value),
            ).props("outlined dense dark").style(
                "margin-top: 6px; width: 100%"
            ).on(
                "click.stop",  # Prevent card click when interacting with input.
                lambda: None,
            )

        # ── Recent projects from existing conversations ─────────
        _seen: set[str] = set()
        _projects: list[str] = []
        current_ws = state.workspace or ""
        for conv in state.local_conversations:
            ws = conv.get("workspace", "")
            if ws and ws not in _seen and ws != current_ws:
                _seen.add(ws)
                _projects.append(ws)

        if _projects:
            ui.label("Recent Projects").style(
                "font-size: 0.75rem; color: #666; margin-top: 8px; "
                "text-transform: uppercase; letter-spacing: 1px"
            )
            with ui.column().style(
                "width: 100%; gap: 4px; max-height: 150px; "
                "overflow-y: auto; padding: 2px 0"
            ):
                for proj_path in _projects[:8]:  # Limit to 8 recent
                    import os
                    basename = os.path.basename(proj_path.rstrip("/\\"))
                    with ui.row().style(
                        "width: 100%; padding: 6px 10px; "
                        "background: #1a1a1a; border: 1px solid #282828; "
                        "border-radius: 6px; cursor: pointer; align-items: center; "
                        "gap: 8px;"
                    ).on(
                        "click",
                        lambda _e, p=proj_path: _select(dialog, on_new_conversation, p),
                    ).on(
                        "mouseenter",
                        lambda e: e.sender.style(
                            "width: 100%; padding: 6px 10px; "
                            "background: #252525; border: 1px solid #444; "
                            "border-radius: 6px; cursor: pointer; align-items: center; "
                            "gap: 8px;"
                        ),
                    ).on(
                        "mouseleave",
                        lambda e: e.sender.style(
                            "width: 100%; padding: 6px 10px; "
                            "background: #1a1a1a; border: 1px solid #282828; "
                            "border-radius: 6px; cursor: pointer; align-items: center; "
                            "gap: 8px;"
                        ),
                    ):
                        ui.icon("folder", size="16px").style("color: #888")
                        with ui.column().style("gap: 0"):
                            ui.label(basename).style(
                                "font-size: 0.9rem; color: #e0e0e0; line-height: 1.2"
                            )
                            ui.label(proj_path).style(
                                "font-size: 0.7rem; color: #555; line-height: 1.2"
                            )

        # ── Option 3: Quick chat ────────────────────────────────
        with ui.card().style(
            "background: #1a1a1a; border: 1px solid #333; cursor: pointer; "
            "width: 100%"
        ).on(
            "click",
            lambda: (_select(dialog, on_new_conversation, None)),
        ):
            ui.label("Quick chat").style(
                "font-size: 1rem; font-weight: 500; color: #e0e0e0"
            )
            ui.label("No workspace attached").style(
                "font-size: 0.85rem; color: #888; margin-top: 2px"
            )

        # ── Cancel button ───────────────────────────────────────
        with ui.row().style("width: 100%; justify-content: flex-end; margin-top: 8px"):
            ui.button("Cancel", on_click=dialog.close).props("flat").style(
                "color: #888"
            )

    return dialog


def _select(
    dialog: ui.dialog,
    callback: Callable[[str | None], None],
    value: str | None,
) -> None:
    """Close the dialog and invoke the callback with the chosen value.

    Args:
        dialog: The dialog to close.
        callback: The new-conversation callback.
        value: Workspace path or ``None``.
    """
    dialog.close()
    callback(value)
