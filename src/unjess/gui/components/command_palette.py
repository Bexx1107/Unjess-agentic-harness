"""Command palette — Cmd+K / Ctrl+K searchable command dialog.

Shows a filterable list of slash commands in a modal dialog.
Typing narrows the list; clicking or pressing Enter selects and
executes the command.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from nicegui import ui


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CommandEntry:
    """A single entry in the command palette.

    Attributes:
        name: Command name (e.g. ``"/help"``).
        description: One-line description.
        shortcut: Optional keyboard shortcut hint.
    """

    name: str
    description: str
    shortcut: str = ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_command_palette(
    commands: dict[str, str],
    on_select: Callable[[str], None],
) -> ui.dialog:
    """Create a command palette dialog.

    The palette is a modal with a search input at the top and a
    scrollable list of matching commands below.  It is designed to be
    opened via a keyboard shortcut (Ctrl+K / Cmd+K).

    Args:
        commands: Mapping of command name → description.
        on_select: Callback invoked with the selected command name.

    Returns:
        The ``ui.dialog`` instance so the caller can bind it to a
        keyboard shortcut.
    """
    entries = [
        CommandEntry(name=name, description=desc)
        for name, desc in commands.items()
    ]

    dialog = ui.dialog().props("position='top'")

    with dialog, ui.card().classes(
        "command-palette q-pa-none"
    ).style(
        "width:500px;max-height:420px;background:#1e1e2e;"
        "border:1px solid rgba(var(--accent-rgb),0.3);border-radius:12px;"
        "margin-top:80px;overflow:hidden"
    ):
        # --- Search input ---
        with ui.row().classes(
            "w-full items-center q-px-md q-py-sm no-wrap"
        ).style(
            "border-bottom:1px solid rgba(255,255,255,0.08)"
        ):
            ui.icon("search").classes("text-grey-6")
            search_input = ui.input(
                placeholder="Type a command…",
            ).props(
                'borderless dense dark input-class="text-white"'
            ).classes("flex-grow q-ml-sm")

        # --- Results list ---
        results_container = ui.scroll_area().classes(
            "w-full"
        ).style("max-height:340px")

        with results_container:
            results_column = ui.column().classes(
                "w-full q-py-xs gap-none"
            )

    # --- Rendering & filtering ---

    def _render_entries(filter_text: str = "") -> None:
        """Re-render the command list based on the filter.

        Args:
            filter_text: Text to filter commands by.
        """
        results_column.clear()
        query = filter_text.strip().lower()

        matching = [
            e for e in entries
            if not query
            or query in e.name.lower()
            or query in e.description.lower()
        ]

        with results_column:
            if not matching:
                ui.label("No matching commands").classes(
                    "text-sm text-grey-6 q-pa-md text-center w-full"
                )
                return

            for entry in matching:
                _render_command_row(entry, on_select, dialog)

    def _render_command_row(
        entry: CommandEntry,
        callback: Callable[[str], None],
        dlg: ui.dialog,
    ) -> None:
        """Render a single command row.

        Args:
            entry: Command entry data.
            callback: Selection callback.
            dlg: Dialog to close after selection.
        """
        with ui.row().classes(
            "w-full items-center q-px-md q-py-sm cursor-pointer command-row"
        ).style(
            "transition:background 0.15s"
        ).on(
            "click",
            lambda _e=entry: (_select(callback, dlg, _e.name)),
        ).on(
            "mouseenter",
            lambda e: e.sender.style("background:rgba(var(--accent-rgb),0.12)"),
        ).on(
            "mouseleave",
            lambda e: e.sender.style("background:transparent"),
        ):
            ui.label(entry.name).classes(
                "text-sm text-white text-weight-medium"
            ).style("font-family:monospace;min-width:100px")
            ui.label(entry.description).classes(
                "text-xs text-grey-6 q-ml-sm"
            )
            ui.space()
            if entry.shortcut:
                ui.badge(entry.shortcut).props(
                    "dense outline color='grey-8'"
                ).classes("text-xs")

    def _select(
        callback: Callable[[str], None],
        dlg: ui.dialog,
        name: str,
    ) -> None:
        """Handle command selection.

        Args:
            callback: Caller's on_select callback.
            dlg: Dialog to close.
            name: Selected command name.
        """
        dlg.close()
        callback(name)

    # --- Wire up search filtering ---
    search_input.on(
        "update:model-value",
        lambda e: _render_entries(e.args or ""),
    )

    # Initial render
    _render_entries()

    # Focus search input when dialog opens
    dialog.on("show", lambda: search_input.run_method("focus"))

    # Enter key selects first match
    def _on_enter() -> None:
        query = (search_input.value or "").strip().lower()
        matching = [
            e for e in entries
            if not query
            or query in e.name.lower()
            or query in e.description.lower()
        ]
        if matching:
            _select(on_select, dialog, matching[0].name)

    search_input.on("keydown.enter", _on_enter)

    # Escape closes
    search_input.on("keydown.escape", dialog.close)

    return dialog
