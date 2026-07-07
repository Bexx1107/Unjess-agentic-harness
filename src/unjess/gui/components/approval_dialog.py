"""Approval dialog — modal for approving, denying, or auto-allowing tool calls.

Shown when the agent requests permission to execute a tool that requires
user approval (e.g. ``run_command``, ``write_file``).  The dialog blocks
the agent thread (via ``GUIInput``) until the user responds.
"""

from __future__ import annotations

import html
from typing import TYPE_CHECKING, Any, Protocol

from nicegui import ui


# ---------------------------------------------------------------------------
# Lightweight protocol for the input handler
# ---------------------------------------------------------------------------

class _InputHandler(Protocol):
    """Minimal duck-type for the GUI input handler."""

    @property
    def has_pending(self) -> bool: ...

    @property
    def pending_type(self) -> str: ...

    @property
    def pending_data(self) -> dict[str, Any]: ...

    def resolve(self, result: Any) -> None: ...


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_approval_dialog(input_handler: _InputHandler) -> ui.dialog:
    """Create (and return) an approval dialog bound to *input_handler*.

    The dialog opens automatically whenever ``input_handler.pending_type``
    is ``'approval'``.  The caller is responsible for watching
    ``input_handler.has_pending`` and calling ``dialog.open()``.

    Buttons:
    - **Allow** → ``input_handler.resolve((True, False))``
    - **Always Allow** → ``input_handler.resolve((True, True))``
    - **Deny** → ``input_handler.resolve((False, False))``

    Args:
        input_handler: The GUI input handler with pending approval data.

    Returns:
        The ``ui.dialog`` instance (caller can ``.open()`` / ``.close()``).
    """

    dialog = ui.dialog().props("persistent")

    with dialog, ui.card().classes(
        "approval-dialog q-pa-md"
    ).style(
        "min-width:420px;max-width:600px;background:#1e1e2e;"
        "border:1px solid rgba(var(--accent-rgb),0.3);border-radius:12px"
    ):

        # --- Header ---
        with ui.row().classes("w-full items-center q-pb-sm").style(
            "border-bottom:1px solid rgba(255,255,255,0.08)"
        ):
            ui.icon("shield").classes("text-amber-5").style("font-size:20px")
            ui.label("Permission Required").classes(
                "text-subtitle1 text-weight-medium text-white q-ml-sm"
            )

        # --- Body (dynamically populated) ---
        body_column = ui.column().classes("w-full q-py-sm gap-sm")

        # --- Button row ---
        with ui.row().classes("w-full justify-end q-pt-sm gap-sm"):
            deny_btn = ui.button(
                "Deny", icon="close", color="red-9",
            ).props("flat dense").classes("text-caption")

            always_btn = ui.button(
                "Always Allow", icon="verified_user", color="blue-7",
            ).props("flat dense").classes("text-caption")

            allow_btn = ui.button(
                "Allow", icon="check", color="green-7",
            ).props("unelevated dense").classes("text-caption")

    # --- Populate body when dialog opens ---

    def _populate() -> None:
        """Fill the dialog body with pending approval data."""
        body_column.clear()
        data = input_handler.pending_data

        tool_name = data.get("tool_name", data.get("name", "unknown"))
        args = data.get("args", {})
        command = args.get("CommandLine", args.get("command", ""))
        cwd = args.get("Cwd", args.get("cwd", ""))
        target_file = args.get("TargetFile", args.get("path", ""))

        with body_column:
            # Tool name
            with ui.row().classes("items-center"):
                ui.label("Tool:").classes("text-xs text-grey-6")
                ui.label(tool_name).classes(
                    "text-sm text-white text-weight-medium q-ml-xs"
                ).style("font-family:monospace")

            # Command (for run_command)
            if command:
                ui.label("Command:").classes("text-xs text-grey-6 q-pt-xs")
                ui.html(
                    f'<pre style="margin:0;padding:6px 10px;'
                    f'background:rgba(0,0,0,0.3);border-radius:6px;'
                    f'color:#e2e8f0;font-size:12px;overflow-x:auto;'
                    f'font-family:monospace">{html.escape(command)}</pre>'
                ).classes("w-full")

            # Working directory
            if cwd:
                with ui.row().classes("items-center"):
                    ui.label("Directory:").classes("text-xs text-grey-6")
                    ui.label(cwd).classes(
                        "text-xs text-grey-5 q-ml-xs"
                    ).style("font-family:monospace")

            # Target file (for write/edit tools)
            if target_file:
                with ui.row().classes("items-center"):
                    ui.label("File:").classes("text-xs text-grey-6")
                    ui.label(target_file).classes(
                        "text-xs text-grey-5 q-ml-xs"
                    ).style("font-family:monospace")

            # Full args (collapsed)
            if args:
                _render_args_detail(args)

    def _render_args_detail(args: dict) -> None:
        """Show full tool arguments in a collapsed expansion panel."""
        with ui.expansion("Show all arguments").classes(
            "w-full text-xs text-grey-7"
        ).props("dense"):
            for key, value in args.items():
                val_str = str(value)
                if len(val_str) > 200:
                    val_str = val_str[:200] + "…"
                with ui.row().classes("items-start q-py-none"):
                    ui.label(f"{key}:").classes(
                        "text-xs text-grey-6"
                    ).style("min-width:80px;font-family:monospace")
                    ui.label(val_str).classes("text-xs text-grey-5").style(
                        "font-family:monospace;word-break:break-all"
                    )

    # --- Wire up buttons ---

    def _approve() -> None:
        input_handler.resolve((True, False))
        dialog.close()

    def _always_approve() -> None:
        input_handler.resolve((True, True))
        dialog.close()

    def _deny() -> None:
        input_handler.resolve((False, False))
        dialog.close()

    allow_btn.on_click(_approve)
    always_btn.on_click(_always_approve)
    deny_btn.on_click(_deny)

    # Expose populate so the caller can refresh before opening
    dialog.populate = _populate  # type: ignore[attr-defined]

    return dialog
