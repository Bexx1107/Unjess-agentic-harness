"""Spawn dialog — modal for approving or cancelling subagent spawns.

Shown when the orchestrator wants to spawn a subagent.  The user can
review the agent type, role, prompt, and override the model or max-turn
count before confirming.  The dialog blocks the agent thread (via
``GUIInput``) until the user responds.
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

def render_spawn_dialog(
    input_handler: _InputHandler,
    router: Any,  # ProviderRouter — typed as Any to avoid circular imports
) -> ui.dialog:
    """Create (and return) a spawn-approval dialog bound to *input_handler*.

    The dialog opens automatically whenever ``input_handler.pending_type``
    is ``'spawn'``.  The caller is responsible for watching
    ``input_handler.has_pending`` and calling ``dialog.open()``.

    Buttons:
    - **Spawn** → resolves with ``{'approved': True, 'model': ..., 'max_turns': ...}``
    - **Cancel** → resolves with ``{'approved': False}``

    Args:
        input_handler: The GUI input handler with pending spawn data.
        router: A ``ProviderRouter`` instance used for ``list_models()``.

    Returns:
        The ``ui.dialog`` instance (caller can ``.open()`` / ``.close()``).
    """

    dialog = ui.dialog().props("persistent")

    # State holders — mutable containers so closures can read current values
    _selected_model: dict[str, str] = {"value": ""}
    _selected_max_turns: dict[str, int] = {"value": 10}

    with dialog, ui.card().classes(
        "spawn-dialog q-pa-md"
    ).style(
        "min-width:420px;max-width:600px;background:#1e1e2e;"
        "border:1px solid rgba(var(--accent-rgb),0.3);border-radius:12px"
    ):

        # --- Header ---
        with ui.row().classes("w-full items-center q-pb-sm").style(
            "border-bottom:1px solid rgba(255,255,255,0.08)"
        ):
            ui.icon("smart_toy").classes("text-blue-5").style("font-size:20px")
            ui.label("Spawn Subagent").classes(
                "text-subtitle1 text-weight-medium text-white q-ml-sm"
            )

        # --- Body (dynamically populated) ---
        body_column = ui.column().classes("w-full q-py-sm gap-sm")

        # --- Button row ---
        with ui.row().classes("w-full justify-end q-pt-sm gap-sm"):
            cancel_btn = ui.button(
                "Cancel", icon="close", color="red-9",
            ).props("flat dense").classes("text-caption")

            spawn_btn = ui.button(
                "Spawn", icon="play_arrow", color="green-7",
            ).props("unelevated dense").classes("text-caption")

    # --- Populate body when dialog opens ---

    def _populate() -> None:
        """Fill the dialog body with pending spawn data."""
        body_column.clear()
        data = input_handler.pending_data

        type_name: str = data.get("type_name", "unknown")
        role: str = data.get("role", "")
        prompt: str = data.get("prompt", "")
        suggested_model: str = data.get("suggested_model", "")
        tools: list[str] = data.get("tools", [])
        max_turns: int = data.get("max_turns", 10)

        # Truncate prompt for display
        display_prompt = prompt
        if len(display_prompt) > 200:
            display_prompt = display_prompt[:200] + "…"

        # Resolve available models from router
        try:
            available_models: list[str] = router.list_models()
        except Exception:
            available_models = []

        # Determine initial model selection
        if suggested_model and suggested_model in available_models:
            initial_model = suggested_model
        elif available_models:
            initial_model = available_models[0]
        else:
            initial_model = suggested_model or ""

        _selected_model["value"] = initial_model
        _selected_max_turns["value"] = max_turns

        with body_column:
            # Agent type
            with ui.row().classes("items-center"):
                ui.label("Type:").classes("text-xs text-grey-6")
                ui.label(type_name).classes(
                    "text-sm text-white text-weight-medium q-ml-xs"
                ).style("font-family:monospace")

            # Role
            if role:
                with ui.row().classes("items-center"):
                    ui.label("Role:").classes("text-xs text-grey-6")
                    ui.label(role).classes(
                        "text-sm text-grey-4 q-ml-xs"
                    )

            # Prompt (truncated)
            if display_prompt:
                ui.label("Prompt:").classes("text-xs text-grey-6 q-pt-xs")
                ui.html(
                    f'<pre style="margin:0;padding:6px 10px;'
                    f'background:rgba(0,0,0,0.3);border-radius:6px;'
                    f'color:#e2e8f0;font-size:12px;overflow-x:auto;'
                    f'white-space:pre-wrap;word-break:break-word;'
                    f'font-family:monospace">{html.escape(display_prompt)}</pre>'
                ).classes("w-full")

            # Tools (read-only labels)
            if tools:
                ui.label("Tools:").classes("text-xs text-grey-6 q-pt-xs")
                with ui.row().classes("w-full gap-xs").style("flex-wrap:wrap"):
                    for tool_name in tools:
                        ui.label(tool_name).classes(
                            "text-xs text-grey-4"
                        ).style(
                            "font-family:monospace;padding:2px 8px;"
                            "background:rgba(255,255,255,0.06);"
                            "border-radius:4px"
                        )

            # Model dropdown
            ui.label("Model:").classes("text-xs text-grey-6 q-pt-sm")
            if available_models:
                model_select = ui.select(
                    options=available_models,
                    value=initial_model,
                ).classes("w-full").props(
                    'dense dark outlined color="blue-5"'
                ).style("font-size:13px")
            else:
                # Fallback: editable text input when no models are listed
                model_select = ui.input(
                    value=initial_model,
                    placeholder="model name",
                ).classes("w-full").props(
                    'dense dark outlined color="blue-5"'
                ).style("font-size:13px")

            def _on_model_change(e: Any) -> None:
                _selected_model["value"] = e.value

            model_select.on_value_change(_on_model_change)

            # Max turns
            ui.label("Max turns:").classes("text-xs text-grey-6 q-pt-sm")
            turns_input = ui.number(
                value=max_turns,
                min=1,
                max=50,
                step=1,
            ).classes("w-full").props(
                'dense dark outlined color="blue-5"'
            ).style("font-size:13px;max-width:120px")

            def _on_turns_change(e: Any) -> None:
                _selected_max_turns["value"] = int(e.value) if e.value else 10

            turns_input.on_value_change(_on_turns_change)

    # --- Wire up buttons ---

    def _spawn() -> None:
        input_handler.resolve({
            "approved": True,
            "model": _selected_model["value"],
            "max_turns": _selected_max_turns["value"],
        })
        dialog.close()

    def _cancel() -> None:
        input_handler.resolve({"approved": False})
        dialog.close()

    spawn_btn.on_click(_spawn)
    cancel_btn.on_click(_cancel)

    # Expose populate so the caller can refresh before opening
    dialog.populate = _populate  # type: ignore[attr-defined]

    return dialog
