"""Settings dialog — scrollable popup with ALL config options.

Opens as a modal overlay from the chat page. Covers:
- Model & Provider (dynamic model lists via ProviderRouter)
- API Keys (masked display, per-provider editing)
- Behavior (max iterations, command timeout, confirm commands)
- Display (verbosity, show diffs/cost/stats)
- Budget & Limits (max cost, max tokens)
- Free Mode (toggle + description)
- RAG (toggle + description)
- Fallback Chain (comma-separated provider/model)
- MCP Servers (read/write ~/.unjess/mcp.json)
- Ignore Patterns (textarea, one per line)
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from nicegui import ui

if TYPE_CHECKING:
    from unjess.config import Settings
    from unjess.gui.state import AppState
    from unjess.llm.router import ProviderRouter

log = logging.getLogger(__name__)

# In-memory cache so switching providers is instant after the first fetch
_model_cache: dict[str, list[str]] = {}

# ── Provider & key constants ──────────────────────────────────────────────

PROVIDERS: list[str] = [
    "google", "openai", "anthropic", "groq",
    "mistral", "xai", "openrouter", "cerebras", "ollama",
]

API_KEY_ENV_NAMES: dict[str, str] = {
    "google": "GOOGLE_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "groq": "GROQ_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "xai": "XAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
}

_MCP_PATH = Path.home() / ".unjess" / "mcp.json"


# ── Helpers ───────────────────────────────────────────────────────────────

def _mask_key(key: str) -> str:
    """Mask an API key, showing only the first 4 and last 4 characters."""
    if not key:
        return ""
    if len(key) <= 10:
        return "*" * len(key)
    return key[:4] + "*" * (len(key) - 8) + key[-4:]


def _section_header(text: str) -> None:
    """Render a styled section header inside the settings dialog."""
    ui.label(text).classes(
        "text-xs font-bold tracking-wider text-gray-500 "
        "dark:text-gray-500 pt-4 pb-2 border-b border-gray-700 w-full"
    )


def _usage_card(label: str, value: str, icon: str, color: str) -> None:
    """Render a compact usage stat card."""
    with ui.column().classes("items-center gap-0 px-3 py-2 rounded-lg").style(
        f"background: rgba(255,255,255,0.03); border: 1px solid #222; "
        f"min-width: 80px;"
    ):
        ui.icon(icon, size="18px").style(f"color: {color};")
        ui.label(value).classes("text-sm font-bold").style(f"color: {color};")
        ui.label(label).classes("text-[10px] text-gray-500")


def _truncate_model(name: str, max_len: int = 28) -> str:
    """Shorten a model name for table display."""
    if len(name) <= max_len:
        return name
    return name[:max_len - 2] + "…"


def _fetch_models_sync(router: "ProviderRouter", provider_name: str) -> list[str]:
    """Fetch model list from the router (blocking). Uses an in-memory cache."""
    if provider_name in _model_cache:
        return _model_cache[provider_name]

    models: list[str] = []
    try:
        models = router.list_models(provider_name)
    except Exception:
        log.debug("Failed to list models for %s, trying list_all_models", provider_name)
        try:
            all_models = router.list_all_models()
            models = all_models.get(provider_name, [])
        except Exception:
            log.debug("list_all_models also failed for %s", provider_name)

    if models:
        _model_cache[provider_name] = models
    return models


async def _fetch_models_async(router: "ProviderRouter", provider_name: str) -> list[str]:
    """Fetch model list in a background thread so the UI stays responsive."""
    if provider_name in _model_cache:
        return _model_cache[provider_name]
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _fetch_models_sync, router, provider_name)


def _load_mcp_config() -> dict[str, dict]:
    """Load MCP server configuration from ~/.unjess/mcp.json."""
    if not _MCP_PATH.exists():
        return {}
    try:
        data = json.loads(_MCP_PATH.read_text(encoding="utf-8"))
        return data.get("mcpServers", {})
    except Exception:
        log.warning("Failed to read MCP config from %s", _MCP_PATH)
        return {}


def _save_mcp_config(servers: dict[str, dict]) -> None:
    """Persist MCP server configuration to ~/.unjess/mcp.json."""
    _MCP_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"mcpServers": servers}
    _MCP_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ── Dialog factory ────────────────────────────────────────────────────────

def create_settings_dialog(
    settings: "Settings",
    state: "AppState",
    router: "ProviderRouter",
) -> ui.dialog:
    """Create a scrollable settings dialog with all config options.

    Args:
        settings: The mutable Settings instance.
        state: App state for live updates.
        router: Provider router for dynamic model lists.

    Returns:
        The dialog instance (call ``.open()`` to show).
    """
    from unjess.config import save_config

    # ── MCP state (mutable list of dicts for the UI) ──────────────────
    mcp_servers_raw = _load_mcp_config()
    mcp_rows: list[dict] = []
    for name, cfg in mcp_servers_raw.items():
        mcp_rows.append({
            "name": name,
            "command": cfg.get("command", ""),
            "args": ", ".join(cfg.get("args", [])),
            "env": cfg.get("env", {}),
            "enabled": cfg.get("enabled", True),
        })

    # ── Build dialog ──────────────────────────────────────────────────
    with ui.dialog() as dialog:
        with ui.card().classes(
            "max-w-none flex flex-col"
        ).style("width: 80vw; height: 80vh; background: #0a0a0a"):
            # ── Header ────────────────────────────────────────────────
            with ui.row().classes(
                "w-full no-wrap items-center px-6 py-3 "
                "border-b border-gray-800 shrink-0"
            ):
                ui.icon("settings", size="20px").classes("text-violet-500")
                ui.label("Settings").classes(
                    "text-base font-semibold text-gray-200 ml-2"
                )
                ui.element("div").classes("flex-grow")
                ui.button(icon="close", on_click=dialog.close).props(
                    "flat round dense"
                ).classes("text-gray-600")

            # ── Scrollable body ───────────────────────────────────────
            with ui.scroll_area().classes("flex-grow w-full"):
                with ui.column().classes("w-full px-6 py-2 gap-0"):

                    # ==========================================================
                    # (a) MODEL & PROVIDER
                    # ==========================================================
                    _section_header("MODEL & PROVIDER")

                    initial_provider = settings.provider or "google"
                    # Use cache if available, otherwise start with current model
                    initial_models = _model_cache.get(initial_provider, [])

                    provider_select = ui.select(
                        label="Provider",
                        options=PROVIDERS,
                        value=initial_provider,
                    ).props("outlined dense dark").classes("w-full")

                    model_select = ui.select(
                        label="Model",
                        options=initial_models or [settings.model] if settings.model else [],
                        value=settings.model,
                        with_input=True,
                    ).props("outlined dense dark").classes("w-full")

                    model_loading = ui.label("Loading models...").classes(
                        "text-xs text-gray-500 italic"
                    )
                    model_loading.set_visibility(not initial_models)

                    async def _populate_models(provider: str) -> None:
                        """Fetch models in background and update the dropdown."""
                        model_loading.set_visibility(True)
                        try:
                            models = await _fetch_models_async(router, provider)
                        except Exception:
                            models = []
                        model_loading.set_visibility(False)
                        model_select.options = models
                        if models and model_select.value not in models:
                            # Keep current value if it's valid, otherwise pick first
                            if settings.model in models:
                                model_select.value = settings.model
                            else:
                                model_select.value = models[0]
                        model_select.update()

                    async def _on_provider_change(_e: object) -> None:
                        prov = provider_select.value
                        await _populate_models(prov)

                    provider_select.on_value_change(_on_provider_change)

                    # Kick off initial async fetch if not cached
                    if not initial_models:
                        asyncio.ensure_future(_populate_models(initial_provider))

                    ollama_input = ui.input(
                        label="Ollama Base URL",
                        value=settings.ollama_base_url,
                    ).props("outlined dense dark").classes("w-full")

                    # ==========================================================
                    # (b) API KEYS
                    # ==========================================================
                    _section_header("API KEYS")

                    key_inputs: dict[str, ui.input] = {}
                    for prov, env_name in API_KEY_ENV_NAMES.items():
                        existing = settings.api_keys.get(prov, "")
                        hint = f"Env: {env_name}"
                        if existing:
                            hint += f"  •  Current: {_mask_key(existing)}"
                        inp = ui.input(
                            label=f"{prov.title()} API Key",
                            value=existing,
                            password=True,
                            password_toggle_button=True,
                        ).props(f'outlined dense dark hint="{hint}"').classes("w-full")
                        key_inputs[prov] = inp

                    ui.label(
                        "Ollama does not require an API key."
                    ).classes("text-xs text-gray-500 -mt-1")

                    # ==========================================================
                    # (c) BEHAVIOR
                    # ==========================================================
                    _section_header("BEHAVIOR")

                    with ui.row().classes("w-full gap-4"):
                        max_iter_input = ui.number(
                            label="Max iterations per turn",
                            value=settings.max_iterations,
                            min=1, max=500, step=1,
                        ).props("outlined dense dark").classes("w-full")

                        cmd_timeout_input = ui.number(
                            label="Command timeout (seconds)",
                            value=settings.command_timeout,
                            min=5, max=600, step=5,
                        ).props("outlined dense dark").classes("w-full")

                    confirm_toggle = ui.switch(
                        "Confirm before running commands",
                        value=settings.confirm_commands,
                    ).classes("text-gray-400")

                    # ==========================================================
                    # (d) DISPLAY
                    # ==========================================================
                    _section_header("DISPLAY")

                    verbosity_select = ui.select(
                        label="Verbosity",
                        options=["quiet", "normal", "verbose"],
                        value=settings.verbosity,
                    ).props("outlined dense dark").classes("w-full")

                    with ui.row().classes("w-full gap-6"):
                        show_diffs_toggle = ui.switch(
                            "Show diffs", value=settings.show_diffs,
                        ).classes("text-gray-400")
                        show_cost_toggle = ui.switch(
                            "Show cost", value=settings.show_cost,
                        ).classes("text-gray-400")
                        show_stats_toggle = ui.switch(
                            "Show stats", value=settings.show_stats,
                        ).classes("text-gray-400")

                    # Accent color picker
                    ui.label("Accent Color").classes("text-xs text-gray-500 mt-2")
                    _PRESETS = [
                        ("#7c3aed", "Purple"),
                        ("#2563eb", "Blue"),
                        ("#0891b2", "Cyan"),
                        ("#059669", "Green"),
                        ("#d97706", "Amber"),
                        ("#dc2626", "Red"),
                        ("#db2777", "Pink"),
                    ]
                    _current_accent = getattr(settings, "accent_color", "#666666")

                    # Live preview + hex input row
                    with ui.row().classes("items-center gap-3"):
                        color_preview = ui.element("div").style(
                            f"background: {_current_accent}; "
                            f"width: 36px; height: 36px; "
                            f"border-radius: 8px; border: 2px solid rgba(255,255,255,0.2); "
                            f"flex-shrink: 0; transition: background 0.2s;"
                        )
                        accent_input = ui.input(
                            label="Hex color",
                            value=_current_accent,
                            on_change=lambda e: color_preview.style(
                                f"background: {e.value}; "
                                f"width: 36px; height: 36px; "
                                f"border-radius: 8px; border: 2px solid rgba(255,255,255,0.2); "
                                f"flex-shrink: 0; transition: background 0.2s;"
                            ),
                        ).props("outlined dense dark").classes("w-40")

                    # Swatch row
                    with ui.row().classes("w-full gap-2 items-center"):
                        for hex_val, label_text in _PRESETS:
                            _is_selected = _current_accent == hex_val

                            def _click_swatch(_e: object, _hex: str = hex_val) -> None:
                                accent_input.set_value(_hex)
                                color_preview.style(
                                    f"background: {_hex}; "
                                    f"width: 36px; height: 36px; "
                                    f"border-radius: 8px; border: 2px solid rgba(255,255,255,0.2); "
                                    f"flex-shrink: 0; transition: background 0.2s;"
                                )

                            ui.element("div").style(
                                f"background: {hex_val}; "
                                f"width: 28px; height: 28px; "
                                f"border-radius: 50%; cursor: pointer; "
                                f"border: 2px solid {'white' if _is_selected else 'transparent'}; "
                                f"flex-shrink: 0;"
                            ).tooltip(label_text).on("click", _click_swatch)

                        ui.color_input(
                            label="Custom",
                            value=_current_accent,
                            on_change=lambda e: (
                                accent_input.set_value(e.value),
                                color_preview.style(
                                    f"background: {e.value}; "
                                    f"width: 36px; height: 36px; "
                                    f"border-radius: 8px; border: 2px solid rgba(255,255,255,0.2); "
                                    f"flex-shrink: 0; transition: background 0.2s;"
                                ),
                            ),
                        ).props("outlined dense dark").classes("w-28").style(
                            "margin-left: 8px"
                        )

                    # ==========================================================
                    # (e) BUDGET & LIMITS
                    # ==========================================================
                    _section_header("BUDGET & LIMITS")

                    with ui.row().classes("w-full gap-4"):
                        max_cost_input = ui.number(
                            label="Max cost per session ($)",
                            value=settings.max_cost_per_session,
                            min=0, step=0.5, format="%.2f",
                        ).props("outlined dense dark").classes("w-full")

                        max_tokens_input = ui.number(
                            label="Max tokens per message",
                            value=settings.max_tokens_per_message,
                            min=0, step=1000,
                        ).props("outlined dense dark").classes("w-full")

                    ui.label(
                        "Set to 0 for unlimited."
                    ).classes("text-xs text-gray-500 -mt-1")

                    # ==========================================================
                    # (f) FREE MODE
                    # ==========================================================
                    _section_header("FREE MODE")

                    free_toggle = ui.switch(
                        "Enable free mode (rotate free-tier API keys)",
                        value=settings.free_mode_enabled,
                    ).classes("text-gray-400")
                    ui.label(
                        "Uses free-tier providers to avoid costs. "
                        "May have rate limits and lower quality."
                    ).classes("text-xs text-gray-500 -mt-1")

                    # ==========================================================
                    # (g) RAG
                    # ==========================================================
                    _section_header("RAG")

                    rag_toggle = ui.switch(
                        "Enable RAG (semantic code search)",
                        value=settings.enable_rag,
                    ).classes("text-gray-400")
                    ui.label(
                        "Uses embeddings for context-aware retrieval "
                        "across your workspace."
                    ).classes("text-xs text-gray-500 -mt-1")

                    # Knowledge graph viewer
                    from pathlib import Path as _Path
                    import json as _json
                    import time as _time

                    _kg_path = _Path.home() / ".unjess" / "memory" / "knowledge_graph.json"
                    _kg_entities: list[dict] = []
                    _kg_rels: list[dict] = []
                    if _kg_path.exists():
                        try:
                            _kg_data = _json.loads(_kg_path.read_text(encoding="utf-8"))
                            _kg_entities = _kg_data.get("entities", [])
                            _kg_rels = _kg_data.get("relationships", [])
                            if isinstance(_kg_entities, dict):
                                _kg_entities = list(_kg_entities.values())
                        except Exception:
                            pass

                    with ui.expansion(
                        f"Knowledge Graph  ({len(_kg_entities)} entities, "
                        f"{len(_kg_rels)} relationships)",
                        icon="hub",
                    ).classes("w-full text-gray-400").style(
                        "background: #161616; border-radius: 8px; margin-top: 4px"
                    ):
                        ui.label(f"📁 {_kg_path}").style(
                            "font-size: 0.7rem; color: #555; "
                            "word-break: break-all; padding: 4px 0 6px 0"
                        )
                        if _kg_entities:
                            with ui.column().style(
                                "width: 100%; max-height: 220px; "
                                "overflow-y: auto; gap: 2px; padding: 4px 0"
                            ):
                                # Sort by mention count descending
                                _sorted = sorted(
                                    _kg_entities,
                                    key=lambda e: e.get("mention_count", 0),
                                    reverse=True,
                                )
                                for ent in _sorted:
                                    _name = ent.get("name", "?")
                                    _etype = ent.get("entity_type", "?")
                                    _mentions = ent.get("mention_count", 0)
                                    _last = ent.get("last_seen", 0)
                                    _ago = ""
                                    if _last:
                                        _diff = _time.time() - _last
                                        if _diff < 3600:
                                            _ago = f"{int(_diff/60)}m ago"
                                        elif _diff < 86400:
                                            _ago = f"{int(_diff/3600)}h ago"
                                        else:
                                            _ago = f"{int(_diff/86400)}d ago"

                                    _type_colors = {
                                        "file": "#4a9eff",
                                        "function": "#a78bfa",
                                        "concept": "#f59e0b",
                                        "library": "#10b981",
                                        "pattern": "#ec4899",
                                        "rule": "#ef4444",
                                    }
                                    _color = _type_colors.get(_etype, "#888")

                                    with ui.row().style(
                                        "width: 100%; padding: 4px 8px; "
                                        "align-items: center; gap: 8px; "
                                        "border-bottom: 1px solid #1e1e1e"
                                    ):
                                        ui.element("div").style(
                                            f"width: 8px; height: 8px; "
                                            f"border-radius: 50%; "
                                            f"background: {_color}; flex-shrink: 0"
                                        )
                                        ui.label(_name).style(
                                            "font-size: 0.85rem; color: #e0e0e0; "
                                            "flex: 1; overflow: hidden; "
                                            "text-overflow: ellipsis; white-space: nowrap"
                                        )
                                        ui.label(_etype).style(
                                            f"font-size: 0.7rem; color: {_color}; "
                                            f"padding: 1px 6px; "
                                            f"border: 1px solid {_color}40; "
                                            f"border-radius: 4px"
                                        )
                                        ui.label(f"×{_mentions}").style(
                                            "font-size: 0.75rem; color: #888; "
                                            "min-width: 30px; text-align: right"
                                        )
                                        if _ago:
                                            ui.label(_ago).style(
                                                "font-size: 0.7rem; color: #555; "
                                                "min-width: 45px; text-align: right"
                                            )

                            # Clear button
                            def _clear_kg() -> None:
                                try:
                                    _kg_path.unlink(missing_ok=True)
                                    ui.notify(
                                        "Knowledge graph cleared",
                                        type="positive", position="top-right",
                                    )
                                except Exception as exc:
                                    ui.notify(
                                        f"Failed: {exc}",
                                        type="negative", position="top-right",
                                    )

                            ui.button(
                                "Clear Knowledge Graph", icon="delete_sweep",
                                on_click=_clear_kg,
                            ).props("flat dense no-caps").style(
                                "color: #ef4444; margin-top: 6px; font-size: 0.8rem"
                            )
                        else:
                            ui.label("No entities tracked yet.").style(
                                "color: #555; font-size: 0.85rem; padding: 8px 0"
                            )

                    # ==========================================================
                    # (h) FALLBACK CHAIN
                    # ==========================================================
                    _section_header("FALLBACK CHAIN")

                    fallback_input = ui.input(
                        label="Fallback chain (comma-separated provider/model)",
                        value=", ".join(settings.fallback_chain),
                    ).props("outlined dense dark").classes("w-full")
                    ui.label(
                        'e.g. "groq/llama-3.3-70b-versatile, '
                        'google/gemini-2.5-flash"'
                    ).classes("text-xs text-gray-500 -mt-1")

                    # ==========================================================
                    # (i) MCP SERVERS
                    # ==========================================================
                    _section_header("MCP SERVERS")

                    mcp_container = ui.column().classes("w-full gap-2")

                    def _render_mcp_rows() -> None:
                        """Rebuild the MCP rows UI from the mutable list."""
                        mcp_container.clear()
                        with mcp_container:
                            if not mcp_rows:
                                ui.label("No MCP servers configured.").classes(
                                    "text-xs text-gray-500 italic"
                                )
                                return
                            for idx, row in enumerate(mcp_rows):
                                _render_single_mcp_row(idx, row)

                    def _render_single_mcp_row(idx: int, row: dict) -> None:
                        """Render one MCP server entry."""
                        with ui.card().classes(
                            "w-full p-3 border border-gray-800"
                        ).style("background: #1a1a1a"):
                            with ui.row().classes("w-full items-center gap-2 no-wrap"):
                                ui.label(row["name"]).classes(
                                    "text-sm font-semibold text-gray-300 flex-grow"
                                )
                                enabled_sw = ui.switch(
                                    "Enabled", value=row["enabled"],
                                ).classes("text-gray-400")

                                def _toggle_enabled(
                                    _e: object, _idx: int = idx, _sw: ui.switch = enabled_sw,
                                ) -> None:
                                    mcp_rows[_idx]["enabled"] = _sw.value

                                enabled_sw.on_value_change(_toggle_enabled)

                                def _edit(
                                    _e: object, _idx: int = idx,
                                ) -> None:
                                    _open_mcp_edit_dialog(_idx)

                                ui.button(icon="edit", on_click=_edit).props(
                                    "flat round dense"
                                ).classes("text-gray-500")

                                def _remove(
                                    _e: object, _idx: int = idx,
                                ) -> None:
                                    mcp_rows.pop(_idx)
                                    _render_mcp_rows()

                                ui.button(icon="delete", on_click=_remove).props(
                                    "flat round dense"
                                ).classes("text-red-400")

                            ui.label(
                                f"cmd: {row['command']}  "
                                f"args: [{row['args']}]"
                            ).classes("text-xs text-gray-500 mt-1")

                    def _open_mcp_edit_dialog(idx: int) -> None:
                        """Open a sub-dialog to edit one MCP server entry."""
                        row = mcp_rows[idx]
                        with ui.dialog() as edit_dlg, ui.card().classes(
                            ""
                        ).style("width: 500px; background: #0a0a0a"):
                            ui.label("Edit MCP Server").classes(
                                "text-sm font-bold text-gray-300 pb-2"
                            )
                            name_inp = ui.input(
                                label="Server name", value=row["name"],
                            ).props("outlined dense dark").classes("w-full")
                            cmd_inp = ui.input(
                                label="Command", value=row["command"],
                                placeholder="e.g. node, npx, python",
                            ).props("outlined dense dark").classes("w-full")
                            ui.label(
                                "The executable to run (not a display name)."
                            ).classes("text-xs text-gray-500 -mt-1")
                            args_inp = ui.input(
                                label="Args (comma-separated)",
                                value=row["args"],
                                placeholder="e.g. E:\\path\\to\\index.js",
                            ).props("outlined dense dark").classes("w-full")

                            with ui.row().classes("w-full justify-end gap-2 pt-2"):
                                ui.button(
                                    "Cancel", on_click=edit_dlg.close,
                                ).props("flat").classes("text-gray-500")

                                def _apply_edit(
                                    _e: object,
                                    _idx: int = idx,
                                    _name: ui.input = name_inp,
                                    _cmd: ui.input = cmd_inp,
                                    _args: ui.input = args_inp,
                                ) -> None:
                                    mcp_rows[_idx]["name"] = _name.value.strip()
                                    mcp_rows[_idx]["command"] = _cmd.value.strip()
                                    mcp_rows[_idx]["args"] = _args.value.strip()
                                    edit_dlg.close()
                                    _render_mcp_rows()

                                ui.button(
                                    "Apply", on_click=_apply_edit,
                                ).props("unelevated").classes(
                                    "bg-violet-600 text-white"
                                )
                        edit_dlg.open()

                    def _add_mcp_server(_e: object) -> None:
                        """Append a blank MCP server entry and re-render."""
                        mcp_rows.append({
                            "name": f"server-{len(mcp_rows) + 1}",
                            "command": "node",
                            "args": "",
                            "env": {},
                            "enabled": True,
                        })
                        _render_mcp_rows()

                    _render_mcp_rows()

                    with ui.row().classes("w-full gap-2 mt-1"):
                        ui.button(
                            "Add Server", icon="add", on_click=_add_mcp_server,
                        ).props("flat dense").classes(
                            "text-violet-400"
                        )

                        def _refresh_mcp_tools(_e: object) -> None:
                            """Re-discover tools from all connected MCP servers."""
                            try:
                                from unjess.mcp.server_manager import ServerManager
                                from pathlib import Path
                                mcp_json = Path.home() / ".unjess" / "mcp.json"
                                if mcp_json.exists():
                                    mgr = ServerManager(config_path=mcp_json)
                                    # Connect to all enabled servers
                                    mgr.connect_all()
                                    connected = mgr.connected_servers
                                    if not connected:
                                        ui.notify(
                                            "No MCP servers connected",
                                            type="warning", position="top",
                                        )
                                        return
                                    # Refresh tools and register on agent
                                    if state.settings and hasattr(state, '_agent_ref'):
                                        results = mgr.refresh_tools(
                                            tool_registry=state._agent_ref.tool_registry
                                        )
                                    else:
                                        results = mgr.refresh_tools()
                                    total = sum(results.values())
                                    ui.notify(
                                        f"Refreshed {total} MCP tools from "
                                        f"{len(results)} server(s)",
                                        type="positive", position="top",
                                    )
                                else:
                                    ui.notify(
                                        "No mcp.json found",
                                        type="warning", position="top",
                                    )
                            except Exception as exc:
                                ui.notify(
                                    f"MCP refresh failed: {exc}",
                                    type="negative", position="top",
                                )

                        ui.button(
                            "Refresh Tools", icon="refresh",
                            on_click=_refresh_mcp_tools,
                        ).props("flat dense").classes(
                            "text-blue-400"
                        ).tooltip(
                            "Re-discover tools from all connected MCP servers"
                        )

                    # ==========================================================
                    # (j) IGNORE PATTERNS
                    # ==========================================================
                    _section_header("IGNORE PATTERNS")

                    ignore_textarea = ui.textarea(
                        label="Ignore patterns (one per line)",
                        value="\n".join(settings.ignore_patterns),
                    ).props("outlined dense dark").classes("w-full font-mono")
                    ui.label(
                        "Glob patterns for files/dirs the agent should ignore."
                    ).classes("text-xs text-gray-500 -mt-1")

                    # ==========================================================
                    # (k) USAGE & COSTS
                    # ==========================================================
                    _section_header("USAGE & COSTS")

                    # Pull data from CostTracker
                    _ct = None
                    if state.conv_logger and hasattr(state.conv_logger, "cost_tracker"):
                        _ct = state.conv_logger.cost_tracker

                    if _ct is None:
                        ui.label("No usage data available").classes(
                            "text-xs text-gray-500 py-2"
                        )
                    else:
                        from unjess.conversation_logger import CostTracker

                        # ── Overview cards ──────────────────────────────
                        with ui.row().classes("w-full flex-wrap gap-2 py-1"):
                            _usage_card(
                                "Session",
                                f"${_ct.total_cost:.4f}" if _ct.total_cost > 0 else "FREE",
                                "payments",
                                "#10b981" if _ct.total_cost == 0 else "#f59e0b",
                            )
                            try:
                                daily = CostTracker.get_daily_cost()
                                _usage_card("Today", f"${daily:.4f}", "today", "#60a5fa")
                            except Exception:
                                _usage_card("Today", "$0.00", "today", "#60a5fa")
                            try:
                                monthly = CostTracker.get_monthly_cost()
                                _usage_card("Month", f"${monthly:.2f}", "calendar_month", "#a78bfa")
                            except Exception:
                                _usage_card("Month", "$0.00", "calendar_month", "#a78bfa")
                            _usage_card(
                                "LLM Calls", str(_ct.llm_calls), "smart_toy", "#f472b6"
                            )
                            _usage_card(
                                "Tool Calls", str(_ct.tool_calls), "build", "#94a3b8"
                            )

                        # ── Session totals ──────────────────────────────
                        with ui.row().classes("w-full gap-4 py-1"):
                            ui.label(
                                f"Total: ↓ {_ct.total_tokens_in:,} in  ·  "
                                f"↑ {_ct.total_tokens_out:,} out"
                            ).classes("text-xs text-gray-400")
                            if _ct.total_thinking_tokens:
                                ui.label(
                                    f"🧠 {_ct.total_thinking_tokens:,} thinking"
                                ).classes("text-xs text-purple-400")
                            if _ct.total_cache_read_tokens:
                                ui.label(
                                    f"🔄 {_ct.total_cache_read_tokens:,} cached"
                                ).classes("text-xs text-orange-400")

                        # ── Per-model breakdown table ──────────────────
                        per_model = _ct._per_model
                        if per_model:
                            ui.label("Per-Model Breakdown").classes(
                                "text-xs font-semibold text-gray-400 pt-2"
                            )

                            # Check if any model has thinking/cache tokens
                            has_thinking = any(
                                m.get("thinking_tokens", 0) for m in per_model.values()
                            )
                            has_cache = any(
                                m.get("cache_read_tokens", 0) for m in per_model.values()
                            )

                            # Sort by cost descending
                            sorted_models = sorted(
                                per_model.items(),
                                key=lambda x: x[1].get("cost", 0),
                                reverse=True,
                            )

                            # Build table
                            cols = ["Model", "Calls", "Tokens In", "Tokens Out"]
                            if has_thinking:
                                cols.append("Thinking")
                            if has_cache:
                                cols.append("Cache")
                            cols.append("Cost")

                            with ui.element("div").classes("w-full").style(
                                "overflow-x: auto;"
                            ):
                                with ui.element("table").classes("w-full").style(
                                    "border-collapse: collapse; font-size: 0.75rem;"
                                ):
                                    # Header
                                    with ui.element("thead"):
                                        with ui.element("tr"):
                                            for col in cols:
                                                _align = (
                                                    "text-align: left;"
                                                    if col == "Model"
                                                    else "text-align: right;"
                                                )
                                                ui.element("th").text(col).style(
                                                    f"padding: 6px 8px; color: #9ca3af; "
                                                    f"font-weight: 600; border-bottom: "
                                                    f"1px solid #333; {_align}"
                                                )

                                    # Body
                                    with ui.element("tbody"):
                                        for model_name, mdata in sorted_models:
                                            is_free = mdata.get("is_free", False)
                                            with ui.element("tr").style(
                                                "border-bottom: 1px solid #1f1f1f;"
                                            ):
                                                # Model name
                                                ui.element("td").text(
                                                    _truncate_model(model_name)
                                                ).style(
                                                    "padding: 5px 8px; color: #d1d5db; "
                                                    "white-space: nowrap;"
                                                )
                                                # Calls
                                                ui.element("td").text(
                                                    str(mdata.get("calls", 0))
                                                ).style(
                                                    "padding: 5px 8px; color: #9ca3af; "
                                                    "text-align: right;"
                                                )
                                                # Tokens In
                                                ui.element("td").text(
                                                    f"{mdata.get('tokens_in', 0):,}"
                                                ).style(
                                                    "padding: 5px 8px; color: #6ee7b7; "
                                                    "text-align: right; font-family: monospace;"
                                                )
                                                # Tokens Out
                                                ui.element("td").text(
                                                    f"{mdata.get('tokens_out', 0):,}"
                                                ).style(
                                                    "padding: 5px 8px; color: #93c5fd; "
                                                    "text-align: right; font-family: monospace;"
                                                )
                                                # Thinking (optional)
                                                if has_thinking:
                                                    t = mdata.get("thinking_tokens", 0)
                                                    ui.element("td").text(
                                                        f"{t:,}" if t else "—"
                                                    ).style(
                                                        "padding: 5px 8px; color: #c4b5fd; "
                                                        "text-align: right; font-family: monospace;"
                                                    )
                                                # Cache (optional)
                                                if has_cache:
                                                    c = mdata.get("cache_read_tokens", 0)
                                                    ui.element("td").text(
                                                        f"{c:,}" if c else "—"
                                                    ).style(
                                                        "padding: 5px 8px; color: #fdba74; "
                                                        "text-align: right; font-family: monospace;"
                                                    )
                                                # Cost
                                                if is_free:
                                                    with ui.element("td").style(
                                                        "padding: 5px 8px; text-align: right;"
                                                    ):
                                                        ui.label("FREE").classes(
                                                            "text-[10px] font-bold px-1 rounded"
                                                        ).style(
                                                            "background: rgba(16,185,129,0.15); "
                                                            "color: #34d399; display: inline-block;"
                                                        )
                                                else:
                                                    cost = mdata.get("cost", 0)
                                                    ui.element("td").text(
                                                        f"${cost:.4f}"
                                                    ).style(
                                                        "padding: 5px 8px; color: #fbbf24; "
                                                        "text-align: right; font-family: monospace;"
                                                    )

            # ── Footer ────────────────────────────────────────────────
            with ui.row().classes(
                "w-full no-wrap justify-end gap-2 px-6 py-3 "
                "border-t border-gray-800 shrink-0"
            ):
                ui.button(
                    "Cancel", on_click=dialog.close,
                ).props("flat").classes("text-gray-500")

                def _save() -> None:
                    """Persist all settings, reload providers if needed."""
                    old_provider = settings.provider
                    old_keys = dict(settings.api_keys)
                    old_accent = getattr(settings, "accent_color", "#666666")

                    # (a) Model & Provider
                    settings.provider = provider_select.value or ""
                    settings.model = model_select.value or ""
                    settings.ollama_base_url = (
                        ollama_input.value or "http://localhost:11434"
                    )

                    # (b) API Keys
                    for prov, inp in key_inputs.items():
                        val = (inp.value or "").strip()
                        if val:
                            settings.api_keys[prov] = val

                    # (c) Behavior
                    settings.max_iterations = int(max_iter_input.value or 50)
                    settings.command_timeout = int(cmd_timeout_input.value or 30)
                    settings.confirm_commands = confirm_toggle.value

                    # (d) Display
                    settings.verbosity = verbosity_select.value or "normal"
                    settings.show_diffs = show_diffs_toggle.value
                    settings.show_cost = show_cost_toggle.value
                    settings.show_stats = show_stats_toggle.value
                    _new_accent = (accent_input.value or "#666666").strip()
                    if _new_accent.startswith("#") and len(_new_accent) in (4, 7):
                        settings.accent_color = _new_accent

                    # (e) Budget & Limits
                    settings.max_cost_per_session = float(max_cost_input.value or 0)
                    settings.max_tokens_per_message = int(
                        max_tokens_input.value or 0
                    )

                    # (f) Free Mode
                    settings.free_mode_enabled = free_toggle.value

                    # (g) RAG
                    settings.enable_rag = rag_toggle.value

                    # (h) Fallback Chain
                    fallback_raw = fallback_input.value or ""
                    settings.fallback_chain = [
                        s.strip() for s in fallback_raw.split(",") if s.strip()
                    ]

                    # (j) Ignore Patterns
                    raw_patterns = ignore_textarea.value or ""
                    settings.ignore_patterns = [
                        p.strip() for p in raw_patterns.splitlines() if p.strip()
                    ]

                    # Persist config.yaml + keys.yaml
                    save_config(settings)

                    # (i) MCP Servers — build dict and write separately
                    mcp_out: dict[str, dict] = {}
                    _mcp_warnings: list[str] = []
                    for row in mcp_rows:
                        name = row["name"].strip()
                        if not name:
                            continue
                        cmd = row["command"].strip()
                        if " " in cmd:
                            _mcp_warnings.append(
                                f"Server '{name}': command '{cmd}' contains "
                                f"spaces — did you mean 'node'?"
                            )
                        args_str = row["args"]
                        args_list = [
                            a.strip() for a in args_str.split(",") if a.strip()
                        ] if args_str else []
                        mcp_out[name] = {
                            "command": cmd,
                            "args": args_list,
                            "env": row.get("env", {}),
                            "enabled": row["enabled"],
                        }
                    _save_mcp_config(mcp_out)
                    for _w in _mcp_warnings:
                        ui.notify(_w, type="warning", position="top-right")

                    # Update live state
                    state.model = settings.model
                    state.provider = settings.provider
                    state.dirty = True

                    # Reload providers when keys or provider changed
                    keys_changed = old_keys != settings.api_keys
                    provider_changed = old_provider != settings.provider
                    if keys_changed or provider_changed:
                        try:
                            router.reload_providers()
                        except Exception:
                            log.warning("Failed to reload providers")

                    # Live-reload accent color
                    if _new_accent != old_accent:
                        from unjess.gui.accent import accent_css
                        ui.add_css(accent_css(settings.accent_color))
                        ui.colors(primary=settings.accent_color)
                        # Full page reload to pick up all accent references
                        ui.run_javascript("setTimeout(() => location.reload(), 300)")

                    ui.notify(
                        "Settings saved", type="positive", position="top-right",
                    )
                    dialog.close()

                ui.button(
                    "Save Changes", icon="save", on_click=_save,
                ).props("unelevated").classes(
                    "bg-violet-600 text-white font-semibold "
                    "rounded-lg px-5"
                )

    return dialog
