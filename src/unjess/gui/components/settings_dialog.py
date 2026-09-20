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
import re
from pathlib import Path
from typing import TYPE_CHECKING

from nicegui import app, ui

logger = logging.getLogger(__name__)

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
    "mistral", "xai", "openrouter", "cerebras", "kimi", "qwen", "ollama", "ollama-api", "lmstudio", "llamacpp",
]

PROVIDER_LABELS: dict[str, str] = {
    "google": "Google",
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "groq": "Groq",
    "mistral": "Mistral",
    "xai": "xAI (Grok)",
    "openrouter": "OpenRouter",
    "cerebras": "Cerebras",
    "kimi": "Kimi (Moonshot)",
    "qwen": "Qwen (DashScope)",
    "ollama": "Ollama",
    "ollama-api": "Ollama API",
    "lmstudio": "LM Studio",
    "llamacpp": "llama.cpp",
}

API_KEY_ENV_NAMES: dict[str, str] = {
    "google": "GOOGLE_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "groq": "GROQ_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "xai": "XAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
    "kimi": "MOONSHOT_API_KEY",
    "qwen": "DASHSCOPE_API_KEY",
    "ollama-api": "OLLAMA_API_KEY",
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


def _fetch_models_sync(router: "ProviderRouter", provider_name: str, force_refresh: bool = False) -> list[str]:
    """Fetch model list from the router (blocking). Uses an in-memory cache."""
    if not force_refresh and provider_name in _model_cache and _model_cache[provider_name]:
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

    if not models:
        try:
            from unjess.first_run import _PROVIDER_MODELS
            if provider_name in _PROVIDER_MODELS:
                models = [m[0] for m in _PROVIDER_MODELS[provider_name]]
        except Exception:
            pass

    if models:
        _model_cache[provider_name] = models
    return models


async def _fetch_models_async(router: "ProviderRouter", provider_name: str, force_refresh: bool = False) -> list[str]:
    """Fetch model list in a background thread so the UI stays responsive."""
    if not force_refresh and provider_name in _model_cache and _model_cache[provider_name]:
        return _model_cache[provider_name]
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _fetch_models_sync, router, provider_name, force_refresh)


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
    agent: "Agent" = None,
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

    # Get existing DMD Studio token for display in the API keys section
    dmd_studio_cfg = mcp_servers_raw.get("dmd-studio", {})
    dmd_env = dmd_studio_cfg.get("env", {})
    existing_dmd_token = dmd_env.get("DMD_STUDIO_TOKEN", "")

    # ── Build dialog ──────────────────────────────────────────────────
    with ui.dialog() as dialog:
        with ui.card().classes(
            "max-w-none flex flex-col"
        ).style("width: 95vw; max-width: 800px; height: 85vh; background: #0a0a0a"):
            # ── Header ────────────────────────────────────────────────
            with ui.row().classes(
                "w-full no-wrap items-center px-4 sm:px-6 py-3 "
                "border-b border-gray-800 shrink-0"
            ):
                ui.icon("settings", size="20px").classes("text-white")
                ui.label("Settings").classes(
                    "text-base font-semibold text-gray-200 ml-2"
                )
                ui.element("div").classes("flex-grow")
                ui.button(icon="close", on_click=dialog.close).props(
                    "flat round dense"
                ).classes("text-gray-400")

            # ── Scrollable body ───────────────────────────────────────
            with ui.scroll_area().classes("flex-grow w-full"):
                with ui.column().classes("w-full px-3 sm:px-6 py-2 gap-0"):

                    # ==========================================================
                    # (a) MODEL & PROVIDER
                    # ==========================================================
                    _section_header("MODEL & PROVIDER")

                    initial_provider = settings.provider or "google"
                    # Use cache if available, otherwise start with current model
                    initial_models = _model_cache.get(initial_provider, [])

                    provider_select = ui.select(
                        label="Provider",
                        options=PROVIDER_LABELS,
                        value=initial_provider if initial_provider in PROVIDER_LABELS else "google",
                    ).props("outlined dense dark").classes("w-full mb-2")

                    model_select = ui.select(
                        label="Model",
                        options=initial_models or [settings.model] if settings.model else [],
                        value=settings.model,
                        with_input=True,
                    ).props("outlined dense dark").classes("w-full mb-2")

                    model_loading = ui.label("Loading models...").classes(
                        "text-xs text-gray-500 italic mb-2"
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
                            if settings.model in models:
                                model_select.value = settings.model
                            else:
                                model_select.value = models[0]
                        elif not models:
                            if settings.model:
                                model_select.options = [settings.model]
                                model_select.value = settings.model
                        model_select.update()

                    async def _on_provider_change(_e: object) -> None:
                        prov = provider_select.value
                        await _populate_models(prov)

                    provider_select.on_value_change(_on_provider_change)

                    # Kick off initial async fetch if not cached
                    if not initial_models:
                        asyncio.ensure_future(_populate_models(initial_provider))

                    ollama_input = ui.input(
                        label="Ollama Base URL (Local)",
                        value=settings.ollama_base_url,
                    ).props("outlined dense dark").classes("w-full mb-2")

                    ollama_api_input = ui.input(
                        label="Ollama API Base URL (Cloud)",
                        value=getattr(settings, "ollama_api_base_url", "https://api.ollama.com"),
                    ).props("outlined dense dark").classes("w-full mb-2")

                    lmstudio_input = ui.input(
                        label="LM Studio Base URL",
                        value=getattr(settings, "lmstudio_base_url", "http://localhost:1234"),
                    ).props("outlined dense dark").classes("w-full mb-4")

                    # ==========================================================
                    # (b) API KEYS
                    # ==========================================================
                    _section_header("API KEYS")

                    key_inputs: dict[str, ui.input] = {}
                    for prov, env_name in API_KEY_ENV_NAMES.items():
                        existing = settings.api_keys.get(prov, "")
                        if prov == "ollama-api" and not existing:
                            existing = settings.api_keys.get("ollama", "")
                        label_name = "Ollama API" if prov == "ollama-api" else prov.title()
                        inp = ui.input(
                            label=f"{label_name} Key",
                            value=existing,
                            password=True,
                            password_toggle_button=True,
                        ).props("outlined dense dark").classes("w-full mt-1")
                        key_inputs[prov] = inp

                        hint = f"Env: {env_name}"
                        if existing:
                            hint += f"  •  Current: {_mask_key(existing)}"
                        ui.label(hint).classes("text-xs text-gray-500 break-all mb-3 pl-1")

                    # DMD Studio Token field
                    dmd_token_input = ui.input(
                        label="DMD Studio Token",
                        value=existing_dmd_token,
                        password=True,
                        password_toggle_button=True,
                    ).props("outlined dense dark").classes("w-full mt-1")

                    dmd_hint = "Env: DMD_STUDIO_TOKEN (stored in mcp.json)"
                    if existing_dmd_token:
                        dmd_hint += f"  •  Current: {_mask_key(existing_dmd_token)}"
                    ui.label(dmd_hint).classes("text-xs text-gray-500 break-all mb-3 pl-1")

                    ui.label(
                        "Ollama (Local) runs on your machine for free without a key. Ollama API connects to Ollama cloud models using your subscription API key."
                    ).classes("text-xs text-gray-500 -mt-1 mb-4")

                    # ==========================================================
                    # (c) BEHAVIOR
                    # ==========================================================
                    _section_header("BEHAVIOR")

                    with ui.column().classes("w-full gap-3 sm:flex-row sm:gap-4"):
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

                    planning_mode_select = ui.select(
                        label="Planning Mode",
                        options=["auto", "on", "off"],
                        value=getattr(settings, "planning_mode", "auto"),
                    ).props("outlined dense dark").classes("w-full mt-2")
                    ui.label(
                        "Auto: plan on complex tasks; On: always plan; Off: never plan."
                    ).classes("text-xs text-gray-500 -mt-1")

                    compaction_toggle = ui.switch(
                        "Enable context auto-compaction",
                        value=getattr(settings, "enable_compaction", True),
                    ).classes("text-gray-400 mt-2")

                    stuck_detection_toggle = ui.switch(
                        "Enable repeating action stuck detection",
                        value=getattr(settings, "enable_stuck_detection", True),
                    ).classes("text-gray-400 mt-2")

                    context_override_input = ui.number(
                        label="Context size override (0 = auto-detect)",
                        value=getattr(settings, "context_window_override", 0),
                        min=0, max=10000000, step=1024,
                    ).props("outlined dense dark").classes("w-full mt-2")

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
                    # (f2) MOBILE COMPANION & PWA
                    # ==========================================================
                    _section_header("MOBILE COMPANION & PWA")

                    import socket
                    try:
                        _s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                        _s.connect(("8.8.8.8", 80))
                        _local_ip = _s.getsockname()[0]
                        _s.close()
                    except Exception:
                        try:
                            _local_ip = socket.gethostbyname(socket.gethostname())
                        except Exception:
                            _local_ip = "192.168.x.x"

                    _mobile_url = f"http://{_local_ip}:{getattr(settings, 'network_port', 8080)}"

                    network_toggle = ui.switch(
                        "Allow local network access (bind 0.0.0.0)",
                        value=getattr(settings, "allow_network_access", True),
                    ).classes("text-gray-400")

                    pin_auth_toggle = ui.switch(
                        "Require 4-digit PIN for remote mobile connections",
                        value=getattr(settings, "enable_pin_auth", False),
                    ).classes("text-gray-400")

                    mobile_pin_input = ui.input(
                        label="Mobile 4-Digit PIN (optional)",
                        placeholder="e.g. 1234",
                        value=getattr(settings, "mobile_pin", ""),
                    ).classes("w-full")

                    with ui.card().classes("w-full bg-[#1e1e24] border border-gray-800 p-3 rounded-lg mt-2 flex flex-col gap-2"):
                        ui.label("📱 Scan to Connect from Smartphone").classes("text-xs font-semibold text-white")
                        ui.label(f"Mobile PWA URL: {_mobile_url}").classes("text-xs text-gray-300 font-mono")
                        ui.label("Open this URL on your phone's browser or scan QR code to install Unjess as a Mobile App.").classes("text-xs text-gray-500")

                        # Lightweight QR Code canvas renderer
                        _qr_api_url = f"https://api.qrserver.com/v1/create-qr-code/?size=160x160&data={_mobile_url}&color=ffffff&bgcolor=1e1e24"
                        ui.image(_qr_api_url).classes("w-32 h-32 self-center rounded border border-gray-700 my-1")

                    # ==========================================================
                    # (g) SCHEDULED TASKS & AUTOMATION
                    # ==========================================================
                    try:
                        from unjess.gui.components.schedule_dashboard import render_schedule_dashboard
                        from unjess.scheduler import Scheduler
                        _sched_inst = getattr(app.storage.user, "scheduler", None) or Scheduler()
                        render_schedule_dashboard(_sched_inst)
                    except Exception as exc:
                        logger.error("Failed to render Schedule Dashboard: %s", exc)

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
                        'google/gemini-3.1-flash"'
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
                                    mgr = ServerManager(config_path=mcp_json, workspace_dir=Path(settings.workspace) if settings.workspace else None)
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
                                                with ui.element("th").style(
                                                    f"padding: 6px 8px; color: #9ca3af; "
                                                    f"font-weight: 600; border-bottom: "
                                                    f"1px solid #333; {_align}"
                                                ):
                                                    ui.label(col)

                                    # Body
                                    with ui.element("tbody"):
                                        for model_name, mdata in sorted_models:
                                            is_free = mdata.get("is_free", False)
                                            with ui.element("tr").style(
                                                "border-bottom: 1px solid #1f1f1f;"
                                            ):
                                                # Model name
                                                with ui.element("td").style(
                                                    "padding: 5px 8px; color: #d1d5db; "
                                                    "white-space: nowrap;"
                                                ):
                                                    ui.label(_truncate_model(model_name))
                                                # Calls
                                                with ui.element("td").style(
                                                    "padding: 5px 8px; color: #9ca3af; "
                                                    "text-align: right;"
                                                ):
                                                    ui.label(str(mdata.get("calls", 0)))
                                                # Tokens In
                                                with ui.element("td").style(
                                                    "padding: 5px 8px; color: #6ee7b7; "
                                                    "text-align: right; font-family: monospace;"
                                                ):
                                                    ui.label(f"{mdata.get('tokens_in', 0):,}")
                                                # Tokens Out
                                                with ui.element("td").style(
                                                    "padding: 5px 8px; color: #93c5fd; "
                                                    "text-align: right; font-family: monospace;"
                                                ):
                                                    ui.label(f"{mdata.get('tokens_out', 0):,}")
                                                # Thinking (optional)
                                                if has_thinking:
                                                    t = mdata.get("thinking_tokens", 0)
                                                    with ui.element("td").style(
                                                        "padding: 5px 8px; color: #c4b5fd; "
                                                        "text-align: right; font-family: monospace;"
                                                    ):
                                                        ui.label(f"{t:,}" if t else "—")
                                                # Cache (optional)
                                                if has_cache:
                                                    c = mdata.get("cache_read_tokens", 0)
                                                    with ui.element("td").style(
                                                        "padding: 5px 8px; color: #fdba74; "
                                                        "text-align: right; font-family: monospace;"
                                                    ):
                                                        ui.label(f"{c:,}" if c else "—")
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
                                                    with ui.element("td").style(
                                                        "padding: 5px 8px; color: #fbbf24; "
                                                        "text-align: right; font-family: monospace;"
                                                    ):
                                                        ui.label(f"${cost:.4f}")

                    # ==========================================================
                    # (l) SKILLS MANAGEMENT
                    # ==========================================================
                    _section_header("SKILLS")

                    skills_container = ui.column().classes("w-full gap-2 mt-2")

                    def _load_and_render_skills():
                        skills_container.clear()
                        skills = []
                        if agent and hasattr(agent, "_skill_engine") and agent._skill_engine:
                            skills = agent._skill_engine.skills

                        with skills_container:
                            if not skills:
                                ui.label("No custom skills discovered.").classes("text-xs text-gray-500 py-1")
                            else:
                                for skill in skills:
                                    # Create card for each skill
                                    with ui.card().classes("w-full p-3 gap-1").style(
                                        "background: #121212; border: 1px solid #222; border-radius: 8px;"
                                    ):
                                        with ui.row().classes("w-full items-center no-wrap"):
                                            ui.label(skill.name).classes("text-sm font-bold text-violet-400")
                                            # Display scope / type
                                            is_workspace = ".agents" in str(skill.path)
                                            scope_label = "Workspace" if is_workspace else "Global"
                                            scope_color = "#3b82f6" if is_workspace else "#8b5cf6"
                                            ui.label(scope_label).classes("text-[10px] px-1.5 py-0.5 rounded font-semibold").style(
                                                f"background: {scope_color}22; color: {scope_color};"
                                            )
                                            ui.element("div").classes("flex-grow")
                                            
                                            # Actions
                                            def make_edit_handler(s=skill):
                                                return lambda _e: _edit_skill(s)
                                            def make_delete_handler(s=skill):
                                                return lambda _e: _delete_skill(s)

                                            ui.button(icon="edit", on_click=make_edit_handler(skill)).props("flat round dense").classes("text-blue-400 w-7 h-7").tooltip("Edit Skill")
                                            ui.button(icon="delete", on_click=make_delete_handler(skill)).props("flat round dense").classes("text-red-400 w-7 h-7").tooltip("Delete Skill")
                                        
                                        ui.label(skill.description).classes("text-xs text-gray-400 line-clamp-2")
                                        if skill.trigger_patterns:
                                            ui.label(f"Triggers: {', '.join(skill.trigger_patterns)}").classes("text-[10px] text-gray-500 font-mono")

                    def _delete_skill(skill):
                        with ui.dialog() as d, ui.card().style("background: #1a1a1a; border: 1px solid #333;"):
                            ui.label(f"Delete skill '{skill.name}'?").classes("text-sm font-bold text-gray-300")
                            ui.label("This will permanently remove the skill files from disk.").classes("text-xs text-gray-500")
                            with ui.row().classes("w-full justify-end gap-2 mt-3"):
                                ui.button("Cancel", on_click=d.close).props("flat dense no-caps").classes("text-gray-500")
                                def _do_delete():
                                    try:
                                        import shutil
                                        if skill.path.is_dir():
                                            shutil.rmtree(skill.path)
                                        else:
                                            skill.path.unlink()
                                        if agent and agent._skill_engine:
                                            agent._skill_engine.discover()
                                        _load_and_render_skills()
                                        ui.notify(f"Skill '{skill.name}' deleted", type="positive", position="top")
                                    except Exception as exc:
                                        ui.notify(f"Delete failed: {exc}", type="negative", position="top")
                                    d.close()
                                ui.button("Delete", on_click=_do_delete).props("unelevated dense no-caps").classes("bg-red-500 text-white")
                        d.open()

                    def _edit_skill(skill):
                        _show_skill_form(skill)

                    def _add_skill():
                        _show_skill_form(None)

                    def _show_skill_form(skill=None):
                        is_edit = skill is not None
                        title = f"Edit Skill: {skill.name}" if is_edit else "Add New Skill"
                        
                        # Load instructions text
                        initial_instr = ""
                        initial_triggers = ""
                        initial_desc = ""
                        initial_name = ""
                        initial_scope = "Global"
                        
                        if is_edit:
                            initial_name = skill.name
                            initial_desc = skill.description
                            initial_triggers = ", ".join(skill.trigger_patterns)
                            is_workspace = ".agents" in str(skill.path)
                            initial_scope = "Workspace" if is_workspace else "Global"
                            if agent and agent._skill_engine:
                                raw_instr = agent._skill_engine.load(skill.name) or ""
                                initial_instr = re.sub(r'^---\s*\n.*?\n---\s*\n?', '', raw_instr, flags=re.DOTALL).strip()

                        with ui.dialog() as form_dialog, ui.card().classes("w-[500px] max-w-none").style("background: #121212; border: 1px solid #333;"):
                            ui.label(title).classes("text-sm font-bold text-violet-400 mb-2")
                            
                            name_input = ui.input("Name", value=initial_name).props("outlined dense dark").classes("w-full mb-2")
                            if is_edit:
                                name_input.props("readonly")
                            
                            desc_input = ui.input("Description", value=initial_desc).props("outlined dense dark").classes("w-full mb-2")
                            triggers_input = ui.input("Triggers (comma separated)", value=initial_triggers).props("outlined dense dark").classes("w-full mb-2")
                            
                            scope_select = ui.select(
                                label="Scope",
                                options=["Global", "Workspace"],
                                value=initial_scope
                            ).props("outlined dense dark").classes("w-full mb-2")
                            if is_edit:
                                scope_select.props("readonly")
                                
                            instr_input = ui.textarea(
                                label="Instructions (Markdown)",
                                value=initial_instr
                            ).props("outlined dense dark").classes("w-full h-48 font-mono mb-3")
                            
                            with ui.row().classes("w-full justify-end gap-2"):
                                ui.button("Cancel", on_click=form_dialog.close).props("flat dense no-caps").classes("text-gray-500")
                                
                                def _save():
                                    name = (name_input.value or "").strip()
                                    desc = (desc_input.value or "").strip()
                                    triggers_raw = (triggers_input.value or "").strip()
                                    scope = scope_select.value
                                    raw_instr = instr_input.value or ""
                                    clean_instr = re.sub(r'^---\s*\n.*?\n---\s*\n?', '', raw_instr, flags=re.DOTALL).strip()
                                    
                                    if not name or not desc or not clean_instr:
                                        ui.notify("Name, Description, and Instructions are required", type="warning")
                                        return
                                        
                                    triggers = [t.strip() for t in triggers_raw.split(",") if t.strip()]
                                    
                                    # Form SKILL.md content
                                    import yaml
                                    frontmatter = {
                                        "name": name,
                                        "description": desc,
                                    }
                                    if triggers:
                                        frontmatter["triggers"] = triggers
                                        
                                    yaml_block = yaml.dump(frontmatter, sort_keys=False).strip()
                                    file_content = f"---\n{yaml_block}\n---\n\n{clean_instr}"
                                    
                                    try:
                                        # Determine path
                                        if is_edit:
                                            target_dir = skill.path if skill.path.is_dir() else skill.path.parent
                                        else:
                                            ws_path = settings.workspace or "."
                                            if scope == "Workspace":
                                                target_dir = Path(ws_path) / ".agents" / "skills" / name
                                            else:
                                                target_dir = Path.home() / ".unjess" / "skills" / name
                                        
                                        target_dir.mkdir(parents=True, exist_ok=True)
                                        skill_md = target_dir / "SKILL.md"
                                        skill_md.write_text(file_content, encoding="utf-8")
                                        
                                        if agent and agent._skill_engine:
                                            # Add root if not exists
                                            agent._skill_engine.add_root(target_dir.parent)
                                            agent._skill_engine.discover()
                                            
                                        _load_and_render_skills()
                                        ui.notify(f"Skill '{name}' saved successfully", type="positive", position="top")
                                        form_dialog.close()
                                    except Exception as exc:
                                        ui.notify(f"Save failed: {exc}", type="negative", position="top")
                                        
                                ui.button("Save", on_click=_save).props("unelevated dense no-caps").classes("bg-violet-600 text-white")
                        form_dialog.open()

                    ui.button("Add Custom Skill", icon="add", on_click=_add_skill).props(
                        "flat dense no-caps"
                    ).classes("text-violet-400 mt-1")

                    _load_and_render_skills()

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
                    # Safeguard settings.model from being overwritten with empty string
                    new_model = model_select.value or ""
                    if new_model:
                        settings.model = new_model
                    elif not settings.model:
                        settings.model = "gemini-3.1-flash"
                    settings.ollama_base_url = (
                        ollama_input.value or "http://localhost:11434"
                    )
                    settings.ollama_api_base_url = (
                        ollama_api_input.value or "https://api.ollama.com"
                    )
                    settings.lmstudio_base_url = (
                        lmstudio_input.value or "http://localhost:1234"
                    ).strip()

                    # (b) API Keys
                    for prov, inp in key_inputs.items():
                        val = (inp.value or "").strip()
                        if val:
                            settings.api_keys[prov] = val
                        elif prov in settings.api_keys:
                            # Allow user to clear their key
                            del settings.api_keys[prov]

                    # (c) Behavior
                    settings.max_iterations = int(max_iter_input.value or 50)
                    settings.command_timeout = int(cmd_timeout_input.value or 30)
                    settings.confirm_commands = confirm_toggle.value
                    settings.planning_mode = planning_mode_select.value or "auto"
                    settings.enable_compaction = compaction_toggle.value
                    settings.enable_stuck_detection = stuck_detection_toggle.value
                    settings.context_window_override = int(context_override_input.value or 0)

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

                    # (f2) Mobile Companion
                    settings.allow_network_access = network_toggle.value
                    settings.enable_pin_auth = pin_auth_toggle.value
                    settings.mobile_pin = (mobile_pin_input.value or "").strip()

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
                    new_dmd_token = dmd_token_input.value.strip()
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
                        
                        env = row.get("env", {})
                        if name == "dmd-studio" and new_dmd_token:
                            env["DMD_STUDIO_TOKEN"] = new_dmd_token
                            
                        mcp_out[name] = {
                            "command": cmd,
                            "args": args_list,
                            "env": env,
                            "enabled": row["enabled"],
                        }
                    # Also make sure we update it if dmd-studio was not modified in rows but is in mcp_out
                    if "dmd-studio" in mcp_out and new_dmd_token:
                        mcp_out["dmd-studio"]["env"]["DMD_STUDIO_TOKEN"] = new_dmd_token
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
