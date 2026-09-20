"""Onboarding page — first-run setup wizard.

Route: ``/onboarding``

A clean, screen-by-screen setup wizard that guides new users through:
1. Welcome & Introduction
2. AI Provider Selection
3. API Key Configuration
4. Model Selection
5. Completion & Launch

On completion the settings are saved and the user is redirected to the chat page.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, TYPE_CHECKING

from nicegui import ui

if TYPE_CHECKING:
    from unjess.config import Settings

logger = logging.getLogger(__name__)

# Provider cards for the selection step
_PROVIDER_CARDS: list[dict[str, str]] = [
    {
        "value": "google",
        "name": "Google Gemini",
        "desc": "Gemini 3.1 Flash — generous free tier available",
        "badge": "FREE TIER",
        "badge_color": "#10b981",
        "icon": "🌐",
    },
    {
        "value": "ollama",
        "name": "Ollama (Local)",
        "desc": "Run open models locally — no key or internet needed",
        "badge": "FREE / LOCAL",
        "badge_color": "#10b981",
        "icon": "💻",
    },
    {
        "value": "ollama-api",
        "name": "Ollama API",
        "desc": "Access Ollama cloud models with your subscription",
        "badge": "CLOUD API",
        "badge_color": "#8b5cf6",
        "icon": "☁️",
    },
    {
        "value": "openrouter",
        "name": "OpenRouter",
        "desc": "50+ free models — unified gateway to open AI",
        "badge": "FREE TIER",
        "badge_color": "#10b981",
        "icon": "🔀",
    },
    {
        "value": "groq",
        "name": "Groq",
        "desc": "Llama 3.3 70B — ultra-fast free tier",
        "badge": "FREE TIER",
        "badge_color": "#10b981",
        "icon": "⚡",
    },
    {
        "value": "cerebras",
        "name": "Cerebras",
        "desc": "ZAI GLM 4.7 — ultra-high throughput free tier",
        "badge": "FREE TIER",
        "badge_color": "#10b981",
        "icon": "🧠",
    },
    {
        "value": "mistral",
        "name": "Mistral",
        "desc": "Codestral — state-of-the-art coding model",
        "badge": "FREE TIER",
        "badge_color": "#10b981",
        "icon": "🇫🇷",
    },
    {
        "value": "openai",
        "name": "OpenAI",
        "desc": "GPT-4o, o4-mini — flagship intelligence",
        "badge": "PAID",
        "badge_color": "#6b7280",
        "icon": "🤖",
    },
    {
        "value": "anthropic",
        "name": "Anthropic",
        "desc": "Claude 3.7 / 3.5 Sonnet — exceptional for code",
        "badge": "PAID",
        "badge_color": "#6b7280",
        "icon": "🏛️",
    },
    {
        "value": "xai",
        "name": "xAI (Grok)",
        "desc": "Grok 4.1 Fast — ultra-low latency & cost",
        "badge": "PAID",
        "badge_color": "#6b7280",
        "icon": "🚀",
    },
]

# Models per provider (value, label, description)
_MODELS_PER_PROVIDER: dict[str, list[tuple[str, str, str]]] = {
    "google": [
        ("gemini-3.1-flash", "Gemini 3.1 Flash", "Ultra-fast & generous free tier (Recommended)"),
        ("gemini-3.1-pro", "Gemini 3.1 Pro", "Deep reasoning and high capability"),
    ],
    "ollama": [
        ("llama3.1", "Llama 3.1", "Meta open weights — runs locally on your machine"),
        ("codestral-v2", "Codestral", "Mistral code generation model — local"),
        ("qwen2.5-coder", "Qwen 2.5 Coder", "Alibaba code model — local"),
    ],
    "ollama-api": [
        ("llama3.3", "Llama 3.3", "Meta flagship model — Ollama Cloud API"),
        ("deepseek-r1", "DeepSeek R1", "High-performance reasoning — Ollama Cloud"),
        ("qwen2.5-coder:32b", "Qwen 2.5 Coder 32B", "Alibaba code model — Ollama Cloud"),
    ],
    "openrouter": [
        ("openrouter/free", "Auto-Free", "Automatically routes to best available free model"),
        ("nvidia/nemotron-3-ultra:free", "Nemotron 3 Ultra", "FREE — 1M context coding model"),
        ("qwen/qwen3-coder:free", "Qwen 3 Coder", "FREE — code generation, 1M context"),
    ],
    "groq": [
        ("llama-3.3-70b-versatile", "Llama 3.3 70B", "Fast & FREE — great general coding model"),
        ("llama-3.1-8b-instant", "Llama 3.1 8B", "Ultra-fast lightweight model"),
    ],
    "cerebras": [
        ("zai-glm-4.7", "ZAI GLM 4.7", "FREE — ultra-fast inference, 131K context"),
        ("llama-3.1-70b", "Llama 3.1 70B", "FREE — Meta open model on Cerebras hardware"),
    ],
    "mistral": [
        ("codestral-latest", "Codestral", "FREE for code — specialized code generation"),
        ("mistral-small-latest", "Mistral Small", "FREE tier — fast general purpose"),
    ],
    "openai": [
        ("gpt-4o-mini", "GPT-4o mini", "Fast & cheap — excellent default"),
        ("gpt-4o", "GPT-4o", "Most capable — great for complex tasks"),
        ("o4-mini", "o4-mini", "Reasoning model — best for hard math/logic"),
    ],
    "anthropic": [
        ("claude-sonnet-4-20250514", "Claude Sonnet 4", "Best balance of speed and code quality"),
        ("claude-3-5-haiku-20241022", "Claude 3.5 Haiku", "Fast & lightweight"),
    ],
    "xai": [
        ("grok-4.1-fast", "Grok 4.1 Fast", "Fast and economical"),
        ("grok-4.3", "Grok 4.3", "Flagship model"),
    ],
}

# Help URLs for getting API keys
_KEY_HELP_URLS: dict[str, str] = {
    "google": "https://aistudio.google.com/apikey",
    "openai": "https://platform.openai.com/api-keys",
    "anthropic": "https://console.anthropic.com/settings/keys",
    "groq": "https://console.groq.com/keys",
    "mistral": "https://console.mistral.ai/api-keys",
    "xai": "https://console.x.ai/",
    "openrouter": "https://openrouter.ai/settings/keys",
    "cerebras": "https://cloud.cerebras.ai/",
    "ollama": "",  # no key needed
    "ollama-api": "https://ollama.com",
}

STEP_TITLES = [
    (1, "Welcome"),
    (2, "Provider"),
    (3, "API Key"),
    (4, "Model"),
    (5, "Ready"),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def setup_onboarding_page(
    state: "Any" = None,
    settings: "Settings | None" = None,
    router: "Any" = None,
) -> None:
    """Build the first-run onboarding wizard at route ``/onboarding``.

    Presents a clean screen-by-screen wizard rather than a tall vertical list,
    guaranteeing full scrollability and proper viewport visibility on all screens.

    Args:
        state: AppState instance (passed by app.py).
        settings: Mutable Settings instance to populate.
        router: ProviderRouter instance (reloaded upon completion).
    """
    ui.dark_mode(True)

    # Force page scrollability and custom styling for the wizard
    ui.add_head_html("""
    <style>
        html, body {
            height: auto !important;
            min-height: 100% !important;
            overflow-y: auto !important;
            overflow-x: hidden !important;
            background-color: #0a0a0a !important;
            margin: 0;
            padding: 0;
        }
        .wizard-card {
            background: #141414;
            border: 1px solid #242424;
            border-radius: 14px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        }
        .prov-card {
            background: #191919;
            border: 1.5px solid #282828;
            border-radius: 10px;
            padding: 10px 12px;
            cursor: pointer;
            transition: all 0.15s ease;
        }
        .prov-card:hover {
            border-color: #7c3aed;
            background: #1f1b29;
            transform: translateY(-1px);
        }
        .prov-card.active {
            border-color: #a855f7 !important;
            background: rgba(168, 85, 247, 0.12) !important;
            box-shadow: 0 0 12px rgba(168, 85, 247, 0.2);
        }
        .model-card {
            background: #191919;
            border: 1.5px solid #282828;
            border-radius: 10px;
            padding: 12px 14px;
            cursor: pointer;
            transition: all 0.15s ease;
        }
        .model-card:hover {
            border-color: #7c3aed;
            background: #1f1b29;
        }
        .model-card.active {
            border-color: #a855f7 !important;
            background: rgba(168, 85, 247, 0.12) !important;
        }
        .step-circle {
            width: 26px;
            height: 26px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 11px;
            font-weight: 700;
            transition: all 0.2s ease;
        }
        .step-circle.active {
            background: #a855f7;
            color: #ffffff;
            box-shadow: 0 0 10px rgba(168, 85, 247, 0.5);
        }
        .step-circle.completed {
            background: #22c55e;
            color: #ffffff;
        }
        .step-circle.pending {
            background: #262626;
            color: #737373;
        }
        .step-connector {
            flex: 1;
            height: 2px;
            background: #262626;
            margin: 0 6px;
        }
        .step-connector.completed {
            background: #a855f7;
        }
    </style>
    """)

    wiz: dict[str, str] = {
        "provider": "google",
        "api_key": "",
        "model": "gemini-3.1-flash",
        "ollama_base_url": getattr(settings, "ollama_base_url", "http://localhost:11434") if settings else "http://localhost:11434",
    }

    current_step_holder: dict[str, int] = {"step": 1}

    # Outer scrollable page wrapper
    with ui.column().classes(
        "w-full min-h-screen items-center justify-start py-8 px-4 sm:px-6"
    ).style("background: #0a0a0a; overflow-y: auto;"):

        with ui.column().classes("w-full max-w-xl mx-auto items-center gap-4"):

            # ── Brand Header ───────────────────────────────────────────
            with ui.row().classes("items-center gap-2 mb-1"):
                ui.label("🟣").style("font-size: 26px;")
                ui.label("njss").classes("text-2xl font-bold tracking-wider").style("color: #c084fc;")
                ui.label("•").classes("text-gray-600 text-sm")
                ui.label("Setup Assistant").classes("text-gray-400 text-sm")

            # ── Step Progress Bar ──────────────────────────────────────
            progress_bar = ui.column().classes("w-full items-center gap-2 mb-2")

            def _render_progress() -> None:
                progress_bar.clear()
                curr = current_step_holder["step"]
                with progress_bar:
                    # Dots and connectors
                    with ui.row().classes("w-full items-center justify-between px-3"):
                        for i, (step_num, title) in enumerate(STEP_TITLES):
                            if step_num < curr:
                                status_class = "completed"
                                label_content = "✓"
                            elif step_num == curr:
                                status_class = "active"
                                label_content = str(step_num)
                            else:
                                status_class = "pending"
                                label_content = str(step_num)

                            ui.label(label_content).classes(f"step-circle {status_class}")

                            if i < len(STEP_TITLES) - 1:
                                line_class = "completed" if step_num < curr else ""
                                ui.element("div").classes(f"step-connector {line_class}")

                    # Step title text
                    curr_title = next(t for n, t in STEP_TITLES if n == curr)
                    ui.label(f"Step {curr} of {len(STEP_TITLES)}: {curr_title}").classes(
                        "text-xs font-medium text-purple-400"
                    )

            # ── Screen Container Card ──────────────────────────────────
            card = ui.column().classes("w-full wizard-card p-6 sm:p-8 gap-5")

            def _render_screen() -> None:
                card.clear()
                curr = current_step_holder["step"]
                _render_progress()

                with card:
                    # ==========================================================
                    # SCREEN 1: WELCOME
                    # ==========================================================
                    if curr == 1:
                        ui.label("Welcome to Unjess (njss)").classes(
                            "text-xl font-bold text-white"
                        )
                        ui.label(
                            "Your personal, autonomous AI pair programmer that runs locally or connects to your favorite cloud models."
                        ).classes("text-sm text-gray-400 -mt-2 leading-relaxed")

                        # Features list
                        with ui.column().classes("w-full gap-3 py-2"):
                            features = [
                                ("📂", "Workspace Native", "Inspects, creates, and refactors project files directly in your repository."),
                                ("⚡", "Local or Cloud", "Free local models via Ollama or state-of-the-art models from Google, Anthropic, OpenAI, and Groq."),
                                ("🔒", "Private & Secure", "All keys and conversation transcripts remain sandboxed on your computer."),
                            ]
                            for icon, title, desc in features:
                                with ui.row().classes("items-start gap-3 p-2.5 rounded-lg").style(
                                    "background: rgba(255,255,255,0.02); border: 1px solid #222;"
                                ):
                                    ui.label(icon).classes("text-lg")
                                    with ui.column().classes("gap-0"):
                                        ui.label(title).classes("text-xs font-semibold text-gray-200")
                                        ui.label(desc).classes("text-[11px] text-gray-400")

                        ui.label("Let's take 60 seconds to configure your default model.").classes(
                            "text-xs text-purple-300 italic"
                        )

                        # Navigation
                        with ui.row().classes("w-full justify-end pt-2"):
                            def _next_step1() -> None:
                                current_step_holder["step"] = 2
                                _render_screen()

                            ui.button(
                                "Get Started →",
                                on_click=_next_step1,
                            ).props('unelevated color="deep-purple-6"').classes("w-full py-2 font-semibold")

                    # ==========================================================
                    # SCREEN 2: CHOOSE PROVIDER
                    # ==========================================================
                    elif curr == 2:
                        ui.label("Choose your AI Provider").classes(
                            "text-xl font-bold text-white"
                        )
                        ui.label(
                            "Select the provider you want to use. You can switch providers or add more keys anytime in Settings."
                        ).classes("text-sm text-gray-400 -mt-2")

                        grid = ui.grid(columns=2).classes("w-full gap-2.5 max-h-[360px] overflow-y-auto pr-1")

                        def _select_provider(val: str) -> None:
                            wiz["provider"] = val
                            # Default to first model for provider
                            models = _MODELS_PER_PROVIDER.get(val, [])
                            if models:
                                wiz["model"] = models[0][0]
                            _render_screen()

                        with grid:
                            for c in _PROVIDER_CARDS:
                                val = c["value"]
                                is_sel = wiz["provider"] == val
                                card_classes = f"prov-card {'active' if is_sel else ''}"

                                with ui.card().classes(card_classes) as p_card:
                                    p_card.on("click", lambda _e, v=val: _select_provider(v))
                                    with ui.row().classes("w-full items-center justify-between no-wrap mb-1"):
                                        with ui.row().classes("items-center gap-2 no-wrap"):
                                            ui.label(c["icon"]).classes("text-lg")
                                            ui.label(c["name"]).classes("text-xs font-semibold text-white")
                                        ui.label(c["badge"]).classes(
                                            "text-[9px] font-bold px-1.5 py-0.5 rounded"
                                        ).style(f"color: {c['badge_color']}; background: {c['badge_color']}18;")
                                    ui.label(c["desc"]).classes("text-[11px] text-gray-400 leading-tight")

                        # Navigation
                        with ui.row().classes("w-full justify-between items-center pt-3 border-t border-gray-800"):
                            def _back_step2() -> None:
                                current_step_holder["step"] = 1
                                _render_screen()

                            def _next_step2() -> None:
                                current_step_holder["step"] = 3
                                _render_screen()

                            ui.button("← Back", on_click=_back_step2).props("flat color=grey-4")
                            ui.button("Continue →", on_click=_next_step2).props(
                                'unelevated color="deep-purple-6"'
                            ).classes("px-6 font-semibold")

                    # ==========================================================
                    # SCREEN 3: API KEY
                    # ==========================================================
                    elif curr == 3:
                        prov = wiz["provider"]
                        prov_info = next((c for c in _PROVIDER_CARDS if c["value"] == prov), None)
                        prov_name = prov_info["name"] if prov_info else prov.title()
                        prov_icon = prov_info["icon"] if prov_info else "🔑"

                        with ui.row().classes("items-center gap-2"):
                            ui.label(prov_icon).classes("text-2xl")
                            ui.label(f"{prov_name} Setup").classes("text-xl font-bold text-white")

                        if prov == "ollama":
                            ui.label("Local execution — no API key or account required.").classes(
                                "text-sm text-gray-400 -mt-2"
                            )

                            with ui.column().classes("w-full p-4 rounded-xl gap-2").style(
                                "background: rgba(16, 185, 129, 0.08); border: 1.5px solid rgba(16, 185, 129, 0.3);"
                            ):
                                with ui.row().classes("items-center gap-2"):
                                    ui.label("✓").classes("text-emerald-400 font-bold text-base")
                                    ui.label("Zero Configuration Required").classes("text-xs font-bold text-emerald-300")
                                ui.label(
                                    "Ollama executes models directly on your machine. Just make sure the Ollama application or daemon is running."
                                ).classes("text-xs text-gray-300 leading-relaxed")
                                ui.label("Default local port: http://localhost:11434").classes(
                                    "text-[11px] text-emerald-400/80 font-mono"
                                )

                            ollama_url_input = ui.input(
                                label="Ollama Base URL (Optional)",
                                value=wiz["ollama_base_url"],
                            ).props("outlined dense dark").classes("w-full mt-2")

                            def _on_ollama_url_change(e: object) -> None:
                                wiz["ollama_base_url"] = getattr(e, "value", "http://localhost:11434")

                            ollama_url_input.on_value_change(_on_ollama_url_change)
                        else:
                            ui.label(f"Enter your {prov_name} API key below to enable agent queries.").classes(
                                "text-sm text-gray-400 -mt-2"
                            )

                            key_input = ui.input(
                                label=f"{prov_name} API Key",
                                value=wiz["api_key"],
                                password=True,
                                password_toggle_button=True,
                            ).props("outlined dense dark").classes("w-full mt-1")

                            def _on_key_change(e: object) -> None:
                                wiz["api_key"] = getattr(e, "value", "").strip()

                            key_input.on_value_change(_on_key_change)

                            url = _KEY_HELP_URLS.get(prov, "")
                            if url:
                                ui.html(
                                    f'<a href="{url}" target="_blank" '
                                    f'style="color:#c084fc; font-size:12px; text-decoration:underline;">'
                                    f'Need a key? Get an API key from {prov_name} →</a>'
                                )

                            with ui.row().classes("items-center gap-2 p-3 rounded-lg w-full").style(
                                "background: rgba(255,255,255,0.02); border: 1px solid #222;"
                            ):
                                ui.label("🔒").classes("text-xs")
                                ui.label(
                                    "Keys are stored securely in ~/.unjess/keys.yaml and are never sent to third-party tracking services."
                                ).classes("text-[11px] text-gray-400")

                        # Navigation
                        with ui.row().classes("w-full justify-between items-center pt-3 border-t border-gray-800"):
                            def _back_step3() -> None:
                                current_step_holder["step"] = 2
                                _render_screen()

                            def _next_step3() -> None:
                                current_step_holder["step"] = 4
                                _render_screen()

                            ui.button("← Back", on_click=_back_step3).props("flat color=grey-4")
                            ui.button("Continue →", on_click=_next_step3).props(
                                'unelevated color="deep-purple-6"'
                            ).classes("px-6 font-semibold")

                    # ==========================================================
                    # SCREEN 4: CHOOSE MODEL
                    # ==========================================================
                    elif curr == 4:
                        prov = wiz["provider"]
                        prov_info = next((c for c in _PROVIDER_CARDS if c["value"] == prov), None)
                        prov_name = prov_info["name"] if prov_info else prov.title()

                        ui.label("Select Default Model").classes("text-xl font-bold text-white")
                        ui.label(f"Choose which model you'd like Unjess to use for {prov_name}.").classes(
                            "text-sm text-gray-400 -mt-2"
                        )

                        models = _MODELS_PER_PROVIDER.get(prov, [])
                        if not wiz["model"] and models:
                            wiz["model"] = models[0][0]

                        def _select_model(m_id: str) -> None:
                            wiz["model"] = m_id
                            _render_screen()

                        with ui.column().classes("w-full gap-2.5 max-h-[340px] overflow-y-auto pr-1"):
                            for m_id, label, desc in models:
                                is_sel = wiz["model"] == m_id
                                m_classes = f"model-card {'active' if is_sel else ''}"
                                with ui.card().classes(m_classes) as m_card:
                                    m_card.on("click", lambda _e, mid=m_id: _select_model(mid))
                                    with ui.row().classes("w-full items-center justify-between no-wrap"):
                                        with ui.column().classes("gap-0"):
                                            ui.label(label).classes("text-xs font-semibold text-white")
                                            ui.label(m_id).classes("text-[10px] text-purple-400 font-mono")
                                        if is_sel:
                                            ui.label("✓").classes("text-xs font-bold text-purple-400")
                                    ui.label(desc).classes("text-[11px] text-gray-400 mt-1")

                        # Custom model input
                        with ui.expansion("Enter a custom model name", icon="tune").classes("w-full text-xs text-gray-400"):
                            custom_inp = ui.input(
                                label="Custom Model ID",
                                value=wiz["model"],
                            ).props("outlined dense dark").classes("w-full mt-1")

                            def _on_custom_model_change(e: object) -> None:
                                val = getattr(e, "value", "").strip()
                                if val:
                                    wiz["model"] = val

                            custom_inp.on_value_change(_on_custom_model_change)

                        # Navigation
                        with ui.row().classes("w-full justify-between items-center pt-3 border-t border-gray-800"):
                            def _back_step4() -> None:
                                current_step_holder["step"] = 3
                                _render_screen()

                            def _next_step4() -> None:
                                current_step_holder["step"] = 5
                                _render_screen()

                            ui.button("← Back", on_click=_back_step4).props("flat color=grey-4")
                            ui.button("Continue →", on_click=_next_step4).props(
                                'unelevated color="deep-purple-6"'
                            ).classes("px-6 font-semibold")

                    # ==========================================================
                    # SCREEN 5: READY & FINISH
                    # ==========================================================
                    elif curr == 5:
                        prov = wiz["provider"]
                        prov_info = next((c for c in _PROVIDER_CARDS if c["value"] == prov), None)
                        prov_name = prov_info["name"] if prov_info else prov.title()
                        prov_icon = prov_info["icon"] if prov_info else "🚀"

                        ui.label("You're All Set!").classes("text-xl font-bold text-white")
                        ui.label("Unjess is configured and ready for your coding sessions.").classes(
                            "text-sm text-gray-400 -mt-2"
                        )

                        # Summary Card
                        has_key = bool(wiz["api_key"] or prov == "ollama")
                        with ui.column().classes("w-full p-4 rounded-xl gap-3").style(
                            "background: rgba(255,255,255,0.02); border: 1px solid #262626;"
                        ):
                            ui.label("CONFIGURATION SUMMARY").classes("text-[10px] font-bold text-purple-400 tracking-wider")

                            items = [
                                ("Provider", f"{prov_icon} {prov_name}"),
                                ("Default Model", wiz["model"]),
                                ("Authentication", "Local / Free ✓" if prov == "ollama" else ("API Key Set ✓" if wiz["api_key"] else "⚠️ No Key Set")),
                                ("Storage", "~/.unjess/config.yaml"),
                            ]
                            for lbl, val in items:
                                with ui.row().classes("w-full justify-between items-center"):
                                    ui.label(lbl).classes("text-xs text-gray-400")
                                    ui.label(val).classes("text-xs font-semibold text-white font-mono")

                        # Quick tips
                        with ui.column().classes("w-full gap-1.5 p-3 rounded-lg").style(
                            "background: rgba(168, 85, 247, 0.05); border: 1px solid rgba(168, 85, 247, 0.15);"
                        ):
                            ui.label("💡 Quick Tips:").classes("text-xs font-semibold text-purple-300")
                            ui.label("• Type / in the chat input to see slash commands (planning, models, provider switch).").classes(
                                "text-[11px] text-gray-300"
                            )
                            ui.label("• Click the gear icon in the top bar anytime to add more API keys or change defaults.").classes(
                                "text-[11px] text-gray-300"
                            )

                        # Launch button
                        with ui.row().classes("w-full justify-between items-center pt-3 border-t border-gray-800"):
                            def _back_step5() -> None:
                                current_step_holder["step"] = 4
                                _render_screen()

                            def _finish_setup() -> None:
                                if settings is not None:
                                    _apply_wizard_settings(settings, wiz, router)
                                ui.notify("Setup complete! Welcome to Unjess 🎉", type="positive", position="top")
                                ui.navigate.to("/")

                            ui.button("← Back", on_click=_back_step5).props("flat color=grey-4")
                            ui.button(
                                "🚀 Start Chatting",
                                on_click=_finish_setup,
                            ).props('unelevated color="deep-purple-6"').classes("px-8 font-bold text-sm")

            # Initial screen render
            _render_screen()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_wizard_settings(
    settings: "Settings",
    wiz: dict[str, str],
    router: Any = None,
) -> None:
    """Write wizard choices into Settings and persist to disk.

    Args:
        settings: The Settings instance to populate.
        wiz: Wizard state dict with ``'provider'``, ``'api_key'``, ``'model'``, ``'ollama_base_url'``.
        router: Optional ProviderRouter instance to reload.
    """
    from unjess.config import save_config

    settings.provider = wiz["provider"]
    settings.model = wiz["model"]

    if wiz["provider"] == "ollama" and wiz.get("ollama_base_url"):
        settings.ollama_base_url = wiz["ollama_base_url"]

    if wiz.get("api_key"):
        settings.api_keys[wiz["provider"]] = wiz["api_key"]
        if wiz["provider"] == "ollama-api":
            os.environ["OLLAMA_API_KEY"] = wiz["api_key"]

    save_config(settings)

    if router and hasattr(router, "reload_providers"):
        try:
            router.reload_providers()
        except Exception as exc:
            logger.debug("Failed to reload router providers: %s", exc)

    logger.info(
        "Onboarding complete: provider=%s model=%s",
        wiz["provider"],
        wiz["model"],
    )
