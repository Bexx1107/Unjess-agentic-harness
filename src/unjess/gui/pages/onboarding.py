"""Onboarding page — first-run setup wizard.

Route: ``/onboarding``

A multi-step stepper flow that guides new users through provider
selection, API key entry, and model choice.  On completion the
settings are saved and the user is redirected to the chat page.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, TYPE_CHECKING

from nicegui import ui

if TYPE_CHECKING:
    from unjess.config import Settings

logger = logging.getLogger(__name__)

_STYLES_PATH = Path(__file__).resolve().parent.parent / "static" / "styles.css"

# Provider cards for the selection step
_PROVIDER_CARDS: list[dict[str, str]] = [
    {
        "value": "google",
        "name": "Google (Gemini)",
        "desc": "Gemini 2.5 Flash — FREE tier available ⭐",
        "icon": "🌐",
    },
    {
        "value": "openrouter",
        "name": "OpenRouter",
        "desc": "50+ FREE models — zero cost to start ⭐",
        "icon": "🔀",
    },
    {
        "value": "groq",
        "name": "Groq",
        "desc": "Llama 3.3 — FREE & ultra-fast inference",
        "icon": "⚡",
    },
    {
        "value": "cerebras",
        "name": "Cerebras",
        "desc": "ZAI GLM 4.7 — FREE & fast",
        "icon": "🧠",
    },
    {
        "value": "mistral",
        "name": "Mistral",
        "desc": "Codestral — FREE for code generation",
        "icon": "🇫🇷",
    },
    {
        "value": "openai",
        "name": "OpenAI",
        "desc": "GPT-4o, o4-mini — paid, most popular",
        "icon": "🤖",
    },
    {
        "value": "anthropic",
        "name": "Anthropic",
        "desc": "Claude Sonnet 4, Opus 4 — paid, excellent for code",
        "icon": "🏛️",
    },
    {
        "value": "xai",
        "name": "xAI / Grok",
        "desc": "Grok 4.1 Fast — cheap & fast",
        "icon": "🚀",
    },
    {
        "value": "ollama",
        "name": "Ollama (Local)",
        "desc": "Run open models locally — no API key needed",
        "icon": "💻",
    },
]

# Models per provider (value, label, description)
_MODELS_PER_PROVIDER: dict[str, list[tuple[str, str, str]]] = {
    "google": [
        ("gemini-2.5-flash", "Gemini 2.5 Flash", "Fast & FREE tier available"),
        ("gemini-2.5-pro", "Gemini 2.5 Pro", "Most capable — limited free tier"),
    ],
    "openai": [
        ("gpt-4o", "GPT-4o", "Most capable — great for complex tasks"),
        ("gpt-4o-mini", "GPT-4o mini", "Fast & cheap — good default"),
        ("o4-mini", "o4-mini", "Reasoning model — best for hard problems"),
    ],
    "anthropic": [
        ("claude-sonnet-4-20250514", "Claude Sonnet 4", "Best balance of speed and quality"),
        ("claude-opus-4-20250514", "Claude Opus 4", "Most capable — slow but powerful"),
    ],
    "groq": [
        ("llama-3.3-70b-versatile", "Llama 3.3 70B", "Fast & FREE — great for coding"),
        ("llama-3.1-8b-instant", "Llama 3.1 8B", "Ultra-fast & FREE"),
        ("mixtral-8x7b-32768", "Mixtral 8x7B", "FREE — 32K context"),
    ],
    "mistral": [
        ("codestral-latest", "Codestral", "FREE for code — best code model"),
        ("mistral-small-latest", "Mistral Small", "FREE tier — fast general"),
    ],
    "xai": [
        ("grok-4.1-fast", "Grok 4.1 Fast", "Cheap & fast"),
        ("grok-4.3", "Grok 4.3", "Flagship — most capable"),
    ],
    "openrouter": [
        ("openrouter/free", "Auto-Free", "Auto-picks best FREE model"),
        ("nvidia/nemotron-3-ultra:free", "Nemotron 3 Ultra", "FREE — 1M context"),
        ("qwen/qwen3-coder:free", "Qwen 3 Coder", "FREE — code gen, 1M context"),
    ],
    "cerebras": [
        ("zai-glm-4.7", "ZAI GLM 4.7", "FREE — fast inference, 131K context"),
        ("llama-3.1-70b", "Llama 3.1 70B", "FREE — Meta's open model"),
    ],
    "ollama": [
        ("llama3.1", "Llama 3.1", "Meta's open model — runs locally"),
        ("codestral", "Codestral", "Mistral's code model — local"),
        ("qwen2.5-coder", "Qwen 2.5 Coder", "Alibaba's code model — local"),
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
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def setup_onboarding_page(
    state: "Any" = None,
    settings: "Settings | None" = None,
    router: "Any" = None,
) -> None:
    """Build the first-run onboarding wizard at route ``/onboarding``.

    Uses NiceGUI's ``ui.stepper`` for a 5-step flow:

    1. **Welcome** — logo + description + "Get Started" button.
    2. **Provider** — cards for each provider with free-tier indicators.
    3. **API Key** — input field + help link.
    4. **Model** — radio list of models for the selected provider.
    5. **Ready** — summary + "Start Chatting" button.

    On completion, saves settings and redirects to ``/`` (chat page).

    Args:
        state: The AppState instance (unused in onboarding but passed by app.py).
        settings: The mutable :class:`Settings` instance to populate.
        router: The provider router (unused in onboarding but passed by app.py).
    """
    if _STYLES_PATH.exists():
        ui.add_css(_STYLES_PATH.read_text(encoding="utf-8"))

    ui.dark_mode(True)

    # Wizard state — mutable dict so closures can share it
    wiz: dict[str, str] = {
        "provider": "google",
        "api_key": "",
        "model": "gemini-2.5-flash",
    }

    with ui.column().classes(
        "w-full max-w-xl mx-auto q-pa-lg"
    ).style("min-height:100vh"):

        # ── Logo ────────────────────────────────────────────────────────
        with ui.column().classes("items-center q-mb-lg"):
            ui.label("🟣").style("font-size:48px")
            ui.label("njss").classes(
                "text-h4 text-weight-bold"
            ).style("color:#c084fc; letter-spacing:2px")
            ui.label(
                "AI coding agent — let's get you set up"
            ).classes("text-subtitle2 text-grey-5 q-mt-xs")

        # ── Stepper ─────────────────────────────────────────────────────
        with ui.stepper().props(
            'vertical animated dark color="deep-purple-6"'
        ).classes("w-full") as stepper:

            # Step 1: Welcome
            with ui.step("Welcome"):
                ui.markdown(
                    "**Welcome to Unjess!**\n\n"
                    "njss is a full-featured AI coding agent that lives in your "
                    "terminal (and now your browser). It can:\n\n"
                    "- 📝 Read, write, and edit files\n"
                    "- 🔧 Run shell commands\n"
                    "- 🔍 Search the web and your codebase\n"
                    "- 💡 Plan and execute complex tasks\n\n"
                    "Let's configure your first AI provider."
                ).classes("text-grey-4")
                with ui.stepper_navigation():
                    ui.button(
                        "Get Started →",
                        on_click=stepper.next,
                    ).props('color="deep-purple-6" unelevated')

            # Step 2: Provider selection
            with ui.step("Choose Provider"):
                ui.label("Pick an AI provider:").classes(
                    "text-weight-medium text-white q-mb-sm"
                )

                for card_info in _PROVIDER_CARDS:
                    _val = card_info["value"]
                    _render_provider_card(card_info, wiz)

                with ui.stepper_navigation():
                    ui.button("Back", on_click=stepper.previous).props("flat")
                    ui.button("Next →", on_click=stepper.next).props(
                        'color="deep-purple-6" unelevated'
                    )

            # Step 3: API key
            with ui.step("API Key"):
                key_info = ui.label("").classes("text-weight-medium text-white q-mb-xs")
                key_help = ui.html("").classes("q-mb-sm")
                key_input = ui.input(
                    label="Paste your API key",
                    password=True,
                    password_toggle_button=True,
                ).classes("w-full").props('dark dense color="purple-4"')

                ollama_note = ui.label(
                    "Ollama runs locally — no API key needed. Just make sure "
                    "Ollama is running on your machine."
                ).classes("text-sm text-grey-5").style("display:none")

                def _refresh_key_step() -> None:
                    """Update the key step UI for the selected provider."""
                    prov = wiz["provider"]
                    if prov == "ollama":
                        key_info.set_text("Ollama — no key required")
                        key_input.set_visibility(False)
                        ollama_note.style("display:block")
                        key_help.content = ""
                    else:
                        key_info.set_text(f"Enter your {prov.title()} API key:")
                        key_input.set_visibility(True)
                        ollama_note.style("display:none")
                        url = _KEY_HELP_URLS.get(prov, "")
                        if url:
                            key_help.content = (
                                f'<a href="{url}" target="_blank" '
                                f'style="color:#c084fc;font-size:13px">'
                                f'Get a {prov.title()} key →</a>'
                            )
                        else:
                            key_help.content = ""

                # Refresh on step entry via timer
                ui.timer(0.3, _refresh_key_step, once=True)

                with ui.stepper_navigation():
                    ui.button("Back", on_click=stepper.previous).props("flat")

                    def _step3_next() -> None:
                        wiz["api_key"] = key_input.value.strip()
                        _refresh_key_step()  # ensure state is current
                        stepper.next()

                    ui.button("Next →", on_click=_step3_next).props(
                        'color="deep-purple-6" unelevated'
                    )

            # Step 4: Model selection
            with ui.step("Choose Model"):
                ui.label("Select a model:").classes(
                    "text-weight-medium text-white q-mb-sm"
                )
                model_container = ui.column().classes("w-full gap-xs")

                def _refresh_models() -> None:
                    """Re-populate model list for the selected provider."""
                    model_container.clear()
                    prov = wiz["provider"]
                    models = _MODELS_PER_PROVIDER.get(prov, [])
                    if not models:
                        with model_container:
                            ui.label("No models available.").classes(
                                "text-sm text-grey-6"
                            )
                        return

                    # Default to first model
                    if not wiz["model"] or not any(
                        m[0] == wiz["model"] for m in models
                    ):
                        wiz["model"] = models[0][0]

                    with model_container:
                        model_radio = ui.radio(
                            options={
                                val: f"{label} — {desc}"
                                for val, label, desc in models
                            },
                            value=wiz["model"],
                        ).classes("w-full").props('dark color="purple-4"')

                        def on_model(e: object) -> None:
                            wiz["model"] = getattr(e, "value", "")

                        model_radio.on_value_change(on_model)

                ui.timer(0.3, _refresh_models, once=True)

                with ui.stepper_navigation():
                    ui.button("Back", on_click=stepper.previous).props("flat")
                    ui.button("Next →", on_click=stepper.next).props(
                        'color="deep-purple-6" unelevated'
                    )

            # Step 5: Ready!
            with ui.step("Ready!"):
                summary = ui.column().classes("w-full gap-xs")

                def _refresh_summary() -> None:
                    summary.clear()
                    key_set = bool(
                        wiz["api_key"] or wiz["provider"] == "ollama"
                    )
                    with summary:
                        ui.markdown(
                            f"**Provider:** {wiz['provider'].title()}\n\n"
                            f"**Model:** `{wiz['model']}`\n\n"
                            f"**API Key:** {'configured ✓' if key_set else '⚠️ not set'}\n\n"
                            "You can change these anytime in **Settings**."
                        ).classes("text-grey-4")

                ui.timer(0.3, _refresh_summary, once=True)

                with ui.stepper_navigation():
                    ui.button("Back", on_click=stepper.previous).props("flat")

                    def _finish() -> None:
                        if settings is None:
                            ui.notify("Settings not available", type="warning")
                            return
                        _apply_wizard_settings(settings, wiz)
                        ui.notify(
                            "Setup complete! 🎉",
                            type="positive",
                            position="top",
                        )
                        ui.navigate.to("/")

                    ui.button(
                        "🚀 Start Chatting",
                        on_click=_finish,
                    ).props('color="deep-purple-6" unelevated').style(
                        "font-weight:600"
                    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _render_provider_card(card_info: dict[str, str], wiz: dict[str, str]) -> None:
    """Render a single provider selection card.

    Args:
        card_info: Dict with ``value``, ``name``, ``desc``, ``icon``.
        wiz: Mutable wizard state dict.
    """
    val = card_info["value"]
    is_selected = wiz["provider"] == val

    border_color = "rgba(124,58,237,0.6)" if is_selected else "rgba(255,255,255,0.06)"
    bg = "rgba(124,58,237,0.08)" if is_selected else "rgba(255,255,255,0.02)"

    card = ui.card().classes("w-full q-pa-sm q-mb-xs cursor-pointer").style(
        f"background:{bg}; border:2px solid {border_color}; "
        "border-radius:10px; transition:all 150ms ease"
    )

    def _select(v: str = val) -> None:
        wiz["provider"] = v
        # Visual update requires page re-render — NiceGUI handles this
        # through the stepper's own state management.

    card.on("click", lambda _, v=val: _select(v))

    with card:
        with ui.row().classes("items-center gap-sm no-wrap"):
            ui.label(card_info["icon"]).style("font-size:22px")
            with ui.column().style("gap:2px"):
                ui.label(card_info["name"]).classes(
                    "text-sm text-white text-weight-medium"
                )
                ui.label(card_info["desc"]).classes("text-xs text-grey-6")


def _apply_wizard_settings(settings: "Settings", wiz: dict[str, str]) -> None:
    """Write wizard choices into Settings and persist to disk.

    Args:
        settings: The Settings instance to populate.
        wiz: Wizard state dict with ``'provider'``, ``'api_key'``, ``'model'``.
    """
    from unjess.config import save_config

    settings.provider = wiz["provider"]
    settings.model = wiz["model"]

    if wiz["api_key"]:
        settings.api_keys[wiz["provider"]] = wiz["api_key"]

    save_config(settings)
    logger.info(
        "Onboarding complete: provider=%s model=%s",
        wiz["provider"],
        wiz["model"],
    )
