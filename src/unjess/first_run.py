"""First-run setup wizard — interactive provider and model selection.

Includes free-tier options and experimental OpenAI OAuth login.
"""

import json
import logging
import os
from typing import Optional
from urllib.request import urlopen, Request
from urllib.error import URLError

logger = logging.getLogger(__name__)

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm

from unjess.config import Settings, save_config


# Known models per provider
_PROVIDER_MODELS: dict[str, list[tuple[str, str]]] = {
    "openai": [
        ("gpt-5.4", "Most capable — frontier reasoning & tool use"),
        ("gpt-5.4-mini", "Fast & efficient — best for subagents"),
        ("o5-preview", "Deep reasoning — for extremely complex tasks"),
    ],
    "openai-oauth": [
        ("gpt-5.4", "Uses your ChatGPT Plus/Pro subscription"),
        ("gpt-5.4-mini", "Fast — uses your subscription"),
        ("o5-preview", "Reasoning model via subscription"),
    ],
    "anthropic": [
        ("claude-sonnet-5-20260610", "Agentic masterpiece — high reliability"),
        ("claude-opus-4.8-20260315", "Ultimate reasoning powerhouse"),
    ],
    "google": [
        ("gemini-3.1-flash", "Ultra-fast & FREE tier available"),
        ("gemini-3.1-pro", "Massive 2M context — free tier limited"),
    ],
    "groq": [
        ("llama-3.5-70b-versatile", "LPU speed — Meta's latest open model"),
        ("mixtral-large-3-32768", "Incredible speed for large context"),
    ],
    "mistral": [
        ("codestral-v2-latest", "Mistral's premier coding agent"),
        ("mistral-large-3.5-latest", "Frontier-class open weights"),
        ("open-mistral-nemo-v2", "128K context, highly efficient"),
    ],
    "xai": [
        ("grok-4.1-fast", "Fast real-time integration"),
        ("grok-4.5", "Flagship flagship — most capable"),
    ],
    "openrouter": [
        ("openrouter/free", "Auto-picks best FREE model ⭐"),
        ("meta-llama/llama-3.5-70b:free", "FREE — Frontier performance"),
        ("mistralai/mistral-large-3.5:free", "FREE — High reasoning"),
    ],
    "cerebras": [
        ("zai-glm-4.7", "FREE — ultra-fast LPU inference"),
        ("llama-3.5-70b", "FREE — Meta's flagship model"),
    ],
    "kimi": [
        ("kimi-k3", "Flagship 2026 — Moonshot AI long-horizon coding model"),
        ("kimi-k2.7-code", "High-speed coding model"),
        ("kimi-latest", "Auto-resolves to Moonshot AI's latest model"),
    ],
    "qwen": [
        ("qwen-max", "Alibaba Cloud flagship model"),
        ("qwen-plus", "Balanced speed & reasoning model"),
        ("qwen-turbo", "High-throughput cost-effective model"),
        ("qwen2.5-coder-32b-instruct", "Specialized open-weight coding model"),
    ],
    "ollama": [
        ("llama3.5", "Meta's latest — runs locally"),
        ("llama3.1", "Meta's open model — runs locally"),
        ("codestral-v2", "Mistral's code model — runs locally"),
        ("qwen3-coder", "Alibaba's latest code model — runs locally"),
    ],
    "ollama-api": [
        ("llama3.3", "Meta Llama 3.3 — Ollama Cloud"),
        ("deepseek-r1", "DeepSeek R1 — Ollama Cloud"),
        ("qwen2.5-coder:32b", "Qwen 2.5 Coder 32B — Ollama Cloud"),
        ("mistral-large", "Mistral Large — Ollama Cloud"),
    ],
}


def _fetch_live_models(
    provider: str,
    settings: Optional[Settings] = None,
    token: Optional[str] = None,
) -> list[tuple[str, str]]:
    """Query a provider's API for available models.

    Args:
        provider: Provider name.
        settings: Settings with saved API keys.
        token: Explicit auth token (e.g. from OAuth flow).

    Returns a list of (model_name, description) tuples.
    Falls back to empty list on any failure.
    """
    try:
        if provider == "ollama":
            return _fetch_ollama_models(settings)
        elif provider == "ollama-api":
            return _fetch_ollama_api_models(settings)
        elif provider == "google":
            return _fetch_google_models(settings)
        elif provider == "groq":
            return _fetch_groq_models(settings)
        elif provider == "mistral":
            return _fetch_mistral_models(settings)
        if provider in ("openai", "openai-oauth"):
            return _fetch_openai_models(settings, token=token)
        if provider == "xai":
            return _fetch_xai_models(settings)
        if provider == "openrouter":
            return _fetch_openrouter_models(settings)
        if provider == "cerebras":
            return _fetch_cerebras_models(settings)
        if provider == "kimi":
            return _fetch_kimi_models(settings)
        if provider == "qwen":
            return _fetch_qwen_models(settings)
    except Exception as exc:
        logger.debug("Failed to fetch live models for %s: %s", provider, exc)
    return []


def _fetch_kimi_models(settings: Optional[Settings] = None) -> list[tuple[str, str]]:
    """Fetch live models from Moonshot AI (Kimi) API."""
    key = _get_api_key("kimi", "MOONSHOT_API_KEY", settings) or _get_api_key("moonshot", "KIMI_API_KEY", settings)
    if not key:
        return []
    url = getattr(settings, "kimi_base_url", "https://api.moonshot.ai/v1") + "/models"
    try:
        raw = _api_get_models(url, key)
        models: list[tuple[str, str]] = []
        for item in raw:
            mid = item.get("id", "")
            if mid:
                models.append((mid, "Moonshot AI / Kimi model"))
        return sorted(models, key=lambda x: x[0])
    except Exception as exc:
        logger.debug("Failed to fetch Kimi models: %s", exc)
        return []


def _fetch_qwen_models(settings: Optional[Settings] = None) -> list[tuple[str, str]]:
    """Fetch live models from Alibaba Cloud DashScope (Qwen) API."""
    key = _get_api_key("qwen", "DASHSCOPE_API_KEY", settings) or _get_api_key("dashscope", "QWEN_API_KEY", settings)
    if not key:
        return []
    url = getattr(settings, "qwen_base_url", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1") + "/models"
    try:
        raw = _api_get_models(url, key)
        models: list[tuple[str, str]] = []
        for item in raw:
            mid = item.get("id", "")
            if mid:
                models.append((mid, "Alibaba Cloud Qwen model"))
        return sorted(models, key=lambda x: x[0])
    except Exception as exc:
        logger.debug("Failed to fetch Qwen models: %s", exc)
        return []


def _fetch_ollama_models(settings: Optional[Settings] = None) -> list[tuple[str, str]]:
    """Query local Ollama API for installed models."""
    base_url = (getattr(settings, "ollama_base_url", None) or "http://localhost:11434").rstrip("/")
    try:
        req = Request(f"{base_url}/api/tags", method="GET")
        with urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())

        models: list[tuple[str, str]] = []
        for m in data.get("models", []):
            name = m.get("name", "")
            display_name = name.replace(":latest", "") if name.endswith(":latest") else name
            size_gb = m.get("size", 0) / (1024 ** 3)
            desc = f"{size_gb:.1f}GB — installed locally"
            models.append((display_name, desc))
        return sorted(models, key=lambda x: x[0])
    except Exception as exc:
        logger.debug("Failed to query local Ollama models: %s", exc)
        return _PROVIDER_MODELS.get("ollama", [])


def _fetch_ollama_api_models(settings: Optional[Settings] = None) -> list[tuple[str, str]]:
    """Fetch cloud models from Ollama Cloud API."""
    key = _get_api_key("ollama-api", "OLLAMA_API_KEY", settings) or _get_api_key("ollama", "OLLAMA_API_KEY", settings)
    if not key:
        return _PROVIDER_MODELS.get("ollama-api", [])
    base_url = (getattr(settings, "ollama_api_base_url", None) or "https://api.ollama.com").rstrip("/")
    try:
        raw = _api_get_models(f"{base_url}/v1/models", key)
        models: list[tuple[str, str]] = []
        for item in raw:
            mid = item.get("id", "")
            if mid:
                models.append((mid, "Ollama Cloud model"))
        if models:
            return sorted(models, key=lambda x: x[0])
    except Exception as exc:
        logger.debug("Failed to fetch Ollama API models: %s", exc)
    return _PROVIDER_MODELS.get("ollama-api", [])


def _api_get_models(url: str, api_key: str, data_key: str = "data") -> list[dict]:
    """Generic helper to GET a models endpoint with Bearer auth."""
    req = Request(url, method="GET")
    req.add_header("Authorization", f"Bearer {api_key}")
    with urlopen(req, timeout=5) as resp:
        result = json.loads(resp.read())
    return result.get(data_key, [])


def _get_api_key(provider: str, env_var: str, settings: Optional[Settings] = None) -> str:
    """Get API key from env or settings."""
    key = os.environ.get(env_var, "")
    if not key and settings:
        key = settings.api_keys.get(provider, "")
    return key


def _fetch_openai_models(
    settings: Optional[Settings] = None,
    token: Optional[str] = None,
) -> list[tuple[str, str]]:
    """Query OpenAI API for available models."""
    api_key = token or _get_api_key("openai", "OPENAI_API_KEY", settings)
    if not api_key:
        return []
    raw = _api_get_models("https://api.openai.com/v1/models", api_key)
    # Filter to chat models only
    chat_prefixes = ("gpt-4", "gpt-3.5", "o1", "o3", "o4")
    models = [(m["id"], "") for m in raw if any(m.get("id", "").startswith(p) for p in chat_prefixes)]
    return sorted(models, key=lambda x: x[0])


def _fetch_google_models(settings: Optional[Settings] = None) -> list[tuple[str, str]]:
    """Query Google AI API for available models."""
    api_key = _get_api_key("google", "GOOGLE_API_KEY", settings)
    if not api_key:
        return []
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    req = Request(url, method="GET")
    with urlopen(req, timeout=5) as resp:
        data = json.loads(resp.read())
    models: list[tuple[str, str]] = []
    for m in data.get("models", []):
        name = m.get("name", "").replace("models/", "")
        if "gemini" in name and "generateContent" in str(m.get("supportedGenerationMethods", [])):
            desc = m.get("description", "")[:60]
            models.append((name, desc))
    return sorted(models, key=lambda x: x[0])


def _fetch_groq_models(settings: Optional[Settings] = None) -> list[tuple[str, str]]:
    """Query Groq API for available models."""
    api_key = _get_api_key("groq", "GROQ_API_KEY", settings)
    if not api_key:
        return []
    raw = _api_get_models("https://api.groq.com/openai/v1/models", api_key)
    models = [(m["id"], "") for m in raw if m.get("active", True)]
    return sorted(models, key=lambda x: x[0])


def _fetch_mistral_models(settings: Optional[Settings] = None) -> list[tuple[str, str]]:
    """Query Mistral API for available models."""
    api_key = _get_api_key("mistral", "MISTRAL_API_KEY", settings)
    if not api_key:
        return []
    raw = _api_get_models("https://api.mistral.ai/v1/models", api_key)
    models = [(m["id"], "") for m in raw]
    return sorted(models, key=lambda x: x[0])


def _fetch_xai_models(settings: Optional[Settings] = None) -> list[tuple[str, str]]:
    """Query xAI API for available models."""
    api_key = _get_api_key("xai", "XAI_API_KEY", settings)
    if not api_key:
        return []
    raw = _api_get_models("https://api.x.ai/v1/models", api_key)
    models = [(m["id"], "") for m in raw if "grok" in m.get("id", "").lower()]
    return sorted(models, key=lambda x: x[0])


def _fetch_openrouter_models(settings: Optional[Settings] = None) -> list[tuple[str, str]]:
    """Query OpenRouter API for available free models."""
    api_key = _get_api_key("openrouter", "OPENROUTER_API_KEY", settings)
    if not api_key:
        return []
    raw = _api_get_models("https://openrouter.ai/api/v1/models", api_key)
    models: list[tuple[str, str]] = []
    for m in raw:
        model_id = m.get("id", "")
        pricing = m.get("pricing", {})
        prompt_cost = float(pricing.get("prompt", "1") or "1")
        # Only show free models (prompt cost == 0)
        if prompt_cost == 0:
            name = m.get("name", model_id)
            ctx = m.get("context_length", 0)
            desc = f"FREE — {name}"
            if ctx:
                desc += f" ({ctx // 1000}K ctx)"
            models.append((model_id, desc))
    return sorted(models, key=lambda x: x[0])[:30]  # cap at 30 for readability


def _fetch_cerebras_models(settings: Optional[Settings] = None) -> list[tuple[str, str]]:
    """Query Cerebras API for available models."""
    api_key = _get_api_key("cerebras", "CEREBRAS_API_KEY", settings)
    if not api_key:
        return []
    raw = _api_get_models("https://api.cerebras.ai/v1/models", api_key)
    models = [(m["id"], "FREE — ultra-fast inference") for m in raw]
    return sorted(models, key=lambda x: x[0])

_PROVIDER_DISPLAY: dict[str, str] = {
    "openai": "OpenAI (API key)",
    "openai-oauth": "OpenAI (browser login)",
    "anthropic": "Anthropic",
    "google": "Google Gemini",
    "groq": "Groq",
    "mistral": "Mistral",
    "xai": "xAI (Grok)",
    "openrouter": "OpenRouter",
    "cerebras": "Cerebras",
    "kimi": "Kimi (Moonshot AI)",
    "qwen": "Qwen (Alibaba DashScope)",
    "ollama": "Ollama",
    "ollama-api": "Ollama API",
}

_ENV_KEY_NAMES: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "groq": "GROQ_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "xai": "XAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
    "kimi": "MOONSHOT_API_KEY",
    "qwen": "DASHSCOPE_API_KEY",
    "ollama": "OLLAMA_API_KEY",
    "ollama-api": "OLLAMA_API_KEY",
}


def _detect_available_providers(settings: Optional[Settings] = None) -> dict[str, bool]:
    """Check which providers have API keys configured.

    Checks both environment variables and saved keys in settings.
    """
    saved = settings.api_keys if settings else {}
    available: dict[str, bool] = {}
    available["openai"] = bool(os.environ.get("OPENAI_API_KEY", "") or saved.get("openai", ""))
    available["openai-oauth"] = True  # Always available (browser login)
    available["anthropic"] = bool(os.environ.get("ANTHROPIC_API_KEY", "") or saved.get("anthropic", ""))
    available["google"] = bool(os.environ.get("GOOGLE_API_KEY", "") or saved.get("google", ""))
    available["groq"] = bool(os.environ.get("GROQ_API_KEY", "") or saved.get("groq", ""))
    available["mistral"] = bool(os.environ.get("MISTRAL_API_KEY", "") or saved.get("mistral", ""))
    available["xai"] = bool(os.environ.get("XAI_API_KEY", "") or saved.get("xai", ""))
    available["openrouter"] = bool(os.environ.get("OPENROUTER_API_KEY", "") or saved.get("openrouter", ""))
    available["cerebras"] = bool(os.environ.get("CEREBRAS_API_KEY", "") or saved.get("cerebras", ""))
    available["kimi"] = bool(os.environ.get("MOONSHOT_API_KEY", "") or os.environ.get("KIMI_API_KEY", "") or saved.get("kimi", ""))
    available["qwen"] = bool(os.environ.get("DASHSCOPE_API_KEY", "") or os.environ.get("QWEN_API_KEY", "") or saved.get("qwen", ""))
    available["ollama"] = True  # Always available (local)
    available["ollama-api"] = bool(
        os.environ.get("OLLAMA_API_KEY", "")
        or saved.get("ollama-api", "")
        or (saved.get("ollama", "") and saved.get("ollama", "") != "ollama")
    )
    return available


def _try_oauth_login(console: Console) -> Optional[str]:
    """Attempt OpenAI OAuth login via browser.

    Returns the access token on success, None on failure.
    """
    try:
        from unjess.llm.oauth_provider import openai_oauth_login
        console.print("\n[bold]Opening your browser to log in with OpenAI...[/bold]")
        console.print("[dim]If the browser doesn't open, check the URL printed below.[/dim]\n")
        token = openai_oauth_login()
        if token:
            console.print("[bold green]Login successful![/bold green]\n")
            return token
        else:
            console.print("[red]Login failed or was cancelled.[/red]\n")
            return None
    except Exception as exc:
        console.print(f"[red]OAuth error: {exc}[/red]")
        console.print("[dim]You can use an API key instead (option 1).[/dim]\n")
        return None



def _collect_extra_keys(
    console: Console,
    settings: "Settings",
    provider_id: str,
    display_name: str,
    url: str,
) -> None:
    """Loop to collect additional API keys for a provider.

    Keeps asking until the user enters an empty key or says no.
    All keys are stored in settings.api_key_pool.
    """
    from rich.prompt import Prompt

    key_num = len(settings.api_key_pool.get(provider_id, []))
    while True:
        key_num += 1
        console.print(
            f"  [dim]Tip: Use a different email/account for each key[/dim]"
        )
        extra_key = Prompt.ask(
            f"  Paste {display_name} key #{key_num + 1} (or Enter to stop)",
            default="",
        )
        if not extra_key.strip():
            break

        if provider_id not in settings.api_key_pool:
            settings.api_key_pool[provider_id] = []

        # Avoid duplicates
        if extra_key.strip() not in settings.api_key_pool[provider_id]:
            settings.api_key_pool[provider_id].append(extra_key.strip())
            console.print(
                f"  [green]✓[/green] {display_name} key #{key_num + 1} saved! "
                f"({len(settings.api_key_pool[provider_id])} keys total)"
            )
        else:
            console.print(f"  [yellow]⚠ Duplicate key — skipped[/yellow]")

        add_more = Prompt.ask(
            f"  Add another {display_name} key? (y/N)",
            default="n",
        )
        if add_more.lower() not in ("y", "yes"):
            break

    pool_count = len(settings.api_key_pool.get(provider_id, []))
    if pool_count > 1:
        console.print(
            f"  [bold green]{pool_count} {display_name} keys[/bold green] "
            f"= {pool_count}x free quota 🎉"
        )


def run_first_setup(settings: Settings, console: Console) -> Settings:
    """Run the interactive first-run setup wizard.

    Guides the user through selecting a provider and model.
    Updates and saves the settings.

    Args:
        settings: Settings instance to update.
        console: Rich console for display.

    Returns:
        Updated Settings instance.
    """
    console.print()
    console.print(Panel(
        "[bold]Welcome to njss![/bold]\n\n"
        "Let's set up your AI coding agent.\n"
        "You can change these settings anytime with [cyan]/settings[/cyan] or [cyan]/model[/cyan].",
        title="First Run Setup",
        title_align="left",
        border_style="cyan",
        expand=False,
    ))
    console.print()

    # --- Detect providers ---
    available = _detect_available_providers(settings)

    table = Table(
        show_header=True,
        border_style="dim",
        expand=False,
        padding=(0, 1),
    )
    table.add_column("#", style="bold", width=3)
    table.add_column("Provider", style="bold cyan")
    table.add_column("Status", min_width=20)
    table.add_column("Cost", style="dim")

    # Option 1 is the special "Free Mode" — auto-rotates between free tiers
    # Options 2+ are individual providers
    provider_order = [
        "google",        # 2 — free tier
        "groq",          # 3 — free tier
        "mistral",       # 4 — free tier
        "openrouter",    # 5 — free tier (50+ models)
        "cerebras",      # 6 — free tier (ultra-fast)
        "xai",           # 7 — cheap paid
        "openai-oauth",  # 8 — uses existing subscription
        "openai",        # 9 — API key (paid)
        "anthropic",     # 10 — API key (paid)
        "ollama",        # 11 — local/free
        "ollama-api",    # 12 — cloud subscription
    ]

    # Row 1: Free Mode (special)
    table.add_row(
        "1",
        "[bold green]Free Mode[/bold green]",
        "[green]Auto-rotate free models[/green]",
        "[bold green]FREE[/bold green]",
    )

    status_map = {
        "google": (
            "[green]API key found[/green]" if available["google"]
            else "[dim]Needs GOOGLE_API_KEY[/dim]"
        ),
        "groq": (
            "[green]API key found[/green]" if available["groq"]
            else "[dim]Needs GROQ_API_KEY[/dim]"
        ),
        "mistral": (
            "[green]API key found[/green]" if available["mistral"]
            else "[dim]Needs MISTRAL_API_KEY[/dim]"
        ),
        "openai-oauth": "[yellow]Experimental[/yellow] — browser login",
        "openai": (
            "[green]API key found[/green]" if available["openai"]
            else "[dim]Needs OPENAI_API_KEY[/dim]"
        ),
        "anthropic": (
            "[green]API key found[/green]" if available["anthropic"]
            else "[dim]Needs ANTHROPIC_API_KEY[/dim]"
        ),
        "xai": (
            "[green]API key found[/green]" if available["xai"]
            else "[dim]Needs XAI_API_KEY[/dim]"
        ),
        "openrouter": (
            "[green]API key found[/green]" if available["openrouter"]
            else "[dim]Needs OPENROUTER_API_KEY[/dim]"
        ),
        "cerebras": (
            "[green]API key found[/green]" if available["cerebras"]
            else "[dim]Needs CEREBRAS_API_KEY[/dim]"
        ),
        "ollama": "[green]Always available[/green]",
        "ollama-api": (
            "[green]API key found[/green]" if available["ollama-api"]
            else "[dim]Needs OLLAMA_API_KEY[/dim]"
        ),
    }

    cost_map = {
        "google": "[bold green]FREE[/bold green] (1500 req/day)",
        "groq": "[bold green]FREE[/bold green] (30 req/min)",
        "mistral": "[bold green]FREE[/bold green] (code models)",
        "openai-oauth": "Uses ChatGPT Plus/Pro sub",
        "openai": "Pay per token",
        "anthropic": "Pay per token",
        "xai": "Pay per token (cheap)",
        "openrouter": "[bold green]FREE[/bold green] (50+ models)",
        "cerebras": "[bold green]FREE[/bold green] (1M tok/day)",
        "ollama": "[bold green]FREE[/bold green] (runs locally)",
        "ollama-api": "Ollama subscription",
    }

    for i, provider in enumerate(provider_order, 2):  # start at 2
        table.add_row(
            str(i),
            _PROVIDER_DISPLAY[provider],
            status_map[provider],
            cost_map[provider],
        )

    console.print(table)
    console.print()

    # --- Suggest free options ---
    has_any_paid_key = available["openai"] or available["anthropic"]
    has_any_free_key = (
        available["google"] or available["groq"] or available["mistral"]
        or available["openrouter"] or available["cerebras"]
    )

    if not has_any_paid_key and not has_any_free_key:
        console.print(
            "[yellow]No API keys detected.[/yellow] Recommended:\n"
            "  [bold]1[/bold] — Free Mode (auto-rotate between free models)\n"
            "  [bold]11[/bold] — Ollama (runs locally, no account needed)\n\n"
            "[dim]For free cloud models, get an API key from:\n"
            "  openrouter.ai | cloud.cerebras.ai | ai.google.dev\n"
            "  console.groq.com | console.mistral.ai[/dim]\n"
        )
    else:
        console.print(
            "[dim]Tip: Option 1 (Free Mode) auto-rotates between available\n"
            "free models so you never hit a rate limit.[/dim]\n"
        )

    # --- Provider selection ---
    # Total choices: 1 (free mode) + len(provider_order)
    total_choices = 1 + len(provider_order)
    default_choice = "1"  # Always default to Free Mode

    choice = Prompt.ask(
        "Pick a provider",
        choices=[str(i) for i in range(1, total_choices + 1)],
        default=default_choice,
    )
    choice_num = int(choice)

    # --- Handle Free Mode selection ---
    if choice_num == 1:
        console.print()
        console.print(Panel(
            "[bold]Free Mode Setup[/bold]\n\n"
            "Free mode rotates between free-tier cloud models so you\n"
            "never hit a rate limit. You can add as many as you want —\n"
            "more keys = more free capacity.\n\n"
            "[dim]All keys are free — no credit card needed.[/dim]",
            border_style="cyan",
            expand=False,
        ))
        console.print()

        # Walk through each free provider
        free_providers = [
            (
                "openrouter", "OPENROUTER_API_KEY",
                "OpenRouter", "openrouter.ai",
                "50+ free models behind one key ⭐ (no credit card)",
            ),
            (
                "cerebras", "CEREBRAS_API_KEY",
                "Cerebras", "cloud.cerebras.ai",
                "Ultra-fast inference, 1M tokens/day free (no credit card)",
            ),
            (
                "google", "GOOGLE_API_KEY",
                "Google Gemini", "ai.google.dev",
                "Generous free tier — 1,500 requests/day",
            ),
            (
                "groq", "GROQ_API_KEY",
                "Groq", "console.groq.com",
                "Ultra-fast Llama 3.3 70B — 30 requests/min",
            ),
            (
                "mistral", "MISTRAL_API_KEY",
                "Mistral", "console.mistral.ai",
                "Codestral code model — free for code tasks",
            ),
        ]

        keys_added: list[str] = []

        for provider_id, env_var, display_name, url, description in free_providers:
            if available[provider_id]:
                # Already has key — still offer to add more
                console.print(
                    f"  [green]✓[/green] {display_name} — API key found"
                )
                keys_added.append(provider_id)

                # Ask if they want to add more keys for this provider
                add_more = Prompt.ask(
                    f"  Add another {display_name} key for extra quota? (y/N)",
                    default="n",
                )
                if add_more.lower() in ("y", "yes"):
                    _collect_extra_keys(
                        console, settings, provider_id, display_name, url
                    )
                continue

            console.print()
            console.print(
                f"  [bold cyan]{display_name}[/bold cyan] — {description}\n"
                f"  Get a free key at: [bold underline]{url}[/bold underline]"
            )

            key = Prompt.ask(
                f"  Paste your {display_name} API key (or Enter to skip)",
                default="",
            )

            if key.strip():
                import os
                os.environ[env_var] = key.strip()
                settings.api_keys[provider_id] = key.strip()
                # Also add to pool
                if provider_id not in settings.api_key_pool:
                    settings.api_key_pool[provider_id] = []
                settings.api_key_pool[provider_id].append(key.strip())
                keys_added.append(provider_id)
                console.print(f"  [green]✓[/green] {display_name} key saved!")

                # Ask if they want to add more keys
                _collect_extra_keys(
                    console, settings, provider_id, display_name, url
                )
            else:
                console.print(f"  [dim]Skipped {display_name}[/dim]")

        console.print()

        # Also mention Ollama
        if not keys_added:
            console.print(
                "[yellow]No free API keys added.[/yellow]\n"
                "You can still use Ollama (local) — no key needed.\n"
                "Or add keys later with /setup\n"
            )

        # Pick the best available free provider as starting model
        if "google" in keys_added:
            start_provider = "google"
            start_model = "gemini-3.1-flash"
        elif "groq" in keys_added:
            start_provider = "groq"
            start_model = "llama-3.3-70b-versatile"
        elif "mistral" in keys_added:
            start_provider = "mistral"
            start_model = "codestral-latest"
        else:
            start_provider = "ollama"
            start_model = "llama3.1"

        # Count total free models available
        free_count = 0
        for pid in keys_added:
            free_count += len(_PROVIDER_MODELS.get(pid, []))
        if not keys_added:
            free_count = len(_PROVIDER_MODELS.get("ollama", []))

        settings.model = start_model
        settings.provider = start_provider
        settings.free_mode_enabled = True

        save_config(settings)

        console.print()
        console.print(Panel(
            f"[bold green]Setup complete![/bold green]\n\n"
            f"  Mode:      [bold green]Free Mode (auto-rotate)[/bold green]\n"
            f"  Starting:  [bold]{start_model}[/bold] via {_PROVIDER_DISPLAY.get(start_provider, start_provider)}\n"
            f"  Models:    [bold]{free_count}[/bold] free models available\n\n"
            f"[dim]Free mode will rotate between available free models\n"
            f"when rate limits are reached. Add more keys with /setup\n"
            f"Disable with /free off | Type / to see all commands[/dim]",
            border_style="green",
            expand=False,
        ))
        console.print()

        return settings

    # --- Normal provider selection (choices 2+) ---
    selected_provider = provider_order[choice_num - 2]

    # --- Handle OAuth login ---
    oauth_token: Optional[str] = None
    if selected_provider == "openai-oauth":
        console.print(Panel(
            "[yellow]Experimental:[/yellow] This uses OpenAI's OAuth flow to log in\n"
            "with your ChatGPT Plus/Pro/Team account via browser.\n"
            "No API key needed — it uses your existing subscription.\n\n"
            "[dim]Note: This may not work in all regions or environments.[/dim]",
            border_style="yellow",
            expand=False,
        ))

        if Confirm.ask("Try browser login now?", default=True):
            oauth_token = _try_oauth_login(console)

            if not oauth_token:
                console.print(
                    "[dim]Falling back to API key method. "
                    "Set OPENAI_API_KEY in your environment.[/dim]\n"
                )
                selected_provider = "openai"

    # --- Check if key is needed but missing (for non-free providers) ---
    if selected_provider in ("openai", "anthropic", "google", "groq", "mistral", "xai", "openrouter", "cerebras", "ollama-api"):
        if not available.get(selected_provider, False):
            env_var = _ENV_KEY_NAMES.get(selected_provider, "")

            # Provider-specific setup guide
            setup_guide = ""
            if selected_provider == "ollama-api":
                setup_guide = (
                    "\n[bold cyan]How to get your Ollama API key:[/bold cyan]\n"
                    "  1. Go to [link=https://ollama.com]ollama.com[/link]\n"
                    "  2. Sign in to your account or subscription\n"
                    "  3. Copy your API key and set OLLAMA_API_KEY\n"
                    "  [dim]Access Ollama cloud models via API[/dim]\n"
                )
            elif selected_provider == "xai":
                setup_guide = (
                    "\n[bold cyan]How to get your xAI API key:[/bold cyan]\n"
                    "  1. Go to [link=https://console.x.ai]console.x.ai[/link]\n"
                    "  2. Sign up with email, Google, or X account\n"
                    "  3. Add a payment method (pay-as-you-go)\n"
                    "  4. Go to API Keys → Create new key\n"
                    "  5. Copy the key and paste it below\n"
                    "  [dim]Grok 4.1 Fast costs ~$0.015 per coding session[/dim]\n"
                )
            elif selected_provider == "openrouter":
                setup_guide = (
                    "\n[bold cyan]How to get your OpenRouter API key:[/bold cyan]\n"
                    "  1. Go to [link=https://openrouter.ai]openrouter.ai[/link]\n"
                    "  2. Sign up (no credit card needed!)\n"
                    "  3. Go to Keys → Create Key\n"
                    "  4. Copy the key and paste it below\n"
                    "  [dim]Access 50+ free models with one key[/dim]\n"
                )
            elif selected_provider == "cerebras":
                setup_guide = (
                    "\n[bold cyan]How to get your Cerebras API key:[/bold cyan]\n"
                    "  1. Go to [link=https://cloud.cerebras.ai]cloud.cerebras.ai[/link]\n"
                    "  2. Sign up (no credit card needed!)\n"
                    "  3. Go to API Keys → Generate API Key\n"
                    "  4. Copy the key and paste it below\n"
                    "  [dim]1M free tokens/day, 1000+ tokens/sec[/dim]\n"
                )

            console.print(
                f"\n[yellow]No API key found for "
                f"{_PROVIDER_DISPLAY.get(selected_provider, selected_provider)}.[/yellow]\n"
                f"{setup_guide}"
                f"Set it in your environment before running njss:\n"
                f"  [bold]export {env_var}=your-key-here[/bold]\n"
            )
            if not Confirm.ask("Continue with this provider anyway?", default=False):
                selected_provider = "ollama"
                console.print("[dim]Using Ollama instead.[/dim]\n")

    # --- Model selection ---
    # Map openai-oauth to openai for model selection if OAuth succeeded
    model_provider = selected_provider
    if selected_provider == "openai-oauth":
        model_provider = "openai-oauth"

    # Try fetching live models first, fall back to hardcoded list
    display_name = _PROVIDER_DISPLAY.get(selected_provider, selected_provider)
    live_models = _fetch_live_models(model_provider, settings, token=oauth_token)
    hardcoded = _PROVIDER_MODELS.get(model_provider, _PROVIDER_MODELS.get("openai", []))
    if live_models:
        models = live_models
        console.print(f"  [dim]Found {len(models)} model(s) from {display_name}[/dim]")
    else:
        models = hardcoded

    if models:
        console.print(f"\n[bold]Models for {display_name}:[/bold]")
        model_table = Table(
            show_header=True,
            border_style="dim",
            expand=False,
            padding=(0, 1),
        )
        model_table.add_column("#", style="bold", width=3)
        model_table.add_column("Model", style="bold green")
        model_table.add_column("Description", style="dim")

        for i, (model_name, desc) in enumerate(models, 1):
            model_table.add_row(str(i), model_name, desc)

        console.print(model_table)
        console.print()

        model_choice = Prompt.ask(
            "Pick a model (or type a custom name)",
            default="1",
        )

        if model_choice.isdigit() and 1 <= int(model_choice) <= len(models):
            selected_model = models[int(model_choice) - 1][0]
        else:
            selected_model = model_choice.strip()
    else:
        selected_model = Prompt.ask("Enter model name", default="llama3.1")

    # --- Save ---
    settings.model = selected_model
    settings.free_mode_enabled = False  # Not free mode — explicit provider
    # Map openai-oauth back to openai provider (the router handles the rest)
    if selected_provider == "openai-oauth":
        settings.provider = "openai"
        # Store OAuth token if we got one
        if oauth_token:
            settings.api_keys["openai"] = oauth_token
    else:
        settings.provider = selected_provider

    save_config(settings)

    console.print()
    console.print(Panel(
        f"[bold green]Setup complete![/bold green]\n\n"
        f"  Provider:  [bold]{display_name}[/bold]\n"
        f"  Model:     [bold]{selected_model}[/bold]\n\n"
        f"[dim]Change anytime: /model <name> or /settings\n"
        f"Try /free to switch to free-tier models\n"
        f"Type / to see all available commands[/dim]",
        border_style="green",
        expand=False,
    ))
    console.print()

    return settings

