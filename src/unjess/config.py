"""Configuration system — YAML config, env vars, CLI overrides."""

import os
import platform
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Optional

import yaml


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_CONFIG_DIR = Path.home() / ".unjess"
DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_DIR / "config.yaml"

DEFAULT_MODEL = ""
DEFAULT_PROVIDER = ""  # auto-detect from model name
DEFAULT_MAX_ITERATIONS = 50
DEFAULT_COMMAND_TIMEOUT = 30

IGNORE_PATTERNS: list[str] = [
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build",
    "*.pyc", "*.pyo", ".DS_Store", "Thumbs.db",
]


# ---------------------------------------------------------------------------
# Settings dataclass
# ---------------------------------------------------------------------------

@dataclass
class Settings:
    """All runtime settings for the agent."""

    # --- Model / Provider ---
    model: str = DEFAULT_MODEL
    provider: str = DEFAULT_PROVIDER  # "" means auto-detect
    api_keys: dict[str, str] = field(default_factory=dict)

    # --- Multiple keys per provider (for free-tier rotation) ---
    api_key_pool: dict[str, list[str]] = field(default_factory=dict)

    # --- Provider base URLs (for Ollama / custom endpoints) ---
    ollama_base_url: str = "http://localhost:11434"
    ollama_api_base_url: str = "https://api.ollama.com"
    llamacpp_base_url: str = "http://localhost:8080"
    lmstudio_base_url: str = "http://localhost:1234"
    kimi_base_url: str = "https://api.moonshot.ai/v1"
    qwen_base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

    # --- Workspace ---
    workspace: str = "."  # resolved to absolute at load time

    # --- Fallback chain (list of "provider/model" strings) ---
    fallback_chain: list[str] = field(default_factory=list)

    # --- Mobile Companion & Security ---
    mobile_pin: str = ""  # 4-digit PIN for remote/mobile client access
    enable_pin_auth: bool = False  # False by default so local & mobile open freely unless enabled
    allow_network_access: bool = True  # bind to 0.0.0.0 for local Wi-Fi / mobile access
    network_port: int = 8080

    # --- Behavior ---
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    subagent_max_turns: int = 10  # default max LLM turns per subagent
    confirm_commands: bool = True  # ask before running shell commands
    command_timeout: int = DEFAULT_COMMAND_TIMEOUT
    enable_stuck_detection: bool = True

    # --- Display ---
    verbosity: str = "normal"  # "quiet", "normal", "verbose"
    show_diffs: bool = True
    show_cost: bool = True
    show_stats: bool = True
    accent_color: str = "#666666"  # user-customizable accent / theme color

    # --- Budget ---
    max_cost_per_session: float = 0.0  # 0 = unlimited
    max_tokens_per_message: int = 0  # 0 = unlimited

    # --- Free mode ---
    free_mode_enabled: bool = False  # start in free-tier rotation mode

    # --- Planning Mode ---
    planning_mode: str = "auto"  # "auto", "on", "off"

    # --- Context Compaction & Override ---
    enable_compaction: bool = True
    context_window_override: int = 0  # 0 = auto-detect based on model

    # --- RAG / Embeddings ---
    enable_rag: bool = False  # opt-in: embedding-powered semantic search

    # --- Ignore patterns ---
    ignore_patterns: list[str] = field(default_factory=lambda: list(IGNORE_PATTERNS))


# ---------------------------------------------------------------------------
# Environment variable mapping
# ---------------------------------------------------------------------------

_ENV_KEY_MAP: dict[str, str] = {
    "OPENAI_API_KEY": "openai",
    "ANTHROPIC_API_KEY": "anthropic",
    "GOOGLE_API_KEY": "google",
    "GROQ_API_KEY": "groq",
    "MISTRAL_API_KEY": "mistral",
    "XAI_API_KEY": "xai",
    "OPENROUTER_API_KEY": "openrouter",
    "CEREBRAS_API_KEY": "cerebras",
    "MOONSHOT_API_KEY": "kimi",
    "KIMI_API_KEY": "kimi",
    "DASHSCOPE_API_KEY": "qwen",
    "QWEN_API_KEY": "qwen",
    "OLLAMA_API_KEY": "ollama-api",
}


def _load_env_keys(settings: Settings) -> None:
    """Merge API keys from environment variables into settings."""
    for env_var, provider_name in _ENV_KEY_MAP.items():
        value = os.environ.get(env_var, "")
        if value and provider_name not in settings.api_keys:
            settings.api_keys[provider_name] = value

    # Backward compatibility: if ollama key was saved previously under 'ollama'
    if "ollama-api" not in settings.api_keys:
        legacy_key = settings.api_keys.get("ollama", "")
        if legacy_key and legacy_key != "ollama":
            settings.api_keys["ollama-api"] = legacy_key


# ---------------------------------------------------------------------------
# Key file (separate from config for security)
# ---------------------------------------------------------------------------

_KEYS_PATH = DEFAULT_CONFIG_PATH.parent / "keys.yaml"


def _load_saved_keys() -> tuple[dict[str, str], dict[str, list[str]]]:
    """Load saved API keys and key pool from the keys file.

    Returns:
        Tuple of (active_keys, key_pool).
    """
    if not _KEYS_PATH.exists():
        return {}, {}
    try:
        with open(_KEYS_PATH, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        pool = data.pop("key_pool", {})
        keys = {k: v for k, v in data.items() if isinstance(v, str) and v}
        return keys, pool
    except Exception:
        return {}, {}


def save_keys(
    api_keys: dict[str, str],
    api_key_pool: dict[str, list[str]] | None = None,
    path: Optional[Path] = None,
) -> None:
    """Persist API keys (and optional key pool) to the keys file.

    Only saves non-empty keys. The key pool is stored under a 'key_pool'
    top-level key so it doesn't clash with provider names.

    Args:
        api_keys: Dict mapping provider name to active API key.
        api_key_pool: Optional dict mapping provider to list of all keys.
        path: Override path for testing.
    """
    keys_path = path or _KEYS_PATH
    keys_path.parent.mkdir(parents=True, exist_ok=True)

    # Only save non-empty keys
    to_save: dict = {k: v for k, v in api_keys.items() if v}

    # Save key pool if provided
    if api_key_pool:
        pool_clean = {k: v for k, v in api_key_pool.items() if v}
        if pool_clean:
            to_save["key_pool"] = pool_clean

    if not to_save:
        if keys_path.exists():
            keys_path.write_text("")
        return

    with open(keys_path, "w", encoding="utf-8") as fh:
        yaml.dump(to_save, fh, default_flow_style=False, sort_keys=False)

    # Restrict permissions on the keys file (best-effort)
    try:
        keys_path.chmod(0o600)
    except (OSError, NotImplementedError):
        pass  # Windows doesn't support Unix permissions


# ---------------------------------------------------------------------------
# YAML load / save
# ---------------------------------------------------------------------------

def _settings_from_dict(data: dict) -> Settings:
    """Create Settings from a raw dict (loaded from YAML), ignoring unknown keys."""
    known_fields = {f.name for f in fields(Settings)}
    filtered = {k: v for k, v in data.items() if k in known_fields}
    return Settings(**filtered)


def load_config(path: Optional[Path] = None) -> Settings:
    """Load settings from a YAML file, then overlay saved keys and env vars.

    Load order (later wins): config.yaml → keys.yaml → environment variables.
    """
    config_path = path or DEFAULT_CONFIG_PATH

    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        settings = _settings_from_dict(raw)
    else:
        settings = Settings()

    # Load saved keys (keys.yaml), then env vars override
    saved_keys, saved_pool = _load_saved_keys()
    for provider, key in saved_keys.items():
        if provider not in settings.api_keys:
            settings.api_keys[provider] = key
    # Restore key pool
    for provider, keys in saved_pool.items():
        if provider not in settings.api_key_pool:
            settings.api_key_pool[provider] = keys

    _load_env_keys(settings)

    # Resolve default model if empty
    if not settings.model and settings.provider:
        defaults = {
            "google": "gemini-3.1-flash",
            "openai": "gpt-4o-mini",
            "anthropic": "claude-3-5-sonnet-latest",
            "groq": "llama-3.3-70b-versatile",
            "mistral": "codestral-latest",
            "xai": "grok-2-1212",
            "openrouter": "openrouter/free",
            "cerebras": "zai-glm-4.7",
            "ollama": "llama3.1",
            "ollama-api": "llama3.3",
            "llamacpp": "llamacpp",
        }
        settings.model = defaults.get(settings.provider, "")

    return settings


def save_config(settings: Settings, path: Optional[Path] = None) -> Path:
    """Persist current settings to YAML and save API keys separately."""
    config_path = path or DEFAULT_CONFIG_PATH
    config_path.parent.mkdir(parents=True, exist_ok=True)

    data = asdict(settings)
    # API keys go to keys.yaml, not config.yaml
    api_keys = data.pop("api_keys", {})
    api_key_pool = data.pop("api_key_pool", {})
    # Workspace is session-specific (derived from cwd), don't persist
    data.pop("workspace", None)

    with open(config_path, "w", encoding="utf-8") as fh:
        yaml.dump(data, fh, default_flow_style=False, sort_keys=False)

    # Save keys to separate file
    if api_keys or api_key_pool:
        save_keys(api_keys, api_key_pool=api_key_pool)

    return config_path


def ensure_config(path: Optional[Path] = None) -> Path:
    """Create default config file if it doesn't exist. Returns the path."""
    config_path = path or DEFAULT_CONFIG_PATH
    if not config_path.exists():
        save_config(Settings(), config_path)
    return config_path


def is_first_run(path: Optional[Path] = None) -> bool:
    """Check if this is a first run (no config file exists or no provider/API key configured)."""
    config_path = path or DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return True
    settings = load_config(config_path)
    if settings.provider == "ollama":
        return False
    # Check if we have at least one non-empty API key (configured in keys.yaml or env vars)
    has_keys = any(val.strip() for val in settings.api_keys.values() if val)
    return not has_keys



# ---------------------------------------------------------------------------
# CLI arg overlay
# ---------------------------------------------------------------------------

def apply_cli_overrides(settings: Settings, **overrides: object) -> Settings:
    """Apply CLI argument overrides onto an existing Settings instance.

    Only non-None values are applied.
    """
    for key, value in overrides.items():
        if value is not None and hasattr(settings, key):
            setattr(settings, key, value)
    return settings


# ---------------------------------------------------------------------------
# Workspace-scoped settings
# ---------------------------------------------------------------------------

def load_workspace_config(workspace: Path) -> dict:
    """Load workspace-specific settings from .agents/settings.yaml."""
    ws_config_path = workspace / ".agents" / "settings.yaml"
    if not ws_config_path.is_file():
        return {}
    try:
        with open(ws_config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data
    except Exception:
        return {}


def merge_workspace_config(settings: Settings, workspace: Path) -> Settings:
    """Merge workspace config on top of global settings."""
    ws_data = load_workspace_config(workspace)
    for key, value in ws_data.items():
        if hasattr(settings, key):
            setattr(settings, key, value)
    return settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_workspace(settings: Settings) -> Path:
    """Return the workspace as an absolute Path."""
    return Path(settings.workspace).resolve()


def get_os_info() -> str:
    """Return a human-readable OS string for the system prompt."""
    system = platform.system().lower()
    if system == "windows":
        return "windows"
    elif system == "darwin":
        return "macOS"
    return system


def get_shell() -> str:
    """Detect the user's shell."""
    if platform.system() == "Windows":
        return "powershell"
    return os.environ.get("SHELL", "/bin/bash").rsplit("/", maxsplit=1)[-1]
