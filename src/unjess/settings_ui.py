"""Settings UI — rich panel display, live editing, and YAML persistence."""

from typing import Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from unjess.config import Settings, save_config


# ---------------------------------------------------------------------------
# Settings categories
# ---------------------------------------------------------------------------

_CATEGORIES: dict[str, list[tuple[str, str, str]]] = {
    "Model / Provider": [
        ("model",              "str",   "Active model"),
        ("provider",           "str",   "Provider override (empty = auto-detect)"),
        ("ollama_base_url",    "str",   "Ollama API base URL"),
    ],
    "Behavior": [
        ("max_iterations",     "int",   "Max tool-call iterations per message"),
        ("confirm_commands",   "bool",  "Ask before running shell commands"),
        ("command_timeout",    "int",   "Command timeout in seconds"),
        ("llm_timeout",        "float", "LLM request timeout in seconds (default 300)"),
    ],
    "Display": [
        ("verbosity",          "str",   "Output verbosity (quiet / normal / verbose)"),
        ("show_diffs",         "bool",  "Show diffs after file edits"),
        ("show_cost",          "bool",  "Show cost estimates"),
        ("show_stats",         "bool",  "Show token stats after responses"),
    ],
    "Budget": [
        ("max_cost_per_session", "float", "Max cost per session (0 = unlimited)"),
        ("max_tokens_per_message", "int", "Max tokens per message (0 = unlimited)"),
    ],
}

# Valid values for constrained fields
_VALID_VALUES: dict[str, list[str]] = {
    "verbosity": ["quiet", "normal", "verbose"],
}


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def show_settings(settings: Settings, console: Console) -> None:
    """Display all current settings in a rich panel.

    Args:
        settings: Current settings.
        console: Rich console for output.
    """
    table = Table(
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
        expand=False,
        padding=(0, 1),
    )
    table.add_column("Setting", style="bold")
    table.add_column("Value", style="green")
    table.add_column("Description", style="dim")

    for category, fields in _CATEGORIES.items():
        # Category header row
        table.add_row(f"[bold magenta]── {category} ──[/]", "", "")

        for field_name, field_type, description in fields:
            value = getattr(settings, field_name, "")
            value_str = _format_value(value, field_type, field_name)
            table.add_row(f"  {field_name}", value_str, description)

    panel = Panel(
        table,
        title="⚙️  Settings",
        title_align="left",
        border_style="cyan",
        expand=False,
    )
    console.print(panel)
    console.print(
        "  [dim]Change a setting: /settings <key> <value>[/dim]\n"
        "  [dim]Example: /settings model gpt-4o[/dim]\n"
    )


def _format_value(value: Any, field_type: str, field_name: str = "") -> str:
    """Format a setting value for display."""
    if isinstance(value, bool):
        return "[green]✓[/green]" if value else "[red]✗[/red]"
    if isinstance(value, float):
        if value == 0.0:
            return "[dim]unlimited[/dim]"
        return f"${value:.2f}" if "cost" in field_name else str(value)
    if isinstance(value, int) and value == 0:
        return "[dim]unlimited[/dim]"
    if isinstance(value, str) and not value:
        return "[dim](auto)[/dim]"
    return str(value)


# ---------------------------------------------------------------------------
# Editing
# ---------------------------------------------------------------------------

def edit_setting(settings: Settings, key: str, value_str: str) -> tuple[bool, str]:
    """Validate and apply a setting change.

    Args:
        settings: Current settings to modify.
        key: Setting name.
        value_str: New value as a string.

    Returns:
        (success, message)
    """
    # Check the key exists
    if not hasattr(settings, key):
        # Fuzzy match
        close_matches = [
            k for k in _all_field_names()
            if key.lower() in k.lower() or k.lower() in key.lower()
        ]
        suggestion = f" Did you mean: {', '.join(close_matches)}?" if close_matches else ""
        return False, f"Unknown setting: '{key}'.{suggestion}"

    # Check valid values
    if key in _VALID_VALUES:
        valid = _VALID_VALUES[key]
        if value_str.lower() not in valid:
            return False, f"Invalid value for '{key}'. Must be one of: {', '.join(valid)}"
        value_str = value_str.lower()

    # Type conversion
    field_type = _get_field_type(key)
    try:
        if field_type == "bool":
            value: Any = value_str.lower() in ("true", "1", "yes", "on")
        elif field_type == "int":
            value = int(value_str)
        elif field_type == "float":
            value = float(value_str)
        else:
            value = value_str
    except (ValueError, TypeError):
        return False, f"Invalid value for '{key}': expected {field_type}, got '{value_str}'"

    old_value = getattr(settings, key)
    setattr(settings, key, value)

    return True, f"{key}: {old_value} → {value}"


def save_and_notify(settings: Settings) -> str:
    """Persist settings to YAML and return confirmation.

    Args:
        settings: Settings to persist.

    Returns:
        Confirmation message.
    """
    path = save_config(settings)
    return f"Settings saved to {path}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _all_field_names() -> list[str]:
    """Get all setting field names across categories."""
    names: list[str] = []
    for fields in _CATEGORIES.values():
        for field_name, _, _ in fields:
            names.append(field_name)
    return names


def _get_field_type(key: str) -> str:
    """Get the type string for a setting field."""
    for fields in _CATEGORIES.values():
        for field_name, field_type, _ in fields:
            if field_name == key:
                return field_type
    return "str"
