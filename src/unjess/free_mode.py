"""Free-tier mode — smart model switching to stay within free usage limits.

Tracks usage against known free-tier quotas, switches models when
approaching limits, and hands off context between models via a brain file.
"""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from unjess.config import Settings
    from unjess.display import Display
    from unjess.llm.router import ProviderRouter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Free-tier definitions (hardcoded defaults, overridable in config)
# ---------------------------------------------------------------------------

@dataclass
class FreeTier:
    """A free-tier model with its quota limits."""

    provider: str
    model: str
    display_name: str
    daily_request_limit: int = 0    # 0 = unlimited
    daily_token_limit: int = 0      # 0 = unlimited
    rpm_limit: int = 0              # requests per minute, 0 = unlimited
    priority: int = 0               # lower = preferred

    @property
    def has_limits(self) -> bool:
        """Whether this tier has any usage limits."""
        return self.daily_request_limit > 0 or self.daily_token_limit > 0


# Known free tiers (as of mid-2026, best-effort)
# Cerebras first (ultra-fast, reliable tool calling), then Gemini Flash,
# then Groq/Mistral. OpenRouter last — its free router picks random models
# that often struggle with tool calling.
_FREE_TIERS: list[FreeTier] = [
    FreeTier(
        provider="cerebras",
        model="zai-glm-4.7",
        display_name="Cerebras GLM-4.7 (ultra-fast)",
        daily_request_limit=0,
        daily_token_limit=1_000_000,
        rpm_limit=30,
        priority=0,
    ),
    FreeTier(
        provider="google",
        model="gemini-3.1-flash",
        display_name="Gemini 3.1 Flash",
        daily_request_limit=1500,
        daily_token_limit=1_000_000,
        rpm_limit=15,
        priority=1,
    ),
    FreeTier(
        provider="groq",
        model="llama-3.3-70b-versatile",
        display_name="Groq Llama 3.3 70B",
        daily_request_limit=14400,
        daily_token_limit=500_000,
        rpm_limit=30,
        priority=2,
    ),
    FreeTier(
        provider="mistral",
        model="codestral-latest",
        display_name="Mistral Codestral",
        daily_request_limit=0,  # per-minute limited, not daily
        daily_token_limit=0,
        rpm_limit=30,
        priority=3,
    ),
    FreeTier(
        provider="google",
        model="gemini-2.5-pro",
        display_name="Gemini 2.5 Pro",
        daily_request_limit=50,
        daily_token_limit=1_000_000,
        rpm_limit=5,
        priority=4,
    ),
    FreeTier(
        provider="groq",
        model="llama-3.1-8b-instant",
        display_name="Groq Llama 3.1 8B",
        daily_request_limit=14400,
        daily_token_limit=500_000,
        rpm_limit=30,
        priority=5,
    ),
    FreeTier(
        provider="mistral",
        model="open-mistral-nemo",
        display_name="Mistral Nemo 128K",
        daily_request_limit=0,
        daily_token_limit=0,
        rpm_limit=30,
        priority=6,
    ),
    FreeTier(
        provider="openrouter",
        model="openrouter/free",
        display_name="OpenRouter Free (fallback)",
        daily_request_limit=0,
        daily_token_limit=0,
        rpm_limit=20,
        priority=8,
    ),
    FreeTier(
        provider="ollama",
        model="llama3.1",
        display_name="Ollama Llama 3.1 (local)",
        priority=10,
    ),
    FreeTier(
        provider="ollama",
        model="qwen2.5-coder",
        display_name="Ollama Qwen 2.5 Coder (local)",
        priority=11,
    ),
]


# ---------------------------------------------------------------------------
# Usage tracking
# ---------------------------------------------------------------------------

@dataclass
class DailyUsage:
    """Tracks daily usage for a specific model."""

    model: str
    date: str = ""  # YYYY-MM-DD
    requests: int = 0
    tokens: int = 0

    def __post_init__(self) -> None:
        if not self.date:
            self.date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def is_today(self) -> bool:
        """Check if this usage is for today."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self.date == today

    def reset_if_new_day(self) -> None:
        """Reset counters if the day has changed."""
        if not self.is_today():
            self.date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            self.requests = 0
            self.tokens = 0


@dataclass
class UsageTracker:
    """Tracks usage across all free-tier models."""

    usage: dict[str, DailyUsage] = field(default_factory=dict)

    def record(self, model: str, tokens: int = 0) -> None:
        """Record a request and tokens for a model."""
        if model not in self.usage:
            self.usage[model] = DailyUsage(model=model)

        entry = self.usage[model]
        entry.reset_if_new_day()
        entry.requests += 1
        entry.tokens += tokens

    def get_usage(self, model: str) -> DailyUsage:
        """Get today's usage for a model."""
        if model not in self.usage:
            self.usage[model] = DailyUsage(model=model)

        entry = self.usage[model]
        entry.reset_if_new_day()
        return entry

    def remaining_pct(self, model: str, tier: FreeTier) -> float:
        """Get the remaining percentage of quota (0.0 to 1.0).

        Returns 1.0 if the tier has no limits.
        """
        if not tier.has_limits:
            return 1.0

        usage = self.get_usage(model)
        pcts: list[float] = []

        if tier.daily_request_limit > 0:
            pcts.append(1.0 - (usage.requests / tier.daily_request_limit))
        if tier.daily_token_limit > 0:
            pcts.append(1.0 - (usage.tokens / tier.daily_token_limit))

        return min(pcts) if pcts else 1.0


# ---------------------------------------------------------------------------
# Brain file (context handoff between models)
# ---------------------------------------------------------------------------

_BRAIN_DIR = Path.home() / ".unjess" / "brain"


def _save_brain(session_summary: str, session_id: str = "") -> Path:
    """Save conversation summary to brain file for model handoff.

    Args:
        session_summary: Compressed context from current conversation.
        session_id: Optional session identifier.

    Returns:
        Path to the saved brain file.
    """
    _BRAIN_DIR.mkdir(parents=True, exist_ok=True)

    if not session_id:
        session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    brain_path = _BRAIN_DIR / f"{session_id}.md"
    brain_path.write_text(
        f"# Session Context (auto-saved by /free)\n\n{session_summary}\n",
        encoding="utf-8",
    )
    return brain_path


def _load_latest_brain() -> str:
    """Load the most recent brain file.

    Returns:
        Content of the latest brain file, or empty string.
    """
    if not _BRAIN_DIR.exists():
        return ""

    brain_files = sorted(_BRAIN_DIR.glob("*.md"), reverse=True)
    if not brain_files:
        return ""

    try:
        return brain_files[0].read_text(encoding="utf-8", errors="replace")[:3000]
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Free Mode
# ---------------------------------------------------------------------------

class FreeMode:
    """Smart free-tier model switching.

    When activated, switches to the best available free-tier model,
    tracks usage, and automatically switches when approaching limits.

    Args:
        settings: Runtime settings.
        router: LLM provider router.
        display: Display layer for output.
    """

    def __init__(
        self,
        settings: "Settings",
        router: "ProviderRouter",
        display: "Display",
    ) -> None:
        self._settings = settings
        self._router = router
        self._display = display
        self._tracker = UsageTracker()
        self._active = False
        self._original_model = ""
        self._original_provider = ""
        self._current_tier: Optional[FreeTier] = None
        # Key rotation: track current key index per provider
        self._key_index: dict[str, int] = {}

    def rotate_key(self, provider: str) -> bool:
        """Rotate to the next API key for a provider.

        Called when a rate limit is hit. Returns True if a new key
        was activated, False if no more keys available.
        """
        pool = self._settings.api_key_pool.get(provider, [])
        if len(pool) <= 1:
            return False

        current_idx = self._key_index.get(provider, 0)
        next_idx = (current_idx + 1) % len(pool)
        self._key_index[provider] = next_idx

        # Swap the active key
        self._settings.api_keys[provider] = pool[next_idx]

        # Tell the router to reload providers with the new key
        if hasattr(self._router, "reload_providers"):
            self._router.reload_providers()

        self._display.show_info(
            f"🔄 Rotated {provider} API key ({next_idx + 1}/{len(pool)})"
        )
        return True

    @property
    def is_active(self) -> bool:
        """Whether free mode is currently active."""
        return self._active

    def activate(self) -> None:
        """Turn on free mode — switch to best free CLOUD option."""
        if self._active:
            self._display.show_info("Free mode is already active.")
            self.show_status()
            return

        # Save original settings
        self._original_model = self._settings.model
        self._original_provider = self._settings.provider

        # Find best available free tier (cloud only — skip Ollama local)
        tier = self._pick_best_cloud_tier()
        if tier is None:
            # No cloud keys — guide the user
            self._display.show_error(
                "No free cloud API keys configured.\n\n"
                "  /free needs a free API key from one of these (all free):\n\n"
                "    🟣 OpenRouter    →  openrouter.ai          (50+ free models) ⭐\n"
                "    ⚡ Cerebras      →  cloud.cerebras.ai      (ultra-fast, 1M tok/day)\n"
                "    🟢 Google Gemini →  ai.google.dev          (best free tier)\n"
                "    🟠 Groq         →  console.groq.com       (fastest inference)\n"
                "    🔵 Mistral      →  console.mistral.ai     (Codestral free)\n\n"
                "  Get a key, then run /setup to configure it.\n"
                "  Or use /model to switch models manually."
            )
            return

        self._switch_to(tier)
        self._active = True

        self._display.show_info(
            f"🆓 Free mode activated — using {tier.display_name}\n"
            f"  Original model saved: {self._original_model}\n"
            f"  Use /free off to return to your previous model."
        )
        self.show_status()

    def deactivate(self) -> None:
        """Turn off free mode — revert to configured model."""
        if not self._active:
            self._display.show_info("Free mode is not active.")
            return

        self._settings.model = self._original_model
        self._settings.provider = self._original_provider
        self._active = False
        self._current_tier = None

        self._display.show_info(
            f"Free mode deactivated — restored model: {self._original_model}"
        )

    def check_and_switch(self, tokens_used: int = 0) -> Optional[str]:
        """Check quota and switch models if needed.

        Call this before each LLM request when free mode is active.

        Args:
            tokens_used: Tokens consumed in the last request.

        Returns:
            New model name if switched, None otherwise.
        """
        if not self._active or self._current_tier is None:
            return None

        # Record usage
        self._tracker.record(self._current_tier.model, tokens_used)

        # Check remaining quota
        remaining = self._tracker.remaining_pct(
            self._current_tier.model, self._current_tier
        )

        # Switch if below 10% remaining
        if remaining < 0.10 and self._current_tier.has_limits:
            self._display.show_warning(
                f"Approaching free-tier limit for {self._current_tier.display_name} "
                f"({remaining:.0%} remaining). Switching models..."
            )

            # Build brain summary before switching
            brain_summary = self._build_switch_summary()
            if brain_summary:
                brain_path = _save_brain(brain_summary)
                self._display.show_info(f"Context saved to {brain_path}")

            # Find next tier
            next_tier = self._pick_best_tier(exclude=self._current_tier.model)
            if next_tier:
                self._switch_to(next_tier)
                self._display.show_info(
                    f"Switched to {next_tier.display_name}"
                )
                return next_tier.model
            else:
                self._display.show_warning(
                    "No more free tiers available. "
                    "Staying on current model until limit is hit."
                )

        return None

    def show_status(self) -> None:
        """Display current free-tier status and usage."""
        from rich.table import Table
        from rich.panel import Panel

        table = Table(
            show_header=True,
            border_style="dim",
            expand=False,
            padding=(0, 1),
        )
        table.add_column("Model", style="bold", no_wrap=True)
        table.add_column("Usage", justify="right")
        table.add_column("", width=10)  # Status

        available_providers = self._router.available_providers

        for tier in sorted(_FREE_TIERS, key=lambda t: t.priority):
            is_available = tier.provider in available_providers
            is_current = (
                self._current_tier is not None
                and tier.model == self._current_tier.model
            )

            usage = self._tracker.get_usage(tier.model)
            remaining = self._tracker.remaining_pct(tier.model, tier)

            # Compact usage string
            if tier.has_limits:
                parts: list[str] = []
                if tier.daily_request_limit > 0:
                    parts.append(f"{usage.requests}/{tier.daily_request_limit}r")
                if tier.daily_token_limit > 0:
                    tok_k = tier.daily_token_limit // 1000
                    used_k = usage.tokens // 1000
                    parts.append(f"{used_k}/{tok_k}K")

                if remaining > 0.5:
                    color = "green"
                elif remaining > 0.2:
                    color = "yellow"
                else:
                    color = "red"
                usage_str = f"[{color}]{' '.join(parts)}[/{color}]"
            else:
                usage_str = "[dim]∞[/dim]"

            if is_current:
                status = "[bold green]◉ Active[/bold green]"
                # Shorten display name for active model
                name = tier.display_name
            elif is_available:
                status = "[dim]Ready[/dim]"
                name = tier.display_name
            else:
                status = "[dim red]No key[/dim red]"
                name = f"[dim]{tier.display_name}[/dim]"

            table.add_row(name, usage_str, status)

        panel = Panel(
            table,
            title="🆓 Free-Tier Status",
            title_align="left",
            border_style="green" if self._active else "dim",
            expand=False,
        )
        self._display.console.print(panel)
        self._display.console.print(
            "  [dim]/model[/dim] switch free model  "
            "[dim]/provider[/dim] switch provider  "
            "[dim]/free off[/dim] exit free mode"
        )

    # ----- Internal helpers -----

    def _pick_best_tier(self, exclude: str = "") -> Optional[FreeTier]:
        """Pick the best available free tier (including local).

        Args:
            exclude: Model name to skip (e.g., current exhausted model).

        Returns:
            Best FreeTier, or None if nothing available.
        """
        available_providers = self._router.available_providers

        candidates = [
            t for t in sorted(_FREE_TIERS, key=lambda t: t.priority)
            if t.provider in available_providers
            and t.model != exclude
            and self._tracker.remaining_pct(t.model, t) > 0.05
        ]

        return candidates[0] if candidates else None

    def _pick_best_cloud_tier(self, exclude: str = "") -> Optional[FreeTier]:
        """Pick the best available free CLOUD tier (excludes Ollama/local).

        Args:
            exclude: Model name to skip.

        Returns:
            Best cloud FreeTier, or None if no cloud keys configured.
        """
        available_providers = self._router.available_providers
        local_providers = {"ollama"}

        candidates = [
            t for t in sorted(_FREE_TIERS, key=lambda t: t.priority)
            if t.provider in available_providers
            and t.provider not in local_providers
            and t.model != exclude
            and self._tracker.remaining_pct(t.model, t) > 0.05
        ]

        return candidates[0] if candidates else None

    def _switch_to(self, tier: FreeTier) -> None:
        """Switch the active model to a free tier."""
        self._settings.model = tier.model
        self._settings.provider = tier.provider
        self._current_tier = tier

    def _build_switch_summary(self) -> str:
        """Build a context summary for model handoff.

        Returns a brief summary that can be injected into the next
        model's context so it understands what's been happening.
        """
        # For now, return a simple message. In the future, this could
        # use the current model to summarize the conversation.
        return (
            f"Model switch summary (auto-generated by /free mode):\n"
            f"- Previous model: {self._settings.model}\n"
            f"- Reason: approaching free-tier quota limit\n"
            f"- Timestamp: {datetime.now(timezone.utc).isoformat()}\n"
        )
