"""Conversation logging — JSONL transcript writer and cost tracking."""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from unjess.config import DEFAULT_CONFIG_DIR

logger = logging.getLogger(__name__)

_CONVERSATIONS_DIR = DEFAULT_CONFIG_DIR / "conversations"


# ---------------------------------------------------------------------------
# Cost tracking
# ---------------------------------------------------------------------------

# Rates per 1M tokens: (input_cost, output_cost)
# Models with (0.0, 0.0) are explicitly free — not "unknown".
_COST_RATES: dict[str, tuple[float, float]] = {
    # OpenAI
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "o3": (2.00, 8.00),
    "o3-mini": (1.10, 4.40),
    "o4-mini": (1.10, 4.40),
    # Anthropic
    "claude-sonnet-4": (3.00, 15.00),
    "claude-opus-4": (15.00, 75.00),
    "claude-3.5-sonnet": (3.00, 15.00),
    "claude-3.5-haiku": (0.80, 4.00),
    # Google
    "gemini-3.1-flash": (0.15, 0.60),
    "gemini-3.1-pro": (1.25, 10.00),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-3.5-flash": (1.50, 9.00),
    "gemini-3-flash": (0.50, 3.00),
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-3.1-pro": (2.00, 12.00),
    # Groq (free tier — paid tier rates negligible)
    "llama-3.3-70b": (0.0, 0.0),
    "llama-3.1-8b": (0.0, 0.0),
    "llama-3.1-70b": (0.0, 0.0),
    "mixtral-8x7b": (0.0, 0.0),
    "gemma-": (0.0, 0.0),  # Groq Gemma models (not Gemini!)
    # Mistral (free tier for code models)
    "codestral": (0.0, 0.0),
    "mistral-small": (0.0, 0.0),
    "open-mistral-nemo": (0.0, 0.0),
    "mistral-large": (2.00, 6.00),
    # xAI / Grok
    "grok-4.1-fast": (0.20, 0.50),
    "grok-4.3": (1.25, 2.50),
    "grok-3": (3.00, 15.00),
    # OpenRouter (free models)
    "openrouter/free": (0.0, 0.0),
    "nvidia/nemotron-3-ultra:free": (0.0, 0.0),
    "poolside/laguna-m.1:free": (0.0, 0.0),
    "qwen/qwen3-coder:free": (0.0, 0.0),
    # Cerebras (free tier)
    "zai-glm-4.7": (0.0, 0.0),
    # Kimi / Moonshot AI
    "kimi": (0.50, 2.00),
    "moonshot": (0.50, 2.00),
    # Qwen / DashScope
    "qwen-max": (0.40, 1.20),
    "qwen-plus": (0.11, 0.33),
    "qwen-turbo": (0.04, 0.12),
    "qwen2.5": (0.10, 0.20),
    "qwen3": (0.10, 0.20),
    "qwen": (0.20, 0.50),
    # Ollama (always free — local)
    "llama3": (0.0, 0.0),
    "qwen2.5": (0.0, 0.0),
    "deepseek-coder": (0.0, 0.0),
    "codestral:latest": (0.0, 0.0),
}

# Models/prefixes known to be free (for display labeling)
_FREE_PREFIXES: list[str] = [
    "llama-3.", "llama3", "mixtral", "gemma-",  # Groq free tier
    "codestral", "mistral-small", "open-mistral",  # Mistral free tier
    "openrouter/",  # OpenRouter free router
    "glm-",  # Cerebras free tier
    "qwen", "deepseek",  # Ollama local
]

# Providers known to always be free
_FREE_PROVIDERS: set[str] = {"ollama", "lmstudio", "llamacpp"}


def is_free_model(model: str, provider: str = "") -> bool:
    """Check if a model is known to be free.

    Args:
        model: Model name.
        provider: Provider name (optional).

    Returns:
        True if the model is on a free tier or local.
    """
    if provider in _FREE_PROVIDERS:
        return True
    m = model.lower()
    for prefix in _FREE_PREFIXES:
        if m.startswith(prefix):
            return True
    # Check if rates are explicitly (0, 0)
    rates = _COST_RATES.get(model)
    if rates is None:
        for key, val in _COST_RATES.items():
            if m.startswith(key):
                rates = val
                break
    return rates == (0.0, 0.0)


class CostTracker:
    """Accumulates token usage and estimates cost with per-model breakdown."""

    def __init__(self) -> None:
        self.total_tokens_in: int = 0
        self.total_tokens_out: int = 0
        self.total_thinking_tokens: int = 0
        self.total_cache_read_tokens: int = 0
        self.total_cache_creation_tokens: int = 0
        self.total_cost: float = 0.0
        self.llm_calls: int = 0
        self.tool_calls: int = 0
        self._per_model: dict[str, dict[str, Any]] = {}

    def record_llm_call(
        self,
        model: str,
        tokens_in: int,
        tokens_out: int,
        thinking_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_creation_tokens: int = 0,
    ) -> float:
        """Record an LLM call and return the estimated cost.

        Args:
            model: Model name.
            tokens_in: Input tokens.
            tokens_out: Output tokens.
            thinking_tokens: Thinking/reasoning tokens (billed at output rate).
            cache_read_tokens: Cache read tokens (typically free or reduced).
            cache_creation_tokens: Cache creation tokens (billed at input rate).

        Returns:
            Estimated cost for this call.
        """
        self.total_tokens_in += tokens_in
        self.total_tokens_out += tokens_out
        self.total_thinking_tokens += thinking_tokens
        self.total_cache_read_tokens += cache_read_tokens
        self.total_cache_creation_tokens += cache_creation_tokens
        self.llm_calls += 1

        cost = self._estimate_cost(model, tokens_in, tokens_out, thinking_tokens)
        self.total_cost += cost

        # Per-model tracking
        if model not in self._per_model:
            self._per_model[model] = {
                "tokens_in": 0, "tokens_out": 0,
                "thinking_tokens": 0,
                "cache_read_tokens": 0,
                "cache_creation_tokens": 0,
                "cost": 0.0, "calls": 0,
                "is_free": is_free_model(model),
            }
        entry = self._per_model[model]
        entry["tokens_in"] += tokens_in
        entry["tokens_out"] += tokens_out
        entry["thinking_tokens"] += thinking_tokens
        entry["cache_read_tokens"] += cache_read_tokens
        entry["cache_creation_tokens"] += cache_creation_tokens
        entry["cost"] += cost
        entry["calls"] += 1

        # Auto-persist to disk
        self._persist_call(model, tokens_in, tokens_out, thinking_tokens, cost)

        return cost

    def record_tool_call(self) -> None:
        """Record a tool call."""
        self.tool_calls += 1

    def estimate_cost(self, model: str, tokens_in: int, tokens_out: int, thinking_tokens: int = 0) -> float:
        """Estimate cost for the given model and token counts.

        Public API — delegates to the internal rate lookup.

        Args:
            model: Model name.
            tokens_in: Input token count.
            tokens_out: Output token count.
            thinking_tokens: Thinking/reasoning token count.

        Returns:
            Estimated cost in USD.
        """
        return self._estimate_cost(model, tokens_in, tokens_out, thinking_tokens)

    def _estimate_cost(self, model: str, tokens_in: int, tokens_out: int, thinking_tokens: int = 0) -> float:
        """Estimate cost based on known rates.

        Thinking tokens are billed at the output token rate.
        Cache read tokens are free (not included in cost).
        Cache creation tokens are billed at the input token rate (handled externally).
        """
        rates = _COST_RATES.get(model)
        if rates is None:
            # Try prefix match
            for key, val in _COST_RATES.items():
                if model.startswith(key):
                    rates = val
                    break
        if rates is None:
            return 0.0

        cost_in, cost_out = rates
        return (
            (tokens_in / 1_000_000) * cost_in
            + (tokens_out / 1_000_000) * cost_out
            + (thinking_tokens / 1_000_000) * cost_out
        )

    def is_session_free(self) -> bool:
        """Whether all calls so far have been to free models."""
        return all(e["is_free"] for e in self._per_model.values())

    def summary(self) -> dict[str, Any]:
        """Return a summary dict of session costs."""
        return {
            "tokens_in": self.total_tokens_in,
            "tokens_out": self.total_tokens_out,
            "thinking_tokens": self.total_thinking_tokens,
            "cache_read_tokens": self.total_cache_read_tokens,
            "cache_creation_tokens": self.total_cache_creation_tokens,
            "total_tokens": self.total_tokens_in + self.total_tokens_out + self.total_thinking_tokens,
            "estimated_cost": round(self.total_cost, 6),
            "is_free": self.is_session_free(),
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
            "per_model": dict(self._per_model),
        }

    def _persist_call(
        self,
        model: str,
        tokens_in: int,
        tokens_out: int,
        thinking_tokens: int,
        cost: float,
    ) -> None:
        """Persist a single LLM call to ~/.unjess/cost_history.jsonl.

        Called automatically after each LLM call so usage data
        survives app restarts.
        """
        history_file = Path.home() / ".unjess" / "cost_history.jsonl"
        history_file.parent.mkdir(parents=True, exist_ok=True)

        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "thinking": thinking_tokens,
            "cost": round(cost, 6),
        }
        try:
            with open(history_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            logger.debug("Failed to persist cost entry", exc_info=True)

    def persist_cost(self) -> None:
        """Legacy method — no longer needed (calls auto-persist now)."""
        pass

    @staticmethod
    def get_usage_summary() -> dict[str, Any]:
        """Aggregate usage stats from cost_history.jsonl.

        Returns a dict with keys:
            today: {cost, tokens_in, tokens_out, calls}
            month: {cost, tokens_in, tokens_out, calls}
            all_time: {cost, tokens_in, tokens_out, calls}
            per_model: {model: {cost, tokens_in, tokens_out, calls}}
        """
        history_file = Path.home() / ".unjess" / "cost_history.jsonl"
        now = datetime.now(timezone.utc)
        today_prefix = now.strftime("%Y-%m-%d")
        month_prefix = now.strftime("%Y-%m")

        buckets: dict[str, dict[str, Any]] = {
            "today": {"cost": 0.0, "tokens_in": 0, "tokens_out": 0, "calls": 0},
            "month": {"cost": 0.0, "tokens_in": 0, "tokens_out": 0, "calls": 0},
            "all_time": {"cost": 0.0, "tokens_in": 0, "tokens_out": 0, "calls": 0},
        }
        per_model: dict[str, dict[str, Any]] = {}

        if not history_file.exists():
            return {**buckets, "per_model": per_model}

        try:
            for line in history_file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                entry = json.loads(line)
                ts = entry.get("ts", entry.get("timestamp", ""))
                cost = entry.get("cost", 0.0)
                t_in = entry.get("tokens_in", 0)
                t_out = entry.get("tokens_out", 0)
                model = entry.get("model", "unknown")

                # All-time
                buckets["all_time"]["cost"] += cost
                buckets["all_time"]["tokens_in"] += t_in
                buckets["all_time"]["tokens_out"] += t_out
                buckets["all_time"]["calls"] += 1

                # Month
                if ts.startswith(month_prefix):
                    buckets["month"]["cost"] += cost
                    buckets["month"]["tokens_in"] += t_in
                    buckets["month"]["tokens_out"] += t_out
                    buckets["month"]["calls"] += 1

                # Today
                if ts.startswith(today_prefix):
                    buckets["today"]["cost"] += cost
                    buckets["today"]["tokens_in"] += t_in
                    buckets["today"]["tokens_out"] += t_out
                    buckets["today"]["calls"] += 1

                # Per-model
                if model not in per_model:
                    per_model[model] = {"cost": 0.0, "tokens_in": 0, "tokens_out": 0, "calls": 0}
                per_model[model]["cost"] += cost
                per_model[model]["tokens_in"] += t_in
                per_model[model]["tokens_out"] += t_out
                per_model[model]["calls"] += 1
        except Exception:
            logger.debug("Failed to read cost history", exc_info=True)

        return {**buckets, "per_model": per_model}

    @staticmethod
    def get_daily_cost() -> float:
        """Sum costs from today's sessions."""
        stats = CostTracker.get_usage_summary()
        return stats["today"]["cost"]

    @staticmethod
    def get_monthly_cost() -> float:
        """Sum costs from this month's sessions."""
        stats = CostTracker.get_usage_summary()
        return stats["month"]["cost"]


# ---------------------------------------------------------------------------
# Log entry types
# ---------------------------------------------------------------------------

class LogType:
    """Constants for log entry types."""

    USER_INPUT = "USER_INPUT"
    MODEL_RESPONSE = "MODEL_RESPONSE"
    TOOL_CALL = "TOOL_CALL"
    TOOL_RESULT = "TOOL_RESULT"
    ERROR = "ERROR"
    SYSTEM = "SYSTEM"


# ---------------------------------------------------------------------------
# Conversation logger
# ---------------------------------------------------------------------------

class ConversationLogger:
    """Append-only JSONL conversation logger.

    Each session gets a unique conversation ID. All steps are logged
    to ``~/.unjess/conversations/{conversation_id}/transcript.jsonl``.

    Args:
        conversation_id: Optional fixed ID. If None, a UUID is generated.
    """

    def __init__(self, conversation_id: Optional[str] = None) -> None:
        self.conversation_id = conversation_id or uuid.uuid4().hex[:12]
        self._step_index = 0
        self._log_dir = _CONVERSATIONS_DIR / self.conversation_id
        self._log_file = self._log_dir / "transcript.jsonl"
        self.cost_tracker = CostTracker()

        # Ensure directory exists
        self._log_dir.mkdir(parents=True, exist_ok=True)

    @property
    def log_path(self) -> Path:
        """Path to the transcript file."""
        return self._log_file

    # ----- Logging methods -----

    def log_user_input(self, content: str) -> None:
        """Log a user message."""
        self._write_entry(LogType.USER_INPUT, content=content)

    def log_model_response(
        self,
        content: str,
        tool_calls: Optional[list[dict[str, Any]]] = None,
        tokens_in: int = 0,
        tokens_out: int = 0,
        thinking_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_creation_tokens: int = 0,
        model: str = "",
    ) -> None:
        """Log a model response."""
        cost = self.cost_tracker.record_llm_call(
            model, tokens_in, tokens_out,
            thinking_tokens=thinking_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_creation_tokens=cache_creation_tokens,
        )

        self._write_entry(
            LogType.MODEL_RESPONSE,
            content=content,
            tool_calls=tool_calls,
            usage={
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "thinking_tokens": thinking_tokens,
                "cache_read_tokens": cache_read_tokens,
                "cache_creation_tokens": cache_creation_tokens,
                "cost": round(cost, 6),
            },
            model=model,
        )

    def log_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> None:
        """Log a tool invocation."""
        self.cost_tracker.record_tool_call()
        self._write_entry(
            LogType.TOOL_CALL,
            content=f"Calling {tool_name}",
            tool_name=tool_name,
            arguments=arguments,
        )

    def log_tool_result(
        self,
        tool_name: str,
        result: str,
        duration_ms: Optional[int] = None,
    ) -> None:
        """Log a tool execution result."""
        self._write_entry(
            LogType.TOOL_RESULT,
            content=result[:500],  # truncate for log readability
            tool_name=tool_name,
            duration_ms=duration_ms,
        )

    def log_error(self, source: str, error: str) -> None:
        """Log an error."""
        self._write_entry(LogType.ERROR, content=error, source=source)

    def log_system(self, message: str) -> None:
        """Log a system event."""
        self._write_entry(LogType.SYSTEM, content=message)

    # ----- Internal -----

    def _write_entry(self, entry_type: str, **fields: Any) -> None:
        """Append a JSON line to the transcript."""
        self._step_index += 1

        entry: dict[str, Any] = {
            "step_index": self._step_index,
            "type": entry_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        entry.update(fields)

        try:
            with open(self._log_file, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        except Exception as exc:
            # Logging should never crash the agent
            logger.warning("Failed to write log entry: %s", exc)
