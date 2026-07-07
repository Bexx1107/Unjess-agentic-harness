"""Context management — token counting, budget tracking, auto-truncation, compaction."""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Known context windows (max tokens) per model family
# ---------------------------------------------------------------------------

_CONTEXT_WINDOWS: dict[str, int] = {
    # OpenAI
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4.1": 1_000_000,
    "gpt-4.1-mini": 1_000_000,
    "gpt-4.1-nano": 1_000_000,
    "o3": 200_000,
    "o3-mini": 200_000,
    "o4-mini": 200_000,
    # Anthropic
    "claude-sonnet-4": 200_000,
    "claude-opus-4": 200_000,
    "claude-3.5-sonnet": 200_000,
    "claude-3.5-haiku": 200_000,
    # Google
    "gemini-2.5-flash": 1_000_000,
    "gemini-2.5-pro": 1_000_000,
    "gemini-2.0-flash": 1_000_000,
    # Groq
    "llama-3.3-70b": 128_000,
    "llama-3.1-8b": 128_000,
    "llama-3.1-70b": 128_000,
    "mixtral-8x7b": 32_768,
    "gemma": 8_192,
    # Mistral
    "codestral": 256_000,
    "mistral-small": 128_000,
    "open-mistral-nemo": 128_000,
    "mistral-large": 128_000,
    # xAI / Grok
    "grok-4.1-fast": 131_072,
    "grok-4.3": 131_072,
    "grok-3": 131_072,
    # OpenRouter
    "openrouter/free": 200_000,
    "nvidia/nemotron-3-ultra:free": 1_000_000,
    "poolside/laguna-m.1:free": 256_000,
    "qwen/qwen3-coder:free": 1_000_000,
    # Cerebras
    "zai-glm-4.7": 131_072,
    # Ollama / local models
    "qwen2.5:14b": 32_768,
    "qwen2.5:7b": 32_768,
    "qwen2.5-coder": 32_768,
    "llama3": 8_192,
    "llama3.1": 128_000,
    "deepseek-coder": 16_384,
    # DeepSeek
    "deepseek-v4-flash": 128_000,
    "deepseek-v3": 128_000,
    "deepseek-r1": 128_000,
    "deepseek-chat": 128_000,
}

# Reserve this fraction of context for the model's response
_RESPONSE_RESERVE = 0.15


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

def count_tokens(text: str, model: str = "") -> int:
    """Count tokens in a text string.

    Uses ``tiktoken`` for OpenAI models, falls back to a chars/4 heuristic.

    Args:
        text: The text to count.
        model: Model name (used to pick the right tokenizer).

    Returns:
        Estimated token count.
    """
    # Try tiktoken for OpenAI models
    if model.startswith(("gpt-", "o1", "o3", "o4")):
        try:
            import tiktoken
            try:
                enc = tiktoken.encoding_for_model(model)
            except KeyError:
                enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except ImportError:
            pass

    # Heuristic fallback: ~4 characters per token
    return max(1, len(text) // 4)


def count_messages_tokens(messages: list[dict[str, Any]], model: str = "") -> int:
    """Count the total tokens across a list of messages.

    Args:
        messages: List of message dicts (OpenAI format).
        model: Model name.

    Returns:
        Total estimated token count.
    """
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += count_tokens(content, model)
        # Tool calls in assistant messages
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                total += count_tokens(json.dumps(tc), model)
        # Overhead per message (~4 tokens for role + formatting)
        total += 4
    return total


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

class ContextManager:
    """Manages context window budget, truncation, and compaction.

    Args:
        model: The active model name.
    """

    def __init__(
        self,
        model: str = "",
        auto_compact_threshold: float = 0.80,
        max_conversation_tokens: int = 80_000,
    ) -> None:
        self._model = model
        self._mode: str = "auto"
        self._auto_compact_threshold = auto_compact_threshold
        self._max_conversation_tokens = max_conversation_tokens

    @property
    def model(self) -> str:
        """Current model."""
        return self._model

    @model.setter
    def model(self, value: str) -> None:
        self._model = value

    def get_context_window(self) -> int:
        """Return the context window size for the current model."""
        # Exact match
        window = _CONTEXT_WINDOWS.get(self._model)
        if window:
            return window

        # Prefix match
        for key, val in _CONTEXT_WINDOWS.items():
            if self._model.startswith(key):
                return val

        # Default
        if ":" in self._model:
            return 32_768  # Ollama default
        return 128_000  # Assume large for unknown models

    def get_available_budget(self, system_prompt: str, messages: list[dict[str, Any]]) -> int:
        """Calculate remaining token budget after system prompt and messages.

        Args:
            system_prompt: The assembled system prompt.
            messages: Current conversation messages.

        Returns:
            Remaining tokens available (after reserving space for the response).
        """
        window = self.get_context_window()
        reserve = int(window * _RESPONSE_RESERVE)

        sys_tokens = count_tokens(system_prompt, self._model)
        msg_tokens = count_messages_tokens(messages, self._model)

        available = window - reserve - sys_tokens - msg_tokens
        return max(0, available)

    def truncate_conversation(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str,
    ) -> list[dict[str, Any]]:
        """Truncate the oldest messages to fit within the context budget.

        Always preserves the first user message (for task continuity)
        and the most recent messages.

        Args:
            messages: Conversation messages (oldest first).
            system_prompt: The system prompt (to account for its tokens).

        Returns:
            Truncated message list.
        """
        window = self.get_context_window()
        reserve = int(window * _RESPONSE_RESERVE)
        sys_tokens = count_tokens(system_prompt, self._model)
        budget = window - reserve - sys_tokens

        if budget <= 0:
            logger.warning("System prompt alone exceeds context budget!")
            return messages[-2:]  # keep only the most recent exchange

        # Check if we even need to truncate
        total = count_messages_tokens(messages, self._model)
        if total <= budget:
            return messages  # fits fine

        # Truncation strategy: keep first message + most recent messages
        # Drop middle messages until we fit
        if len(messages) <= 2:
            return messages

        # Always keep the first message
        first = messages[0]
        rest = messages[1:]

        # Remove from the front of `rest` until we fit
        total = count_messages_tokens([first] + rest, self._model)
        while rest and total > budget:
            # Group tool call/result pairs into atomic units.
            # If the next message is an assistant with tool_calls, also remove
            # the subsequent tool-role messages that reference those calls.
            # If the next message is a tool result, also remove its companion
            # assistant message if that assistant has no other remaining results.
            removed_unit: list[dict[str, Any]] = []
            head = rest[0]

            if head.get("role") == "assistant" and head.get("tool_calls"):
                # Remove the assistant message and all following tool results
                # that belong to this call group.
                call_ids = {tc.get("id") for tc in head["tool_calls"] if tc.get("id")}
                removed_unit.append(rest.pop(0))
                while rest and rest[0].get("role") == "tool" and rest[0].get("tool_call_id") in call_ids:
                    removed_unit.append(rest.pop(0))
            elif head.get("role") == "tool":
                # Orphaned tool result — remove it.
                removed_unit.append(rest.pop(0))
            else:
                removed_unit.append(rest.pop(0))

            removed_tokens = count_messages_tokens(removed_unit, self._model)
            total -= removed_tokens
            for rm in removed_unit:
                logger.debug("Truncated message: %s...", str(rm.get("content", ""))[:50])

        result = [first] + rest

        if len(result) < len(messages):
            # Insert a truncation notice
            truncated_count = len(messages) - len(result)
            notice = {
                "role": "user",
                "content": f"[{truncated_count} earlier messages were truncated to fit the context window]",
            }
            result.insert(1, notice)

        return result

    def compact(
        self,
        messages: list[dict[str, Any]],
        summarize_fn: Callable[[str], str] | None = None,
        keep_recent: int = 4,
    ) -> list[dict[str, Any]]:
        """Aggressively compact conversation by summarizing old messages.

        If a ``summarize_fn`` is provided, it's called to generate a summary
        of old messages via a cheap LLM call. Otherwise, a simple
        concatenation + truncation is used.

        The split point between old (summarized) and recent (kept) messages
        is chosen to avoid breaking tool call/response groups.  Gemini
        requires that an assistant message with ``tool_calls`` is immediately
        followed by the corresponding ``tool`` result messages.

        Args:
            messages: Current conversation messages.
            summarize_fn: Optional callable(text) -> str for LLM summarization.
            keep_recent: Minimum number of recent messages to keep intact.

        Returns:
            Compacted message list.
        """
        if len(messages) <= keep_recent:
            return messages  # too few to compact

        # Find a valid split point: we want at least `keep_recent` messages
        # at the end, but we must not split inside a tool call group.
        # A valid split point is an index where messages[index] is NOT
        # a tool result, and messages[index-1] is NOT an assistant with
        # pending tool_calls whose results haven't appeared yet.
        split = len(messages) - keep_recent

        # Walk backwards from the initial split to find a safe boundary.
        # A safe boundary is a user message, or a text-only assistant message.
        while split > 1:
            msg = messages[split]
            role = msg.get("role", "")

            # If this message is a tool result, we can't start here —
            # its parent assistant message would be missing.
            if role == "tool":
                split -= 1
                continue

            # If this is an assistant with tool_calls, we can't start here —
            # the tool results that follow would be orphaned from context.
            if role == "assistant" and msg.get("tool_calls"):
                split -= 1
                continue

            # Safe: user message, text-only assistant, or system message
            break

        if split <= 1:
            # Can't compact without breaking structure
            return messages

        old_messages = messages[:split]
        recent_messages = messages[split:]

        # Build text from old messages
        old_text_parts: list[str] = []
        for msg in old_messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, str) and content:
                old_text_parts.append(f"{role}: {content[:200]}")

        old_text = "\n".join(old_text_parts)

        # Summarize
        if summarize_fn and callable(summarize_fn):
            try:
                summary = summarize_fn(
                    f"Summarize this conversation history in 2-3 concise sentences:\n\n{old_text}"
                )
            except Exception as exc:
                logger.warning("LLM summarization failed: %s", exc)
                summary = old_text[:500] + "..."
        else:
            # Simple truncation summary
            summary = old_text[:500]
            if len(old_text) > 500:
                summary += f"\n\n[...{len(old_messages)} messages summarized]"

        # Build compacted conversation.
        # Use role "user" for the summary — Gemini rejects mid-conversation
        # "system" messages and requires conversations to start with "user".
        summary_msg = {
            "role": "user",
            "content": f"[Conversation summary of {len(old_messages)} earlier messages]:\n{summary}",
        }

        return [summary_msg] + recent_messages

    def get_context_breakdown(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Get a detailed breakdown of context usage.

        Returns a dict with token counts for each section.
        """
        window = self.get_context_window()
        reserve = int(window * _RESPONSE_RESERVE)
        sys_tokens = count_tokens(system_prompt, self._model)
        msg_tokens = count_messages_tokens(messages, self._model)
        used = sys_tokens + msg_tokens
        available = max(0, window - reserve - used)

        return {
            "context_window": window,
            "response_reserve": reserve,
            "system_prompt_tokens": sys_tokens,
            "conversation_tokens": msg_tokens,
            "total_used": used,
            "available": available,
            "utilization_pct": round((used / window) * 100, 1) if window > 0 else 0,
            "message_count": len(messages),
        }

    def needs_compaction(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        threshold: float | None = None,
    ) -> bool:
        """Check if context usage exceeds the compaction threshold.

        Triggers compaction when either:
        1. Overall utilization exceeds the threshold (% of context window), OR
        2. Conversation tokens alone exceed ``max_conversation_tokens`` — this
           prevents runaway growth on large-context models where the percentage
           threshold would never fire (e.g. 80% of 1M = 800K).

        Args:
            system_prompt: The current system prompt.
            messages: Current conversation messages.
            threshold: Override threshold (0-1). Defaults to auto_compact_threshold.

        Returns:
            True if utilization exceeds the threshold.
        """
        threshold = threshold if threshold is not None else self._auto_compact_threshold
        breakdown = self.get_context_breakdown(system_prompt, messages)
        utilization = breakdown.get("utilization_pct", 0) / 100.0

        # Hard cap: compact when conversation tokens exceed absolute limit
        conv_tokens = breakdown.get("conversation_tokens", 0)
        if conv_tokens > self._max_conversation_tokens:
            logger.info(
                "Conversation tokens (%d) exceed hard cap (%d), triggering compaction",
                conv_tokens,
                self._max_conversation_tokens,
            )
            return True

        return utilization > threshold

    # ------------------------------------------------------------------
    # Context Modes (Phase 4e)
    # ------------------------------------------------------------------

    def set_mode(self, mode: str) -> None:
        """Set the context assembly mode.

        Modes:
        - ``full``: Include all sections (default for large-context models).
        - ``lean``: Only identity + tools + conversation (for small models).
        - ``auto``: Auto-select based on model context window.
        - ``manual``: User controls which sections to include.
        - ``adaptive``: Dynamic — include sections based on relevance.

        Args:
            mode: One of "full", "lean", "auto", "manual", "adaptive".
        """
        valid = {"full", "lean", "auto", "manual", "adaptive"}
        if mode not in valid:
            raise ValueError(f"Invalid context mode: {mode}. Must be one of: {valid}")
        self._mode = mode

    @property
    def mode(self) -> str:
        """Current context assembly mode."""
        return self._mode

    def get_effective_mode(self) -> str:
        """Resolve 'auto' mode into an actual mode based on model capabilities."""
        mode = self.mode
        if mode != "auto":
            return mode

        window = self.get_context_window()
        if window >= 200_000:
            return "full"
        elif window >= 32_000:
            return "adaptive"
        else:
            return "lean"


# ---------------------------------------------------------------------------
# Section prioritization for context assembly
# ---------------------------------------------------------------------------

@dataclass
class ContextSection:
    """A section of the system prompt with priority and size metadata.

    Sections are included in priority order until the token budget
    is exhausted.

    Args:
        name: Section identifier.
        content: Text content of the section.
        priority: Higher priority sections are included first (1-10).
        required: If True, always included regardless of budget.
    """

    name: str
    content: str
    priority: int = 5
    required: bool = False
    _token_count: Optional[int] = field(default=None, init=False, repr=False)

    def token_count(self, model: str = "") -> int:
        """Count tokens in this section (cached)."""
        if self._token_count is None:
            self._token_count = count_tokens(self.content, model)
        return self._token_count


# Default section priorities (higher = more important)
SECTION_PRIORITIES: dict[str, int] = {
    "identity": 10,            # Always include
    "user_info": 9,            # Always include
    "tools": 9,                # Always include
    "rules": 8,                # User-defined rules
    "communication_style": 8,  # How to respond
    "slash_commands": 6,       # Available commands
    "skills_catalog": 5,       # Available skills
    "plugins_catalog": 5,      # Available plugins
    "mcp_servers": 5,          # MCP tool catalog
    "planning_mode": 4,        # Planning instructions
    "repo_map": 4,             # Codebase structure
    "artifacts_docs": 3,       # Artifact formatting
    "transcript_docs": 3,      # Conversation log docs
    "web_dev_guidelines": 2,   # Web dev guidelines
    "customization_docs": 2,   # Skill/rule system docs
    "subagent_catalog": 2,     # Subagent types
}


def assemble_sections(
    sections: list[ContextSection],
    token_budget: int,
    model: str = "",
    mode: str = "full",
) -> list[ContextSection]:
    """Select sections to include based on mode and token budget.

    Args:
        sections: All available sections.
        token_budget: Maximum tokens for the system prompt.
        model: Model name for token counting.
        mode: Context mode ("full", "lean", "adaptive").

    Returns:
        List of sections to include, ordered by priority.
    """
    # Sort by priority (required first, then by priority descending)
    sorted_sections = sorted(
        sections,
        key=lambda s: (not s.required, -s.priority),
    )

    if mode == "full":
        return sorted_sections  # include everything

    if mode == "lean":
        # Only required sections + tools
        return [s for s in sorted_sections if s.required or s.priority >= 9]

    # Adaptive: fit as many sections as possible within budget
    included: list[ContextSection] = []
    used_tokens = 0

    for section in sorted_sections:
        tokens = section.token_count(model)

        if section.required:
            included.append(section)
            used_tokens += tokens
            continue

        if used_tokens + tokens <= token_budget:
            included.append(section)
            used_tokens += tokens

    return included
