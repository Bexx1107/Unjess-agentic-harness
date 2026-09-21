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
    "gemini-3.1-flash": 2_000_000,
    "gemini-3.1-pro": 2_000_000,
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
    "deepseek-v4.1-flash": 1_000_000,
    "deepseek-v4.1": 1_000_000,
    "deepseek-v4-flash": 1_000_000,
    "deepseek-v4-pro": 128_000,
    "deepseek-v4": 128_000,
    "deepseek-v3": 128_000,
    "deepseek-r1": 128_000,
    "deepseek-chat": 128_000,
    "deepseek": 64_000,
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
    if "gpt" in model.lower() or "o1" in model.lower() or "o3" in model.lower() or "o4" in model.lower():
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except Exception:
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
        max_conversation_tokens: Optional[int] = None,
        context_window_override: int = 0,
        enable_compaction: bool = True,
    ) -> None:
        self._model = model
        self._mode: str = "auto"
        self._auto_compact_threshold = auto_compact_threshold
        self._max_conversation_tokens = max_conversation_tokens
        self._context_window_override = context_window_override
        self.enable_compaction = enable_compaction

    @property
    def model(self) -> str:
        """Current model."""
        return self._model

    @model.setter
    def model(self, value: str) -> None:
        self._model = value

    def get_context_window(self) -> int:
        """Return the context window size for the current model."""
        if self._context_window_override > 0:
            return self._context_window_override

        # Strip common provider prefixes e.g. "ollama:deepseek-v4.1-flash:cloud" -> "deepseek-v4.1-flash:cloud"
        clean = self._model
        for prefix in ("ollama:", "ollama-api:", "ollama_api:", "openrouter:", "groq:", "mistral:", "openai:", "google:", "anthropic:"):
            if clean.lower().startswith(prefix):
                clean = clean[len(prefix):]
                break

        # Exact match
        window = _CONTEXT_WINDOWS.get(clean) or _CONTEXT_WINDOWS.get(self._model)
        if window:
            # Remote Ollama cloud models (:cloud) default to 64,000 tokens when no override is set
            if ":cloud" in self._model.lower() or ":cloud" in clean.lower():
                return min(window, 64_000)
            return window

        # Prefix match
        for key, val in _CONTEXT_WINDOWS.items():
            if clean.startswith(key) or self._model.startswith(key):
                if ":cloud" in self._model.lower() or ":cloud" in clean.lower():
                    return min(val, 64_000)
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
        and the most recent user instruction (to prevent forgetting the user's request).

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

        # For small local Ollama models (<=128k), enforce a safety ceiling of 48,000 tokens
        # to ensure conversations never stall low-memory servers. Never enforce this if the
        # user explicitly set context_window_override or if the model has a large context window (>128k).
        if self._context_window_override <= 0 and window <= 128_000:
            if ":cloud" in self._model.lower() or ":" in self._model:
                budget = min(budget, 48_000)

        if budget <= 0:
            logger.warning("System prompt alone exceeds context budget!")
            return messages[-2:]  # keep only the most recent exchange

        # Check if we even need to truncate
        total = count_messages_tokens(messages, self._model)
        if total <= budget:
            return messages  # fits fine

        if len(messages) <= 2:
            return messages

        # Find the latest user message index
        latest_user_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                latest_user_idx = i
                break

        # Structure:
        # [0]: first message (initial goal)
        # [1 : latest_user_idx]: middle history (droppable)
        # [latest_user_idx : ]: protected recent turn
        if latest_user_idx > 1:
            first = messages[0]
            middle = list(messages[1:latest_user_idx])
            protected = list(messages[latest_user_idx:])

            # Drop from middle first until budget fits
            while middle and count_messages_tokens([first] + middle + protected, self._model) > budget:
                head = middle[0]
                removed_unit: list[dict[str, Any]] = []

                if head.get("role") == "assistant" and head.get("tool_calls"):
                    call_ids = {tc.get("id") for tc in head["tool_calls"] if tc.get("id")}
                    removed_unit.append(middle.pop(0))
                    while middle and middle[0].get("role") == "tool" and middle[0].get("tool_call_id") in call_ids:
                        removed_unit.append(middle.pop(0))
                else:
                    removed_unit.append(middle.pop(0))

            result = [first] + middle + protected
            if len(result) < len(messages):
                truncated_count = len(messages) - len(result)
                notice = {
                    "role": "user",
                    "content": f"[{truncated_count} earlier middle messages were truncated to fit the context window]",
                }
                result.insert(1, notice)
            messages = result

        # Handle tool loop case: single user message at start (latest_user_idx <= 1),
        # but many assistant+tool turns exceeding context budget
        elif len(messages) > 4 and count_messages_tokens(messages, self._model) > budget:
            first = messages[0]
            split = max(1, len(messages) - 4)
            while split > 1 and messages[split].get("role") == "tool":
                split -= 1
            if split > 1:
                middle = list(messages[1:split])
                protected = list(messages[split:])
                while middle and count_messages_tokens([first] + middle + protected, self._model) > budget:
                    head = middle[0]
                    if head.get("role") == "assistant" and head.get("tool_calls"):
                        call_ids = {tc.get("id") for tc in head["tool_calls"] if tc.get("id")}
                        middle.pop(0)
                        while middle and middle[0].get("role") == "tool" and middle[0].get("tool_call_id") in call_ids:
                            middle.pop(0)
                    else:
                        middle.pop(0)

                result = [first] + middle + protected
                if len(result) < len(messages):
                    truncated_count = len(messages) - len(result)
                    notice = {
                        "role": "user",
                        "content": f"[{truncated_count} earlier tool execution turns were truncated to fit the context window]",
                    }
                    result.insert(1, notice)
                messages = result

        # If still over budget after dropping middle messages (because tool outputs in protected are huge),
        # truncate the content of tool results inside protected messages instead of dropping user prompts.
        if count_messages_tokens(messages, self._model) > budget:
            trimmed: list[dict[str, Any]] = []
            for msg in messages:
                if msg.get("role") == "tool" and isinstance(msg.get("content"), str) and len(msg["content"]) > 1000:
                    m = dict(msg)
                    m["content"] = m["content"][:1000] + "\n[...tool result truncated to fit context budget...]"
                    trimmed.append(m)
                else:
                    trimmed.append(msg)
            messages = trimmed

        # Emergency secondary trim if still over budget: cut tool content down to 300 chars
        if count_messages_tokens(messages, self._model) > budget:
            trimmed = []
            for msg in messages:
                if msg.get("role") == "tool" and isinstance(msg.get("content"), str) and len(msg["content"]) > 300:
                    m = dict(msg)
                    m["content"] = m["content"][:300] + "\n[...tool result truncated...]"
                    trimmed.append(m)
                else:
                    trimmed.append(msg)
            messages = trimmed

        return messages

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
        is chosen to avoid breaking tool call/response groups and to keep
        the latest user prompt intact.

        Args:
            messages: Current conversation messages.
            summarize_fn: Optional callable(text) -> str for LLM summarization.
            keep_recent: Minimum number of recent messages to keep intact.

        Returns:
            Compacted message list.
        """
        if not self.enable_compaction or len(messages) <= keep_recent:
            return messages  # too few to compact or disabled

        # Ensure split point is before the latest user message if possible
        latest_user_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                latest_user_idx = i
                break

        split = len(messages) - keep_recent
        # If there are multiple user messages in history, don't split past the latest user message
        if latest_user_idx > 1 and split > latest_user_idx:
            split = latest_user_idx

        # If there are no user messages at all and message list is small (<= 4), keep original
        if latest_user_idx == -1 and len(messages) <= 4:
            return messages

        # Walk backwards from the initial split to find a safe boundary.
        # If split lands on a 'tool' result, walk back to the assistant that made the tool call.
        while split > 1 and messages[split].get("role") == "tool":
            split -= 1

        if split <= 1:
            return messages

        old_messages = messages[:split]
        recent_messages = messages[split:]

        if len(old_messages) <= 1:
            return messages

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
                if not summary or not summary.strip():
                    summary = old_text[:500] + "..."
            except Exception as exc:
                logger.warning("LLM summarization failed: %s", exc)
                summary = old_text[:500] + "..."
        else:
            summary = old_text[:500]
            if len(old_text) > 500:
                summary += f"\n\n[...{len(old_messages)} messages summarized]"

        summary_msg = {
            "role": "user",
            "content": f"[Conversation summary of {len(old_messages)} earlier messages]:\n{summary}",
        }

        # Truncate overly long tool outputs in recent_messages to guarantee token reduction
        compacted_recent: list[dict[str, Any]] = []
        for msg in recent_messages:
            if msg.get("role") == "tool" and isinstance(msg.get("content"), str) and len(msg["content"]) > 1000:
                m = dict(msg)
                m["content"] = m["content"][:1000] + "\n[...tool output truncated during compaction...]"
                compacted_recent.append(m)
            else:
                compacted_recent.append(msg)

        return [summary_msg] + compacted_recent

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
           threshold would never fire.

        Args:
            system_prompt: The current system prompt.
            messages: Current conversation messages.
            threshold: Override threshold (0-1). Defaults to auto_compact_threshold.

        Returns:
            True if utilization exceeds the threshold.
        """
        if not self.enable_compaction:
            return False

        threshold = threshold if threshold is not None else self._auto_compact_threshold
        breakdown = self.get_context_breakdown(system_prompt, messages)
        utilization = breakdown.get("utilization_pct", 0) / 100.0

        # Dynamic hard cap:
        # If max_conversation_tokens was explicitly specified (e.g. in tests), use it directly.
        # Otherwise, scale with the context window: for large windows (>128k) or overrides,
        # allow conversation up to auto_compact_threshold (e.g. 80% = 800k tokens for 1M context),
        # avoiding premature compaction at small 80k token limits.
        window = self.get_context_window()
        if self._max_conversation_tokens is not None:
            effective_max = self._max_conversation_tokens
        elif self._context_window_override > 0 or window > 128_000:
            effective_max = int(window * self._auto_compact_threshold)
        else:
            effective_max = min(80_000, int(window * 0.7))

        conv_tokens = breakdown.get("conversation_tokens", 0)
        if conv_tokens > effective_max:
            logger.info(
                "Conversation tokens (%d) exceed hard cap (%d), triggering compaction",
                conv_tokens,
                effective_max,
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
