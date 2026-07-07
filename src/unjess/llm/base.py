"""Abstract base for all LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generator


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    """A single tool/function call requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]
    thought_signature: str | None = None  # Google Gemini 3+ requirement


@dataclass
class Usage:
    """Token usage for a single LLM response."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    thinking_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Return total tokens used.

        Note: ``thinking_tokens`` is a *breakdown* of ``completion_tokens``,
        not additive.  Providers already include reasoning tokens in
        ``completion_tokens``.
        """
        return self.prompt_tokens + self.completion_tokens


@dataclass
class LLMResponse:
    """Complete response from an LLM call."""

    text: str = ""
    thinking_text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    finish_reason: str = ""
    model: str = ""


# ---------------------------------------------------------------------------
# Chunk type for streaming
# ---------------------------------------------------------------------------

@dataclass
class StreamChunk:
    """A single chunk from a streaming response.

    Exactly one of ``text``, ``thinking``, or ``tool_call`` will be set per chunk.
    ``done`` is True on the final chunk.
    """

    text: str = ""
    thinking: str = ""
    tool_call: ToolCall | None = None
    done: bool = False
    usage: Usage | None = None


# ---------------------------------------------------------------------------
# Abstract provider
# ---------------------------------------------------------------------------

class LLMProvider(ABC):
    """Unified interface that every LLM provider must implement."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Short identifier for the provider (e.g. 'openai', 'anthropic')."""

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str = "",
    ) -> LLMResponse:
        """Send messages and return a complete response.

        Args:
            messages: Conversation history in OpenAI message format.
            tools: JSON-schema tool definitions (OpenAI function-calling format).
            model: Override the default model for this call.

        Returns:
            A fully populated ``LLMResponse``.
        """

    @abstractmethod
    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str = "",
    ) -> Generator[StreamChunk, None, None]:
        """Stream a response, yielding ``StreamChunk`` objects.

        The final chunk has ``done=True`` and may include aggregated ``usage``.
        """

    def list_models(self) -> list[str]:
        """List available models from this provider's API.

        Returns:
            Sorted list of model ID strings. Returns empty list if
            the provider doesn't support model listing.
        """
        return []

