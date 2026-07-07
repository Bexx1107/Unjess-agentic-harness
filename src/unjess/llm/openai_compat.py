"""OpenAI-compatible provider — handles OpenAI API and Ollama (same protocol)."""

import json
import logging
import random
from typing import Any, Generator

from openai import OpenAI

from unjess.llm.base import (
    LLMProvider,
    LLMResponse,
    StreamChunk,
    ToolCall,
    Usage,
)

from unjess.llm.retry import call_with_retries

logger = logging.getLogger(__name__)


def _convert_tools_to_openai(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert our tool schema format to OpenAI function-calling format.

    Our tools already use the OpenAI shape, so this is mostly pass-through
    with validation.
    """
    converted: list[dict[str, Any]] = []
    for tool in tools:
        converted.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
            },
        })
    return converted


def _parse_tool_calls(raw_calls: Any) -> list[ToolCall]:
    """Parse OpenAI tool call objects into our ToolCall dataclass."""
    results: list[ToolCall] = []
    if not raw_calls:
        return results

    for tc in raw_calls:
        try:
            args = json.loads(tc.function.arguments) if tc.function.arguments else {}
        except json.JSONDecodeError:
            args = {"_raw": tc.function.arguments}
        results.append(ToolCall(
            id=tc.id or "",
            name=tc.function.name or "",
            arguments=args,
        ))
    return results


class OpenAICompatibleProvider(LLMProvider):
    """Provider for OpenAI API and any OpenAI-compatible endpoint (e.g. Ollama).

    Args:
        api_key: The API key.
        base_url: The base URL for the API (default: OpenAI).
        default_model: The default model to use.
        name: Display name for this provider instance.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        default_model: str = "gpt-4o-mini",
        name: str = "openai",
    ) -> None:
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=60.0)
        self._default_model = default_model
        self._name = name

    @property
    def provider_name(self) -> str:
        """Short identifier for this provider."""
        return self._name

    def list_models(self) -> list[str]:
        """List available models from this provider's API.

        Uses the standard OpenAI /v1/models endpoint, which is also
        supported by Groq, Mistral, and Ollama.

        Filters results per-provider to only show chat/code models
        useful for a coding agent (excludes OCR, TTS, embedding, etc.).

        Returns:
            Sorted list of model ID strings.
        """
        try:
            response = self._client.models.list()
            model_ids = [m.id for m in response]

            # OpenRouter — only free models
            if self._name == "openrouter":
                free_ids = [
                    mid for mid in model_ids
                    if ":free" in mid or mid.startswith("openrouter/")
                ]
                return sorted(free_ids)

            # Google — only gemini chat models
            if self._name == "google":
                _GOOGLE_SKIP = (
                    "robotics", "lyria", "nano-banana", "tts",
                    "deep-research", "antigravity", "gemma",
                    "computer-use",
                )
                model_ids = [
                    mid for mid in model_ids
                    if mid.startswith("gemini-")
                    and not any(skip in mid for skip in _GOOGLE_SKIP)
                ]

            # Groq — only chat/instruct models
            elif self._name == "groq":
                _GROQ_SKIP = (
                    "whisper", "orpheus", "prompt-guard",
                    "safeguard", "allam",
                )
                model_ids = [
                    mid for mid in model_ids
                    if not any(skip in mid.lower() for skip in _GROQ_SKIP)
                ]

            # Mistral — only chat/code models
            elif self._name == "mistral":
                _MISTRAL_SKIP = (
                    "ocr", "embed", "moderation", "voxtral",
                    "vibe-cli", "fim", "tiny",
                )
                model_ids = [
                    mid for mid in model_ids
                    if not any(skip in mid.lower() for skip in _MISTRAL_SKIP)
                ]
                # Deduplicate (Mistral returns aliases as separate entries)
                model_ids = list(dict.fromkeys(model_ids))

            return sorted(model_ids)
        except Exception as exc:
            logger.warning("Failed to list models from %s: %s", self._name, exc)
            return []

    # ----- non-streaming -----

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str = "",
    ) -> LLMResponse:
        """Send a chat completion request and return the full response."""
        model = model or self._default_model
        kwargs: dict[str, Any] = {"model": model, "messages": messages}

        # Ollama: bust KV cache to prevent old context from leaking
        if self._name == "ollama":
            kwargs["seed"] = random.randint(0, 2**31)

        if tools:
            kwargs["tools"] = _convert_tools_to_openai(tools)

        raw = self._call_with_retries(**kwargs)

        # Extract reasoning tokens from completion_tokens_details (o3/o4-mini)
        thinking_tokens = 0
        if raw.usage and hasattr(raw.usage, 'completion_tokens_details') and raw.usage.completion_tokens_details:
            thinking_tokens = getattr(raw.usage.completion_tokens_details, 'reasoning_tokens', 0) or 0

        if not raw.choices:
            usage = Usage(
                prompt_tokens=raw.usage.prompt_tokens if raw.usage else 0,
                completion_tokens=raw.usage.completion_tokens if raw.usage else 0,
                thinking_tokens=thinking_tokens,
            )
            return LLMResponse(
                text="",
                tool_calls=[],
                usage=usage,
                finish_reason="empty_response",
                model=raw.model or model,
            )

        choice = raw.choices[0]
        usage = Usage(
            prompt_tokens=raw.usage.prompt_tokens if raw.usage else 0,
            completion_tokens=raw.usage.completion_tokens if raw.usage else 0,
            thinking_tokens=thinking_tokens,
        )

        return LLMResponse(
            text=choice.message.content or "",
            tool_calls=_parse_tool_calls(choice.message.tool_calls),
            usage=usage,
            finish_reason=choice.finish_reason or "",
            model=raw.model or model,
        )

    # ----- streaming -----

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str = "",
    ) -> Generator[StreamChunk, None, None]:
        """Stream a chat completion, yielding StreamChunk objects."""
        model = model or self._default_model
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            # Ollama: bust KV cache to prevent context leaking between sessions
            **(({"seed": random.randint(0, 2**31)}) if self._name == "ollama" else {}),
        }

        if tools:
            kwargs["tools"] = _convert_tools_to_openai(tools)

        # Try with stream_options first; fall back without if the provider
        # doesn't support it (Ollama, Groq, Mistral).
        try:
            stream = self._call_with_retries(**kwargs)
        except Exception as exc:
            if "stream_options" in str(exc).lower():
                kwargs.pop("stream_options", None)
                logger.info("Provider does not support stream_options, retrying without it.")
                stream = self._call_with_retries(**kwargs)
            else:
                raise

        # Accumulate tool calls across chunks (they arrive in pieces)
        pending_tool_calls: dict[int, dict[str, str]] = {}
        usage: Usage | None = None

        for chunk in stream:
            # Usage comes on the final chunk (with stream_options.include_usage)
            if chunk.usage:
                thinking_tokens = 0
                if hasattr(chunk.usage, 'completion_tokens_details') and chunk.usage.completion_tokens_details:
                    thinking_tokens = getattr(chunk.usage.completion_tokens_details, 'reasoning_tokens', 0) or 0
                usage = Usage(
                    prompt_tokens=chunk.usage.prompt_tokens or 0,
                    completion_tokens=chunk.usage.completion_tokens or 0,
                    thinking_tokens=thinking_tokens,
                )

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            # Text content
            if delta.content:
                yield StreamChunk(text=delta.content)

            # Tool calls (arrive incrementally)
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in pending_tool_calls:
                        pending_tool_calls[idx] = {
                            "id": tc_delta.id or "",
                            "name": "",
                            "arguments": "",
                        }
                    if tc_delta.id:
                        pending_tool_calls[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            pending_tool_calls[idx]["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            pending_tool_calls[idx]["arguments"] += tc_delta.function.arguments

            # Check if finished
            if chunk.choices[0].finish_reason:
                break

        # Emit accumulated tool calls
        for _idx in sorted(pending_tool_calls):
            tc_data = pending_tool_calls[_idx]
            try:
                args = json.loads(tc_data["arguments"]) if tc_data["arguments"] else {}
            except json.JSONDecodeError:
                args = {"_raw": tc_data["arguments"]}

            yield StreamChunk(tool_call=ToolCall(
                id=tc_data["id"],
                name=tc_data["name"],
                arguments=args,
            ))

        # Final done chunk
        yield StreamChunk(done=True, usage=usage)

    # ----- retry logic -----

    def _call_with_retries(self, **kwargs: Any) -> Any:
        """Call the OpenAI client with exponential backoff on retryable errors.

        Rate-limit errors are NOT retried internally — they bubble up to
        the router which handles API key rotation.
        """
        return call_with_retries(
            self._client.chat.completions.create,
            skip_rate_limit_retry=True,
            **kwargs,
        )
