"""OpenAI-compatible provider — handles OpenAI API and Ollama (same protocol)."""

import base64
import json
import logging
import random
from typing import Any, Generator

import httpx
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


def _prepare_messages_for_openai(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Format messages and image payloads for OpenAI-compatible APIs.

    Converts internal image dicts to OpenAI's standard image_url format:
    {"type": "image_url", "image_url": {"url": "data:<mime>;base64,<b64>"}}
    ensuring that raw bytes never reach json serialization.
    """
    prepared: list[dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            new_content: list[dict[str, Any]] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image":
                    raw_data = part.get("data", "")
                    mime = part.get("mime_type", "image/png")
                    if isinstance(raw_data, bytes):
                        b64_str = base64.b64encode(raw_data).decode("ascii")
                    else:
                        b64_str = str(raw_data)
                    new_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{b64_str}",
                        },
                    })
                else:
                    new_content.append(part)
            prepared.append({**msg, "content": new_content})
        else:
            prepared.append(msg)
    return prepared


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
        timeout: float = 300.0,
    ) -> None:
        client_timeout = (
            httpx.Timeout(float(timeout), connect=60.0)
            if isinstance(timeout, (int, float))
            else timeout
        )
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=client_timeout)
        self._default_model = default_model
        self._name = name
        self._timeout = float(timeout) if isinstance(timeout, (int, float)) else 300.0
        self._active_stream: Any = None
        self._aborted: bool = False

    def abort(self) -> None:
        """Abort the active stream / HTTP connection immediately."""
        self._aborted = True
        stream = self._active_stream
        if stream is not None:
            try:
                stream.close()
            except Exception:
                pass
            self._active_stream = None

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

            # llama.cpp — return loaded models or fallback to 'llamacpp'
            if self._name == "llamacpp":
                return sorted(model_ids) if model_ids else ["llamacpp"]

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

    def _create_completion(self, **kwargs: Any) -> Any:
        """Call chat.completions.create with fallback handling for stream_options and reasoning_effort."""
        try:
            return self._call_with_retries(**kwargs)
        except Exception as exc:
            err_msg = str(exc).lower()
            # Case 1: Provider does not support stream_options
            if "stream_options" in err_msg and "stream_options" in kwargs:
                kwargs_copy = dict(kwargs)
                kwargs_copy.pop("stream_options", None)
                logger.info("Provider does not support stream_options, retrying without it.")
                return self._create_completion(**kwargs_copy)
            # Case 2: Provider requires reasoning_effort="none" when function tools are passed
            if "reasoning_effort" in err_msg and kwargs.get("reasoning_effort") != "none":
                kwargs_copy = dict(kwargs)
                kwargs_copy["reasoning_effort"] = "none"
                logger.info("Function tools require reasoning_effort='none', retrying with reasoning_effort='none'.")
                return self._create_completion(**kwargs_copy)
            # Case 3: Provider rejects reasoning_effort parameter as unrecognized
            if "reasoning_effort" in err_msg and "reasoning_effort" in kwargs:
                kwargs_copy = dict(kwargs)
                kwargs_copy.pop("reasoning_effort", None)
                logger.info("Provider rejected reasoning_effort parameter, retrying without it.")
                return self._create_completion(**kwargs_copy)
            raise

    # ----- non-streaming -----

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str = "",
    ) -> LLMResponse:
        """Send a chat completion request and return the full response."""
        model = model or self._default_model
        kwargs: dict[str, Any] = {"model": model, "messages": _prepare_messages_for_openai(messages)}

        # Ollama: bust KV cache to prevent old context from leaking
        if self._name in ("ollama", "ollama-api"):
            kwargs["seed"] = random.randint(0, 2**31)

        # Local providers: enforce low temperature for deterministic tool calling & no rambling
        if self._name in ("ollama", "ollama-api", "llamacpp", "lmstudio"):
            kwargs["temperature"] = 0.1

        if tools:
            kwargs["tools"] = _convert_tools_to_openai(tools)
            if self._name == "openai" and any(k in model.lower() for k in ("gpt-5", "o1", "o3", "o4", "luna", "reasoning")):
                kwargs["reasoning_effort"] = "none"

        raw = self._create_completion(**kwargs)

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
            "messages": _prepare_messages_for_openai(messages),
            "stream": True,
            "stream_options": {"include_usage": True},
            # Local providers: low temperature for deterministic tool calling & no rambling
            **(({"temperature": 0.1}) if self._name in ("ollama", "ollama-api", "llamacpp", "lmstudio") else {}),
        }

        if tools:
            kwargs["tools"] = _convert_tools_to_openai(tools)
            if self._name == "openai" and any(k in model.lower() for k in ("gpt-5", "o1", "o3", "o4", "luna", "reasoning")):
                kwargs["reasoning_effort"] = "none"

        self._aborted = False
        stream = self._create_completion(**kwargs)
        self._active_stream = stream

        # Accumulate tool calls across chunks (they arrive in pieces)
        pending_tool_calls: dict[int, dict[str, str]] = {}
        usage: Usage | None = None
        full_text = ""

        try:
            for chunk in stream:
                if self._aborted:
                    break

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

                # Reasoning / Thinking content (e.g., DeepSeek R1 via Ollama)
                reasoning = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
                if reasoning:
                    yield StreamChunk(thinking=reasoning)

                # Text content
                if delta.content:
                    full_text += delta.content
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
        except Exception as exc:
            if self._aborted or "closed" in str(exc).lower() or type(exc).__name__ in ("StreamClosed", "ResponseClosed"):
                logger.debug("Stream successfully closed on abort.")
                return
            raise
        finally:
            self._active_stream = None
            try:
                stream.close()
            except Exception:
                pass

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

        # Estimate usage if missing
        if not usage or (usage.prompt_tokens == 0 and usage.completion_tokens == 0):
            try:
                from unjess.context_manager import count_messages_tokens, count_tokens
                prompt_tokens = count_messages_tokens(messages, model=model)
                completion_tokens = count_tokens(full_text, model=model)
                usage = Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
            except Exception as exc:
                logger.debug("Failed to estimate streamed token usage: %s", exc)

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
