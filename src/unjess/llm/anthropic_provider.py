"""Anthropic / Claude provider — handles the different message format and content blocks."""

import json
import logging
from typing import Any, Generator

import anthropic

from unjess.llm.base import (
    LLMProvider,
    LLMResponse,
    StreamChunk,
    ToolCall,
    Usage,
)

from unjess.llm.retry import call_with_retries

logger = logging.getLogger(__name__)


def _convert_tools_to_anthropic(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert our tool schema format to Anthropic's tool format."""
    converted: list[dict[str, Any]] = []
    for tool in tools:
        params = tool.get("parameters", {"type": "object", "properties": {}})
        converted.append({
            "name": tool["name"],
            "description": tool.get("description", ""),
            "input_schema": params,
        })
    return converted


def _split_system_message(
    messages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Anthropic requires the system message to be separate.

    Returns (system_text, remaining_messages).
    """
    system_parts: list[str] = []
    remaining: list[dict[str, Any]] = []

    for msg in messages:
        if msg.get("role") == "system":
            system_parts.append(msg.get("content", ""))
        else:
            remaining.append(msg)

    return "\n\n".join(system_parts), remaining


def _convert_messages_for_anthropic(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert OpenAI-format messages to Anthropic format.

    Handles tool_calls (assistant) and tool results (tool role).
    """
    converted: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "user")

        if role == "assistant":
            content_blocks: list[dict[str, Any]] = []
            # Text content
            if msg.get("content"):
                content_blocks.append({"type": "text", "text": msg["content"]})
            # Tool use blocks
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    func = tc.get("function", {})
                    try:
                        args = json.loads(func.get("arguments", "{}"))
                    except json.JSONDecodeError:
                        args = {}
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": func.get("name", ""),
                        "input": args,
                    })
            converted.append({"role": "assistant", "content": content_blocks or msg.get("content", "")})

        elif role == "tool":
            # Tool result → Anthropic uses "user" role with tool_result content block.
            # Merge consecutive tool results into the same user message to
            # satisfy Anthropic's alternating user/assistant requirement.
            tool_result_block = {
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id", ""),
                "content": msg.get("content", ""),
            }
            if converted and converted[-1]["role"] == "user" and isinstance(converted[-1]["content"], list):
                converted[-1]["content"].append(tool_result_block)
            else:
                converted.append({
                    "role": "user",
                    "content": [tool_result_block],
                })

        else:
            # user messages pass through
            converted.append({"role": role, "content": msg.get("content", "")})

    return converted


class AnthropicProvider(LLMProvider):
    """Provider for Anthropic Claude models.

    Args:
        api_key: Anthropic API key.
        default_model: Default model name.
    """

    def __init__(
        self,
        api_key: str,
        default_model: str = "claude-sonnet-4-20250514",
    ) -> None:
        self._client = anthropic.Anthropic(api_key=api_key, timeout=60.0)
        self._default_model = default_model

    @property
    def provider_name(self) -> str:
        """Short identifier for this provider."""
        return "anthropic"

    def list_models(self) -> list[str]:
        """List available Claude models.

        Anthropic doesn't expose a list-models API, so this returns
        a curated list of current models.

        Returns:
            List of model ID strings.
        """
        try:
            # The anthropic sdk has a models.list() endpoint in recent versions
            response = self._client.models.list()
            # The response is a SyncPage[Model] where each item has an `id`
            return sorted([m.id for m in response.data])
        except Exception as e:
            logger.warning(f"Failed to fetch Anthropic models: {e}")
            return [
                "claude-opus-4-20250514",
                "claude-sonnet-4-20250514",
                "claude-3.5-sonnet-20241022",
                "claude-3.5-haiku-20241022",
            ]

    # ----- non-streaming -----

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str = "",
    ) -> LLMResponse:
        """Send a chat request and return the full response."""
        model = model or self._default_model
        system_text, user_messages = _split_system_message(messages)
        ant_messages = _convert_messages_for_anthropic(user_messages)

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": ant_messages,
            "max_tokens": 8192,
        }
        if system_text:
            # Use structured content blocks with cache_control for prompt caching
            kwargs["system"] = [
                {
                    "type": "text",
                    "text": system_text,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        if tools:
            ant_tools = _convert_tools_to_anthropic(tools)
            # Mark the last tool with cache_control so tool definitions are cached
            if ant_tools:
                ant_tools[-1]["cache_control"] = {"type": "ephemeral"}
            kwargs["tools"] = ant_tools

        raw = self._call_with_retries(**kwargs)

        # Parse response content blocks
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in raw.content:
            if block.type == "thinking":
                thinking_parts.append(block.thinking)
            elif block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input if isinstance(block.input, dict) else {},
                ))

        # Extract cache token counts
        cache_read = 0
        cache_creation = 0
        if raw.usage:
            cache_read = getattr(raw.usage, 'cache_read_input_tokens', 0) or 0
            cache_creation = getattr(raw.usage, 'cache_creation_input_tokens', 0) or 0

        usage = Usage(
            prompt_tokens=raw.usage.input_tokens if raw.usage else 0,
            completion_tokens=raw.usage.output_tokens if raw.usage else 0,
            cache_read_tokens=cache_read,
            cache_creation_tokens=cache_creation,
        )

        return LLMResponse(
            text="\n".join(text_parts),
            thinking_text="\n".join(thinking_parts),
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=raw.stop_reason or "",
            model=raw.model or model,
        )

    # ----- streaming -----

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str = "",
    ) -> Generator[StreamChunk, None, None]:
        """Stream a response, yielding StreamChunk objects."""
        model = model or self._default_model
        system_text, user_messages = _split_system_message(messages)
        ant_messages = _convert_messages_for_anthropic(user_messages)

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": ant_messages,
            "max_tokens": 8192,
            "stream": True,
        }
        if system_text:
            # Use structured content blocks with cache_control for prompt caching
            kwargs["system"] = [
                {
                    "type": "text",
                    "text": system_text,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        if tools:
            ant_tools = _convert_tools_to_anthropic(tools)
            if ant_tools:
                ant_tools[-1]["cache_control"] = {"type": "ephemeral"}
            kwargs["tools"] = ant_tools

        stream = self._call_with_retries(**kwargs)

        # Track tool use blocks and thinking blocks as they stream in
        current_tool: dict[str, Any] | None = None
        current_block_type: str = ""
        input_tokens = 0
        output_tokens = 0
        cache_read = 0
        cache_creation = 0

        with stream as event_stream:
            for event in event_stream:
                event_type = event.type

                if event_type == "message_start":
                    if hasattr(event, "message") and hasattr(event.message, "usage"):
                        usage_data = event.message.usage
                        input_tokens = getattr(usage_data, "input_tokens", 0)
                        cache_read = getattr(usage_data, 'cache_read_input_tokens', 0) or 0
                        cache_creation = getattr(usage_data, 'cache_creation_input_tokens', 0) or 0

                elif event_type == "content_block_start":
                    block = event.content_block
                    if block.type == "thinking":
                        current_block_type = "thinking"
                    elif block.type == "tool_use":
                        current_block_type = "tool_use"
                        current_tool = {
                            "id": block.id,
                            "name": block.name,
                            "input_json": "",
                        }
                    else:
                        current_block_type = block.type

                elif event_type == "content_block_delta":
                    delta = event.delta
                    if current_block_type == "thinking":
                        delta_text = getattr(delta, 'thinking', '')
                        if delta_text:
                            yield StreamChunk(thinking=delta_text)
                    elif current_block_type == "text" and hasattr(delta, "text"):
                        yield StreamChunk(text=delta.text)
                    elif hasattr(delta, "partial_json") and current_tool is not None:
                        current_tool["input_json"] += delta.partial_json

                elif event_type == "content_block_stop":
                    if current_block_type == "tool_use" and current_tool is not None:
                        try:
                            args = json.loads(current_tool["input_json"]) if current_tool["input_json"] else {}
                        except json.JSONDecodeError:
                            args = {"_raw": current_tool["input_json"]}
                        yield StreamChunk(tool_call=ToolCall(
                            id=current_tool["id"],
                            name=current_tool["name"],
                            arguments=args,
                        ))
                        current_tool = None
                    current_block_type = ""

                elif event_type == "message_delta":
                    if hasattr(event, "usage"):
                        output_tokens = getattr(event.usage, "output_tokens", 0)

                elif event_type == "message_stop":
                    pass

        yield StreamChunk(
            done=True,
            usage=Usage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                cache_read_tokens=cache_read,
                cache_creation_tokens=cache_creation,
            ),
        )

    # ----- retry logic -----

    def _call_with_retries(self, **kwargs: Any) -> Any:
        """Call Anthropic API with exponential backoff on retryable errors.

        Rate-limit errors are NOT retried internally — they bubble up to
        the router which handles API key rotation.
        """
        is_stream = kwargs.get("stream", False)

        if is_stream:
            stream_kwargs = {k: v for k, v in kwargs.items() if k != "stream"}
            return call_with_retries(
                self._client.messages.stream,
                skip_rate_limit_retry=True,
                **stream_kwargs,
            )
        return call_with_retries(
            self._client.messages.create,
            skip_rate_limit_retry=True,
            **kwargs,
        )
