"""Google Gemini provider — uses the google-genai SDK."""

import base64
import json
import logging
from typing import Any, Generator

from google import genai
from google.genai import types as genai_types

from unjess.llm.base import (
    LLMProvider,
    LLMResponse,
    StreamChunk,
    ToolCall,
    Usage,
)
from unjess.llm.retry import call_with_retries

logger = logging.getLogger(__name__)

# Fields that Google's genai SDK rejects in JSON schemas
_SCHEMA_STRIP_KEYS = {"$schema", "additionalProperties", "$ref", "definitions", "$defs"}


def _sanitize_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Recursively strip non-standard JSON Schema fields that Google rejects.

    MCP tools (especially those using zod's z.any()) may include fields like
    '$schema' that cause pydantic validation errors in the genai SDK.
    """
    cleaned: dict[str, Any] = {}
    for key, value in schema.items():
        if key in _SCHEMA_STRIP_KEYS:
            continue
        if isinstance(value, dict):
            cleaned[key] = _sanitize_schema(value)
        elif isinstance(value, list):
            cleaned[key] = [
                _sanitize_schema(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            cleaned[key] = value
    return cleaned


def _convert_tools_to_google(
    tools: list[dict[str, Any]],
) -> list[genai_types.Tool]:
    """Convert our tool schemas to Google function declarations."""
    declarations: list[genai_types.FunctionDeclaration] = []

    for tool in tools:
        params_schema = tool.get("parameters", {"type": "object", "properties": {}})
        # Strip fields that Google's genai SDK doesn't accept
        params_schema = _sanitize_schema(params_schema)
        declarations.append(genai_types.FunctionDeclaration(
            name=tool["name"],
            description=tool.get("description", ""),
            parameters=params_schema,
        ))

    return [genai_types.Tool(function_declarations=declarations)]


def _extract_system_text(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Separate system messages from the rest."""
    system_parts: list[str] = []
    remaining: list[dict[str, Any]] = []

    for msg in messages:
        if msg.get("role") == "system":
            system_parts.append(msg.get("content", ""))
        else:
            remaining.append(msg)

    return "\n\n".join(system_parts), remaining


def _convert_messages_for_google(
    messages: list[dict[str, Any]],
) -> list[genai_types.Content]:
    """Convert OpenAI-format messages to Google Content objects."""
    contents: list[genai_types.Content] = []

    for msg in messages:
        role = msg.get("role", "user")
        # Google uses "user" and "model" roles
        google_role = "model" if role == "assistant" else "user"

        parts: list[genai_types.Part] = []

        # Handle tool results (role == "tool")
        if role == "tool":
            parts.append(genai_types.Part.from_function_response(
                name=msg.get("name", "unknown"),
                response={"result": msg.get("content", "")},
            ))
            google_role = "user"

        # Handle assistant messages with tool calls
        elif role == "assistant" and msg.get("tool_calls"):
            if msg.get("content"):
                parts.append(genai_types.Part.from_text(text=msg["content"]))
            for tc in msg["tool_calls"]:
                func = tc.get("function", {})
                try:
                    args = json.loads(func.get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {}
                fc_part = genai_types.Part.from_function_call(
                    name=func.get("name", ""),
                    args=args,
                )
                # Preserve thought_signature for Gemini 3+ models
                ts = tc.get("thought_signature")
                if ts:
                    # Decode from base64 string back to bytes
                    fc_part.thought_signature = base64.b64decode(ts) if isinstance(ts, str) else ts
                parts.append(fc_part)

        # Text and/or multi-part (image) messages
        else:
            content = msg.get("content", "")
            if isinstance(content, list):
                # Multi-part content: text + images
                for part_item in content:
                    ptype = part_item.get("type", "text")
                    if ptype == "text":
                        text_val = part_item.get("text", "")
                        if text_val:
                            parts.append(genai_types.Part.from_text(text=text_val))
                    elif ptype == "image":
                        img_data = part_item.get("data", b"")
                        if isinstance(img_data, str):
                            img_data = base64.b64decode(img_data)
                        mime = part_item.get("mime_type", "image/png")
                        parts.append(genai_types.Part.from_bytes(
                            data=img_data,
                            mime_type=mime,
                        ))
            elif content:
                parts.append(genai_types.Part.from_text(text=content))

        if parts:
            contents.append(genai_types.Content(role=google_role, parts=parts))

    return contents


class GoogleProvider(LLMProvider):
    """Provider for Google Gemini models via the google-genai SDK.

    Args:
        api_key: Google API key.
        default_model: Default model name.
    """

    def __init__(
        self,
        api_key: str,
        default_model: str = "gemini-3.1-flash",
    ) -> None:
        self._client = genai.Client(api_key=api_key)
        self._default_model = default_model

    @property
    def provider_name(self) -> str:
        """Short identifier for this provider."""
        return "google"

    def list_models(self) -> list[str]:
        """List available Gemini models.

        Returns:
            Sorted list of model names (e.g. 'gemini-3.1-flash').
        """
        try:
            models = self._client.models.list()
            model_ids: list[str] = []
            for m in models:
                # Only include models that support generateContent
                actions = getattr(m, "supported_actions", None) or []
                if not actions or "generateContent" in actions:
                    name = getattr(m, "name", "")
                    # Strip "models/" prefix
                    if name.startswith("models/"):
                        name = name[7:]
                    if name:
                        model_ids.append(name)
            return sorted(model_ids)
        except Exception as exc:
            logger.warning("Failed to list Google models: %s", exc)
            return []

    # ----- non-streaming -----

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str = "",
    ) -> LLMResponse:
        """Send a chat request and return the full response."""
        model = model or self._default_model
        system_text, user_messages = _extract_system_text(messages)
        contents = _convert_messages_for_google(user_messages)

        config = genai_types.GenerateContentConfig(
            system_instruction=system_text if system_text else None,
            tools=_convert_tools_to_google(tools) if tools else None,
        )

        raw = self._call_with_retries(model, contents, config)

        # Parse response
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        if raw.candidates:
            candidate = raw.candidates[0]
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) if content else None
            if parts:
                for i, part in enumerate(parts):
                    # Gemini 3+ models: part.thought is True when the
                    # text is internal reasoning (not user-facing).
                    if getattr(part, 'thought', False) and part.text:
                        thinking_parts.append(part.text)
                    elif part.text:
                        text_parts.append(part.text)
                    elif part.function_call:
                        fc = part.function_call
                        # Capture thought_signature as base64 string for JSON serialization
                        raw_ts = getattr(part, 'thought_signature', None)
                        ts_str = base64.b64encode(raw_ts).decode('ascii') if isinstance(raw_ts, bytes) else raw_ts
                        tool_calls.append(ToolCall(
                            id=f"{fc.name}_{i}",  # Unique ID per call
                            name=fc.name,
                            arguments=dict(fc.args) if fc.args else {},
                            thought_signature=ts_str,
                        ))

        thinking_tokens = 0
        cache_read = 0
        if raw.usage_metadata:
            thinking_tokens = getattr(raw.usage_metadata, 'thoughts_token_count', 0) or 0
            cache_read = getattr(raw.usage_metadata, 'cached_content_token_count', 0) or 0

        usage = Usage(
            prompt_tokens=raw.usage_metadata.prompt_token_count if raw.usage_metadata else 0,
            completion_tokens=raw.usage_metadata.candidates_token_count if raw.usage_metadata else 0,
            thinking_tokens=thinking_tokens,
            cache_read_tokens=cache_read,
        )

        return LLMResponse(
            text="\n".join(text_parts),
            thinking_text="\n".join(thinking_parts),
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=str(raw.candidates[0].finish_reason) if raw.candidates else "",
            model=model,
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
        system_text, user_messages = _extract_system_text(messages)
        contents = _convert_messages_for_google(user_messages)

        config = genai_types.GenerateContentConfig(
            system_instruction=system_text if system_text else None,
            tools=_convert_tools_to_google(tools) if tools else None,
        )

        stream = self._call_with_retries(model, contents, config, stream=True)

        total_usage = Usage()

        for chunk in stream:
            if chunk.usage_metadata:
                thinking_tokens = getattr(chunk.usage_metadata, 'thoughts_token_count', 0) or 0
                cache_read = getattr(chunk.usage_metadata, 'cached_content_token_count', 0) or 0
                total_usage = Usage(
                    prompt_tokens=chunk.usage_metadata.prompt_token_count or 0,
                    completion_tokens=chunk.usage_metadata.candidates_token_count or 0,
                    thinking_tokens=thinking_tokens,
                    cache_read_tokens=cache_read,
                )

            if not chunk.candidates:
                continue

            candidate = chunk.candidates[0]
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) if content else None
            if not parts:
                continue

            for i, part in enumerate(parts):
                # Gemini 3+: part.thought marks internal reasoning text
                if getattr(part, 'thought', False) and part.text:
                    yield StreamChunk(thinking=part.text)
                elif part.text:
                    yield StreamChunk(text=part.text)
                elif part.function_call:
                    fc = part.function_call
                    # Capture thought_signature as base64 string for JSON serialization
                    raw_ts = getattr(part, 'thought_signature', None)
                    ts_str = base64.b64encode(raw_ts).decode('ascii') if isinstance(raw_ts, bytes) else raw_ts
                    yield StreamChunk(tool_call=ToolCall(
                        id=f"{fc.name}_{i}",
                        name=fc.name,
                        arguments=dict(fc.args) if fc.args else {},
                        thought_signature=ts_str,
                    ))

        yield StreamChunk(done=True, usage=total_usage)

    # ----- retry logic -----

    def _call_with_retries(
        self,
        model: str,
        contents: list[genai_types.Content],
        config: genai_types.GenerateContentConfig,
        stream: bool = False,
    ) -> Any:
        """Call Google API with retries.

        Rate-limit errors are NOT retried internally — they bubble up to
        the router which handles API key rotation.
        """
        if stream:
            return call_with_retries(
                self._client.models.generate_content_stream,
                model=model,
                contents=contents,
                config=config,
                skip_rate_limit_retry=True,
            )
        return call_with_retries(
            self._client.models.generate_content,
            model=model,
            contents=contents,
            config=config,
            skip_rate_limit_retry=True,
        )
