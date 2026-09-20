"""Tests for the Google Gemini LLM provider."""

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from unjess.llm.base import LLMProvider, LLMResponse, StreamChunk, ToolCall, Usage
from unjess.llm.google_provider import (
    GoogleProvider,
    _convert_messages_for_google,
    _convert_tools_to_google,
    _extract_system_text,
    _sanitize_schema,
)


# ---------------------------------------------------------------------------
# Helpers — mock factories
# ---------------------------------------------------------------------------


def _make_text_part(text: str = "Hello") -> SimpleNamespace:
    return SimpleNamespace(text=text, function_call=None)


def _make_function_call_part(name: str = "read_file", args: dict[str, Any] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        text=None,
        function_call=SimpleNamespace(name=name, args=args or {"path": "x.py"}),
    )


def _make_usage_metadata(
    prompt: int = 10,
    candidates: int = 20,
    thinking: int = 0,
    cached: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        prompt_token_count=prompt,
        candidates_token_count=candidates,
        thoughts_token_count=thinking,
        cached_content_token_count=cached,
    )


def _make_candidate(parts: list[SimpleNamespace] | None = None, finish_reason: str = "STOP") -> SimpleNamespace:
    if parts is None:
        parts = [_make_text_part()]
    content = SimpleNamespace(parts=parts)
    return SimpleNamespace(content=content, finish_reason=finish_reason)


def _make_response(
    parts: list[SimpleNamespace] | None = None,
    model: str = "gemini-3.1-flash",
    finish_reason: str = "STOP",
    usage: SimpleNamespace | None = None,
    candidates: list[SimpleNamespace] | None = None,
) -> SimpleNamespace:
    if candidates is None:
        candidates = [_make_candidate(parts, finish_reason)]
    if usage is None:
        usage = _make_usage_metadata()
    return SimpleNamespace(candidates=candidates, usage_metadata=usage, model=model)


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def provider() -> GoogleProvider:
    """Create a GoogleProvider with a mocked genai client."""
    with patch("unjess.llm.google_provider.genai") as mock_genai:
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        p = GoogleProvider(api_key="test-key", default_model="gemini-3.1-flash")
    p._mock_client = mock_client  # type: ignore[attr-defined]
    return p


# ===================================================================
# _sanitize_schema
# ===================================================================


class TestSanitizeSchema:
    """Tests for the schema sanitization helper."""

    def test_strips_top_level_forbidden_keys(self) -> None:
        schema = {
            "type": "object",
            "$schema": "http://json-schema.org/draft-07/schema#",
            "additionalProperties": False,
            "properties": {"x": {"type": "string"}},
        }
        result = _sanitize_schema(schema)
        assert "$schema" not in result
        assert "additionalProperties" not in result
        assert result["type"] == "object"
        assert result["properties"]["x"]["type"] == "string"

    def test_strips_nested_forbidden_keys(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "nested": {
                    "type": "object",
                    "$ref": "#/definitions/Foo",
                    "properties": {"a": {"type": "string"}},
                },
            },
        }
        result = _sanitize_schema(schema)
        assert "$ref" not in result["properties"]["nested"]

    def test_strips_in_list_items(self) -> None:
        schema = {
            "anyOf": [
                {"type": "string", "$defs": {"x": {}}},
                {"type": "number"},
            ],
        }
        result = _sanitize_schema(schema)
        assert "$defs" not in result["anyOf"][0]

    def test_preserves_valid_keys(self) -> None:
        schema = {"type": "object", "required": ["a"], "description": "Test"}
        result = _sanitize_schema(schema)
        assert result == schema

    def test_empty_schema(self) -> None:
        assert _sanitize_schema({}) == {}

    def test_definitions_stripped(self) -> None:
        schema = {"definitions": {"Foo": {"type": "string"}}, "type": "object"}
        result = _sanitize_schema(schema)
        assert "definitions" not in result


# ===================================================================
# _extract_system_text
# ===================================================================


class TestExtractSystemText:
    """Tests for system message extraction."""

    def test_no_system(self) -> None:
        msgs = [{"role": "user", "content": "Hi"}]
        system, remaining = _extract_system_text(msgs)
        assert system == ""
        assert remaining == msgs

    def test_single_system(self) -> None:
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        system, remaining = _extract_system_text(msgs)
        assert system == "You are helpful."
        assert len(remaining) == 1

    def test_multiple_systems_concatenated(self) -> None:
        msgs = [
            {"role": "system", "content": "Part A"},
            {"role": "system", "content": "Part B"},
            {"role": "user", "content": "Hi"},
        ]
        system, remaining = _extract_system_text(msgs)
        assert system == "Part A\n\nPart B"
        assert len(remaining) == 1


# ===================================================================
# _convert_messages_for_google
# ===================================================================


class TestConvertMessagesForGoogle:
    """Tests for OpenAI-to-Google message format conversion."""

    def test_user_message(self) -> None:
        with patch("unjess.llm.google_provider.genai_types") as mock_types:
            mock_types.Part.from_text.return_value = "text_part"
            mock_types.Content.return_value = "content_obj"
            msgs = [{"role": "user", "content": "Hello"}]
            result = _convert_messages_for_google(msgs)
            mock_types.Content.assert_called_once_with(role="user", parts=["text_part"])
            assert len(result) == 1

    def test_assistant_role_maps_to_model(self) -> None:
        with patch("unjess.llm.google_provider.genai_types") as mock_types:
            mock_types.Part.from_text.return_value = "text_part"
            mock_types.Content.return_value = "content_obj"
            msgs = [{"role": "assistant", "content": "I'll help"}]
            result = _convert_messages_for_google(msgs)
            mock_types.Content.assert_called_once_with(role="model", parts=["text_part"])

    def test_tool_role_uses_function_response(self) -> None:
        with patch("unjess.llm.google_provider.genai_types") as mock_types:
            mock_types.Part.from_function_response.return_value = "func_resp_part"
            mock_types.Content.return_value = "content_obj"
            msgs = [{"role": "tool", "name": "read_file", "content": "file contents"}]
            result = _convert_messages_for_google(msgs)
            mock_types.Part.from_function_response.assert_called_once_with(
                name="read_file",
                response={"result": "file contents"},
            )
            mock_types.Content.assert_called_once_with(role="user", parts=["func_resp_part"])

    def test_assistant_with_tool_calls(self) -> None:
        with patch("unjess.llm.google_provider.genai_types") as mock_types:
            mock_types.Part.from_text.return_value = "text_part"
            mock_types.Part.from_function_call.return_value = "fc_part"
            mock_types.Content.return_value = "content_obj"
            msgs = [{
                "role": "assistant",
                "content": "Let me check.",
                "tool_calls": [{
                    "function": {"name": "read_file", "arguments": '{"path":"x.py"}'},
                }],
            }]
            result = _convert_messages_for_google(msgs)
            mock_types.Part.from_function_call.assert_called_once_with(
                name="read_file", args={"path": "x.py"},
            )

    def test_assistant_tool_calls_invalid_json(self) -> None:
        with patch("unjess.llm.google_provider.genai_types") as mock_types:
            mock_types.Part.from_function_call.return_value = "fc_part"
            mock_types.Content.return_value = "content_obj"
            msgs = [{
                "role": "assistant",
                "tool_calls": [{"function": {"name": "f", "arguments": "bad{json"}}],
            }]
            result = _convert_messages_for_google(msgs)
            mock_types.Part.from_function_call.assert_called_once_with(name="f", args={})

    def test_empty_text_message_skipped(self) -> None:
        with patch("unjess.llm.google_provider.genai_types") as mock_types:
            msgs = [{"role": "user", "content": ""}]
            result = _convert_messages_for_google(msgs)
            # No Content should be created for empty messages
            assert len(result) == 0

    def test_missing_tool_name_defaults(self) -> None:
        with patch("unjess.llm.google_provider.genai_types") as mock_types:
            mock_types.Part.from_function_response.return_value = "part"
            mock_types.Content.return_value = "content"
            msgs = [{"role": "tool", "content": "result"}]
            result = _convert_messages_for_google(msgs)
            mock_types.Part.from_function_response.assert_called_once_with(
                name="unknown", response={"result": "result"},
            )


# ===================================================================
# _convert_tools_to_google
# ===================================================================


class TestConvertToolsToGoogle:
    """Tests for tool schema conversion to Google format."""

    def test_basic_conversion(self) -> None:
        with patch("unjess.llm.google_provider.genai_types") as mock_types:
            mock_decl = MagicMock()
            mock_types.FunctionDeclaration.return_value = mock_decl
            mock_tool = MagicMock()
            mock_types.Tool.return_value = mock_tool

            tools = [{
                "name": "read_file",
                "description": "Read a file",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            }]
            result = _convert_tools_to_google(tools)
            assert len(result) == 1
            mock_types.FunctionDeclaration.assert_called_once()
            call_kwargs = mock_types.FunctionDeclaration.call_args.kwargs
            assert call_kwargs["name"] == "read_file"
            assert call_kwargs["description"] == "Read a file"

    def test_sanitizes_schema(self) -> None:
        with patch("unjess.llm.google_provider.genai_types") as mock_types:
            mock_types.FunctionDeclaration.return_value = MagicMock()
            mock_types.Tool.return_value = MagicMock()

            tools = [{
                "name": "f",
                "parameters": {
                    "type": "object",
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "properties": {"x": {"type": "string"}},
                },
            }]
            result = _convert_tools_to_google(tools)
            call_kwargs = mock_types.FunctionDeclaration.call_args.kwargs
            assert "$schema" not in call_kwargs["parameters"]

    def test_missing_fields_default(self) -> None:
        with patch("unjess.llm.google_provider.genai_types") as mock_types:
            mock_types.FunctionDeclaration.return_value = MagicMock()
            mock_types.Tool.return_value = MagicMock()
            tools = [{"name": "foo"}]
            _convert_tools_to_google(tools)
            call_kwargs = mock_types.FunctionDeclaration.call_args.kwargs
            assert call_kwargs["description"] == ""
            assert call_kwargs["parameters"] == {"type": "object", "properties": {}}

    def test_empty_list(self) -> None:
        with patch("unjess.llm.google_provider.genai_types") as mock_types:
            mock_types.Tool.return_value = MagicMock()
            result = _convert_tools_to_google([])
            assert len(result) == 1  # still wraps in a Tool
            mock_types.Tool.assert_called_once_with(function_declarations=[])


# ===================================================================
# Provider basics
# ===================================================================


class TestProviderBasics:
    """Tests for basic provider properties and interface compliance."""

    def test_implements_llm_provider(self, provider: GoogleProvider) -> None:
        assert isinstance(provider, LLMProvider)

    def test_provider_name(self, provider: GoogleProvider) -> None:
        assert provider.provider_name == "google"


# ===================================================================
# list_models()
# ===================================================================


class TestListModels:
    """Tests for the list_models method."""

    def test_list_models_basic(self, provider: GoogleProvider) -> None:
        mock_model = SimpleNamespace(
            name="models/gemini-3.1-flash",
            supported_actions=["generateContent"],
        )
        provider._mock_client.models.list.return_value = [mock_model]  # type: ignore[attr-defined]
        result = provider.list_models()
        assert result == ["gemini-3.1-flash"]

    def test_list_models_strips_prefix(self, provider: GoogleProvider) -> None:
        mock_model = SimpleNamespace(name="models/gemini-pro", supported_actions=["generateContent"])
        provider._mock_client.models.list.return_value = [mock_model]  # type: ignore[attr-defined]
        result = provider.list_models()
        assert result == ["gemini-pro"]

    def test_list_models_no_prefix(self, provider: GoogleProvider) -> None:
        mock_model = SimpleNamespace(name="gemini-pro", supported_actions=["generateContent"])
        provider._mock_client.models.list.return_value = [mock_model]  # type: ignore[attr-defined]
        result = provider.list_models()
        assert result == ["gemini-pro"]

    def test_list_models_filters_non_generate_content(self, provider: GoogleProvider) -> None:
        models = [
            SimpleNamespace(name="models/gemini-pro", supported_actions=["generateContent"]),
            SimpleNamespace(name="models/embedding-001", supported_actions=["embedContent"]),
        ]
        provider._mock_client.models.list.return_value = models  # type: ignore[attr-defined]
        result = provider.list_models()
        assert "gemini-pro" in result
        assert "embedding-001" not in result

    def test_list_models_empty_actions_included(self, provider: GoogleProvider) -> None:
        mock_model = SimpleNamespace(name="models/gemini-x", supported_actions=[])
        provider._mock_client.models.list.return_value = [mock_model]  # type: ignore[attr-defined]
        result = provider.list_models()
        assert "gemini-x" in result

    def test_list_models_no_supported_actions_included(self, provider: GoogleProvider) -> None:
        mock_model = SimpleNamespace(name="models/gemini-y")
        # no supported_actions attribute at all
        provider._mock_client.models.list.return_value = [mock_model]  # type: ignore[attr-defined]
        result = provider.list_models()
        assert "gemini-y" in result

    def test_list_models_empty_name_excluded(self, provider: GoogleProvider) -> None:
        mock_model = SimpleNamespace(name="", supported_actions=["generateContent"])
        provider._mock_client.models.list.return_value = [mock_model]  # type: ignore[attr-defined]
        result = provider.list_models()
        assert result == []

    def test_list_models_exception_returns_empty(self, provider: GoogleProvider) -> None:
        provider._mock_client.models.list.side_effect = Exception("Network error")  # type: ignore[attr-defined]
        assert provider.list_models() == []

    def test_list_models_sorted(self, provider: GoogleProvider) -> None:
        models = [
            SimpleNamespace(name="models/gemini-pro", supported_actions=["generateContent"]),
            SimpleNamespace(name="models/gemini-flash", supported_actions=["generateContent"]),
        ]
        provider._mock_client.models.list.return_value = models  # type: ignore[attr-defined]
        result = provider.list_models()
        assert result == ["gemini-flash", "gemini-pro"]


# ===================================================================
# chat()
# ===================================================================


class TestChat:
    """Tests for the non-streaming chat method."""

    def test_chat_returns_llm_response(self, provider: GoogleProvider) -> None:
        raw = _make_response(parts=[_make_text_part("Hi there")])
        provider._mock_client.models.generate_content.return_value = raw  # type: ignore[attr-defined]
        resp = provider.chat([{"role": "user", "content": "Hello"}])
        assert isinstance(resp, LLMResponse)
        assert resp.text == "Hi there"
        assert resp.model == "gemini-3.1-flash"

    def test_chat_uses_default_model(self, provider: GoogleProvider) -> None:
        raw = _make_response()
        provider._mock_client.models.generate_content.return_value = raw  # type: ignore[attr-defined]
        provider.chat([{"role": "user", "content": "Hi"}])
        call_args = provider._mock_client.models.generate_content.call_args  # type: ignore[attr-defined]
        assert "gemini-3.1-flash" in str(call_args)

    def test_chat_model_override(self, provider: GoogleProvider) -> None:
        raw = _make_response()
        provider._mock_client.models.generate_content.return_value = raw  # type: ignore[attr-defined]
        resp = provider.chat([{"role": "user", "content": "Hi"}], model="gemini-3.1-pro")
        assert resp.model == "gemini-3.1-pro"

    def test_chat_with_tool_calls(self, provider: GoogleProvider) -> None:
        parts = [
            _make_text_part("Let me read that."),
            _make_function_call_part("read_file", {"path": "a.py"}),
        ]
        raw = _make_response(parts=parts)
        provider._mock_client.models.generate_content.return_value = raw  # type: ignore[attr-defined]
        resp = provider.chat([{"role": "user", "content": "Read a.py"}])
        assert resp.text == "Let me read that."
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "read_file"
        assert resp.tool_calls[0].arguments == {"path": "a.py"}
        # ID should be name_index format
        assert resp.tool_calls[0].id == "read_file_1"

    def test_chat_tool_call_no_args(self, provider: GoogleProvider) -> None:
        parts = [_make_function_call_part("no_args")]
        parts[0].function_call.args = None
        raw = _make_response(parts=parts)
        provider._mock_client.models.generate_content.return_value = raw  # type: ignore[attr-defined]
        resp = provider.chat([{"role": "user", "content": "run"}])
        assert resp.tool_calls[0].arguments == {}

    def test_chat_usage_tokens(self, provider: GoogleProvider) -> None:
        usage = _make_usage_metadata(100, 50, thinking=30, cached=20)
        raw = _make_response(usage=usage)
        provider._mock_client.models.generate_content.return_value = raw  # type: ignore[attr-defined]
        resp = provider.chat([{"role": "user", "content": "Hi"}])
        assert resp.usage.prompt_tokens == 100
        assert resp.usage.completion_tokens == 50
        assert resp.usage.thinking_tokens == 30
        assert resp.usage.cache_read_tokens == 20

    def test_chat_none_usage_metadata(self, provider: GoogleProvider) -> None:
        raw = _make_response()
        raw.usage_metadata = None
        # Also need candidates for finish_reason
        raw.candidates = []
        provider._mock_client.models.generate_content.return_value = raw  # type: ignore[attr-defined]
        resp = provider.chat([{"role": "user", "content": "Hi"}])
        assert resp.usage.prompt_tokens == 0
        assert resp.usage.completion_tokens == 0

    def test_chat_no_candidates(self, provider: GoogleProvider) -> None:
        raw = _make_response()
        raw.candidates = []
        provider._mock_client.models.generate_content.return_value = raw  # type: ignore[attr-defined]
        resp = provider.chat([{"role": "user", "content": "Hi"}])
        assert resp.text == ""
        assert resp.tool_calls == []
        assert resp.finish_reason == ""

    def test_chat_candidate_no_content(self, provider: GoogleProvider) -> None:
        candidate = SimpleNamespace(content=None, finish_reason="STOP")
        raw = _make_response(candidates=[candidate])
        provider._mock_client.models.generate_content.return_value = raw  # type: ignore[attr-defined]
        resp = provider.chat([{"role": "user", "content": "Hi"}])
        assert resp.text == ""

    def test_chat_candidate_no_parts(self, provider: GoogleProvider) -> None:
        candidate = SimpleNamespace(content=SimpleNamespace(parts=None), finish_reason="STOP")
        raw = _make_response(candidates=[candidate])
        provider._mock_client.models.generate_content.return_value = raw  # type: ignore[attr-defined]
        resp = provider.chat([{"role": "user", "content": "Hi"}])
        assert resp.text == ""

    def test_chat_multiple_text_parts_joined(self, provider: GoogleProvider) -> None:
        parts = [_make_text_part("Part 1"), _make_text_part("Part 2")]
        raw = _make_response(parts=parts)
        provider._mock_client.models.generate_content.return_value = raw  # type: ignore[attr-defined]
        resp = provider.chat([{"role": "user", "content": "Hi"}])
        assert resp.text == "Part 1\nPart 2"

    def test_chat_finish_reason_from_candidate(self, provider: GoogleProvider) -> None:
        raw = _make_response(finish_reason="MAX_TOKENS")
        provider._mock_client.models.generate_content.return_value = raw  # type: ignore[attr-defined]
        resp = provider.chat([{"role": "user", "content": "Hi"}])
        assert resp.finish_reason == "MAX_TOKENS"

    def test_chat_thinking_tokens(self, provider: GoogleProvider) -> None:
        usage = _make_usage_metadata(10, 200, thinking=150)
        raw = _make_response(usage=usage)
        provider._mock_client.models.generate_content.return_value = raw  # type: ignore[attr-defined]
        resp = provider.chat([{"role": "user", "content": "think"}])
        assert resp.usage.thinking_tokens == 150


# ===================================================================
# chat_stream()
# ===================================================================


class TestChatStream:
    """Tests for the streaming chat method."""

    def test_stream_text_chunks(self, provider: GoogleProvider) -> None:
        chunk1 = _make_response(parts=[_make_text_part("Hello ")])
        chunk2 = _make_response(parts=[_make_text_part("world")], usage=_make_usage_metadata(10, 5))
        provider._mock_client.models.generate_content_stream.return_value = iter([chunk1, chunk2])  # type: ignore[attr-defined]

        results = list(provider.chat_stream([{"role": "user", "content": "Hi"}]))
        texts = [r.text for r in results if r.text]
        assert texts == ["Hello ", "world"]
        assert results[-1].done is True
        assert results[-1].usage is not None

    def test_stream_tool_call_chunks(self, provider: GoogleProvider) -> None:
        chunk = _make_response(parts=[_make_function_call_part("read_file", {"path": "a.py"})])
        provider._mock_client.models.generate_content_stream.return_value = iter([chunk])  # type: ignore[attr-defined]

        results = list(provider.chat_stream([{"role": "user", "content": "read"}]))
        tool_chunks = [r for r in results if r.tool_call is not None]
        assert len(tool_chunks) == 1
        assert tool_chunks[0].tool_call.name == "read_file"
        assert tool_chunks[0].tool_call.arguments == {"path": "a.py"}

    def test_stream_no_candidates_skipped(self, provider: GoogleProvider) -> None:
        chunk = SimpleNamespace(
            candidates=[],
            usage_metadata=_make_usage_metadata(5, 2),
        )
        provider._mock_client.models.generate_content_stream.return_value = iter([chunk])  # type: ignore[attr-defined]

        results = list(provider.chat_stream([{"role": "user", "content": "Hi"}]))
        assert results[-1].done is True
        # Only the done chunk
        assert len([r for r in results if r.text]) == 0

    def test_stream_no_parts_skipped(self, provider: GoogleProvider) -> None:
        candidate = SimpleNamespace(content=SimpleNamespace(parts=None), finish_reason="STOP")
        chunk = SimpleNamespace(candidates=[candidate], usage_metadata=_make_usage_metadata())
        provider._mock_client.models.generate_content_stream.return_value = iter([chunk])  # type: ignore[attr-defined]

        results = list(provider.chat_stream([{"role": "user", "content": "Hi"}]))
        assert results[-1].done is True
        assert len([r for r in results if r.text]) == 0

    def test_stream_no_content_skipped(self, provider: GoogleProvider) -> None:
        candidate = SimpleNamespace(content=None, finish_reason="STOP")
        chunk = SimpleNamespace(candidates=[candidate], usage_metadata=_make_usage_metadata())
        provider._mock_client.models.generate_content_stream.return_value = iter([chunk])  # type: ignore[attr-defined]

        results = list(provider.chat_stream([{"role": "user", "content": "Hi"}]))
        assert results[-1].done is True

    def test_stream_usage_updates_with_last_chunk(self, provider: GoogleProvider) -> None:
        chunk1 = _make_response(parts=[_make_text_part("a")], usage=_make_usage_metadata(10, 1))
        chunk2 = _make_response(parts=[_make_text_part("b")], usage=_make_usage_metadata(10, 5, thinking=3, cached=2))
        provider._mock_client.models.generate_content_stream.return_value = iter([chunk1, chunk2])  # type: ignore[attr-defined]

        results = list(provider.chat_stream([{"role": "user", "content": "Hi"}]))
        final = results[-1]
        assert final.usage.completion_tokens == 5
        assert final.usage.thinking_tokens == 3
        assert final.usage.cache_read_tokens == 2

    def test_stream_no_usage_metadata_defaults_zero(self, provider: GoogleProvider) -> None:
        chunk = _make_response(parts=[_make_text_part("x")])
        chunk.usage_metadata = None
        provider._mock_client.models.generate_content_stream.return_value = iter([chunk])  # type: ignore[attr-defined]

        results = list(provider.chat_stream([{"role": "user", "content": "Hi"}]))
        final = results[-1]
        assert final.usage.prompt_tokens == 0
        assert final.usage.completion_tokens == 0

    def test_stream_function_call_no_args(self, provider: GoogleProvider) -> None:
        part = _make_function_call_part("no_args")
        part.function_call.args = None
        chunk = _make_response(parts=[part])
        provider._mock_client.models.generate_content_stream.return_value = iter([chunk])  # type: ignore[attr-defined]

        results = list(provider.chat_stream([{"role": "user", "content": "run"}]))
        tool_chunks = [r for r in results if r.tool_call is not None]
        assert tool_chunks[0].tool_call.arguments == {}


# ===================================================================
# Error handling
# ===================================================================


class TestErrorHandling:
    """Tests for error propagation and edge cases."""

    def test_api_error_propagates(self, provider: GoogleProvider) -> None:
        provider._mock_client.models.generate_content.side_effect = RuntimeError("API error")  # type: ignore[attr-defined]
        with pytest.raises(RuntimeError, match="API error"):
            provider.chat([{"role": "user", "content": "Hi"}])

    def test_stream_error_propagates(self, provider: GoogleProvider) -> None:
        provider._mock_client.models.generate_content_stream.side_effect = RuntimeError("Stream error")  # type: ignore[attr-defined]
        with pytest.raises(RuntimeError, match="Stream error"):
            list(provider.chat_stream([{"role": "user", "content": "Hi"}]))
