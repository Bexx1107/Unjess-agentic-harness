"""Tests for the Anthropic LLM provider."""

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from unjess.llm.base import LLMProvider, LLMResponse, StreamChunk, ToolCall, Usage
from unjess.llm.anthropic_provider import (
    AnthropicProvider,
    _convert_tools_to_anthropic,
    _split_system_message,
    _convert_messages_for_anthropic,
)


# ---------------------------------------------------------------------------
# Helpers — mock object factories
# ---------------------------------------------------------------------------


def _make_text_block(text: str = "Hello") -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def _make_thinking_block(thinking: str = "Let me think...") -> SimpleNamespace:
    return SimpleNamespace(type="thinking", thinking=thinking)


def _make_tool_use_block(
    block_id: str = "toolu_1",
    name: str = "read_file",
    input_data: dict[str, Any] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        type="tool_use",
        id=block_id,
        name=name,
        input=input_data if input_data is not None else {"path": "x.py"},
    )


def _make_usage(
    input_tokens: int = 10,
    output_tokens: int = 20,
    cache_read: int = 0,
    cache_creation: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_creation,
    )


def _make_response(
    content: list[SimpleNamespace] | None = None,
    model: str = "claude-sonnet-4-20250514",
    stop_reason: str = "end_turn",
    usage: SimpleNamespace | None = None,
) -> SimpleNamespace:
    if content is None:
        content = [_make_text_block("Hello")]
    if usage is None:
        usage = _make_usage()
    return SimpleNamespace(
        content=content,
        model=model,
        stop_reason=stop_reason,
        usage=usage,
    )


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def provider() -> AnthropicProvider:
    """Create a provider with a mocked Anthropic client."""
    with patch("unjess.llm.anthropic_provider.anthropic") as mock_mod:
        mock_client = MagicMock()
        mock_mod.Anthropic.return_value = mock_client
        p = AnthropicProvider(api_key="test-key", default_model="claude-sonnet-4-20250514")
    p._mock_client = mock_client  # type: ignore[attr-defined]
    return p


# ===================================================================
# _split_system_message
# ===================================================================


class TestSplitSystemMessage:
    """Tests for extracting system messages from the message list."""

    def test_no_system_message(self) -> None:
        msgs = [{"role": "user", "content": "Hi"}]
        system, remaining = _split_system_message(msgs)
        assert system == ""
        assert remaining == msgs

    def test_single_system_message(self) -> None:
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        system, remaining = _split_system_message(msgs)
        assert system == "You are helpful."
        assert len(remaining) == 1
        assert remaining[0]["role"] == "user"

    def test_multiple_system_messages_concatenated(self) -> None:
        msgs = [
            {"role": "system", "content": "Part 1"},
            {"role": "user", "content": "Hi"},
            {"role": "system", "content": "Part 2"},
        ]
        system, remaining = _split_system_message(msgs)
        assert system == "Part 1\n\nPart 2"
        assert len(remaining) == 1

    def test_empty_content(self) -> None:
        msgs = [{"role": "system"}]
        system, remaining = _split_system_message(msgs)
        assert system == ""
        assert remaining == []


# ===================================================================
# _convert_messages_for_anthropic
# ===================================================================


class TestConvertMessagesForAnthropic:
    """Tests for OpenAI-to-Anthropic message format conversion."""

    def test_user_message_passthrough(self) -> None:
        msgs = [{"role": "user", "content": "Hello"}]
        result = _convert_messages_for_anthropic(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello"

    def test_assistant_text_only(self) -> None:
        msgs = [{"role": "assistant", "content": "I'll help you"}]
        result = _convert_messages_for_anthropic(msgs)
        assert result[0]["role"] == "assistant"
        assert len(result[0]["content"]) == 1
        assert result[0]["content"][0] == {"type": "text", "text": "I'll help you"}

    def test_assistant_with_tool_calls(self) -> None:
        msgs = [{
            "role": "assistant",
            "content": "Let me read that.",
            "tool_calls": [{
                "id": "call_1",
                "function": {
                    "name": "read_file",
                    "arguments": '{"path": "x.py"}',
                },
            }],
        }]
        result = _convert_messages_for_anthropic(msgs)
        blocks = result[0]["content"]
        assert len(blocks) == 2
        assert blocks[0]["type"] == "text"
        assert blocks[1]["type"] == "tool_use"
        assert blocks[1]["id"] == "call_1"
        assert blocks[1]["name"] == "read_file"
        assert blocks[1]["input"] == {"path": "x.py"}

    def test_assistant_tool_calls_invalid_json(self) -> None:
        msgs = [{
            "role": "assistant",
            "tool_calls": [{
                "id": "call_1",
                "function": {"name": "f", "arguments": "not-json"},
            }],
        }]
        result = _convert_messages_for_anthropic(msgs)
        assert result[0]["content"][0]["input"] == {}

    def test_assistant_no_content_no_tools_uses_original(self) -> None:
        msgs = [{"role": "assistant"}]
        result = _convert_messages_for_anthropic(msgs)
        assert result[0]["content"] == ""

    def test_tool_result_single(self) -> None:
        msgs = [{"role": "tool", "tool_call_id": "call_1", "content": "file contents"}]
        result = _convert_messages_for_anthropic(msgs)
        assert result[0]["role"] == "user"
        assert isinstance(result[0]["content"], list)
        assert result[0]["content"][0]["type"] == "tool_result"
        assert result[0]["content"][0]["tool_use_id"] == "call_1"
        assert result[0]["content"][0]["content"] == "file contents"

    def test_consecutive_tool_results_merged(self) -> None:
        msgs = [
            {"role": "tool", "tool_call_id": "call_1", "content": "result1"},
            {"role": "tool", "tool_call_id": "call_2", "content": "result2"},
        ]
        result = _convert_messages_for_anthropic(msgs)
        # Should be merged into a single user message
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert len(result[0]["content"]) == 2

    def test_tool_result_after_non_tool_not_merged(self) -> None:
        msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "tool", "tool_call_id": "call_1", "content": "result"},
        ]
        result = _convert_messages_for_anthropic(msgs)
        assert len(result) == 2

    def test_default_role_is_user(self) -> None:
        msgs = [{"content": "no role specified"}]
        result = _convert_messages_for_anthropic(msgs)
        assert result[0]["role"] == "user"


# ===================================================================
# _convert_tools_to_anthropic
# ===================================================================


class TestConvertToolsToAnthropic:
    """Tests for tool schema conversion to Anthropic format."""

    def test_basic_conversion(self) -> None:
        tools = [{
            "name": "read_file",
            "description": "Read a file",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
        }]
        result = _convert_tools_to_anthropic(tools)
        assert len(result) == 1
        assert result[0]["name"] == "read_file"
        assert result[0]["description"] == "Read a file"
        assert result[0]["input_schema"] == tools[0]["parameters"]

    def test_missing_fields_default(self) -> None:
        tools = [{"name": "foo"}]
        result = _convert_tools_to_anthropic(tools)
        assert result[0]["description"] == ""
        assert result[0]["input_schema"] == {"type": "object", "properties": {}}

    def test_empty_list(self) -> None:
        assert _convert_tools_to_anthropic([]) == []


# ===================================================================
# Provider basics
# ===================================================================


class TestProviderBasics:
    """Tests for basic provider properties and interface compliance."""

    def test_implements_llm_provider(self, provider: AnthropicProvider) -> None:
        assert isinstance(provider, LLMProvider)

    def test_provider_name(self, provider: AnthropicProvider) -> None:
        assert provider.provider_name == "anthropic"

    def test_list_models_fallback_on_error(self, provider: AnthropicProvider) -> None:
        with patch.object(provider._client.models, "list", side_effect=Exception("API Error")):
            models = provider.list_models()
        assert len(models) == 4
        assert "claude-sonnet-4-20250514" in models
        assert "claude-opus-4-20250514" in models


# ===================================================================
# chat()
# ===================================================================


class TestChat:
    """Tests for the non-streaming chat method."""

    def test_chat_returns_llm_response(self, provider: AnthropicProvider) -> None:
        provider._mock_client.messages.create.return_value = _make_response()  # type: ignore[attr-defined]
        resp = provider.chat([{"role": "user", "content": "Hi"}])
        assert isinstance(resp, LLMResponse)
        assert resp.text == "Hello"
        assert resp.finish_reason == "end_turn"
        assert resp.model == "claude-sonnet-4-20250514"

    def test_chat_with_system_message(self, provider: AnthropicProvider) -> None:
        provider._mock_client.messages.create.return_value = _make_response()  # type: ignore[attr-defined]
        msgs = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hi"},
        ]
        provider.chat(msgs)
        call_kwargs = provider._mock_client.messages.create.call_args  # type: ignore[attr-defined]
        # System should be passed as structured block with cache_control
        assert "system" in str(call_kwargs)

    def test_chat_without_system_message(self, provider: AnthropicProvider) -> None:
        provider._mock_client.messages.create.return_value = _make_response()  # type: ignore[attr-defined]
        provider.chat([{"role": "user", "content": "Hi"}])
        call_kwargs = provider._mock_client.messages.create.call_args  # type: ignore[attr-defined]
        # system should NOT be in kwargs when no system message
        kwargs_dict = call_kwargs.kwargs if call_kwargs.kwargs else call_kwargs[1]
        assert "system" not in kwargs_dict

    def test_chat_with_tools(self, provider: AnthropicProvider) -> None:
        provider._mock_client.messages.create.return_value = _make_response()  # type: ignore[attr-defined]
        tools = [{"name": "read_file", "description": "Read", "parameters": {"type": "object", "properties": {}}}]
        provider.chat([{"role": "user", "content": "Hi"}], tools=tools)
        call_kwargs = provider._mock_client.messages.create.call_args  # type: ignore[attr-defined]
        assert "tools" in str(call_kwargs)

    def test_chat_tools_have_cache_control_on_last(self, provider: AnthropicProvider) -> None:
        provider._mock_client.messages.create.return_value = _make_response()  # type: ignore[attr-defined]
        tools = [{"name": "a"}, {"name": "b"}]
        provider.chat([{"role": "user", "content": "Hi"}], tools=tools)
        call_kwargs = provider._mock_client.messages.create.call_args  # type: ignore[attr-defined]
        kwargs_dict = call_kwargs.kwargs if call_kwargs.kwargs else call_kwargs[1]
        ant_tools = kwargs_dict["tools"]
        assert "cache_control" not in ant_tools[0]
        assert ant_tools[-1]["cache_control"] == {"type": "ephemeral"}

    def test_chat_with_thinking_blocks(self, provider: AnthropicProvider) -> None:
        content = [
            _make_thinking_block("Step 1: analyze"),
            _make_text_block("Here's the answer"),
        ]
        provider._mock_client.messages.create.return_value = _make_response(content=content)  # type: ignore[attr-defined]
        resp = provider.chat([{"role": "user", "content": "Think hard"}])
        assert resp.thinking_text == "Step 1: analyze"
        assert resp.text == "Here's the answer"

    def test_chat_with_tool_use_blocks(self, provider: AnthropicProvider) -> None:
        content = [
            _make_text_block("I'll read the file."),
            _make_tool_use_block("toolu_1", "read_file", {"path": "a.py"}),
        ]
        provider._mock_client.messages.create.return_value = _make_response(content=content)  # type: ignore[attr-defined]
        resp = provider.chat([{"role": "user", "content": "Read a.py"}])
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].id == "toolu_1"
        assert resp.tool_calls[0].name == "read_file"
        assert resp.tool_calls[0].arguments == {"path": "a.py"}

    def test_chat_tool_use_non_dict_input(self, provider: AnthropicProvider) -> None:
        block = SimpleNamespace(type="tool_use", id="t1", name="f", input="not-a-dict")
        provider._mock_client.messages.create.return_value = _make_response(content=[block])  # type: ignore[attr-defined]
        resp = provider.chat([{"role": "user", "content": "Hi"}])
        assert resp.tool_calls[0].arguments == {}

    def test_chat_usage_tokens(self, provider: AnthropicProvider) -> None:
        usage = _make_usage(100, 50, cache_read=30, cache_creation=10)
        provider._mock_client.messages.create.return_value = _make_response(usage=usage)  # type: ignore[attr-defined]
        resp = provider.chat([{"role": "user", "content": "Hi"}])
        assert resp.usage.prompt_tokens == 100
        assert resp.usage.completion_tokens == 50
        assert resp.usage.cache_read_tokens == 30
        assert resp.usage.cache_creation_tokens == 10

    def test_chat_none_usage(self, provider: AnthropicProvider) -> None:
        raw = _make_response()
        raw.usage = None
        provider._mock_client.messages.create.return_value = raw  # type: ignore[attr-defined]
        resp = provider.chat([{"role": "user", "content": "Hi"}])
        assert resp.usage.prompt_tokens == 0
        assert resp.usage.completion_tokens == 0

    def test_chat_none_stop_reason(self, provider: AnthropicProvider) -> None:
        raw = _make_response(stop_reason=None)
        provider._mock_client.messages.create.return_value = raw  # type: ignore[attr-defined]
        resp = provider.chat([{"role": "user", "content": "Hi"}])
        assert resp.finish_reason == ""

    def test_chat_none_model_fallback(self, provider: AnthropicProvider) -> None:
        raw = _make_response(model=None)
        provider._mock_client.messages.create.return_value = raw  # type: ignore[attr-defined]
        resp = provider.chat([{"role": "user", "content": "Hi"}])
        assert resp.model == "claude-sonnet-4-20250514"

    def test_chat_model_override(self, provider: AnthropicProvider) -> None:
        provider._mock_client.messages.create.return_value = _make_response()  # type: ignore[attr-defined]
        provider.chat([{"role": "user", "content": "Hi"}], model="claude-opus-4-20250514")
        call_kwargs = provider._mock_client.messages.create.call_args  # type: ignore[attr-defined]
        assert "claude-opus-4-20250514" in str(call_kwargs)

    def test_chat_multiple_text_blocks_joined(self, provider: AnthropicProvider) -> None:
        content = [_make_text_block("Part 1"), _make_text_block("Part 2")]
        provider._mock_client.messages.create.return_value = _make_response(content=content)  # type: ignore[attr-defined]
        resp = provider.chat([{"role": "user", "content": "Hi"}])
        assert resp.text == "Part 1\nPart 2"


# ===================================================================
# chat_stream()
# ===================================================================


class TestChatStream:
    """Tests for the streaming chat method."""

    def _make_event(self, event_type: str, **kwargs: Any) -> SimpleNamespace:
        """Create a mock streaming event."""
        return SimpleNamespace(type=event_type, **kwargs)

    def test_stream_text_chunks(self, provider: AnthropicProvider) -> None:
        events = [
            self._make_event("message_start", message=SimpleNamespace(
                usage=SimpleNamespace(input_tokens=10, cache_read_input_tokens=0, cache_creation_input_tokens=0),
            )),
            self._make_event("content_block_start", content_block=SimpleNamespace(type="text")),
            self._make_event("content_block_delta", delta=SimpleNamespace(text="Hello ")),
            self._make_event("content_block_delta", delta=SimpleNamespace(text="world")),
            self._make_event("content_block_stop"),
            self._make_event("message_delta", usage=SimpleNamespace(output_tokens=5)),
            self._make_event("message_stop"),
        ]

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=iter(events))
        mock_stream.__exit__ = MagicMock(return_value=False)
        provider._mock_client.messages.stream.return_value = mock_stream  # type: ignore[attr-defined]

        results = list(provider.chat_stream([{"role": "user", "content": "Hi"}]))
        texts = [r.text for r in results if r.text]
        assert texts == ["Hello ", "world"]
        assert results[-1].done is True
        assert results[-1].usage is not None
        assert results[-1].usage.prompt_tokens == 10
        assert results[-1].usage.completion_tokens == 5

    def test_stream_thinking_chunks(self, provider: AnthropicProvider) -> None:
        events = [
            self._make_event("message_start", message=SimpleNamespace(
                usage=SimpleNamespace(input_tokens=5, cache_read_input_tokens=0, cache_creation_input_tokens=0),
            )),
            self._make_event("content_block_start", content_block=SimpleNamespace(type="thinking")),
            self._make_event("content_block_delta", delta=SimpleNamespace(thinking="Let me think...")),
            self._make_event("content_block_stop"),
            self._make_event("content_block_start", content_block=SimpleNamespace(type="text")),
            self._make_event("content_block_delta", delta=SimpleNamespace(text="Answer")),
            self._make_event("content_block_stop"),
            self._make_event("message_delta", usage=SimpleNamespace(output_tokens=10)),
            self._make_event("message_stop"),
        ]

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=iter(events))
        mock_stream.__exit__ = MagicMock(return_value=False)
        provider._mock_client.messages.stream.return_value = mock_stream  # type: ignore[attr-defined]

        results = list(provider.chat_stream([{"role": "user", "content": "Think"}]))
        thinking_chunks = [r for r in results if r.thinking]
        text_chunks = [r for r in results if r.text]
        assert len(thinking_chunks) == 1
        assert thinking_chunks[0].thinking == "Let me think..."
        assert len(text_chunks) == 1
        assert text_chunks[0].text == "Answer"

    def test_stream_tool_use(self, provider: AnthropicProvider) -> None:
        events = [
            self._make_event("message_start", message=SimpleNamespace(
                usage=SimpleNamespace(input_tokens=10, cache_read_input_tokens=0, cache_creation_input_tokens=0),
            )),
            self._make_event("content_block_start", content_block=SimpleNamespace(
                type="tool_use", id="toolu_1", name="read_file",
            )),
            self._make_event("content_block_delta", delta=SimpleNamespace(partial_json='{"path":')),
            self._make_event("content_block_delta", delta=SimpleNamespace(partial_json='"x.py"}')),
            self._make_event("content_block_stop"),
            self._make_event("message_delta", usage=SimpleNamespace(output_tokens=15)),
            self._make_event("message_stop"),
        ]

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=iter(events))
        mock_stream.__exit__ = MagicMock(return_value=False)
        provider._mock_client.messages.stream.return_value = mock_stream  # type: ignore[attr-defined]

        results = list(provider.chat_stream([{"role": "user", "content": "Read x.py"}]))
        tool_chunks = [r for r in results if r.tool_call is not None]
        assert len(tool_chunks) == 1
        assert tool_chunks[0].tool_call.name == "read_file"
        assert tool_chunks[0].tool_call.arguments == {"path": "x.py"}

    def test_stream_tool_invalid_json(self, provider: AnthropicProvider) -> None:
        events = [
            self._make_event("message_start", message=SimpleNamespace(
                usage=SimpleNamespace(input_tokens=5, cache_read_input_tokens=0, cache_creation_input_tokens=0),
            )),
            self._make_event("content_block_start", content_block=SimpleNamespace(
                type="tool_use", id="t1", name="f",
            )),
            self._make_event("content_block_delta", delta=SimpleNamespace(partial_json="bad{json")),
            self._make_event("content_block_stop"),
            self._make_event("message_delta", usage=SimpleNamespace(output_tokens=5)),
            self._make_event("message_stop"),
        ]

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=iter(events))
        mock_stream.__exit__ = MagicMock(return_value=False)
        provider._mock_client.messages.stream.return_value = mock_stream  # type: ignore[attr-defined]

        results = list(provider.chat_stream([{"role": "user", "content": "bad"}]))
        tool_chunks = [r for r in results if r.tool_call is not None]
        assert tool_chunks[0].tool_call.arguments == {"_raw": "bad{json"}

    def test_stream_empty_tool_input(self, provider: AnthropicProvider) -> None:
        events = [
            self._make_event("message_start", message=SimpleNamespace(
                usage=SimpleNamespace(input_tokens=5, cache_read_input_tokens=0, cache_creation_input_tokens=0),
            )),
            self._make_event("content_block_start", content_block=SimpleNamespace(
                type="tool_use", id="t1", name="no_args",
            )),
            # No partial_json deltas
            self._make_event("content_block_stop"),
            self._make_event("message_delta", usage=SimpleNamespace(output_tokens=5)),
            self._make_event("message_stop"),
        ]

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=iter(events))
        mock_stream.__exit__ = MagicMock(return_value=False)
        provider._mock_client.messages.stream.return_value = mock_stream  # type: ignore[attr-defined]

        results = list(provider.chat_stream([{"role": "user", "content": "no args"}]))
        tool_chunks = [r for r in results if r.tool_call is not None]
        assert tool_chunks[0].tool_call.arguments == {}

    def test_stream_cache_tokens(self, provider: AnthropicProvider) -> None:
        events = [
            self._make_event("message_start", message=SimpleNamespace(
                usage=SimpleNamespace(
                    input_tokens=100,
                    cache_read_input_tokens=50,
                    cache_creation_input_tokens=20,
                ),
            )),
            self._make_event("content_block_start", content_block=SimpleNamespace(type="text")),
            self._make_event("content_block_delta", delta=SimpleNamespace(text="Hi")),
            self._make_event("content_block_stop"),
            self._make_event("message_delta", usage=SimpleNamespace(output_tokens=5)),
            self._make_event("message_stop"),
        ]

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=iter(events))
        mock_stream.__exit__ = MagicMock(return_value=False)
        provider._mock_client.messages.stream.return_value = mock_stream  # type: ignore[attr-defined]

        results = list(provider.chat_stream([{"role": "user", "content": "Hi"}]))
        final = results[-1]
        assert final.usage.cache_read_tokens == 50
        assert final.usage.cache_creation_tokens == 20

    def test_stream_thinking_delta_empty_ignored(self, provider: AnthropicProvider) -> None:
        events = [
            self._make_event("message_start", message=SimpleNamespace(
                usage=SimpleNamespace(input_tokens=5, cache_read_input_tokens=0, cache_creation_input_tokens=0),
            )),
            self._make_event("content_block_start", content_block=SimpleNamespace(type="thinking")),
            self._make_event("content_block_delta", delta=SimpleNamespace(thinking="")),
            self._make_event("content_block_stop"),
            self._make_event("message_delta", usage=SimpleNamespace(output_tokens=0)),
            self._make_event("message_stop"),
        ]

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=iter(events))
        mock_stream.__exit__ = MagicMock(return_value=False)
        provider._mock_client.messages.stream.return_value = mock_stream  # type: ignore[attr-defined]

        results = list(provider.chat_stream([{"role": "user", "content": "Hi"}]))
        thinking_chunks = [r for r in results if r.thinking]
        assert len(thinking_chunks) == 0


# ===================================================================
# Error handling
# ===================================================================


class TestErrorHandling:
    """Tests for error propagation and edge cases."""

    def test_api_error_propagates(self, provider: AnthropicProvider) -> None:
        provider._mock_client.messages.create.side_effect = RuntimeError("API failed")  # type: ignore[attr-defined]
        with pytest.raises(RuntimeError, match="API failed"):
            provider.chat([{"role": "user", "content": "Hi"}])
