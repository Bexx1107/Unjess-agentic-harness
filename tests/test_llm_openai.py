"""Tests for the OpenAI-compatible LLM provider."""

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from unjess.llm.base import LLMProvider, LLMResponse, StreamChunk, ToolCall, Usage
from unjess.llm.openai_compat import (
    OpenAICompatibleProvider,
    _convert_tools_to_openai,
    _parse_tool_calls,
)


# ---------------------------------------------------------------------------
# Helpers — mock object factories
# ---------------------------------------------------------------------------


def _make_usage(
    prompt: int = 10,
    completion: int = 20,
    reasoning: int = 0,
) -> SimpleNamespace:
    """Build a mock usage object matching the OpenAI SDK shape."""
    details = SimpleNamespace(reasoning_tokens=reasoning) if reasoning else None
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        completion_tokens_details=details,
    )


def _make_tool_call(
    tc_id: str = "call_1",
    name: str = "read_file",
    arguments: str = '{"path": "/tmp/x.py"}',
) -> SimpleNamespace:
    return SimpleNamespace(
        id=tc_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _make_choice(
    content: str = "Hello",
    tool_calls: list[SimpleNamespace] | None = None,
    finish_reason: str = "stop",
) -> SimpleNamespace:
    return SimpleNamespace(
        message=SimpleNamespace(content=content, tool_calls=tool_calls),
        finish_reason=finish_reason,
    )


def _make_response(
    content: str = "Hello",
    tool_calls: list[SimpleNamespace] | None = None,
    finish_reason: str = "stop",
    model: str = "gpt-4o-mini",
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
    reasoning_tokens: int = 0,
    choices: list[SimpleNamespace] | None = None,
) -> SimpleNamespace:
    """Build a full mock ChatCompletion response."""
    if choices is None:
        choices = [_make_choice(content, tool_calls, finish_reason)]
    return SimpleNamespace(
        choices=choices,
        model=model,
        usage=_make_usage(prompt_tokens, completion_tokens, reasoning_tokens),
    )


def _make_stream_chunk(
    content: str | None = None,
    tool_calls: list[SimpleNamespace] | None = None,
    finish_reason: str | None = None,
    usage: SimpleNamespace | None = None,
) -> SimpleNamespace:
    """Build a mock streaming chunk."""
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=usage)


def _make_stream_tc_delta(
    index: int = 0,
    tc_id: str | None = None,
    name: str | None = None,
    arguments: str | None = None,
) -> SimpleNamespace:
    func = SimpleNamespace(name=name, arguments=arguments) if (name or arguments) else None
    return SimpleNamespace(index=index, id=tc_id, function=func)


# ---------------------------------------------------------------------------
# Fixture: patched provider
# ---------------------------------------------------------------------------


@pytest.fixture
def provider() -> OpenAICompatibleProvider:
    """Create a provider with a mocked OpenAI client."""
    with patch("unjess.llm.openai_compat.OpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        p = OpenAICompatibleProvider(
            api_key="test-key",
            default_model="gpt-4o-mini",
            name="openai",
            base_url="https://api.openai.com/v1",
        )
    # Expose the mock client for per-test configuration
    p._mock_client = mock_client  # type: ignore[attr-defined]
    return p


# ===================================================================
# _convert_tools_to_openai
# ===================================================================


class TestConvertToolsToOpenAI:
    """Tests for the standalone tool-conversion helper."""

    def test_basic_conversion(self) -> None:
        tools = [
            {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            }
        ]
        result = _convert_tools_to_openai(tools)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "read_file"
        assert result[0]["function"]["description"] == "Read a file"
        assert result[0]["function"]["parameters"]["required"] == ["path"]

    def test_missing_description_defaults_empty(self) -> None:
        tools = [{"name": "foo"}]
        result = _convert_tools_to_openai(tools)
        assert result[0]["function"]["description"] == ""

    def test_missing_parameters_defaults_empty_object(self) -> None:
        tools = [{"name": "foo"}]
        result = _convert_tools_to_openai(tools)
        assert result[0]["function"]["parameters"] == {"type": "object", "properties": {}}

    def test_multiple_tools(self) -> None:
        tools = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
        result = _convert_tools_to_openai(tools)
        assert [r["function"]["name"] for r in result] == ["a", "b", "c"]

    def test_empty_list(self) -> None:
        assert _convert_tools_to_openai([]) == []


# ===================================================================
# _parse_tool_calls
# ===================================================================


class TestParseToolCalls:
    """Tests for parsing raw OpenAI tool call objects."""

    def test_valid_tool_call(self) -> None:
        raw = [_make_tool_call("call_1", "read_file", '{"path": "x.py"}')]
        result = _parse_tool_calls(raw)
        assert len(result) == 1
        assert result[0] == ToolCall(id="call_1", name="read_file", arguments={"path": "x.py"})

    def test_invalid_json_arguments(self) -> None:
        raw = [_make_tool_call("call_2", "do_stuff", "not-json")]
        result = _parse_tool_calls(raw)
        assert result[0].arguments == {"_raw": "not-json"}

    def test_empty_arguments(self) -> None:
        raw = [_make_tool_call("call_3", "no_args", "")]
        result = _parse_tool_calls(raw)
        assert result[0].arguments == {}

    def test_none_returns_empty(self) -> None:
        assert _parse_tool_calls(None) == []

    def test_empty_list_returns_empty(self) -> None:
        assert _parse_tool_calls([]) == []

    def test_none_id_defaults_empty(self) -> None:
        raw = [SimpleNamespace(id=None, function=SimpleNamespace(name="f", arguments="{}"))]
        result = _parse_tool_calls(raw)
        assert result[0].id == ""

    def test_none_name_defaults_empty(self) -> None:
        raw = [SimpleNamespace(id="c1", function=SimpleNamespace(name=None, arguments="{}"))]
        result = _parse_tool_calls(raw)
        assert result[0].name == ""


# ===================================================================
# OpenAICompatibleProvider — basics
# ===================================================================


class TestProviderBasics:
    """Tests for basic provider properties and interface compliance."""

    def test_implements_llm_provider(self, provider: OpenAICompatibleProvider) -> None:
        assert isinstance(provider, LLMProvider)

    def test_provider_name(self, provider: OpenAICompatibleProvider) -> None:
        assert provider.provider_name == "openai"

    def test_custom_provider_name(self) -> None:
        with patch("unjess.llm.openai_compat.OpenAI"):
            p = OpenAICompatibleProvider(api_key="k", name="ollama")
        assert p.provider_name == "ollama"


# ===================================================================
# chat()
# ===================================================================


class TestChat:
    """Tests for the non-streaming chat method."""

    def test_chat_returns_llm_response(self, provider: OpenAICompatibleProvider) -> None:
        provider._mock_client.chat.completions.create.return_value = _make_response("Hi there")  # type: ignore[attr-defined]
        resp = provider.chat([{"role": "user", "content": "Hello"}])
        assert isinstance(resp, LLMResponse)
        assert resp.text == "Hi there"
        assert resp.finish_reason == "stop"
        assert resp.model == "gpt-4o-mini"

    def test_chat_uses_default_model(self, provider: OpenAICompatibleProvider) -> None:
        provider._mock_client.chat.completions.create.return_value = _make_response()  # type: ignore[attr-defined]
        provider.chat([{"role": "user", "content": "Hi"}])
        call_kwargs = provider._mock_client.chat.completions.create.call_args  # type: ignore[attr-defined]
        assert call_kwargs.kwargs.get("model") or call_kwargs[1].get("model") == "gpt-4o-mini"

    def test_chat_model_override(self, provider: OpenAICompatibleProvider) -> None:
        provider._mock_client.chat.completions.create.return_value = _make_response()  # type: ignore[attr-defined]
        provider.chat([{"role": "user", "content": "Hi"}], model="gpt-4o")
        call_kwargs = provider._mock_client.chat.completions.create.call_args  # type: ignore[attr-defined]
        # model should be gpt-4o
        assert "gpt-4o" in str(call_kwargs)

    def test_chat_with_tools(self, provider: OpenAICompatibleProvider) -> None:
        provider._mock_client.chat.completions.create.return_value = _make_response()  # type: ignore[attr-defined]
        tools = [{"name": "read_file", "description": "Read", "parameters": {"type": "object", "properties": {}}}]
        provider.chat([{"role": "user", "content": "Hi"}], tools=tools)
        call_kwargs = provider._mock_client.chat.completions.create.call_args  # type: ignore[attr-defined]
        assert "tools" in str(call_kwargs)

    def test_chat_with_tool_calls_in_response(self, provider: OpenAICompatibleProvider) -> None:
        tc = _make_tool_call("call_42", "write_file", '{"path":"a.py","content":"x"}')
        raw = _make_response(content="", tool_calls=[tc], finish_reason="tool_calls")
        provider._mock_client.chat.completions.create.return_value = raw  # type: ignore[attr-defined]
        resp = provider.chat([{"role": "user", "content": "write"}])
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "write_file"
        assert resp.tool_calls[0].id == "call_42"
        assert resp.finish_reason == "tool_calls"

    def test_chat_usage_tokens(self, provider: OpenAICompatibleProvider) -> None:
        raw = _make_response(prompt_tokens=100, completion_tokens=50)
        provider._mock_client.chat.completions.create.return_value = raw  # type: ignore[attr-defined]
        resp = provider.chat([{"role": "user", "content": "Hi"}])
        assert resp.usage.prompt_tokens == 100
        assert resp.usage.completion_tokens == 50
        assert resp.usage.total_tokens == 150

    def test_chat_reasoning_tokens(self, provider: OpenAICompatibleProvider) -> None:
        raw = _make_response(prompt_tokens=50, completion_tokens=200, reasoning_tokens=150)
        provider._mock_client.chat.completions.create.return_value = raw  # type: ignore[attr-defined]
        resp = provider.chat([{"role": "user", "content": "think"}])
        assert resp.usage.thinking_tokens == 150

    def test_chat_empty_choices(self, provider: OpenAICompatibleProvider) -> None:
        raw = _make_response()
        raw.choices = []
        provider._mock_client.chat.completions.create.return_value = raw  # type: ignore[attr-defined]
        resp = provider.chat([{"role": "user", "content": "Hi"}])
        assert resp.text == ""
        assert resp.tool_calls == []
        assert resp.finish_reason == "empty_response"

    def test_chat_none_content_defaults_empty(self, provider: OpenAICompatibleProvider) -> None:
        raw = _make_response()
        raw.choices[0].message.content = None
        provider._mock_client.chat.completions.create.return_value = raw  # type: ignore[attr-defined]
        resp = provider.chat([{"role": "user", "content": "Hi"}])
        assert resp.text == ""

    def test_chat_none_usage_defaults_zero(self, provider: OpenAICompatibleProvider) -> None:
        raw = _make_response()
        raw.usage = None
        provider._mock_client.chat.completions.create.return_value = raw  # type: ignore[attr-defined]
        resp = provider.chat([{"role": "user", "content": "Hi"}])
        assert resp.usage.prompt_tokens == 0
        assert resp.usage.completion_tokens == 0

    def test_chat_ollama_adds_seed(self) -> None:
        with patch("unjess.llm.openai_compat.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            p = OpenAICompatibleProvider(api_key="k", name="ollama")
        mock_client.chat.completions.create.return_value = _make_response()
        p.chat([{"role": "user", "content": "Hi"}])
        call_kwargs = mock_client.chat.completions.create.call_args
        assert "seed" in str(call_kwargs)


# ===================================================================
# chat_stream()
# ===================================================================


class TestChatStream:
    """Tests for the streaming chat method."""

    def test_stream_text_chunks(self, provider: OpenAICompatibleProvider) -> None:
        chunks = [
            _make_stream_chunk(content="Hello"),
            _make_stream_chunk(content=" world"),
            _make_stream_chunk(finish_reason="stop", usage=_make_usage(5, 10)),
        ]
        provider._mock_client.chat.completions.create.return_value = iter(chunks)  # type: ignore[attr-defined]
        results = list(provider.chat_stream([{"role": "user", "content": "Hi"}]))
        texts = [r.text for r in results if r.text]
        assert texts == ["Hello", " world"]
        assert results[-1].done is True
        assert results[-1].usage is not None
        assert results[-1].usage.prompt_tokens == 5

    def test_stream_tool_calls(self, provider: OpenAICompatibleProvider) -> None:
        chunks = [
            _make_stream_chunk(tool_calls=[_make_stream_tc_delta(0, tc_id="c1", name="read_file")]),
            _make_stream_chunk(tool_calls=[_make_stream_tc_delta(0, arguments='{"path":')]),
            _make_stream_chunk(tool_calls=[_make_stream_tc_delta(0, arguments='"x.py"}')]),
            _make_stream_chunk(finish_reason="tool_calls"),
        ]
        provider._mock_client.chat.completions.create.return_value = iter(chunks)  # type: ignore[attr-defined]
        results = list(provider.chat_stream([{"role": "user", "content": "read"}]))
        tool_chunks = [r for r in results if r.tool_call is not None]
        assert len(tool_chunks) == 1
        assert tool_chunks[0].tool_call.name == "read_file"
        assert tool_chunks[0].tool_call.arguments == {"path": "x.py"}

    def test_stream_multiple_tool_calls(self, provider: OpenAICompatibleProvider) -> None:
        chunks = [
            _make_stream_chunk(tool_calls=[_make_stream_tc_delta(0, tc_id="c1", name="foo")]),
            _make_stream_chunk(tool_calls=[_make_stream_tc_delta(0, arguments='{}')]),
            _make_stream_chunk(tool_calls=[_make_stream_tc_delta(1, tc_id="c2", name="bar")]),
            _make_stream_chunk(tool_calls=[_make_stream_tc_delta(1, arguments='{"x":1}')]),
            _make_stream_chunk(finish_reason="tool_calls"),
        ]
        provider._mock_client.chat.completions.create.return_value = iter(chunks)  # type: ignore[attr-defined]
        results = list(provider.chat_stream([{"role": "user", "content": "multi"}]))
        tool_chunks = [r for r in results if r.tool_call is not None]
        assert len(tool_chunks) == 2
        assert tool_chunks[0].tool_call.name == "foo"
        assert tool_chunks[1].tool_call.name == "bar"
        assert tool_chunks[1].tool_call.arguments == {"x": 1}

    def test_stream_invalid_tool_json_produces_raw(self, provider: OpenAICompatibleProvider) -> None:
        chunks = [
            _make_stream_chunk(tool_calls=[_make_stream_tc_delta(0, tc_id="c1", name="f")]),
            _make_stream_chunk(tool_calls=[_make_stream_tc_delta(0, arguments="broken{json")]),
            _make_stream_chunk(finish_reason="tool_calls"),
        ]
        provider._mock_client.chat.completions.create.return_value = iter(chunks)  # type: ignore[attr-defined]
        results = list(provider.chat_stream([{"role": "user", "content": "bad"}]))
        tool_chunks = [r for r in results if r.tool_call is not None]
        assert tool_chunks[0].tool_call.arguments == {"_raw": "broken{json"}

    def test_stream_no_choices_skipped(self, provider: OpenAICompatibleProvider) -> None:
        empty_chunk = SimpleNamespace(choices=[], usage=None)
        done_chunk = _make_stream_chunk(finish_reason="stop")
        provider._mock_client.chat.completions.create.return_value = iter([empty_chunk, done_chunk])  # type: ignore[attr-defined]
        results = list(provider.chat_stream([{"role": "user", "content": "Hi"}]))
        assert results[-1].done is True

    def test_stream_usage_with_reasoning_tokens(self, provider: OpenAICompatibleProvider) -> None:
        usage = _make_usage(10, 30, reasoning=20)
        chunks = [
            _make_stream_chunk(content="x", finish_reason="stop", usage=usage),
        ]
        provider._mock_client.chat.completions.create.return_value = iter(chunks)  # type: ignore[attr-defined]
        results = list(provider.chat_stream([{"role": "user", "content": "Hi"}]))
        assert results[-1].usage is not None
        assert results[-1].usage.thinking_tokens == 20

    def test_stream_fallback_removes_stream_options(self, provider: OpenAICompatibleProvider) -> None:
        def side_effect(**kwargs: Any) -> Any:
            if "stream_options" in kwargs:
                raise Exception("stream_options is not supported")
            return iter([_make_stream_chunk(content="ok", finish_reason="stop")])

        provider._mock_client.chat.completions.create.side_effect = side_effect  # type: ignore[attr-defined]
        results = list(provider.chat_stream([{"role": "user", "content": "Hi"}]))
        texts = [r.text for r in results if r.text]
        assert texts == ["ok"]

    def test_stream_non_stream_options_error_reraises(self, provider: OpenAICompatibleProvider) -> None:
        provider._mock_client.chat.completions.create.side_effect = ValueError("something else broke")  # type: ignore[attr-defined]
        with pytest.raises(ValueError, match="something else broke"):
            list(provider.chat_stream([{"role": "user", "content": "Hi"}]))


# ===================================================================
# list_models()
# ===================================================================


class TestListModels:
    """Tests for list_models with provider-specific filtering."""

    def test_list_models_basic(self, provider: OpenAICompatibleProvider) -> None:
        provider._mock_client.models.list.return_value = [  # type: ignore[attr-defined]
            SimpleNamespace(id="gpt-4o"),
            SimpleNamespace(id="gpt-4o-mini"),
            SimpleNamespace(id="gpt-3.5-turbo"),
        ]
        result = provider.list_models()
        assert result == ["gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini"]

    def test_list_models_openrouter_free_only(self) -> None:
        with patch("unjess.llm.openai_compat.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            p = OpenAICompatibleProvider(api_key="k", name="openrouter")
        mock_client.models.list.return_value = [
            SimpleNamespace(id="meta/llama:free"),
            SimpleNamespace(id="openrouter/auto"),
            SimpleNamespace(id="anthropic/claude-3"),
        ]
        result = p.list_models()
        assert "anthropic/claude-3" not in result
        assert "meta/llama:free" in result
        assert "openrouter/auto" in result

    def test_list_models_google_filters_non_gemini(self) -> None:
        with patch("unjess.llm.openai_compat.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            p = OpenAICompatibleProvider(api_key="k", name="google")
        mock_client.models.list.return_value = [
            SimpleNamespace(id="gemini-3.1-flash"),
            SimpleNamespace(id="gemini-3.1-pro"),
            SimpleNamespace(id="gemini-tts-model"),
            SimpleNamespace(id="gemma-3n-e4b"),
            SimpleNamespace(id="lyria-something"),
        ]
        result = p.list_models()
        assert "gemini-3.1-flash" in result
        assert "gemini-3.1-pro" in result
        assert "gemini-tts-model" not in result
        assert "gemma-3n-e4b" not in result
        assert "lyria-something" not in result

    def test_list_models_groq_filters_whisper(self) -> None:
        with patch("unjess.llm.openai_compat.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            p = OpenAICompatibleProvider(api_key="k", name="groq")
        mock_client.models.list.return_value = [
            SimpleNamespace(id="llama-3.3-70b"),
            SimpleNamespace(id="whisper-large-v3"),
            SimpleNamespace(id="orpheus-tts"),
        ]
        result = p.list_models()
        assert "llama-3.3-70b" in result
        assert "whisper-large-v3" not in result
        assert "orpheus-tts" not in result

    def test_list_models_mistral_filters_and_deduplicates(self) -> None:
        with patch("unjess.llm.openai_compat.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            p = OpenAICompatibleProvider(api_key="k", name="mistral")
        mock_client.models.list.return_value = [
            SimpleNamespace(id="mistral-large-latest"),
            SimpleNamespace(id="mistral-large-latest"),  # duplicate
            SimpleNamespace(id="mistral-ocr-latest"),
            SimpleNamespace(id="mistral-embed-latest"),
            SimpleNamespace(id="codestral-latest"),
        ]
        result = p.list_models()
        assert result.count("mistral-large-latest") == 1
        assert "mistral-ocr-latest" not in result
        assert "mistral-embed-latest" not in result
        assert "codestral-latest" in result

    def test_list_models_exception_returns_empty(self, provider: OpenAICompatibleProvider) -> None:
        provider._mock_client.models.list.side_effect = Exception("Network error")  # type: ignore[attr-defined]
        assert provider.list_models() == []


# ===================================================================
# Error handling
# ===================================================================


class TestErrorHandling:
    """Tests for error propagation through the provider."""

    def test_api_error_propagates(self, provider: OpenAICompatibleProvider) -> None:
        provider._mock_client.chat.completions.create.side_effect = RuntimeError("API error")  # type: ignore[attr-defined]
        with pytest.raises(RuntimeError, match="API error"):
            provider.chat([{"role": "user", "content": "Hi"}])

    def test_none_model_in_response_falls_back(self, provider: OpenAICompatibleProvider) -> None:
        raw = _make_response()
        raw.model = None
        provider._mock_client.chat.completions.create.return_value = raw  # type: ignore[attr-defined]
        resp = provider.chat([{"role": "user", "content": "Hi"}])
        assert resp.model == "gpt-4o-mini"

    def test_none_finish_reason_defaults_empty(self, provider: OpenAICompatibleProvider) -> None:
        raw = _make_response()
        raw.choices[0].finish_reason = None
        provider._mock_client.chat.completions.create.return_value = raw  # type: ignore[attr-defined]
        resp = provider.chat([{"role": "user", "content": "Hi"}])
        assert resp.finish_reason == ""

    def test_reasoning_effort_auto_set_for_gpt5_luna_with_tools(self, provider: OpenAICompatibleProvider) -> None:
        tools = [{"name": "read_file", "description": "read", "parameters": {}}]
        raw = _make_response()
        provider._mock_client.chat.completions.create.return_value = raw  # type: ignore[attr-defined]
        provider.chat([{"role": "user", "content": "Hi"}], tools=tools, model="gpt-5.6-luna")
        call_kwargs = provider._mock_client.chat.completions.create.call_args[1]  # type: ignore[attr-defined]
        assert call_kwargs.get("reasoning_effort") == "none"

    def test_reasoning_effort_error_retries_with_none(self, provider: OpenAICompatibleProvider) -> None:
        tools = [{"name": "read_file", "description": "read", "parameters": {}}]
        raw = _make_response()

        def side_effect(**kwargs: Any) -> Any:
            if kwargs.get("reasoning_effort") != "none":
                raise Exception("Function tools with reasoning_effort are not supported. Set reasoning_effort to 'none'.")
            return raw

        provider._mock_client.chat.completions.create.side_effect = side_effect  # type: ignore[attr-defined]
        resp = provider.chat([{"role": "user", "content": "Hi"}], tools=tools, model="custom-reasoning-model")
        assert resp.text == "Hello"

    def test_reasoning_effort_unsupported_param_removes_param(self, provider: OpenAICompatibleProvider) -> None:
        tools = [{"name": "read_file", "description": "read", "parameters": {}}]
        raw = _make_response()

        def side_effect(**kwargs: Any) -> Any:
            if "reasoning_effort" in kwargs:
                raise Exception("Unrecognized request argument: reasoning_effort")
            return raw

        provider._mock_client.chat.completions.create.side_effect = side_effect  # type: ignore[attr-defined]
        resp = provider.chat([{"role": "user", "content": "Hi"}], tools=tools, model="gpt-5.6-luna")
        assert resp.text == "Hello"
