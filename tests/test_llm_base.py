"""Tests for unjess.llm.base — LLM provider interface and data classes."""

from typing import Any, Generator
from unittest.mock import MagicMock

import pytest

from unjess.llm.base import (
    LLMProvider,
    LLMResponse,
    StreamChunk,
    ToolCall,
    Usage,
)


class TestToolCall:
    """ToolCall dataclass construction and field access."""

    def test_construction(self) -> None:
        tc = ToolCall(id="call_1", name="read_file", arguments={"path": "/tmp/x"})
        assert tc.id == "call_1"
        assert tc.name == "read_file"
        assert tc.arguments == {"path": "/tmp/x"}

    def test_empty_arguments(self) -> None:
        tc = ToolCall(id="c", name="list_files", arguments={})
        assert tc.arguments == {}

    def test_complex_arguments(self) -> None:
        args: dict[str, Any] = {"nested": {"a": [1, 2]}, "flag": True}
        tc = ToolCall(id="c2", name="complex_tool", arguments=args)
        assert tc.arguments["nested"]["a"] == [1, 2]
        assert tc.arguments["flag"] is True

    def test_equality(self) -> None:
        a = ToolCall(id="x", name="t", arguments={"k": "v"})
        b = ToolCall(id="x", name="t", arguments={"k": "v"})
        assert a == b

    def test_inequality(self) -> None:
        a = ToolCall(id="1", name="t", arguments={})
        b = ToolCall(id="2", name="t", arguments={})
        assert a != b


class TestUsage:
    """Usage dataclass and total_tokens property."""

    def test_defaults_all_zero(self) -> None:
        u = Usage()
        assert u.prompt_tokens == 0
        assert u.completion_tokens == 0
        assert u.thinking_tokens == 0
        assert u.cache_read_tokens == 0
        assert u.cache_creation_tokens == 0

    def test_total_tokens_prompt_plus_completion(self) -> None:
        u = Usage(prompt_tokens=100, completion_tokens=50)
        assert u.total_tokens == 150

    def test_thinking_tokens_not_additive(self) -> None:
        u = Usage(prompt_tokens=100, completion_tokens=50, thinking_tokens=30)
        # thinking_tokens is a breakdown of completion_tokens, NOT added
        assert u.total_tokens == 150

    def test_cache_tokens_not_in_total(self) -> None:
        u = Usage(
            prompt_tokens=10,
            completion_tokens=5,
            cache_read_tokens=20,
            cache_creation_tokens=15,
        )
        assert u.total_tokens == 15  # only prompt + completion

    def test_total_tokens_zero(self) -> None:
        assert Usage().total_tokens == 0

    def test_large_values(self) -> None:
        u = Usage(prompt_tokens=1_000_000, completion_tokens=500_000)
        assert u.total_tokens == 1_500_000

    def test_all_fields_set(self) -> None:
        u = Usage(
            prompt_tokens=10,
            completion_tokens=20,
            thinking_tokens=5,
            cache_read_tokens=3,
            cache_creation_tokens=2,
        )
        assert u.prompt_tokens == 10
        assert u.completion_tokens == 20
        assert u.thinking_tokens == 5
        assert u.cache_read_tokens == 3
        assert u.cache_creation_tokens == 2
        assert u.total_tokens == 30


class TestLLMResponse:
    """LLMResponse dataclass construction and defaults."""

    def test_defaults(self) -> None:
        r = LLMResponse()
        assert r.text == ""
        assert r.thinking_text == ""
        assert r.tool_calls == []
        assert isinstance(r.usage, Usage)
        assert r.usage.total_tokens == 0
        assert r.finish_reason == ""
        assert r.model == ""

    def test_text_response(self) -> None:
        r = LLMResponse(text="Hello!", model="gpt-4", finish_reason="stop")
        assert r.text == "Hello!"
        assert r.model == "gpt-4"
        assert r.finish_reason == "stop"

    def test_with_tool_calls(self) -> None:
        tc = ToolCall(id="c1", name="search", arguments={"q": "test"})
        r = LLMResponse(tool_calls=[tc], finish_reason="tool_calls")
        assert len(r.tool_calls) == 1
        assert r.tool_calls[0].name == "search"

    def test_multiple_tool_calls(self) -> None:
        calls = [
            ToolCall(id="1", name="read", arguments={}),
            ToolCall(id="2", name="write", arguments={"text": "x"}),
        ]
        r = LLMResponse(tool_calls=calls)
        assert len(r.tool_calls) == 2

    def test_with_thinking(self) -> None:
        r = LLMResponse(
            text="answer",
            thinking_text="let me think...",
            usage=Usage(prompt_tokens=50, completion_tokens=30, thinking_tokens=10),
        )
        assert r.thinking_text == "let me think..."
        assert r.usage.thinking_tokens == 10

    def test_tool_calls_list_is_independent(self) -> None:
        r1 = LLMResponse()
        r2 = LLMResponse()
        r1.tool_calls.append(ToolCall(id="x", name="t", arguments={}))
        assert len(r2.tool_calls) == 0  # field(default_factory) ensures independence

    def test_usage_is_independent(self) -> None:
        r1 = LLMResponse()
        r2 = LLMResponse()
        r1.usage.prompt_tokens = 999
        assert r2.usage.prompt_tokens == 0


class TestStreamChunk:
    """StreamChunk dataclass construction and defaults."""

    def test_defaults(self) -> None:
        c = StreamChunk()
        assert c.text == ""
        assert c.thinking == ""
        assert c.tool_call is None
        assert c.done is False
        assert c.usage is None

    def test_text_chunk(self) -> None:
        c = StreamChunk(text="hello ")
        assert c.text == "hello "
        assert c.done is False

    def test_thinking_chunk(self) -> None:
        c = StreamChunk(thinking="reasoning...")
        assert c.thinking == "reasoning..."

    def test_tool_call_chunk(self) -> None:
        tc = ToolCall(id="c1", name="run", arguments={"cmd": "ls"})
        c = StreamChunk(tool_call=tc)
        assert c.tool_call is not None
        assert c.tool_call.name == "run"

    def test_done_chunk_with_usage(self) -> None:
        u = Usage(prompt_tokens=10, completion_tokens=5)
        c = StreamChunk(done=True, usage=u)
        assert c.done is True
        assert c.usage is not None
        assert c.usage.total_tokens == 15

    def test_done_chunk_without_usage(self) -> None:
        c = StreamChunk(done=True)
        assert c.done is True
        assert c.usage is None


class TestLLMProvider:
    """LLMProvider ABC enforcement and default implementations."""

    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            LLMProvider()  # type: ignore[abstract]

    def test_must_implement_provider_name(self) -> None:
        class Missing(LLMProvider):
            def chat(self, messages, tools=None, model=""):
                return LLMResponse()

            def chat_stream(self, messages, tools=None, model=""):
                yield StreamChunk(done=True)

        with pytest.raises(TypeError):
            Missing()  # type: ignore[abstract]

    def test_must_implement_chat(self) -> None:
        class Missing(LLMProvider):
            @property
            def provider_name(self):
                return "test"

            def chat_stream(self, messages, tools=None, model=""):
                yield StreamChunk(done=True)

        with pytest.raises(TypeError):
            Missing()  # type: ignore[abstract]

    def test_must_implement_chat_stream(self) -> None:
        class Missing(LLMProvider):
            @property
            def provider_name(self):
                return "test"

            def chat(self, messages, tools=None, model=""):
                return LLMResponse()

        with pytest.raises(TypeError):
            Missing()  # type: ignore[abstract]

    def test_concrete_subclass_works(self) -> None:
        class FakeProvider(LLMProvider):
            @property
            def provider_name(self) -> str:
                return "fake"

            def chat(
                self,
                messages: list[dict[str, Any]],
                tools: list[dict[str, Any]] | None = None,
                model: str = "",
            ) -> LLMResponse:
                return LLMResponse(text="ok", model=model or "fake-1")

            def chat_stream(
                self,
                messages: list[dict[str, Any]],
                tools: list[dict[str, Any]] | None = None,
                model: str = "",
            ) -> Generator[StreamChunk, None, None]:
                yield StreamChunk(text="ok")
                yield StreamChunk(done=True)

        provider = FakeProvider()
        assert provider.provider_name == "fake"

    def test_default_list_models_returns_empty(self) -> None:
        class MinimalProvider(LLMProvider):
            @property
            def provider_name(self) -> str:
                return "minimal"

            def chat(self, messages, tools=None, model=""):
                return LLMResponse()

            def chat_stream(self, messages, tools=None, model=""):
                yield StreamChunk(done=True)

        p = MinimalProvider()
        assert p.list_models() == []

    def test_list_models_can_be_overridden(self) -> None:
        class RichProvider(LLMProvider):
            @property
            def provider_name(self) -> str:
                return "rich"

            def chat(self, messages, tools=None, model=""):
                return LLMResponse()

            def chat_stream(self, messages, tools=None, model=""):
                yield StreamChunk(done=True)

            def list_models(self) -> list[str]:
                return ["model-a", "model-b"]

        p = RichProvider()
        assert p.list_models() == ["model-a", "model-b"]

    def test_chat_returns_llm_response(self) -> None:
        class SimpleProvider(LLMProvider):
            @property
            def provider_name(self) -> str:
                return "simple"

            def chat(self, messages, tools=None, model=""):
                return LLMResponse(
                    text="response",
                    usage=Usage(prompt_tokens=5, completion_tokens=3),
                    finish_reason="stop",
                    model="simple-1",
                )

            def chat_stream(self, messages, tools=None, model=""):
                yield StreamChunk(text="response", done=True)

        p = SimpleProvider()
        resp = p.chat([{"role": "user", "content": "hi"}])
        assert isinstance(resp, LLMResponse)
        assert resp.text == "response"
        assert resp.usage.total_tokens == 8
        assert resp.finish_reason == "stop"

    def test_chat_stream_yields_chunks(self) -> None:
        class StreamProvider(LLMProvider):
            @property
            def provider_name(self) -> str:
                return "stream"

            def chat(self, messages, tools=None, model=""):
                return LLMResponse(text="full")

            def chat_stream(self, messages, tools=None, model=""):
                yield StreamChunk(text="hel")
                yield StreamChunk(text="lo")
                yield StreamChunk(done=True, usage=Usage(prompt_tokens=1, completion_tokens=2))

        p = StreamProvider()
        chunks = list(p.chat_stream([{"role": "user", "content": "hi"}]))
        assert len(chunks) == 3
        assert chunks[0].text == "hel"
        assert chunks[1].text == "lo"
        assert chunks[2].done is True
        assert chunks[2].usage is not None
        assert chunks[2].usage.total_tokens == 3

    def test_chat_with_tools_parameter(self) -> None:
        class ToolProvider(LLMProvider):
            @property
            def provider_name(self) -> str:
                return "tool"

            def chat(self, messages, tools=None, model=""):
                if tools:
                    return LLMResponse(
                        tool_calls=[ToolCall(id="c1", name=tools[0]["name"], arguments={})],
                        finish_reason="tool_calls",
                    )
                return LLMResponse(text="no tools")

            def chat_stream(self, messages, tools=None, model=""):
                yield StreamChunk(done=True)

        p = ToolProvider()
        tool_def = {"name": "search", "parameters": {}}
        resp = p.chat([{"role": "user", "content": "search"}], tools=[tool_def])
        assert resp.finish_reason == "tool_calls"
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "search"
