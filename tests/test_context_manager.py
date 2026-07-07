"""Tests for unjess.context_manager — token counting, budget tracking, truncation, compaction."""

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from unjess.context_manager import (
    ContextManager,
    ContextSection,
    SECTION_PRIORITIES,
    _CONTEXT_WINDOWS,
    _RESPONSE_RESERVE,
    assemble_sections,
    count_messages_tokens,
    count_tokens,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _msg(role: str, content: str, **kwargs: Any) -> dict[str, Any]:
    """Build a message dict."""
    m: dict[str, Any] = {"role": role, "content": content}
    m.update(kwargs)
    return m


def _make_messages(n: int, chars_per: int = 40) -> list[dict[str, Any]]:
    """Create *n* user/assistant message pairs (total 2*n messages)."""
    msgs: list[dict[str, Any]] = []
    for i in range(n):
        msgs.append(_msg("user", "x" * chars_per))
        msgs.append(_msg("assistant", "y" * chars_per))
    return msgs


# ===========================================================================
# count_tokens
# ===========================================================================

class TestCountTokens:
    """Token counting with tiktoken and heuristic fallback."""

    def test_heuristic_fallback_basic(self) -> None:
        # Non-OpenAI model → chars / 4
        text = "a" * 100
        assert count_tokens(text, model="gemini-2.5-flash") == 25

    def test_heuristic_minimum_one(self) -> None:
        assert count_tokens("ab", model="gemini-2.5-flash") == 1

    def test_heuristic_empty_string(self) -> None:
        # len("") // 4 == 0, max(1, 0) => 1
        assert count_tokens("", model="gemini-2.5-flash") == 1

    def test_heuristic_no_model(self) -> None:
        assert count_tokens("a" * 80) == 20

    def test_tiktoken_openai_model(self) -> None:
        # gpt-4o should try tiktoken
        result = count_tokens("Hello world", model="gpt-4o")
        assert isinstance(result, int)
        assert result >= 1

    def test_tiktoken_o3_model(self) -> None:
        result = count_tokens("Hello world", model="o3")
        assert isinstance(result, int)
        assert result >= 1

    def test_tiktoken_o4_model(self) -> None:
        result = count_tokens("Hello world", model="o4-mini")
        assert isinstance(result, int)
        assert result >= 1

    def test_tiktoken_import_error_fallback(self) -> None:
        with patch.dict("sys.modules", {"tiktoken": None}):
            result = count_tokens("a" * 100, model="gpt-4o")
            # Should fall through to heuristic
            assert result == 25

    def test_tiktoken_encoding_key_error(self) -> None:
        """When tiktoken can't find model encoding, falls back to cl100k_base."""
        result = count_tokens("Hello!", model="gpt-99-turbo")
        assert isinstance(result, int)
        assert result >= 1


# ===========================================================================
# count_messages_tokens
# ===========================================================================

class TestCountMessagesTokens:
    """Token counting across a list of messages."""

    def test_empty_messages(self) -> None:
        assert count_messages_tokens([], model="gemini-2.5-flash") == 0

    def test_single_message(self) -> None:
        msgs = [_msg("user", "a" * 80)]
        # content tokens: 80//4=20, overhead: 4 => 24
        assert count_messages_tokens(msgs, model="gemini-2.5-flash") == 24

    def test_multiple_messages(self) -> None:
        msgs = [_msg("user", "a" * 80), _msg("assistant", "b" * 40)]
        # user: 20 + 4 = 24; assistant: 10 + 4 = 14 => 38
        assert count_messages_tokens(msgs, model="gemini-2.5-flash") == 38

    def test_message_with_tool_calls(self) -> None:
        tc = {"id": "call_1", "function": {"name": "read_file", "arguments": "{}"}}
        msgs = [_msg("assistant", "", tool_calls=[tc])]
        result = count_messages_tokens(msgs, model="gemini-2.5-flash")
        expected_content = 1  # empty string => max(1, 0)
        expected_tc = max(1, len(json.dumps(tc)) // 4)
        expected = expected_content + expected_tc + 4
        assert result == expected

    def test_missing_content_key(self) -> None:
        msgs = [{"role": "user"}]
        # content defaults to "" → 1 token + 4 overhead
        assert count_messages_tokens(msgs, model="gemini-2.5-flash") == 5

    def test_non_string_content_ignored(self) -> None:
        # list content (multimodal) → not isinstance str → skipped, 0 content tokens
        msgs = [{"role": "user", "content": [{"type": "image"}]}]
        result = count_messages_tokens(msgs, model="gemini-2.5-flash")
        assert result == 4  # only overhead


# ===========================================================================
# ContextManager — model property
# ===========================================================================

class TestContextManagerModel:
    """Model property get/set."""

    def test_model_default(self) -> None:
        cm = ContextManager()
        assert cm.model == ""

    def test_model_constructor(self) -> None:
        cm = ContextManager(model="gpt-4o")
        assert cm.model == "gpt-4o"

    def test_model_setter(self) -> None:
        cm = ContextManager()
        cm.model = "claude-sonnet-4"
        assert cm.model == "claude-sonnet-4"


# ===========================================================================
# ContextManager — get_context_window
# ===========================================================================

class TestGetContextWindow:
    """Context window lookup: exact, prefix, defaults."""

    def test_exact_match(self) -> None:
        cm = ContextManager(model="gpt-4o")
        assert cm.get_context_window() == 128_000

    def test_exact_match_gemini(self) -> None:
        cm = ContextManager(model="gemini-2.5-flash")
        assert cm.get_context_window() == 1_000_000

    def test_exact_match_mixtral(self) -> None:
        cm = ContextManager(model="mixtral-8x7b")
        assert cm.get_context_window() == 32_768

    def test_prefix_match(self) -> None:
        # "gpt-4o-2024-01-01" starts with "gpt-4o"
        cm = ContextManager(model="gpt-4o-2024-01-01")
        # Note: "gpt-4o-2024-01-01" doesn't match "gpt-4o" exactly,
        # but prefix check: "gpt-4o" is in _CONTEXT_WINDOWS and model starts with it
        assert cm.get_context_window() == 128_000

    def test_prefix_match_claude(self) -> None:
        cm = ContextManager(model="claude-sonnet-4-20250514")
        assert cm.get_context_window() == 200_000

    def test_default_ollama_model(self) -> None:
        cm = ContextManager(model="my-custom-model:latest")
        assert cm.get_context_window() == 32_768

    def test_default_unknown_model(self) -> None:
        cm = ContextManager(model="totally-unknown-model")
        assert cm.get_context_window() == 128_000

    def test_all_known_models_have_positive_window(self) -> None:
        for model_name, window in _CONTEXT_WINDOWS.items():
            assert window > 0, f"Model {model_name} has non-positive window"

    def test_gemma_small_window(self) -> None:
        cm = ContextManager(model="gemma")
        assert cm.get_context_window() == 8_192


# ===========================================================================
# ContextManager — get_available_budget
# ===========================================================================

class TestGetAvailableBudget:
    """Budget = window - reserve - system - messages."""

    def test_empty_conversation(self) -> None:
        cm = ContextManager(model="mixtral-8x7b")  # 32768
        budget = cm.get_available_budget("system", [])
        window = 32_768
        reserve = int(window * _RESPONSE_RESERVE)
        sys_tokens = count_tokens("system", "mixtral-8x7b")
        assert budget == window - reserve - sys_tokens

    def test_with_messages(self) -> None:
        cm = ContextManager(model="mixtral-8x7b")
        msgs = [_msg("user", "hello")]
        budget = cm.get_available_budget("system prompt", msgs)
        assert budget > 0

    def test_budget_floors_at_zero(self) -> None:
        cm = ContextManager(model="gemma")  # 8192
        # Enormous system prompt should exhaust budget
        huge = "x" * 100_000
        budget = cm.get_available_budget(huge, [])
        assert budget == 0

    def test_budget_decreases_with_more_messages(self) -> None:
        cm = ContextManager(model="mixtral-8x7b")
        sys_prompt = "You are an assistant."
        budget_0 = cm.get_available_budget(sys_prompt, [])
        budget_1 = cm.get_available_budget(sys_prompt, [_msg("user", "hi")])
        assert budget_0 > budget_1


# ===========================================================================
# ContextManager — truncate_conversation
# ===========================================================================

class TestTruncateConversation:
    """Truncation preserves first message + most recent, drops middle."""

    def test_no_truncation_needed(self) -> None:
        cm = ContextManager(model="gemini-2.5-flash")  # 1M window
        msgs = [_msg("user", "hello"), _msg("assistant", "hi")]
        result = cm.truncate_conversation(msgs, "system")
        assert result is msgs  # exact same list when no truncation

    def test_truncation_preserves_first_and_last(self) -> None:
        cm = ContextManager(model="gemma")  # 8192 tokens
        # Fill a lot of messages to exceed budget
        msgs = [_msg("user", "first task")] + _make_messages(50, chars_per=200)
        result = cm.truncate_conversation(msgs, "system prompt")
        assert result[0]["content"] == "first task"
        assert result[-1] == msgs[-1]

    def test_truncation_inserts_notice(self) -> None:
        cm = ContextManager(model="gemma")  # 8192
        msgs = [_msg("user", "first")] + _make_messages(50, chars_per=200)
        result = cm.truncate_conversation(msgs, "system prompt")
        if len(result) < len(msgs):
            assert "[" in result[1]["content"] and "truncated" in result[1]["content"]

    def test_two_messages_never_truncated(self) -> None:
        cm = ContextManager(model="gemma")
        msgs = [_msg("user", "x" * 10000), _msg("assistant", "y" * 10000)]
        result = cm.truncate_conversation(msgs, "x" * 5000)
        assert len(result) == 2

    def test_budget_zero_returns_last_two(self) -> None:
        cm = ContextManager(model="gemma")
        # System prompt so huge the budget is <= 0
        msgs = [_msg("user", "a"), _msg("assistant", "b"), _msg("user", "c")]
        result = cm.truncate_conversation(msgs, "x" * 100_000)
        assert len(result) == 2

    def test_tool_call_group_removed_atomically(self) -> None:
        cm = ContextManager(model="gemma")  # small window
        tool_calls = [{"id": "tc_1", "function": {"name": "f", "arguments": "{}"}}]
        msgs = [
            _msg("user", "first"),
            _msg("assistant", "", tool_calls=tool_calls),
            _msg("tool", "result text", tool_call_id="tc_1"),
            _msg("user", "second " * 200),
            _msg("assistant", "done " * 200),
        ]
        # Make sure the messages are big enough that truncation occurs on gemma
        big_msgs = [msgs[0]]
        for _ in range(20):
            big_msgs.extend(msgs[1:])
        result = cm.truncate_conversation(big_msgs, "system prompt")
        # Verify no orphaned tool results without their assistant
        for i, m in enumerate(result):
            if m.get("role") == "tool":
                # There must be a preceding assistant with matching tool_calls
                assert any(
                    r.get("role") == "assistant" and r.get("tool_calls")
                    for r in result[:i]
                )


# ===========================================================================
# ContextManager — compact
# ===========================================================================

class TestCompact:
    """Conversation compaction with optional LLM summarization."""

    def test_too_few_messages_unchanged(self) -> None:
        cm = ContextManager(model="gpt-4o")
        msgs = [_msg("user", "hi"), _msg("assistant", "hello")]
        result = cm.compact(msgs, keep_recent=4)
        assert result is msgs

    def test_compact_without_summarize_fn(self) -> None:
        cm = ContextManager(model="gemini-2.5-flash")
        msgs = _make_messages(10, chars_per=40)
        result = cm.compact(msgs, keep_recent=4)
        assert len(result) < len(msgs)
        # First message should be a summary
        assert "[Conversation summary" in result[0]["content"]

    def test_compact_with_summarize_fn(self) -> None:
        cm = ContextManager(model="gemini-2.5-flash")
        msgs = _make_messages(10, chars_per=40)
        summary_fn = MagicMock(return_value="Summary of old stuff")
        result = cm.compact(msgs, summarize_fn=summary_fn, keep_recent=4)
        summary_fn.assert_called_once()
        assert "Summary of old stuff" in result[0]["content"]

    def test_compact_summarize_fn_failure_falls_back(self) -> None:
        cm = ContextManager(model="gemini-2.5-flash")
        msgs = _make_messages(10, chars_per=40)
        summary_fn = MagicMock(side_effect=RuntimeError("LLM down"))
        result = cm.compact(msgs, summarize_fn=summary_fn, keep_recent=4)
        assert "[Conversation summary" in result[0]["content"]
        # Fallback: truncated text + "..."
        assert "..." in result[0]["content"]

    def test_compact_preserves_recent_messages(self) -> None:
        cm = ContextManager(model="gemini-2.5-flash")
        msgs = _make_messages(10, chars_per=40)
        keep = 4
        result = cm.compact(msgs, keep_recent=keep)
        # The last `keep` messages (approximately) should be preserved
        for orig, compacted in zip(msgs[-keep:], result[-keep:]):
            assert orig["content"] == compacted["content"]

    def test_compact_keeps_recent_exact(self) -> None:
        cm = ContextManager(model="gemini-2.5-flash")
        # 6 user messages: all user role (no tool_calls confusion)
        msgs = [_msg("user", f"msg{i}") for i in range(6)]
        result = cm.compact(msgs, keep_recent=4)
        # summary + 4 recent
        assert len(result) == 5
        assert result[-1]["content"] == "msg5"
        assert result[-4]["content"] == "msg2"

    def test_compact_avoids_breaking_tool_call_group(self) -> None:
        cm = ContextManager(model="gemini-2.5-flash")
        tc = [{"id": "tc_1", "function": {"name": "f", "arguments": "{}"}}]
        msgs = [
            _msg("user", "do something"),
            _msg("assistant", "ok"),
            _msg("user", "another thing"),
            _msg("assistant", "", tool_calls=tc),
            _msg("tool", "result", tool_call_id="tc_1"),
            _msg("user", "follow up"),
            _msg("assistant", "done"),
        ]
        result = cm.compact(msgs, keep_recent=3)
        # Should not split tool_calls from their results
        for i, m in enumerate(result):
            if m.get("role") == "assistant" and m.get("tool_calls"):
                # Next message should be the tool result
                assert i + 1 < len(result)
                assert result[i + 1].get("role") == "tool"

    def test_compact_split_too_early_returns_original(self) -> None:
        cm = ContextManager(model="gemini-2.5-flash")
        # Only tool-call groups — can't find a safe split
        tc = [{"id": "tc_1", "function": {"name": "f", "arguments": "{}"}}]
        msgs = [
            _msg("assistant", "", tool_calls=tc),
            _msg("tool", "r1", tool_call_id="tc_1"),
            _msg("assistant", "", tool_calls=tc),
            _msg("tool", "r2", tool_call_id="tc_1"),
        ]
        result = cm.compact(msgs, keep_recent=2)
        # split walks back to <= 1, returns original
        assert result is msgs


# ===========================================================================
# ContextManager — get_context_breakdown
# ===========================================================================

class TestGetContextBreakdown:
    """Context breakdown returns correct token accounting."""

    def test_breakdown_keys(self) -> None:
        cm = ContextManager(model="gemini-2.5-flash")
        breakdown = cm.get_context_breakdown("sys", [_msg("user", "hi")])
        expected_keys = {
            "context_window", "response_reserve", "system_prompt_tokens",
            "conversation_tokens", "total_used", "available",
            "utilization_pct", "message_count",
        }
        assert set(breakdown.keys()) == expected_keys

    def test_breakdown_message_count(self) -> None:
        cm = ContextManager(model="gemini-2.5-flash")
        msgs = _make_messages(3)
        breakdown = cm.get_context_breakdown("sys", msgs)
        assert breakdown["message_count"] == 6

    def test_breakdown_total_used(self) -> None:
        cm = ContextManager(model="gemini-2.5-flash")
        breakdown = cm.get_context_breakdown("system", [_msg("user", "hello")])
        assert breakdown["total_used"] == breakdown["system_prompt_tokens"] + breakdown["conversation_tokens"]

    def test_breakdown_utilization_pct(self) -> None:
        cm = ContextManager(model="gemini-2.5-flash")
        breakdown = cm.get_context_breakdown("sys", [])
        assert 0 <= breakdown["utilization_pct"] <= 100


# ===========================================================================
# ContextManager — needs_compaction
# ===========================================================================

class TestNeedsCompaction:
    """Compaction trigger based on utilization or hard token cap."""

    def test_below_threshold(self) -> None:
        cm = ContextManager(model="gemini-2.5-flash")
        assert cm.needs_compaction("sys", []) is False

    def test_above_threshold(self) -> None:
        cm = ContextManager(model="gemma", auto_compact_threshold=0.10)
        # gemma = 8192; fill enough to exceed 10%
        msgs = _make_messages(10, chars_per=400)
        assert cm.needs_compaction("system", msgs) is True

    def test_custom_threshold_override(self) -> None:
        cm = ContextManager(model="gemma")  # 8192 window so small content is measurable
        # 0.0 threshold → triggers if utilization > 0
        assert cm.needs_compaction("system prompt", [_msg("user", "hello world")], threshold=0.0) is True

    def test_hard_cap_triggers_compaction(self) -> None:
        cm = ContextManager(model="gemini-2.5-flash", max_conversation_tokens=10)
        msgs = [_msg("user", "a" * 200)]  # way more than 10 tokens
        assert cm.needs_compaction("sys", msgs) is True

    def test_hard_cap_not_triggered(self) -> None:
        cm = ContextManager(model="gemini-2.5-flash", max_conversation_tokens=1_000_000)
        msgs = [_msg("user", "hello")]
        assert cm.needs_compaction("sys", msgs) is False


# ===========================================================================
# ContextManager — modes
# ===========================================================================

class TestContextModes:
    """Context assembly modes: full, lean, auto, manual, adaptive."""

    def test_default_mode_is_auto(self) -> None:
        cm = ContextManager()
        assert cm.mode == "auto"

    def test_set_mode_valid(self) -> None:
        cm = ContextManager()
        for mode in ("full", "lean", "auto", "manual", "adaptive"):
            cm.set_mode(mode)
            assert cm.mode == mode

    def test_set_mode_invalid_raises(self) -> None:
        cm = ContextManager()
        with pytest.raises(ValueError, match="Invalid context mode"):
            cm.set_mode("turbo")

    def test_effective_mode_full_for_large_model(self) -> None:
        cm = ContextManager(model="gemini-2.5-flash")  # 1M >= 200K
        assert cm.get_effective_mode() == "full"

    def test_effective_mode_full_for_200k_model(self) -> None:
        cm = ContextManager(model="claude-sonnet-4")  # 200K
        assert cm.get_effective_mode() == "full"

    def test_effective_mode_adaptive_for_medium_model(self) -> None:
        cm = ContextManager(model="mixtral-8x7b")  # 32768 >= 32K
        assert cm.get_effective_mode() == "adaptive"

    def test_effective_mode_lean_for_small_model(self) -> None:
        cm = ContextManager(model="gemma")  # 8192 < 32K
        assert cm.get_effective_mode() == "lean"

    def test_effective_mode_non_auto_passthrough(self) -> None:
        cm = ContextManager(model="gemma")
        cm.set_mode("full")
        assert cm.get_effective_mode() == "full"


# ===========================================================================
# ContextSection
# ===========================================================================

class TestContextSection:
    """ContextSection dataclass with cached token counting."""

    def test_defaults(self) -> None:
        s = ContextSection(name="test", content="hello")
        assert s.priority == 5
        assert s.required is False

    def test_token_count_cached(self) -> None:
        s = ContextSection(name="test", content="a" * 80)
        first = s.token_count(model="gemini-2.5-flash")
        assert first == 20
        # Second call should return the same cached value
        assert s.token_count(model="gemini-2.5-flash") == 20

    def test_required_section(self) -> None:
        s = ContextSection(name="identity", content="I am njss", priority=10, required=True)
        assert s.required is True
        assert s.priority == 10


# ===========================================================================
# assemble_sections
# ===========================================================================

class TestAssembleSections:
    """Section assembly by mode and token budget."""

    def _make_sections(self) -> list[ContextSection]:
        return [
            ContextSection("identity", "I am an agent", priority=10, required=True),
            ContextSection("tools", "tool list here", priority=9, required=True),
            ContextSection("rules", "user rules", priority=8),
            ContextSection("repo_map", "a" * 400, priority=4),
            ContextSection("docs", "b" * 400, priority=2),
        ]

    def test_full_mode_includes_all(self) -> None:
        sections = self._make_sections()
        result = assemble_sections(sections, token_budget=10_000, mode="full")
        assert len(result) == len(sections)

    def test_full_mode_sorted_by_priority(self) -> None:
        sections = self._make_sections()
        result = assemble_sections(sections, token_budget=10_000, mode="full")
        # Required first, then by descending priority
        assert result[0].required is True
        assert result[1].required is True
        priorities = [s.priority for s in result]
        # After required sections, priorities should be non-increasing
        non_req = [p for s, p in zip(result, priorities) if not s.required]
        assert non_req == sorted(non_req, reverse=True)

    def test_lean_mode_only_high_priority(self) -> None:
        sections = self._make_sections()
        result = assemble_sections(sections, token_budget=10_000, mode="lean")
        # Only required or priority >= 9
        for s in result:
            assert s.required or s.priority >= 9
        # Should NOT include low-priority sections
        names = {s.name for s in result}
        assert "docs" not in names
        assert "repo_map" not in names

    def test_adaptive_mode_respects_budget(self) -> None:
        sections = self._make_sections()
        # Very small budget — should include required sections but skip large ones
        result = assemble_sections(sections, token_budget=20, model="gemini-2.5-flash", mode="adaptive")
        # Required sections are always included
        required_names = {s.name for s in result if s.required}
        assert "identity" in required_names
        assert "tools" in required_names

    def test_adaptive_mode_large_budget_includes_more(self) -> None:
        sections = self._make_sections()
        result = assemble_sections(sections, token_budget=100_000, model="gemini-2.5-flash", mode="adaptive")
        assert len(result) == len(sections)


# ===========================================================================
# SECTION_PRIORITIES
# ===========================================================================

class TestSectionPriorities:
    """The default section priority mapping."""

    def test_identity_highest(self) -> None:
        assert SECTION_PRIORITIES["identity"] == 10

    def test_tools_high(self) -> None:
        assert SECTION_PRIORITIES["tools"] == 9

    def test_known_keys_present(self) -> None:
        for key in ("identity", "user_info", "tools", "rules", "repo_map"):
            assert key in SECTION_PRIORITIES


# ===========================================================================
# _RESPONSE_RESERVE constant
# ===========================================================================

class TestResponseReserve:
    """The response reserve fraction."""

    def test_value(self) -> None:
        assert _RESPONSE_RESERVE == 0.15

    def test_reserve_used_in_budget(self) -> None:
        cm = ContextManager(model="gemma")
        window = cm.get_context_window()
        budget = cm.get_available_budget("", [])
        reserve = int(window * _RESPONSE_RESERVE)
        sys_tokens = count_tokens("", "gemma")
        assert budget == window - reserve - sys_tokens
