"""Tests for unjess.agent — the core agent loop."""

import pytest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, PropertyMock
from collections import deque

from unjess.agent import (
    Agent, _MAX_TOOL_RETRIES, _STUCK_WINDOW, _PARALLEL_SAFE_TOOLS,
    _READ_ONLY_TOOLS, _CONSECUTIVE_READ_NUDGE, _CONSECUTIVE_READ_HARD_LIMIT,
)
from unjess.config import Settings
from unjess.llm.base import LLMResponse, ToolCall, Usage
from unjess.tools import ToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_settings(**overrides: Any) -> Settings:
    """Create a Settings with sensible test defaults."""
    defaults = {
        "model": "test-model",
        "provider": "openai",
        "workspace": ".",
        "max_iterations": 5,
    }
    defaults.update(overrides)
    s = Settings()
    for k, v in defaults.items():
        setattr(s, k, v)
    return s


def _make_agent(
    settings: Settings | None = None,
    **kwargs: Any,
) -> Agent:
    """Create an Agent with fully mocked dependencies."""
    if settings is None:
        settings = _make_settings()

    router = MagicMock()
    tools = kwargs.pop("tool_registry", ToolRegistry())
    display = MagicMock()
    permissions = MagicMock()
    permissions.check.return_value = True

    # Suppress heavy init by marking as child agent
    return Agent(
        settings=settings,
        router=router,
        tool_registry=tools,
        display=display,
        permissions=permissions,
        _is_child=True,  # skip model_router/architect/subagent init
        **kwargs,
    )


# ===================================================================
# Construction
# ===================================================================

class TestAgentConstruction:
    """Tests for Agent.__init__."""

    def test_child_agent_skips_heavy_init(self) -> None:
        agent = _make_agent()
        assert agent._model_router is None
        assert agent._architect is None
        assert agent._subagent_manager is None

    def test_max_iterations_from_settings(self) -> None:
        s = _make_settings(max_iterations=42)
        agent = _make_agent(settings=s)
        assert agent._max_iterations == 42

    def test_max_iterations_explicit_override(self) -> None:
        s = _make_settings(max_iterations=42)
        agent = _make_agent(settings=s, _max_iterations=99)
        assert agent._max_iterations == 99

    def test_conversation_starts_empty(self) -> None:
        agent = _make_agent()
        assert agent.conversation == []
        assert agent.conversation_length == 0

    def test_default_flags(self) -> None:
        agent = _make_agent()
        assert agent._planning_mode is False
        assert agent._goal_mode is False
        assert agent._abort_requested is False

    def test_context_manager_created_automatically(self) -> None:
        agent = _make_agent()
        assert agent._context_manager is not None

    def test_custom_context_manager(self) -> None:
        cm = MagicMock()
        agent = _make_agent(context_manager=cm)
        assert agent._context_manager is cm


# ===================================================================
# Conversation management
# ===================================================================

class TestConversationManagement:
    """Tests for conversation property and clear_history."""

    def test_conversation_setter(self) -> None:
        agent = _make_agent()
        new_conv = [{"role": "user", "content": "hi"}]
        agent.conversation = new_conv
        assert agent.conversation is new_conv
        assert agent.conversation_length == 1

    def test_clear_history(self) -> None:
        agent = _make_agent()
        agent._conversation = [{"role": "user", "content": "test"}]
        agent._recent_tool_calls.append("read_file")
        agent.clear_history()
        assert agent.conversation == []
        assert len(agent._recent_tool_calls) == 0

    def test_tool_registry_property(self) -> None:
        reg = ToolRegistry()
        agent = _make_agent(tool_registry=reg)
        assert agent.tool_registry is reg


# ===================================================================
# Stuck detection
# ===================================================================

class TestStuckDetection:
    """Tests for the _is_stuck method."""

    def test_not_stuck_when_empty(self) -> None:
        agent = _make_agent()
        assert agent._is_stuck() is False

    def test_not_stuck_with_varied_calls(self) -> None:
        agent = _make_agent()
        agent._recent_tool_calls.extend(["read_file:a", "write_file:b", "grep_search:c"])
        assert agent._is_stuck() is False

    def test_stuck_when_repeating(self) -> None:
        agent = _make_agent()
        # Fill with identical entries beyond the window
        for _ in range(_STUCK_WINDOW * 2):
            agent._recent_tool_calls.append("read_file:/same/path")
        assert agent._is_stuck() is True

    def test_stuck_window_size(self) -> None:
        agent = _make_agent()
        # Just under the window — not stuck
        unique_call = "read_file:/same/path"
        for _ in range(_STUCK_WINDOW - 1):
            agent._recent_tool_calls.append(unique_call)
        agent._recent_tool_calls.append("different_tool:other")
        assert agent._is_stuck() is False


# ===================================================================
# Constants
# ===================================================================

class TestConstants:
    """Tests for module-level constants."""

    def test_max_tool_retries(self) -> None:
        assert _MAX_TOOL_RETRIES >= 1

    def test_stuck_window(self) -> None:
        assert _STUCK_WINDOW >= 2

    def test_parallel_safe_tools(self) -> None:
        assert "read_file" in _PARALLEL_SAFE_TOOLS
        assert "list_dir" in _PARALLEL_SAFE_TOOLS
        assert "grep_search" in _PARALLEL_SAFE_TOOLS
        assert "write_file" not in _PARALLEL_SAFE_TOOLS
        assert "run_command" not in _PARALLEL_SAFE_TOOLS


# ===================================================================
# System prompt (child agent)
# ===================================================================

class TestSystemPromptChild:
    """Tests for _build_full_system_prompt on child agents."""

    def test_child_with_custom_prompt(self) -> None:
        agent = _make_agent()
        agent._custom_system_prompt = "You are a helper."
        prompt = agent._build_full_system_prompt()
        assert "You are a helper." in prompt

    def test_child_without_custom_prompt_uses_build_system_prompt(self) -> None:
        agent = _make_agent()
        agent._custom_system_prompt = ""
        # Should not crash — falls through to build_system_prompt
        prompt = agent._build_full_system_prompt()
        assert isinstance(prompt, str)


# ===================================================================
# Run method (integration-like)
# ===================================================================

class TestRunBasics:
    """Lightweight tests for the Agent.run() method."""

    def test_run_text_response_exits_loop(self) -> None:
        agent = _make_agent()
        # Mock the streaming to return a simple text response
        response = LLMResponse(
            text="Hello!",
            tool_calls=[],
            usage=Usage(prompt_tokens=10, completion_tokens=5),
            model="test-model",
        )
        agent._stream_response = MagicMock(return_value=response)
        agent._context_manager.needs_compaction = MagicMock(return_value=False)
        agent._context_manager.truncate_conversation = MagicMock(
            side_effect=lambda msgs, sp: msgs
        )

        agent.run("Hello!")
        assert agent.conversation_length >= 1  # at least the user message
        assert agent._conversation[0]["content"] == "Hello!"

    def test_run_appends_user_message(self) -> None:
        agent = _make_agent()
        response = LLMResponse(
            text="Reply", tool_calls=[], model="test-model",
            usage=Usage(prompt_tokens=5, completion_tokens=5),
        )
        agent._stream_response = MagicMock(return_value=response)
        agent._context_manager.needs_compaction = MagicMock(return_value=False)
        agent._context_manager.truncate_conversation = MagicMock(
            side_effect=lambda msgs, sp: msgs
        )

        agent.run("test input")
        assert any(
            m.get("content") == "test input" for m in agent._conversation
        )

    def test_run_abort_stops_loop(self) -> None:
        agent = _make_agent()

        # Set abort during the loop via a side effect
        def set_abort(*a, **kw):
            agent._abort_requested = True
            return False

        agent._context_manager.needs_compaction = MagicMock(side_effect=set_abort)
        agent._context_manager.truncate_conversation = MagicMock(
            side_effect=lambda msgs, sp: msgs
        )
        agent._stream_response = MagicMock()  # should NOT be called

        agent.run("should abort")
        # The display should show abort message
        agent._display.show_info.assert_called()

    def test_run_max_iterations_warning(self) -> None:
        settings = _make_settings(max_iterations=1)
        agent = _make_agent(settings=settings)

        # Return tool calls so the loop wants to continue
        tc = ToolCall(id="tc1", name="read_file", arguments={"path": "x"})
        response = LLMResponse(
            text="", tool_calls=[tc], model="test-model",
            usage=Usage(prompt_tokens=5, completion_tokens=5),
        )
        agent._stream_response = MagicMock(return_value=response)
        agent._context_manager.needs_compaction = MagicMock(return_value=False)
        agent._context_manager.truncate_conversation = MagicMock(
            side_effect=lambda msgs, sp: msgs
        )
        agent._handle_tool_calls = MagicMock(return_value=0)
        agent._is_stuck = MagicMock(return_value=False)

        agent.run("test")
        agent._display.show_warning.assert_called()

    def test_run_with_logger(self) -> None:
        agent = _make_agent()
        logger = MagicMock()
        agent._logger = logger

        response = LLMResponse(
            text="Response", tool_calls=[], model="test-model",
            usage=Usage(prompt_tokens=10, completion_tokens=5),
        )
        agent._stream_response = MagicMock(return_value=response)
        agent._context_manager.needs_compaction = MagicMock(return_value=False)
        agent._context_manager.truncate_conversation = MagicMock(
            side_effect=lambda msgs, sp: msgs
        )

        agent.run("test")
        logger.log_user_input.assert_called_once_with("test")
        logger.log_model_response.assert_called_once()

    def test_run_with_mention_resolver(self) -> None:
        agent = _make_agent()
        resolver = MagicMock()
        resolver.process_message.return_value = ("enriched @file", [])
        agent._mention_resolver = resolver

        response = LLMResponse(
            text="Reply", tool_calls=[], model="test-model",
            usage=Usage(prompt_tokens=5, completion_tokens=5),
        )
        agent._stream_response = MagicMock(return_value=response)
        agent._context_manager.needs_compaction = MagicMock(return_value=False)
        agent._context_manager.truncate_conversation = MagicMock(
            side_effect=lambda msgs, sp: msgs
        )

        agent.run("check @file.py")
        resolver.process_message.assert_called_once()

    def test_run_with_undo_checkpoint(self) -> None:
        agent = _make_agent()
        undo = MagicMock()
        agent._undo_manager = undo

        response = LLMResponse(
            text="Done", tool_calls=[], model="test-model",
            usage=Usage(prompt_tokens=5, completion_tokens=5),
        )
        agent._stream_response = MagicMock(return_value=response)
        agent._context_manager.needs_compaction = MagicMock(return_value=False)
        agent._context_manager.truncate_conversation = MagicMock(
            side_effect=lambda msgs, sp: msgs
        )

        agent.run("make changes")
        undo.checkpoint.assert_called_once()


# ===================================================================
# Error handling in run()
# ===================================================================

class TestRunErrorHandling:
    """Tests for error recovery in Agent.run()."""

    def test_run_llm_error_displays_error(self) -> None:
        agent = _make_agent()
        agent._stream_response = MagicMock(
            side_effect=Exception("API Error 500")
        )
        agent._context_manager.needs_compaction = MagicMock(return_value=False)
        agent._context_manager.truncate_conversation = MagicMock(
            side_effect=lambda msgs, sp: msgs
        )

        agent.run("test")
        agent._display.show_error.assert_called()

    def test_run_context_length_exceeded_auto_compacts(self) -> None:
        agent = _make_agent()
        # First call: context_length_exceeded, second call: success
        response = LLMResponse(
            text="OK", tool_calls=[], model="test-model",
            usage=Usage(prompt_tokens=5, completion_tokens=5),
        )
        call_count = [0]
        def stream_side_effect(*a, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("context_length_exceeded: too long")
            return response

        agent._stream_response = MagicMock(side_effect=stream_side_effect)
        agent._context_manager.needs_compaction = MagicMock(return_value=False)
        agent._context_manager.truncate_conversation = MagicMock(
            side_effect=lambda msgs, sp: msgs
        )
        # Pre-fill conversation so truncation is visible
        agent._conversation = [
            {"role": "user", "content": f"msg{i}"}
            for i in range(10)
        ]

        agent.run("trigger compact")
        # Should have auto-compacted
        agent._display.show_info.assert_called()

    def test_run_timeout_retries_and_succeeds(self) -> None:
        agent = _make_agent()
        response = LLMResponse(
            text="Done after timeout retry", tool_calls=[], model="test-model",
            usage=Usage(prompt_tokens=5, completion_tokens=5),
        )
        call_count = [0]
        def stream_side_effect(*a, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("Request timed out.")
            return response

        agent._stream_response = MagicMock(side_effect=stream_side_effect)
        agent._context_manager.needs_compaction = MagicMock(return_value=False)
        agent._context_manager.truncate_conversation = MagicMock(
            side_effect=lambda msgs, sp: msgs
        )

        with patch("time.sleep", return_value=None):
            agent.run("trigger timeout retry")

        assert call_count[0] == 2
        agent._display.show_info.assert_called()
        info_msgs = [str(call[0][0]) for call in agent._display.show_info.call_args_list]
        assert any("timed out" in m.lower() or "retrying" in m.lower() for m in info_msgs)


# ===================================================================
# Cost enforcement
# ===================================================================

class TestCostEnforcement:
    """Tests for session cost budget enforcement."""

    def test_cost_limit_stops_agent(self) -> None:
        settings = _make_settings(max_cost_per_session=0.01)
        agent = _make_agent(settings=settings)

        logger = MagicMock()
        logger.cost_tracker.total_cost = 0.05  # over budget
        agent._logger = logger

        response = LLMResponse(
            text="Reply", tool_calls=[], model="test-model",
            usage=Usage(prompt_tokens=10, completion_tokens=5),
        )
        agent._stream_response = MagicMock(return_value=response)
        agent._context_manager.needs_compaction = MagicMock(return_value=False)
        agent._context_manager.truncate_conversation = MagicMock(
            side_effect=lambda msgs, sp: msgs
        )

        agent.run("expensive request")
        agent._display.show_warning.assert_called()
        # Should have mentioned cost in the warning
        warning_text = agent._display.show_warning.call_args[0][0]
        assert "cost" in warning_text.lower() or "limit" in warning_text.lower()


# ===================================================================
# Anti-Paralysis Guards
# ===================================================================

class TestAntiParalysisGuards:
    """Tests for consecutive read limits and anti-paralysis guards."""

    def test_consecutive_read_nudge_injected(self) -> None:
        """When an agent executes pure read calls for N consecutive turns, a nudge is injected."""
        settings = _make_settings(max_iterations=20)
        agent = _make_agent(settings=settings)

        # Mock tools so read_file returns content
        agent._tools.execute = MagicMock(return_value="file contents line 1\nline 2")

        # Create responses: 10 consecutive turns of read_file, then a stop
        turn_count = [0]

        def fake_stream(messages, tools):
            turn_count[0] += 1
            if turn_count[0] <= 10:
                return LLMResponse(
                    text="",
                    tool_calls=[ToolCall(id=f"c{turn_count[0]}", name="read_file", arguments={"path": f"foo{turn_count[0]}.py"})],
                    model="test-model",
                    usage=Usage(prompt_tokens=5, completion_tokens=5),
                )
            # 11th turn: returns final text
            return LLMResponse(
                text="Here is the answer based on foo files.",
                tool_calls=[],
                model="test-model",
                usage=Usage(prompt_tokens=5, completion_tokens=5),
            )

        agent._stream_response = MagicMock(side_effect=fake_stream)
        agent._context_manager.needs_compaction = MagicMock(return_value=False)
        agent._context_manager.truncate_conversation = MagicMock(side_effect=lambda msgs, sp: msgs)

        agent.run("Inspect the codebase")

        # Turn 10 tool result should contain the nudge directive
        tool_msgs = [m for m in agent.conversation if m.get("role") == "tool"]
        assert len(tool_msgs) >= 10
        last_nudge_msg = tool_msgs[9]["content"]
        assert "SYSTEM NOTICE" in last_nudge_msg
        assert "consecutive rounds of file reading" in last_nudge_msg

    def test_consecutive_read_hard_limit_forces_synthesis(self) -> None:
        """When consecutive read turns reach hard limit and no action tools exist, loop forces synthesis without tools."""
        settings = _make_settings(max_iterations=30)
        agent = _make_agent(settings=settings)
        agent._tools.execute = MagicMock(return_value="sample content")

        tools_passed_to_stream = []

        def fake_stream(messages, tools):
            tools_passed_to_stream.append(tools)
            # Keep returning read_file tool calls
            return LLMResponse(
                text="",
                tool_calls=[ToolCall(id=f"call_{len(tools_passed_to_stream)}", name="read_file", arguments={"path": f"f_{len(tools_passed_to_stream)}.py"})],
                model="test-model",
                usage=Usage(prompt_tokens=5, completion_tokens=5),
            )

        agent._stream_response = MagicMock(side_effect=fake_stream)
        agent._context_manager.needs_compaction = MagicMock(return_value=False)
        agent._context_manager.truncate_conversation = MagicMock(side_effect=lambda msgs, sp: msgs)

        agent.run("Read everything")

        # Display should show warning about analysis paralysis
        agent._display.show_warning.assert_called()
        warning_text = agent._display.show_warning.call_args[0][0]
        assert "paralysis" in warning_text.lower() or "read turns" in warning_text.lower()

        # The final call to _stream_response must have tools=None (forcing synthesis text)
        assert tools_passed_to_stream[-1] is None

    def test_consecutive_read_hard_limit_transitions_to_action_tools(self) -> None:
        """When hard read limit is reached but action tools exist, agent transitions to action tools instead of halting."""
        settings = _make_settings(max_iterations=30)
        registry = ToolRegistry()
        registry.register("read_file", "Read file", {"type": "object", "properties": {"path": {"type": "string"}}}, lambda **kw: "content")
        registry.register("edit_file", "Edit file", {"type": "object", "properties": {"path": {"type": "string"}}}, lambda **kw: "edited")

        agent = _make_agent(settings=settings, tool_registry=registry)
        tools_passed = []

        def fake_stream(messages, tools):
            tools_passed.append(tools)
            if len(tools_passed) <= 20:
                resp = LLMResponse(
                    text="",
                    tool_calls=[ToolCall(id=f"c{len(tools_passed)}", name="read_file", arguments={"path": f"f_{len(tools_passed)}.py"})],
                    model="test-model",
                    usage=Usage(prompt_tokens=5, completion_tokens=5),
                )
            elif len(tools_passed) == 21:
                # Transition turn: model invokes edit_file using available action tools
                resp = LLMResponse(
                    text="",
                    tool_calls=[ToolCall(id="edit_call", name="edit_file", arguments={"path": "f_1.py"})],
                    model="test-model",
                    usage=Usage(prompt_tokens=5, completion_tokens=5),
                )
            else:
                resp = LLMResponse(
                    text="Completed the edits successfully.",
                    tool_calls=[],
                    model="test-model",
                    usage=Usage(prompt_tokens=5, completion_tokens=5),
                )
            agent._conversation.append({"role": "assistant", "content": resp.text})
            return resp

        agent._stream_response = MagicMock(side_effect=fake_stream)
        agent._context_manager.needs_compaction = MagicMock(return_value=False)
        agent._context_manager.truncate_conversation = MagicMock(side_effect=lambda msgs, sp: msgs)

        agent.run("Refactor files")

        # Turn 21 must only be offered action tools (edit_file), read_file must be excluded
        transition_tools = tools_passed[20]
        assert transition_tools is not None
        tool_names = [t["name"] for t in transition_tools]
        assert "edit_file" in tool_names
        assert "read_file" not in tool_names

        # The loop continued and executed the edit instead of dying
        tool_msgs = [m for m in agent.conversation if m.get("role") == "tool"]
        assert any("edited" in m["content"] for m in tool_msgs)
        # Agent finished with final response
        assert any("Completed the edits successfully." in m.get("content", "") for m in agent.conversation if m.get("role") == "assistant")

    def test_repeated_file_path_read_loop_guard(self) -> None:
        """Reading multiple different slices of a file is permitted up to threshold (6), then blocked."""
        agent = _make_agent()
        agent._tools.execute = MagicMock(return_value="file content")

        # Simulate reads of the same file path with different line numbers
        tcs = [
            ToolCall(id=str(i), name="read_file", arguments={"path": "plan.md", "start_line": i * 50, "end_line": (i + 1) * 50})
            for i in range(7)
        ]

        for tc in tcs:
            resp = LLMResponse(text="", tool_calls=[tc], model="test", usage=Usage())
            agent._handle_tool_calls(resp, 0)

        tool_msgs = [m for m in agent.conversation if m.get("role") == "tool"]
        assert len(tool_msgs) == 7
        # Early slices should have executed normally without false-positive blocking
        assert "file content" in tool_msgs[0]["content"]
        assert "file content" in tool_msgs[1]["content"]
        assert "file content" in tool_msgs[2]["content"]
        # The 7th tool call exceeds the 6-read threshold and should return the notice
        assert "already read" in tool_msgs[-1]["content"].lower() or "notice" in tool_msgs[-1]["content"].lower()

    def test_parallel_tool_timeout_recovers_gracefully(self) -> None:
        """When a tool in parallel execution times out or raises, it produces an error result without crashing."""
        agent = _make_agent()
        import time

        def slow_or_failing_tool(name, args):
            if args.get("path") == "slow.txt":
                raise TimeoutError("Simulated timeout")
            return "ok content"

        agent._tools.execute = MagicMock(side_effect=slow_or_failing_tool)

        tc1 = ToolCall(id="t1", name="read_file", arguments={"path": "slow.txt"})
        tc2 = ToolCall(id="t2", name="read_file", arguments={"path": "fast.txt"})

        agent._execute_parallel([tc1, tc2], 0)

        tool_msgs = [m for m in agent.conversation if m.get("role") == "tool"]
        assert len(tool_msgs) == 2
        # One tool should report error, the other should have succeeded
        assert any("Error" in m["content"] for m in tool_msgs)
        assert any("ok content" in m["content"] for m in tool_msgs)

    def test_multi_turn_auto_compaction_triggers_after_iteration_1(self) -> None:
        """Auto-compaction evaluates and triggers on turns > 1 when context exceeds threshold."""
        settings = _make_settings(max_iterations=5)
        agent = _make_agent(settings=settings)
        agent._context_manager.enable_compaction = True

        compaction_checks = []

        def mock_needs_compaction(sp, convo):
            compaction_checks.append(len(convo))
            # Trigger compaction on turn 2 (when convo has accumulated tool messages)
            return len(convo) >= 2

        agent._context_manager.needs_compaction = MagicMock(side_effect=mock_needs_compaction)
        compact_mock = MagicMock(side_effect=lambda convo, summarize_fn: convo[:2])
        agent._context_manager.compact = compact_mock

        # Turn 1: tool call
        # Turn 2: tool call (needs_compaction fires here!)
        # Turn 3: text response (stops)
        turns = [0]
        def fake_stream(messages, tools):
            turns[0] += 1
            if turns[0] == 1:
                return LLMResponse(text="", tool_calls=[ToolCall(id="1", name="read_file", arguments={"path": "f1.txt"})], model="test", usage=Usage())
            elif turns[0] == 2:
                return LLMResponse(text="", tool_calls=[ToolCall(id="2", name="read_file", arguments={"path": "f2.txt"})], model="test", usage=Usage())
            return LLMResponse(text="Done", tool_calls=[], model="test", usage=Usage())

        agent._stream_response = MagicMock(side_effect=fake_stream)
        agent._tools.execute = MagicMock(return_value="content")
        agent._context_manager.truncate_conversation = MagicMock(side_effect=lambda msgs, sp: msgs)

        agent.run("Run multi turn task")

        # Compaction should have been checked on iterations after 1
        assert len(compaction_checks) >= 2
        # And compact() should have been called
        assert compact_mock.called

    def test_agent_abort_calls_router_abort(self) -> None:
        """Calling agent.abort() sets _abort_requested and invokes router.abort()."""
        agent = _make_agent()
        agent._router.abort = MagicMock()
        agent.abort()
        assert agent._abort_requested is True
        agent._router.abort.assert_called_once()

