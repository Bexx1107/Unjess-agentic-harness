"""Tests for unjess.subagents.manager — subagent lifecycle and messaging."""

import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from unjess.subagents.manager import (
    AgentInfo,
    AgentStatus,
    MAX_CONCURRENT_SUBAGENTS,
    SubagentManager,
)
from unjess.subagents.messaging import Message, MessageBus


# ---------------------------------------------------------------------------
# AgentInfo dataclass
# ---------------------------------------------------------------------------

class TestAgentInfo:
    """Tests for the AgentInfo dataclass and its properties."""

    def test_default_status_is_pending(self) -> None:
        a = AgentInfo(conversation_id="abc", type_name="research", role="r", prompt="do stuff")
        assert a.status == AgentStatus.PENDING

    def test_created_at_auto_set(self) -> None:
        before = time.time()
        a = AgentInfo(conversation_id="abc", type_name="t", role="r", prompt="p")
        assert a.created_at >= before

    def test_is_active_pending(self) -> None:
        a = AgentInfo(conversation_id="a", type_name="t", role="r", prompt="p", status=AgentStatus.PENDING)
        assert a.is_active is True

    def test_is_active_running(self) -> None:
        a = AgentInfo(conversation_id="a", type_name="t", role="r", prompt="p", status=AgentStatus.RUNNING)
        assert a.is_active is True

    def test_is_active_idle(self) -> None:
        a = AgentInfo(conversation_id="a", type_name="t", role="r", prompt="p", status=AgentStatus.IDLE)
        assert a.is_active is True

    def test_not_active_completed(self) -> None:
        a = AgentInfo(conversation_id="a", type_name="t", role="r", prompt="p", status=AgentStatus.COMPLETED)
        assert a.is_active is False

    def test_not_active_failed(self) -> None:
        a = AgentInfo(conversation_id="a", type_name="t", role="r", prompt="p", status=AgentStatus.FAILED)
        assert a.is_active is False

    def test_not_active_killed(self) -> None:
        a = AgentInfo(conversation_id="a", type_name="t", role="r", prompt="p", status=AgentStatus.KILLED)
        assert a.is_active is False

    def test_runtime_seconds_while_running(self) -> None:
        a = AgentInfo(conversation_id="a", type_name="t", role="r", prompt="p", created_at=time.time() - 5)
        assert a.runtime_seconds >= 4.5

    def test_runtime_seconds_when_finished(self) -> None:
        now = time.time()
        a = AgentInfo(
            conversation_id="a", type_name="t", role="r", prompt="p",
            created_at=now - 10, finished_at=now,
        )
        assert abs(a.runtime_seconds - 10.0) < 0.5

    def test_summary_includes_status(self) -> None:
        a = AgentInfo(conversation_id="abcdefgh12345678", type_name="coder", role="Coder", prompt="p")
        s = a.summary()
        assert "coder" in s
        assert "pending" in s

    def test_summary_includes_icon(self) -> None:
        a = AgentInfo(conversation_id="abc12345", type_name="t", role="r", prompt="p", status=AgentStatus.COMPLETED)
        assert "✅" in a.summary()


# ---------------------------------------------------------------------------
# AgentStatus enum
# ---------------------------------------------------------------------------

class TestAgentStatus:
    """Tests for the AgentStatus enum values."""

    def test_all_values(self) -> None:
        assert AgentStatus.PENDING.value == "pending"
        assert AgentStatus.RUNNING.value == "running"
        assert AgentStatus.IDLE.value == "idle"
        assert AgentStatus.COMPLETED.value == "completed"
        assert AgentStatus.FAILED.value == "failed"
        assert AgentStatus.KILLED.value == "killed"


# ---------------------------------------------------------------------------
# SubagentManager — define_type
# ---------------------------------------------------------------------------

class TestDefineType:
    """Tests for defining agent types."""

    def test_define_type_stores_config(self, tmp_path: Path) -> None:
        mgr = SubagentManager("parent-1", tmp_path)
        mgr.define_type("research", "Research agent", "You are a researcher.", tools=["search"])
        assert "research" in mgr._agent_types
        assert mgr._agent_types["research"]["description"] == "Research agent"

    def test_define_type_with_write_tools(self, tmp_path: Path) -> None:
        mgr = SubagentManager("parent-1", tmp_path)
        mgr.define_type("coder", "Coding agent", "You write code.", enable_write_tools=True)
        assert mgr._agent_types["coder"]["enable_write_tools"] is True

    def test_define_multiple_types(self, tmp_path: Path) -> None:
        mgr = SubagentManager("parent-1", tmp_path)
        mgr.define_type("a", "Agent A", "System A")
        mgr.define_type("b", "Agent B", "System B")
        assert len(mgr._agent_types) == 2


# ---------------------------------------------------------------------------
# SubagentManager — invoke
# ---------------------------------------------------------------------------

class TestInvoke:
    """Tests for spawning subagents."""

    def test_invoke_returns_conversation_id(self, tmp_path: Path) -> None:
        mgr = SubagentManager("parent", tmp_path)
        cid = mgr.invoke(type_name="research", prompt="Find info")
        assert isinstance(cid, str)
        assert len(cid) == 16

    def test_invoke_stores_agent_info(self, tmp_path: Path) -> None:
        mgr = SubagentManager("parent", tmp_path)
        cid = mgr.invoke(type_name="research", prompt="Find X", role="Researcher")
        agent = mgr.get_agent(cid)
        assert agent is not None
        assert agent.type_name == "research"
        assert agent.role == "Researcher"
        assert agent.prompt == "Find X"

    def test_invoke_default_role_is_type_name(self, tmp_path: Path) -> None:
        mgr = SubagentManager("parent", tmp_path)
        cid = mgr.invoke(type_name="coder", prompt="Write code")
        agent = mgr.get_agent(cid)
        assert agent is not None
        assert agent.role == "coder"

    def test_invoke_model_override(self, tmp_path: Path) -> None:
        mgr = SubagentManager("parent", tmp_path)
        cid = mgr.invoke(type_name="t", prompt="p", model_override="gpt-4o")
        agent = mgr.get_agent(cid)
        assert agent is not None
        assert agent.model_override == "gpt-4o"

    def test_invoke_tool_filter(self, tmp_path: Path) -> None:
        mgr = SubagentManager("parent", tmp_path)
        cid = mgr.invoke(type_name="t", prompt="p", tool_filter=["read_file", "search"])
        agent = mgr.get_agent(cid)
        assert agent is not None
        assert agent.tool_filter == ["read_file", "search"]

    def test_invoke_max_turns(self, tmp_path: Path) -> None:
        mgr = SubagentManager("parent", tmp_path)
        cid = mgr.invoke(type_name="t", prompt="p", max_turns=5)
        agent = mgr.get_agent(cid)
        assert agent is not None
        assert agent.max_turns == 5

    def test_invoke_concurrency_limit(self, tmp_path: Path) -> None:
        mgr = SubagentManager("parent", tmp_path)
        # Spawn MAX_CONCURRENT_SUBAGENTS agents
        blocker = threading.Event()

        def blocking_factory(info, config, ws, bus):
            def runner():
                blocker.wait(timeout=30)
                return "done"
            return runner

        mgr.set_agent_factory(blocking_factory)

        ids = []
        for i in range(MAX_CONCURRENT_SUBAGENTS):
            cid = mgr.invoke(type_name="t", prompt=f"task {i}")
            ids.append(cid)

        # The next one should raise
        with pytest.raises(RuntimeError, match="concurrency limit"):
            mgr.invoke(type_name="t", prompt="one too many")

        # Cleanup
        blocker.set()
        for cid in ids:
            thread = mgr._threads.get(cid)
            if thread:
                thread.join(timeout=5.0)

    def test_invoke_inherit_workspace(self, tmp_path: Path) -> None:
        mgr = SubagentManager("parent", tmp_path)
        cid = mgr.invoke(type_name="t", prompt="p", workspace_mode="inherit")
        agent = mgr.get_agent(cid)
        assert agent is not None
        assert agent.workspace == str(tmp_path)

    def test_invoke_branch_workspace_creates_dir(self, tmp_path: Path) -> None:
        mgr = SubagentManager("parent", tmp_path)
        cid = mgr.invoke(type_name="t", prompt="p", workspace_mode="branch")
        agent = mgr.get_agent(cid)
        assert agent is not None
        branch_path = Path(agent.workspace)
        assert branch_path.exists()
        assert ".unjess" in str(branch_path)

    def test_invoke_share_workspace(self, tmp_path: Path) -> None:
        mgr = SubagentManager("parent", tmp_path)
        cid = mgr.invoke(type_name="t", prompt="p", workspace_mode="share")
        agent = mgr.get_agent(cid)
        assert agent is not None
        assert agent.workspace == str(tmp_path)


# ---------------------------------------------------------------------------
# SubagentManager — kill
# ---------------------------------------------------------------------------

class TestKill:
    """Tests for killing subagents."""

    def test_kill_existing_agent(self, tmp_path: Path) -> None:
        mgr = SubagentManager("parent", tmp_path)
        cid = mgr.invoke(type_name="t", prompt="p")
        result = mgr.kill(cid)
        assert result is True
        agent = mgr.get_agent(cid)
        assert agent is not None
        assert agent.status == AgentStatus.KILLED
        assert agent.finished_at > 0

    def test_kill_nonexistent_returns_false(self, tmp_path: Path) -> None:
        mgr = SubagentManager("parent", tmp_path)
        assert mgr.kill("nonexistent") is False

    def test_kill_removes_thread_reference(self, tmp_path: Path) -> None:
        mgr = SubagentManager("parent", tmp_path)
        cid = mgr.invoke(type_name="t", prompt="p")
        assert cid in mgr._threads
        mgr.kill(cid)
        assert cid not in mgr._threads


# ---------------------------------------------------------------------------
# SubagentManager — kill_all
# ---------------------------------------------------------------------------

class TestKillAll:
    """Tests for killing all active subagents."""

    def test_kill_all_returns_count(self, tmp_path: Path) -> None:
        mgr = SubagentManager("parent", tmp_path)
        blocker = threading.Event()

        def blocking_factory(info, config, ws, bus):
            def runner():
                blocker.wait(timeout=30)
                return "done"
            return runner

        mgr.set_agent_factory(blocking_factory)
        mgr.invoke(type_name="t", prompt="a")
        mgr.invoke(type_name="t", prompt="b")
        time.sleep(0.1)  # let threads start
        count = mgr.kill_all()
        assert count == 2
        blocker.set()

    def test_kill_all_empty_manager(self, tmp_path: Path) -> None:
        mgr = SubagentManager("parent", tmp_path)
        assert mgr.kill_all() == 0

    def test_kill_all_skips_already_killed(self, tmp_path: Path) -> None:
        mgr = SubagentManager("parent", tmp_path)
        cid = mgr.invoke(type_name="t", prompt="p")
        mgr.kill(cid)
        count = mgr.kill_all()
        assert count == 0


# ---------------------------------------------------------------------------
# SubagentManager — list_active / list_all / active_count
# ---------------------------------------------------------------------------

class TestListing:
    """Tests for listing subagents."""

    def test_list_active_excludes_killed(self, tmp_path: Path) -> None:
        mgr = SubagentManager("parent", tmp_path)
        cid1 = mgr.invoke(type_name="t", prompt="a")
        cid2 = mgr.invoke(type_name="t", prompt="b")
        mgr.kill(cid1)
        active = mgr.list_active()
        active_ids = [a.conversation_id for a in active]
        assert cid1 not in active_ids

    def test_list_all_includes_killed(self, tmp_path: Path) -> None:
        mgr = SubagentManager("parent", tmp_path)
        cid = mgr.invoke(type_name="t", prompt="p")
        mgr.kill(cid)
        all_agents = mgr.list_all()
        assert len(all_agents) == 1
        assert all_agents[0].status == AgentStatus.KILLED

    def test_active_count_property(self, tmp_path: Path) -> None:
        mgr = SubagentManager("parent", tmp_path)
        blocker = threading.Event()

        def blocking_factory(info, config, ws, bus):
            def runner():
                blocker.wait(timeout=30)
                return "done"
            return runner

        mgr.set_agent_factory(blocking_factory)
        mgr.invoke(type_name="t", prompt="a")
        mgr.invoke(type_name="t", prompt="b")
        time.sleep(0.1)
        assert mgr.active_count == 2
        mgr.kill_all()
        assert mgr.active_count == 0
        blocker.set()


# ---------------------------------------------------------------------------
# SubagentManager — messaging
# ---------------------------------------------------------------------------

class TestMessaging:
    """Tests for inter-agent message passing."""

    def test_send_message_to_existing_agent(self, tmp_path: Path) -> None:
        mgr = SubagentManager("parent", tmp_path)
        cid = mgr.invoke(type_name="t", prompt="p")
        result = mgr.send_message(cid, "Hello agent!")
        assert result is True

    def test_send_message_to_nonexistent_returns_false(self, tmp_path: Path) -> None:
        mgr = SubagentManager("parent", tmp_path)
        result = mgr.send_message("nonexistent", "Hello?")
        assert result is False

    def test_send_message_delivered_via_bus(self, tmp_path: Path) -> None:
        bus = MessageBus()
        mgr = SubagentManager("parent", tmp_path, message_bus=bus)
        cid = mgr.invoke(type_name="t", prompt="p")
        mgr.send_message(cid, "Test message content")
        messages = bus.receive(cid)
        assert len(messages) == 1
        assert messages[0].content == "Test message content"
        assert messages[0].sender == "parent"

    def test_receive_messages_from_subagents(self, tmp_path: Path) -> None:
        bus = MessageBus()
        mgr = SubagentManager("parent", tmp_path, message_bus=bus)
        # Simulate a subagent sending a message to parent
        bus.send(Message(sender="sub-1", recipient="parent", content="Result from sub"))
        messages = mgr.receive_messages()
        assert len(messages) == 1
        assert messages[0].content == "Result from sub"

    def test_receive_clears_mailbox(self, tmp_path: Path) -> None:
        bus = MessageBus()
        mgr = SubagentManager("parent", tmp_path, message_bus=bus)
        bus.send(Message(sender="sub", recipient="parent", content="msg1"))
        first = mgr.receive_messages()
        assert len(first) == 1
        second = mgr.receive_messages()
        assert len(second) == 0


# ---------------------------------------------------------------------------
# SubagentManager — agent factory and _run_agent
# ---------------------------------------------------------------------------

class TestAgentFactory:
    """Tests for agent factory and background execution."""

    def test_set_agent_factory(self, tmp_path: Path) -> None:
        mgr = SubagentManager("parent", tmp_path)
        factory = MagicMock()
        mgr.set_agent_factory(factory)
        assert mgr._agent_factory is factory

    def test_no_factory_sends_error_message(self, tmp_path: Path) -> None:
        bus = MessageBus()
        mgr = SubagentManager("parent", tmp_path, message_bus=bus)
        cid = mgr.invoke(type_name="t", prompt="do something")
        # Wait for the thread to complete
        thread = mgr._threads.get(cid)
        if thread:
            thread.join(timeout=5.0)
        # Should have sent error message to parent
        messages = bus.receive("parent")
        assert len(messages) >= 1
        assert "No agent factory" in messages[0].content

    def test_factory_result_stored(self, tmp_path: Path) -> None:
        def factory(info, config, ws, bus):
            def runner():
                return "factory result"
            return runner

        mgr = SubagentManager("parent", tmp_path)
        mgr.set_agent_factory(factory)
        cid = mgr.invoke(type_name="t", prompt="p")
        thread = mgr._threads.get(cid)
        if thread:
            thread.join(timeout=5.0)
        agent = mgr.get_agent(cid)
        assert agent is not None
        assert agent.result == "factory result"
        assert agent.status == AgentStatus.COMPLETED

    def test_factory_exception_sets_failed(self, tmp_path: Path) -> None:
        def factory(info, config, ws, bus):
            def runner():
                raise ValueError("factory boom")
            return runner

        bus = MessageBus()
        mgr = SubagentManager("parent", tmp_path, message_bus=bus)
        mgr.set_agent_factory(factory)
        cid = mgr.invoke(type_name="t", prompt="p")
        thread = mgr._threads.get(cid)
        if thread:
            thread.join(timeout=5.0)
        agent = mgr.get_agent(cid)
        assert agent is not None
        assert agent.status == AgentStatus.FAILED
        assert "factory boom" in agent.error


# ---------------------------------------------------------------------------
# SubagentManager — properties and catalog
# ---------------------------------------------------------------------------

class TestManagerProperties:
    """Tests for manager properties and catalog_for_prompt."""

    def test_parent_id_property(self, tmp_path: Path) -> None:
        mgr = SubagentManager("my-parent-id", tmp_path)
        assert mgr.parent_id == "my-parent-id"

    def test_message_bus_property(self, tmp_path: Path) -> None:
        bus = MessageBus()
        mgr = SubagentManager("parent", tmp_path, message_bus=bus)
        assert mgr.message_bus is bus

    def test_default_message_bus_created(self, tmp_path: Path) -> None:
        mgr = SubagentManager("parent", tmp_path)
        assert isinstance(mgr.message_bus, MessageBus)

    def test_catalog_empty_when_no_types(self, tmp_path: Path) -> None:
        mgr = SubagentManager("parent", tmp_path)
        assert mgr.catalog_for_prompt() == ""

    def test_catalog_lists_defined_types(self, tmp_path: Path) -> None:
        mgr = SubagentManager("parent", tmp_path)
        mgr.define_type("research", "Research assistant", "System prompt")
        mgr.define_type("coder", "Coding assistant", "System prompt")
        catalog = mgr.catalog_for_prompt()
        assert "research" in catalog
        assert "coder" in catalog
        assert "Research assistant" in catalog

    def test_catalog_shows_active_agents(self, tmp_path: Path) -> None:
        mgr = SubagentManager("parent", tmp_path)
        mgr.define_type("t", "Test type", "sys")
        blocker = threading.Event()

        def blocking_factory(info, config, ws, bus):
            def runner():
                blocker.wait(timeout=30)
                return "done"
            return runner

        mgr.set_agent_factory(blocking_factory)
        mgr.invoke(type_name="t", prompt="task", role="Worker")
        time.sleep(0.1)
        catalog = mgr.catalog_for_prompt()
        assert "Active subagents: 1" in catalog
        mgr.kill_all()
        blocker.set()


# ---------------------------------------------------------------------------
# SubagentManager — fan_out
# ---------------------------------------------------------------------------

class TestFanOut:
    """Tests for fan-out agent spawning."""

    def test_fan_out_spawns_multiple(self, tmp_path: Path) -> None:
        mgr = SubagentManager("parent", tmp_path)
        ids = mgr.fan_out(["prompt1", "prompt2"], agent_type="research", role="Researcher")
        assert len(ids) == 2
        for cid in ids:
            agent = mgr.get_agent(cid)
            assert agent is not None
        # Cleanup
        mgr.kill_all()

    def test_fan_out_assigns_numbered_roles(self, tmp_path: Path) -> None:
        mgr = SubagentManager("parent", tmp_path)
        ids = mgr.fan_out(["a", "b"], role="Worker")
        agents = [mgr.get_agent(cid) for cid in ids]
        assert agents[0] is not None
        assert agents[0].role == "Worker 1"
        assert agents[1] is not None
        assert agents[1].role == "Worker 2"
        mgr.kill_all()

    def test_fan_out_respects_concurrency_limit(self, tmp_path: Path) -> None:
        mgr = SubagentManager("parent", tmp_path)
        blocker = threading.Event()

        def blocking_factory(info, config, ws, bus):
            def runner():
                blocker.wait(timeout=30)
                return "done"
            return runner

        mgr.set_agent_factory(blocking_factory)
        prompts = [f"task {i}" for i in range(MAX_CONCURRENT_SUBAGENTS + 1)]
        with pytest.raises(RuntimeError, match="concurrency limit"):
            mgr.fan_out(prompts)
        mgr.kill_all()
        blocker.set()


# ---------------------------------------------------------------------------
# SubagentManager — get_agent
# ---------------------------------------------------------------------------

class TestGetAgent:
    """Tests for retrieving agent info."""

    def test_get_existing_agent(self, tmp_path: Path) -> None:
        mgr = SubagentManager("parent", tmp_path)
        cid = mgr.invoke(type_name="t", prompt="p")
        assert mgr.get_agent(cid) is not None

    def test_get_nonexistent_returns_none(self, tmp_path: Path) -> None:
        mgr = SubagentManager("parent", tmp_path)
        assert mgr.get_agent("does-not-exist") is None
