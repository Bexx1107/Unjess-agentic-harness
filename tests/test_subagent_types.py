"""Tests for unjess.subagents.types — agent types and orchestration patterns."""

from unittest.mock import MagicMock, call

import pytest

from unjess.subagents.types import (
    BUILTIN_TYPES,
    CODER_AGENT,
    RESEARCH_AGENT,
    REVIEWER_AGENT,
    SELF_AGENT,
    TESTER_AGENT,
    AgentType,
    FanOut,
    FanOutResult,
    Pipeline,
    PipelineStep,
    Supervisor,
    register_builtin_types,
)


# ---------------------------------------------------------------------------
# AgentType
# ---------------------------------------------------------------------------


class TestAgentType:
    """Tests for the AgentType dataclass."""

    def test_fields(self) -> None:
        at = AgentType(
            name="test",
            description="desc",
            system_prompt="prompt",
            tools=["a", "b"],
            enable_write_tools=True,
        )
        assert at.name == "test"
        assert at.description == "desc"
        assert at.system_prompt == "prompt"
        assert at.tools == ["a", "b"]
        assert at.enable_write_tools is True

    def test_default_tools_empty(self) -> None:
        at = AgentType(name="t", description="d", system_prompt="p")
        assert at.tools == []
        assert at.enable_write_tools is False

    def test_register_calls_define_type(self) -> None:
        at = AgentType(
            name="custom",
            description="Custom agent",
            system_prompt="Do stuff",
            tools=["read_file"],
            enable_write_tools=False,
        )
        manager = MagicMock()
        at.register(manager)
        manager.define_type.assert_called_once_with(
            name="custom",
            description="Custom agent",
            system_prompt="Do stuff",
            tools=["read_file"],
            enable_write_tools=False,
        )

    def test_register_none_tools_when_empty(self) -> None:
        at = AgentType(name="t", description="d", system_prompt="p", tools=[])
        manager = MagicMock()
        at.register(manager)
        manager.define_type.assert_called_once_with(
            name="t",
            description="d",
            system_prompt="p",
            tools=None,
            enable_write_tools=False,
        )


# ---------------------------------------------------------------------------
# Built-in agent types
# ---------------------------------------------------------------------------


class TestBuiltinTypes:
    """Tests for built-in agent type definitions."""

    def test_builtin_count(self) -> None:
        assert len(BUILTIN_TYPES) == 5

    def test_research_agent(self) -> None:
        assert RESEARCH_AGENT.name == "research"
        assert not RESEARCH_AGENT.enable_write_tools
        assert "read_file" in RESEARCH_AGENT.tools
        assert "write_file" not in RESEARCH_AGENT.tools

    def test_coder_agent(self) -> None:
        assert CODER_AGENT.name == "coder"
        assert CODER_AGENT.enable_write_tools
        assert "write_file" in CODER_AGENT.tools
        assert "run_command" in CODER_AGENT.tools

    def test_reviewer_agent(self) -> None:
        assert REVIEWER_AGENT.name == "reviewer"
        assert not REVIEWER_AGENT.enable_write_tools
        assert "write_file" not in REVIEWER_AGENT.tools

    def test_tester_agent(self) -> None:
        assert TESTER_AGENT.name == "tester"
        assert TESTER_AGENT.enable_write_tools
        assert "run_command" in TESTER_AGENT.tools

    def test_self_agent(self) -> None:
        assert SELF_AGENT.name == "self"
        assert SELF_AGENT.enable_write_tools
        assert SELF_AGENT.system_prompt == ""
        assert SELF_AGENT.tools == []

    def test_all_have_names(self) -> None:
        for at in BUILTIN_TYPES:
            assert at.name
            assert at.description

    def test_unique_names(self) -> None:
        names = [at.name for at in BUILTIN_TYPES]
        assert len(names) == len(set(names))


# ---------------------------------------------------------------------------
# register_builtin_types
# ---------------------------------------------------------------------------


class TestRegisterBuiltinTypes:
    """Tests for registering all built-in types with a manager."""

    def test_registers_all(self) -> None:
        manager = MagicMock()
        register_builtin_types(manager)
        assert manager.define_type.call_count == len(BUILTIN_TYPES)

    def test_registers_by_name(self) -> None:
        manager = MagicMock()
        register_builtin_types(manager)
        registered_names = [
            c.kwargs["name"] for c in manager.define_type.call_args_list
        ]
        expected_names = [at.name for at in BUILTIN_TYPES]
        assert registered_names == expected_names


# ---------------------------------------------------------------------------
# FanOutResult
# ---------------------------------------------------------------------------


class TestFanOutResult:
    """Tests for FanOutResult dataclass."""

    def test_all_succeeded_empty(self) -> None:
        r = FanOutResult()
        assert r.all_succeeded

    def test_all_succeeded_true(self) -> None:
        r = FanOutResult(results={"a": "ok", "b": "ok"})
        assert r.all_succeeded

    def test_all_succeeded_false(self) -> None:
        r = FanOutResult(results={"a": "ok"}, errors={"b": "failed"})
        assert not r.all_succeeded

    def test_summary_all_success(self) -> None:
        r = FanOutResult(results={"a": "ok", "b": "ok"})
        assert "2/2 succeeded" in r.summary()
        assert "0 failed" in r.summary()

    def test_summary_with_errors(self) -> None:
        r = FanOutResult(results={"a": "ok"}, errors={"b": "err"})
        assert "1/2 succeeded" in r.summary()
        assert "1 failed" in r.summary()

    def test_summary_all_failed(self) -> None:
        r = FanOutResult(errors={"a": "err", "b": "err"})
        assert "0/2 succeeded" in r.summary()

    def test_default_empty(self) -> None:
        r = FanOutResult()
        assert r.results == {}
        assert r.errors == {}


# ---------------------------------------------------------------------------
# FanOut
# ---------------------------------------------------------------------------


class TestFanOut:
    """Tests for the FanOut orchestration pattern."""

    def test_spawn_creates_agents(self) -> None:
        manager = MagicMock()
        manager.invoke.side_effect = ["id-1", "id-2", "id-3"]
        fan = FanOut(manager, agent_type="research")
        ids = fan.spawn(["task1", "task2", "task3"])
        assert ids == ["id-1", "id-2", "id-3"]
        assert manager.invoke.call_count == 3

    def test_spawn_with_custom_roles(self) -> None:
        manager = MagicMock()
        manager.invoke.side_effect = ["id-1", "id-2"]
        fan = FanOut(manager)
        ids = fan.spawn(["t1", "t2"], roles=["Alice", "Bob"])
        assert manager.invoke.call_args_list[0].kwargs["role"] == "Alice"
        assert manager.invoke.call_args_list[1].kwargs["role"] == "Bob"

    def test_spawn_default_roles(self) -> None:
        manager = MagicMock()
        manager.invoke.return_value = "id"
        fan = FanOut(manager)
        fan.spawn(["task1", "task2"])
        roles = [c.kwargs["role"] for c in manager.invoke.call_args_list]
        assert roles == ["Worker 1", "Worker 2"]

    def test_spawn_empty(self) -> None:
        manager = MagicMock()
        fan = FanOut(manager)
        ids = fan.spawn([])
        assert ids == []
        manager.invoke.assert_not_called()


# ---------------------------------------------------------------------------
# PipelineStep
# ---------------------------------------------------------------------------


class TestPipelineStep:
    """Tests for PipelineStep dataclass."""

    def test_fields(self) -> None:
        step = PipelineStep(
            name="plan",
            agent_type="research",
            prompt_template="Plan: {previous_result}",
            role="planner",
        )
        assert step.name == "plan"
        assert step.agent_type == "research"
        assert "{previous_result}" in step.prompt_template
        assert step.role == "planner"

    def test_default_role_empty(self) -> None:
        step = PipelineStep(name="s", agent_type="coder", prompt_template="x")
        assert step.role == ""


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class TestPipeline:
    """Tests for the Pipeline orchestration pattern."""

    def test_start_invokes_first_step(self) -> None:
        manager = MagicMock()
        manager.invoke.return_value = "step-0-id"
        steps = [
            PipelineStep(name="Plan", agent_type="research", prompt_template="Do: {previous_result}"),
        ]
        pipe = Pipeline(manager, steps)
        cid = pipe.start("initial input")
        assert cid == "step-0-id"
        manager.invoke.assert_called_once()
        # Check prompt has the initial input substituted
        prompt_arg = manager.invoke.call_args.kwargs["prompt"]
        assert "initial input" in prompt_arg

    def test_start_empty_pipeline(self) -> None:
        manager = MagicMock()
        pipe = Pipeline(manager, [])
        assert pipe.start("input") == ""
        manager.invoke.assert_not_called()

    def test_next_step_advances(self) -> None:
        manager = MagicMock()
        manager.invoke.return_value = "step-1-id"
        steps = [
            PipelineStep(name="Plan", agent_type="research", prompt_template="Plan"),
            PipelineStep(name="Code", agent_type="coder", prompt_template="Code: {previous_result}"),
        ]
        pipe = Pipeline(manager, steps)
        cid = pipe.next_step(1, "plan output")
        assert cid == "step-1-id"
        prompt_arg = manager.invoke.call_args.kwargs["prompt"]
        assert "plan output" in prompt_arg

    def test_next_step_beyond_end(self) -> None:
        manager = MagicMock()
        steps = [
            PipelineStep(name="Only", agent_type="research", prompt_template="x"),
        ]
        pipe = Pipeline(manager, steps)
        assert pipe.next_step(1, "result") is None

    def test_step_uses_role_or_name(self) -> None:
        manager = MagicMock()
        manager.invoke.return_value = "id"
        step_with_role = PipelineStep(
            name="Step1", agent_type="coder",
            prompt_template="x", role="custom-role",
        )
        pipe = Pipeline(manager, [step_with_role])
        pipe.start("input")
        assert manager.invoke.call_args.kwargs["role"] == "custom-role"

    def test_step_falls_back_to_name(self) -> None:
        manager = MagicMock()
        manager.invoke.return_value = "id"
        step_no_role = PipelineStep(
            name="MyStep", agent_type="coder",
            prompt_template="x",
        )
        pipe = Pipeline(manager, [step_no_role])
        pipe.start("input")
        assert manager.invoke.call_args.kwargs["role"] == "MyStep"


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------


class TestSupervisor:
    """Tests for the Supervisor orchestration pattern."""

    def _mock_agent(self, active: bool = True, status: str = "running", result: str = "") -> MagicMock:
        agent = MagicMock()
        agent.is_active = active
        agent.status.value = status
        agent.result = result
        return agent

    def test_delegate_creates_worker(self) -> None:
        manager = MagicMock()
        manager.invoke.return_value = "worker-1"
        manager.get_agent.return_value = None  # no active workers
        sup = Supervisor(manager, max_workers=5)
        cid = sup.delegate("task", "coder", role="Builder")
        assert cid == "worker-1"
        manager.invoke.assert_called_once()

    def test_delegate_at_capacity(self) -> None:
        manager = MagicMock()
        active_agent = self._mock_agent(active=True)
        manager.get_agent.return_value = active_agent
        manager.invoke.return_value = "w"

        sup = Supervisor(manager, max_workers=2)
        # Fill up
        sup.delegate("t1", "coder")
        sup.delegate("t2", "coder")
        # Should be at capacity now
        result = sup.delegate("t3", "coder")
        assert result is None

    def test_delegate_default_role(self) -> None:
        manager = MagicMock()
        manager.invoke.return_value = "w"
        manager.get_agent.return_value = None
        sup = Supervisor(manager)
        sup.delegate("task", "research")
        role_arg = manager.invoke.call_args.kwargs["role"]
        assert "research" in role_arg.lower()

    def test_active_workers_count(self) -> None:
        manager = MagicMock()
        active = self._mock_agent(active=True)
        inactive = self._mock_agent(active=False)
        manager.invoke.side_effect = ["w1", "w2"]
        manager.get_agent.side_effect = lambda aid: active if aid == "w1" else inactive

        sup = Supervisor(manager, max_workers=5)
        # Manually add workers
        sup._worker_ids = ["w1", "w2"]
        assert sup.active_workers == 1

    def test_active_workers_none_agents(self) -> None:
        manager = MagicMock()
        manager.get_agent.return_value = None
        sup = Supervisor(manager)
        sup._worker_ids = ["w1", "w2"]
        assert sup.active_workers == 0

    def test_collect_results(self) -> None:
        manager = MagicMock()
        completed = self._mock_agent(active=False, status="completed", result="done!")
        running = self._mock_agent(active=True, status="running", result="")
        manager.get_agent.side_effect = lambda aid: completed if aid == "w1" else running

        sup = Supervisor(manager)
        sup._worker_ids = ["w1", "w2"]
        results = sup.collect_results()
        assert results == {"w1": "done!"}

    def test_collect_results_empty(self) -> None:
        manager = MagicMock()
        manager.get_agent.return_value = None
        sup = Supervisor(manager)
        sup._worker_ids = ["w1"]
        assert sup.collect_results() == {}

    def test_max_workers_default(self) -> None:
        manager = MagicMock()
        sup = Supervisor(manager)
        assert sup._max_workers == 5
