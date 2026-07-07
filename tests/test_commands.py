"""Tests for unjess.commands — slash command registry, dispatch, and handlers."""

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from unjess.commands import (
    CommandContext,
    SlashCommand,
    register,
    dispatch,
    get_all_commands,
    register_builtin_commands,
    _cmd_exit,
    _cmd_help,
    _cmd_clear,
    _cmd_compact,
    _cmd_verbose,
    _cmd_quiet,
    _cmd_undo,
    _cmd_learn,
    _cmd_map,
    _cmd_budget,
    _cmd_goal,
    _cmd_browser,
    _cmd_diff,
    _cmd_test,
    _cmd_cost,
    _cmd_context,
    _cmd_schedule,
    _COMMANDS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ctx(**overrides: Any) -> CommandContext:
    """Build a CommandContext with MagicMocks for all dependencies."""
    mock_agent = MagicMock()
    mock_agent.conversation_length = 10
    mock_agent.conversation = [{"role": "user", "content": "hi"}] * 10
    mock_agent._conversation = mock_agent.conversation
    mock_agent.tool_registry = MagicMock()

    mock_settings = MagicMock()
    mock_settings.workspace = "/tmp/test"
    mock_settings.model = "test-model"
    mock_settings.provider = "test-provider"
    mock_settings.verbosity = "normal"
    mock_settings.max_cost_per_session = 0.0
    mock_settings.api_keys = {}
    mock_settings.api_key_pool = {}

    mock_display = MagicMock()
    mock_display.console = MagicMock()
    mock_display._verbose = False

    mock_router = MagicMock()
    mock_router.active_provider_name = "test-provider"
    mock_router.available_providers = ["test-provider"]

    ctx = CommandContext(
        agent=overrides.get("agent", mock_agent),
        settings=overrides.get("settings", mock_settings),
        display=overrides.get("display", mock_display),
        router=overrides.get("router", mock_router),
        undo_manager=overrides.get("undo_manager", None),
        context_manager=overrides.get("context_manager", None),
        logger=overrides.get("logger", None),
        repo_map=overrides.get("repo_map", None),
        memory_store=overrides.get("memory_store", None),
        input_handler=overrides.get("input_handler", None),
    )
    return ctx


# ---------------------------------------------------------------------------
# Tests: SlashCommand dataclass
# ---------------------------------------------------------------------------

class TestSlashCommand:
    """SlashCommand dataclass."""

    def test_basic_creation(self) -> None:
        handler = lambda ctx, args: True
        cmd = SlashCommand(name="/test", description="Test command", handler=handler)
        assert cmd.name == "/test"
        assert cmd.description == "Test command"
        assert cmd.handler is handler
        assert cmd.usage == ""

    def test_with_usage(self) -> None:
        cmd = SlashCommand(name="/model", description="Switch model", handler=lambda c, a: True, usage="<name>")
        assert cmd.usage == "<name>"


# ---------------------------------------------------------------------------
# Tests: Registration and lookup
# ---------------------------------------------------------------------------

class TestCommandRegistry:
    """Command registration and retrieval."""

    def test_register_custom_command(self) -> None:
        handler = lambda ctx, args: True
        original_commands = dict(_COMMANDS)
        try:
            register("/test_custom_xyz", "A test command", handler, usage="<arg>")
            assert "/test_custom_xyz" in _COMMANDS
            cmd = _COMMANDS["/test_custom_xyz"]
            assert cmd.description == "A test command"
            assert cmd.usage == "<arg>"
            assert cmd.handler is handler
        finally:
            # Clean up
            _COMMANDS.pop("/test_custom_xyz", None)

    def test_get_all_commands_returns_dict(self) -> None:
        all_cmds = get_all_commands()
        assert isinstance(all_cmds, dict)
        # Should contain at least the built-in commands
        assert "/help" in all_cmds
        assert "/exit" in all_cmds

    def test_get_all_commands_returns_copy(self) -> None:
        cmds1 = get_all_commands()
        cmds2 = get_all_commands()
        assert cmds1 is not cmds2

    def test_builtin_commands_registered(self) -> None:
        expected = [
            "/exit", "/quit", "/help", "/model", "/clear", "/settings",
            "/compact", "/context", "/cost", "/undo", "/verbose", "/quiet",
            "/init", "/setup", "/skill", "/mcp", "/map", "/learn",
            "/schedule", "/serve", "/plan", "/free", "/provider",
            "/goal", "/browser", "/diff", "/test", "/agents", "/budget",
        ]
        all_cmds = get_all_commands()
        for cmd_name in expected:
            assert cmd_name in all_cmds, f"Missing built-in command: {cmd_name}"


# ---------------------------------------------------------------------------
# Tests: dispatch
# ---------------------------------------------------------------------------

class TestDispatch:
    """Command dispatch system."""

    def test_dispatch_known_command(self) -> None:
        ctx = _make_ctx()
        result = dispatch("/help", ctx)
        assert result is True

    def test_dispatch_unknown_command(self) -> None:
        ctx = _make_ctx()
        result = dispatch("/nonexistent_xyz", ctx)
        assert result is True  # unknown commands return True (continue REPL)
        ctx.display.show_error.assert_called_once()
        assert "Unknown command" in ctx.display.show_error.call_args[0][0]

    def test_dispatch_with_args(self) -> None:
        ctx = _make_ctx()
        # /model with a direct name should work
        result = dispatch("/model fast-model", ctx)
        assert result is True

    def test_dispatch_strips_whitespace(self) -> None:
        ctx = _make_ctx()
        result = dispatch("  /help  ", ctx)
        assert result is True

    def test_dispatch_case_insensitive_command(self) -> None:
        ctx = _make_ctx()
        result = dispatch("/HELP", ctx)
        assert result is True

    def test_dispatch_exit_returns_false(self) -> None:
        ctx = _make_ctx()
        result = dispatch("/exit", ctx)
        assert result is False

    def test_dispatch_quit_returns_false(self) -> None:
        ctx = _make_ctx()
        result = dispatch("/quit", ctx)
        assert result is False


# ---------------------------------------------------------------------------
# Tests: _cmd_exit
# ---------------------------------------------------------------------------

class TestCmdExit:
    """Exit command handler."""

    def test_returns_false(self) -> None:
        ctx = _make_ctx()
        result = _cmd_exit(ctx, "")
        assert result is False

    def test_shows_goodbye(self) -> None:
        ctx = _make_ctx()
        _cmd_exit(ctx, "")
        ctx.display.show_info.assert_called_once()
        assert "Goodbye" in ctx.display.show_info.call_args[0][0]


# ---------------------------------------------------------------------------
# Tests: _cmd_help
# ---------------------------------------------------------------------------

class TestCmdHelp:
    """Help command handler."""

    def test_returns_true(self) -> None:
        ctx = _make_ctx()
        result = _cmd_help(ctx, "")
        assert result is True

    def test_prints_panel(self) -> None:
        ctx = _make_ctx()
        _cmd_help(ctx, "")
        ctx.display.console.print.assert_called_once()
        from rich.panel import Panel
        arg = ctx.display.console.print.call_args[0][0]
        assert isinstance(arg, Panel)


# ---------------------------------------------------------------------------
# Tests: _cmd_clear
# ---------------------------------------------------------------------------

class TestCmdClear:
    """Clear command handler."""

    def test_returns_true(self) -> None:
        ctx = _make_ctx()
        result = _cmd_clear(ctx, "")
        assert result is True

    def test_clears_history(self) -> None:
        ctx = _make_ctx()
        _cmd_clear(ctx, "")
        ctx.agent.clear_history.assert_called_once()

    def test_shows_info_message(self) -> None:
        ctx = _make_ctx()
        _cmd_clear(ctx, "")
        ctx.display.show_info.assert_called_once()
        assert "cleared" in ctx.display.show_info.call_args[0][0].lower()


# ---------------------------------------------------------------------------
# Tests: _cmd_compact
# ---------------------------------------------------------------------------

class TestCmdCompact:
    """Compact command handler."""

    def test_returns_true(self) -> None:
        ctx = _make_ctx(context_manager=MagicMock())
        result = _cmd_compact(ctx, "")
        assert result is True

    def test_error_without_context_manager(self) -> None:
        ctx = _make_ctx(context_manager=None)
        result = _cmd_compact(ctx, "")
        assert result is True
        ctx.display.show_error.assert_called_once()

    def test_calls_compact_on_context_manager(self) -> None:
        mock_cm = MagicMock()
        mock_cm.compact.return_value = [{"role": "system", "content": "summary"}]
        ctx = _make_ctx(context_manager=mock_cm)
        _cmd_compact(ctx, "")
        mock_cm.compact.assert_called_once()

    def test_shows_message_count_change(self) -> None:
        mock_cm = MagicMock()
        mock_cm.compact.return_value = [{"role": "system", "content": "summary"}]
        mock_agent = MagicMock()
        mock_agent.conversation_length = 10
        # After compact, simulate reduced conversation
        type(mock_agent).conversation_length = PropertyMock(side_effect=[10, 1])
        ctx = _make_ctx(agent=mock_agent, context_manager=mock_cm)
        _cmd_compact(ctx, "")
        ctx.display.show_info.assert_called_once()
        info_msg = ctx.display.show_info.call_args[0][0]
        assert "10" in info_msg
        assert "1" in info_msg


# ---------------------------------------------------------------------------
# Tests: _cmd_verbose / _cmd_quiet
# ---------------------------------------------------------------------------

class TestCmdVerboseQuiet:
    """Verbose and quiet mode toggling."""

    def test_verbose_sets_flag(self) -> None:
        ctx = _make_ctx()
        result = _cmd_verbose(ctx, "")
        assert result is True
        assert ctx.settings.verbosity == "verbose"
        assert ctx.display._verbose is True

    def test_quiet_sets_flag(self) -> None:
        ctx = _make_ctx()
        result = _cmd_quiet(ctx, "")
        assert result is True
        assert ctx.settings.verbosity == "quiet"
        assert ctx.display._verbose is False


# ---------------------------------------------------------------------------
# Tests: _cmd_undo
# ---------------------------------------------------------------------------

class TestCmdUndo:
    """Undo command handler."""

    def test_error_without_undo_manager(self) -> None:
        ctx = _make_ctx(undo_manager=None)
        result = _cmd_undo(ctx, "")
        assert result is True
        ctx.display.show_error.assert_called_once()

    def test_undo_with_default_steps(self) -> None:
        mock_undo = MagicMock()
        mock_undo.undo.return_value = (True, "Undid 1 change")
        ctx = _make_ctx(undo_manager=mock_undo)
        result = _cmd_undo(ctx, "")
        assert result is True
        mock_undo.undo.assert_called_once_with(steps=1)
        ctx.display.show_info.assert_called_once()

    def test_undo_with_numeric_steps(self) -> None:
        mock_undo = MagicMock()
        mock_undo.undo.return_value = (True, "Undid 3 changes")
        ctx = _make_ctx(undo_manager=mock_undo)
        _cmd_undo(ctx, "3")
        mock_undo.undo.assert_called_once_with(steps=3)

    def test_undo_failure_shows_error(self) -> None:
        mock_undo = MagicMock()
        mock_undo.undo.return_value = (False, "Nothing to undo")
        ctx = _make_ctx(undo_manager=mock_undo)
        _cmd_undo(ctx, "")
        ctx.display.show_error.assert_called_once()

    def test_undo_list_subcommand(self) -> None:
        mock_undo = MagicMock()
        mock_undo.list_entries.return_value = []
        ctx = _make_ctx(undo_manager=mock_undo)
        _cmd_undo(ctx, "list")
        mock_undo.list_entries.assert_called_once_with(n=10)

    def test_undo_list_with_entries(self) -> None:
        mock_entry = MagicMock()
        mock_entry.display.return_value = "Checkpoint: file.py modified"
        mock_undo = MagicMock()
        mock_undo.list_entries.return_value = [mock_entry]
        ctx = _make_ctx(undo_manager=mock_undo)
        _cmd_undo(ctx, "list")
        ctx.display.console.print.assert_called()


# ---------------------------------------------------------------------------
# Tests: _cmd_learn
# ---------------------------------------------------------------------------

class TestCmdLearn:
    """Learn command handler."""

    def test_error_without_memory_store(self) -> None:
        ctx = _make_ctx(memory_store=None)
        result = _cmd_learn(ctx, "Always use type hints")
        assert result is True
        ctx.display.show_error.assert_called_once()

    def test_learn_new_rule(self) -> None:
        mock_mem = MagicMock()
        mock_rule = MagicMock()
        mock_rule.rule = "Use ruff for linting"
        mock_rule.id = "abc123"
        mock_mem.learn.return_value = mock_rule
        ctx = _make_ctx(memory_store=mock_mem)
        result = _cmd_learn(ctx, "Use ruff for linting")
        assert result is True
        mock_mem.learn.assert_called_once_with(
            rule="Use ruff for linting",
            workspace=ctx.settings.workspace,
            source="user",
        )
        ctx.display.show_info.assert_called_once()

    def test_learn_no_args_shows_rules(self) -> None:
        mock_mem = MagicMock()
        mock_mem.get_rules.return_value = []
        ctx = _make_ctx(memory_store=mock_mem)
        _cmd_learn(ctx, "")
        mock_mem.get_rules.assert_called_once()

    def test_learn_no_args_with_existing_rules(self) -> None:
        mock_rule = MagicMock()
        mock_rule.id = "r1"
        mock_rule.rule = "Always type-hint"
        mock_rule.workspace = "/project"
        mock_mem = MagicMock()
        mock_mem.get_rules.return_value = [mock_rule]
        ctx = _make_ctx(memory_store=mock_mem)
        _cmd_learn(ctx, "")
        ctx.display.console.print.assert_called()

    def test_learn_forget_rule(self) -> None:
        mock_mem = MagicMock()
        mock_mem.forget.return_value = True
        ctx = _make_ctx(memory_store=mock_mem)
        _cmd_learn(ctx, "--forget rule_id_123")
        mock_mem.forget.assert_called_once_with("rule_id_123")
        ctx.display.show_info.assert_called_once()

    def test_learn_forget_nonexistent(self) -> None:
        mock_mem = MagicMock()
        mock_mem.forget.return_value = False
        ctx = _make_ctx(memory_store=mock_mem)
        _cmd_learn(ctx, "--forget nope")
        ctx.display.show_error.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: _cmd_map
# ---------------------------------------------------------------------------

class TestCmdMap:
    """Map command handler."""

    def test_error_without_repo_map(self) -> None:
        ctx = _make_ctx(repo_map=None)
        result = _cmd_map(ctx, "")
        assert result is True
        ctx.display.show_error.assert_called_once()

    def test_shows_map(self) -> None:
        mock_map = MagicMock()
        mock_map.format.return_value = "src/\n  app.py\n  utils.py"
        ctx = _make_ctx(repo_map=mock_map)
        _cmd_map(ctx, "")
        mock_map.format.assert_called_once_with(max_tokens=2000)

    def test_custom_token_budget(self) -> None:
        mock_map = MagicMock()
        mock_map.format.return_value = "tree"
        ctx = _make_ctx(repo_map=mock_map)
        _cmd_map(ctx, "5000")
        mock_map.format.assert_called_once_with(max_tokens=5000)

    def test_refresh_flag(self) -> None:
        mock_map = MagicMock()
        mock_map.index.return_value = 42
        ctx = _make_ctx(repo_map=mock_map)
        _cmd_map(ctx, "--refresh")
        mock_map.index.assert_called_once()

    def test_empty_map_shows_info(self) -> None:
        mock_map = MagicMock()
        mock_map.format.return_value = ""
        ctx = _make_ctx(repo_map=mock_map)
        _cmd_map(ctx, "")
        ctx.display.show_info.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: _cmd_budget
# ---------------------------------------------------------------------------

class TestCmdBudget:
    """Budget command handler."""

    def test_set_budget(self) -> None:
        ctx = _make_ctx()
        result = _cmd_budget(ctx, "5.00")
        assert result is True
        assert ctx.settings.max_cost_per_session == 5.0

    def test_set_budget_with_dollar_sign(self) -> None:
        ctx = _make_ctx()
        _cmd_budget(ctx, "$10")
        assert ctx.settings.max_cost_per_session == 10.0

    def test_show_budget_when_set(self) -> None:
        ctx = _make_ctx()
        ctx.settings.max_cost_per_session = 2.0
        _cmd_budget(ctx, "")
        ctx.display.show_info.assert_called_once()
        msg = ctx.display.show_info.call_args[0][0]
        assert "$2.00" in msg

    def test_show_budget_when_unlimited(self) -> None:
        ctx = _make_ctx()
        ctx.settings.max_cost_per_session = 0.0
        _cmd_budget(ctx, "")
        ctx.display.show_info.assert_called_once()
        msg = ctx.display.show_info.call_args[0][0]
        assert "No budget" in msg

    def test_invalid_budget_shows_error(self) -> None:
        ctx = _make_ctx()
        _cmd_budget(ctx, "abc")
        ctx.display.show_error.assert_called_once()

    def test_show_budget_with_logger(self) -> None:
        mock_logger = MagicMock()
        mock_logger.cost_tracker.total_cost = 0.5
        ctx = _make_ctx(logger=mock_logger)
        ctx.settings.max_cost_per_session = 5.0
        _cmd_budget(ctx, "")
        msg = ctx.display.show_info.call_args[0][0]
        assert "Spent" in msg


# ---------------------------------------------------------------------------
# Tests: _cmd_cost
# ---------------------------------------------------------------------------

class TestCmdCost:
    """Cost command handler."""

    def test_error_without_logger(self) -> None:
        ctx = _make_ctx(logger=None)
        result = _cmd_cost(ctx, "")
        assert result is True
        ctx.display.show_error.assert_called_once()

    def test_shows_cost_panel(self) -> None:
        mock_logger = MagicMock()
        mock_logger.cost_tracker.summary.return_value = {
            "tokens_in": 1000,
            "tokens_out": 500,
            "total_tokens": 1500,
            "estimated_cost": 0.01,
            "llm_calls": 3,
            "tool_calls": 5,
            "is_free": False,
            "per_model": {},
        }
        ctx = _make_ctx(logger=mock_logger)
        result = _cmd_cost(ctx, "")
        assert result is True
        ctx.display.console.print.assert_called()

    def test_free_mode_label(self) -> None:
        mock_logger = MagicMock()
        mock_logger.cost_tracker.summary.return_value = {
            "tokens_in": 100,
            "tokens_out": 50,
            "total_tokens": 150,
            "estimated_cost": 0.0,
            "llm_calls": 1,
            "tool_calls": 0,
            "is_free": True,
            "per_model": {},
        }
        ctx = _make_ctx(logger=mock_logger)
        _cmd_cost(ctx, "")
        # Should render without error
        ctx.display.console.print.assert_called()


# ---------------------------------------------------------------------------
# Tests: _cmd_context
# ---------------------------------------------------------------------------

class TestCmdContext:
    """Context usage command handler."""

    def test_error_without_context_manager(self) -> None:
        ctx = _make_ctx(context_manager=None)
        result = _cmd_context(ctx, "")
        assert result is True
        ctx.display.show_error.assert_called_once()

    @patch("unjess.system_prompt.build_system_prompt", return_value="mock prompt")
    def test_shows_context_breakdown(self, mock_build: MagicMock) -> None:
        mock_cm = MagicMock()
        mock_cm.get_context_breakdown.return_value = {
            "context_window": 128000,
            "response_reserve": 4096,
            "system_prompt_tokens": 2000,
            "conversation_tokens": 5000,
            "message_count": 10,
            "available": 116904,
            "utilization_pct": 8,
        }
        ctx = _make_ctx(context_manager=mock_cm)
        # Ensure the agent has the attributes the function accesses
        ctx.agent.conversation = []
        ctx.agent.tool_registry = MagicMock()
        result = _cmd_context(ctx, "")
        assert result is True


# ---------------------------------------------------------------------------
# Tests: _cmd_goal
# ---------------------------------------------------------------------------

class TestCmdGoal:
    """Goal mode command handler."""

    def test_error_without_task(self) -> None:
        ctx = _make_ctx()
        result = _cmd_goal(ctx, "")
        assert result is True
        ctx.display.show_error.assert_called_once()

    def test_runs_agent_with_task(self) -> None:
        ctx = _make_ctx()
        result = _cmd_goal(ctx, "Refactor the entire codebase")
        assert result is True
        ctx.agent.run.assert_called_once_with("Refactor the entire codebase")

    def test_resets_goal_mode_after_run(self) -> None:
        ctx = _make_ctx()
        _cmd_goal(ctx, "Some task")
        assert ctx.agent._goal_mode is False

    def test_resets_goal_mode_on_error(self) -> None:
        ctx = _make_ctx()
        ctx.agent.run.side_effect = RuntimeError("fail")
        with pytest.raises(RuntimeError):
            _cmd_goal(ctx, "Task that fails")
        assert ctx.agent._goal_mode is False


# ---------------------------------------------------------------------------
# Tests: _cmd_browser
# ---------------------------------------------------------------------------

class TestCmdBrowser:
    """Browser command handler."""

    def test_with_url_navigates(self) -> None:
        ctx = _make_ctx()
        result = _cmd_browser(ctx, "https://example.com")
        assert result is True
        ctx.agent.run.assert_called_once()
        assert "example.com" in ctx.agent.run.call_args[0][0]

    def test_without_args_shows_info(self) -> None:
        ctx = _make_ctx()
        result = _cmd_browser(ctx, "")
        assert result is True
        assert ctx.display.show_info.call_count == 2


# ---------------------------------------------------------------------------
# Tests: _cmd_diff
# ---------------------------------------------------------------------------

class TestCmdDiff:
    """Diff command handler."""

    def test_error_without_undo_manager(self) -> None:
        ctx = _make_ctx(undo_manager=None)
        result = _cmd_diff(ctx, "")
        assert result is True
        ctx.display.show_error.assert_called_once()

    def test_no_changes_message(self) -> None:
        mock_undo = MagicMock()
        mock_undo.list_checkpoints.return_value = []
        ctx = _make_ctx(undo_manager=mock_undo)
        result = _cmd_diff(ctx, "")
        assert result is True
        ctx.display.show_info.assert_called_once()
        assert "No changes" in ctx.display.show_info.call_args[0][0]


# ---------------------------------------------------------------------------
# Tests: _cmd_test
# ---------------------------------------------------------------------------

class TestCmdTest:
    """Test runner command handler."""

    def test_with_custom_command(self) -> None:
        ctx = _make_ctx()
        result = _cmd_test(ctx, "pytest -v")
        assert result is True
        ctx.agent.run.assert_called_once()
        assert "pytest -v" in ctx.agent.run.call_args[0][0]

    def test_without_args_auto_detects(self) -> None:
        ctx = _make_ctx()
        result = _cmd_test(ctx, "")
        assert result is True
        ctx.agent.run.assert_called_once()
        assert "test" in ctx.agent.run.call_args[0][0].lower()


# ---------------------------------------------------------------------------
# Tests: _cmd_schedule
# ---------------------------------------------------------------------------

class TestCmdSchedule:
    """Schedule command handler."""

    def test_no_args_shows_usage(self) -> None:
        ctx = _make_ctx()
        result = _cmd_schedule(ctx, "")
        assert result is True
        ctx.display.show_info.assert_called_once()
        assert "Usage" in ctx.display.show_info.call_args[0][0]


# ---------------------------------------------------------------------------
# Tests: _cmd_model (quick switch)
# ---------------------------------------------------------------------------

class TestCmdModel:
    """Model switching command handler."""

    def test_quick_switch_with_name(self) -> None:
        from unjess.commands import _cmd_model
        ctx = _make_ctx()
        ctx.settings.model = "old-model"
        result = _cmd_model(ctx, "gpt-4o")
        assert result is True
        assert ctx.settings.model == "gpt-4o"
        ctx.display.show_info.assert_called_once()
        msg = ctx.display.show_info.call_args[0][0]
        assert "old-model" in msg
        assert "gpt-4o" in msg

    def test_quick_switch_updates_context_manager(self) -> None:
        from unjess.commands import _cmd_model
        mock_cm = MagicMock()
        ctx = _make_ctx(context_manager=mock_cm)
        ctx.settings.model = "old"
        _cmd_model(ctx, "new-model")
        assert mock_cm.model == "new-model"


# ---------------------------------------------------------------------------
# Tests: CommandContext dataclass
# ---------------------------------------------------------------------------

class TestCommandContext:
    """CommandContext dataclass."""

    def test_required_fields(self) -> None:
        ctx = CommandContext(
            agent=MagicMock(),
            settings=MagicMock(),
            display=MagicMock(),
            router=MagicMock(),
        )
        assert ctx.undo_manager is None
        assert ctx.context_manager is None
        assert ctx.logger is None
        assert ctx.repo_map is None
        assert ctx.memory_store is None
        assert ctx.input_handler is None

    def test_all_fields(self) -> None:
        ctx = CommandContext(
            agent=MagicMock(),
            settings=MagicMock(),
            display=MagicMock(),
            router=MagicMock(),
            undo_manager=MagicMock(),
            context_manager=MagicMock(),
            logger=MagicMock(),
            repo_map=MagicMock(),
            memory_store=MagicMock(),
            input_handler=MagicMock(),
        )
        assert ctx.undo_manager is not None
        assert ctx.input_handler is not None


# ---------------------------------------------------------------------------
# Tests: register_builtin_commands idempotency
# ---------------------------------------------------------------------------

class TestRegisterBuiltinCommands:
    """Built-in command registration."""

    def test_idempotent(self) -> None:
        count_before = len(get_all_commands())
        register_builtin_commands()
        count_after = len(get_all_commands())
        assert count_before == count_after

    def test_all_commands_have_handlers(self) -> None:
        for name, cmd in get_all_commands().items():
            assert callable(cmd.handler), f"{name} handler is not callable"
            assert cmd.name == name
            assert isinstance(cmd.description, str)
            assert len(cmd.description) > 0
