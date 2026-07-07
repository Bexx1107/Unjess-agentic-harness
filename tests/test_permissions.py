"""Tests for unjess.permissions — PermissionManager and _FallbackTerminalInput."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from unjess.permissions import PermissionManager, _FallbackTerminalInput

# Re-use the MockInput from conftest (injected via mock_input fixture).


# ---------------------------------------------------------------------------
# PermissionManager.check — always-approve tools
# ---------------------------------------------------------------------------


class TestCheckAlwaysApprove:
    """Tools in _ALWAYS_APPROVE should pass check() unconditionally."""

    @pytest.mark.parametrize("tool", ["read_file", "list_dir", "grep_search"])
    def test_always_approve_tools_pass_check(self, tool: str) -> None:
        pm = PermissionManager()
        assert pm.check(tool, {}) is True

    @pytest.mark.parametrize("tool", ["read_file", "list_dir", "grep_search"])
    def test_always_approve_tools_pass_with_arbitrary_args(self, tool: str) -> None:
        pm = PermissionManager()
        assert pm.check(tool, {"path": "/some/file", "extra": 42}) is True


# ---------------------------------------------------------------------------
# PermissionManager.check — workspace-approve tools
# ---------------------------------------------------------------------------


class TestCheckWorkspaceApprove:
    """Tools in _WORKSPACE_APPROVE should pass check() (sandbox handled elsewhere)."""

    @pytest.mark.parametrize("tool", ["write_file", "edit_file"])
    def test_workspace_approve_tools_pass_check(self, tool: str) -> None:
        pm = PermissionManager()
        assert pm.check(tool, {}) is True

    @pytest.mark.parametrize("tool", ["write_file", "edit_file"])
    def test_workspace_approve_tools_pass_with_args(self, tool: str) -> None:
        pm = PermissionManager()
        assert pm.check(tool, {"path": "/workspace/foo.py", "content": "x"}) is True


# ---------------------------------------------------------------------------
# PermissionManager.check — run_command
# ---------------------------------------------------------------------------


class TestCheckRunCommand:
    """run_command requires explicit approval unless overridden."""

    def test_run_command_fails_check_by_default(self) -> None:
        pm = PermissionManager()
        assert pm.check("run_command", {"command": "ls"}) is False

    def test_run_command_passes_after_approve_all_commands(self) -> None:
        pm = PermissionManager()
        pm.approve_all_commands()
        assert pm.check("run_command", {"command": "rm -rf /"}) is True

    def test_run_command_passes_with_approved_prefix(self) -> None:
        pm = PermissionManager()
        pm._auto_approved_commands.add("git")
        assert pm.check("run_command", {"command": "git status"}) is True

    def test_run_command_fails_with_unapproved_prefix(self) -> None:
        pm = PermissionManager()
        pm._auto_approved_commands.add("git")
        assert pm.check("run_command", {"command": "rm -rf /"}) is False

    def test_run_command_prefix_match_is_startswith(self) -> None:
        pm = PermissionManager()
        pm._auto_approved_commands.add("python")
        # startswith means "python" also matches "python3"
        assert pm.check("run_command", {"command": "python3 script.py"}) is True
        assert pm.check("run_command", {"command": "python script.py"}) is True
        assert pm.check("run_command", {"command": "ruby script.rb"}) is False

    def test_run_command_empty_command(self) -> None:
        pm = PermissionManager()
        assert pm.check("run_command", {"command": ""}) is False

    def test_run_command_missing_command_key(self) -> None:
        pm = PermissionManager()
        assert pm.check("run_command", {}) is False

    def test_run_command_approve_all_overrides_even_without_prefix(self) -> None:
        pm = PermissionManager()
        pm.approve_all_commands()
        assert pm.check("run_command", {}) is True


# ---------------------------------------------------------------------------
# PermissionManager.check — unknown tools
# ---------------------------------------------------------------------------


class TestCheckUnknownTools:
    """Unknown / unrecognised tools should require approval."""

    @pytest.mark.parametrize("tool", ["deploy", "custom_tool", "send_email", ""])
    def test_unknown_tool_fails_check(self, tool: str) -> None:
        pm = PermissionManager()
        assert pm.check(tool, {}) is False


# ---------------------------------------------------------------------------
# PermissionManager.request_approval
# ---------------------------------------------------------------------------


class TestRequestApproval:
    """request_approval delegates to the input handler and stores prefix on 'always'."""

    def test_request_approval_approved(self, mock_input: Any) -> None:
        pm = PermissionManager(input_handler=mock_input)
        result = pm.request_approval("run_command", {"command": "ls -la"})
        assert result is True
        assert len(mock_input.approval_requests) == 1
        assert mock_input.approval_requests[0] == ("run_command", {"command": "ls -la"})

    def test_request_approval_denied(self) -> None:
        class DenyInput:
            """Always denies."""
            def ask_approval(self, tool_name: str, args: dict[str, Any]) -> tuple[bool, bool]:
                return (False, False)

        pm = PermissionManager(input_handler=DenyInput())  # type: ignore[arg-type]
        result = pm.request_approval("run_command", {"command": "rm -rf /"})
        assert result is False

    def test_request_approval_always_stores_prefix(self) -> None:
        class AlwaysInput:
            """Returns (True, True) — always approve."""

            def ask_approval(self, tool_name: str, args: dict[str, Any]) -> tuple[bool, bool]:
                return (True, True)

        pm = PermissionManager(input_handler=AlwaysInput())  # type: ignore[arg-type]
        pm.request_approval("run_command", {"command": "git push origin main"})
        assert "git" in pm.auto_approved

    def test_request_approval_always_with_empty_command(self) -> None:
        class AlwaysInput:
            def ask_approval(self, tool_name: str, args: dict[str, Any]) -> tuple[bool, bool]:
                return (True, True)

        pm = PermissionManager(input_handler=AlwaysInput())  # type: ignore[arg-type]
        pm.request_approval("run_command", {"command": ""})
        # Empty string split produces empty prefix — should NOT add empty string
        assert pm.auto_approved == set()

    def test_request_approval_always_non_run_command_tool(self) -> None:
        class AlwaysInput:
            def ask_approval(self, tool_name: str, args: dict[str, Any]) -> tuple[bool, bool]:
                return (True, True)

        pm = PermissionManager(input_handler=AlwaysInput())  # type: ignore[arg-type]
        pm.request_approval("deploy", {"target": "prod"})
        # 'always' only stores prefix for run_command
        assert pm.auto_approved == set()

    def test_request_approval_logs_all_requests(self, mock_input: Any) -> None:
        pm = PermissionManager(input_handler=mock_input)
        pm.request_approval("run_command", {"command": "echo hello"})
        pm.request_approval("deploy", {"env": "staging"})
        assert len(mock_input.approval_requests) == 2

    def test_auto_approved_prefix_makes_future_check_pass(self) -> None:
        class AlwaysInput:
            def ask_approval(self, tool_name: str, args: dict[str, Any]) -> tuple[bool, bool]:
                return (True, True)

        pm = PermissionManager(input_handler=AlwaysInput())  # type: ignore[arg-type]
        # First time — needs approval
        assert pm.check("run_command", {"command": "npm install"}) is False
        pm.request_approval("run_command", {"command": "npm install"})
        # Now prefix "npm" is remembered
        assert pm.check("run_command", {"command": "npm test"}) is True
        assert pm.check("run_command", {"command": "npm run build"}) is True


# ---------------------------------------------------------------------------
# PermissionManager.approve_all_commands
# ---------------------------------------------------------------------------


class TestApproveAllCommands:
    """approve_all_commands globally disables command prompts."""

    def test_approve_all_commands_flag(self) -> None:
        pm = PermissionManager()
        assert pm._always_approve_all_commands is False
        pm.approve_all_commands()
        assert pm._always_approve_all_commands is True

    def test_approve_all_commands_applies_to_any_command(self) -> None:
        pm = PermissionManager()
        pm.approve_all_commands()
        assert pm.check("run_command", {"command": "curl evil.com"}) is True
        assert pm.check("run_command", {"command": "sudo rm -rf /"}) is True

    def test_approve_all_commands_does_not_affect_unknown_tools(self) -> None:
        pm = PermissionManager()
        pm.approve_all_commands()
        assert pm.check("deploy", {}) is False


# ---------------------------------------------------------------------------
# PermissionManager.auto_approved property
# ---------------------------------------------------------------------------


class TestAutoApprovedProperty:
    """The auto_approved property returns a copy of approved prefixes."""

    def test_auto_approved_initially_empty(self) -> None:
        pm = PermissionManager()
        assert pm.auto_approved == set()

    def test_auto_approved_returns_copy(self) -> None:
        pm = PermissionManager()
        pm._auto_approved_commands.add("git")
        result = pm.auto_approved
        result.add("malicious")
        # Internal set should be unchanged
        assert "malicious" not in pm.auto_approved
        assert "git" in pm.auto_approved

    def test_auto_approved_reflects_added_prefixes(self) -> None:
        pm = PermissionManager()
        pm._auto_approved_commands.add("python")
        pm._auto_approved_commands.add("pip")
        assert pm.auto_approved == {"python", "pip"}


# ---------------------------------------------------------------------------
# PermissionManager constructor
# ---------------------------------------------------------------------------


class TestPermissionManagerInit:
    """Constructor handles default and custom arguments."""

    def test_default_console_created(self) -> None:
        pm = PermissionManager()
        assert isinstance(pm._console, Console)

    def test_custom_console(self) -> None:
        console = Console()
        pm = PermissionManager(console=console)
        assert pm._console is console

    def test_default_fallback_input(self) -> None:
        pm = PermissionManager()
        assert isinstance(pm._input_handler, _FallbackTerminalInput)

    def test_custom_input_handler(self, mock_input: Any) -> None:
        pm = PermissionManager(input_handler=mock_input)
        assert pm._input_handler is mock_input


# ---------------------------------------------------------------------------
# _FallbackTerminalInput
# ---------------------------------------------------------------------------


class TestFallbackTerminalInput:
    """The built-in terminal prompt for permission approval."""

    def test_run_command_yes(self) -> None:
        console = MagicMock(spec=Console)
        fti = _FallbackTerminalInput(console)
        with patch("builtins.input", return_value="y"):
            approved, always = fti.ask_approval("run_command", {"command": "ls"})
        assert approved is True
        assert always is False

    def test_run_command_no(self) -> None:
        console = MagicMock(spec=Console)
        fti = _FallbackTerminalInput(console)
        with patch("builtins.input", return_value="n"):
            approved, always = fti.ask_approval("run_command", {"command": "ls"})
        assert approved is False
        assert always is False

    def test_run_command_always(self) -> None:
        console = MagicMock(spec=Console)
        fti = _FallbackTerminalInput(console)
        with patch("builtins.input", return_value="always"):
            approved, always = fti.ask_approval("run_command", {"command": "ls"})
        assert approved is True
        assert always is True

    def test_shorthand_responses(self) -> None:
        console = MagicMock(spec=Console)
        fti = _FallbackTerminalInput(console)
        for resp, expected in [("yes", (True, False)), ("no", (False, False)), ("a", (True, True))]:
            with patch("builtins.input", return_value=resp):
                assert fti.ask_approval("run_command", {"command": "ls"}) == expected

    def test_invalid_then_valid_response(self) -> None:
        console = MagicMock(spec=Console)
        fti = _FallbackTerminalInput(console)
        with patch("builtins.input", side_effect=["maybe", "sure", "y"]):
            approved, always = fti.ask_approval("run_command", {"command": "ls"})
        assert approved is True
        assert always is False

    def test_eof_error_returns_false(self) -> None:
        console = MagicMock(spec=Console)
        fti = _FallbackTerminalInput(console)
        with patch("builtins.input", side_effect=EOFError):
            approved, always = fti.ask_approval("run_command", {"command": "ls"})
        assert approved is False
        assert always is False

    def test_keyboard_interrupt_returns_false(self) -> None:
        console = MagicMock(spec=Console)
        fti = _FallbackTerminalInput(console)
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            approved, always = fti.ask_approval("run_command", {"command": "ls"})
        assert approved is False
        assert always is False

    def test_non_run_command_tool_preview(self) -> None:
        console = MagicMock(spec=Console)
        fti = _FallbackTerminalInput(console)
        with patch("builtins.input", return_value="y"):
            approved, _ = fti.ask_approval("deploy", {"env": "prod"})
        assert approved is True
        # Panel was printed with generic preview (not command-specific)
        console.print.assert_called()

    def test_run_command_missing_command_key(self) -> None:
        console = MagicMock(spec=Console)
        fti = _FallbackTerminalInput(console)
        with patch("builtins.input", return_value="y"):
            approved, _ = fti.ask_approval("run_command", {})
        assert approved is True

    def test_whitespace_stripped_from_input(self) -> None:
        console = MagicMock(spec=Console)
        fti = _FallbackTerminalInput(console)
        with patch("builtins.input", return_value="  Y  "):
            approved, always = fti.ask_approval("run_command", {"command": "ls"})
        # 'Y' is not recognised — only lowercase matches
        # Actually .strip().lower() → 'y' which IS recognised
        assert approved is True
        assert always is False


# ---------------------------------------------------------------------------
# Integration: full approval flow
# ---------------------------------------------------------------------------


class TestIntegrationApprovalFlow:
    """End-to-end flows combining check → request_approval → check."""

    def test_full_flow_run_command_deny_then_approve(self) -> None:
        call_count = 0

        class SequencedInput:
            def ask_approval(self, tool_name: str, args: dict[str, Any]) -> tuple[bool, bool]:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return (False, False)  # denied
                return (True, True)  # always approve

        pm = PermissionManager(input_handler=SequencedInput())  # type: ignore[arg-type]
        args = {"command": "pytest -x"}

        assert pm.check("run_command", args) is False
        denied = pm.request_approval("run_command", args)
        assert denied is False
        assert pm.check("run_command", args) is False  # still not auto-approved

        approved = pm.request_approval("run_command", args)
        assert approved is True
        assert "pytest" in pm.auto_approved
        assert pm.check("run_command", args) is True  # now auto-approved

    def test_multiple_prefixes_accumulated(self) -> None:
        class AlwaysInput:
            def ask_approval(self, tool_name: str, args: dict[str, Any]) -> tuple[bool, bool]:
                return (True, True)

        pm = PermissionManager(input_handler=AlwaysInput())  # type: ignore[arg-type]
        pm.request_approval("run_command", {"command": "git status"})
        pm.request_approval("run_command", {"command": "pip install foo"})
        pm.request_approval("run_command", {"command": "python main.py"})

        assert pm.auto_approved == {"git", "pip", "python"}
        assert pm.check("run_command", {"command": "git push"}) is True
        assert pm.check("run_command", {"command": "pip freeze"}) is True
        assert pm.check("run_command", {"command": "cargo build"}) is False
