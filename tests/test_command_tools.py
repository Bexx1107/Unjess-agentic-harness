"""Tests for unjess.tools.command_tools."""

import platform
import sys
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from unjess.sandbox import Sandbox, SandboxResult
from unjess.tools.command_tools import _run_command, _DEFAULT_TIMEOUT, _MAX_OUTPUT_CHARS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reset_file_tools_globals(workspace: Optional[Path] = None) -> None:
    """Reset module-level globals for file_tools (used by command_tools)."""
    import unjess.tools.file_tools as ft
    ft._permissions = None
    ft._approved_external_paths.clear()
    ft._workspace_ref["current"] = workspace.resolve() if workspace else None


def _make_permission_manager(auto_approve: bool = True) -> MagicMock:
    """Create a mock PermissionManager."""
    pm = MagicMock()
    pm.check.return_value = auto_approve
    pm.request_approval.return_value = auto_approve
    return pm


def _echo_cmd(text: str) -> str:
    """Return a platform-appropriate echo command."""
    if platform.system() == "Windows":
        return f'echo {text}'
    return f'echo "{text}"'


def _exit_cmd(code: int) -> str:
    """Return a platform-appropriate command that exits with a code."""
    if platform.system() == "Windows":
        return f"cmd /c exit {code}"
    return f"bash -c 'exit {code}'"


def _python_cmd(script: str) -> str:
    """Return a command to run an inline Python script."""
    # Use single quotes inside to avoid Windows double-quote issues
    return f"{sys.executable} -c \"{script}\""


# ===================================================================
# Basic execution
# ===================================================================

class TestRunCommandBasic:
    """Basic command execution tests."""

    def test_echo_returns_output(self, tmp_workspace: Path) -> None:
        _reset_file_tools_globals(tmp_workspace)
        pm = _make_permission_manager()
        result = _run_command(tmp_workspace, pm, None, _echo_cmd("hello"))
        assert "hello" in result

    def test_nonzero_exit_code_shown(self, tmp_workspace: Path) -> None:
        _reset_file_tools_globals(tmp_workspace)
        pm = _make_permission_manager()
        result = _run_command(tmp_workspace, pm, None, _exit_cmd(42))
        assert "Exit code: 42" in result

    def test_no_output_command(self, tmp_workspace: Path) -> None:
        _reset_file_tools_globals(tmp_workspace)
        pm = _make_permission_manager()
        # A command that succeeds with no output (cross-platform)
        cmd = f"{sys.executable} -c pass"
        result = _run_command(tmp_workspace, pm, None, cmd)
        assert "(no output)" in result or result.strip() == ""

    def test_stderr_included(self, tmp_workspace: Path) -> None:
        _reset_file_tools_globals(tmp_workspace)
        pm = _make_permission_manager()
        cmd = _python_cmd("import sys; sys.stderr.write('err_msg')")
        result = _run_command(tmp_workspace, pm, None, cmd)
        assert "err_msg" in result

    def test_combined_stdout_stderr(self, tmp_workspace: Path) -> None:
        _reset_file_tools_globals(tmp_workspace)
        pm = _make_permission_manager()
        cmd = _python_cmd(
            "import sys; print('out_msg'); sys.stderr.write('err_msg')"
        )
        result = _run_command(tmp_workspace, pm, None, cmd)
        assert "out_msg" in result
        assert "err_msg" in result


# ===================================================================
# Working directory
# ===================================================================

class TestRunCommandCwd:
    """Working directory validation."""

    def test_cwd_relative(self, tmp_workspace: Path) -> None:
        _reset_file_tools_globals(tmp_workspace)
        pm = _make_permission_manager()
        # Use echo to verify cwd ends with 'src' (cross-platform)
        if platform.system() == "Windows":
            cmd = "cd"
        else:
            cmd = "pwd"
        result = _run_command(tmp_workspace, pm, None, cmd, cwd="src")
        assert "src" in result

    def test_cwd_outside_workspace_blocked(self, tmp_workspace: Path) -> None:
        _reset_file_tools_globals(tmp_workspace)
        pm = _make_permission_manager()
        result = _run_command(tmp_workspace, pm, None, _echo_cmd("test"), cwd="..")
        assert "Error" in result
        assert "outside the workspace" in result

    def test_cwd_nonexistent_dir(self, tmp_workspace: Path) -> None:
        _reset_file_tools_globals(tmp_workspace)
        pm = _make_permission_manager()
        result = _run_command(
            tmp_workspace, pm, None, _echo_cmd("test"), cwd="no_such_dir"
        )
        assert "Error: Working directory not found" in result


# ===================================================================
# Sandbox blocking
# ===================================================================

class TestRunCommandSandbox:
    """Sandbox integration for dangerous command blocking."""

    def test_sandbox_blocks_dangerous_command(self, tmp_workspace: Path) -> None:
        _reset_file_tools_globals(tmp_workspace)
        pm = _make_permission_manager()
        sandbox = Sandbox()
        result = _run_command(
            tmp_workspace, pm, sandbox, "rm -rf /", timeout=5
        )
        assert "Blocked by sandbox" in result

    def test_sandbox_allows_safe_command(self, tmp_workspace: Path) -> None:
        _reset_file_tools_globals(tmp_workspace)
        pm = _make_permission_manager()
        sandbox = Sandbox()
        result = _run_command(
            tmp_workspace, pm, sandbox, _echo_cmd("safe"), timeout=5
        )
        assert "safe" in result

    def test_sandbox_with_custom_blocklist(self, tmp_workspace: Path) -> None:
        _reset_file_tools_globals(tmp_workspace)
        pm = _make_permission_manager()
        sandbox = Sandbox(extra_blocked=[r"my_dangerous_command"])
        result = _run_command(
            tmp_workspace, pm, sandbox, "my_dangerous_command --nuke"
        )
        assert "Blocked by sandbox" in result

    def test_sandbox_scrubs_secrets(self, tmp_workspace: Path) -> None:
        _reset_file_tools_globals(tmp_workspace)
        pm = _make_permission_manager()
        sandbox = Sandbox()
        # Output an OpenAI-style key
        cmd = _python_cmd(
            "print('api_key=sk-abcdefghijklmnopqrstuvwxyz1234567890')"
        )
        result = _run_command(tmp_workspace, pm, sandbox, cmd, timeout=5)
        assert "REDACTED" in result
        assert "sk-abcdefghijklmnopqrstuvwxyz1234567890" not in result

    def test_no_sandbox_passes_output_through(self, tmp_workspace: Path) -> None:
        _reset_file_tools_globals(tmp_workspace)
        pm = _make_permission_manager()
        cmd = _python_cmd("print('token=short')")
        result = _run_command(tmp_workspace, pm, None, cmd, timeout=5)
        # With no sandbox, output is not scrubbed
        assert "token=short" in result


# ===================================================================
# Permission handling
# ===================================================================

class TestRunCommandPermissions:
    """Permission gate (PermissionManager integration)."""

    def test_permission_denied(self, tmp_workspace: Path) -> None:
        _reset_file_tools_globals(tmp_workspace)
        pm = _make_permission_manager(auto_approve=False)
        result = _run_command(tmp_workspace, pm, None, _echo_cmd("hello"))
        assert "denied" in result.lower()

    def test_permission_check_called_first(self, tmp_workspace: Path) -> None:
        _reset_file_tools_globals(tmp_workspace)
        pm = _make_permission_manager()
        _run_command(tmp_workspace, pm, None, _echo_cmd("hello"))
        pm.check.assert_called_once()

    def test_approval_requested_when_not_pre_approved(
        self, tmp_workspace: Path
    ) -> None:
        _reset_file_tools_globals(tmp_workspace)
        pm = MagicMock()
        pm.check.return_value = False
        pm.request_approval.return_value = True
        _run_command(tmp_workspace, pm, None, _echo_cmd("hello"))
        pm.request_approval.assert_called_once()


# ===================================================================
# Output truncation
# ===================================================================

class TestRunCommandTruncation:
    """Output truncation for very large outputs."""

    def test_output_truncated_when_too_long(self, tmp_workspace: Path) -> None:
        _reset_file_tools_globals(tmp_workspace)
        pm = _make_permission_manager()
        # Generate output larger than _MAX_OUTPUT_CHARS
        line_count = (_MAX_OUTPUT_CHARS // 10) + 1000
        cmd = _python_cmd(
            f"[print('A' * 10) for _ in range({line_count})]"
        )
        result = _run_command(tmp_workspace, pm, None, cmd, timeout=10)
        assert "truncated" in result


# ===================================================================
# Error handling
# ===================================================================

class TestRunCommandErrors:
    """Edge cases and error conditions."""

    def test_invalid_command(self, tmp_workspace: Path) -> None:
        _reset_file_tools_globals(tmp_workspace)
        pm = _make_permission_manager()
        result = _run_command(
            tmp_workspace, pm, None,
            "this_command_does_not_exist_xyz123",
            timeout=5,
        )
        # Should get an error (either from the shell or exit code != 0)
        assert "Exit code" in result or "Error" in result or "not recognized" in result.lower() or "not found" in result.lower()


# ===================================================================
# Constants
# ===================================================================

class TestConstants:
    """Verify default constants."""

    def test_default_timeout(self) -> None:
        assert _DEFAULT_TIMEOUT == 120

    def test_max_output_chars(self) -> None:
        assert _MAX_OUTPUT_CHARS == 20_000
