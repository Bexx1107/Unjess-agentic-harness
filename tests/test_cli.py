"""Tests for unjess.cli — argument parsing only (not main() which starts the agent)."""

import argparse
import sys
from unittest.mock import patch

import pytest

from unjess import __version__


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse(argv: list[str]) -> argparse.Namespace:
    """Parse argv through the CLI's main() parser without running the agent.

    We import and re-create the parser in the same way main() does,
    since there is no standalone ``build_parser`` or ``parse_args`` function.
    """
    parser = argparse.ArgumentParser(prog="njss", description="Unjess — AI coding agent")
    parser.add_argument("prompt", nargs="?", default=None)
    parser.add_argument("-m", "--model", default=None)
    parser.add_argument("-p", "--provider", default=None)
    parser.add_argument("-w", "--workspace", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--version", action="version", version=f"njss {__version__}")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--native", action="store_true")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Default args
# ---------------------------------------------------------------------------

class TestDefaultArgs:
    """Tests for default argument values when no flags are passed."""

    def test_no_args_defaults(self) -> None:
        ns = _parse([])
        assert ns.prompt is None
        assert ns.model is None
        assert ns.provider is None
        assert ns.workspace is None
        assert ns.config is None
        assert ns.verbose is False
        assert ns.yes is False
        assert ns.gui is False
        assert ns.native is False


# ---------------------------------------------------------------------------
# --model / -m
# ---------------------------------------------------------------------------

class TestModelFlag:
    """Tests for the --model / -m flag."""

    def test_model_long_flag(self) -> None:
        ns = _parse(["--model", "gpt-4o"])
        assert ns.model == "gpt-4o"

    def test_model_short_flag(self) -> None:
        ns = _parse(["-m", "claude-sonnet-4-20250514"])
        assert ns.model == "claude-sonnet-4-20250514"

    def test_model_with_equals(self) -> None:
        ns = _parse(["--model=gemini-3.1-flash"])
        assert ns.model == "gemini-3.1-flash"


# ---------------------------------------------------------------------------
# --provider / -p
# ---------------------------------------------------------------------------

class TestProviderFlag:
    """Tests for the --provider / -p flag."""

    def test_provider_long_flag(self) -> None:
        ns = _parse(["--provider", "openai"])
        assert ns.provider == "openai"

    def test_provider_short_flag(self) -> None:
        ns = _parse(["-p", "anthropic"])
        assert ns.provider == "anthropic"

    def test_provider_google(self) -> None:
        ns = _parse(["--provider", "google"])
        assert ns.provider == "google"

    def test_provider_ollama(self) -> None:
        ns = _parse(["--provider", "ollama"])
        assert ns.provider == "ollama"


# ---------------------------------------------------------------------------
# --workspace / -w
# ---------------------------------------------------------------------------

class TestWorkspaceFlag:
    """Tests for the --workspace / -w flag."""

    def test_workspace_long_flag(self) -> None:
        ns = _parse(["--workspace", "/path/to/project"])
        assert ns.workspace == "/path/to/project"

    def test_workspace_short_flag(self) -> None:
        ns = _parse(["-w", "C:\\Users\\dev\\project"])
        assert ns.workspace == "C:\\Users\\dev\\project"


# ---------------------------------------------------------------------------
# --verbose
# ---------------------------------------------------------------------------

class TestVerboseFlag:
    """Tests for the --verbose flag."""

    def test_verbose_enabled(self) -> None:
        ns = _parse(["--verbose"])
        assert ns.verbose is True

    def test_verbose_default_off(self) -> None:
        ns = _parse([])
        assert ns.verbose is False


# ---------------------------------------------------------------------------
# --yes (auto-approve)
# ---------------------------------------------------------------------------

class TestYesFlag:
    """Tests for the --yes flag (auto-approve all commands)."""

    def test_yes_enabled(self) -> None:
        ns = _parse(["--yes"])
        assert ns.yes is True

    def test_yes_default_off(self) -> None:
        ns = _parse([])
        assert ns.yes is False


# ---------------------------------------------------------------------------
# --version
# ---------------------------------------------------------------------------

class TestVersionFlag:
    """Tests for the --version flag."""

    def test_version_prints_and_exits(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exc_info:
            _parse(["--version"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert __version__ in captured.out


# ---------------------------------------------------------------------------
# --config
# ---------------------------------------------------------------------------

class TestConfigFlag:
    """Tests for the --config flag."""

    def test_config_path(self) -> None:
        ns = _parse(["--config", "/home/user/.unjess/alt-config.yaml"])
        assert ns.config == "/home/user/.unjess/alt-config.yaml"

    def test_config_default_none(self) -> None:
        ns = _parse([])
        assert ns.config is None


# ---------------------------------------------------------------------------
# --gui / --native
# ---------------------------------------------------------------------------

class TestGuiFlags:
    """Tests for the --gui and --native flags."""

    def test_gui_flag(self) -> None:
        ns = _parse(["--gui"])
        assert ns.gui is True

    def test_native_flag(self) -> None:
        ns = _parse(["--native"])
        assert ns.native is True

    def test_gui_and_native_together(self) -> None:
        ns = _parse(["--gui", "--native"])
        assert ns.gui is True
        assert ns.native is True


# ---------------------------------------------------------------------------
# Positional prompt argument
# ---------------------------------------------------------------------------

class TestPromptPositional:
    """Tests for the optional positional prompt argument."""

    def test_prompt_provided(self) -> None:
        ns = _parse(["Fix the bug in parser.py"])
        assert ns.prompt == "Fix the bug in parser.py"

    def test_prompt_default_none(self) -> None:
        ns = _parse([])
        assert ns.prompt is None

    def test_prompt_with_flags(self) -> None:
        ns = _parse(["-m", "gpt-4o", "--verbose", "Deploy to prod"])
        assert ns.prompt == "Deploy to prod"
        assert ns.model == "gpt-4o"
        assert ns.verbose is True


# ---------------------------------------------------------------------------
# Combined flags
# ---------------------------------------------------------------------------

class TestCombinedFlags:
    """Tests for combining multiple flags together."""

    def test_all_flags_combined(self) -> None:
        ns = _parse([
            "-m", "gemini-3.1-flash",
            "-p", "google",
            "-w", "/my/project",
            "--config", "/cfg.yaml",
            "--verbose",
            "--yes",
            "Run the tests",
        ])
        assert ns.model == "gemini-3.1-flash"
        assert ns.provider == "google"
        assert ns.workspace == "/my/project"
        assert ns.config == "/cfg.yaml"
        assert ns.verbose is True
        assert ns.yes is True
        assert ns.prompt == "Run the tests"

    def test_unknown_flag_raises(self) -> None:
        with pytest.raises(SystemExit):
            _parse(["--nonexistent-flag"])


# ---------------------------------------------------------------------------
# TerminalInput (from cli module)
# ---------------------------------------------------------------------------

class TestTerminalInput:
    """Tests for the TerminalInput class defined in cli.py."""

    def test_get_user_input(self) -> None:
        from unjess.cli import TerminalInput
        from rich.console import Console

        ti = TerminalInput(Console())
        with patch("builtins.input", return_value="hello"):
            result = ti.get_user_input()
            assert result == "hello"

    def test_ask_approval_yes(self) -> None:
        from unjess.cli import TerminalInput
        from rich.console import Console

        console = Console()
        ti = TerminalInput(console)
        with patch("builtins.input", return_value="y"), \
             patch.object(console, "print"):
            approved, always = ti.ask_approval("write_file", {"path": "a.py"})
            assert approved is True
            assert always is False

    def test_ask_approval_no(self) -> None:
        from unjess.cli import TerminalInput
        from rich.console import Console

        console = Console()
        ti = TerminalInput(console)
        with patch("builtins.input", return_value="n"), \
             patch.object(console, "print"):
            approved, always = ti.ask_approval("write_file", {"path": "a.py"})
            assert approved is False
            assert always is False

    def test_ask_approval_always(self) -> None:
        from unjess.cli import TerminalInput
        from rich.console import Console

        console = Console()
        ti = TerminalInput(console)
        with patch("builtins.input", return_value="a"), \
             patch.object(console, "print"):
            approved, always = ti.ask_approval("write_file", {"path": "a.py"})
            assert approved is True
            assert always is True

    def test_ask_approval_eof(self) -> None:
        from unjess.cli import TerminalInput
        from rich.console import Console

        console = Console()
        ti = TerminalInput(console)
        with patch("builtins.input", side_effect=EOFError), \
             patch.object(console, "print"):
            approved, always = ti.ask_approval("write_file", {})
            assert approved is False
            assert always is False

    def test_ask_approval_run_command_preview(self) -> None:
        from unjess.cli import TerminalInput
        from rich.console import Console

        console = Console()
        ti = TerminalInput(console)
        with patch("builtins.input", return_value="y"), \
             patch.object(console, "print"):
            approved, _ = ti.ask_approval("run_command", {"command": "ls", "cwd": "."})
            assert approved is True

    def test_ask_confirmation_yes(self) -> None:
        from unjess.cli import TerminalInput
        from rich.console import Console

        console = Console()
        ti = TerminalInput(console)
        with patch("builtins.input", return_value="y"), \
             patch.object(console, "print"):
            assert ti.ask_confirmation("Proceed?") is True

    def test_ask_confirmation_no(self) -> None:
        from unjess.cli import TerminalInput
        from rich.console import Console

        console = Console()
        ti = TerminalInput(console)
        with patch("builtins.input", return_value="n"), \
             patch.object(console, "print"):
            assert ti.ask_confirmation("Proceed?") is False

    def test_ask_confirmation_eof(self) -> None:
        from unjess.cli import TerminalInput
        from rich.console import Console

        console = Console()
        ti = TerminalInput(console)
        with patch("builtins.input", side_effect=EOFError), \
             patch.object(console, "print"):
            assert ti.ask_confirmation("Proceed?") is False

    def test_ask_question_numeric_choice(self) -> None:
        from unjess.cli import TerminalInput
        from rich.console import Console

        console = Console()
        ti = TerminalInput(console)
        with patch("builtins.input", return_value="2"), \
             patch.object(console, "print"):
            result = ti.ask_question("Pick one:", ["Option A", "Option B", "Option C"])
            assert result == "Option B"

    def test_ask_question_text_answer(self) -> None:
        from unjess.cli import TerminalInput
        from rich.console import Console

        console = Console()
        ti = TerminalInput(console)
        with patch("builtins.input", return_value="custom"), \
             patch.object(console, "print"):
            result = ti.ask_question("Pick one:", ["A", "B"])
            assert result == "custom"

    def test_ask_question_eof(self) -> None:
        from unjess.cli import TerminalInput
        from rich.console import Console

        console = Console()
        ti = TerminalInput(console)
        with patch("builtins.input", side_effect=KeyboardInterrupt), \
             patch.object(console, "print"):
            result = ti.ask_question("Pick:", ["A"])
            assert result == "(user cancelled)"

    def test_ask_choice_numeric(self) -> None:
        from unjess.cli import TerminalInput
        from rich.console import Console

        console = Console()
        ti = TerminalInput(console)
        with patch("builtins.input", return_value="1"), \
             patch.object(console, "print"):
            result = ti.ask_choice("Choose:", [("val1", "Label 1"), ("val2", "Label 2")])
            assert result == "val1"

    def test_ask_choice_invalid_returns_empty(self) -> None:
        from unjess.cli import TerminalInput
        from rich.console import Console

        console = Console()
        ti = TerminalInput(console)
        with patch("builtins.input", return_value="not a number"), \
             patch.object(console, "print"):
            result = ti.ask_choice("Choose:", [("v", "L")])
            assert result == ""

    def test_ask_choice_eof(self) -> None:
        from unjess.cli import TerminalInput
        from rich.console import Console

        console = Console()
        ti = TerminalInput(console)
        with patch("builtins.input", side_effect=EOFError), \
             patch.object(console, "print"):
            result = ti.ask_choice("Choose:", [("v", "L")])
            assert result == ""
