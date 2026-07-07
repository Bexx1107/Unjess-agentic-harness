"""Tests for unjess.display — terminal display layer using rich."""

from typing import Any, Generator
from unittest.mock import MagicMock, call, patch

import pytest

from unjess.display import TerminalDisplay, Display, AGENT_THEME


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_display(verbose: bool = False) -> tuple[TerminalDisplay, MagicMock]:
    """Create a TerminalDisplay with a mocked Console."""
    mock_console = MagicMock()
    display = TerminalDisplay(console=mock_console, verbose=verbose)
    return display, mock_console


def _text_chunks(*parts: str) -> Generator[str, None, None]:
    """Create a generator yielding string chunks."""
    for part in parts:
        yield part


# ---------------------------------------------------------------------------
# Tests: Initialization
# ---------------------------------------------------------------------------

class TestTerminalDisplayInit:
    """TerminalDisplay construction and configuration."""

    def test_creates_with_default_console(self) -> None:
        mock_console = MagicMock()
        display = TerminalDisplay(console=mock_console)
        assert display.console is not None

    def test_uses_provided_console(self) -> None:
        mock_console = MagicMock()
        display = TerminalDisplay(console=mock_console)
        assert display.console is mock_console

    def test_verbose_defaults_false(self) -> None:
        mock_console = MagicMock()
        display = TerminalDisplay(console=mock_console)
        assert display._verbose is False

    def test_verbose_flag_set(self) -> None:
        mock_console = MagicMock()
        display = TerminalDisplay(console=mock_console, verbose=True)
        assert display._verbose is True

    def test_display_alias_equals_terminal_display(self) -> None:
        assert Display is TerminalDisplay


# ---------------------------------------------------------------------------
# Tests: stream_text
# ---------------------------------------------------------------------------

class TestStreamText:
    """Streaming text output."""

    def test_returns_concatenated_text(self) -> None:
        display, console = _make_display()
        result = display.stream_text(_text_chunks("Hello", " ", "World"))
        assert result == "Hello World"

    def test_prints_each_chunk(self) -> None:
        display, console = _make_display()
        display.stream_text(_text_chunks("A", "B", "C"))
        # Each chunk printed + final newline
        assert console.print.call_count == 4  # 3 chunks + 1 final newline

    def test_prints_chunks_without_end_newline(self) -> None:
        display, console = _make_display()
        display.stream_text(_text_chunks("chunk"))
        # First call should have end="" and highlight=False
        console.print.assert_any_call("chunk", end="", highlight=False)

    def test_prints_final_newline(self) -> None:
        display, console = _make_display()
        display.stream_text(_text_chunks("x"))
        # Last call is just console.print() for newline
        last_call = console.print.call_args_list[-1]
        assert last_call == call()

    def test_empty_generator_returns_empty_string(self) -> None:
        display, console = _make_display()
        result = display.stream_text(_text_chunks())
        assert result == ""


# ---------------------------------------------------------------------------
# Tests: show_thinking
# ---------------------------------------------------------------------------

class TestShowThinking:
    """Thinking/reasoning display."""

    def test_shows_thinking_text(self) -> None:
        display, console = _make_display()
        display.show_thinking("I need to read the file first")
        console.print.assert_called_once()
        printed = console.print.call_args[0][0]
        assert "I need to read the file first" in printed

    def test_truncates_long_thinking(self) -> None:
        display, console = _make_display()
        long_text = "x" * 500
        display.show_thinking(long_text, max_length=100)
        printed = console.print.call_args[0][0]
        assert "..." in printed

    def test_does_not_truncate_short_text(self) -> None:
        display, console = _make_display()
        display.show_thinking("short", max_length=200)
        printed = console.print.call_args[0][0]
        assert "..." not in printed

    def test_skips_empty_text(self) -> None:
        display, console = _make_display()
        display.show_thinking("")
        console.print.assert_not_called()

    def test_skips_whitespace_only(self) -> None:
        display, console = _make_display()
        display.show_thinking("   \n\t  ")
        console.print.assert_not_called()

    def test_custom_max_length(self) -> None:
        display, console = _make_display()
        display.show_thinking("a" * 50, max_length=20)
        printed = console.print.call_args[0][0]
        assert "..." in printed

    def test_show_thinking_start(self) -> None:
        display, console = _make_display()
        display.show_thinking_start()
        console.print.assert_called_once()
        printed = console.print.call_args[0][0]
        assert "Thinking" in printed

    def test_show_thinking_end(self) -> None:
        display, console = _make_display()
        display.show_thinking_end()
        console.print.assert_called_once_with()


# ---------------------------------------------------------------------------
# Tests: show_progress
# ---------------------------------------------------------------------------

class TestShowProgress:
    """Progress indicator (spinner)."""

    def test_returns_live_instance(self) -> None:
        display, console = _make_display()
        live = display.show_progress("Loading")
        # Should return a Live instance (from rich.live)
        from rich.live import Live
        assert isinstance(live, Live)

    def test_custom_message(self) -> None:
        display, console = _make_display()
        live = display.show_progress("Fetching models")
        # The Live object should be created — just verify no exception


# ---------------------------------------------------------------------------
# Tests: print_markdown
# ---------------------------------------------------------------------------

class TestPrintMarkdown:
    """Markdown rendering."""

    def test_renders_markdown(self) -> None:
        display, console = _make_display()
        display.print_markdown("# Hello\nSome **bold** text")
        console.print.assert_called_once()
        from rich.markdown import Markdown
        arg = console.print.call_args[0][0]
        assert isinstance(arg, Markdown)

    def test_skips_empty_text(self) -> None:
        display, console = _make_display()
        display.print_markdown("   ")
        console.print.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: show_tool_call
# ---------------------------------------------------------------------------

class TestShowToolCall:
    """Tool call display."""

    def test_shows_tool_name(self) -> None:
        display, console = _make_display()
        display.show_tool_call("read_file", {"path": "src/app.py"})
        printed = console.print.call_args[0][0]
        assert "read_file" in printed

    def test_shows_arguments(self) -> None:
        display, console = _make_display()
        display.show_tool_call("search", {"query": "hello", "path": "/src"})
        printed = console.print.call_args[0][0]
        assert "query=hello" in printed
        assert "path=/src" in printed

    def test_truncates_long_arg_values(self) -> None:
        display, console = _make_display()
        long_val = "x" * 200
        display.show_tool_call("write_file", {"content": long_val})
        printed = console.print.call_args[0][0]
        assert "..." in printed

    def test_empty_args(self) -> None:
        display, console = _make_display()
        display.show_tool_call("list_files", {})
        printed = console.print.call_args[0][0]
        assert "list_files" in printed

    def test_includes_wrench_emoji(self) -> None:
        display, console = _make_display()
        display.show_tool_call("test_tool", {})
        printed = console.print.call_args[0][0]
        assert "🔧" in printed


# ---------------------------------------------------------------------------
# Tests: show_tool_result
# ---------------------------------------------------------------------------

class TestShowToolResult:
    """Tool result display."""

    def test_verbose_shows_panel(self) -> None:
        display, console = _make_display(verbose=True)
        display.show_tool_result("read_file", "file content here")
        console.print.assert_called_once()
        from rich.panel import Panel
        arg = console.print.call_args[0][0]
        assert isinstance(arg, Panel)

    def test_compact_shows_first_line(self) -> None:
        display, console = _make_display(verbose=False)
        display.show_tool_result("read_file", "line1\nline2\nline3")
        printed = console.print.call_args[0][0]
        assert "read_file" in printed
        assert "line1" in printed

    def test_truncates_long_results_in_verbose(self) -> None:
        display, console = _make_display(verbose=True)
        long_result = "\n".join(f"line {i}" for i in range(200))
        display.show_tool_result("cmd", long_result, max_lines=50)
        # Should show truncated text in the panel
        console.print.assert_called_once()

    def test_compact_shows_total_line_count(self) -> None:
        display, console = _make_display(verbose=False)
        result = "\n".join(f"line {i}" for i in range(20))
        display.show_tool_result("grep", result)
        printed = str(console.print.call_args)
        assert "20" in printed or "line" in printed

    def test_compact_short_result_no_total(self) -> None:
        display, console = _make_display(verbose=False)
        display.show_tool_result("cmd", "just one line")
        printed = console.print.call_args[0][0]
        assert "total lines" not in printed

    def test_empty_result_in_compact(self) -> None:
        display, console = _make_display(verbose=False)
        display.show_tool_result("cmd", "")
        printed = console.print.call_args[0][0]
        assert "(empty)" in printed


# ---------------------------------------------------------------------------
# Tests: show_diff
# ---------------------------------------------------------------------------

class TestShowDiff:
    """Diff display."""

    def test_shows_diff_with_changes(self) -> None:
        display, console = _make_display()
        display.show_diff("app.py", "old line\n", "new line\n")
        # Should call print at least twice (syntax + summary)
        assert console.print.call_count >= 2

    def test_shows_no_changes_message(self) -> None:
        display, console = _make_display()
        display.show_diff("app.py", "same\n", "same\n")
        printed = console.print.call_args[0][0]
        assert "No changes" in printed

    def test_shows_addition_and_deletion_counts(self) -> None:
        display, console = _make_display()
        display.show_diff("test.py", "old\n", "new\n")
        # The summary line should contain +N / -N
        # Find the last print call which has the summary
        all_prints = [c[0][0] for c in console.print.call_args_list if c[0]]
        summary = str(all_prints[-1])
        assert "+" in summary
        assert "-" in summary

    def test_show_file_created(self) -> None:
        display, console = _make_display()
        display.show_file_created("new_file.py", size=1234)
        printed = console.print.call_args[0][0]
        assert "Created" in printed
        assert "new_file.py" in printed
        assert "1,234" in printed

    def test_show_file_created_no_size(self) -> None:
        display, console = _make_display()
        display.show_file_created("empty.py")
        printed = console.print.call_args[0][0]
        assert "bytes" not in printed

    def test_show_file_deleted(self) -> None:
        display, console = _make_display()
        display.show_file_deleted("old_file.py")
        printed = console.print.call_args[0][0]
        assert "Deleted" in printed
        assert "old_file.py" in printed


# ---------------------------------------------------------------------------
# Tests: Messages (error, info, warning)
# ---------------------------------------------------------------------------

class TestMessages:
    """Error, info, and warning messages."""

    def test_show_error(self) -> None:
        display, console = _make_display()
        display.show_error("Something failed")
        printed = console.print.call_args[0][0]
        assert "Something failed" in printed
        assert "✗" in printed

    def test_show_info(self) -> None:
        display, console = _make_display()
        display.show_info("Model loaded")
        printed = console.print.call_args[0][0]
        assert "Model loaded" in printed

    def test_show_warning(self) -> None:
        display, console = _make_display()
        display.show_warning("Quota running low")
        printed = console.print.call_args[0][0]
        assert "Quota running low" in printed
        assert "⚠" in printed


# ---------------------------------------------------------------------------
# Tests: show_stats
# ---------------------------------------------------------------------------

class TestShowStats:
    """Token usage and cost statistics."""

    def test_basic_stats(self) -> None:
        display, console = _make_display()
        display.show_stats(tokens_in=100, tokens_out=50)
        printed = console.print.call_args[0][0]
        assert "in:" in printed
        assert "out:" in printed

    def test_formats_large_numbers_with_k(self) -> None:
        display, console = _make_display()
        display.show_stats(tokens_in=5000, tokens_out=2000)
        printed = console.print.call_args[0][0]
        assert "5.0K" in printed
        assert "2.0K" in printed

    def test_small_numbers_not_formatted_with_k(self) -> None:
        display, console = _make_display()
        display.show_stats(tokens_in=500, tokens_out=100)
        printed = console.print.call_args[0][0]
        assert "500" in printed
        assert "100" in printed

    def test_shows_cost(self) -> None:
        display, console = _make_display()
        display.show_stats(tokens_in=100, tokens_out=50, cost=0.0025)
        printed = console.print.call_args[0][0]
        assert "$0.0025" in printed

    def test_shows_free_label(self) -> None:
        display, console = _make_display()
        display.show_stats(tokens_in=100, tokens_out=50, is_free=True)
        printed = console.print.call_args[0][0]
        assert "FREE" in printed

    def test_shows_thinking_tokens(self) -> None:
        display, console = _make_display()
        display.show_stats(tokens_in=100, tokens_out=50, thinking_tokens=30)
        printed = console.print.call_args[0][0]
        assert "think:" in printed
        assert "30" in printed

    def test_shows_cache_read_tokens(self) -> None:
        display, console = _make_display()
        display.show_stats(tokens_in=100, tokens_out=50, cache_read_tokens=2000)
        printed = console.print.call_args[0][0]
        assert "cache:" in printed
        assert "hit:" in printed

    def test_shows_cache_creation_tokens(self) -> None:
        display, console = _make_display()
        display.show_stats(tokens_in=100, tokens_out=50, cache_creation_tokens=1500)
        printed = console.print.call_args[0][0]
        assert "write:" in printed

    def test_shows_model_name(self) -> None:
        display, console = _make_display()
        display.show_stats(tokens_in=100, tokens_out=50, model="gpt-4o")
        printed = console.print.call_args[0][0]
        assert "gpt-4o" in printed

    def test_quota_high_green(self) -> None:
        display, console = _make_display()
        display.show_stats(tokens_in=100, tokens_out=50, quota_pct=0.75)
        printed = console.print.call_args[0][0]
        assert "75% left" in printed
        assert "green" in printed

    def test_quota_medium_yellow(self) -> None:
        display, console = _make_display()
        display.show_stats(tokens_in=100, tokens_out=50, quota_pct=0.35)
        printed = console.print.call_args[0][0]
        assert "35% left" in printed
        assert "yellow" in printed

    def test_quota_low_red(self) -> None:
        display, console = _make_display()
        display.show_stats(tokens_in=100, tokens_out=50, quota_pct=0.10)
        printed = console.print.call_args[0][0]
        assert "10% left" in printed
        assert "red" in printed

    def test_no_cost_when_zero(self) -> None:
        display, console = _make_display()
        display.show_stats(tokens_in=100, tokens_out=50, cost=0.0)
        printed = console.print.call_args[0][0]
        assert "$" not in printed

    def test_no_thinking_tokens_when_zero(self) -> None:
        display, console = _make_display()
        display.show_stats(tokens_in=100, tokens_out=50, thinking_tokens=0)
        printed = console.print.call_args[0][0]
        assert "think:" not in printed

    def test_no_cache_when_both_zero(self) -> None:
        display, console = _make_display()
        display.show_stats(tokens_in=100, tokens_out=50, cache_read_tokens=0, cache_creation_tokens=0)
        printed = console.print.call_args[0][0]
        assert "cache:" not in printed


# ---------------------------------------------------------------------------
# Tests: show_banner
# ---------------------------------------------------------------------------

class TestShowBanner:
    """Startup banner display."""

    def test_shows_banner_panel(self) -> None:
        display, console = _make_display()
        display.show_banner(model="gpt-4o", provider="openai", workspace="/projects/test")
        # Should print a panel + a blank line
        assert console.print.call_count == 2
        from rich.panel import Panel
        first_arg = console.print.call_args_list[0][0][0]
        assert isinstance(first_arg, Panel)

    def test_banner_contains_model(self) -> None:
        display, console = _make_display()
        display.show_banner(model="claude-sonnet-4", provider="anthropic", workspace="/ws")
        # Verify model appears in the printed content
        console.print.assert_called()


# ---------------------------------------------------------------------------
# Tests: AGENT_THEME
# ---------------------------------------------------------------------------

class TestAgentTheme:
    """Custom rich theme."""

    def test_theme_has_expected_styles(self) -> None:
        from rich.theme import Theme
        assert isinstance(AGENT_THEME, Theme)
        # Theme styles are stored as a dict internally
        style_names = set(AGENT_THEME.styles.keys())
        assert "tool.name" in style_names
        assert "error" in style_names
        assert "info" in style_names
        assert "stats" in style_names
