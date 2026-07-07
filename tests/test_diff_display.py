"""Tests for unjess.gui.components.diff_view — _count_changes, _colorize_line, render_diff."""

import html
import sys
from unittest.mock import MagicMock, patch

import pytest

# The diff_view module imports nicegui.ui at module level, so we must
# mock it before importing.  We insert a fake 'nicegui' package into
# sys.modules so the import succeeds without the real dependency.
_nicegui_mock = MagicMock()
sys.modules.setdefault("nicegui", _nicegui_mock)
sys.modules.setdefault("nicegui.ui", _nicegui_mock.ui)

from unjess.gui.components.diff_view import (
    _colorize_line,
    _count_changes,
    render_diff,
)


# ---------------------------------------------------------------------------
# _count_changes
# ---------------------------------------------------------------------------

class TestCountChanges:
    """Tests for the _count_changes helper that tallies + / - lines."""

    def test_empty_diff(self) -> None:
        assert _count_changes("") == (0, 0)

    def test_additions_only(self) -> None:
        diff = "+line1\n+line2\n+line3\n"
        assert _count_changes(diff) == (3, 0)

    def test_deletions_only(self) -> None:
        diff = "-old1\n-old2\n"
        assert _count_changes(diff) == (0, 2)

    def test_mixed_changes(self) -> None:
        diff = "+new\n-old\n context\n+new2\n"
        assert _count_changes(diff) == (2, 1)

    def test_headers_excluded(self) -> None:
        diff = "--- a/file.py\n+++ b/file.py\n@@ -1,3 +1,4 @@\n+added\n"
        additions, deletions = _count_changes(diff)
        assert additions == 1
        assert deletions == 0

    def test_context_lines_not_counted(self) -> None:
        diff = " context1\n context2\n+added\n"
        assert _count_changes(diff) == (1, 0)


# ---------------------------------------------------------------------------
# _colorize_line
# ---------------------------------------------------------------------------

class TestColorizeLine:
    """Tests for the _colorize_line helper that wraps lines in colored HTML spans."""

    def test_addition_line(self) -> None:
        result = _colorize_line("+added line")
        assert "color:#3fb950" in result
        assert html.escape("+added line") in result

    def test_deletion_line(self) -> None:
        result = _colorize_line("-removed line")
        assert "color:#f85149" in result
        assert html.escape("-removed line") in result

    def test_header_minus_three(self) -> None:
        result = _colorize_line("--- a/file.py")
        assert "font-weight:bold" in result
        assert "color:#8b949e" in result

    def test_header_plus_three(self) -> None:
        result = _colorize_line("+++ b/file.py")
        assert "font-weight:bold" in result
        assert "color:#8b949e" in result

    def test_hunk_header(self) -> None:
        result = _colorize_line("@@ -1,3 +1,4 @@")
        assert "color:#79c0ff" in result

    def test_context_line(self) -> None:
        result = _colorize_line(" unchanged context")
        assert "color:#8b949e" in result
        # Should NOT have the addition/deletion backgrounds
        assert "3fb950" not in result
        assert "f85149" not in result

    def test_html_entities_escaped(self) -> None:
        result = _colorize_line("+<script>alert('xss')</script>")
        assert "&lt;script&gt;" in result
        assert "<script>" not in result

    def test_empty_line(self) -> None:
        result = _colorize_line("")
        # An empty string isn't a +/- line, so gets context styling
        assert "color:#8b949e" in result


# ---------------------------------------------------------------------------
# render_diff
# ---------------------------------------------------------------------------

class TestRenderDiff:
    """Tests for the render_diff function (UI calls are mocked via nicegui stub)."""

    def test_render_diff_empty_text_is_noop(self) -> None:
        # Should return without errors for empty/blank diff
        render_diff("", "file.py")
        render_diff("   \n  \n", "file.py")

    def test_render_diff_calls_nicegui_ui(self) -> None:
        diff_text = "+++ b/file.py\n--- a/file.py\n+added line\n-removed line\n"
        # render_diff uses ui.card(), ui.row(), etc. — these are mocked
        # Just verify it doesn't raise
        render_diff(diff_text, "file.py")

    def test_render_diff_whitespace_only_is_noop(self) -> None:
        render_diff("   ", "some/path.py")
