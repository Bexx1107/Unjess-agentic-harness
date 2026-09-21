"""Tests for unjess.tools.file_tools."""

import os
import textwrap
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from unjess.tools.file_tools import (
    _edit_file,
    _format_size,
    _list_dir,
    _read_file,
    _resolve_safe,
    _write_file,
    get_current_workspace,
    update_workspace,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reset_file_tools_globals(workspace: Optional[Path] = None) -> None:
    """Reset module-level globals to a clean state for isolation."""
    import unjess.tools.file_tools as ft
    ft._permissions = None
    ft._approved_external_paths.clear()
    ft._workspace_ref["current"] = workspace.resolve() if workspace else None


# ===================================================================
# _resolve_safe
# ===================================================================

class TestResolveSafe:
    """Path resolution and sandbox enforcement."""

    def test_relative_path_inside_workspace(self, tmp_workspace: Path) -> None:
        _reset_file_tools_globals(tmp_workspace)
        result = _resolve_safe("src/app.py", tmp_workspace)
        assert result == (tmp_workspace / "src" / "app.py").resolve()

    def test_absolute_path_inside_workspace(self, tmp_workspace: Path) -> None:
        _reset_file_tools_globals(tmp_workspace)
        abs_path = str(tmp_workspace / "README.md")
        result = _resolve_safe(abs_path, tmp_workspace)
        assert result == (tmp_workspace / "README.md").resolve()

    def test_workspace_root_itself(self, tmp_workspace: Path) -> None:
        _reset_file_tools_globals(tmp_workspace)
        result = _resolve_safe(".", tmp_workspace)
        assert result == tmp_workspace.resolve()

    def test_fuzzy_unicode_ellipsis_matching(self, tmp_workspace: Path) -> None:
        _reset_file_tools_globals(tmp_workspace)
        real_file = tmp_workspace / "File With…Ellipsis.csv"
        real_file.write_text("hello", encoding="utf-8")
        # Try resolving path with ASCII dot instead of unicode ellipsis
        ascii_path = str(tmp_workspace / "File With.Ellipsis.csv")
        result = _resolve_safe(ascii_path, tmp_workspace)
        assert result == real_file.resolve()

    def test_outside_workspace_denied_no_permissions(self, tmp_workspace: Path) -> None:
        _reset_file_tools_globals(tmp_workspace)
        with pytest.raises(PermissionError, match="outside the workspace"):
            _resolve_safe("/tmp/evil.txt", tmp_workspace)

    def test_outside_workspace_denied_by_user(self, tmp_workspace: Path) -> None:
        import unjess.tools.file_tools as ft
        _reset_file_tools_globals(tmp_workspace)
        mock_pm = MagicMock()
        mock_pm.request_approval.return_value = False
        ft._permissions = mock_pm

        with pytest.raises(PermissionError, match="outside the workspace"):
            _resolve_safe("/tmp/evil.txt", tmp_workspace)

    def test_outside_workspace_approved_by_user(self, tmp_workspace: Path) -> None:
        import unjess.tools.file_tools as ft
        _reset_file_tools_globals(tmp_workspace)
        mock_pm = MagicMock()
        mock_pm.request_approval.return_value = True
        ft._permissions = mock_pm

        result = _resolve_safe("/tmp/approved.txt", tmp_workspace)
        assert result == Path("/tmp/approved.txt").resolve()

    def test_approved_path_cached(self, tmp_workspace: Path) -> None:
        import unjess.tools.file_tools as ft
        _reset_file_tools_globals(tmp_workspace)
        mock_pm = MagicMock()
        mock_pm.request_approval.return_value = True
        ft._permissions = mock_pm

        _resolve_safe("/tmp/approved.txt", tmp_workspace)
        # Second call should not ask permission again
        _resolve_safe("/tmp/approved.txt", tmp_workspace)
        assert mock_pm.request_approval.call_count == 1

    def test_parent_dir_approval_covers_children(self, tmp_workspace: Path) -> None:
        import unjess.tools.file_tools as ft
        _reset_file_tools_globals(tmp_workspace)
        mock_pm = MagicMock()
        mock_pm.request_approval.return_value = True
        ft._permissions = mock_pm

        parent = Path("/tmp/mydir").resolve()
        ft._approved_external_paths.add(str(parent))
        # Child should be auto-approved
        result = _resolve_safe(str(parent / "child.txt"), tmp_workspace)
        assert result == (parent / "child.txt").resolve()
        mock_pm.request_approval.assert_not_called()

    def test_path_traversal_blocked(self, tmp_workspace: Path) -> None:
        _reset_file_tools_globals(tmp_workspace)
        with pytest.raises(PermissionError):
            _resolve_safe("../../etc/passwd", tmp_workspace)


# ===================================================================
# _read_file
# ===================================================================

class TestReadFile:
    """File reading with line ranges and error handling."""

    def test_read_full_file(self, tmp_workspace: Path) -> None:
        _reset_file_tools_globals(tmp_workspace)
        result = _read_file(tmp_workspace, "README.md")
        assert "# Test Project" in result
        assert "1 lines total" in result

    def test_read_with_line_range(self, tmp_workspace: Path) -> None:
        _reset_file_tools_globals(tmp_workspace)
        result = _read_file(tmp_workspace, "src/app.py", start_line=3, end_line=5)
        assert "showing 3-5" in result
        # Line 3 should be "def greet(..."
        assert "3:" in result

    def test_read_start_line_only(self, tmp_workspace: Path) -> None:
        _reset_file_tools_globals(tmp_workspace)
        result = _read_file(tmp_workspace, "src/app.py", start_line=2)
        assert "showing 2-" in result
        assert "1:" not in result

    def test_read_end_line_only(self, tmp_workspace: Path) -> None:
        _reset_file_tools_globals(tmp_workspace)
        result = _read_file(tmp_workspace, "src/app.py", end_line=3)
        assert "showing 1-3" in result

    def test_read_nonexistent_file(self, tmp_workspace: Path) -> None:
        _reset_file_tools_globals(tmp_workspace)
        result = _read_file(tmp_workspace, "no_such_file.py")
        assert "Error: File not found" in result

    def test_read_directory_as_file(self, tmp_workspace: Path) -> None:
        _reset_file_tools_globals(tmp_workspace)
        result = _read_file(tmp_workspace, "src")
        assert "Error: Not a file" in result

    def test_read_start_line_exceeds_length(self, tmp_workspace: Path) -> None:
        _reset_file_tools_globals(tmp_workspace)
        result = _read_file(tmp_workspace, "README.md", start_line=9999)
        assert "Error: start_line 9999 exceeds file length" in result

    def test_read_caps_at_1000_lines(self, tmp_workspace: Path) -> None:
        _reset_file_tools_globals(tmp_workspace)
        big_file = tmp_workspace / "big.txt"
        big_file.write_text("\n".join(f"Line {i}" for i in range(1500)), encoding="utf-8")
        result = _read_file(tmp_workspace, "big.txt")
        # Should show 1000 lines, not 1500
        assert "showing 1-1000" in result

    def test_read_empty_file(self, tmp_workspace: Path) -> None:
        _reset_file_tools_globals(tmp_workspace)
        empty = tmp_workspace / "empty.txt"
        empty.write_text("", encoding="utf-8")
        result = _read_file(tmp_workspace, "empty.txt")
        # Empty file: 0 lines, the source may return an error or empty content
        assert "0 lines" in result or "empty" in result.lower() or "Error" in result

    def test_read_with_line_numbers(self, tmp_workspace: Path) -> None:
        _reset_file_tools_globals(tmp_workspace)
        f = tmp_workspace / "numbered.txt"
        f.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
        result = _read_file(tmp_workspace, "numbered.txt")
        assert "1: alpha" in result
        assert "2: beta" in result
        assert "3: gamma" in result


# ===================================================================
# _write_file
# ===================================================================

class TestWriteFile:
    """File writing, overwrite guards, parent directory creation."""

    def test_create_new_file(self, tmp_workspace: Path) -> None:
        _reset_file_tools_globals(tmp_workspace)
        result = _write_file(tmp_workspace, "new.txt", "hello world\n")
        assert "Successfully wrote" in result
        assert (tmp_workspace / "new.txt").read_text(encoding="utf-8") == "hello world\n"

    def test_refuse_overwrite_by_default(self, tmp_workspace: Path) -> None:
        _reset_file_tools_globals(tmp_workspace)
        result = _write_file(tmp_workspace, "README.md", "overwritten")
        assert "Error: File already exists" in result
        # Original content preserved
        assert (tmp_workspace / "README.md").read_text(encoding="utf-8") == "# Test Project\n"

    def test_overwrite_when_flag_set(self, tmp_workspace: Path) -> None:
        _reset_file_tools_globals(tmp_workspace)
        result = _write_file(tmp_workspace, "README.md", "new content", overwrite=True)
        assert "Successfully wrote" in result
        assert (tmp_workspace / "README.md").read_text(encoding="utf-8") == "new content"

    def test_creates_parent_directories(self, tmp_workspace: Path) -> None:
        _reset_file_tools_globals(tmp_workspace)
        result = _write_file(tmp_workspace, "deep/nested/dir/file.py", "# code\n")
        assert "Successfully wrote" in result
        assert (tmp_workspace / "deep" / "nested" / "dir" / "file.py").exists()

    def test_line_count_in_result(self, tmp_workspace: Path) -> None:
        _reset_file_tools_globals(tmp_workspace)
        result = _write_file(tmp_workspace, "multi.txt", "a\nb\nc\n")
        assert "3 lines" in result

    def test_line_count_no_trailing_newline(self, tmp_workspace: Path) -> None:
        _reset_file_tools_globals(tmp_workspace)
        result = _write_file(tmp_workspace, "notrl.txt", "a\nb\nc")
        # "a\nb\nc" has 2 newlines + 1 (no trailing newline) = 3 lines
        assert "3 lines" in result

    def test_write_empty_content(self, tmp_workspace: Path) -> None:
        _reset_file_tools_globals(tmp_workspace)
        result = _write_file(tmp_workspace, "empty.txt", "")
        assert "Successfully wrote 0 lines" in result
        assert (tmp_workspace / "empty.txt").read_text(encoding="utf-8") == ""


# ===================================================================
# _edit_file
# ===================================================================

class TestEditFile:
    """Exact-match editing with diff output."""

    def test_simple_replacement(self, tmp_workspace: Path) -> None:
        _reset_file_tools_globals(tmp_workspace)
        result = _edit_file(tmp_workspace, "README.md", "# Test Project", "# My Project")
        assert "Successfully edited" in result
        assert (tmp_workspace / "README.md").read_text(encoding="utf-8") == "# My Project\n"
        # Should contain unified diff
        assert "---" in result or "+++" in result

    def test_target_not_found(self, tmp_workspace: Path) -> None:
        _reset_file_tools_globals(tmp_workspace)
        result = _edit_file(tmp_workspace, "README.md", "NONEXISTENT STRING", "replacement")
        assert "Error: Target string not found" in result
        assert "File starts with:" in result

    def test_multiple_matches_error(self, tmp_workspace: Path) -> None:
        _reset_file_tools_globals(tmp_workspace)
        dup = tmp_workspace / "dup.txt"
        dup.write_text("foo\nbar\nfoo\n", encoding="utf-8")
        result = _edit_file(tmp_workspace, "dup.txt", "foo", "baz")
        assert "found 2 times" in result

    def test_edit_nonexistent_file(self, tmp_workspace: Path) -> None:
        _reset_file_tools_globals(tmp_workspace)
        result = _edit_file(tmp_workspace, "ghost.py", "x", "y")
        assert "Error: File not found" in result

    def test_edit_multiline_target(self, tmp_workspace: Path) -> None:
        _reset_file_tools_globals(tmp_workspace)
        multi = tmp_workspace / "multi.py"
        multi.write_text("def foo():\n    return 1\n", encoding="utf-8")
        result = _edit_file(
            tmp_workspace, "multi.py",
            "def foo():\n    return 1",
            "def foo():\n    return 42",
        )
        assert "Successfully edited" in result
        assert "return 42" in multi.read_text(encoding="utf-8")

    def test_edit_produces_diff(self, tmp_workspace: Path) -> None:
        _reset_file_tools_globals(tmp_workspace)
        f = tmp_workspace / "diffme.txt"
        f.write_text("old_value = 10\n", encoding="utf-8")
        result = _edit_file(tmp_workspace, "diffme.txt", "old_value = 10", "new_value = 20")
        assert "-old_value = 10" in result
        assert "+new_value = 20" in result


# ===================================================================
# _list_dir
# ===================================================================

class TestListDir:
    """Directory listing with ignore patterns and formatting."""

    def test_list_workspace_root(self, tmp_workspace: Path) -> None:
        _reset_file_tools_globals(tmp_workspace)
        result = _list_dir(tmp_workspace, ".")
        assert "README.md" in result
        assert "src" in result
        assert "tests" in result
        assert "📁" in result  # directory icon
        assert "📄" in result  # file icon

    def test_list_subdirectory(self, tmp_workspace: Path) -> None:
        _reset_file_tools_globals(tmp_workspace)
        result = _list_dir(tmp_workspace, "src")
        assert "app.py" in result
        assert "utils.py" in result

    def test_list_nonexistent_dir(self, tmp_workspace: Path) -> None:
        _reset_file_tools_globals(tmp_workspace)
        result = _list_dir(tmp_workspace, "nope")
        assert "Error: Directory not found" in result

    def test_list_file_not_dir(self, tmp_workspace: Path) -> None:
        _reset_file_tools_globals(tmp_workspace)
        result = _list_dir(tmp_workspace, "README.md")
        assert "Error: Not a directory" in result

    def test_list_empty_dir(self, tmp_workspace: Path) -> None:
        _reset_file_tools_globals(tmp_workspace)
        (tmp_workspace / "empty_dir").mkdir()
        result = _list_dir(tmp_workspace, "empty_dir")
        assert "(empty)" in result

    def test_hidden_files_excluded(self, tmp_workspace: Path) -> None:
        _reset_file_tools_globals(tmp_workspace)
        (tmp_workspace / ".hidden").write_text("secret", encoding="utf-8")
        result = _list_dir(tmp_workspace, ".")
        assert ".hidden" not in result

    def test_ignored_patterns_excluded(self, tmp_workspace: Path) -> None:
        _reset_file_tools_globals(tmp_workspace)
        (tmp_workspace / "__pycache__").mkdir()
        (tmp_workspace / "node_modules").mkdir()
        result = _list_dir(tmp_workspace, ".")
        assert "__pycache__" not in result
        assert "node_modules" not in result

    def test_list_shows_item_count(self, tmp_workspace: Path) -> None:
        _reset_file_tools_globals(tmp_workspace)
        result = _list_dir(tmp_workspace, ".")
        assert "items)" in result  # header like "Directory: . (N items)"

    def test_directories_sorted_before_files(self, tmp_workspace: Path) -> None:
        _reset_file_tools_globals(tmp_workspace)
        result = _list_dir(tmp_workspace, ".")
        lines = result.strip().split("\n")
        # Find first dir line and first file line
        first_dir = next((i for i, l in enumerate(lines) if "📁" in l), None)
        first_file = next((i for i, l in enumerate(lines) if "📄" in l), None)
        if first_dir is not None and first_file is not None:
            assert first_dir < first_file


# ===================================================================
# _format_size
# ===================================================================

class TestFormatSize:
    """Human-readable byte formatting."""

    def test_bytes(self) -> None:
        assert _format_size(0) == "0 B"
        assert _format_size(512) == "512 B"
        assert _format_size(1023) == "1023 B"

    def test_kilobytes(self) -> None:
        assert _format_size(1024) == "1.0 KB"
        assert _format_size(2048) == "2.0 KB"
        assert _format_size(1536) == "1.5 KB"

    def test_megabytes(self) -> None:
        assert _format_size(1024 * 1024) == "1.0 MB"
        assert _format_size(5 * 1024 * 1024) == "5.0 MB"


# ===================================================================
# update_workspace / get_current_workspace
# ===================================================================

class TestWorkspaceManagement:
    """Workspace reference management."""

    def test_update_and_get(self, tmp_workspace: Path) -> None:
        update_workspace(tmp_workspace)
        assert get_current_workspace() == tmp_workspace.resolve()

    def test_get_raises_when_unset(self) -> None:
        import unjess.tools.file_tools as ft
        ft._workspace_ref["current"] = None
        with pytest.raises(RuntimeError, match="No workspace has been set"):
            get_current_workspace()

    def test_update_clears_approved_paths(self, tmp_workspace: Path) -> None:
        import unjess.tools.file_tools as ft
        ft._approved_external_paths.add("/some/path")
        update_workspace(tmp_workspace)
        assert len(ft._approved_external_paths) == 0
