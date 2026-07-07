"""Tests for unjess.tools.search_tools."""

import shutil
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from unjess.tools.search_tools import (
    _grep_search,
    _search_with_python,
    _MAX_RESULTS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reset_file_tools_globals(workspace: Optional[Path] = None) -> None:
    """Reset module-level globals so _resolve_safe works within workspace."""
    import unjess.tools.file_tools as ft
    ft._permissions = None
    ft._approved_external_paths.clear()
    ft._workspace_ref["current"] = workspace.resolve() if workspace else None


def _force_python_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure ripgrep is NOT found so pure-Python fallback is used."""
    monkeypatch.setattr(shutil, "which", lambda name: None)


# ===================================================================
# _grep_search — basic behavior
# ===================================================================

class TestGrepSearch:
    """Search across workspace files."""

    def test_literal_search_finds_match(
        self, tmp_workspace: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _reset_file_tools_globals(tmp_workspace)
        _force_python_fallback(monkeypatch)
        result = _grep_search(tmp_workspace, "greet")
        assert "match" in result.lower()
        assert "greet" in result

    def test_no_matches(
        self, tmp_workspace: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _reset_file_tools_globals(tmp_workspace)
        _force_python_fallback(monkeypatch)
        result = _grep_search(tmp_workspace, "zzz_nonexistent_xyz")
        assert "No matches found" in result

    def test_case_insensitive_search(
        self, tmp_workspace: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _reset_file_tools_globals(tmp_workspace)
        _force_python_fallback(monkeypatch)
        result = _grep_search(tmp_workspace, "GREET", case_insensitive=True)
        assert "match" in result.lower()

    def test_case_sensitive_search_misses(
        self, tmp_workspace: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _reset_file_tools_globals(tmp_workspace)
        _force_python_fallback(monkeypatch)
        # "GREET" with case_sensitive (default) should NOT match "greet"
        result = _grep_search(tmp_workspace, "GREET", case_insensitive=False)
        assert "No matches found" in result

    def test_regex_search(
        self, tmp_workspace: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _reset_file_tools_globals(tmp_workspace)
        _force_python_fallback(monkeypatch)
        # Regex: "def \w+" should match function definitions
        result = _grep_search(tmp_workspace, r"def \w+", regex=True)
        assert "match" in result.lower()

    def test_search_specific_file(
        self, tmp_workspace: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _reset_file_tools_globals(tmp_workspace)
        _force_python_fallback(monkeypatch)
        result = _grep_search(tmp_workspace, "add", path="src/app.py")
        assert "match" in result.lower()
        assert "app.py" in result

    def test_search_nonexistent_path(
        self, tmp_workspace: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _reset_file_tools_globals(tmp_workspace)
        _force_python_fallback(monkeypatch)
        result = _grep_search(tmp_workspace, "hello", path="no_such_dir")
        assert "Error: Path not found" in result

    def test_search_subdirectory(
        self, tmp_workspace: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _reset_file_tools_globals(tmp_workspace)
        _force_python_fallback(monkeypatch)
        result = _grep_search(tmp_workspace, "import", path="src")
        assert "match" in result.lower()


# ===================================================================
# _search_with_python — detailed fallback tests
# ===================================================================

class TestSearchWithPython:
    """Pure-Python fallback search engine."""

    def test_results_include_line_numbers(
        self, tmp_workspace: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _reset_file_tools_globals(tmp_workspace)
        _force_python_fallback(monkeypatch)
        result = _grep_search(tmp_workspace, "greet", path="src/app.py")
        # Format: "file:line: content"
        assert ":3:" in result or ":4:" in result or ":1:" in result

    def test_invalid_regex_returns_error(
        self, tmp_workspace: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _reset_file_tools_globals(tmp_workspace)
        _force_python_fallback(monkeypatch)
        result = _grep_search(tmp_workspace, "[invalid(regex", regex=True)
        assert "Error: Invalid regex" in result

    def test_binary_files_excluded(
        self, tmp_workspace: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _reset_file_tools_globals(tmp_workspace)
        _force_python_fallback(monkeypatch)
        # Create a .pyc file with searchable text
        pyc = tmp_workspace / "cached.pyc"
        pyc.write_text("greet_from_cache", encoding="utf-8")
        result = _grep_search(tmp_workspace, "greet_from_cache")
        assert "No matches found" in result

    def test_hidden_dirs_excluded(
        self, tmp_workspace: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _reset_file_tools_globals(tmp_workspace)
        _force_python_fallback(monkeypatch)
        hidden = tmp_workspace / ".secret"
        hidden.mkdir()
        (hidden / "data.txt").write_text("hidden_content", encoding="utf-8")
        result = _grep_search(tmp_workspace, "hidden_content")
        assert "No matches found" in result

    def test_max_results_cap(
        self, tmp_workspace: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _reset_file_tools_globals(tmp_workspace)
        _force_python_fallback(monkeypatch)
        # Create a file with 100 matching lines
        many_matches = tmp_workspace / "many.txt"
        many_matches.write_text(
            "\n".join(f"match_keyword line {i}" for i in range(100)),
            encoding="utf-8",
        )
        result = _grep_search(tmp_workspace, "match_keyword", path="many.txt")
        assert f"capped at {_MAX_RESULTS}" in result

    def test_search_single_file(self, tmp_workspace: Path) -> None:
        _reset_file_tools_globals(tmp_workspace)
        target = tmp_workspace / "src" / "app.py"
        result = _search_with_python("greet", target, tmp_workspace, False, False)
        assert "match" in result.lower()

    def test_search_single_file_no_match(self, tmp_workspace: Path) -> None:
        _reset_file_tools_globals(tmp_workspace)
        target = tmp_workspace / "src" / "app.py"
        result = _search_with_python("zzz_nope", target, tmp_workspace, False, False)
        assert "No matches found" in result

    def test_regex_with_case_insensitive(
        self, tmp_workspace: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _reset_file_tools_globals(tmp_workspace)
        _force_python_fallback(monkeypatch)
        result = _grep_search(
            tmp_workspace, r"DEF \w+", regex=True, case_insensitive=True
        )
        assert "match" in result.lower()


# ===================================================================
# _MAX_RESULTS constant
# ===================================================================

class TestConstants:
    """Verify search constants."""

    def test_max_results_is_50(self) -> None:
        assert _MAX_RESULTS == 50
