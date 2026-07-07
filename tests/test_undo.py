"""Tests for unjess.undo — undo/rollback system with git and snapshot strategies."""

from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from unjess.undo import UndoEntry, UndoManager, _SNAPSHOT_DIR


# ---------------------------------------------------------------------------
# UndoEntry dataclass
# ---------------------------------------------------------------------------


class TestUndoEntry:
    """Tests for the UndoEntry dataclass and its display() method."""

    def test_display_short_id(self) -> None:
        entry = UndoEntry(id="abcdef1234567890", label="edit file", timestamp="2026-01-01", source="git")
        result = entry.display()
        assert result == "[abcdef12] edit file (2026-01-01)"

    def test_display_short_id_when_id_short(self) -> None:
        entry = UndoEntry(id="abc", label="snap", timestamp="now", source="snapshot")
        result = entry.display()
        assert result == "[abc] snap (now)"

    def test_display_exactly_eight_chars(self) -> None:
        entry = UndoEntry(id="12345678", label="test", timestamp="t", source="git")
        result = entry.display()
        assert result == "[12345678] test (t)"

    def test_fields(self) -> None:
        entry = UndoEntry(id="x", label="y", timestamp="z", source="git")
        assert entry.id == "x"
        assert entry.label == "y"
        assert entry.timestamp == "z"
        assert entry.source == "git"


# ---------------------------------------------------------------------------
# UndoManager — git mode
# ---------------------------------------------------------------------------


def _make_mock_git(is_repo: bool = True) -> MagicMock:
    """Create a mock GitIntegration."""
    mock = MagicMock()
    type(mock).is_git_repo = PropertyMock(return_value=is_repo)
    return mock


class TestUndoManagerGitMode:
    """Tests for UndoManager when backed by git."""

    def test_uses_git_true(self, tmp_path: Path) -> None:
        git = _make_mock_git(is_repo=True)
        mgr = UndoManager(tmp_path, git)
        assert mgr.uses_git is True

    def test_checkpoint_delegates_to_git(self, tmp_path: Path) -> None:
        git = _make_mock_git(is_repo=True)
        git.checkpoint.return_value = "abc123"
        mgr = UndoManager(tmp_path, git)
        result = mgr.checkpoint("test edit")
        git.checkpoint.assert_called_once_with("test edit")
        assert result == "abc123"

    def test_checkpoint_returns_none_on_git_failure(self, tmp_path: Path) -> None:
        git = _make_mock_git(is_repo=True)
        git.checkpoint.return_value = None
        mgr = UndoManager(tmp_path, git)
        assert mgr.checkpoint("label") is None

    def test_undo_git_reverts_checkpoints(self, tmp_path: Path) -> None:
        git = _make_mock_git(is_repo=True)
        cp = MagicMock()
        cp.commit_hash = "abc123"
        git.list_checkpoints.return_value = [cp]
        git.revert_checkpoint.return_value = True
        mgr = UndoManager(tmp_path, git)
        success, msg = mgr.undo()
        assert success is True
        assert "Reverted 1 checkpoint(s)" in msg
        git.revert_checkpoint.assert_called_once_with("abc123")

    def test_undo_git_no_checkpoints(self, tmp_path: Path) -> None:
        git = _make_mock_git(is_repo=True)
        git.list_checkpoints.return_value = []
        mgr = UndoManager(tmp_path, git)
        success, msg = mgr.undo()
        assert success is False
        assert "No checkpoints" in msg

    def test_undo_git_revert_fails(self, tmp_path: Path) -> None:
        git = _make_mock_git(is_repo=True)
        cp = MagicMock()
        cp.commit_hash = "def456"
        git.list_checkpoints.return_value = [cp]
        git.revert_checkpoint.return_value = False
        mgr = UndoManager(tmp_path, git)
        success, msg = mgr.undo()
        assert success is False
        assert "Failed to revert" in msg

    def test_undo_git_multiple_steps(self, tmp_path: Path) -> None:
        git = _make_mock_git(is_repo=True)
        cp1, cp2 = MagicMock(), MagicMock()
        cp1.commit_hash = "aaa"
        cp2.commit_hash = "bbb"
        git.list_checkpoints.return_value = [cp1, cp2]
        git.revert_checkpoint.return_value = True
        mgr = UndoManager(tmp_path, git)
        success, msg = mgr.undo(steps=2)
        assert success is True
        assert "Reverted 2 checkpoint(s)" in msg

    def test_undo_git_partial_revert(self, tmp_path: Path) -> None:
        git = _make_mock_git(is_repo=True)
        cp1, cp2 = MagicMock(), MagicMock()
        cp1.commit_hash = "aaa"
        cp2.commit_hash = "bbb"
        git.list_checkpoints.return_value = [cp1, cp2]
        git.revert_checkpoint.side_effect = [True, False]
        mgr = UndoManager(tmp_path, git)
        success, msg = mgr.undo(steps=2)
        assert success is True
        assert "Reverted 1 checkpoint(s)" in msg

    def test_list_entries_git(self, tmp_path: Path) -> None:
        git = _make_mock_git(is_repo=True)
        cp = MagicMock()
        cp.commit_hash = "hash123"
        cp.label = "my edit"
        cp.timestamp = "2026-01-01"
        git.list_checkpoints.return_value = [cp]
        mgr = UndoManager(tmp_path, git)
        entries = mgr.list_entries()
        assert len(entries) == 1
        assert entries[0].source == "git"
        assert entries[0].id == "hash123"
        assert entries[0].label == "my edit"


# ---------------------------------------------------------------------------
# UndoManager — snapshot mode
# ---------------------------------------------------------------------------


class TestUndoManagerSnapshotMode:
    """Tests for UndoManager when no git is available (file snapshot strategy)."""

    def test_uses_git_false(self, tmp_path: Path) -> None:
        git = _make_mock_git(is_repo=False)
        mgr = UndoManager(tmp_path, git)
        assert mgr.uses_git is False

    def test_snapshot_checkpoint_creates_files(self, tmp_path: Path) -> None:
        # Create a file in the workspace
        (tmp_path / "hello.py").write_text("print('hi')\n", encoding="utf-8")

        git = _make_mock_git(is_repo=False)
        mgr = UndoManager(tmp_path, git)

        with patch("unjess.undo._SNAPSHOT_DIR", tmp_path / ".snapshots"):
            result = mgr.checkpoint("before edit")

        assert result is not None
        assert len(result) > 0  # timestamp string returned

    def test_snapshot_checkpoint_writes_meta(self, tmp_path: Path) -> None:
        (tmp_path / "code.py").write_text("x = 1\n", encoding="utf-8")
        snap_dir = tmp_path / ".test_snapshots"

        git = _make_mock_git(is_repo=False)
        mgr = UndoManager(tmp_path, git)

        with patch("unjess.undo._SNAPSHOT_DIR", snap_dir):
            ts = mgr.checkpoint("test label")

        meta = (snap_dir / ts / ".undo_meta").read_text(encoding="utf-8")
        assert "label=test label" in meta
        assert "files=" in meta
        assert "manifest=" in meta

    def test_snapshot_checkpoint_skips_hidden_dirs(self, tmp_path: Path) -> None:
        hidden = tmp_path / ".hidden"
        hidden.mkdir()
        (hidden / "secret.py").write_text("secret\n", encoding="utf-8")
        (tmp_path / "visible.py").write_text("ok\n", encoding="utf-8")

        git = _make_mock_git(is_repo=False)
        mgr = UndoManager(tmp_path, git)
        snap_dir = tmp_path / ".test_snapshots2"

        with patch("unjess.undo._SNAPSHOT_DIR", snap_dir):
            ts = mgr.checkpoint("label")

        # visible.py should be snapshotted, secret.py should not
        assert (snap_dir / ts / "visible.py").exists()
        assert not (snap_dir / ts / ".hidden" / "secret.py").exists()

    def test_snapshot_checkpoint_skips_common_dirs(self, tmp_path: Path) -> None:
        for dirname in ("node_modules", "__pycache__", "venv"):
            d = tmp_path / dirname
            d.mkdir()
            (d / "file.py").write_text("x\n", encoding="utf-8")
        (tmp_path / "main.py").write_text("y\n", encoding="utf-8")

        git = _make_mock_git(is_repo=False)
        mgr = UndoManager(tmp_path, git)
        snap_dir = tmp_path / ".snaps3"

        with patch("unjess.undo._SNAPSHOT_DIR", snap_dir):
            ts = mgr.checkpoint("label")

        assert (snap_dir / ts / "main.py").exists()
        for dirname in ("node_modules", "__pycache__", "venv"):
            assert not (snap_dir / ts / dirname).exists()

    def test_snapshot_undo_restores_files(self, tmp_path: Path) -> None:
        (tmp_path / "app.py").write_text("original\n", encoding="utf-8")

        git = _make_mock_git(is_repo=False)
        mgr = UndoManager(tmp_path, git)
        snap_dir = tmp_path / ".snaps4"

        with patch("unjess.undo._SNAPSHOT_DIR", snap_dir):
            mgr.checkpoint("before")
            # Modify the file
            (tmp_path / "app.py").write_text("modified\n", encoding="utf-8")
            success, msg = mgr.undo()

        assert success is True
        assert "Restored" in msg
        assert (tmp_path / "app.py").read_text(encoding="utf-8") == "original\n"

    def test_snapshot_undo_no_snapshots(self, tmp_path: Path) -> None:
        git = _make_mock_git(is_repo=False)
        mgr = UndoManager(tmp_path, git)
        snap_dir = tmp_path / ".nonexistent_snaps"

        with patch("unjess.undo._SNAPSHOT_DIR", snap_dir):
            success, msg = mgr.undo()

        assert success is False
        assert "No snapshots" in msg

    def test_snapshot_undo_deletes_new_files(self, tmp_path: Path) -> None:
        (tmp_path / "original.py").write_text("x\n", encoding="utf-8")

        git = _make_mock_git(is_repo=False)
        mgr = UndoManager(tmp_path, git)
        snap_dir = tmp_path / ".snaps5"

        with patch("unjess.undo._SNAPSHOT_DIR", snap_dir):
            mgr.checkpoint("before")
            # Create a new file after snapshot
            (tmp_path / "new_file.py").write_text("new\n", encoding="utf-8")
            success, msg = mgr.undo()

        assert success is True
        assert not (tmp_path / "new_file.py").exists()
        assert "Deleted" in msg

    def test_list_entries_snapshot(self, tmp_path: Path) -> None:
        (tmp_path / "file.py").write_text("x\n", encoding="utf-8")

        git = _make_mock_git(is_repo=False)
        mgr = UndoManager(tmp_path, git)
        snap_dir = tmp_path / ".snaps6"

        with patch("unjess.undo._SNAPSHOT_DIR", snap_dir):
            mgr.checkpoint("my snapshot")
            entries = mgr.list_entries()

        assert len(entries) == 1
        assert entries[0].source == "snapshot"
        assert entries[0].label == "my snapshot"

    def test_list_entries_empty(self, tmp_path: Path) -> None:
        git = _make_mock_git(is_repo=False)
        mgr = UndoManager(tmp_path, git)
        snap_dir = tmp_path / ".empty_snaps"

        with patch("unjess.undo._SNAPSHOT_DIR", snap_dir):
            entries = mgr.list_entries()

        assert entries == []

    def test_snapshot_undo_from_disk_when_stack_empty(self, tmp_path: Path) -> None:
        (tmp_path / "f.py").write_text("data\n", encoding="utf-8")
        snap_dir = tmp_path / ".snaps7"

        git = _make_mock_git(is_repo=False)
        mgr1 = UndoManager(tmp_path, git)

        with patch("unjess.undo._SNAPSHOT_DIR", snap_dir):
            mgr1.checkpoint("saved")

        # New manager instance with empty stack
        mgr2 = UndoManager(tmp_path, git)
        (tmp_path / "f.py").write_text("changed\n", encoding="utf-8")

        with patch("unjess.undo._SNAPSHOT_DIR", snap_dir):
            success, msg = mgr2.undo()

        assert success is True
        assert (tmp_path / "f.py").read_text(encoding="utf-8") == "data\n"

    def test_snapshot_checkpoint_returns_none_on_error(self, tmp_path: Path) -> None:
        git = _make_mock_git(is_repo=False)
        mgr = UndoManager(tmp_path, git)

        # Point snapshot dir to an invalid path to trigger an error
        with patch("unjess.undo._SNAPSHOT_DIR", Path("/\x00invalid")):
            result = mgr.checkpoint("will fail")

        assert result is None
