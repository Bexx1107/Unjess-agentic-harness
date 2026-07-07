"""Tests for unjess.git_integration — GitIntegration, GitStatus, Commit, Checkpoint."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from unjess.git_integration import (
    Checkpoint,
    Commit,
    GitIntegration,
    GitStatus,
    _CHECKPOINT_PREFIX,
)


# ---------------------------------------------------------------------------
# GitStatus dataclass
# ---------------------------------------------------------------------------

class TestGitStatus:
    """Tests for the GitStatus dataclass and its summary method."""

    def test_summary_clean(self) -> None:
        gs = GitStatus(branch="main", is_clean=True)
        assert gs.summary() == "on main — clean"

    def test_summary_modified_only(self) -> None:
        gs = GitStatus(branch="dev", modified=["a.py"], is_clean=False)
        assert "1 modified" in gs.summary()
        assert "on dev" in gs.summary()

    def test_summary_staged_only(self) -> None:
        gs = GitStatus(branch="main", staged=["b.py", "c.py"], is_clean=False)
        assert "2 staged" in gs.summary()

    def test_summary_untracked_only(self) -> None:
        gs = GitStatus(branch="main", untracked=["new.py"], is_clean=False)
        assert "1 untracked" in gs.summary()

    def test_summary_deleted_only(self) -> None:
        gs = GitStatus(branch="main", deleted=["old.py"], is_clean=False)
        assert "1 deleted" in gs.summary()

    def test_summary_mixed(self) -> None:
        gs = GitStatus(
            branch="feature",
            modified=["a.py"],
            staged=["b.py"],
            untracked=["c.py"],
            deleted=["d.py"],
            is_clean=False,
        )
        s = gs.summary()
        assert "1 modified" in s
        assert "1 staged" in s
        assert "1 untracked" in s
        assert "1 deleted" in s
        assert "on feature" in s

    def test_default_values(self) -> None:
        gs = GitStatus()
        assert gs.branch == ""
        assert gs.modified == []
        assert gs.staged == []
        assert gs.untracked == []
        assert gs.deleted == []
        assert gs.is_clean is True


# ---------------------------------------------------------------------------
# Commit dataclass
# ---------------------------------------------------------------------------

class TestCommit:
    """Tests for the Commit dataclass and its is_checkpoint property."""

    def test_is_checkpoint_true(self) -> None:
        c = Commit(
            hash="abc123", short_hash="abc",
            author="Test", date="2026-01-01",
            message=f"{_CHECKPOINT_PREFIX} auto",
        )
        assert c.is_checkpoint is True

    def test_is_checkpoint_false(self) -> None:
        c = Commit(
            hash="abc123", short_hash="abc",
            author="Test", date="2026-01-01",
            message="Fix bug",
        )
        assert c.is_checkpoint is False


# ---------------------------------------------------------------------------
# GitIntegration — is_git_repo
# ---------------------------------------------------------------------------

class TestIsGitRepo:
    """Tests for the is_git_repo property."""

    def test_returns_true_when_dot_git_exists(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        gi = GitIntegration(tmp_path)
        assert gi.is_git_repo is True

    def test_returns_false_when_no_dot_git(self, tmp_path: Path) -> None:
        gi = GitIntegration(tmp_path)
        assert gi.is_git_repo is False

    def test_is_cached(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        gi = GitIntegration(tmp_path)
        _ = gi.is_git_repo
        # Remove .git — cached value should persist
        (tmp_path / ".git").rmdir()
        assert gi.is_git_repo is True


# ---------------------------------------------------------------------------
# GitIntegration._run — subprocess wrappers
# ---------------------------------------------------------------------------

class TestRun:
    """Tests for the internal _run method and error handling."""

    def test_run_calls_subprocess(self, tmp_path: Path) -> None:
        gi = GitIntegration(tmp_path)
        with patch("unjess.git_integration.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                ["git", "status"], 0, stdout="ok", stderr="",
            )
            result = gi._run("status")
            assert result.returncode == 0
            assert result.stdout == "ok"
            mock_run.assert_called_once_with(
                ["git", "status"],
                capture_output=True,
                text=True,
                cwd=str(tmp_path),
                timeout=10,
            )

    def test_run_timeout_returns_error(self, tmp_path: Path) -> None:
        gi = GitIntegration(tmp_path)
        with patch("unjess.git_integration.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(["git"], 10)
            result = gi._run("status")
            assert result.returncode == 1
            assert result.stdout == ""

    def test_run_file_not_found_returns_error(self, tmp_path: Path) -> None:
        gi = GitIntegration(tmp_path)
        with patch("unjess.git_integration.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("git not found")
            result = gi._run("status")
            assert result.returncode == 1


# ---------------------------------------------------------------------------
# GitIntegration.status
# ---------------------------------------------------------------------------

class TestStatus:
    """Tests for the status() method and porcelain-v1 output parsing."""

    def _gi_with_output(self, tmp_path: Path, stdout: str, rc: int = 0) -> GitIntegration:
        (tmp_path / ".git").mkdir()
        gi = GitIntegration(tmp_path)
        with patch.object(gi, "_run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                ["git"], rc, stdout=stdout, stderr="",
            )
            gi._status_result = gi.status()
            gi._mock_run = mock_run
        return gi

    def test_status_not_a_repo(self, tmp_path: Path) -> None:
        gi = GitIntegration(tmp_path)
        st = gi.status()
        assert st.is_clean is True
        assert st.branch == ""

    def test_status_parse_branch(self, tmp_path: Path) -> None:
        gi = self._gi_with_output(tmp_path, "## main...origin/main\n")
        assert gi._status_result.branch == "main"

    def test_status_parse_modified(self, tmp_path: Path) -> None:
        gi = self._gi_with_output(tmp_path, "## main\n M src/app.py\n")
        st = gi._status_result
        assert "src/app.py" in st.modified
        assert st.is_clean is False

    def test_status_parse_staged(self, tmp_path: Path) -> None:
        gi = self._gi_with_output(tmp_path, "## main\nM  src/app.py\n")
        st = gi._status_result
        assert "src/app.py" in st.staged
        assert st.is_clean is False

    def test_status_parse_untracked(self, tmp_path: Path) -> None:
        gi = self._gi_with_output(tmp_path, "## main\n?? new_file.py\n")
        st = gi._status_result
        assert "new_file.py" in st.untracked
        assert st.is_clean is False

    def test_status_parse_deleted(self, tmp_path: Path) -> None:
        gi = self._gi_with_output(tmp_path, "## main\n D old.py\n")
        st = gi._status_result
        assert "old.py" in st.deleted
        assert st.is_clean is False

    def test_status_clean_repo(self, tmp_path: Path) -> None:
        gi = self._gi_with_output(tmp_path, "## main\n")
        st = gi._status_result
        assert st.is_clean is True

    def test_status_error_returns_default(self, tmp_path: Path) -> None:
        gi = self._gi_with_output(tmp_path, "", rc=1)
        st = gi._status_result
        assert st.is_clean is True
        assert st.branch == ""

    def test_status_short_line_skipped(self, tmp_path: Path) -> None:
        gi = self._gi_with_output(tmp_path, "## dev\nX\n")
        st = gi._status_result
        # Short lines (< 4 chars) are ignored; should still parse cleanly
        assert st.branch == "dev"
        assert st.is_clean is True

    def test_status_mixed_indicators(self, tmp_path: Path) -> None:
        output = "## main\nMM both.py\nA  added.py\nR  renamed.py\n"
        gi = self._gi_with_output(tmp_path, output)
        st = gi._status_result
        assert "both.py" in st.staged      # M in index position
        assert "both.py" in st.modified     # M in working tree position
        assert "added.py" in st.staged
        assert "renamed.py" in st.staged


# ---------------------------------------------------------------------------
# GitIntegration.diff
# ---------------------------------------------------------------------------

class TestDiff:
    """Tests for the diff() method."""

    def test_diff_not_a_repo(self, tmp_path: Path) -> None:
        gi = GitIntegration(tmp_path)
        assert gi.diff() == ""

    def test_diff_returns_stdout(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        gi = GitIntegration(tmp_path)
        with patch.object(gi, "_run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                ["git"], 0, stdout="diff --git a/f b/f\n+line\n", stderr="",
            )
            result = gi.diff()
            assert "+line" in result
            mock_run.assert_called_once_with("diff", "HEAD")

    def test_diff_with_file_filter(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        gi = GitIntegration(tmp_path)
        with patch.object(gi, "_run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                ["git"], 0, stdout="diff text", stderr="",
            )
            gi.diff(file="src/app.py")
            mock_run.assert_called_once_with("diff", "HEAD", "--", "src/app.py")

    def test_diff_error_returns_empty(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        gi = GitIntegration(tmp_path)
        with patch.object(gi, "_run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                ["git"], 1, stdout="", stderr="error",
            )
            assert gi.diff() == ""


# ---------------------------------------------------------------------------
# GitIntegration.log
# ---------------------------------------------------------------------------

class TestLog:
    """Tests for the log() method and commit parsing."""

    def test_log_not_a_repo(self, tmp_path: Path) -> None:
        gi = GitIntegration(tmp_path)
        assert gi.log() == []

    def test_log_parses_commits(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        gi = GitIntegration(tmp_path)
        log_output = (
            "aaa111aaa111aaa111aaa111aaa111aaa111aaa111a\n"
            "aaa111a\n"
            "Alice\n"
            "2026-01-01 10:00:00 +0000\n"
            "Initial commit\n"
            "bbb222bbb222bbb222bbb222bbb222bbb222bbb222b\n"
            "bbb222b\n"
            "Bob\n"
            "2026-01-02 10:00:00 +0000\n"
            "Second commit\n"
        )
        with patch.object(gi, "_run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                ["git"], 0, stdout=log_output, stderr="",
            )
            commits = gi.log(n=5)
            assert len(commits) == 2
            assert commits[0].author == "Alice"
            assert commits[0].message == "Initial commit"
            assert commits[1].short_hash == "bbb222b"

    def test_log_error_returns_empty(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        gi = GitIntegration(tmp_path)
        with patch.object(gi, "_run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                ["git"], 128, stdout="", stderr="fatal",
            )
            assert gi.log() == []

    def test_log_incomplete_record_skipped(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        gi = GitIntegration(tmp_path)
        # 5 full lines + 3 incomplete => only 1 commit parsed
        log_output = (
            "hash1\nshort1\nauthor1\ndate1\nmessage1\n"
            "hash2\nshort2\nauthor2\n"
        )
        with patch.object(gi, "_run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                ["git"], 0, stdout=log_output, stderr="",
            )
            commits = gi.log()
            assert len(commits) == 1

    def test_log_empty_output(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        gi = GitIntegration(tmp_path)
        with patch.object(gi, "_run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                ["git"], 0, stdout="", stderr="",
            )
            assert gi.log() == []


# ---------------------------------------------------------------------------
# GitIntegration.current_branch
# ---------------------------------------------------------------------------

class TestCurrentBranch:
    """Tests for the current_branch() method."""

    def test_current_branch_not_a_repo(self, tmp_path: Path) -> None:
        gi = GitIntegration(tmp_path)
        assert gi.current_branch() == ""

    def test_current_branch_returns_name(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        gi = GitIntegration(tmp_path)
        with patch.object(gi, "_run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                ["git"], 0, stdout="feature/xyz\n", stderr="",
            )
            assert gi.current_branch() == "feature/xyz"

    def test_current_branch_error_returns_empty(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        gi = GitIntegration(tmp_path)
        with patch.object(gi, "_run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                ["git"], 1, stdout="", stderr="fatal",
            )
            assert gi.current_branch() == ""


# ---------------------------------------------------------------------------
# GitIntegration.checkpoint
# ---------------------------------------------------------------------------

class TestCheckpoint:
    """Tests for the checkpoint() method."""

    def test_checkpoint_not_a_repo(self, tmp_path: Path) -> None:
        gi = GitIntegration(tmp_path)
        assert gi.checkpoint() is None

    def test_checkpoint_clean_repo_returns_none(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        gi = GitIntegration(tmp_path)
        with patch.object(gi, "status") as mock_status:
            mock_status.return_value = GitStatus(is_clean=True)
            assert gi.checkpoint() is None

    def test_checkpoint_success(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        gi = GitIntegration(tmp_path)
        with patch.object(gi, "status") as mock_status, \
             patch.object(gi, "_run") as mock_run:
            mock_status.return_value = GitStatus(
                modified=["a.py"], is_clean=False,
            )
            # _run called for: add -A, commit, rev-parse HEAD
            mock_run.side_effect = [
                subprocess.CompletedProcess(["git"], 0, stdout="", stderr=""),
                subprocess.CompletedProcess(["git"], 0, stdout="", stderr=""),
                subprocess.CompletedProcess(["git"], 0, stdout="abc123def\n", stderr=""),
            ]
            result = gi.checkpoint("test-label")
            assert result == "abc123def"
            # Verify commit message includes prefix and label
            commit_call = mock_run.call_args_list[1]
            assert _CHECKPOINT_PREFIX in commit_call[0][2]
            assert "test-label" in commit_call[0][2]

    def test_checkpoint_commit_fails(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        gi = GitIntegration(tmp_path)
        with patch.object(gi, "status") as mock_status, \
             patch.object(gi, "_run") as mock_run:
            mock_status.return_value = GitStatus(
                modified=["a.py"], is_clean=False,
            )
            mock_run.side_effect = [
                subprocess.CompletedProcess(["git"], 0, stdout="", stderr=""),   # add
                subprocess.CompletedProcess(["git"], 1, stdout="", stderr="err"), # commit fails
            ]
            assert gi.checkpoint() is None


# ---------------------------------------------------------------------------
# GitIntegration.revert_checkpoint
# ---------------------------------------------------------------------------

class TestRevertCheckpoint:
    """Tests for the revert_checkpoint() method."""

    def test_revert_not_a_repo(self, tmp_path: Path) -> None:
        gi = GitIntegration(tmp_path)
        assert gi.revert_checkpoint("abc123") is False

    def test_revert_success(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        gi = GitIntegration(tmp_path)
        with patch.object(gi, "_run") as mock_run:
            mock_run.side_effect = [
                subprocess.CompletedProcess(["git"], 0, stdout="", stderr=""),  # revert
                subprocess.CompletedProcess(["git"], 0, stdout="", stderr=""),  # commit
            ]
            assert gi.revert_checkpoint("abc12345") is True

    def test_revert_failure_aborts(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        gi = GitIntegration(tmp_path)
        with patch.object(gi, "_run") as mock_run:
            mock_run.side_effect = [
                subprocess.CompletedProcess(["git"], 1, stdout="", stderr="conflict"),  # revert fails
                subprocess.CompletedProcess(["git"], 0, stdout="", stderr=""),  # abort
            ]
            assert gi.revert_checkpoint("abc12345") is False


# ---------------------------------------------------------------------------
# GitIntegration.list_checkpoints
# ---------------------------------------------------------------------------

class TestListCheckpoints:
    """Tests for the list_checkpoints() method."""

    def test_list_checkpoints_not_a_repo(self, tmp_path: Path) -> None:
        gi = GitIntegration(tmp_path)
        assert gi.list_checkpoints() == []

    def test_list_checkpoints_filters_and_limits(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        gi = GitIntegration(tmp_path)
        commits = [
            Commit("h1", "s1", "A", "d1", f"{_CHECKPOINT_PREFIX} cp1"),
            Commit("h2", "s2", "A", "d2", "Regular commit"),
            Commit("h3", "s3", "A", "d3", f"{_CHECKPOINT_PREFIX} cp2"),
        ]
        with patch.object(gi, "log", return_value=commits):
            cps = gi.list_checkpoints(n=1)
            assert len(cps) == 1
            assert cps[0].commit_hash == "h1"
            assert cps[0].label == "cp1"

    def test_list_checkpoints_strips_prefix(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        gi = GitIntegration(tmp_path)
        commits = [
            Commit("h1", "s1", "A", "d1", f"{_CHECKPOINT_PREFIX} my label"),
        ]
        with patch.object(gi, "log", return_value=commits):
            cps = gi.list_checkpoints()
            assert cps[0].label == "my label"
            assert cps[0].timestamp == "d1"


# ---------------------------------------------------------------------------
# GitIntegration.context_summary
# ---------------------------------------------------------------------------

class TestContextSummary:
    """Tests for the context_summary() method."""

    def test_context_summary_not_a_repo(self, tmp_path: Path) -> None:
        gi = GitIntegration(tmp_path)
        assert gi.context_summary() == "Not a git repository"

    def test_context_summary_includes_git_prefix(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        gi = GitIntegration(tmp_path)
        with patch.object(gi, "status") as mock_status:
            mock_status.return_value = GitStatus(branch="main", is_clean=True)
            s = gi.context_summary()
            assert s.startswith("Git:")
            assert "main" in s
            assert "clean" in s


# ---------------------------------------------------------------------------
# Integration test with real git (sample_git_repo fixture)
# ---------------------------------------------------------------------------

class TestGitIntegrationReal:
    """Integration tests using the sample_git_repo fixture with real git."""

    def test_is_git_repo_real(self, sample_git_repo: Path) -> None:
        gi = GitIntegration(sample_git_repo)
        assert gi.is_git_repo is True

    def test_status_real_clean(self, sample_git_repo: Path) -> None:
        gi = GitIntegration(sample_git_repo)
        st = gi.status()
        assert st.is_clean is True
        assert st.branch != ""

    def test_current_branch_real(self, sample_git_repo: Path) -> None:
        gi = GitIntegration(sample_git_repo)
        branch = gi.current_branch()
        # Initial branch is usually "main" or "master"
        assert branch in ("main", "master")

    def test_log_real(self, sample_git_repo: Path) -> None:
        gi = GitIntegration(sample_git_repo)
        commits = gi.log(n=5)
        assert len(commits) >= 1
        assert commits[0].message == "Initial commit"

    def test_diff_real_clean(self, sample_git_repo: Path) -> None:
        gi = GitIntegration(sample_git_repo)
        assert gi.diff() == ""

    def test_diff_real_with_changes(self, sample_git_repo: Path) -> None:
        (sample_git_repo / "src" / "app.py").write_text(
            "# changed\n", encoding="utf-8",
        )
        gi = GitIntegration(sample_git_repo)
        d = gi.diff()
        assert "changed" in d

    def test_checkpoint_real(self, sample_git_repo: Path) -> None:
        (sample_git_repo / "new_file.txt").write_text("hello\n", encoding="utf-8")
        gi = GitIntegration(sample_git_repo)
        commit_hash = gi.checkpoint("test-checkpoint")
        assert commit_hash is not None
        assert len(commit_hash) == 40  # full SHA

    def test_context_summary_real(self, sample_git_repo: Path) -> None:
        gi = GitIntegration(sample_git_repo)
        summary = gi.context_summary()
        assert "Git:" in summary
