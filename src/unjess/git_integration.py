"""Git integration — status, diff, log, checkpoints, and branch info.

All git operations shell out to the ``git`` CLI via subprocess.
"""

import logging
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_CHECKPOINT_PREFIX = "[njss-checkpoint]"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GitStatus:
    """Parsed output of ``git status``."""

    branch: str = ""
    modified: list[str] = field(default_factory=list)
    staged: list[str] = field(default_factory=list)
    untracked: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    is_clean: bool = True

    def summary(self) -> str:
        """One-line summary of the git status."""
        if self.is_clean:
            return f"on {self.branch} — clean"
        parts = []
        if self.modified:
            parts.append(f"{len(self.modified)} modified")
        if self.staged:
            parts.append(f"{len(self.staged)} staged")
        if self.untracked:
            parts.append(f"{len(self.untracked)} untracked")
        if self.deleted:
            parts.append(f"{len(self.deleted)} deleted")
        return f"on {self.branch} — {', '.join(parts)}"


@dataclass
class Commit:
    """A single git commit."""

    hash: str
    short_hash: str
    author: str
    date: str
    message: str

    @property
    def is_checkpoint(self) -> bool:
        """Whether this commit is an njss checkpoint."""
        return _CHECKPOINT_PREFIX in self.message


@dataclass
class Checkpoint:
    """An njss auto-checkpoint."""

    commit_hash: str
    label: str
    timestamp: str


# ---------------------------------------------------------------------------
# GitIntegration
# ---------------------------------------------------------------------------

class GitIntegration:
    """Wrapper around git CLI operations.

    All methods are safe to call on non-git directories — they will
    return empty/default results.

    Args:
        workspace: The project root (must contain a ``.git`` directory for git ops).
    """

    def __init__(self, workspace: Path) -> None:
        self._workspace = workspace
        self._is_repo: Optional[bool] = None

    @property
    def is_git_repo(self) -> bool:
        """Check if the workspace is a git repository."""
        if self._is_repo is None:
            self._is_repo = (self._workspace / ".git").exists()
        return self._is_repo

    def _run(self, *args: str, check: bool = False) -> subprocess.CompletedProcess[str]:
        """Run a git command in the workspace directory."""
        cmd = ["git"] + list(args)
        try:
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self._workspace),
                timeout=10,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            logger.warning("Git command failed: %s", exc)
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr=str(exc))

    # ----- Status -----

    def status(self) -> GitStatus:
        """Get the current git status."""
        if not self.is_git_repo:
            return GitStatus()

        result = self._run("status", "--porcelain=v1", "--branch")
        if result.returncode != 0:
            return GitStatus()

        status = GitStatus()
        lines = result.stdout.strip().splitlines()

        for line in lines:
            if line.startswith("##"):
                # Branch line: ## main...origin/main
                branch_part = line[3:].split("...")[0]
                status.branch = branch_part.strip()
                continue

            if len(line) < 4:
                continue

            indicator = line[:2]
            filepath = line[3:].strip()

            if indicator[0] in ("M", "A", "R", "C"):
                status.staged.append(filepath)
            if indicator[1] == "M":
                status.modified.append(filepath)
            elif indicator[1] == "D":
                status.deleted.append(filepath)
            elif indicator == "??":
                status.untracked.append(filepath)

        status.is_clean = not (status.modified or status.staged or status.untracked or status.deleted)
        return status

    # ----- Diff -----

    def diff(self, file: Optional[str] = None) -> str:
        """Show uncommitted changes (unstaged + staged).

        Args:
            file: Optional file path to restrict the diff.

        Returns:
            Unified diff text, or empty string.
        """
        if not self.is_git_repo:
            return ""

        args = ["diff", "HEAD"]
        if file:
            args.extend(["--", file])

        result = self._run(*args)
        return result.stdout if result.returncode == 0 else ""

    # ----- Log -----

    def log(self, n: int = 10) -> list[Commit]:
        """Get the last N commits.

        Args:
            n: Number of commits to return.

        Returns:
            List of Commit objects, newest first.
        """
        if not self.is_git_repo:
            return []

        fmt = "%H%n%h%n%an%n%ai%n%s"
        result = self._run("log", f"-{n}", f"--format={fmt}", "--no-merges")

        if result.returncode != 0:
            return []

        commits: list[Commit] = []
        lines = result.stdout.strip().splitlines()

        # Each commit is 5 lines
        for i in range(0, len(lines), 5):
            if i + 4 >= len(lines):
                break
            commits.append(Commit(
                hash=lines[i],
                short_hash=lines[i + 1],
                author=lines[i + 2],
                date=lines[i + 3],
                message=lines[i + 4],
            ))

        return commits

    # ----- Branch -----

    def current_branch(self) -> str:
        """Get the current branch name."""
        if not self.is_git_repo:
            return ""

        result = self._run("rev-parse", "--abbrev-ref", "HEAD")
        return result.stdout.strip() if result.returncode == 0 else ""

    # ----- Checkpoints -----

    def checkpoint(self, label: str = "auto") -> Optional[str]:
        """Create a silent checkpoint commit.

        Stages all changes and commits with a ``[njss-checkpoint]`` prefix.
        Returns the commit hash, or None if nothing to commit.
        """
        if not self.is_git_repo:
            return None

        # Check if there's anything to commit
        st = self.status()
        if st.is_clean:
            return None

        # Stage everything
        self._run("add", "-A")

        # Commit
        message = f"{_CHECKPOINT_PREFIX} {label}"
        result = self._run("commit", "-m", message, "--no-verify")

        if result.returncode != 0:
            logger.warning("Checkpoint commit failed: %s", result.stderr.strip())
            return None

        # Get the commit hash
        hash_result = self._run("rev-parse", "HEAD")
        commit_hash = hash_result.stdout.strip() if hash_result.returncode == 0 else ""

        logger.info("Created checkpoint: %s (%s)", label, commit_hash[:8])
        return commit_hash

    def revert_checkpoint(self, commit_hash: str) -> bool:
        """Revert to a checkpoint by resetting to the commit before it.

        Uses ``git revert --no-commit`` to undo the checkpoint's changes
        without losing history.

        Args:
            commit_hash: The checkpoint commit to revert.

        Returns:
            True if successful.
        """
        if not self.is_git_repo:
            return False

        # Revert the checkpoint commit (undoes its changes)
        result = self._run("revert", "--no-commit", commit_hash)
        if result.returncode != 0:
            logger.warning("Revert failed: %s", result.stderr.strip())
            # Try hard reset as fallback
            self._run("revert", "--abort")
            return False

        # Commit the revert
        self._run("commit", "-m", f"{_CHECKPOINT_PREFIX} revert {commit_hash[:8]}")
        return True

    def list_checkpoints(self, n: int = 20) -> list[Checkpoint]:
        """List recent njss checkpoints.

        Args:
            n: Max number of checkpoints to return.

        Returns:
            List of Checkpoint objects, newest first.
        """
        if not self.is_git_repo:
            return []

        commits = self.log(n=50)
        checkpoints: list[Checkpoint] = []

        for commit in commits:
            if commit.is_checkpoint:
                label = commit.message.replace(_CHECKPOINT_PREFIX, "").strip()
                checkpoints.append(Checkpoint(
                    commit_hash=commit.hash,
                    label=label,
                    timestamp=commit.date,
                ))
                if len(checkpoints) >= n:
                    break

        return checkpoints

    # ----- Context info for system prompt -----

    def context_summary(self) -> str:
        """Build a short git context string for the system prompt."""
        if not self.is_git_repo:
            return "Not a git repository"

        st = self.status()
        return f"Git: {st.summary()}"
