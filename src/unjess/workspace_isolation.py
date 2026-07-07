"""Workspace isolation — branching, sharing, and sandboxing for subagents.

Provides 4 isolation strategies:
- inherit: Same workspace directory (no isolation)
- branch: Git worktree or full copy (full isolation)
- share: Same repo, different branch (shared storage)
- sandbox: Temporary directory with copied files (disposable)
"""

import logging
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class IsolatedWorkspace:
    """An isolated workspace created for a subagent."""

    path: Path
    mode: str  # "inherit", "branch", "share", "sandbox"
    parent_workspace: Path = Path(".")
    branch_name: str = ""
    is_temporary: bool = False

    @property
    def is_git_worktree(self) -> bool:
        """Whether this is a git worktree."""
        return self.mode == "branch" and self.branch_name != ""


# ---------------------------------------------------------------------------
# Workspace Isolator
# ---------------------------------------------------------------------------

class WorkspaceIsolator:
    """Creates and manages isolated workspaces for subagents.

    Args:
        parent_workspace: The parent agent's workspace root.
    """

    def __init__(self, parent_workspace: Path) -> None:
        self._parent = parent_workspace
        self._workspaces: dict[str, IsolatedWorkspace] = {}

    def create(self, mode: str, agent_id: str = "") -> IsolatedWorkspace:
        """Create an isolated workspace.

        Args:
            mode: Isolation strategy ("inherit", "branch", "share", "sandbox").
            agent_id: Agent conversation ID (for naming).

        Returns:
            IsolatedWorkspace with the path to use.
        """
        short_id = agent_id[:8] if agent_id else uuid.uuid4().hex[:8]

        if mode == "inherit":
            ws = IsolatedWorkspace(
                path=self._parent,
                mode="inherit",
                parent_workspace=self._parent,
            )

        elif mode == "branch":
            ws = self._create_branch(short_id)

        elif mode == "share":
            ws = self._create_shared(short_id)

        elif mode == "sandbox":
            ws = self._create_sandbox(short_id)

        else:
            logger.warning("Unknown isolation mode '%s', falling back to inherit", mode)
            ws = IsolatedWorkspace(
                path=self._parent,
                mode="inherit",
                parent_workspace=self._parent,
            )

        self._workspaces[agent_id or short_id] = ws
        return ws

    def cleanup(self, agent_id: str) -> bool:
        """Clean up an isolated workspace.

        Only removes temporary workspaces (sandbox, branch).
        Does not touch inherited or shared workspaces.

        Args:
            agent_id: Agent whose workspace to clean up.

        Returns:
            True if cleanup was performed.
        """
        ws = self._workspaces.pop(agent_id, None)
        if not ws:
            return False

        if not ws.is_temporary:
            return False

        if ws.is_git_worktree:
            return self._remove_worktree(ws)

        if ws.path.exists() and ws.path != self._parent:
            try:
                shutil.rmtree(ws.path)
                logger.info("Cleaned up workspace: %s", ws.path)
                return True
            except OSError as exc:
                logger.warning("Failed to clean up workspace: %s", exc)

        return False

    def cleanup_all(self) -> int:
        """Clean up all temporary workspaces.

        Returns:
            Number of workspaces cleaned up.
        """
        count = 0
        for agent_id in list(self._workspaces.keys()):
            if self.cleanup(agent_id):
                count += 1
        return count

    def merge_back(self, agent_id: str) -> tuple[bool, str]:
        """Merge a branched workspace back to the parent.

        Only works for "branch" and "share" modes with git.

        Args:
            agent_id: Agent whose workspace to merge.

        Returns:
            (success, message)
        """
        ws = self._workspaces.get(agent_id)
        if not ws:
            return False, "Workspace not found"

        if ws.mode not in ("branch", "share"):
            return False, "Only branch/share workspaces can be merged"

        if not ws.branch_name:
            return False, "No branch to merge"

        try:
            # Merge the branch back
            result = subprocess.run(
                ["git", "merge", ws.branch_name, "--no-edit"],
                capture_output=True,
                text=True,
                cwd=str(self._parent),
            )
            if result.returncode != 0:
                return False, f"Merge failed: {result.stderr.strip()}"

            return True, f"Merged branch '{ws.branch_name}' back to parent"

        except Exception as exc:
            return False, f"Merge error: {exc}"

    # ----- Internal -----

    def _create_branch(self, short_id: str) -> IsolatedWorkspace:
        """Create a git worktree branch, or fall back to directory copy."""
        branch_name = f"njss-agent-{short_id}"
        worktree_dir = self._parent / ".unjess" / "worktrees" / short_id

        # Try git worktree first
        if (self._parent / ".git").exists():
            try:
                worktree_dir.parent.mkdir(parents=True, exist_ok=True)
                result = subprocess.run(
                    ["git", "worktree", "add", str(worktree_dir), "-b", branch_name],
                    capture_output=True,
                    text=True,
                    cwd=str(self._parent),
                )
                if result.returncode == 0:
                    logger.info("Created git worktree: %s", worktree_dir)
                    return IsolatedWorkspace(
                        path=worktree_dir,
                        mode="branch",
                        parent_workspace=self._parent,
                        branch_name=branch_name,
                        is_temporary=True,
                    )
            except Exception as exc:
                logger.debug("Git worktree failed, falling back to copy: %s", exc)

        # Fall back to directory copy
        copy_dir = self._parent / ".unjess" / "workspaces" / short_id
        copy_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Copy workspace (excluding .git, .unjess, node_modules, etc.)
            self._copy_workspace(self._parent, copy_dir)
            logger.info("Created workspace copy: %s", copy_dir)
        except Exception as exc:
            logger.warning("Workspace copy failed: %s", exc)

        return IsolatedWorkspace(
            path=copy_dir,
            mode="branch",
            parent_workspace=self._parent,
            is_temporary=True,
        )

    def _create_shared(self, short_id: str) -> IsolatedWorkspace:
        """Create a shared workspace (same directory, different git branch)."""
        if not (self._parent / ".git").exists():
            # No git — fall back to inherit
            return IsolatedWorkspace(
                path=self._parent,
                mode="inherit",
                parent_workspace=self._parent,
            )

        branch_name = f"njss-agent-{short_id}"

        try:
            subprocess.run(
                ["git", "branch", branch_name],
                capture_output=True,
                text=True,
                cwd=str(self._parent),
            )
        except Exception:
            pass

        return IsolatedWorkspace(
            path=self._parent,
            mode="share",
            parent_workspace=self._parent,
            branch_name=branch_name,
        )

    def _create_sandbox(self, short_id: str) -> IsolatedWorkspace:
        """Create a temporary sandbox directory."""
        sandbox_dir = self._parent / ".unjess" / "sandboxes" / short_id
        sandbox_dir.mkdir(parents=True, exist_ok=True)

        return IsolatedWorkspace(
            path=sandbox_dir,
            mode="sandbox",
            parent_workspace=self._parent,
            is_temporary=True,
        )

    def _remove_worktree(self, ws: IsolatedWorkspace) -> bool:
        """Remove a git worktree."""
        try:
            subprocess.run(
                ["git", "worktree", "remove", str(ws.path), "--force"],
                capture_output=True,
                text=True,
                cwd=str(ws.parent_workspace),
            )
            # Delete the branch too
            if ws.branch_name:
                subprocess.run(
                    ["git", "branch", "-D", ws.branch_name],
                    capture_output=True,
                    text=True,
                    cwd=str(ws.parent_workspace),
                )
            logger.info("Removed git worktree: %s", ws.path)
            return True
        except Exception as exc:
            logger.warning("Failed to remove worktree: %s", exc)
            return False

    def _copy_workspace(self, src: Path, dst: Path) -> None:
        """Copy a workspace, skipping heavy directories."""
        skip_dirs = {".git", ".unjess", "node_modules", "__pycache__", ".venv", "venv", "target", "dist", "build"}

        for item in src.iterdir():
            if item.name in skip_dirs:
                continue

            dest = dst / item.name
            try:
                if item.is_dir():
                    shutil.copytree(item, dest, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, dest)
            except (OSError, shutil.Error) as exc:
                logger.debug("Skip copying %s: %s", item.name, exc)
