"""Undo / rollback system — hybrid git-based and file-snapshot strategy."""

import logging
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from unjess.git_integration import GitIntegration

logger = logging.getLogger(__name__)

_SNAPSHOT_DIR = Path.home() / ".unjess" / "snapshots"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class UndoEntry:
    """A single undoable operation."""

    id: str  # checkpoint hash or snapshot timestamp
    label: str
    timestamp: str
    source: str  # "git" or "snapshot"

    def display(self) -> str:
        """Human-readable display string."""
        short_id = self.id[:8] if len(self.id) > 8 else self.id
        return f"[{short_id}] {self.label} ({self.timestamp})"


# ---------------------------------------------------------------------------
# Undo manager
# ---------------------------------------------------------------------------

class UndoManager:
    """Manages undo/rollback operations.

    Uses a hybrid strategy:
    - **Git repos**: Creates checkpoint commits before agent edits,
      reverts them on undo.
    - **Non-git dirs**: Copies modified files to a snapshot directory
      before edits, restores them on undo.

    Args:
        workspace: The project root.
        git: Git integration instance.
    """

    def __init__(self, workspace: Path, git: GitIntegration) -> None:
        self._workspace = workspace
        self._git = git
        self._snapshot_stack: list[str] = []  # timestamps of file snapshots

    @property
    def uses_git(self) -> bool:
        """Whether this workspace uses git-based undo."""
        return self._git.is_git_repo

    # ----- Checkpoint -----

    def checkpoint(self, label: str = "auto") -> Optional[str]:
        """Save state before agent makes changes.

        Args:
            label: Human-readable label for the checkpoint.

        Returns:
            Checkpoint ID (commit hash or snapshot timestamp), or None.
        """
        if self.uses_git:
            return self._git_checkpoint(label)
        return self._snapshot_checkpoint(label)

    def _git_checkpoint(self, label: str) -> Optional[str]:
        """Create a git checkpoint commit."""
        return self._git.checkpoint(label)

    def _snapshot_checkpoint(self, label: str) -> Optional[str]:
        """Snapshot modified files to the snapshot directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snap_dir = _SNAPSHOT_DIR / timestamp

        try:
            snap_dir.mkdir(parents=True, exist_ok=True)

            # Copy all files in workspace (excluding hidden dirs and common ignores)
            file_count = 0
            manifest_paths: list[str] = []
            for fpath in self._workspace.rglob("*"):
                if not fpath.is_file():
                    continue
                if any(p.startswith(".") for p in fpath.relative_to(self._workspace).parts):
                    continue
                if any(p in ("node_modules", "__pycache__", "venv", ".venv", "dist", "build")
                       for p in fpath.relative_to(self._workspace).parts):
                    continue

                rel = fpath.relative_to(self._workspace)
                dest = snap_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(fpath), str(dest))
                manifest_paths.append(str(rel).replace("\\", "/"))
                file_count += 1

            # Write metadata (includes manifest of all files at snapshot time)
            meta_file = snap_dir / ".undo_meta"
            meta_file.write_text(
                f"label={label}\n"
                f"timestamp={timestamp}\n"
                f"files={file_count}\n"
                f"workspace={self._workspace}\n"
                f"manifest={'|'.join(manifest_paths)}\n",
                encoding="utf-8",
            )

            self._snapshot_stack.append(timestamp)
            logger.info("Created snapshot: %s (%d files)", label, file_count)
            return timestamp

        except Exception as exc:
            logger.warning("Snapshot failed: %s", exc)
            return None

    # ----- Undo -----

    def undo(self, steps: int = 1) -> tuple[bool, str]:
        """Revert the last N agent operations.

        Args:
            steps: Number of operations to undo.

        Returns:
            (success, message)
        """
        if self.uses_git:
            return self._git_undo(steps)
        return self._snapshot_undo(steps)

    def _git_undo(self, steps: int) -> tuple[bool, str]:
        """Undo using git checkpoint revert."""
        checkpoints = self._git.list_checkpoints(n=steps)

        if not checkpoints:
            return False, "No checkpoints found to undo."

        reverted = 0
        for cp in checkpoints[:steps]:
            if self._git.revert_checkpoint(cp.commit_hash):
                reverted += 1
            else:
                break

        if reverted == 0:
            return False, "Failed to revert any checkpoints."

        return True, f"Reverted {reverted} checkpoint(s)."

    # Dirs to skip when scanning for files created after a snapshot
    _SKIP_DIRS = {".git", ".unjess", "node_modules", "__pycache__", "venv", ".venv", "dist", "build"}

    def _snapshot_undo(self, steps: int) -> tuple[bool, str]:
        """Undo using file snapshot restoration."""
        if not self._snapshot_stack:
            # Check if there are snapshots on disk
            if not _SNAPSHOT_DIR.exists():
                return False, "No snapshots found to undo."

            snap_dirs = sorted(_SNAPSHOT_DIR.iterdir(), reverse=True)
            if not snap_dirs:
                return False, "No snapshots found to undo."

            timestamp = snap_dirs[0].name
        else:
            timestamp = self._snapshot_stack.pop()

        snap_dir = _SNAPSHOT_DIR / timestamp
        if not snap_dir.exists():
            return False, f"Snapshot not found: {timestamp}"

        # Restore files
        restored = 0
        try:
            for fpath in snap_dir.rglob("*"):
                if not fpath.is_file() or fpath.name == ".undo_meta":
                    continue

                rel = fpath.relative_to(snap_dir)
                dest = self._workspace / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(fpath), str(dest))
                restored += 1
        except Exception as exc:
            return False, f"Restore failed: {exc}"

        # Delete files created after the snapshot
        deleted = self._delete_new_files(snap_dir)

        msg = f"Restored {restored} files from snapshot {timestamp}."
        if deleted:
            msg += f" Deleted {deleted} file(s) created after snapshot."
        return True, msg

    def _delete_new_files(self, snap_dir: Path) -> int:
        """Delete files that were created after the snapshot was taken.

        Reads the manifest from .undo_meta and removes any workspace file
        not present in it (skipping hidden dirs, node_modules, etc.).

        Args:
            snap_dir: Path to the snapshot directory.

        Returns:
            Number of files deleted.
        """
        # Read manifest from .undo_meta
        meta_file = snap_dir / ".undo_meta"
        if not meta_file.exists():
            return 0

        manifest: set[str] = set()
        for line in meta_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("manifest="):
                raw = line.split("=", 1)[1]
                if raw:
                    manifest = set(raw.split("|"))
                break

        if not manifest:
            return 0

        deleted = 0
        for fpath in self._workspace.rglob("*"):
            if not fpath.is_file():
                continue

            parts = fpath.relative_to(self._workspace).parts
            # Skip hidden dirs and common build/venv dirs
            if any(p.startswith(".") or p in self._SKIP_DIRS for p in parts):
                continue

            rel = str(fpath.relative_to(self._workspace)).replace("\\", "/")
            if rel not in manifest:
                try:
                    fpath.unlink()
                    logger.info("Deleted new file: %s", rel)
                    deleted += 1
                except OSError as exc:
                    logger.debug("Could not delete %s: %s", rel, exc)

        if deleted:
            logger.info("Deleted %d file(s) created after snapshot", deleted)
        return deleted

    # ----- List checkpoints -----

    def list_entries(self, n: int = 10) -> list[UndoEntry]:
        """List recent undo-able entries.

        Args:
            n: Max entries to return.

        Returns:
            List of UndoEntry objects, newest first.
        """
        entries: list[UndoEntry] = []

        if self.uses_git:
            for cp in self._git.list_checkpoints(n=n):
                entries.append(UndoEntry(
                    id=cp.commit_hash,
                    label=cp.label,
                    timestamp=cp.timestamp,
                    source="git",
                ))
        else:
            if _SNAPSHOT_DIR.exists():
                snap_dirs = sorted(_SNAPSHOT_DIR.iterdir(), reverse=True)[:n]
                for snap_dir in snap_dirs:
                    meta_file = snap_dir / ".undo_meta"
                    label = "snapshot"
                    if meta_file.exists():
                        for line in meta_file.read_text(encoding="utf-8").splitlines():
                            if line.startswith("label="):
                                label = line.split("=", 1)[1]
                                break
                    entries.append(UndoEntry(
                        id=snap_dir.name,
                        label=label,
                        timestamp=snap_dir.name,
                        source="snapshot",
                    ))

        return entries
