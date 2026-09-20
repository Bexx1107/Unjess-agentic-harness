"""Background task manager — track and control long-running operations."""

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    KILLED = "killed"


@dataclass
class TaskInfo:
    """Information about a background task."""
    id: str
    name: str
    status: TaskStatus = TaskStatus.PENDING
    started_at: float = 0.0
    completed_at: float = 0.0
    output: str = ""
    error: str = ""
    _thread: Optional[threading.Thread] = field(default=None, repr=False)
    _kill_event: threading.Event = field(default_factory=threading.Event, repr=False)
    parent_conversation_id: str = ""

    @property
    def elapsed(self) -> float:
        if self.completed_at:
            return self.completed_at - self.started_at
        if self.started_at:
            return time.time() - self.started_at
        return 0.0

    @property 
    def is_alive(self) -> bool:
        return self.status in (TaskStatus.PENDING, TaskStatus.RUNNING)


class TaskManager:
    """Manage background tasks.
    
    Tracks running tasks, allows listing, killing, and retrieving output.
    Thread-safe.
    """

    def __init__(self) -> None:
        self._tasks: dict[str, TaskInfo] = {}
        self._lock = threading.Lock()

    def start_task(
        self,
        name: str,
        fn: Callable[..., str],
        *args: Any,
        task_id: str = "",
        **kwargs: Any,
    ) -> str:
        """Start a background task.
        
        Args:
            name: Human-readable task name.
            fn: Callable to run — should return a string result.
            task_id: Optional custom ID.
            
        Returns:
            Task ID.
        """
        tid = task_id or f"task-{uuid.uuid4().hex[:8]}"
        parent_id = getattr(self, "current_parent_id", "")
        info = TaskInfo(id=tid, name=name, parent_conversation_id=parent_id)
        
        def runner() -> None:
            info.status = TaskStatus.RUNNING
            info.started_at = time.time()
            try:
                result = fn(*args, **kwargs)
                if not info._kill_event.is_set():
                    info.output = result
                    info.status = TaskStatus.COMPLETED
            except Exception as exc:
                info.error = f"{type(exc).__name__}: {exc}"
                info.status = TaskStatus.FAILED
                logger.warning("Task %s failed: %s", tid, exc)
            finally:
                info.completed_at = time.time()
        
        thread = threading.Thread(target=runner, daemon=True, name=f"task-{tid}")
        info._thread = thread
        
        with self._lock:
            self._tasks[tid] = info
        
        thread.start()
        return tid

    def list_tasks(self, active_only: bool = False) -> list[TaskInfo]:
        """List all tasks, optionally only active ones."""
        with self._lock:
            tasks = list(self._tasks.values())
        if active_only:
            tasks = [t for t in tasks if t.is_alive]
        return sorted(tasks, key=lambda t: t.started_at, reverse=True)

    def get_task(self, task_id: str) -> Optional[TaskInfo]:
        """Get task info by ID."""
        with self._lock:
            return self._tasks.get(task_id)

    def kill_task(self, task_id: str) -> bool:
        """Kill a running task.
        
        Returns True if the task was found and killed.
        """
        with self._lock:
            task = self._tasks.get(task_id)
        if not task or not task.is_alive:
            return False
        task._kill_event.set()
        task.status = TaskStatus.KILLED
        task.completed_at = time.time()
        logger.info("Killed task %s (%s)", task_id, task.name)
        return True

    def kill_all(self) -> int:
        """Kill all active tasks. Returns count killed."""
        count = 0
        with self._lock:
            for task in self._tasks.values():
                if task.is_alive:
                    task._kill_event.set()
                    task.status = TaskStatus.KILLED
                    task.completed_at = time.time()
                    count += 1
        return count

    def cleanup(self, max_age: float = 3600) -> int:
        """Remove completed tasks older than max_age seconds."""
        now = time.time()
        to_remove = []
        with self._lock:
            for tid, task in self._tasks.items():
                if not task.is_alive and (now - task.completed_at) > max_age:
                    to_remove.append(tid)
            for tid in to_remove:
                del self._tasks[tid]
        return len(to_remove)

    def get_status_summary(self) -> str:
        """Get a formatted status summary of all tasks."""
        tasks = self.list_tasks()
        if not tasks:
            return "No tasks."
        
        lines = []
        for t in tasks:
            status_icon = {
                TaskStatus.PENDING: "⏳",
                TaskStatus.RUNNING: "🔄",
                TaskStatus.COMPLETED: "✅",
                TaskStatus.FAILED: "❌",
                TaskStatus.KILLED: "🛑",
            }.get(t.status, "?")
            elapsed = f"{t.elapsed:.1f}s" if t.elapsed else ""
            lines.append(f"  {status_icon} {t.id}: {t.name} [{t.status.value}] {elapsed}")
        return "\n".join(lines)

    def __len__(self) -> int:
        with self._lock:
            return len(self._tasks)
