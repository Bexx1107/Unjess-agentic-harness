"""Tests for unjess.task_manager — background task lifecycle."""

import threading
import time
from typing import Any

import pytest

from unjess.task_manager import TaskInfo, TaskManager, TaskStatus


# ---------------------------------------------------------------------------
# TaskInfo dataclass
# ---------------------------------------------------------------------------

class TestTaskInfo:
    """Tests for the TaskInfo dataclass and its computed properties."""

    def test_default_status_is_pending(self) -> None:
        t = TaskInfo(id="t1", name="test")
        assert t.status == TaskStatus.PENDING

    def test_is_alive_when_pending(self) -> None:
        t = TaskInfo(id="t1", name="test", status=TaskStatus.PENDING)
        assert t.is_alive is True

    def test_is_alive_when_running(self) -> None:
        t = TaskInfo(id="t1", name="test", status=TaskStatus.RUNNING)
        assert t.is_alive is True

    def test_not_alive_when_completed(self) -> None:
        t = TaskInfo(id="t1", name="test", status=TaskStatus.COMPLETED)
        assert t.is_alive is False

    def test_not_alive_when_failed(self) -> None:
        t = TaskInfo(id="t1", name="test", status=TaskStatus.FAILED)
        assert t.is_alive is False

    def test_not_alive_when_killed(self) -> None:
        t = TaskInfo(id="t1", name="test", status=TaskStatus.KILLED)
        assert t.is_alive is False

    def test_elapsed_zero_before_start(self) -> None:
        t = TaskInfo(id="t1", name="test")
        assert t.elapsed == 0.0

    def test_elapsed_computed_while_running(self) -> None:
        t = TaskInfo(id="t1", name="test", started_at=time.time() - 2.0)
        assert t.elapsed >= 1.5

    def test_elapsed_fixed_when_completed(self) -> None:
        now = time.time()
        t = TaskInfo(id="t1", name="test", started_at=now - 5.0, completed_at=now)
        assert abs(t.elapsed - 5.0) < 0.1


# ---------------------------------------------------------------------------
# TaskStatus enum
# ---------------------------------------------------------------------------

class TestTaskStatus:
    """Tests for the TaskStatus enum values."""

    def test_all_status_values(self) -> None:
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.RUNNING.value == "running"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"
        assert TaskStatus.KILLED.value == "killed"


# ---------------------------------------------------------------------------
# TaskManager — start_task
# ---------------------------------------------------------------------------

class TestStartTask:
    """Tests for starting tasks via TaskManager."""

    def test_start_returns_task_id(self) -> None:
        tm = TaskManager()
        tid = tm.start_task("greet", lambda: "hello")
        assert isinstance(tid, str)
        assert len(tid) > 0

    def test_custom_task_id(self) -> None:
        tm = TaskManager()
        tid = tm.start_task("greet", lambda: "hello", task_id="my-task")
        assert tid == "my-task"

    def test_task_completes_with_output(self) -> None:
        tm = TaskManager()
        tid = tm.start_task("greet", lambda: "hello world")
        # Wait for completion
        task = tm.get_task(tid)
        assert task is not None
        task._thread.join(timeout=5.0)
        assert task.status == TaskStatus.COMPLETED
        assert task.output == "hello world"

    def test_task_with_args(self) -> None:
        def adder(a: int, b: int) -> str:
            return str(a + b)

        tm = TaskManager()
        tid = tm.start_task("add", adder, 3, 7)
        task = tm.get_task(tid)
        assert task is not None
        task._thread.join(timeout=5.0)
        assert task.output == "10"

    def test_task_with_kwargs(self) -> None:
        def greeter(greeting: str = "Hello") -> str:
            return f"{greeting}, World!"

        tm = TaskManager()
        tid = tm.start_task("greet", greeter, greeting="Hi")
        task = tm.get_task(tid)
        assert task is not None
        task._thread.join(timeout=5.0)
        assert task.output == "Hi, World!"

    def test_task_failure_sets_failed_status(self) -> None:
        def failing() -> str:
            raise ValueError("boom")

        tm = TaskManager()
        tid = tm.start_task("fail", failing)
        task = tm.get_task(tid)
        assert task is not None
        task._thread.join(timeout=5.0)
        assert task.status == TaskStatus.FAILED
        assert "ValueError" in task.error
        assert "boom" in task.error

    def test_task_sets_timestamps(self) -> None:
        tm = TaskManager()
        before = time.time()
        tid = tm.start_task("quick", lambda: "done")
        task = tm.get_task(tid)
        assert task is not None
        task._thread.join(timeout=5.0)
        assert task.started_at >= before
        assert task.completed_at >= task.started_at

    def test_auto_generated_id_has_prefix(self) -> None:
        tm = TaskManager()
        tid = tm.start_task("test", lambda: "ok")
        assert tid.startswith("task-")


# ---------------------------------------------------------------------------
# TaskManager — get_task
# ---------------------------------------------------------------------------

class TestGetTask:
    """Tests for task retrieval."""

    def test_get_existing_task(self) -> None:
        tm = TaskManager()
        tid = tm.start_task("test", lambda: "ok")
        task = tm.get_task(tid)
        assert task is not None
        assert task.name == "test"

    def test_get_nonexistent_task_returns_none(self) -> None:
        tm = TaskManager()
        assert tm.get_task("nonexistent") is None


# ---------------------------------------------------------------------------
# TaskManager — list_tasks
# ---------------------------------------------------------------------------

class TestListTasks:
    """Tests for listing tasks."""

    def test_list_empty(self) -> None:
        tm = TaskManager()
        assert tm.list_tasks() == []

    def test_list_all_tasks(self) -> None:
        tm = TaskManager()
        tm.start_task("t1", lambda: "a")
        tm.start_task("t2", lambda: "b")
        tasks = tm.list_tasks()
        assert len(tasks) == 2

    def test_list_active_only(self) -> None:
        tm = TaskManager()
        blocker = threading.Event()
        tid1 = tm.start_task("slow", lambda: (blocker.wait(timeout=10), "done")[1])
        tid2 = tm.start_task("fast", lambda: "done")
        # Wait for fast task to complete
        tm.get_task(tid2)._thread.join(timeout=5.0)
        active = tm.list_tasks(active_only=True)
        # slow should still be active
        active_ids = [t.id for t in active]
        assert tid2 not in active_ids
        # Cleanup
        blocker.set()
        tm.get_task(tid1)._thread.join(timeout=5.0)

    def test_list_sorted_by_started_at_descending(self) -> None:
        tm = TaskManager()
        tid1 = tm.start_task("first", lambda: "a")
        tm.get_task(tid1)._thread.join(timeout=5.0)
        time.sleep(0.01)
        tid2 = tm.start_task("second", lambda: "b")
        tm.get_task(tid2)._thread.join(timeout=5.0)
        tasks = tm.list_tasks()
        assert tasks[0].started_at >= tasks[1].started_at


# ---------------------------------------------------------------------------
# TaskManager — kill_task
# ---------------------------------------------------------------------------

class TestKillTask:
    """Tests for killing tasks."""

    def test_kill_active_task(self) -> None:
        tm = TaskManager()
        blocker = threading.Event()
        tid = tm.start_task("slow", lambda: (blocker.wait(timeout=10), "done")[1])
        time.sleep(0.05)  # let it start
        result = tm.kill_task(tid)
        assert result is True
        task = tm.get_task(tid)
        assert task is not None
        assert task.status == TaskStatus.KILLED
        assert task.completed_at > 0
        blocker.set()

    def test_kill_nonexistent_task_returns_false(self) -> None:
        tm = TaskManager()
        assert tm.kill_task("nonexistent") is False

    def test_kill_completed_task_returns_false(self) -> None:
        tm = TaskManager()
        tid = tm.start_task("fast", lambda: "done")
        tm.get_task(tid)._thread.join(timeout=5.0)
        assert tm.kill_task(tid) is False

    def test_kill_sets_kill_event(self) -> None:
        tm = TaskManager()
        blocker = threading.Event()
        tid = tm.start_task("slow", lambda: (blocker.wait(timeout=10), "done")[1])
        time.sleep(0.05)
        tm.kill_task(tid)
        task = tm.get_task(tid)
        assert task is not None
        assert task._kill_event.is_set()
        blocker.set()


# ---------------------------------------------------------------------------
# TaskManager — kill_all
# ---------------------------------------------------------------------------

class TestKillAll:
    """Tests for killing all active tasks at once."""

    def test_kill_all_returns_count(self) -> None:
        tm = TaskManager()
        blocker = threading.Event()
        tm.start_task("a", lambda: (blocker.wait(timeout=10), "a")[1])
        tm.start_task("b", lambda: (blocker.wait(timeout=10), "b")[1])
        time.sleep(0.05)
        count = tm.kill_all()
        assert count == 2
        blocker.set()

    def test_kill_all_on_empty_manager(self) -> None:
        tm = TaskManager()
        assert tm.kill_all() == 0

    def test_kill_all_skips_completed(self) -> None:
        tm = TaskManager()
        tid = tm.start_task("fast", lambda: "done")
        tm.get_task(tid)._thread.join(timeout=5.0)
        blocker = threading.Event()
        tm.start_task("slow", lambda: (blocker.wait(timeout=10), "slow")[1])
        time.sleep(0.05)
        count = tm.kill_all()
        assert count == 1
        blocker.set()


# ---------------------------------------------------------------------------
# TaskManager — cleanup
# ---------------------------------------------------------------------------

class TestCleanup:
    """Tests for cleaning up old completed tasks."""

    def test_cleanup_removes_old_tasks(self) -> None:
        tm = TaskManager()
        tid = tm.start_task("old", lambda: "done")
        task = tm.get_task(tid)
        assert task is not None
        task._thread.join(timeout=5.0)
        # Backdate the completion
        task.completed_at = time.time() - 7200  # 2 hours ago
        removed = tm.cleanup(max_age=3600)
        assert removed == 1
        assert tm.get_task(tid) is None

    def test_cleanup_keeps_recent_tasks(self) -> None:
        tm = TaskManager()
        tid = tm.start_task("recent", lambda: "done")
        tm.get_task(tid)._thread.join(timeout=5.0)
        removed = tm.cleanup(max_age=3600)
        assert removed == 0
        assert tm.get_task(tid) is not None

    def test_cleanup_keeps_active_tasks(self) -> None:
        tm = TaskManager()
        blocker = threading.Event()
        tid = tm.start_task("active", lambda: (blocker.wait(timeout=10), "ok")[1])
        time.sleep(0.05)
        removed = tm.cleanup(max_age=0)
        assert removed == 0
        blocker.set()
        tm.get_task(tid)._thread.join(timeout=5.0)


# ---------------------------------------------------------------------------
# TaskManager — __len__ and get_status_summary
# ---------------------------------------------------------------------------

class TestManagerMisc:
    """Tests for TaskManager utility methods."""

    def test_len_reflects_task_count(self) -> None:
        tm = TaskManager()
        assert len(tm) == 0
        tid = tm.start_task("a", lambda: "ok")
        assert len(tm) == 1
        tm.get_task(tid)._thread.join(timeout=5.0)
        assert len(tm) == 1  # completed tasks still count

    def test_status_summary_empty(self) -> None:
        tm = TaskManager()
        assert tm.get_status_summary() == "No tasks."

    def test_status_summary_with_tasks(self) -> None:
        tm = TaskManager()
        tid = tm.start_task("test-task", lambda: "result", task_id="abc123")
        tm.get_task(tid)._thread.join(timeout=5.0)
        summary = tm.get_status_summary()
        assert "abc123" in summary
        assert "test-task" in summary
        assert "completed" in summary
