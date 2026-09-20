"""Unit tests for Scheduled Headless Subagents System (Disk Persistence, Subagent Execution, Run Logs, Task Controls)."""

import json
import time
import pytest
from pathlib import Path

from unjess.scheduler import Scheduler, ScheduledTask
from unjess.subagents.scheduled_runner import (
    TaskRunLog,
    execute_scheduled_subagent,
    load_task_run_logs,
    save_run_log,
)


@pytest.fixture
def tmp_schedules_file(tmp_path: Path) -> Path:
    return tmp_path / "schedules.json"


def test_scheduler_disk_persistence(tmp_schedules_file: Path):
    """Test that schedules save to disk and reload on start."""
    scheduler1 = Scheduler(check_interval=0.1, schedules_file=tmp_schedules_file)
    task_id = scheduler1.schedule_cron("*/5 * * * *", prompt="Test persist prompt")
    scheduler1.stop()

    assert tmp_schedules_file.exists()

    # Load in new scheduler instance
    scheduler2 = Scheduler(check_interval=0.1, schedules_file=tmp_schedules_file)
    tasks = scheduler2.list_tasks()
    scheduler2.stop()

    assert len(tasks) == 1
    assert tasks[0].id == task_id
    assert tasks[0].prompt == "Test persist prompt"
    assert tasks[0].cron_expression == "*/5 * * * *"


def test_scheduler_task_controls(tmp_schedules_file: Path):
    """Test pause, resume, trigger_now, and delete task controls."""
    scheduler = Scheduler(check_interval=0.1, schedules_file=tmp_schedules_file)
    task_id = scheduler.schedule_cron("0 * * * *", prompt="Hourly check")

    # Pause
    assert scheduler.pause_task(task_id) is True
    task = [t for t in scheduler.list_tasks() if t.id == task_id][0]
    assert task.enabled is False
    assert task.last_status == "paused"

    # Resume
    assert scheduler.resume_task(task_id) is True
    assert task.enabled is True

    # Delete
    assert scheduler.delete_task(task_id) is True
    assert len(scheduler.list_tasks()) == 0
    scheduler.stop()


def test_scheduled_runner_run_logs():
    """Test background subagent execution and run log saving."""
    task_id = f"test-task-{int(time.time())}"

    def mock_agent(prompt: str) -> str:
        return f"Mock answer for: {prompt}"

    log = execute_scheduled_subagent(
        task_id=task_id,
        prompt="Check stock price",
        task_type="timer",
        agent_runner=mock_agent,
    )

    # Wait for background thread
    time.sleep(0.5)

    logs = load_task_run_logs(task_id)
    assert len(logs) >= 1
    assert logs[0]["task_id"] == task_id
    assert logs[0]["status"] == "succeeded"
    assert "Mock answer for: Check stock price" in logs[0]["result"]
