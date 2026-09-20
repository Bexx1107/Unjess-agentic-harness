"""Tests for schedule tools — one-shot timers and cron tasks."""

import time
import pytest
from pathlib import Path

from unjess.scheduler import Scheduler
from unjess.tools import ToolRegistry
from unjess.tools.schedule_tools import register_schedule_tools


@pytest.fixture
def scheduler() -> Scheduler:
    s = Scheduler(check_interval=0.1)
    s.start()
    yield s
    s.stop()


def test_schedule_timer_tool(scheduler: Scheduler) -> None:
    registry = ToolRegistry()
    register_schedule_tools(registry, scheduler)

    tool = registry.get_tool("schedule")
    assert tool is not None

    result = tool.handler(duration_seconds=60, prompt="Test timer prompt")
    assert "Timer scheduled successfully" in result
    assert "Test timer prompt" in result

    tasks = scheduler.list_tasks()
    assert len(tasks) == 1
    assert tasks[0].prompt == "Test timer prompt"
    assert tasks[0].task_type == "timer"


def test_schedule_cron_tool(scheduler: Scheduler) -> None:
    registry = ToolRegistry()
    register_schedule_tools(registry, scheduler)

    tool = registry.get_tool("schedule")
    assert tool is not None

    result = tool.handler(cron_expression="0 */6 * * *", prompt="Check issues every 6h")
    assert "Cron task scheduled successfully" in result

    tasks = scheduler.list_tasks()
    assert len(tasks) == 1
    assert tasks[0].cron_expression == "0 */6 * * *"
    assert tasks[0].task_type == "cron"


def test_invalid_cron_expression(scheduler: Scheduler) -> None:
    registry = ToolRegistry()
    register_schedule_tools(registry, scheduler)

    tool = registry.get_tool("schedule")
    result = tool.handler(cron_expression="invalid cron expr", prompt="Fail prompt")
    assert "Error: Invalid cron expression" in result
