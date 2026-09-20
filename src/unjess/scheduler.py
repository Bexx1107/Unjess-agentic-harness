"""Scheduler — one-shot timers and recurring cron-based tasks.

Runs in a background thread and fires callbacks or sends
messages when timers expire or cron triggers fire.
"""

import json
import itertools
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class ScheduledTask:
    """A scheduled task — one-shot timer or recurring cron."""

    id: str
    task_type: str  # "timer" or "cron"
    prompt: str = ""
    callback: Optional[Callable[[], None]] = None
    duration_seconds: float = 0.0
    cron_expression: str = ""
    max_iterations: int = 0  # 0 = unlimited
    created_at: float = 0.0
    next_fire: float = 0.0
    fire_count: int = 0
    cancelled: bool = False
    enabled: bool = True
    last_run: float = 0.0
    last_status: str = "pending"  # pending, running, succeeded, failed, paused
    last_result: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = time.time()

    @property
    def is_active(self) -> bool:
        """Whether this task is still active."""
        if self.cancelled or not self.enabled:
            return False
        if self.task_type == "timer":
            return self.fire_count == 0
        if self.max_iterations > 0:
            return self.fire_count < self.max_iterations
        return True

    def to_dict(self) -> dict[str, Any]:
        """Serialize task to dictionary for disk persistence."""
        return {
            "id": self.id,
            "task_type": self.task_type,
            "prompt": self.prompt,
            "duration_seconds": self.duration_seconds,
            "cron_expression": self.cron_expression,
            "max_iterations": self.max_iterations,
            "created_at": self.created_at,
            "next_fire": self.next_fire,
            "fire_count": self.fire_count,
            "cancelled": self.cancelled,
            "enabled": self.enabled,
            "last_run": self.last_run,
            "last_status": self.last_status,
            "last_result": self.last_result,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScheduledTask":
        """Deserialize task from dictionary."""
        return cls(
            id=data.get("id", ""),
            task_type=data.get("task_type", "cron"),
            prompt=data.get("prompt", ""),
            duration_seconds=float(data.get("duration_seconds", 0.0)),
            cron_expression=data.get("cron_expression", ""),
            max_iterations=int(data.get("max_iterations", 0)),
            created_at=float(data.get("created_at", time.time())),
            next_fire=float(data.get("next_fire", 0.0)),
            fire_count=int(data.get("fire_count", 0)),
            cancelled=bool(data.get("cancelled", False)),
            enabled=bool(data.get("enabled", True)),
            last_run=float(data.get("last_run", 0.0)),
            last_status=data.get("last_status", "pending"),
            last_result=data.get("last_result", ""),
        )


def parse_cron(expression: str) -> dict[str, list[int]]:
    """Parse a 5-field cron expression into field ranges.

    Fields: minute hour day-of-month month day-of-week

    Args:
        expression: Cron expression (e.g., "*/5 * * * *").

    Returns:
        Dict mapping field names to lists of valid values.
    """
    fields = expression.strip().split()
    if len(fields) != 5:
        raise ValueError(f"Expected 5 cron fields, got {len(fields)}: {expression}")

    names = ["minute", "hour", "day", "month", "weekday"]
    ranges = [
        (0, 59),   # minute
        (0, 23),   # hour
        (1, 31),   # day
        (1, 12),   # month
        (0, 6),    # weekday (0=Sunday)
    ]

    result: dict[str, list[int]] = {}

    for i, (field_str, (lo, hi)) in enumerate(zip(fields, ranges)):
        result[names[i]] = _parse_cron_field(field_str, lo, hi)

    return result


def _parse_cron_field(field_str: str, lo: int, hi: int) -> list[int]:
    """Parse a single cron field into a list of valid values."""
    values: set[int] = set()

    for part in field_str.split(","):
        # */N — every N
        step_match = re.match(r'\*/(\d+)', part)
        if step_match:
            step = int(step_match.group(1))
            values.update(range(lo, hi + 1, step))
            continue

        # N-M — range
        range_match = re.match(r'(\d+)-(\d+)', part)
        if range_match:
            start = int(range_match.group(1))
            end = int(range_match.group(2))
            values.update(range(start, end + 1))
            continue

        # * — wildcard
        if part == "*":
            values.update(range(lo, hi + 1))
            continue

        # N — single value
        try:
            values.add(int(part))
        except ValueError:
            pass

    return sorted(values)


def cron_matches_now(parsed: dict[str, list[int]], now: Optional[time.struct_time] = None) -> bool:
    """Check if the current time matches a parsed cron expression.

    Args:
        parsed: Parsed cron fields from parse_cron().
        now: Override for current time.

    Returns:
        True if all fields match.
    """
    if now is None:
        now = time.localtime()

    # Python's tm_wday: 0=Monday, cron weekday: 0=Sunday — convert
    cron_weekday = (now.tm_wday + 1) % 7

    return (
        now.tm_min in parsed["minute"]
        and now.tm_hour in parsed["hour"]
        and now.tm_mday in parsed["day"]
        and now.tm_mon in parsed["month"]
        and cron_weekday in parsed["weekday"]
    )


class Scheduler:
    """Background scheduler for timers and cron jobs.

    Runs in a daemon thread and fires callbacks when tasks trigger.

    Args:
        check_interval: How often to check for triggers (seconds).
        schedules_file: Path to JSON persistence file.
    """

    def __init__(
        self,
        check_interval: float = 1.0,
        schedules_file: Optional[Path] = None,
    ) -> None:
        self._check_interval = check_interval
        self._schedules_file = schedules_file or (Path.home() / ".unjess" / "schedules.json")
        self._tasks: dict[str, ScheduledTask] = {}
        self._lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._counter = itertools.count(1)
        self.load_schedules()

    def save_schedules(self) -> None:
        """Persist active schedules to disk."""
        try:
            self._schedules_file.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                data = {t_id: t.to_dict() for t_id, t in self._tasks.items()}
            with open(self._schedules_file, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
        except Exception as exc:
            logger.error("Failed to save schedules to %s: %s", self._schedules_file, exc)

    def load_schedules(self) -> None:
        """Load persisted schedules from disk."""
        if not self._schedules_file.exists():
            return
        try:
            with open(self._schedules_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            with self._lock:
                for t_id, t_dict in data.items():
                    if t_id not in self._tasks:
                        task = ScheduledTask.from_dict(t_dict)
                        self._tasks[t_id] = task
            logger.info("Loaded %d schedules from %s", len(self._tasks), self._schedules_file)
        except Exception as exc:
            logger.error("Failed to load schedules from %s: %s", self._schedules_file, exc)

    def start(self) -> None:
        """Start the scheduler background thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop,
            name="scheduler",
            daemon=True,
        )
        self._thread.start()
        logger.info("Scheduler started")

    def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Scheduler stopped")

    def schedule_timer(
        self,
        duration_seconds: float,
        prompt: str = "",
        callback: Optional[Callable[[], None]] = None,
    ) -> str:
        """Schedule a one-shot timer.

        Args:
            duration_seconds: Seconds until firing.
            prompt: Message to deliver when fired.
            callback: Optional callback to invoke.

        Returns:
            Task ID.
        """
        task_id = f"timer-{next(self._counter)}"

        task = ScheduledTask(
            id=task_id,
            task_type="timer",
            prompt=prompt,
            callback=callback,
            duration_seconds=duration_seconds,
            next_fire=time.time() + duration_seconds,
        )

        with self._lock:
            self._tasks[task_id] = task

        self.save_schedules()

        if not self._running:
            self.start()

        logger.info("Scheduled timer '%s' for %.0fs", task_id, duration_seconds)
        return task_id

    def schedule_cron(
        self,
        expression: str,
        prompt: str = "",
        callback: Optional[Callable[[], None]] = None,
        max_iterations: int = 0,
    ) -> str:
        """Schedule a recurring cron job.

        Args:
            expression: Cron expression (5 fields).
            prompt: Message to deliver on each trigger.
            callback: Optional callback to invoke.
            max_iterations: Max number of fires (0 = unlimited).

        Returns:
            Task ID.
        """
        # Validate cron expression
        parse_cron(expression)

        task_id = f"cron-{next(self._counter)}"

        task = ScheduledTask(
            id=task_id,
            task_type="cron",
            prompt=prompt,
            callback=callback,
            cron_expression=expression,
            max_iterations=max_iterations,
        )

        with self._lock:
            self._tasks[task_id] = task

        self.save_schedules()

        if not self._running:
            self.start()

        logger.info("Scheduled cron '%s': %s", task_id, expression)
        return task_id

    def cancel(self, task_id: str) -> bool:
        """Cancel a scheduled task."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.cancelled = True
                task.enabled = False
                self.save_schedules()
                return True
        return False

    def pause_task(self, task_id: str) -> bool:
        """Pause a scheduled task."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.enabled = False
                task.last_status = "paused"
                self.save_schedules()
                return True
        return False

    def resume_task(self, task_id: str) -> bool:
        """Resume a paused task."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.enabled = True
                task.last_status = "pending"
                self.save_schedules()
                return True
        return False

    def trigger_now(self, task_id: str) -> bool:
        """Trigger a task immediately on-demand."""
        with self._lock:
            task = self._tasks.get(task_id)
        if task:
            self._fire(task)
            return True
        return False

    def delete_task(self, task_id: str) -> bool:
        """Delete a task permanently."""
        with self._lock:
            if task_id in self._tasks:
                del self._tasks[task_id]
                self.save_schedules()
                return True
        return False

    def list_tasks(self) -> list[ScheduledTask]:
        """List all tasks (active or paused)."""
        with self._lock:
            return list(self._tasks.values())

    # ----- Internal -----

    def _run_loop(self) -> None:
        """Main scheduler loop."""
        last_cron_minute = -1

        while self._running:
            now = time.time()
            now_struct = time.localtime(now)

            with self._lock:
                for task in list(self._tasks.values()):
                    if not task.is_active:
                        continue

                    if task.task_type == "timer":
                        if now >= task.next_fire:
                            self._fire(task)

                    elif task.task_type == "cron":
                        # Only check cron once per minute
                        if now_struct.tm_min != last_cron_minute:
                            try:
                                parsed = parse_cron(task.cron_expression)
                                if cron_matches_now(parsed, now_struct):
                                    self._fire(task)
                            except ValueError:
                                pass

                if now_struct.tm_min != last_cron_minute:
                    last_cron_minute = now_struct.tm_min

            time.sleep(self._check_interval)

    def _fire(self, task: ScheduledTask) -> None:
        """Fire a task as an isolated headless subagent."""
        task.fire_count += 1
        task.last_run = time.time()
        task.last_status = "running"
        self.save_schedules()
        logger.info("Task '%s' fired (count=%d): %s", task.id, task.fire_count, task.prompt[:80])

        if task.callback:
            try:
                task.callback()
            except Exception as exc:
                logger.error("Task '%s' callback failed: %s", task.id, exc)

        # Run headless subagent task
        try:
            from unjess.subagents.scheduled_runner import execute_scheduled_subagent

            def _on_run_complete(run_log: Any) -> None:
                task.last_status = run_log.status
                task.last_result = run_log.result or run_log.error
                self.save_schedules()

            execute_scheduled_subagent(
                task_id=task.id,
                prompt=task.prompt,
                task_type=task.task_type,
                on_complete=_on_run_complete,
            )
        except Exception as exc:
            task.last_status = "failed"
            task.last_result = str(exc)
            self.save_schedules()
            logger.error("Failed to launch headless subagent for %s: %s", task.id, exc)
