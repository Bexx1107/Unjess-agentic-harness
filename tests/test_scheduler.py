"""Tests for the scheduler module — timers, cron parsing, and the Scheduler class."""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from unjess.scheduler import (
    Scheduler,
    ScheduledTask,
    cron_matches_now,
    parse_cron,
)


# ---------------------------------------------------------------------------
# ScheduledTask dataclass
# ---------------------------------------------------------------------------

class TestScheduledTask:
    """Tests for the ScheduledTask dataclass and its is_active property."""

    def test_post_init_sets_created_at_when_zero(self):
        before = time.time()
        task = ScheduledTask(id="t1", task_type="timer")
        after = time.time()
        assert before <= task.created_at <= after

    def test_post_init_keeps_explicit_created_at(self):
        task = ScheduledTask(id="t1", task_type="timer", created_at=1000.0)
        assert task.created_at == 1000.0

    def test_is_active_timer_not_yet_fired(self):
        task = ScheduledTask(id="t1", task_type="timer", fire_count=0)
        assert task.is_active is True

    def test_is_active_timer_already_fired(self):
        task = ScheduledTask(id="t1", task_type="timer", fire_count=1)
        assert task.is_active is False

    def test_is_active_cancelled(self):
        task = ScheduledTask(id="t1", task_type="timer", cancelled=True)
        assert task.is_active is False

    def test_is_active_cron_unlimited(self):
        task = ScheduledTask(id="c1", task_type="cron", max_iterations=0, fire_count=50)
        assert task.is_active is True

    def test_is_active_cron_within_limit(self):
        task = ScheduledTask(id="c1", task_type="cron", max_iterations=5, fire_count=3)
        assert task.is_active is True

    def test_is_active_cron_at_limit(self):
        task = ScheduledTask(id="c1", task_type="cron", max_iterations=5, fire_count=5)
        assert task.is_active is False

    def test_is_active_cron_cancelled_overrides(self):
        task = ScheduledTask(id="c1", task_type="cron", max_iterations=0, cancelled=True)
        assert task.is_active is False

    def test_default_field_values(self):
        task = ScheduledTask(id="x", task_type="timer")
        assert task.prompt == ""
        assert task.callback is None
        assert task.duration_seconds == 0.0
        assert task.cron_expression == ""
        assert task.max_iterations == 0
        assert task.next_fire == 0.0
        assert task.fire_count == 0
        assert task.cancelled is False


# ---------------------------------------------------------------------------
# parse_cron
# ---------------------------------------------------------------------------

class TestParseCron:
    """Tests for the parse_cron function."""

    def test_every_5_minutes(self):
        parsed = parse_cron("*/5 * * * *")
        assert parsed["minute"] == [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
        assert parsed["hour"] == list(range(0, 24))
        assert parsed["day"] == list(range(1, 32))
        assert parsed["month"] == list(range(1, 13))
        assert parsed["weekday"] == list(range(0, 7))

    def test_weekday_mornings(self):
        parsed = parse_cron("0 9 * * 1-5")
        assert parsed["minute"] == [0]
        assert parsed["hour"] == [9]
        assert parsed["day"] == list(range(1, 32))
        assert parsed["month"] == list(range(1, 13))
        assert parsed["weekday"] == [1, 2, 3, 4, 5]

    def test_specific_values(self):
        parsed = parse_cron("30 12 15 6 3")
        assert parsed["minute"] == [30]
        assert parsed["hour"] == [12]
        assert parsed["day"] == [15]
        assert parsed["month"] == [6]
        assert parsed["weekday"] == [3]

    def test_comma_separated(self):
        parsed = parse_cron("0,15,30,45 * * * *")
        assert parsed["minute"] == [0, 15, 30, 45]

    def test_step_in_hour(self):
        parsed = parse_cron("0 */6 * * *")
        assert parsed["hour"] == [0, 6, 12, 18]

    def test_range_in_month(self):
        parsed = parse_cron("0 0 * 3-7 *")
        assert parsed["month"] == [3, 4, 5, 6, 7]

    def test_wrong_field_count_raises(self):
        with pytest.raises(ValueError, match="Expected 5 cron fields"):
            parse_cron("*/5 * *")

    def test_too_many_fields_raises(self):
        with pytest.raises(ValueError, match="Expected 5 cron fields"):
            parse_cron("* * * * * *")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="Expected 5 cron fields"):
            parse_cron("")

    def test_invalid_field_token_ignored(self):
        # An invalid token like "abc" is silently ignored per the impl
        parsed = parse_cron("abc * * * *")
        assert parsed["minute"] == []

    def test_results_are_sorted(self):
        parsed = parse_cron("45,10,30,5 * * * *")
        assert parsed["minute"] == [5, 10, 30, 45]

    def test_comma_and_range_mixed(self):
        parsed = parse_cron("0 0 1,15,20-25 * *")
        assert parsed["day"] == [1, 15, 20, 21, 22, 23, 24, 25]

    def test_star_gives_full_range(self):
        parsed = parse_cron("* * * * *")
        assert parsed["minute"] == list(range(0, 60))
        assert parsed["hour"] == list(range(0, 24))
        assert parsed["day"] == list(range(1, 32))
        assert parsed["month"] == list(range(1, 13))
        assert parsed["weekday"] == list(range(0, 7))


# ---------------------------------------------------------------------------
# cron_matches_now
# ---------------------------------------------------------------------------

class TestCronMatchesNow:
    """Tests for cron_matches_now with explicit time overrides."""

    def _make_time(
        self,
        minute: int = 0,
        hour: int = 0,
        day: int = 1,
        month: int = 1,
        wday: int = 0,
    ) -> time.struct_time:
        """Build a struct_time with the given fields.

        wday is Python's convention: 0=Monday.
        """
        return time.struct_time((2026, month, day, hour, minute, 0, wday, 1, -1))

    def test_matches_every_5_min_at_0(self):
        parsed = parse_cron("*/5 * * * *")
        now = self._make_time(minute=0)
        assert cron_matches_now(parsed, now) is True

    def test_matches_every_5_min_at_5(self):
        parsed = parse_cron("*/5 * * * *")
        now = self._make_time(minute=5)
        assert cron_matches_now(parsed, now) is True

    def test_no_match_every_5_min_at_3(self):
        parsed = parse_cron("*/5 * * * *")
        now = self._make_time(minute=3)
        assert cron_matches_now(parsed, now) is False

    def test_weekday_match_monday(self):
        # cron weekday: 0=Sun, 1=Mon.  Python wday: 0=Mon
        # Monday: python wday=0, cron_weekday = (0+1)%7 = 1
        parsed = parse_cron("0 9 * * 1-5")
        now = self._make_time(minute=0, hour=9, wday=0)  # Monday
        assert cron_matches_now(parsed, now) is True

    def test_weekday_no_match_sunday(self):
        # Sunday: python wday=6, cron_weekday = (6+1)%7 = 0
        parsed = parse_cron("0 9 * * 1-5")
        now = self._make_time(minute=0, hour=9, wday=6)  # Sunday
        assert cron_matches_now(parsed, now) is False

    def test_weekday_match_friday(self):
        # Friday: python wday=4, cron_weekday = (4+1)%7 = 5
        parsed = parse_cron("0 9 * * 1-5")
        now = self._make_time(minute=0, hour=9, wday=4)  # Friday
        assert cron_matches_now(parsed, now) is True

    def test_weekday_no_match_saturday(self):
        # Saturday: python wday=5, cron_weekday = (5+1)%7 = 6
        parsed = parse_cron("0 9 * * 1-5")
        now = self._make_time(minute=0, hour=9, wday=5)  # Saturday
        assert cron_matches_now(parsed, now) is False

    def test_hour_mismatch(self):
        parsed = parse_cron("0 9 * * *")
        now = self._make_time(minute=0, hour=10)
        assert cron_matches_now(parsed, now) is False

    def test_day_mismatch(self):
        parsed = parse_cron("0 0 15 * *")
        now = self._make_time(day=14)
        assert cron_matches_now(parsed, now) is False

    def test_month_mismatch(self):
        parsed = parse_cron("0 0 1 6 *")
        now = self._make_time(month=7)
        assert cron_matches_now(parsed, now) is False

    def test_all_wildcards_always_matches(self):
        parsed = parse_cron("* * * * *")
        now = self._make_time(minute=37, hour=14, day=22, month=11, wday=3)
        assert cron_matches_now(parsed, now) is True

    def test_uses_current_time_when_now_is_none(self):
        parsed = parse_cron("* * * * *")
        # Wildcard always matches, so this should be True regardless of wall-clock
        assert cron_matches_now(parsed) is True


# ---------------------------------------------------------------------------
# Scheduler class
# ---------------------------------------------------------------------------

class TestScheduler:
    """Tests for the Scheduler class — timers, crons, start/stop, cancellation."""

    def test_init_defaults(self):
        sched = Scheduler()
        assert sched._check_interval == 1.0
        assert sched._running is False
        assert sched._tasks == {}

    def test_init_custom_interval(self):
        sched = Scheduler(check_interval=0.5)
        assert sched._check_interval == 0.5

    def test_schedule_timer_returns_id(self):
        sched = Scheduler(check_interval=60)
        task_id = sched.schedule_timer(10, prompt="test")
        assert task_id.startswith("timer-")
        sched.stop()

    def test_schedule_timer_auto_starts(self):
        sched = Scheduler(check_interval=60)
        assert sched._running is False
        sched.schedule_timer(10, prompt="test")
        assert sched._running is True
        sched.stop()

    def test_schedule_timer_stores_task(self):
        sched = Scheduler(check_interval=60)
        task_id = sched.schedule_timer(10, prompt="hello")
        assert task_id in sched._tasks
        task = sched._tasks[task_id]
        assert task.task_type == "timer"
        assert task.prompt == "hello"
        assert task.duration_seconds == 10
        sched.stop()

    def test_schedule_cron_returns_id(self):
        sched = Scheduler(check_interval=60)
        task_id = sched.schedule_cron("*/5 * * * *", prompt="cron test")
        assert task_id.startswith("cron-")
        sched.stop()

    def test_schedule_cron_auto_starts(self):
        sched = Scheduler(check_interval=60)
        sched.schedule_cron("*/5 * * * *")
        assert sched._running is True
        sched.stop()

    def test_schedule_cron_invalid_expression_raises(self):
        sched = Scheduler(check_interval=60)
        with pytest.raises(ValueError):
            sched.schedule_cron("bad cron")

    def test_schedule_cron_stores_task(self):
        sched = Scheduler(check_interval=60)
        task_id = sched.schedule_cron("0 9 * * 1-5", prompt="work", max_iterations=3)
        task = sched._tasks[task_id]
        assert task.task_type == "cron"
        assert task.cron_expression == "0 9 * * 1-5"
        assert task.max_iterations == 3
        sched.stop()

    def test_cancel_existing_task(self):
        sched = Scheduler(check_interval=60)
        task_id = sched.schedule_timer(100, prompt="cancel me")
        assert sched.cancel(task_id) is True
        assert sched._tasks[task_id].cancelled is True
        sched.stop()

    def test_cancel_nonexistent_task(self):
        sched = Scheduler(check_interval=60)
        assert sched.cancel("no-such-id") is False

    def test_list_tasks_returns_only_active(self):
        sched = Scheduler(check_interval=60)
        id1 = sched.schedule_timer(100, prompt="active")
        id2 = sched.schedule_timer(100, prompt="will cancel")
        sched.cancel(id2)
        active = sched.list_tasks(only_active=True)
        assert len(active) == 1
        assert active[0].id == id1
        sched.stop()

    def test_list_tasks_empty_initially(self):
        sched = Scheduler(check_interval=60)
        assert sched.list_tasks() == []

    def test_start_is_idempotent(self):
        sched = Scheduler(check_interval=60)
        sched.start()
        thread1 = sched._thread
        sched.start()  # second call should be a no-op
        assert sched._thread is thread1
        sched.stop()

    def test_stop_when_not_running(self):
        sched = Scheduler(check_interval=60)
        sched.stop()  # should not raise
        assert sched._running is False

    def test_timer_fires_callback(self):
        callback = MagicMock()
        sched = Scheduler(check_interval=0.05)
        sched.schedule_timer(0.01, prompt="fire!", callback=callback)
        time.sleep(0.3)
        sched.stop()
        callback.assert_called()

    def test_timer_fires_only_once(self):
        callback = MagicMock()
        sched = Scheduler(check_interval=0.05)
        sched.schedule_timer(0.01, prompt="once", callback=callback)
        time.sleep(0.4)
        sched.stop()
        assert callback.call_count == 1

    def test_cancelled_timer_does_not_fire(self):
        callback = MagicMock()
        sched = Scheduler(check_interval=0.05)
        task_id = sched.schedule_timer(0.2, prompt="cancel me", callback=callback)
        sched.cancel(task_id)
        time.sleep(0.4)
        sched.stop()
        callback.assert_not_called()

    def test_callback_exception_is_caught(self):
        def exploding_callback() -> None:
            raise RuntimeError("boom")

        sched = Scheduler(check_interval=0.05)
        sched.schedule_timer(0.01, prompt="boom", callback=exploding_callback)
        time.sleep(0.3)
        sched.stop()
        # No exception should propagate; scheduler should still be stable

    def test_multiple_timers_fire_independently(self):
        results: list[str] = []
        sched = Scheduler(check_interval=0.05)
        sched.schedule_timer(0.01, prompt="a", callback=lambda: results.append("a"))
        sched.schedule_timer(0.01, prompt="b", callback=lambda: results.append("b"))
        time.sleep(0.3)
        sched.stop()
        assert "a" in results
        assert "b" in results

    def test_task_ids_are_unique(self):
        sched = Scheduler(check_interval=60)
        ids = set()
        for _ in range(20):
            ids.add(sched.schedule_timer(100, prompt="t"))
        for _ in range(20):
            ids.add(sched.schedule_cron("* * * * *", prompt="c"))
        assert len(ids) == 40
        sched.stop()

    def test_schedule_timer_with_no_callback(self):
        sched = Scheduler(check_interval=0.05)
        task_id = sched.schedule_timer(0.01, prompt="no cb")
        time.sleep(0.3)
        sched.stop()
        task = sched._tasks[task_id]
        assert task.fire_count == 1

    def test_fire_increments_fire_count(self):
        sched = Scheduler(check_interval=60)
        task = ScheduledTask(id="t1", task_type="timer")
        sched._fire(task)
        assert task.fire_count == 1
        sched._fire(task)
        assert task.fire_count == 2

    def test_fire_calls_callback(self):
        sched = Scheduler(check_interval=60)
        cb = MagicMock()
        task = ScheduledTask(id="t1", task_type="timer", callback=cb)
        sched._fire(task)
        cb.assert_called_once()

    def test_fire_handles_callback_exception(self):
        sched = Scheduler(check_interval=60)
        cb = MagicMock(side_effect=RuntimeError("fail"))
        task = ScheduledTask(id="t1", task_type="timer", callback=cb)
        sched._fire(task)  # should not raise
        assert task.fire_count == 1

    def test_fire_with_no_callback(self):
        sched = Scheduler(check_interval=60)
        task = ScheduledTask(id="t1", task_type="timer", callback=None)
        sched._fire(task)  # should not raise
        assert task.fire_count == 1

    def test_cron_fires_callback_when_matching(self):
        callback = MagicMock()
        sched = Scheduler(check_interval=0.05)

        # Schedule a cron that matches every minute
        sched.schedule_cron("* * * * *", prompt="always", callback=callback, max_iterations=1)
        time.sleep(0.3)
        sched.stop()
        callback.assert_called()

    def test_scheduler_thread_is_daemon(self):
        sched = Scheduler(check_interval=60)
        sched.start()
        assert sched._thread is not None
        assert sched._thread.daemon is True
        sched.stop()
