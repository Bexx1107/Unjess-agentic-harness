"""Schedule tools — non-blocking one-shot timers and recurring cron jobs."""

from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING

from unjess.tools import ToolRegistry

if TYPE_CHECKING:
    from unjess.scheduler import Scheduler

logger = logging.getLogger(__name__)


def register_schedule_tools(registry: ToolRegistry, scheduler: "Scheduler") -> None:
    """Register the native schedule tool on the tool registry.

    Args:
        registry: Target tool registry.
        scheduler: Active Scheduler instance.
    """

    def _schedule(
        prompt: str,
        duration_seconds: Optional[float] = None,
        cron_expression: Optional[str] = None,
        max_iterations: int = 0,
    ) -> str:
        """Schedule a non-blocking timer or recurring cron task.

        Args:
            prompt: Message/task to deliver when fired.
            duration_seconds: Seconds to wait for a one-shot timer (e.g. 60 for 1 min).
            cron_expression: Cron expression (e.g. '0 */6 * * *' for every 6h).
            max_iterations: Max number of times cron fires (0 = unlimited).

        Returns:
            Task ID and confirmation string.
        """
        if not prompt:
            return "Error: 'prompt' parameter is required for scheduled tasks."

        if duration_seconds is not None and duration_seconds > 0:
            task_id = scheduler.schedule_timer(
                duration_seconds=duration_seconds,
                prompt=prompt,
            )
            mins = duration_seconds / 60
            time_str = f"{mins:.1f} minute(s)" if mins >= 1 else f"{duration_seconds:.0f} second(s)"
            return f"Timer scheduled successfully (ID: {task_id}). Will trigger in {time_str} with prompt: '{prompt}'"

        if cron_expression:
            try:
                task_id = scheduler.schedule_cron(
                    expression=cron_expression,
                    prompt=prompt,
                    max_iterations=max_iterations,
                )
                return f"Cron task scheduled successfully (ID: {task_id}, schedule: '{cron_expression}'). Prompt: '{prompt}'"
            except ValueError as exc:
                return f"Error: Invalid cron expression '{cron_expression}': {exc}"

        return "Error: Specify either 'duration_seconds' (for a timer) or 'cron_expression' (for a recurring schedule)."

    registry.register(
        name="schedule",
        description=(
            "Schedule a non-blocking background timer or recurring cron task. "
            "Returns immediately without blocking the conversation. "
            "When the schedule triggers, your prompt notification will fire in the background."
        ),
        parameters={
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The message/task prompt to run when the schedule triggers.",
                },
                "duration_seconds": {
                    "type": "number",
                    "description": "Seconds to wait for a one-shot timer (e.g. 60 for 1 min, 3600 for 1 hr).",
                },
                "cron_expression": {
                    "type": "string",
                    "description": "Standard 5-field cron expression (e.g. '0 */6 * * *' for every 6 hours).",
                },
                "max_iterations": {
                    "type": "integer",
                    "description": "Optional max number of times a cron task fires (0 = unlimited).",
                },
            },
            "required": ["prompt"],
        },
        handler=_schedule,
    )
