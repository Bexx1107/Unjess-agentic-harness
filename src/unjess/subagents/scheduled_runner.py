"""Scheduled subagent runner — executes scheduled tasks as isolated background subagents with persistent run logs."""

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

SCHEDULES_DIR = Path.home() / ".unjess" / "schedules"


@dataclass
class TaskRunLog:
    """Audit log record for a single run of a scheduled task."""

    run_id: str
    task_id: str
    task_type: str
    prompt: str
    started_at: float
    finished_at: float = 0.0
    status: str = "running"  # running, succeeded, failed
    result: str = ""
    error: str = ""
    duration_seconds: float = 0.0
    tool_calls: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def get_task_runs_dir(task_id: str) -> Path:
    """Get the runs directory for a task."""
    d = SCHEDULES_DIR / task_id / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_run_log(log: TaskRunLog) -> Path:
    """Save a run log record to disk."""
    runs_dir = get_task_runs_dir(log.task_id)
    file_path = runs_dir / f"{int(log.started_at)}_{log.run_id[:8]}.json"
    try:
        with open(file_path, "w", encoding="utf-8") as fh:
            json.dump(log.to_dict(), fh, indent=2)
    except Exception as exc:
        logger.error("Failed to save run log to %s: %s", file_path, exc)
    return file_path


def load_task_run_logs(task_id: str, limit: int = 50) -> list[dict[str, Any]]:
    """Load past run logs for a scheduled task from disk."""
    runs_dir = get_task_runs_dir(task_id)
    if not runs_dir.exists():
        return []

    logs: list[dict[str, Any]] = []
    for p in sorted(runs_dir.glob("*.json"), reverse=True)[:limit]:
        try:
            with open(p, "r", encoding="utf-8") as fh:
                logs.append(json.load(fh))
        except Exception:
            pass
    return logs


class DummyHeadlessInput:
    """Headless input protocol implementation for background scheduled tasks."""

    def get_user_input(self, prompt: str = "❯ ") -> str:
        return ""

    def ask_approval(self, tool_name: str, args: dict[str, Any]) -> tuple[bool, bool]:
        return True, True  # Auto-approve tool calls in background scheduled tasks

    def ask_question(self, questions: list[dict[str, Any]]) -> list[str]:
        return []


def _default_agent_runner(prompt: str) -> str:
    """Run prompt using real LLM agent and tool registry."""
    try:
        from unjess.config import load_config
        from unjess.llm.router import ProviderRouter
        from unjess.tools import ToolRegistry
        from unjess.agent import Agent
        from unjess.display import TerminalDisplay

        settings = load_config()
        router = ProviderRouter(settings)
        tools = ToolRegistry()
        display = TerminalDisplay()
        user_input = DummyHeadlessInput()

        agent = Agent(
            settings=settings,
            provider_router=router,
            tools=tools,
            display=display,
            user_input=user_input,
        )
        return agent.run(prompt)
    except Exception as exc:
        logger.error("Error in default LLM agent runner: %s", exc)
        raise exc


def execute_scheduled_subagent(
    task_id: str,
    prompt: str,
    task_type: str = "cron",
    agent_runner: Optional[Callable[[str], str]] = None,
    on_complete: Optional[Callable[[TaskRunLog], None]] = None,
) -> TaskRunLog:
    """Run a scheduled task in a dedicated background thread with isolated logging.

    Args:
        task_id: Unique task identifier.
        prompt: Task prompt to execute.
        task_type: "timer" or "cron".
        agent_runner: Callable that accepts prompt and returns string result.
        on_complete: Callback invoked with completed TaskRunLog.

    Returns:
        TaskRunLog instance.
    """
    run_id = f"run-{int(time.time())}"
    start_time = time.time()

    log = TaskRunLog(
        run_id=run_id,
        task_id=task_id,
        task_type=task_type,
        prompt=prompt,
        started_at=start_time,
    )
    save_run_log(log)

    runner_fn = agent_runner or _default_agent_runner

    def _worker() -> None:
        try:
            res = runner_fn(prompt)
            log.result = str(res or "Task executed successfully")
            log.status = "succeeded"
        except Exception as exc:
            log.status = "failed"
            log.error = str(exc)
            logger.error("Scheduled subagent run failed for %s: %s", task_id, exc)

        log.finished_at = time.time()
        log.duration_seconds = round(log.finished_at - start_time, 2)
        save_run_log(log)

        if on_complete:
            try:
                on_complete(log)
            except Exception as exc:
                logger.error("on_complete callback failed: %s", exc)

        # Dispatch NiceGUI Toast Alert
        try:
            from nicegui import ui
            icon = "check_circle" if log.status == "succeeded" else "error"
            color = "positive" if log.status == "succeeded" else "negative"
            summary = log.result[:60] if log.status == "succeeded" else log.error[:60]
            msg = f"⏱️ Scheduled Task [{task_id}] Finished: {summary}"
            ui.notify(msg, type=color, position="top-right", duration=6.0)
        except Exception:
            pass  # Non-GUI context

    thread = threading.Thread(target=_worker, name=f"sched-subagent-{task_id}", daemon=True)
    thread.start()
    return log
