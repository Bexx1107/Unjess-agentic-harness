"""Workflow definition — steps with dependencies, conditions, and retry logic."""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class StepStatus(Enum):
    """Execution status of a workflow step."""

    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


@dataclass
class WorkflowStep:
    """A single step in a workflow.

    Args:
        id: Unique step identifier.
        action: What to do (agent type or callable name).
        prompt: Task description for the agent.
        depends_on: List of step IDs that must complete first.
        condition: Optional callable that returns True if step should run.
        max_retries: Number of retries on failure.
        timeout: Timeout in seconds (0 = no timeout).
        agent_type: Agent type to use (default: "self").
    """

    id: str
    action: str
    prompt: str = ""
    depends_on: list[str] = field(default_factory=list)
    condition: Optional[Callable[["Workflow"], bool]] = None
    condition_str: str = ""  # String condition, e.g. "step_id.success" or "!step_id.success"
    max_retries: int = 0
    timeout: int = 0
    agent_type: str = "self"

    # Runtime state
    status: StepStatus = StepStatus.PENDING
    result: str = ""
    error: str = ""
    attempt: int = 0
    conversation_id: str = ""
    started_at: float = 0.0  # Unix timestamp when step started running

    @property
    def is_terminal(self) -> bool:
        """Whether this step has reached a terminal state."""
        return self.status in (StepStatus.COMPLETED, StepStatus.FAILED, StepStatus.SKIPPED)

    @property
    def can_retry(self) -> bool:
        """Whether this step can be retried."""
        return self.attempt < self.max_retries and self.status == StepStatus.FAILED


@dataclass
class Workflow:
    """A workflow — a DAG of steps with dependencies.

    Steps are executed respecting their dependency ordering.
    Independent steps can run in parallel.

    Args:
        name: Workflow name.
        description: Human-readable description.
        steps: Ordered list of steps.
    """

    name: str
    description: str = ""
    steps: list[WorkflowStep] = field(default_factory=list)

    def add_step(self, step: WorkflowStep) -> None:
        """Add a step to the workflow."""
        # Validate dependencies exist
        existing_ids = {s.id for s in self.steps}
        for dep in step.depends_on:
            if dep not in existing_ids:
                raise ValueError(f"Step '{step.id}' depends on unknown step '{dep}'")
        self.steps.append(step)

    def get_step(self, step_id: str) -> Optional[WorkflowStep]:
        """Get a step by ID."""
        for step in self.steps:
            if step.id == step_id:
                return step
        return None

    def get_ready_steps(self) -> list[WorkflowStep]:
        """Get all steps whose dependencies are satisfied and are ready to run.

        Returns:
            Steps that can be started now.
        """
        ready: list[WorkflowStep] = []

        for step in self.steps:
            if step.status != StepStatus.PENDING:
                continue

            # Check all dependencies have reached a terminal state.
            # Note: deps may be COMPLETED, FAILED, or SKIPPED. The executor's
            # _evaluate_condition() decides whether the step should actually run
            # based on the condition_str (e.g. "!run_tests.success").
            deps_satisfied = all(
                self.get_step(dep_id) is not None and self.get_step(dep_id).is_terminal  # type: ignore[union-attr]
                for dep_id in step.depends_on
            )

            if deps_satisfied:
                # Check callable condition if present
                if step.condition is not None:
                    try:
                        if not step.condition(self):
                            step.status = StepStatus.SKIPPED
                            continue
                    except Exception as exc:
                        logger.warning("Condition check failed for step '%s': %s", step.id, exc)

                step.status = StepStatus.READY
                ready.append(step)

        return ready

    @property
    def is_complete(self) -> bool:
        """Whether all steps have reached a terminal state."""
        return all(step.is_terminal for step in self.steps)

    @property
    def has_failures(self) -> bool:
        """Whether any step has failed."""
        return any(step.status == StepStatus.FAILED for step in self.steps)

    @property
    def progress(self) -> tuple[int, int]:
        """(completed, total) step counts."""
        completed = sum(1 for s in self.steps if s.status == StepStatus.COMPLETED)
        return completed, len(self.steps)

    def summary(self) -> str:
        """Human-readable workflow summary."""
        completed, total = self.progress
        failed = sum(1 for s in self.steps if s.status == StepStatus.FAILED)
        running = sum(1 for s in self.steps if s.status == StepStatus.RUNNING)
        skipped = sum(1 for s in self.steps if s.status == StepStatus.SKIPPED)

        parts = [f"{completed}/{total} completed"]
        if running:
            parts.append(f"{running} running")
        if failed:
            parts.append(f"{failed} failed")
        if skipped:
            parts.append(f"{skipped} skipped")

        return f"Workflow '{self.name}': {', '.join(parts)}"

    def display(self) -> str:
        """Detailed step-by-step display."""
        status_icons = {
            StepStatus.PENDING: "⏳",
            StepStatus.READY: "▶️",
            StepStatus.RUNNING: "🔄",
            StepStatus.COMPLETED: "✅",
            StepStatus.FAILED: "❌",
            StepStatus.SKIPPED: "⏭️",
            StepStatus.RETRYING: "🔁",
        }

        lines = [f"## {self.name}", ""]
        for step in self.steps:
            icon = status_icons.get(step.status, "?")
            deps = f" (after: {', '.join(step.depends_on)})" if step.depends_on else ""
            lines.append(f"  {icon} {step.id}: {step.action}{deps} — {step.status.value}")
            if step.error:
                lines.append(f"     Error: {step.error}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pre-built workflow templates
# ---------------------------------------------------------------------------

def create_code_review_workflow(description: str) -> Workflow:
    """Create a plan → code → review → test workflow.

    Args:
        description: What the workflow should accomplish.

    Returns:
        A Workflow with 4 steps.
    """
    wf = Workflow(name="code_review", description=description)

    wf.add_step(WorkflowStep(
        id="plan",
        action="plan",
        prompt=f"Create a detailed plan for: {description}",
        agent_type="research",
    ))
    wf.add_step(WorkflowStep(
        id="code",
        action="implement",
        prompt="Implement the plan from the previous step. {{previous_result}}",
        depends_on=["plan"],
        agent_type="coder",
        max_retries=2,
    ))
    wf.add_step(WorkflowStep(
        id="review",
        action="review",
        prompt="Review the code changes from the implementation step.",
        depends_on=["code"],
        agent_type="reviewer",
    ))
    wf.add_step(WorkflowStep(
        id="test",
        action="test",
        prompt="Run the test suite and report results.",
        depends_on=["code"],
        agent_type="tester",
    ))

    return wf
