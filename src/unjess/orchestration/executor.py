"""Workflow executor — runs workflow steps respecting dependencies.

Executes ready steps in parallel (via subagents), waits for completion,
handles retries, and tracks progress.
"""

import logging
import time
from typing import Any, Optional

from unjess.orchestration.workflow import StepStatus, Workflow, WorkflowStep
from unjess.subagents.manager import AgentStatus, SubagentManager

logger = logging.getLogger(__name__)


class WorkflowExecutor:
    """Executes a workflow by dispatching steps to subagents.

    Respects dependencies — only starts a step when all its
    dependencies have completed. Independent steps run in parallel.

    Args:
        workflow: The workflow to execute.
        manager: SubagentManager for spawning agents.
        poll_interval: How often to check step status (seconds).
    """

    def __init__(
        self,
        workflow: Workflow,
        manager: SubagentManager,
        poll_interval: float = 1.0,
    ) -> None:
        self._workflow = workflow
        self._manager = manager
        self._poll_interval = poll_interval
        self._started = False

    @property
    def workflow(self) -> Workflow:
        """The workflow being executed."""
        return self._workflow

    def tick(self) -> bool:
        """Advance the workflow by one tick.

        Checks for completed steps, starts new ready steps,
        handles retries.

        Returns:
            True if the workflow is still running,
            False if all steps are complete.
        """
        if self._workflow.is_complete:
            return False

        # Check status of running steps
        self._update_running_steps()

        # Find and start ready steps
        ready = self._workflow.get_ready_steps()
        for step in ready:
            # Evaluate string-based condition before starting
            if step.condition_str and not self._evaluate_condition(step.condition_str):
                step.status = StepStatus.SKIPPED
                logger.info("Skipped step '%s' — condition '%s' not met", step.id, step.condition_str)
                continue
            self._start_step(step)

        # Check for retries
        for step in self._workflow.steps:
            if step.can_retry:
                logger.info("Retrying step '%s' (attempt %d)", step.id, step.attempt + 1)
                step.status = StepStatus.RETRYING
                self._start_step(step)

        return not self._workflow.is_complete

    def run_to_completion(self, max_ticks: int = 1000) -> bool:
        """Run the workflow to completion.

        Blocks until all steps are complete or max_ticks is reached.

        Args:
            max_ticks: Safety limit on number of ticks.

        Returns:
            True if all steps completed successfully.
        """
        self._started = True
        tick_count = 0

        while tick_count < max_ticks:
            still_running = self.tick()
            if not still_running:
                break

            tick_count += 1
            time.sleep(self._poll_interval)

        return not self._workflow.has_failures

    # ----- Internal -----

    def _start_step(self, step: WorkflowStep) -> None:
        """Start a workflow step by spawning a subagent.

        Args:
            step: The step to start.
        """
        step.status = StepStatus.RUNNING
        step.attempt += 1
        step.started_at = time.time()

        # Build prompt with previous results injected
        prompt = self._build_prompt(step)

        # Spawn the agent
        conversation_id = self._manager.invoke(
            type_name=step.agent_type,
            prompt=prompt,
            role=f"{self._workflow.name}/{step.id}",
        )
        step.conversation_id = conversation_id

        logger.info(
            "Started step '%s' (agent %s, type %s)",
            step.id,
            conversation_id[:8],
            step.agent_type,
        )

    def _update_running_steps(self) -> None:
        """Check on running steps and update their status."""
        for step in self._workflow.steps:
            if step.status != StepStatus.RUNNING:
                continue

            # Timeout enforcement
            if step.timeout > 0 and step.started_at > 0:
                elapsed = time.time() - step.started_at
                if elapsed > step.timeout:
                    logger.warning(
                        "Step '%s' timed out after %.1fs (limit: %ds)",
                        step.id, elapsed, step.timeout,
                    )
                    # Kill the agent if it's still alive
                    if step.conversation_id:
                        self._manager.kill(step.conversation_id)
                    step.status = StepStatus.FAILED
                    step.error = f"Timed out after {elapsed:.0f}s (limit: {step.timeout}s)"
                    continue

            if not step.conversation_id:
                continue

            agent = self._manager.get_agent(step.conversation_id)
            if not agent:
                step.status = StepStatus.FAILED
                step.error = "Agent not found"
                continue

            if agent.status == AgentStatus.COMPLETED:
                step.status = StepStatus.COMPLETED
                step.result = agent.result
                logger.info("Step '%s' completed", step.id)

            elif agent.status == AgentStatus.FAILED:
                step.status = StepStatus.FAILED
                step.error = agent.error
                logger.warning("Step '%s' failed: %s", step.id, agent.error)

            elif agent.status == AgentStatus.KILLED:
                step.status = StepStatus.FAILED
                step.error = "Agent was killed"

    def _build_prompt(self, step: WorkflowStep) -> str:
        """Build the prompt for a step, injecting dependency results.

        Args:
            step: The step whose prompt to build.

        Returns:
            Enriched prompt string.
        """
        prompt = step.prompt

        # Inject previous results
        if step.depends_on:
            context_parts: list[str] = []
            for dep_id in step.depends_on:
                dep_step = self._workflow.get_step(dep_id)
                if dep_step and dep_step.result:
                    context_parts.append(f"[Result from '{dep_id}']: {dep_step.result}")

            if context_parts:
                context = "\n\n".join(context_parts)
                prompt = prompt.replace("{{previous_result}}", context)

                # Also append if template variable wasn't used
                if "{{previous_result}}" not in step.prompt:
                    prompt += f"\n\nContext from previous steps:\n{context}"

        return prompt

    def _evaluate_condition(self, condition: str) -> bool:
        """Evaluate a step condition string.

        Condition format:
        - ``"step_id.success"`` — True if step completed without error.
        - ``"!step_id.success"`` — True if step failed (negate).
        - Empty string or None — always True.

        Args:
            condition: The condition string to evaluate.

        Returns:
            Whether the condition is satisfied.
        """
        if not condition:
            return True

        negate = condition.startswith('!')
        if negate:
            condition = condition[1:]

        parts = condition.split('.')
        step_id = parts[0]
        prop = parts[1] if len(parts) > 1 else 'success'

        step = self._workflow.get_step(step_id)
        if not step:
            # Unknown step — don't block, allow the dependent to run
            return True

        if prop == 'success':
            result = step.status == StepStatus.COMPLETED
        else:
            # Unknown property — default to True
            result = True

        return not result if negate else result
