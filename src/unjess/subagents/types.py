"""Pre-defined agent types and orchestration patterns.

Provides:
- 5 built-in agent types (research, coder, reviewer, tester, self)
- 3 orchestration patterns (fan-out, pipeline, supervisor)
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent type definitions
# ---------------------------------------------------------------------------

@dataclass
class AgentType:
    """Definition of a reusable agent type."""

    name: str
    description: str
    system_prompt: str
    tools: list[str] = field(default_factory=list)
    enable_write_tools: bool = False

    def register(self, manager: Any) -> None:
        """Register this type with a SubagentManager."""
        manager.define_type(
            name=self.name,
            description=self.description,
            system_prompt=self.system_prompt,
            tools=self.tools or None,
            enable_write_tools=self.enable_write_tools,
        )


# ---------------------------------------------------------------------------
# Built-in agent types
# ---------------------------------------------------------------------------

RESEARCH_AGENT = AgentType(
    name="research",
    description="Read-only agent for exploring the codebase, searching the web, and reading files. "
                "Cannot modify files or run commands.",
    system_prompt=(
        "You are a research agent. Your job is to thoroughly explore the codebase and gather "
        "information to answer questions. You have read-only access — you cannot modify files "
        "or run commands. Be thorough and report your findings clearly."
    ),
    tools=["read_file", "list_dir", "grep_search", "search_web"],
    enable_write_tools=False,
)

CODER_AGENT = AgentType(
    name="coder",
    description="Coding agent with full read/write access. Can edit files and run commands.",
    system_prompt=(
        "You are a coding agent. Your job is to implement code changes as instructed. "
        "You can read, write, and edit files, and run commands. Follow the instructions "
        "precisely and report what you've done."
    ),
    tools=["read_file", "write_file", "edit_file", "list_dir", "grep_search", "run_command"],
    enable_write_tools=True,
)

REVIEWER_AGENT = AgentType(
    name="reviewer",
    description="Code review agent. Reads code and provides feedback — does not modify files.",
    system_prompt=(
        "You are a code review agent. Your job is to review code changes and provide "
        "constructive feedback. Check for:\n"
        "- Bugs and edge cases\n"
        "- Code quality and readability\n"
        "- Missing error handling\n"
        "- Security issues\n"
        "- Performance concerns\n"
        "Do NOT modify any files. Report your findings clearly."
    ),
    tools=["read_file", "list_dir", "grep_search"],
    enable_write_tools=False,
)

TESTER_AGENT = AgentType(
    name="tester",
    description="Testing agent. Runs tests, reports results, and can write test code.",
    system_prompt=(
        "You are a testing agent. Your job is to:\n"
        "1. Run the project's test suite\n"
        "2. Report which tests pass and fail\n"
        "3. Write new tests if requested\n"
        "4. Suggest fixes for failing tests\n"
        "Be thorough and precise in your test reports."
    ),
    tools=["read_file", "write_file", "list_dir", "grep_search", "run_command"],
    enable_write_tools=True,
)

SELF_AGENT = AgentType(
    name="self",
    description="Clone of the parent agent with the same configuration and capabilities.",
    system_prompt="",  # inherits parent's prompt
    tools=[],  # inherits parent's tools
    enable_write_tools=True,
)

BUILTIN_TYPES: list[AgentType] = [
    RESEARCH_AGENT,
    CODER_AGENT,
    REVIEWER_AGENT,
    TESTER_AGENT,
    SELF_AGENT,
]


def register_builtin_types(manager: Any) -> None:
    """Register all built-in agent types with a SubagentManager.

    Args:
        manager: SubagentManager instance.
    """
    for agent_type in BUILTIN_TYPES:
        agent_type.register(manager)
    logger.info("Registered %d built-in agent types", len(BUILTIN_TYPES))


# ---------------------------------------------------------------------------
# Orchestration Patterns
# ---------------------------------------------------------------------------

@dataclass
class FanOutResult:
    """Result from a fan-out (parallel) operation."""

    results: dict[str, str] = field(default_factory=dict)  # agent_id -> result
    errors: dict[str, str] = field(default_factory=dict)  # agent_id -> error

    @property
    def all_succeeded(self) -> bool:
        """Whether all agents completed successfully."""
        return len(self.errors) == 0

    def summary(self) -> str:
        """One-line summary."""
        total = len(self.results) + len(self.errors)
        return f"{len(self.results)}/{total} succeeded, {len(self.errors)} failed"


class FanOut:
    """Fan-out pattern — spawn multiple agents in parallel.

    Each agent gets a different prompt but the same type.
    Useful for parallel research or exploring multiple approaches.

    Args:
        manager: SubagentManager instance.
        agent_type: Agent type to use for all spawned agents.
    """

    def __init__(self, manager: Any, agent_type: str = "research") -> None:
        self._manager = manager
        self._agent_type = agent_type

    def spawn(self, prompts: list[str], roles: list[str] | None = None) -> list[str]:
        """Spawn agents for each prompt.

        Args:
            prompts: List of task prompts.
            roles: Optional list of role names (one per prompt).

        Returns:
            List of conversation IDs.
        """
        roles = roles or [f"Worker {i+1}" for i in range(len(prompts))]
        ids: list[str] = []

        for prompt, role in zip(prompts, roles):
            cid = self._manager.invoke(
                type_name=self._agent_type,
                prompt=prompt,
                role=role,
            )
            ids.append(cid)

        return ids


@dataclass
class PipelineStep:
    """A single step in a pipeline."""

    name: str
    agent_type: str
    prompt_template: str  # can contain {previous_result}
    role: str = ""


class Pipeline:
    """Pipeline pattern — sequential chain of agents.

    Each agent's output feeds into the next as context.
    Useful for plan → code → review → test workflows.

    Args:
        manager: SubagentManager instance.
        steps: Ordered list of pipeline steps.
    """

    def __init__(self, manager: Any, steps: list[PipelineStep]) -> None:
        self._manager = manager
        self._steps = steps

    def start(self, initial_input: str) -> str:
        """Start the pipeline with the first step.

        Only starts the first step — subsequent steps must be triggered
        as each agent completes.

        Args:
            initial_input: Input for the first step.

        Returns:
            Conversation ID of the first agent.
        """
        if not self._steps:
            return ""

        step = self._steps[0]
        prompt = step.prompt_template.replace("{previous_result}", initial_input)

        return self._manager.invoke(
            type_name=step.agent_type,
            prompt=prompt,
            role=step.role or step.name,
        )

    def next_step(self, step_index: int, previous_result: str) -> Optional[str]:
        """Trigger the next step in the pipeline.

        Args:
            step_index: Index of the step to run.
            previous_result: Output from the previous step.

        Returns:
            Conversation ID of the next agent, or None if pipeline is complete.
        """
        if step_index >= len(self._steps):
            return None

        step = self._steps[step_index]
        prompt = step.prompt_template.replace("{previous_result}", previous_result)

        return self._manager.invoke(
            type_name=step.agent_type,
            prompt=prompt,
            role=step.role or step.name,
        )


class Supervisor:
    """Supervisor pattern — a manager agent that delegates to workers.

    The supervisor decides how to break down a task and which
    agent types to use, then coordinates the results.

    Args:
        manager: SubagentManager instance.
        max_workers: Maximum concurrent workers.
    """

    def __init__(self, manager: Any, max_workers: int = 5) -> None:
        self._manager = manager
        self._max_workers = max_workers
        self._worker_ids: list[str] = []

    def delegate(self, task: str, agent_type: str, role: str = "") -> Optional[str]:
        """Delegate a subtask to a worker agent.

        Args:
            task: Task description.
            agent_type: Agent type for the worker.
            role: Worker's role.

        Returns:
            Conversation ID, or None if at capacity.
        """
        active = [
            aid for aid in self._worker_ids
            if self._manager.get_agent(aid) and self._manager.get_agent(aid).is_active
        ]

        if len(active) >= self._max_workers:
            return None

        cid = self._manager.invoke(
            type_name=agent_type,
            prompt=task,
            role=role or f"Worker ({agent_type})",
        )
        self._worker_ids.append(cid)
        return cid

    @property
    def active_workers(self) -> int:
        """Number of currently active workers."""
        return sum(
            1 for aid in self._worker_ids
            if self._manager.get_agent(aid) and self._manager.get_agent(aid).is_active
        )

    def collect_results(self) -> dict[str, str]:
        """Collect results from all completed workers.

        Returns:
            Dict of conversation_id -> result.
        """
        results: dict[str, str] = {}
        for aid in self._worker_ids:
            agent = self._manager.get_agent(aid)
            if agent and agent.status.value == "completed":
                results[aid] = agent.result
        return results
