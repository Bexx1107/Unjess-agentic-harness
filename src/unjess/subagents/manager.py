"""Subagent manager — define, invoke, track, and kill child agents."""

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from unjess.subagents.messaging import Message, MessageBus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class AgentStatus(Enum):
    """Lifecycle status of a subagent."""

    PENDING = "pending"
    RUNNING = "running"
    IDLE = "idle"
    COMPLETED = "completed"
    FAILED = "failed"
    KILLED = "killed"


@dataclass
class AgentInfo:
    """Runtime information about a subagent instance."""

    conversation_id: str
    type_name: str
    role: str
    prompt: str
    status: AgentStatus = AgentStatus.PENDING
    workspace: str = ""
    workspace_mode: str = "inherit"  # inherit, branch, share
    created_at: float = 0.0
    finished_at: float = 0.0
    result: str = ""
    error: str = ""
    model_override: str = ""           # empty = use parent's model
    tool_filter: list[str] = field(default_factory=list)  # empty = use type defaults
    max_turns: int = 10                 # max LLM round-trips
    parent_conversation_id: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = time.time()

    @property
    def is_active(self) -> bool:
        """Whether this agent is still running or idle."""
        return self.status in (AgentStatus.PENDING, AgentStatus.RUNNING, AgentStatus.IDLE)

    @property
    def runtime_seconds(self) -> float:
        """How long this agent has been running."""
        end = self.finished_at or time.time()
        return end - self.created_at

    def summary(self) -> str:
        """One-line summary for display."""
        status_icons = {
            AgentStatus.PENDING: "⏳",
            AgentStatus.RUNNING: "🔄",
            AgentStatus.IDLE: "💤",
            AgentStatus.COMPLETED: "✅",
            AgentStatus.FAILED: "❌",
            AgentStatus.KILLED: "💀",
        }
        icon = status_icons.get(self.status, "?")
        elapsed = f"{self.runtime_seconds:.0f}s"
        return f"{icon} [{self.conversation_id[:8]}] {self.role} ({self.type_name}) — {self.status.value} ({elapsed})"


# ---------------------------------------------------------------------------
# Subagent Manager
# ---------------------------------------------------------------------------

# Maximum number of concurrent subagents to prevent system overload
# (especially important with local LLMs that consume heavy GPU/RAM per instance)
MAX_CONCURRENT_SUBAGENTS = 3


class SubagentManager:
    """Manages the lifecycle of child agents.

    Handles definition, invocation, messaging, and cleanup of subagents.
    Each subagent runs in its own thread with its own conversation context.
    Enforces a concurrency limit of :data:`MAX_CONCURRENT_SUBAGENTS`.

    Args:
        parent_id: Conversation ID of the parent agent.
        workspace: Parent's workspace directory.
        message_bus: Shared message bus for inter-agent communication.
    """

    def __init__(
        self,
        parent_id: str,
        workspace: Path,
        message_bus: Optional[MessageBus] = None,
    ) -> None:
        self._parent_id = parent_id
        self._workspace = workspace
        self._message_bus = message_bus or MessageBus()
        self._agents: dict[str, AgentInfo] = {}
        self._agent_types: dict[str, dict[str, Any]] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._lock = threading.Lock()
        self._agent_factory: Optional[Callable] = None

    @property
    def parent_id(self) -> str:
        """Conversation ID of the parent agent."""
        return self._parent_id

    @property
    def message_bus(self) -> MessageBus:
        """The shared message bus."""
        return self._message_bus

    # ----- Agent type definition -----

    def define_type(
        self,
        name: str,
        description: str,
        system_prompt: str,
        tools: list[str] | None = None,
        enable_write_tools: bool = False,
    ) -> None:
        """Define a new agent type.

        Args:
            name: Unique type name (e.g. "research", "coder").
            description: Human-readable description.
            system_prompt: System prompt for agents of this type.
            tools: List of tool names to enable (None = read-only).
            enable_write_tools: Whether to enable file write and command tools.
        """
        self._agent_types[name] = {
            "name": name,
            "description": description,
            "system_prompt": system_prompt,
            "tools": tools,
            "enable_write_tools": enable_write_tools,
        }
        logger.info("Defined agent type: %s", name)

    def set_agent_factory(self, factory: Callable) -> None:
        """Set the factory function for creating agent instances.

        The factory is called with (agent_info, agent_type_config, workspace, message_bus)
        and should return a callable that runs the agent loop.

        Args:
            factory: Agent instance factory.
        """
        self._agent_factory = factory

    # ----- Invocation -----

    def invoke(
        self,
        type_name: str,
        prompt: str,
        role: str = "",
        workspace_mode: str = "inherit",
        model_override: str = "",
        tool_filter: list[str] | None = None,
        max_turns: int = 10,
    ) -> str:
        """Spawn a new subagent.

        Enforces :data:`MAX_CONCURRENT_SUBAGENTS` to prevent resource
        exhaustion (especially with local LLMs).

        Args:
            type_name: The agent type to spawn.
            prompt: The task prompt.
            role: Human-readable role (e.g. "Codebase Researcher").
            workspace_mode: "inherit", "branch", or "share".

        Returns:
            The conversation ID of the new agent.

        Raises:
            RuntimeError: If the concurrency limit is reached.
        """
        # Enforce concurrency limit
        with self._lock:
            active_count = sum(1 for a in self._agents.values() if a.is_active)
            if active_count >= MAX_CONCURRENT_SUBAGENTS:
                raise RuntimeError(
                    f"Cannot spawn subagent: concurrency limit reached "
                    f"({active_count}/{MAX_CONCURRENT_SUBAGENTS} active). "
                    f"Wait for existing subagents to finish or kill some first."
                )

        conversation_id = uuid.uuid4().hex[:16]

        parent_id = getattr(self, "current_parent_id", "")
        agent_info = AgentInfo(
            conversation_id=conversation_id,
            type_name=type_name,
            role=role or type_name,
            prompt=prompt,
            workspace_mode=workspace_mode,
            workspace=str(self._resolve_workspace(workspace_mode)),
            model_override=model_override,
            tool_filter=tool_filter or [],
            max_turns=max_turns,
            parent_conversation_id=parent_id,
        )

        with self._lock:
            self._agents[conversation_id] = agent_info

        # Start agent in a background thread
        thread = threading.Thread(
            target=self._run_agent,
            args=(conversation_id,),
            name=f"agent-{conversation_id[:8]}",
            daemon=True,
        )
        self._threads[conversation_id] = thread
        thread.start()

        logger.info("Invoked subagent: %s (%s)", conversation_id[:8], type_name)
        return conversation_id

    # ----- Messaging -----

    def send_message(self, conversation_id: str, content: str) -> bool:
        """Send a message to a subagent.

        Args:
            conversation_id: Target agent's conversation ID.
            content: Message content.

        Returns:
            True if the agent exists and message was sent.
        """
        if conversation_id not in self._agents:
            return False

        self._message_bus.send(Message(
            sender=self._parent_id,
            recipient=conversation_id,
            content=content,
        ))
        return True

    def receive_messages(self) -> list[Message]:
        """Read all pending messages for the parent agent.

        Returns:
            List of messages from subagents.
        """
        return self._message_bus.receive(self._parent_id)

    # ----- Lifecycle -----

    def list_active(self) -> list[AgentInfo]:
        """List all active (running or idle) subagents."""
        with self._lock:
            return [a for a in self._agents.values() if a.is_active]

    @property
    def active_count(self) -> int:
        """Number of currently active subagents."""
        return len(self.list_active())

    def list_all(self) -> list[AgentInfo]:
        """List all subagents (active and completed)."""
        with self._lock:
            return list(self._agents.values())

    def get_agent(self, conversation_id: str) -> Optional[AgentInfo]:
        """Get info about a specific subagent."""
        return self._agents.get(conversation_id)

    def kill(self, conversation_id: str) -> bool:
        """Kill a subagent.

        Args:
            conversation_id: Agent to kill.

        Returns:
            True if the agent was found and killed.
        """
        with self._lock:
            agent = self._agents.get(conversation_id)
            if not agent:
                return False

            agent.status = AgentStatus.KILLED
            agent.finished_at = time.time()

            # Clean up thread reference
            self._threads.pop(conversation_id, None)

        logger.info("Killed subagent: %s", conversation_id[:8])
        return True

    def kill_all(self) -> int:
        """Kill all active subagents.

        Returns:
            Number of agents killed.
        """
        count = 0
        for agent_id in list(self._agents.keys()):
            agent = self._agents[agent_id]
            if agent.is_active:
                self.kill(agent_id)
                count += 1
        return count

    # ----- Fan-out & Pipeline helpers -----

    def fan_out(
        self,
        prompts: list[str],
        agent_type: str = "research",
        role: str = "Research Agent",
    ) -> list[str]:
        """Spawn multiple agents in parallel and return their conversation IDs.

        Args:
            prompts: List of prompts, one per agent.
            agent_type: Agent type name to use.
            role: Role description.

        Returns:
            List of conversation IDs.
        """
        ids: list[str] = []
        for i, prompt in enumerate(prompts):
            cid = self.invoke(
                type_name=agent_type,
                prompt=prompt,
                role=f"{role} {i + 1}",
            )
            ids.append(cid)
        return ids

    def fan_out_and_gather(
        self,
        prompts: list[str],
        agent_type: str = "research",
        role: str = "Research Agent",
        timeout: float = 300,
        poll_interval: float = 2.0,
    ) -> list[str]:
        """Spawn multiple agents, wait for all to complete, return results.

        Args:
            prompts: List of prompts.
            agent_type: Agent type name.
            role: Role description.
            timeout: Max seconds to wait.
            poll_interval: Seconds between status checks.

        Returns:
            List of result strings (one per agent).
        """
        ids = self.fan_out(prompts, agent_type, role)
        deadline = time.time() + timeout

        while time.time() < deadline:
            all_done = True
            for cid in ids:
                agent = self.get_agent(cid)
                if agent and agent.is_active:
                    all_done = False
                    break
            if all_done:
                break
            time.sleep(poll_interval)

        results: list[str] = []
        for cid in ids:
            agent = self.get_agent(cid)
            if agent and agent.result:
                results.append(agent.result)
            elif agent and agent.error:
                results.append(f"Error: {agent.error}")
            else:
                results.append("Timed out")
        return results

    def pipeline(
        self,
        steps: list[tuple[str, str, str]],
        timeout_per_step: float = 300,
    ) -> str:
        """Execute agents in sequence, passing each result to the next.

        Args:
            steps: List of (agent_type, role, prompt_template) tuples.
                   Use {{previous_result}} in prompt_template for injection.
            timeout_per_step: Max seconds per step.

        Returns:
            Final step's result.
        """
        previous_result = ""

        for agent_type, role, prompt_template in steps:
            prompt = prompt_template.replace("{{previous_result}}", previous_result)
            cid = self.invoke(type_name=agent_type, prompt=prompt, role=role)

            deadline = time.time() + timeout_per_step
            while time.time() < deadline:
                agent = self.get_agent(cid)
                if agent and not agent.is_active:
                    break
                time.sleep(2.0)

            agent = self.get_agent(cid)
            if agent and agent.result:
                previous_result = agent.result
            elif agent and agent.error:
                previous_result = f"Error: {agent.error}"
                break
            else:
                previous_result = "Step timed out"
                break

        return previous_result

    # ----- Internal -----

    def _run_agent(self, conversation_id: str) -> None:
        """Run a subagent in a background thread."""
        agent_info = self._agents.get(conversation_id)
        if not agent_info:
            return

        agent_info.status = AgentStatus.RUNNING
        type_config = self._agent_types.get(agent_info.type_name, {})

        try:
            if self._agent_factory:
                runner = self._agent_factory(
                    agent_info,
                    type_config,
                    Path(agent_info.workspace),
                    self._message_bus,
                )
                if callable(runner):
                    result = runner()
                    agent_info.result = str(result) if result else ""
            else:
                # No factory — just send the prompt result back
                self._message_bus.send(Message(
                    sender=conversation_id,
                    recipient=self._parent_id,
                    content=f"[Agent {conversation_id[:8]}] No agent factory configured. Prompt was: {agent_info.prompt[:200]}",
                    message_type="error",
                ))

            # Only set COMPLETED if not already killed
            if agent_info.status != AgentStatus.KILLED:
                agent_info.status = AgentStatus.COMPLETED

        except Exception as exc:
            agent_info.status = AgentStatus.FAILED
            agent_info.error = str(exc)
            logger.error("Subagent %s failed: %s", conversation_id[:8], exc)

            # Send error back to parent
            self._message_bus.send(Message(
                sender=conversation_id,
                recipient=self._parent_id,
                content=f"[Agent {conversation_id[:8]}] Error: {exc}",
                message_type="error",
            ))

        finally:
            agent_info.finished_at = time.time()

    def _resolve_workspace(self, mode: str) -> Path:
        """Resolve the workspace path for a given isolation mode."""
        if mode == "inherit":
            return self._workspace
        elif mode == "branch":
            # Create a temporary workspace directory
            branch_dir = self._workspace / ".unjess" / "workspaces" / uuid.uuid4().hex[:8]
            branch_dir.mkdir(parents=True, exist_ok=True)
            return branch_dir
        elif mode == "share":
            return self._workspace
        return self._workspace

    # ----- Context info -----

    def catalog_for_prompt(self) -> str:
        """Build subagent catalog for the system prompt."""
        if not self._agent_types:
            return ""

        lines = ["Available subagent types:"]
        for name, config in self._agent_types.items():
            desc = config.get("description", "")
            lines.append(f"  - {name}: {desc}")

        active = self.list_active()
        if active:
            lines.append(f"\nActive subagents: {len(active)}")
            for agent in active:
                lines.append(f"  {agent.summary()}")

        return "\n".join(lines)
