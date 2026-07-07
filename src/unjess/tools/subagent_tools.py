"""LLM-facing tools for subagent management.

Registers tools that let the LLM spawn subagents, list them,
send messages to them, read their replies, and kill them.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from unjess.tools import ToolRegistry
from unjess.subagents.manager import SubagentManager


def register_subagent_tools(
    registry: ToolRegistry,
    manager: SubagentManager,
    input_handler: Any = None,
    default_max_turns: int = 10,
) -> None:
    """Register all subagent management tools with the given registry.

    Args:
        registry: The tool registry to register tools into.
        manager: The subagent manager instance that handles lifecycle operations.
        input_handler: Optional GUI input handler for spawn approval dialogs.
            When provided (GUI mode), spawning shows a popup for model selection.
        default_max_turns: Default max LLM turns for subagents (from settings).
    """

    # ------------------------------------------------------------------
    # spawn_agent
    # ------------------------------------------------------------------
    def _spawn_agent(
        type_name: str,
        prompt: str,
        role: Optional[str] = None,
        model: str = "",
        tools: Optional[list[str]] = None,
        max_turns: int = default_max_turns,
    ) -> str:
        """Spawn a new subagent.

        Args:
            type_name: Agent type – one of: research, coder, reviewer, tester, self.
            prompt: Task description for the subagent.
            role: Optional human-readable role label, e.g. 'Codebase Researcher'.
            model: Model override for this subagent (empty = inherit parent's model).
            tools: Explicit tool list (empty = use type defaults).
            max_turns: Max LLM round-trips (default 10).

        Returns:
            JSON string with conversation_id and status, or error.
        """
        effective_model = model
        effective_max_turns = max_turns
        effective_tools = tools

        # In GUI mode, show spawn approval dialog for user confirmation
        if input_handler and hasattr(input_handler, "ask_spawn_approval"):
            spawn_result = input_handler.ask_spawn_approval(
                type_name=type_name,
                role=role or type_name,
                prompt=prompt,
                suggested_model=model,
                tools=tools,
                max_turns=max_turns,
            )
            if not spawn_result.get("approved", False):
                return json.dumps({"error": "Spawn cancelled by user."})
            # User may have overridden model and max_turns
            effective_model = spawn_result.get("model", model)
            effective_max_turns = spawn_result.get("max_turns", max_turns)

        try:
            conversation_id = manager.invoke(
                type_name=type_name,
                prompt=prompt,
                role=role or type_name,
                model_override=effective_model,
                tool_filter=effective_tools or [],
                max_turns=effective_max_turns,
            )
            return json.dumps({
                "conversation_id": conversation_id,
                "status": "spawned",
                "model": effective_model or "(parent default)",
                "max_turns": effective_max_turns,
            })
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    registry.register(
        name="spawn_agent",
        description=(
            "Spawn a subagent to work on a task in parallel. "
            "Max 3 concurrent subagents allowed — wait for existing ones to "
            "finish or kill them before spawning more. "
            "The user will be asked to approve and can override the model."
        ),
        parameters={
            "type": "object",
            "properties": {
                "type_name": {
                    "type": "string",
                    "description": (
                        "Agent type to spawn. "
                        "One of: research, coder, reviewer, tester, self."
                    ),
                },
                "prompt": {
                    "type": "string",
                    "description": "Task description for the subagent.",
                },
                "role": {
                    "type": "string",
                    "description": (
                        "Optional human-readable role label, "
                        "e.g. 'Codebase Researcher'."
                    ),
                },
                "model": {
                    "type": "string",
                    "description": (
                        "Model to use for this subagent. "
                        "Leave empty to inherit the parent's model. "
                        "Use a smaller model for simple tasks."
                    ),
                },
                "tools": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Explicit list of tool names for this subagent. "
                        "Leave empty to use the type's default tools."
                    ),
                },
                "max_turns": {
                    "type": "integer",
                    "description": (
                        "Max LLM round-trips for this subagent (default 10). "
                        "Use fewer for simple tasks, more for complex ones."
                    ),
                },
            },
            "required": ["type_name", "prompt"],
        },
        handler=_spawn_agent,
    )

    # ------------------------------------------------------------------
    # list_agents
    # ------------------------------------------------------------------
    def _list_agents() -> str:
        """List all active subagents.

        Returns:
            JSON array of agent summary dicts.
        """
        try:
            agents = manager.list_all()
            return json.dumps([
                {
                    "conversation_id": a.conversation_id,
                    "type_name": a.type_name,
                    "role": a.role,
                    "status": a.status.value,
                }
                for a in agents
            ])
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    registry.register(
        name="list_agents",
        description="List all active subagents and their statuses.",
        parameters={
            "type": "object",
            "properties": {},
        },
        handler=_list_agents,
    )

    # ------------------------------------------------------------------
    # send_to_agent
    # ------------------------------------------------------------------
    def _send_to_agent(conversation_id: str, message: str) -> str:
        """Send a message to a running subagent.

        Args:
            conversation_id: Target subagent's conversation ID.
            message: The message content to send.

        Returns:
            Success or failure message string.
        """
        try:
            manager.send_message(conversation_id, message)
            return json.dumps({
                "status": "sent",
                "conversation_id": conversation_id,
            })
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    registry.register(
        name="send_to_agent",
        description="Send a message to a subagent.",
        parameters={
            "type": "object",
            "properties": {
                "conversation_id": {
                    "type": "string",
                    "description": "The conversation ID of the target subagent.",
                },
                "message": {
                    "type": "string",
                    "description": "The message to send to the subagent.",
                },
            },
            "required": ["conversation_id", "message"],
        },
        handler=_send_to_agent,
    )

    # ------------------------------------------------------------------
    # kill_agent
    # ------------------------------------------------------------------
    def _kill_agent(conversation_id: str) -> str:
        """Kill a running subagent.

        Args:
            conversation_id: The conversation ID of the subagent to kill.

        Returns:
            Success or failure message string.
        """
        try:
            manager.kill(conversation_id)
            return json.dumps({
                "status": "killed",
                "conversation_id": conversation_id,
            })
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    registry.register(
        name="kill_agent",
        description="Kill a running subagent.",
        parameters={
            "type": "object",
            "properties": {
                "conversation_id": {
                    "type": "string",
                    "description": "The conversation ID of the subagent to kill.",
                },
            },
            "required": ["conversation_id"],
        },
        handler=_kill_agent,
    )

    # ------------------------------------------------------------------
    # read_agent_messages
    # ------------------------------------------------------------------
    def _read_agent_messages() -> str:
        """Read pending messages from subagents.

        Returns:
            JSON array of message dicts.
        """
        try:
            messages = manager.receive_messages()
            return json.dumps([
                {
                    "sender": m.sender,
                    "content": m.content,
                    "type": m.message_type,
                }
                for m in messages
            ])
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    registry.register(
        name="read_agent_messages",
        description="Read pending messages received from subagents.",
        parameters={
            "type": "object",
            "properties": {},
        },
        handler=_read_agent_messages,
    )
