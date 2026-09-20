"""Agent loop — the core think/act/observe cycle.

Orchestrates LLM calls, tool execution, conversation management,
context budgeting, logging, loop control, repo map injection,
model routing, and memory.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

from unjess.config import Settings
from unjess.context_manager import ContextManager
from unjess.conversation_logger import ConversationLogger
from unjess.display import Display

if TYPE_CHECKING:
    from unjess.protocols import DisplayProtocol, InputProtocol
from unjess.llm.base import LLMResponse, StreamChunk, ToolCall, Usage
from unjess.llm.router import ProviderRouter
from unjess.model_router import ArchitectMode, ModelRouter
from unjess.permissions import PermissionManager
from unjess.system_prompt import build_system_prompt, load_project_context, load_user_rules
from unjess.tools import ToolRegistry
from unjess.undo import UndoManager
from unjess.mentions import MentionResolver
from unjess.conversation_logger import is_free_model

logger = logging.getLogger(__name__)

_MAX_TOOL_RETRIES = 3
_STUCK_WINDOW = 5  # check last N tool calls for repetition
_GOAL_MAX_ITERATIONS = 200  # extended limit for /goal mode
_MAX_TOOL_RESULT_CHARS = 4000  # truncate tool results stored in conversation
_CONSECUTIVE_READ_NUDGE = 3  # turns of pure reading before injecting a stop-reading notice
_CONSECUTIVE_READ_HARD_LIMIT = 5  # turns of pure reading before halting and forcing synthesis

# Tools safe to execute in parallel (read-only)
_PARALLEL_SAFE_TOOLS = frozenset({
    "read_file", "view_file", "list_dir", "grep_search", "find_by_name",
    "search_web", "read_url",
    "browser_get_text", "browser_screenshot",
})

# All read-only reconnaissance tools
_READ_ONLY_TOOLS = frozenset({
    "read_file", "view_file", "list_dir", "grep_search", "find_by_name",
    "search_web", "read_url",
    "browser_get_text", "browser_screenshot",
    "list_agents", "read_agent_messages",
})


class Agent:
    """The main agent — runs the think/act/observe loop.

    Args:
        settings: Runtime configuration.
        router: LLM provider router.
        tool_registry: Registered tools.
        display: Display layer for output (implements DisplayProtocol).
        permissions: Permission manager.
        context_manager: Context window manager.
        conv_logger: Conversation logger.
        undo_manager: Undo/rollback manager.
        mention_resolver: Optional @ mention resolver.
        repo_map: Optional pre-built repo map instance.
        memory_store: Optional cross-session memory store.
        rag_engine: Optional RAG engine for semantic context retrieval.
        knowledge_graph: Optional knowledge graph for entity tracking.
        input_handler: Optional InputProtocol for UI-agnostic user input.
    """

    def __init__(
        self,
        settings: Settings,
        router: ProviderRouter,
        tool_registry: ToolRegistry,
        display: DisplayProtocol | Display,
        permissions: PermissionManager,
        context_manager: Optional[ContextManager] = None,
        conv_logger: Optional[ConversationLogger] = None,
        undo_manager: Optional[UndoManager] = None,
        mention_resolver: Optional[MentionResolver] = None,
        repo_map: Optional[Any] = None,
        memory_store: Optional[Any] = None,
        rag_engine: Optional[Any] = None,
        knowledge_graph: Optional[Any] = None,
        skill_engine: Optional[Any] = None,
        input_handler: Optional[InputProtocol] = None,
        _is_child: bool = False,
        _max_iterations: int = 0,
    ) -> None:
        self._settings = settings
        self._router = router
        self._tools = tool_registry
        self._display = display
        self._permissions = permissions
        self._context_manager = context_manager or ContextManager(
            model=settings.model,
            context_window_override=getattr(settings, "context_window_override", 0),
            enable_compaction=getattr(settings, "enable_compaction", True),
        )
        self._logger = conv_logger
        self._undo_manager = undo_manager
        self._mention_resolver = mention_resolver
        self._repo_map = repo_map
        self._memory_store = memory_store
        self._rag_engine = rag_engine
        self._knowledge_graph = knowledge_graph
        self._skill_engine = skill_engine
        self._input_handler = input_handler
        self._conversation: list[dict[str, Any]] = []
        self._recent_tool_calls: deque[str] = deque(maxlen=_STUCK_WINDOW * 2)  # for stuck detection
        self._is_child = _is_child

        # Max iterations: explicit override > settings default
        self._max_iterations = _max_iterations or settings.max_iterations

        # Heavy setup — skip for child agents (they don't need planning/routing)
        if not _is_child:
            self._model_router = ModelRouter(settings)
            self._architect = ArchitectMode(settings, self._model_router)
            workspace = Path(settings.workspace)
            self._project_context = load_project_context(workspace)
            self._user_rules = load_user_rules(workspace)
        else:
            self._model_router = None  # type: ignore[assignment]
            self._architect = None  # type: ignore[assignment]
            self._project_context = ""
            self._user_rules = ""

        self._planning_mode = False  # set per-request based on complexity
        self._goal_mode = False  # extended autonomy for /goal command
        self._abort_requested = False  # set by GUI stop button to abort generation

        # --- Subagent Manager ---
        if not _is_child:
            self._init_subagent_manager(Path(settings.workspace))
            self._init_scheduler()
        else:
            self._subagent_manager = None
            self._scheduler = None
            self._custom_system_prompt: str = ""

    def _init_scheduler(self) -> None:
        """Initialize background scheduler and register native schedule tool."""
        try:
            from unjess.scheduler import Scheduler
            from unjess.tools.schedule_tools import register_schedule_tools
            self._scheduler = Scheduler()
            self._scheduler.start()
            register_schedule_tools(self._tools, self._scheduler)
        except Exception as exc:
            logger.warning("Failed to initialize scheduler: %s", exc)
            self._scheduler = None

    def _init_subagent_manager(self, workspace: Path) -> None:
        """Initialize the subagent manager with built-in types and a factory."""
        from unjess.subagents.manager import SubagentManager
        from unjess.subagents.types import register_builtin_types

        self._subagent_manager = SubagentManager(
            parent_id="main",
            workspace=workspace,
        )
        register_builtin_types(self._subagent_manager)
        self._subagent_manager.set_agent_factory(self._create_child_agent)

        # Register LLM-facing subagent tools
        try:
            from unjess.tools.subagent_tools import register_subagent_tools
            register_subagent_tools(
                self._tools, self._subagent_manager,
                input_handler=self._input_handler,
                default_max_turns=self._settings.subagent_max_turns,
            )
        except Exception as exc:
            logger.warning("Failed to register subagent tools: %s", exc)

    def _create_child_agent(
        self,
        agent_info: Any,
        type_config: dict[str, Any],
        workspace: Path,
        message_bus: Any,
    ) -> Any:
        """Factory function that creates a child agent runner.

        Returns a callable that runs the child agent's task and returns a result.
        The child agent uses a silent SubagentDisplay so its output doesn't
        pollute the parent's chat. Results are sent back via the message bus.

        Supports model override, tool filtering, and turn limits from
        ``agent_info`` fields set by the spawn dialog.
        """
        from dataclasses import replace as _replace
        from unjess.subagents.display import SubagentDisplay
        from unjess.subagents.messaging import Message

        # Subagent tool names — never copied to children
        _SUBAGENT_TOOLS = frozenset({
            "spawn_agent", "list_agents", "send_to_agent",
            "kill_agent", "read_agent_messages",
        })

        def _runner() -> str:
            """Run the child agent and return its result."""
            # Create a silent display for this child
            child_display = SubagentDisplay(agent_id=agent_info.conversation_id)

            # --- Tool filtering ---
            # Priority: agent_info.tool_filter > type_config tools > "self" fallback
            child_tools = ToolRegistry()
            if getattr(agent_info, "tool_filter", None):
                allowed = agent_info.tool_filter
            elif type_config.get("tools"):
                allowed = type_config["tools"]
            else:
                # "self" type — copy all parent tools except subagent tools
                allowed = [
                    n for n in self._tools.tool_names
                    if n not in _SUBAGENT_TOOLS
                ]

            for tool_name in allowed:
                parent_tool = self._tools.get_tool(tool_name)
                if parent_tool:
                    child_tools.register(
                        name=parent_tool.name,
                        description=parent_tool.description,
                        parameters=parent_tool.parameters,
                        handler=parent_tool.handler,
                    )

            # --- Model override ---
            child_settings = self._settings
            model_override = getattr(agent_info, "model_override", "")
            if model_override:
                child_settings = _replace(self._settings, model=model_override)

            # --- Turn limit ---
            max_turns = getattr(agent_info, "max_turns", 10)

            # Build system prompt
            child_prompt = type_config.get("system_prompt", "")
            if not child_prompt:
                child_prompt = (
                    "You are a helpful AI coding assistant. "
                    "Complete the task described below."
                )
            child_prompt += f"\n\nYour workspace is: {workspace}"

            # Create a child agent instance with the silent display
            child_agent = Agent(
                settings=child_settings,
                router=self._router,
                tool_registry=child_tools,
                display=child_display,
                permissions=self._permissions,
                _is_child=True,
                _max_iterations=max_turns,
            )
            # Inject the type's system prompt
            child_agent._custom_system_prompt = child_prompt

            # Run the prompt
            result = ""
            try:
                child_agent.run(agent_info.prompt)
                # Extract the last assistant message as the result
                for msg in reversed(child_agent._conversation):
                    if msg.get("role") == "assistant" and msg.get("content"):
                        result = msg["content"]
                        break
                if not result:
                    result = child_display.captured_output or "Task completed (no text response)."
            except Exception as exc:
                result = f"Error: {exc}"

            # Send result back to parent via message bus
            message_bus.send(Message(
                sender=agent_info.conversation_id,
                recipient=self._subagent_manager.parent_id,
                content=result[:4000],  # Cap at 4k chars
                message_type="result",
            ))

            return result

        return _runner

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, user_input: str, images: list[dict[str, str]] | None = None) -> None:
        """Process a user message through the agent loop.

        1. Create undo checkpoint
        2. Build system prompt (with repo map, context, rules, memory)
        3. Append user message
        4. Truncate conversation if needed
        5. Call LLM (streaming)
        6. If tool calls → execute, append results, loop back
        7. If text only → display, break

        Args:
            user_input: The user's message.
            images: Optional list of image dicts with 'data' (base64),
                'mime_type', and 'name' keys.
        """
        # Log user input
        if self._logger:
            self._logger.log_user_input(user_input)

        # Reset abort flag and recent tool calls cache at start of each turn
        self._abort_requested = False
        self._recent_tool_calls.clear()

        # Resolve @ mentions (inject file/git/dir context)
        enriched_input = user_input
        if self._mention_resolver and "@" in user_input:
            enriched_input, mentions = self._mention_resolver.process_message(user_input)
            if mentions:
                resolved = [m for m in mentions if m.is_resolved]
                if resolved:
                    self._display.show_info(
                        f"Resolved {len(resolved)} mention(s): "
                        + ", ".join(m.raw for m in resolved)
                    )

        # Create undo checkpoint before any changes
        if self._undo_manager:
            self._undo_manager.checkpoint(label=user_input[:50])

        # Build conversation message — multi-part if images attached
        if images:
            import base64
            content_parts: list[dict[str, Any]] = [
                {"type": "text", "text": enriched_input},
            ]
            for img in images:
                raw_d = img.get("data", "")
                b64_str = base64.b64encode(raw_d).decode("ascii") if isinstance(raw_d, bytes) else str(raw_d)
                content_parts.append({
                    "type": "image",
                    "data": b64_str,
                    "mime_type": img.get("mime_type", "image/png"),
                })
            self._conversation.append({"role": "user", "content": content_parts})
        else:
            self._conversation.append({"role": "user", "content": enriched_input})

        # Route the request — detect if this is a complex/planning task
        # Skip for child agents (they just execute, no planning)
        planning_setting = getattr(self._settings, "planning_mode", "auto")
        if planning_setting == "on":
            self._planning_mode = True
            self._display.show_info("Planning mode active (forced by settings)")
        elif planning_setting == "off":
            self._planning_mode = False
        else:
            # Auto-detect
            if self._model_router and self._architect:
                route = self._model_router.route(enriched_input)
                self._planning_mode = self._architect.should_use(
                    route.task_type, route.complexity
                )

                if self._planning_mode:
                    self._display.show_info(
                        f"Complex task detected ({route.task_type}/{route.complexity}) — planning mode active"
                    )

        # Build system prompt with all context (includes planning mode if active)
        system_prompt = self._build_full_system_prompt()
        iteration = 0
        consecutive_errors = 0
        consecutive_read_turns = 0

        max_iter = _GOAL_MAX_ITERATIONS if self._goal_mode else self._max_iterations

        while iteration < max_iter:
            iteration += 1

            # Check abort flag (set by GUI stop button)
            if self._abort_requested:
                self._display.show_info("⏹ Generation stopped by user.")
                self._abort_requested = False
                break

            # Auto-compact if context is getting full (only at the start of a turn, and if enabled)
            self._context_manager.enable_compaction = getattr(self._settings, "enable_compaction", True)
            if (
                iteration == 1
                and self._context_manager.enable_compaction
                and self._context_manager.needs_compaction(system_prompt, self._conversation)
            ):
                pre = self._context_manager.get_context_breakdown(
                    system_prompt, self._conversation
                ).get("utilization_pct", 0)

                # Use LLM to summarize old messages
                def _summarize(text: str) -> str:
                    resp = self._router.chat(
                        messages=[{"role": "user", "content": text}],
                        tools=None,
                    )
                    return resp.text

                self._conversation = self._context_manager.compact(
                    self._conversation, summarize_fn=_summarize
                )
                post = self._context_manager.get_context_breakdown(
                    system_prompt, self._conversation
                ).get("utilization_pct", 0)
                self._display.show_info(
                    f"⚡ Context auto-compacted ({pre:.0f}% → {post:.0f}%)"
                )

            # Truncate conversation if approaching context limit
            self._conversation = self._context_manager.truncate_conversation(
                self._conversation, system_prompt
            )

            # Build messages list: system + conversation
            messages = [{"role": "system", "content": system_prompt}] + self._conversation

            # Get tool schemas
            tools = self._tools.get_tools() or None

            # Call LLM (streaming)
            try:
                response = self._stream_response(messages, tools)
            except Exception as exc:
                error_msg = str(exc)
                logger.exception("LLM call failed")
                if self._logger:
                    self._logger.log_error("llm", error_msg)

                # --- Auto-rotate on model errors in free mode ---
                if (
                    getattr(self._settings, "free_mode_enabled", False)
                    and ("404" in error_msg or "not_found" in error_msg.lower()
                         or "does not exist" in error_msg.lower())
                ):
                    self._display.show_error(
                        f"Model unavailable: {self._settings.model}"
                    )
                    # Try to rotate via free mode
                    try:
                        from unjess.free_mode import FreeMode
                        fm = FreeMode(
                            settings=self._settings,
                            router=self._router,
                            display=self._display,
                        )
                        # Force switch away from current model
                        next_tier = fm._pick_best_tier(exclude=self._settings.model)
                        if next_tier:
                            fm._switch_to(next_tier)
                            self._display.show_info(
                                f"Auto-switched to: {next_tier.display_name}"
                            )
                            continue  # retry with new model
                    except Exception as exc:
                        logger.debug("Free mode auto-rotation failed: %s", exc)
                    self._display.show_info(
                        "No fallback available. Try /model to switch manually."
                    )
                    break

                # --- Auto-compact on context length exceeded ---
                if "context_length_exceeded" in error_msg.lower() or (
                    "reduce the length" in error_msg.lower()
                    and "400" in error_msg
                ):
                    if consecutive_errors < 2:  # only retry once
                        self._display.show_info(
                            "⚡ Context too long — auto-compacting conversation..."
                        )
                        # Emergency truncation: keep only last 4 messages
                        if len(self._conversation) > 4:
                            self._conversation = self._conversation[-4:]
                        consecutive_errors += 1
                        continue  # retry with shorter context
                    else:
                        self._display.show_error(
                            "Context still too long after compaction. "
                            "Try /clear to start fresh or /model to switch to a model with more context."
                        )
                        break

                # Suggest fixes for common errors
                self._display.show_error(f"LLM error: {error_msg}")

                # --- Auto-retry on transient server errors (500/502/503) ---
                if any(code in error_msg for code in ("500", "502", "503")) or "internal" in error_msg.lower():
                    consecutive_errors += 1
                    if consecutive_errors < 3:
                        import time as _time
                        wait = min(2 ** consecutive_errors, 10)
                        self._display.show_info(
                            f"⚡ Server error — retrying in {wait}s (attempt {consecutive_errors}/3)..."
                        )
                        _time.sleep(wait)
                        continue  # retry
                    self._display.print_markdown(
                        "\n\n⚠️ **Server error** — the provider returned an internal error after 3 retries. "
                        "Try again in a moment, or use `/model` to switch models.\n"
                    )
                    break

                if "failed_generation" in error_msg.lower():
                    self._display.show_info(
                        "This model couldn't generate a valid tool call.\n"
                        "  Try: /model to switch models, or /provider for a different provider.\n"
                        "  Groq's Llama models sometimes struggle with complex tool schemas."
                    )
                    consecutive_errors += 1
                    if consecutive_errors < 3:
                        continue  # retry
                    break
                elif "403" in error_msg or "401" in error_msg or "PermissionDenied" in error_msg:
                    self._display.print_markdown(
                        "\n\n⚠️ **Authentication error** — run `/setup` to reconfigure your provider, "
                        "or `/model` to switch models.\n"
                    )
                elif "rate" in error_msg.lower() or "429" in error_msg:
                    # Attempt key rotation before giving up
                    prov_name = self._settings.provider or "google"
                    rotated = False
                    if hasattr(self._router, '_rotate_key'):
                        rotated = self._router._rotate_key(prov_name)
                    if rotated:
                        self._display.show_info(
                            f"🔄 Rate limited — rotated {prov_name} API key, retrying..."
                        )
                        continue  # retry with new key

                    # Extract retry delay from error message (e.g. "retryDelay: '58s'")
                    import re as _re
                    delay_match = _re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+)", error_msg)
                    if delay_match and consecutive_errors < 2:
                        wait_secs = min(int(delay_match.group(1)), 120)  # cap at 2 min
                        self._display.print_markdown(
                            f"\n\n⏳ **Rate limited** — waiting {wait_secs}s before retrying...\n"
                        )
                        import time as _time
                        _time.sleep(wait_secs)
                        consecutive_errors += 1
                        continue  # retry after waiting

                    self._display.print_markdown(
                        "\n\n⚠️ **Rate limited** — no more keys to rotate. "
                        "Wait a moment or try `/free` for free-tier models.\n"
                    )
                else:
                    # Generic error — show in chat so user sees it
                    self._display.print_markdown(
                        f"\n\n⚠️ **Error**: {error_msg[:300]}\n"
                    )
                break

            # Log model response
            if self._logger and response.usage:
                tc_dicts = [
                    {"name": tc.name, "arguments": tc.arguments}
                    for tc in response.tool_calls
                ] if response.tool_calls else None

                self._logger.log_model_response(
                    content=response.text,
                    tool_calls=tc_dicts,
                    tokens_in=response.usage.prompt_tokens,
                    tokens_out=response.usage.completion_tokens,
                    model=response.model,
                    thinking_tokens=response.usage.thinking_tokens,
                    cache_read_tokens=response.usage.cache_read_tokens,
                    cache_creation_tokens=response.usage.cache_creation_tokens,
                )

            # Budget enforcement
            if (
                self._settings.max_cost_per_session > 0
                and self._logger
                and self._logger.cost_tracker.total_cost >= self._settings.max_cost_per_session
            ):
                self._display.show_warning(
                    f"Session cost limit reached "
                    f"(${self._logger.cost_tracker.total_cost:.4f} >= "
                    f"${self._settings.max_cost_per_session:.4f}). Stopping."
                )
                break

            # Handle tool calls
            if response.tool_calls:
                # Track consecutive read-only turns to prevent analysis paralysis
                is_pure_read = all(tc.name in _READ_ONLY_TOOLS for tc in response.tool_calls)
                if is_pure_read:
                    consecutive_read_turns += 1
                else:
                    consecutive_read_turns = 0

                consecutive_errors = self._handle_tool_calls(
                    response, consecutive_errors, consecutive_read_turns=consecutive_read_turns
                )
                if consecutive_errors >= _MAX_TOOL_RETRIES:
                    self._display.show_error(
                        f"Too many consecutive tool errors ({_MAX_TOOL_RETRIES}). Stopping."
                    )
                    break

                # Hard limit for consecutive read reconnaissance turns
                hard_read_limit = (
                    _CONSECUTIVE_READ_HARD_LIMIT * 2
                    if self._goal_mode
                    else _CONSECUTIVE_READ_HARD_LIMIT
                )
                if is_pure_read and consecutive_read_turns >= hard_read_limit:
                    self._display.show_warning(
                        f"Analysis paralysis detected ({consecutive_read_turns} consecutive read turns). Forcing final synthesis."
                    )
                    self._conversation.append({
                        "role": "user",
                        "content": (
                            "[SYSTEM DIRECTIVE: Maximum exploration budget reached. "
                            "Do NOT call any more tools. Synthesize the findings from the files you inspected "
                            "and output your implementation plan or answer now.]"
                        ),
                    })
                    final_messages = [{"role": "system", "content": system_prompt}] + self._conversation
                    try:
                        final_response = self._stream_response(final_messages, tools=None)
                        if self._settings.show_stats and final_response.usage:
                            self._display.show_stats(
                                final_response.usage.prompt_tokens,
                                final_response.usage.completion_tokens,
                                model=final_response.model,
                            )
                    except Exception as exc:
                        logger.debug("Failed to get forced final response: %s", exc)
                    break

                # Stuck detection
                if self._is_stuck():
                    self._display.show_warning(
                        "Agent appears stuck repeating the same actions. Stopping."
                    )
                    break

                continue  # loop back to let model see results

            # Text-only response — show per-request stats and break
            if self._settings.show_stats and response.usage:
                cost = None
                model_is_free = False
                quota_pct = None
                if self._logger:
                    cost = self._logger.cost_tracker.estimate_cost(
                        response.model,
                        response.usage.prompt_tokens,
                        response.usage.completion_tokens,
                    )
                    model_is_free = is_free_model(response.model)

                # Get free quota remaining if in free mode
                free_mode = getattr(self, '_free_mode', None)
                if free_mode is None and hasattr(self._settings, '_free_mode_ref'):
                    free_mode = self._settings._free_mode_ref
                if free_mode and free_mode.is_active and free_mode._current_tier:
                    quota_pct = free_mode._tracker.remaining_pct(
                        response.model, free_mode._current_tier
                    )

                self._display.show_stats(
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                    cost=cost,
                    model=response.model,
                    is_free=model_is_free,
                    thinking_tokens=response.usage.thinking_tokens,
                    cache_read_tokens=response.usage.cache_read_tokens,
                    cache_creation_tokens=response.usage.cache_creation_tokens,
                    quota_pct=quota_pct,
                )

            # In goal mode, only stop if the model explicitly finishes
            if self._goal_mode and response.text:
                # Keep going — model will decide when to stop
                continue

            break  # wait for next user input

        if iteration >= max_iter:
            self._display.show_warning(
                f"Reached max iterations ({max_iter}). Stopping."
            )

        # Save knowledge graph at the end of the turn
        if self._knowledge_graph:
            try:
                self._knowledge_graph.save()
            except Exception as exc:
                logger.debug("Failed to save knowledge graph at end of turn: %s", exc)

    def clear_history(self) -> None:
        """Clear the conversation history."""
        self._conversation.clear()
        self._recent_tool_calls.clear()

    @property
    def conversation(self) -> list[dict[str, Any]]:
        """The current conversation messages (mutable reference)."""
        return self._conversation

    @conversation.setter
    def conversation(self, value: list[dict[str, Any]]) -> None:
        """Replace the conversation history."""
        self._conversation = value

    @property
    def conversation_length(self) -> int:
        """Number of messages in the conversation."""
        return len(self._conversation)

    @property
    def tool_registry(self) -> ToolRegistry:
        """The agent's tool registry."""
        return self._tools

    # ------------------------------------------------------------------
    # System prompt assembly
    # ------------------------------------------------------------------

    def _build_full_system_prompt(self) -> str:
        """Build the complete system prompt with all optional sections."""
        # Child agents use a lightweight custom prompt
        if self._is_child and getattr(self, '_custom_system_prompt', ''):
            return self._custom_system_prompt

        # Repo map
        repo_map_str = ""
        if self._repo_map:
            try:
                repo_map_str = self._repo_map.format(max_tokens=1500)
            except Exception as exc:
                logger.debug("Repo map formatting failed: %s", exc)

        # Memory context
        memory_str = ""
        if self._memory_store:
            try:
                learned_rules = self._memory_store.build_rules_block(
                    workspace=self._settings.workspace,
                )
                memory_str = self._memory_store.build_context_block(
                    max_summaries=3,
                    workspace=self._settings.workspace,
                )
                # Append learned rules to user_rules
                if learned_rules:
                    user_rules = self._user_rules + "\n\n" + learned_rules if self._user_rules else learned_rules
                else:
                    user_rules = self._user_rules
            except Exception:
                user_rules = self._user_rules
        else:
            user_rules = self._user_rules

        # RAG context — semantic retrieval from past sessions
        rag_context = ""
        if self._rag_engine and self._rag_engine.is_available and self._conversation:
            try:
                # Use the latest user message as the query
                latest_user = ""
                for msg in reversed(self._conversation):
                    if msg.get("role") == "user":
                        content = msg.get("content", "")
                        if isinstance(content, str):
                            latest_user = content
                        break
                if latest_user:
                    rag_context = self._rag_engine.build_context_block(
                        query=latest_user, max_tokens=500
                    )
            except Exception as exc:
                logger.debug("RAG retrieval failed: %s", exc)

        # Knowledge graph context
        kg_context = ""
        if self._knowledge_graph and self._conversation:
            try:
                latest_user = ""
                for msg in reversed(self._conversation):
                    if msg.get("role") == "user":
                        content = msg.get("content", "")
                        if isinstance(content, str):
                            latest_user = content
                        break
                if latest_user:
                    kg_context = self._knowledge_graph.get_context_for(
                        query=latest_user, max_entities=8
                    )
            except Exception as exc:
                logger.debug("Knowledge graph retrieval failed: %s", exc)

        # Combine memory context with RAG and KG
        if rag_context:
            memory_str = (memory_str + "\n\n" + rag_context) if memory_str else rag_context
        if kg_context:
            memory_str = (memory_str + "\n\n" + kg_context) if memory_str else kg_context

        # --- Skill matching ---
        skill_context = ""
        if self._skill_engine and not self._is_child:
            try:
                # Build the catalog (always present — just names + one-liner)
                catalog = self._skill_engine.catalog_for_prompt()

                # Match skills against latest user message
                latest_user = ""
                for msg in reversed(self._conversation):
                    if msg.get("role") == "user":
                        content = msg.get("content", "")
                        if isinstance(content, str):
                            latest_user = content
                        break

                matched_instructions = ""
                if latest_user:
                    matched = self._skill_engine.match(latest_user)
                    if matched:
                        # Load full instructions only for the top match
                        top_skill = matched[0]
                        instructions = self._skill_engine.load(top_skill.name)
                        if instructions:
                            matched_instructions = (
                                f"\n\n## Active Skill: {top_skill.name}\n"
                                f"{instructions}"
                            )
                            logger.debug("Skill matched: %s (score matched)", top_skill.name)

                if catalog or matched_instructions:
                    skill_context = catalog + matched_instructions
            except Exception as exc:
                logger.debug("Skill matching failed: %s", exc)

        user_profile_context = ""
        if self._memory_store:
            try:
                user_profile_context = self._memory_store.get_user_profile().to_context()
            except Exception as exc:
                logger.debug("Failed to get user profile context: %s", exc)

        # Detect if running in GUI mode
        _gui_mode = type(self._display).__name__ == "GUIDisplay"

        return build_system_prompt(
            self._settings,
            self._tools,
            repo_map=repo_map_str,
            project_context=self._project_context,
            user_rules=user_rules,
            memory_context=memory_str,
            planning_mode=self._planning_mode,
            gui_mode=_gui_mode,
            skill_context=skill_context,
            user_profile_context=user_profile_context,
        )

    # ------------------------------------------------------------------
    # Loop control
    # ------------------------------------------------------------------

    def _is_stuck(self) -> bool:
        """Detect if the agent is repeating the same tool calls.

        Checks the last ``_STUCK_WINDOW`` tool call signatures for
        repetition. In goal mode, uses a higher threshold to allow
        more persistence.
        """
        if not getattr(self._settings, "enable_stuck_detection", True):
            return False

        window_size = _STUCK_WINDOW * 2 if self._goal_mode else _STUCK_WINDOW
        if len(self._recent_tool_calls) < window_size:
            return False

        window = list(self._recent_tool_calls)[-window_size:]
        unique_ratio = len(set(window)) / len(window)
        max_repeats = max(window.count(call) for call in set(window))
        return unique_ratio <= 0.3 or max_repeats >= len(window)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _stream_response(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> LLMResponse:
        """Call the LLM with streaming and assemble the response.

        Text chunks are printed as they arrive. Tool calls are accumulated
        and returned in the final LLMResponse.
        """
        stream = self._router.chat_stream(messages=messages, tools=tools)

        text_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        usage = None
        model = self._settings.model
        is_thinking = False
        first_text_chunk = True  # Track if we need a separator

        for chunk in stream:
            # Handle thinking/reasoning content
            if chunk.thinking:
                if not is_thinking:
                    is_thinking = True
                    self._display.show_thinking_start()
                thinking_parts.append(chunk.thinking)

            if chunk.text:
                # End thinking indicator if we were thinking
                if is_thinking:
                    is_thinking = False
                    self._display.show_thinking_end()
                    # Show abbreviated thinking summary
                    full_thinking = "".join(thinking_parts)
                    if full_thinking:
                        self._display.show_thinking(full_thinking)

                # Visual separator before AI response starts
                if first_text_chunk:
                    self._display.console.print()
                    first_text_chunk = False

                self._display.console.print(chunk.text, end="", highlight=False)
                text_parts.append(chunk.text)

            if chunk.tool_call:
                # End thinking indicator if we were thinking
                if is_thinking:
                    is_thinking = False
                    self._display.show_thinking_end()
                    full_thinking = "".join(thinking_parts)
                    if full_thinking:
                        self._display.show_thinking(full_thinking)
                tool_calls.append(chunk.tool_call)

            if chunk.usage:
                usage = chunk.usage

            if chunk.done:
                break

        # Handle thinking that continued to the end
        if is_thinking:
            self._display.show_thinking_end()
            full_thinking = "".join(thinking_parts)
            if full_thinking:
                self._display.show_thinking(full_thinking)

        # Print newline after streamed text
        if text_parts:
            self._display.console.print()

        full_text = "".join(text_parts)
        full_thinking_text = "".join(thinking_parts)

        # Append assistant message to conversation
        assistant_msg: dict[str, Any] = {"role": "assistant", "content": full_text}
        
        # Fallback for models that output raw JSON tool calls in text/thinking
        if not tool_calls:
            combined = full_thinking_text + "\n" + full_text
            import uuid
            # Try parsing line-by-line first (common for multiple raw tool calls)
            for line in combined.splitlines():
                line = line.strip()
                if line.startswith("{") and line.endswith("}"):
                    try:
                        parsed = json.loads(line)
                        if isinstance(parsed, dict) and "name" in parsed and "arguments" in parsed:
                            tool_calls.append(
                                ToolCall(
                                    id=f"call_{uuid.uuid4().hex[:8]}",
                                    name=parsed["name"],
                                    arguments=parsed["arguments"] if isinstance(parsed["arguments"], dict) else {},
                                )
                            )
                    except Exception:
                        pass
            
            # If still nothing, try extracting any JSON object from the combined text
            if not tool_calls and "{" in combined and "}" in combined:
                try:
                    import re
                    # Look for { "name": ..., "arguments": ... }
                    match = re.search(r'\{\s*"name"\s*:.*"arguments"\s*:.*\}', combined, re.DOTALL)
                    if match:
                        parsed = json.loads(match.group(0))
                        if isinstance(parsed, dict) and "name" in parsed and "arguments" in parsed:
                            tool_calls.append(
                                ToolCall(
                                    id=f"call_{uuid.uuid4().hex[:8]}",
                                    name=parsed["name"],
                                    arguments=parsed["arguments"] if isinstance(parsed["arguments"], dict) else {},
                                )
                            )
                except Exception:
                    pass

        if tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                    **({"thought_signature": tc.thought_signature}
                       if tc.thought_signature else {}),
                }
                for tc in tool_calls
            ]
        self._conversation.append(assistant_msg)

        return LLMResponse(
            text=full_text,
            thinking_text=full_thinking_text,
            tool_calls=tool_calls,
            usage=usage or Usage(),
            finish_reason="tool_calls" if tool_calls else "stop",
            model=model,
        )

    def _handle_tool_calls(
        self,
        response: LLMResponse,
        consecutive_errors: int,
        consecutive_read_turns: int = 0,
    ) -> int:
        """Execute tool calls and append results to conversation.

        Read-only tools are executed in parallel via ThreadPoolExecutor.
        Write tools are executed sequentially after all reads complete.

        Returns the updated consecutive_errors count.
        """
        # Determine if we should append a nudge notice to the final tool result
        nudge_limit = (
            _CONSECUTIVE_READ_NUDGE * 2
            if self._goal_mode
            else _CONSECUTIVE_READ_NUDGE
        )
        nudge_notice = ""
        if consecutive_read_turns >= nudge_limit:
            nudge_notice = (
                f"\n\n[SYSTEM DIRECTIVE: You have completed {consecutive_read_turns} consecutive rounds of file reading. "
                "You now have sufficient context. STOP calling read tools. Either: 1) present your implementation plan / response "
                "to the user, or 2) begin making the necessary code edits.]"
            )

        # Partition into parallel-safe reads and sequential writes
        read_calls = [tc for tc in response.tool_calls if tc.name in _PARALLEL_SAFE_TOOLS]
        write_calls = [tc for tc in response.tool_calls if tc.name not in _PARALLEL_SAFE_TOOLS]

        # Execute read calls in parallel
        if len(read_calls) > 1:
            consecutive_errors = self._execute_parallel(
                read_calls, consecutive_errors,
                nudge_notice=nudge_notice if not write_calls else "",
            )
        elif read_calls:
            consecutive_errors = self._execute_sequential(
                read_calls, consecutive_errors,
                nudge_notice=nudge_notice if not write_calls else "",
            )

        # Execute write calls sequentially
        if write_calls:
            consecutive_errors = self._execute_sequential(
                write_calls, consecutive_errors,
                nudge_notice=nudge_notice,
            )

        return consecutive_errors

    def _execute_sequential(
        self,
        tool_calls: list[ToolCall],
        consecutive_errors: int,
        nudge_notice: str = "",
    ) -> int:
        """Execute tool calls one at a time."""
        for tc in tool_calls:
            self._display.show_tool_call(tc.name, tc.arguments)

            if self._logger:
                self._logger.log_tool_call(tc.name, tc.arguments)

            # Track for stuck detection
            call_sig = f"{tc.name}({json.dumps(tc.arguments, sort_keys=True)})"
            recent_calls = list(self._recent_tool_calls)

            # Target path check
            target_p = tc.arguments.get("path", tc.arguments.get("file_path", tc.arguments.get("AbsolutePath", "")))
            norm_target = str(target_p).replace("\\", "/").strip().lower() if target_p else ""

            # Duplicate read guard:
            # 1. Exact duplicate call signature (2+ times)
            # 2. Same file read across multiple different slices/line ranges (3+ times)
            is_exact_dup = recent_calls.count(call_sig) >= 2
            is_path_loop = False
            if norm_target and tc.name in ("read_file", "view_file"):
                path_read_count = sum(
                    1 for sig in recent_calls
                    if ("read_file" in sig or "view_file" in sig) and norm_target in sig.lower()
                )
                if path_read_count >= 2:
                    is_path_loop = True

            if tc.name in ("read_file", "view_file", "list_dir", "grep_search") and (is_exact_dup or is_path_loop):
                self._recent_tool_calls.append(call_sig)
                self._display.show_info(f"Notice: Redundant re-read skipped for {tc.name} ({target_p or 'same args'})")
                notice_result = (
                    f"Notice: You have ALREADY inspected '{target_p or tc.name}' multiple times in the recent context. "
                    "Do NOT re-read the same file repeatedly. Proceed directly to making necessary code edits or providing your final response."
                )
                self._conversation.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.name,
                    "content": notice_result,
                })
                continue

            self._recent_tool_calls.append(call_sig)

            # Invalidate recent read signatures if writing to a file so updated contents can be read
            if tc.name in ("write_file", "write_to_file", "edit_file", "create_file", "replace_file_content", "multi_replace_file_content"):
                target_p = tc.arguments.get("path", tc.arguments.get("file_path", tc.arguments.get("TargetFile", "")))
                if target_p:
                    self._recent_tool_calls = deque(
                        [sig for sig in self._recent_tool_calls if target_p not in sig],
                        maxlen=_STUCK_WINDOW * 2,
                    )

            # Planning mode enforcement: block write/edit on non-plan files
            # when planning mode is active and no plan.md has been created yet.
            if (
                self._planning_mode
                and tc.name in ("write_file", "edit_file", "create_file", "replace_file_content", "multi_replace_file_content", "write_to_file")
            ):
                path = tc.arguments.get("path", tc.arguments.get("TargetFile", ""))
                if not path.endswith("plan.md"):
                    # Check if plan.md exists in workspace
                    plan_path = Path(self._settings.workspace) / "plan.md"
                    if not plan_path.exists():
                        self._display.show_warning(
                            "Planning mode: blocked write without a plan"
                        )
                        result = (
                            "Error: PLANNING MODE ENFORCEMENT - You must create 'plan.md' "
                            "with your implementation plan and ask for user approval FIRST "
                            "before modifying other files."
                        )
                        consecutive_errors += 1
                        
                        self._conversation.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": tc.name,
                            "content": result,
                        })
                        
                        if self._logger:
                            self._logger.log_tool_result(tc.name, result, duration_ms=0)
                            
                        continue

            planning_warning = ""

            start = time.monotonic()
            result = self._tools.execute(tc.name, tc.arguments)
            duration_ms = int((time.monotonic() - start) * 1000)

            if result.startswith("Error"):
                consecutive_errors += 1
            else:
                consecutive_errors = 0

            self._display.show_tool_result(tc.name, result)

            if self._logger:
                self._logger.log_tool_result(tc.name, result, duration_ms=duration_ms)

            final_content = planning_warning + self._truncate_tool_result(tc.name, result)
            if tc == tool_calls[-1] and nudge_notice:
                final_content += nudge_notice

            self._conversation.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tc.name,
                "content": final_content,
            })

            # Track in knowledge graph
            if self._knowledge_graph:
                try:
                    args = tc.arguments
                    # Read tools: read_file, view_file, list_dir, grep_search
                    if tc.name in ("read_file", "view_file", "list_dir", "grep_search"):
                        path = args.get("path", args.get("file_path", args.get("AbsolutePath", "")))
                        if path:
                            self._knowledge_graph.record_file_read(path)
                    # Write tools: write_file, write_to_file, edit_file, create_file, replace_file_content, multi_replace_file_content
                    elif tc.name in ("write_file", "write_to_file", "edit_file", "create_file", "replace_file_content", "multi_replace_file_content"):
                        path = args.get("path", args.get("file_path", args.get("TargetFile", "")))
                        if path:
                            self._knowledge_graph.record_file_write(path)
                except Exception as exc:
                    logger.debug("Knowledge graph tracking failed: %s", exc)

            # Register notable files as sidebar artifacts
            if (
                tc.name in ("write_file", "create_file", "write_to_file")
                and not result.startswith("Error")
            ):
                path = tc.arguments.get("path", tc.arguments.get("file_path", tc.arguments.get("TargetFile", "")))
                if path and hasattr(self._display, "_register_artifact"):
                    self._display._register_artifact(path, tool_name=tc.name)

        return consecutive_errors

    def _execute_parallel(
        self,
        tool_calls: list[ToolCall],
        consecutive_errors: int,
        nudge_notice: str = "",
    ) -> int:
        """Execute read-only tool calls in parallel."""
        # Filter out redundant re-reads in parallel execution
        valid_calls: list[ToolCall] = []
        recent_calls = list(self._recent_tool_calls)
        for tc in tool_calls:
            self._display.show_tool_call(tc.name, tc.arguments)
            if self._logger:
                self._logger.log_tool_call(tc.name, tc.arguments)
            call_sig = f"{tc.name}({json.dumps(tc.arguments, sort_keys=True)})"

            target_p = tc.arguments.get("path", tc.arguments.get("file_path", tc.arguments.get("AbsolutePath", "")))
            norm_target = str(target_p).replace("\\", "/").strip().lower() if target_p else ""

            is_exact_dup = recent_calls.count(call_sig) >= 2
            is_path_loop = False
            if norm_target and tc.name in ("read_file", "view_file"):
                path_read_count = sum(
                    1 for sig in recent_calls
                    if ("read_file" in sig or "view_file" in sig) and norm_target in sig.lower()
                )
                if path_read_count >= 2:
                    is_path_loop = True

            if tc.name in ("read_file", "view_file", "list_dir", "grep_search") and (is_exact_dup or is_path_loop):
                self._recent_tool_calls.append(call_sig)
                self._display.show_info(f"Notice: Redundant parallel re-read skipped for {tc.name} ({target_p or 'same args'})")
                notice_result = (
                    f"Notice: You have ALREADY inspected '{target_p or tc.name}' multiple times in the recent context. "
                    "Do NOT re-read the same file repeatedly. Proceed directly to making necessary code edits or providing your final response."
                )
                self._conversation.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.name,
                    "content": notice_result,
                })
            else:
                self._recent_tool_calls.append(call_sig)
                valid_calls.append(tc)

        if not valid_calls:
            return consecutive_errors

        tool_calls = valid_calls
        self._display.show_info(f"Running {len(tool_calls)} tools in parallel...")

        results: dict[str, tuple[str, int]] = {}  # tc.id -> (result, duration_ms)

        def _run_one(tc: ToolCall) -> tuple[str, str, int]:
            start = time.monotonic()
            result = self._tools.execute(tc.name, tc.arguments)
            duration_ms = int((time.monotonic() - start) * 1000)
            return tc.id, result, duration_ms

        with ThreadPoolExecutor(max_workers=min(len(tool_calls), 8)) as pool:
            futures = {pool.submit(_run_one, tc): tc for tc in tool_calls}
            for future in as_completed(futures):
                tc = futures[future]
                try:
                    tc_id, result, duration_ms = future.result()
                    results[tc_id] = (result, duration_ms)
                except Exception as exc:
                    results[tc.id] = (f"Error: {exc}", 0)

        # Append results in original order (important for conversation coherence)
        for tc in tool_calls:
            result, duration_ms = results.get(tc.id, ("Error: no result", 0))

            if result.startswith("Error"):
                consecutive_errors += 1
            else:
                consecutive_errors = 0

            self._display.show_tool_result(tc.name, result)

            if self._logger:
                self._logger.log_tool_result(tc.name, result, duration_ms=duration_ms)

            content_str = self._truncate_tool_result(tc.name, result)
            if tc == tool_calls[-1] and nudge_notice:
                content_str += nudge_notice

            self._conversation.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tc.name,
                "content": content_str,
            })

            # Register notable files as sidebar artifacts
            if (
                tc.name in ("write_file", "create_file")
                and not result.startswith("Error")
            ):
                path = tc.arguments.get("path", tc.arguments.get("file_path", ""))
                if path and hasattr(self._display, "_register_artifact"):
                    self._display._register_artifact(path, tool_name=tc.name)

        return consecutive_errors

    @staticmethod
    def _truncate_tool_result(
        tool_name: str,
        result: str,
        max_chars: int = _MAX_TOOL_RESULT_CHARS,
    ) -> str:
        """Truncate a tool result for storage in conversation history.

        Large MCP responses, file reads, and command outputs can bloat
        the conversation to 60K+ tokens.  The full result is still shown
        to the user and logged — only the copy stored in the conversation
        list is truncated.

        Args:
            tool_name: Name of the tool (for the truncation notice).
            result: The raw result string.
            max_chars: Maximum characters to keep (default 4000 ≈ ~1000 tokens).

        Returns:
            Original result if under the limit, or a truncated version
            with a notice appended.
        """
        if len(result) <= max_chars:
            return result

        truncated_len = len(result) - max_chars
        return (
            result[:max_chars]
            + f"\n\n[... truncated {truncated_len:,} chars from {tool_name} result]"
        )

    # ------------------------------------------------------------------
    # Interactive question tool
    # ------------------------------------------------------------------

    def get_conversation(self) -> list[dict[str, Any]]:
        """Return a copy of the current conversation history.

        Used by the auto-summary system to generate session summaries.

        Returns:
            Copy of the conversation messages list.
        """
        return list(self._conversation)

    def register_ask_question(self) -> None:
        """Register the ask_question tool for interactive user queries.

        Uses the ``InputProtocol`` when available, otherwise falls back
        to ``prompt_toolkit`` for terminal input.
        """
        input_handler = self._input_handler

        def _ask_question(
            question: str,
            options: list[str] | None = None,
            multi_select: bool = False,
        ) -> str:
            """Ask the user a question, optionally with numbered options."""
            # Route through InputProtocol when available (GUI mode)
            if input_handler is not None:
                if options:
                    return input_handler.ask_question(question, options)
                else:
                    return "Error: You called ask_question without any options. For open-ended questions, DO NOT use this tool. Just output your question as normal conversational text instead!"

            # Fallback: terminal mode via prompt_toolkit
            from prompt_toolkit import prompt as pt_prompt

            self._display.show_info(f"\n  ❓ {question}")

            if options:
                for i, opt in enumerate(options, 1):
                    self._display.show_info(f"    {i}. {opt}")

                if multi_select:
                    self._display.show_info("    (comma-separated numbers for multiple)")

                try:
                    answer = pt_prompt("  Your choice: ").strip()
                    # Parse numbered selection
                    if multi_select and "," in answer:
                        indices = [s.strip() for s in answer.split(",")]
                        selected = []
                        for s in indices:
                            if s.isdigit():
                                idx = int(s) - 1
                                if 0 <= idx < len(options):
                                    selected.append(options[idx])
                        return ", ".join(selected) if selected else answer
                    if answer.isdigit():
                        idx = int(answer) - 1
                        if 0 <= idx < len(options):
                            return options[idx]
                    return answer
                except (EOFError, KeyboardInterrupt):
                    return "(user cancelled)"
            else:
                try:
                    answer = pt_prompt("  Your answer: ").strip()
                    return answer or "(no answer)"
                except (EOFError, KeyboardInterrupt):
                    return "(user cancelled)"

        self._tools.register(
            name="ask_question",
            description=(
                "Ask the user a question to clarify requirements or get a decision. "
                "Optionally provide numbered options for the user to choose from. "
                "CRITICAL: DO NOT use this tool for general conversational questions "
                "(e.g. 'How can I help you?'). For normal conversation, just output text directly!"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to ask the user.",
                    },
                    "options": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of choices for the user.",
                    },
                    "multi_select": {
                        "type": "boolean",
                        "description": "Allow selecting multiple options. Default: false.",
                    },
                },
                "required": ["question"],
            },
            handler=_ask_question,
        )
