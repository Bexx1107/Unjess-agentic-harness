"""GUI application state — shared reactive state for all pages and components.

Holds the message list, thinking indicator, sidebar visibility, and other
UI state.  Components read from this; the agent thread and UI timers
write to it.

The ``dirty`` flag is the primary change-detection mechanism: the agent
thread sets it after mutations, and the periodic UI timer checks it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import threading
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from unjess.config import Settings
    from unjess.memory import MemoryStore
    from unjess.subagents.manager import SubagentManager
    from unjess.task_manager import TaskManager
    from unjess.scheduler import Scheduler
    from unjess.conversation_logger import ConversationLogger


# ---------------------------------------------------------------------------
# Data classes used inside messages
# ---------------------------------------------------------------------------

@dataclass
class TraceStep:
    """A single step in the agent's execution trace.

    Steps are stored in chronological order in ``ChatMessage.trace``
    to enable an AG-style ordered trace view in the UI.

    Attributes:
        step_type: One of ``'thinking'``, ``'tool_call'``, ``'tool_result'``,
            ``'text'``, ``'error'``, ``'info'``.
        name: Tool name (for tool steps), or descriptive label.
        content: Thinking text, tool result, response text, or error message.
        args: Tool arguments dict (for ``'tool_call'`` steps).
        diff: Unified diff string (for file-modifying tool results).
        duration_ms: Execution time in milliseconds.
        collapsed: Whether the UI expansion should render collapsed.
        timestamp: ``time.monotonic()`` when this step was created.
    """

    step_type: str = ""
    name: str = ""
    content: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    diff: str = ""
    duration_ms: int = 0
    collapsed: bool = True
    timestamp: float = 0.0


@dataclass
class ToolCallDisplay:
    """Display data for a single tool invocation inside a message.

    Attributes:
        name: Tool function name.
        args: Tool arguments dict.
        result: Tool result text (populated after execution).
        diff: Unified diff string, if the tool modified a file.
        collapsed: Whether the card should render collapsed.
    """

    name: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    result: str = ""
    diff: str = ""
    collapsed: bool = True


@dataclass
class ChatMessage:
    """A single message in the chat history.

    Attributes:
        role: One of ``'user'``, ``'assistant'``, ``'system'``.
        content: Display text (markdown for assistant, plain for user).
        is_streaming: ``True`` while the assistant message is being
            streamed (shows a typing indicator).
        tool_calls: Inline tool-call display data (legacy flat list).
        trace: Ordered execution trace — chronological list of
            :class:`TraceStep` objects for AG-style rendering.
        stats: Per-turn token/cost stats dict, attached after the
            response completes.
        checkpoint_id: Undo checkpoint ID associated with this user turn.
            Set when the agent creates a checkpoint before processing.
        msg_index: Sequential index of this message in the conversation.
    """

    role: str = "assistant"
    content: str = ""
    thinking: str = ""
    thinking_duration: float = 0.0
    is_streaming: bool = False
    tool_calls: list[ToolCallDisplay] = field(default_factory=list)
    trace: list[TraceStep] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    images: list[dict[str, str]] = field(default_factory=list)
    """Attached images/files. Each dict has keys:
    ``data`` (base64 string), ``mime_type``, ``name``."""
    msg_index: int = -1
    checkpoint_id: str = ""


@dataclass
class Notification:
    """A toast notification queued by the agent thread.

    Attributes:
        level: Severity — ``'info'``, ``'warning'``, or ``'error'``.
        message: Human-readable notification text.
    """

    level: str = "info"
    message: str = ""


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

class AppState:
    """Shared mutable state for the NiceGUI frontend.

    This is **not** thread-safe by itself — the ``dirty`` flag is the
    cheap synchronisation primitive.  The agent thread sets ``dirty = True``
    after writes; the NiceGUI timer checks it and re-renders.

    Args:
        model: Active model name.
        provider: Active provider name.
        workspace: Workspace root path string.
        settings: The global :class:`~unjess.config.Settings` instance.
    """

    # Persistence paths
    _UNJESS_DIR = None  # lazily resolved

    @classmethod
    def _get_data_dir(cls) -> "Path":
        """Return ~/.unjess, creating if needed."""
        if cls._UNJESS_DIR is None:
            from pathlib import Path
            cls._UNJESS_DIR = Path.home() / ".unjess"
            cls._UNJESS_DIR.mkdir(parents=True, exist_ok=True)
        return cls._UNJESS_DIR

    def __init__(
        self,
        model: str = "",
        provider: str = "",
        workspace: str = "",
        settings: "Settings | None" = None,
        memory_store: "MemoryStore | None" = None,
        subagent_manager: "SubagentManager | None" = None,
        task_manager: "TaskManager | None" = None,
        scheduler: "Scheduler | None" = None,
        conv_logger: "ConversationLogger | None" = None,
    ) -> None:
        # Identity
        self.model: str = model
        self.provider: str = provider
        self.workspace: str = workspace
        self.settings: Settings | None = settings  # type: ignore[assignment]

        # Backend references for sidebar panels
        self.memory_store: MemoryStore | None = memory_store  # type: ignore[assignment]
        self.subagent_manager: SubagentManager | None = subagent_manager  # type: ignore[assignment]
        self.task_manager: TaskManager | None = task_manager  # type: ignore[assignment]
        self.scheduler: Scheduler | None = scheduler  # type: ignore[assignment]
        self.conv_logger: ConversationLogger | None = conv_logger  # type: ignore[assignment]  # for transcript persistence

        # Messages — protected by _msg_lock for thread safety
        self.messages: list[ChatMessage] = []
        self._msg_lock = threading.Lock()

        # UI flags
        self.is_thinking: bool = False
        self.dirty: bool = False
        self.sidebar_visible: bool = True

        # Current conversation tracking
        self.current_conversation_id: str = ""       # what the user is VIEWING
        self._agent_conversation_id: str = ""        # what the agent's LLM context has loaded
        self._conversation_renamed: bool = False

        # Session stats (cumulative)
        self.tokens_in: int = 0
        self.tokens_out: int = 0
        self.session_cost: float = 0.0
        self.is_free: bool = False
        self.free_quota_pct: float | None = None

        # Files changed during this session
        self.files_changed: list[dict[str, Any]] = []

        # Notification queue (drained by UI timer)
        self.notifications: list[Notification] = []

        # Artifacts created during this session
        self.artifacts: list[dict[str, Any]] = []

        # Local conversation entries (fallback when memory_store is None)
        self.local_conversations: list[dict[str, Any]] = []

        # Per-conversation caches (survives switching)
        self._message_cache: dict[str, list[ChatMessage]] = {}
        self._files_changed_cache: dict[str, list[dict[str, Any]]] = {}
        self._artifacts_cache: dict[str, list[dict[str, Any]]] = {}

        # Load persisted data from disk
        self._load_conversations()

    # ------------------------------------------------------------------
    # Thread-safe message access
    # ------------------------------------------------------------------

    def append_message(self, msg: ChatMessage) -> None:
        """Thread-safe append to the message list."""
        with self._msg_lock:
            self.messages.append(msg)

    def get_messages_snapshot(self) -> list[ChatMessage]:
        """Return a shallow copy of messages for safe iteration."""
        with self._msg_lock:
            return list(self.messages)

    def clear_messages(self) -> None:
        """Thread-safe clear of the message list."""
        with self._msg_lock:
            self.messages.clear()

    # ------------------------------------------------------------------
    # Conversation persistence
    # ------------------------------------------------------------------

    def _conversations_file(self) -> "Path":
        """Path to the conversations index file."""
        from pathlib import Path
        return self._get_data_dir() / "gui_conversations.json"

    def _messages_dir(self) -> "Path":
        """Directory for per-conversation message files."""
        from pathlib import Path
        d = self._get_data_dir() / "gui_messages"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _load_conversations(self) -> None:
        """Load conversation list and message caches from disk."""
        import json
        import logging
        log = logging.getLogger(__name__)

        cf = self._conversations_file()
        if cf.exists():
            try:
                data = json.loads(cf.read_text(encoding="utf-8"))
                self.local_conversations = data.get("conversations", [])
                log.info("Loaded %d conversations from disk", len(self.local_conversations))
            except Exception as e:
                log.warning("Failed to load conversations: %s", e)

        # Load message caches
        md = self._messages_dir()
        for msg_file in md.glob("*.json"):
            conv_id = msg_file.stem
            try:
                entries = json.loads(msg_file.read_text(encoding="utf-8"))
                msgs = []
                for entry in entries:
                    msgs.append(ChatMessage(
                        role=entry.get("role", "assistant"),
                        content=entry.get("content", ""),
                    ))
                if msgs:
                    self._message_cache[conv_id] = msgs
            except Exception as e:
                log.warning("Failed to load messages for %s: %s", conv_id, e)

    def save_conversations(self) -> None:
        """Persist conversation list to disk."""
        import json
        import logging
        log = logging.getLogger(__name__)

        try:
            cf = self._conversations_file()
            data = {"conversations": self.local_conversations}
            cf.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as e:
            log.warning("Failed to save conversations: %s", e)

    def save_messages(self, conversation_id: str) -> None:
        """Persist messages for a specific conversation to disk."""
        import json
        import logging
        log = logging.getLogger(__name__)

        msgs = self._message_cache.get(conversation_id, [])
        if not msgs:
            # If current conversation matches, save current messages
            if conversation_id == self.current_conversation_id and self.messages:
                msgs = self.messages

        if not msgs:
            return

        try:
            msg_file = self._messages_dir() / f"{conversation_id}.json"
            entries = [
                {"role": m.role, "content": m.content}
                for m in msgs
                if m.content.strip()  # skip empty messages
            ]
            msg_file.write_text(json.dumps(entries, indent=2), encoding="utf-8")
        except Exception as e:
            log.warning("Failed to save messages for %s: %s", conversation_id, e)

    def save_all(self) -> None:
        """Save conversation list and current messages to disk."""
        self.save_conversations()
        if self.current_conversation_id:
            # Update cache with current messages before saving
            if self.messages:
                self._message_cache[self.current_conversation_id] = list(self.messages)
            self.save_messages(self.current_conversation_id)
