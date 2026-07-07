"""Silent display for subagents — captures output instead of showing it.

Subagents run in the background and shouldn't pollute the parent's chat.
This display captures all output into a buffer and sends the final result
back to the parent via the message bus.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any, Iterator

logger = logging.getLogger(__name__)


class SubagentDisplay:
    """Display implementation for child agents.

    Captures all output (text, tool calls, errors) into a buffer.
    The parent agent reads the result when the subagent completes.

    Args:
        agent_id: The conversation ID of this subagent.
    """

    def __init__(self, agent_id: str = "") -> None:
        self._agent_id = agent_id
        self._buffer: list[str] = []
        self._tool_calls: list[dict[str, Any]] = []
        self.console = self  # Commands access display.console.print(...)

    def print(self, *args: Any, **kwargs: Any) -> None:
        """Capture console.print() output into the buffer."""
        for arg in args:
            text = str(arg).strip()
            if text:
                self._buffer.append(text)

    @property
    def captured_output(self) -> str:
        """All captured output as a single string."""
        return "\n".join(self._buffer)

    # --- Text output ---

    def stream_response(self, chunks: Any) -> str:
        """Collect streamed chunks into a single response.

        Args:
            chunks: Iterable of text chunks from the LLM.

        Returns:
            The full response text.
        """
        full_text: list[str] = []
        for chunk in chunks:
            full_text.append(chunk)
        result = "".join(full_text)
        if result.strip():
            self._buffer.append(result)
        return result

    def show_thinking(self, text: str, max_length: int = 200) -> None:
        """Capture thinking text (silently)."""
        pass  # Don't clutter subagent output with thinking

    def show_thinking_start(self) -> None:
        """No-op for subagents."""
        pass

    def show_thinking_end(self) -> None:
        """No-op for subagents."""
        pass

    # --- Tool calls ---

    def show_tool_call(self, name: str, args: dict[str, Any]) -> None:
        """Record a tool call."""
        self._tool_calls.append({"name": name, "args": args})

    def show_tool_result(self, name: str, result: str, max_lines: int = 50) -> None:
        """Record a tool result (silently — don't buffer huge results)."""
        pass  # Tool results go into conversation, not display

    def show_diff(self, path: str, old_content: str, new_content: str) -> None:
        """Record a file diff."""
        self._buffer.append(f"Modified: {path}")

    # --- Messages ---

    def show_info(self, message: str) -> None:
        """Capture an info message."""
        logger.debug("[subagent %s] info: %s", self._agent_id[:8], message)

    def show_warning(self, message: str) -> None:
        """Capture a warning."""
        logger.debug("[subagent %s] warning: %s", self._agent_id[:8], message)
        self._buffer.append(f"⚠️ {message}")

    def show_error(self, message: str) -> None:
        """Capture an error."""
        logger.debug("[subagent %s] error: %s", self._agent_id[:8], message)
        self._buffer.append(f"❌ {message}")

    def show_stats(self, *args: Any, **kwargs: Any) -> None:
        """No-op — subagents don't show token stats."""
        pass

    def show_progress(self, message: str = "Thinking") -> Any:
        """Return a no-op context manager."""
        @contextlib.contextmanager
        def _noop() -> Iterator[None]:
            yield
        return _noop()

    def show_user_input(self, text: str) -> None:
        """No-op for subagents."""
        pass

    def show_welcome(self, *args: Any, **kwargs: Any) -> None:
        """No-op."""
        pass

    def show_cost_summary(self, *args: Any, **kwargs: Any) -> None:
        """No-op."""
        pass
