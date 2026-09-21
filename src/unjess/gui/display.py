"""GUI display adapter — bridges DisplayProtocol to reactive AppState.

Instead of printing to a terminal, :class:`GUIDisplay` pushes data into
:class:`~unjess.gui.state.AppState` and fires a callback so the NiceGUI
event loop knows to re-render.

**Threading**: All methods are called from the agent's background thread.
The ``notify_fn`` callback must be safe to call from any thread — typically
it schedules work on the NiceGUI main loop via ``ui.timer`` or
``app.call_later``.
"""

from __future__ import annotations

import difflib
import logging
import re
import time
from contextlib import contextmanager
from typing import Any, Callable, Generator, Iterator

from unjess.gui.state import AppState, ChatMessage, Notification, ToolCallDisplay, TraceStep

logger = logging.getLogger(__name__)

# How many streamed chunks to accumulate before notifying the UI.
_STREAM_NOTIFY_INTERVAL = 4


class GUIDisplay:
    """Display adapter that writes into :class:`AppState`.

    Implements :class:`~unjess.protocols.DisplayProtocol`.

    Args:
        state: The shared application state instance.
        notify_fn: Zero-arg callable invoked (from *any* thread) to tell
            the UI it should refresh.  The implementation must be
            thread-safe — e.g. ``functools.partial(ui.timer, 0, ...)``.
    """

    def __init__(self, state: AppState, notify_fn: Callable[[], None]) -> None:
        self._state = state
        self._notify = notify_fn
        self.console = self._ConsoleShim(self)
        self._thinking_start: float = 0.0

    class _ConsoleShim:
        """Minimal shim so ``self._display.console.print()`` works in GUI mode."""

        def __init__(self, display: "GUIDisplay") -> None:
            self._display = display

        def print(self, *args: Any, **kwargs: Any) -> None:
            """Route console.print() calls to the streaming text display."""
            text = " ".join(str(a) for a in args)
            if text:
                self._display.print_markdown(text)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _track_file_change(
        self, path: str, additions: int, deletions: int
    ) -> None:
        """Track a file change for the sidebar panel."""
        from pathlib import PurePosixPath, PureWindowsPath
        try:
            p = PureWindowsPath(path) if "\\" in path else PurePosixPath(path)
            filename = p.name
            parent = str(p.parent)
        except Exception:
            filename = path.rsplit("/", 1)[-1] if "/" in path else path
            parent = ""

        # Update existing entry or add new one
        for entry in self._state.files_changed:
            if entry.get("full_path") == path:
                entry["additions"] += additions
                entry["deletions"] += deletions
                return
        self._state.files_changed.append({
            "filename": filename,
            "path": parent,
            "full_path": path,
            "additions": additions,
            "deletions": deletions,
        })

    def _register_artifact(self, path: str, tool_name: str = "") -> None:
        """Register a file as a session artifact in the sidebar.

        Args:
            path: Absolute or workspace-relative file path.
            tool_name: The tool that created it (e.g. 'write_file').
        """
        from datetime import datetime
        from pathlib import Path as _Path

        # Resolve relative paths against the workspace root
        p = _Path(path)
        if not p.is_absolute() and self._state.workspace:
            p = _Path(self._state.workspace) / p
        full_path = str(p.resolve())

        filename = p.name
        ext = p.suffix.lower()

        # Deduplicate by full path
        for existing in self._state.artifacts:
            if existing.get("full_path") == full_path:
                existing["updated_at"] = datetime.now()
                return

        self._state.artifacts.append({
            "name": filename,
            "full_path": full_path,
            "ext": ext,
            "tool": tool_name,
            "created_at": datetime.now(),
        })

    def _ensure_assistant_message(self) -> ChatMessage:
        """Return the current assistant message, creating one if needed."""
        with self._state._msg_lock:
            msgs = self._state.messages
            if msgs and msgs[-1].role == "assistant":
                return msgs[-1]
            msg = ChatMessage(role="assistant", content="")
            msgs.append(msg)
        self._state.dirty = True
        return msg

    def _add_notification(self, level: str, message: str) -> None:
        """Append a notification and signal the UI."""
        self._state.notifications.append(Notification(level=level, message=message))
        self._state.dirty = True
        self._notify()

    # ------------------------------------------------------------------
    # DisplayProtocol implementation
    # ------------------------------------------------------------------

    def stream_text(self, chunks: Generator[str, None, None]) -> str:
        """Accumulate streamed tokens into the current assistant message.

        Creates a new :class:`ChatMessage` with ``is_streaming=True``,
        appends each chunk to its ``content``, and periodically fires
        ``_notify()`` so the UI can show partial output.

        Args:
            chunks: Generator yielding text fragments from the LLM.

        Returns:
            The full concatenated response text.
        """
        msg = self._ensure_assistant_message()
        msg.is_streaming = True
        self._state.dirty = True
        self._notify()

        full_text: list[str] = []
        count = 0

        for chunk in chunks:
            full_text.append(chunk)
            msg.content += chunk
            count += 1
            if count % _STREAM_NOTIFY_INTERVAL == 0:
                self._state.dirty = True
                self._notify()

        msg.is_streaming = False
        self._state.dirty = True
        self._notify()
        return "".join(full_text)

    def show_thinking(self, text: str, max_length: int = 0) -> None:
        """Store LLM thinking/reasoning content in the message's thinking field.

        Also appends a ``TraceStep`` for the AG-style trace view.

        Args:
            text: The thinking text from the model.
            max_length: Ignored (kept for API compat).
        """
        msg = self._ensure_assistant_message()
        msg.thinking = text

        # Trace: add or update thinking step
        duration = 0.0
        if self._thinking_start:
            duration = time.monotonic() - self._thinking_start
        msg.trace.append(TraceStep(
            step_type="thinking",
            name="Reasoning",
            content=text,
            duration_ms=int(duration * 1000),
            collapsed=True,
            timestamp=time.monotonic(),
        ))

        self._state.dirty = True
        self._notify()

    def show_thinking_start(self) -> None:
        """Indicate that the model is reasoning and start timing."""
        self._thinking_start = time.monotonic()
        self._state.is_thinking = True
        self._state.dirty = True
        self._notify()

    def show_thinking_end(self) -> None:
        """End the thinking indicator and record duration."""
        self._state.is_thinking = False
        if self._thinking_start:
            duration = time.monotonic() - self._thinking_start
            msg = self._ensure_assistant_message()
            msg.thinking_duration = duration
            self._thinking_start = 0.0
        self._state.dirty = True
        self._notify()

    def show_progress(self, message: str = "Thinking") -> Any:
        """Return a no-op context manager.

        The GUI renders its own activity indicators based on
        ``state.is_thinking``, so the terminal-style spinner is
        unnecessary.

        Args:
            message: Status text (unused in GUI mode).

        Returns:
            A context manager that does nothing.
        """
        @contextmanager
        def _noop() -> Iterator[None]:
            yield

        return _noop()

    # Regex to strip <think>...</think> blocks that some models (Gemma, QwQ,
    # DeepSeek R1) embed inline in their text output.
    _THINK_TAG_RE = re.compile(
        r"<think>\s*(.*?)\s*</think>",
        re.DOTALL | re.IGNORECASE,
    )

    def print_markdown(self, text: str) -> None:
        """Append rendered markdown to the current assistant message.

        If the text contains ``<think>...</think>`` tags (from models that
        embed reasoning inline), the thinking content is extracted into
        ``msg.thinking`` and removed from the visible text.

        Args:
            text: Markdown string.
        """
        msg = self._ensure_assistant_message()

        # Strip inline <think> blocks → move to msg.thinking
        think_matches = self._THINK_TAG_RE.findall(text)
        if think_matches:
            for match in think_matches:
                if match.strip():
                    msg.thinking += ("\n" if msg.thinking else "") + match.strip()
            text = self._THINK_TAG_RE.sub("", text)

        msg.content += text
        self._state.dirty = True
        self._notify()

    def show_tool_call(self, name: str, args: dict[str, Any]) -> None:
        """Append a tool-call indicator to the current assistant message.

        Also appends a ``TraceStep`` for the AG-style trace view.

        Args:
            name: Tool name.
            args: Tool arguments.
        """
        msg = self._ensure_assistant_message()
        msg.tool_calls.append(ToolCallDisplay(name=name, args=args))

        # Trace: tool_call step (result will be paired later)
        msg.trace.append(TraceStep(
            step_type="tool_call",
            name=name,
            args=dict(args),  # defensive copy
            timestamp=time.monotonic(),
        ))

        self._state.dirty = True
        self._notify()

    def show_tool_result(self, name: str, result: str, max_lines: int = 50) -> None:
        """Update the most recent tool call with its result.

        Also appends a ``TraceStep`` for the AG-style trace view and
        computes execution duration from the matching ``tool_call`` step.

        Args:
            name: Tool name (used for validation).
            result: Result text.
            max_lines: Max lines to keep (excess is truncated).
        """
        msg = self._ensure_assistant_message()
        if not msg.tool_calls:
            logger.warning("show_tool_result(%s) called with no pending tool calls", name)
            return

        # Truncate long results
        lines = result.splitlines(keepends=True)
        if len(lines) > max_lines:
            truncated = "".join(lines[:max_lines])
            truncated += f"\n… ({len(lines) - max_lines} more lines)\n"
        else:
            truncated = result

        # Find the matching tool call (search from end, find first without result)
        tc = None
        for candidate in reversed(msg.tool_calls):
            if candidate.name == name and not candidate.result:
                tc = candidate
                break
        if tc is None:
            # Fallback: last tool call without a result
            for candidate in reversed(msg.tool_calls):
                if not candidate.result:
                    tc = candidate
                    break
        if tc is None:
            tc = msg.tool_calls[-1]

        tc.result = truncated

        # Trace: tool_result step — compute duration from matching tool_call
        now = time.monotonic()
        duration_ms = 0
        for step in reversed(msg.trace):
            if step.step_type == "tool_call" and step.name == name:
                duration_ms = int((now - step.timestamp) * 1000)
                break
        msg.trace.append(TraceStep(
            step_type="tool_result",
            name=name,
            content=truncated,
            duration_ms=duration_ms,
            collapsed=True,
            timestamp=now,
        ))

        self._state.dirty = True
        self._notify()

    def show_diff(self, path: str, old_content: str, new_content: str) -> None:
        """Generate a unified diff and attach it to the latest tool call.

        Also updates the most recent ``tool_result`` trace step with the diff.

        Args:
            path: File path that was modified.
            old_content: Original file content.
            new_content: Updated file content.
        """
        diff_lines = list(difflib.unified_diff(
            old_content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            lineterm="",
        ))
        diff_str = "\n".join(diff_lines) if diff_lines else "(no changes)"

        msg = self._ensure_assistant_message()
        if msg.tool_calls:
            msg.tool_calls[-1].diff = diff_str
        else:
            # No tool call context — append as markdown
            msg.content += f"\n```diff\n{diff_str}\n```\n"

        # Trace: attach diff to the most recent tool_result step
        for step in reversed(msg.trace):
            if step.step_type == "tool_result":
                step.diff = diff_str
                break

        self._state.dirty = True
        self._notify()

        # Track file change for sidebar
        additions = sum(1 for l in diff_lines if l.startswith('+') and not l.startswith('+++'))
        deletions = sum(1 for l in diff_lines if l.startswith('-') and not l.startswith('---'))
        self._track_file_change(path, additions, deletions)

    def show_file_created(self, path: str, size: int = 0) -> None:
        """Append a file-creation notice to the current message.

        Args:
            path: Created file path.
            size: File size in bytes.
        """
        size_str = f" ({size:,} bytes)" if size else ""
        msg = self._ensure_assistant_message()
        msg.content += f"\n📄 **Created** `{path}`{size_str}\n"
        self._state.dirty = True
        self._notify()

        # Track file change for sidebar
        self._track_file_change(path, 1, 0)

    def show_file_deleted(self, path: str) -> None:
        """Append a file-deletion notice to the current message.

        Args:
            path: Deleted file path.
        """
        msg = self._ensure_assistant_message()
        msg.content += f"\n🗑️ **Deleted** `{path}`\n"
        self._state.dirty = True
        self._notify()

        # Track file change for sidebar
        self._track_file_change(path, 0, 1)

    # ----- Notifications -----

    def show_error(self, message: str) -> None:
        """Display an error notification.

        Args:
            message: Error text.
        """
        msg = self._ensure_assistant_message()
        msg.trace.append(TraceStep(
            step_type="error",
            content=message,
            timestamp=time.monotonic(),
        ))
        # Errors: both inline AND toast so they're impossible to miss
        self.print_markdown(f"\n\n> ❌ **Error:** {message}\n\n")
        self._add_notification("negative", message)

    def show_info(self, message: str) -> None:
        """Display an informational message in the execution trace.

        Args:
            message: Info text.
        """
        msg = self._ensure_assistant_message()
        msg.trace.append(TraceStep(
            step_type="info",
            content=message,
            timestamp=time.monotonic(),
        ))
        self._state.dirty = True
        self._notify()

    def show_warning(self, message: str) -> None:
        """Display a warning message in the execution trace and toast notification.

        Args:
            message: Warning text.
        """
        msg = self._ensure_assistant_message()
        msg.trace.append(TraceStep(
            step_type="info",
            name="warning",
            content=message,
            timestamp=time.monotonic(),
        ))
        self._add_notification("warning", message)

    # ----- Statistics -----

    def show_stats(
        self,
        tokens_in: int,
        tokens_out: int,
        cost: float | None = None,
        model: str = "",
        is_free: bool = False,
        thinking_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_creation_tokens: int = 0,
        quota_pct: float | None = None,
    ) -> None:
        """Update cumulative session statistics in state.

        Also attaches a per-turn stats snapshot to the current assistant
        message so the UI can render inline token counts.

        Args:
            tokens_in: Input tokens for this turn.
            tokens_out: Output tokens for this turn.
            cost: Estimated cost in USD for this turn.
            model: Model name.
            is_free: Whether this was a free-tier response.
            thinking_tokens: Reasoning tokens used.
            cache_read_tokens: Tokens read from cache.
            cache_creation_tokens: Tokens written to cache.
            quota_pct: Remaining free-tier quota (0.0–1.0).
        """
        state = self._state
        state.tokens_in += tokens_in
        state.tokens_out += tokens_out
        if cost is not None:
            state.session_cost += cost
        state.is_free = is_free
        state.free_quota_pct = quota_pct
        if model:
            state.model = model

        # Attach per-turn snapshot to the current message
        msg = self._ensure_assistant_message()
        msg.stats = {
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost": cost,
            "model": model,
            "is_free": is_free,
            "thinking_tokens": thinking_tokens,
            "cache_read_tokens": cache_read_tokens,
            "cache_creation_tokens": cache_creation_tokens,
            "quota_pct": quota_pct,
        }

        self._state.dirty = True
        self._notify()

    # ----- Banner (no-op) -----

    def show_banner(self, model: str, provider: str, workspace: str) -> None:
        """No-op — the GUI header replaces the terminal banner.

        State fields are set directly by :func:`gui.app.launch`.

        Args:
            model: Model name (stored but not rendered as a banner).
            provider: Provider name.
            workspace: Workspace path.
        """
        self._state.model = model
        self._state.provider = provider
        self._state.workspace = workspace
