"""GUI input handler — bridges InputProtocol to the NiceGUI event loop.

The agent's background thread calls methods like :meth:`get_user_input` and
:meth:`ask_approval`, which **block** using :class:`threading.Event` until
the user interacts with the web UI.  The NiceGUI main thread calls
:meth:`resolve` to unblock the agent.

This two-thread handshake is the core synchronisation primitive between
the synchronous agent loop and the async NiceGUI frontend.
"""

from __future__ import annotations

import threading
from typing import Any


class GUIInput:
    """Blocking input handler for the NiceGUI frontend.

    Implements :class:`~unjess.protocols.InputProtocol`.

    The agent thread calls one of the ``get_*`` / ``ask_*`` methods, which
    store metadata about what the UI should show, then block on an
    internal :class:`threading.Event`.  When the user responds in the
    browser, the UI calls :meth:`resolve` with the answer, setting the
    event and allowing the agent to continue.
    """

    def __init__(self) -> None:
        self._event = threading.Event()
        self._result: Any = None
        self._pending_type: str = ""      # 'input' | 'approval' | 'question' | 'confirmation' | 'choice' | 'spawn'
        self._pending_data: dict[str, Any] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Properties (read by the UI thread)
    # ------------------------------------------------------------------

    @property
    def has_pending(self) -> bool:
        """``True`` when the agent is blocked waiting for user input."""
        return self._pending_type != ""

    @property
    def pending_type(self) -> str:
        """Type of the pending request.

        One of ``'input'``, ``'approval'``, ``'question'``,
        ``'confirmation'``, ``'choice'``, or ``''`` if nothing is pending.
        """
        return self._pending_type

    @property
    def pending_data(self) -> dict[str, Any]:
        """Metadata for the pending request.

        Contents depend on :attr:`pending_type`:

        - ``'input'``: ``{"prompt": str}``
        - ``'approval'``: ``{"tool_name": str, "args": dict}``
        - ``'question'``: ``{"question": str, "options": list[str]}``
        - ``'confirmation'``: ``{"message": str}``
        - ``'choice'``: ``{"title": str, "choices": list[tuple[str, str]]}``
        """
        return self._pending_data

    # ------------------------------------------------------------------
    # InputProtocol methods (called from agent thread — BLOCKING)
    # ------------------------------------------------------------------

    def _wait_for_result(self) -> Any:
        """Block until :meth:`resolve` is called, then return the result.

        Uses a 5-minute timeout to prevent permanent deadlocks if the
        GUI is closed while the agent is waiting for input.
        """
        if not self._event.wait(timeout=300):
            # Timed out — return None so the caller can handle it
            return None
        with self._lock:
            result = self._result
            self._result = None
            self._pending_type = ""
            self._pending_data = {}
        return result

    def get_user_input(self, prompt: str = "❯ ") -> str:
        """Block until the user submits a chat message.

        Args:
            prompt: Prompt string (shown as placeholder in the input box).

        Returns:
            The user's text input.

        Raises:
            EOFError: If the GUI is closed while waiting.
        """
        with self._lock:
            self._event.clear()
            self._pending_type = "input"
            self._pending_data = {"prompt": prompt}
        result = self._wait_for_result()
        if result is None:
            raise EOFError("GUI closed")
        return str(result)

    def ask_approval(self, tool_name: str, args: dict[str, Any]) -> tuple[bool, bool]:
        """Block until the user approves or denies a tool call.

        Args:
            tool_name: The tool requesting approval.
            args: The tool's arguments.

        Returns:
            ``(approved, always)`` — *approved* is ``True`` to proceed,
            *always* is ``True`` to auto-approve similar future calls.
        """
        with self._lock:
            self._event.clear()
            self._pending_type = "approval"
            self._pending_data = {"tool_name": tool_name, "args": args}
        result = self._wait_for_result()
        if result is None:
            return (False, False)
        # result expected as (bool, bool)
        return tuple(result)  # type: ignore[return-value]

    def ask_question(self, question: str, options: list[str]) -> str:
        """Block until the user selects an option.

        Args:
            question: The question text.
            options: Available answer strings.

        Returns:
            The chosen option string.
        """
        with self._lock:
            self._event.clear()
            self._pending_type = "question"
            self._pending_data = {"question": question, "options": options}
        result = self._wait_for_result()
        return str(result) if result is not None else (options[0] if options else "")

    def ask_confirmation(self, message: str) -> bool:
        """Block until the user confirms or cancels.

        Args:
            message: Confirmation prompt text.

        Returns:
            ``True`` if confirmed, ``False`` otherwise.
        """
        with self._lock:
            self._event.clear()
            self._pending_type = "confirmation"
            self._pending_data = {"message": message}
        result = self._wait_for_result()
        return bool(result)

    def ask_choice(self, title: str, choices: list[tuple[str, str]]) -> str:
        """Block until the user picks from a labelled list.

        Args:
            title: Dialog title.
            choices: List of ``(value, display_label)`` tuples.

        Returns:
            The selected *value* string, or ``""`` if cancelled.
        """
        with self._lock:
            self._event.clear()
            self._pending_type = "choice"
            self._pending_data = {"title": title, "choices": choices}
        result = self._wait_for_result()
        return str(result) if result is not None else ""

    def ask_spawn_approval(
        self,
        type_name: str,
        role: str,
        prompt: str,
        suggested_model: str = "",
        tools: list[str] | None = None,
        max_turns: int = 10,
    ) -> dict[str, Any]:
        """Block until the user approves or cancels a subagent spawn.

        Args:
            type_name: Agent type (e.g. 'research', 'coder').
            role: Human-readable role label.
            prompt: The task prompt for the subagent.
            suggested_model: LLM-suggested model (pre-selected in dropdown).
            tools: Suggested tool list.
            max_turns: Suggested max LLM turns.

        Returns:
            Dict with ``'approved'`` (bool), and if approved,
            ``'model'`` (str) and ``'max_turns'`` (int).
        """
        with self._lock:
            self._event.clear()
            self._pending_type = "spawn"
            self._pending_data = {
                "type_name": type_name,
                "role": role,
                "prompt": prompt,
                "suggested_model": suggested_model,
                "tools": tools or [],
                "max_turns": max_turns,
            }
        result = self._wait_for_result()
        if result is None or not isinstance(result, dict):
            return {"approved": False}
        return result

    # ------------------------------------------------------------------
    # Resolution (called from UI / main thread)
    # ------------------------------------------------------------------

    def resolve(self, result: Any) -> None:
        """Unblock the agent thread with the given result.

        Must be called from the NiceGUI main thread when the user
        interacts with the pending input widget.

        Args:
            result: The user's response.  Type depends on
                :attr:`pending_type` — ``str`` for input/question/choice,
                ``(bool, bool)`` for approval, ``bool`` for confirmation.
        """
        with self._lock:
            self._result = result
        self._event.set()

    def cancel(self) -> None:
        """Unblock the agent thread with ``None`` (e.g. on GUI shutdown)."""
        self.resolve(None)
