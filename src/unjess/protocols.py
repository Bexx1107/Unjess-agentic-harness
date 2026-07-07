"""UI protocol definitions — abstract interfaces for display and input.

Any UI backend (terminal, web GUI, native app) implements these protocols.
The agent and all components interact only through these interfaces,
making the core engine UI-agnostic.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generator, Iterator, Protocol, runtime_checkable


@runtime_checkable
class DisplayProtocol(Protocol):
    """Abstract display interface — all agent output goes through this.

    Maps 1:1 with the rendering methods needed by the agent loop,
    tool execution, and slash commands.
    """

    # ----- Streaming text -----

    def stream_text(self, chunks: Generator[str, None, None]) -> str:
        """Print text chunks as they arrive, returning the full text.

        Args:
            chunks: Generator yielding text strings.

        Returns:
            The complete concatenated text.
        """
        ...

    # ----- Thinking display -----

    def show_thinking(self, text: str, max_length: int = 200) -> None:
        """Display LLM thinking/reasoning content.

        Args:
            text: The thinking text from the model.
            max_length: Max characters to display.
        """
        ...

    def show_thinking_start(self) -> None:
        """Indicate that the model is thinking."""
        ...

    def show_thinking_end(self) -> None:
        """End the thinking indicator."""
        ...

    # ----- Progress indicators -----

    def show_progress(self, message: str = "Thinking") -> Any:
        """Start an animated progress indicator.

        Returns a context manager. Use with ``with``:

            with display.show_progress("Calling LLM"):
                response = llm.call(...)

        Args:
            message: Status message to display.

        Returns:
            A context manager for the progress indicator.
        """
        ...

    # ----- Markdown -----

    def print_markdown(self, text: str) -> None:
        """Render a complete markdown string."""
        ...

    # ----- Tool calls -----

    def show_tool_call(self, name: str, args: dict[str, Any]) -> None:
        """Display a tool invocation.

        Args:
            name: Tool name.
            args: Tool arguments.
        """
        ...

    def show_tool_result(self, name: str, result: str, max_lines: int = 50) -> None:
        """Display a tool execution result, truncated if long.

        Args:
            name: Tool name.
            result: Result string.
            max_lines: Max lines to display.
        """
        ...

    # ----- Diffs & files -----

    def show_diff(self, path: str, old_content: str, new_content: str) -> None:
        """Display a syntax-highlighted unified diff with change summary.

        Args:
            path: File path.
            old_content: Original content.
            new_content: New content.
        """
        ...

    def show_file_created(self, path: str, size: int = 0) -> None:
        """Display a file creation indicator.

        Args:
            path: File path.
            size: File size in bytes.
        """
        ...

    def show_file_deleted(self, path: str) -> None:
        """Display a file deletion indicator.

        Args:
            path: File path.
        """
        ...

    # ----- Messages -----

    def show_error(self, message: str) -> None:
        """Display an error message."""
        ...

    def show_info(self, message: str) -> None:
        """Display an info message."""
        ...

    def show_warning(self, message: str) -> None:
        """Display a warning."""
        ...

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
        """Display token usage and cost after a response.

        Args:
            tokens_in: Input/prompt tokens.
            tokens_out: Output/completion tokens.
            cost: Estimated cost in USD.
            model: Model name.
            is_free: Whether this was a free model.
            thinking_tokens: Reasoning/thinking tokens used.
            cache_read_tokens: Tokens read from cache.
            cache_creation_tokens: Tokens written to cache.
            quota_pct: Remaining free-tier quota (0.0-1.0), or None if not free.
        """
        ...

    # ----- Banner -----

    def show_banner(self, model: str, provider: str, workspace: str) -> None:
        """Display the startup banner.

        Args:
            model: Current model name.
            provider: Current provider name.
            workspace: Current workspace path.
        """
        ...


@runtime_checkable
class InputProtocol(Protocol):
    """Abstract input interface — all user input goes through this.

    Methods are synchronous and blocking. For GUI backends, implementations
    use threading primitives (e.g. ``threading.Event``) to block until the
    user responds in the UI.  This keeps the agent loop synchronous while
    allowing any frontend to provide input.
    """

    def get_user_input(self, prompt: str = "❯ ") -> str:
        """Get a line of input from the user.

        Args:
            prompt: The prompt string to display.

        Returns:
            The user's input string.

        Raises:
            EOFError: If input is exhausted.
            KeyboardInterrupt: If the user cancels.
        """
        ...

    def ask_approval(self, tool_name: str, args: dict[str, Any]) -> tuple[bool, bool]:
        """Prompt the user to approve a tool call.

        Args:
            tool_name: The tool being called.
            args: The tool arguments.

        Returns:
            A tuple of (approved, always). ``approved`` is True if the
            user approved, ``always`` is True if the user wants to
            auto-approve similar future calls.
        """
        ...

    def ask_question(self, question: str, options: list[str]) -> str:
        """Ask the user a question with predefined options.

        Args:
            question: The question text.
            options: List of option strings.

        Returns:
            The selected option string.
        """
        ...

    def ask_confirmation(self, message: str) -> bool:
        """Ask the user a yes/no confirmation.

        Args:
            message: The confirmation message.

        Returns:
            True if confirmed, False otherwise.
        """
        ...

    def ask_choice(self, title: str, choices: list[tuple[str, str]]) -> str:
        """Ask the user to pick from a list of labelled choices.

        Args:
            title: Dialog/prompt title.
            choices: List of (value, display_label) tuples.

        Returns:
            The selected value string, or empty string if cancelled.
        """
        ...
