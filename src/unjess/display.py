"""Display layer — streaming output, diffs, tool call formatting, banners.

Uses ``rich`` for all terminal rendering.
"""

# ruff: noqa: E501

import difflib
from typing import Any, Generator

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.live import Live
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.text import Text
from rich.theme import Theme

from unjess import __version__


# Custom theme
AGENT_THEME = Theme({
    "tool.name": "bold cyan",
    "tool.args": "dim",
    "info": "dim",
    "error": "bold red",
    "stats": "dim green",
    "banner.name": "bold magenta",
    "banner.model": "bold cyan",
    "banner.version": "dim",
})


class TerminalDisplay:
    """Handles all terminal output for the agent.

    Implements :class:`~unjess.protocols.DisplayProtocol` using ``rich``
    for terminal rendering.  Provides streaming text output, tool call
    display, diffs, errors, info messages, and statistics.
    """

    def __init__(self, console: Console | None = None, verbose: bool = False) -> None:
        self._console = console or Console(theme=AGENT_THEME, word_wrap=True)
        self._verbose = verbose

    @property
    def console(self) -> Console:
        """The underlying rich Console."""
        return self._console

    # ----- Streaming text -----

    def stream_text(self, chunks: Generator[str, None, None]) -> str:
        """Print text chunks as they arrive, returning the full text.

        Args:
            chunks: Generator yielding text strings.

        Returns:
            The complete concatenated text.
        """
        full_text: list[str] = []

        for chunk in chunks:
            self._console.print(chunk, end="", highlight=False)
            full_text.append(chunk)

        # Final newline
        self._console.print()
        return "".join(full_text)

    # ----- Thinking display -----

    def show_thinking(self, text: str, max_length: int = 200) -> None:
        """Display LLM thinking/reasoning content.

        Shows a dimmed, abbreviated version of the model's internal reasoning.

        Args:
            text: The thinking text from the model.
            max_length: Max characters to display.
        """
        if not text or not text.strip():
            return
        display_text = text.strip()
        if len(display_text) > max_length:
            display_text = display_text[:max_length] + "..."
        self._console.print(f"  [dim]💭 {display_text}[/dim]")

    def show_thinking_start(self) -> None:
        """Indicate that the model is thinking."""
        self._console.print("  [dim]💭 Thinking...[/dim]", end="")

    def show_thinking_end(self) -> None:
        """End the thinking indicator."""
        self._console.print()  # newline after thinking

    # ----- Progress indicators -----

    def show_progress(self, message: str = "Thinking") -> Live:
        """Start an animated progress indicator.

        Returns a Rich Live context manager. Use with `with`:
            with display.show_progress("Calling LLM"):
                response = llm.call(...)

        Args:
            message: Status message to display.

        Returns:
            A Rich Live instance (use as context manager).
        """
        spinner = Spinner("dots", text=f" [dim]{message}...[/dim]", style="cyan")
        return Live(spinner, console=self._console, transient=True)

    def print_markdown(self, text: str) -> None:
        """Render a complete markdown string."""
        if text.strip():
            self._console.print(Markdown(text))

    # ----- Tool calls -----

    def show_tool_call(self, name: str, args: dict[str, Any]) -> None:
        """Display a tool invocation."""
        # Build a compact argument summary
        arg_parts: list[str] = []
        for k, v in args.items():
            val_str = str(v)
            if len(val_str) > 80:
                val_str = val_str[:77] + "..."
            arg_parts.append(f"{k}={val_str}")

        args_str = ", ".join(arg_parts)
        self._console.print(f"  🔧 [tool.name]{name}[/]([tool.args]{args_str}[/])")

    def show_tool_result(self, name: str, result: str, max_lines: int = 50) -> None:
        """Display a tool execution result, truncated if long."""
        lines = result.splitlines()

        if len(lines) > max_lines:
            truncated = "\n".join(lines[:max_lines])
            truncated += f"\n\n... ({len(lines) - max_lines} more lines truncated)"
        else:
            truncated = result

        if self._verbose:
            self._console.print(Panel(
                truncated,
                title=f"📋 {name} result",
                title_align="left",
                border_style="dim",
                expand=False,
            ))
        else:
            # Compact: just show first few lines
            preview_lines = lines[:5]
            preview = "\n".join(preview_lines)
            if len(lines) > 5:
                preview += f"\n  [dim]... ({len(lines)} total lines)[/dim]"
            first_line = preview.splitlines()[0] if preview else "(empty)"
            self._console.print(f"  [dim]→ {name}: {first_line}[/dim]")

    # ----- Diffs -----

    def show_diff(self, path: str, old_content: str, new_content: str) -> None:
        """Display a syntax-highlighted unified diff with change summary."""
        diff_lines = list(difflib.unified_diff(
            old_content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        ))
        diff_text = "".join(diff_lines)

        if not diff_text.strip():
            self._console.print(f"  [dim]No changes in {path}[/dim]")
            return

        # Count additions and deletions
        additions = sum(1 for line in diff_lines if line.startswith("+") and not line.startswith("+++"))
        deletions = sum(1 for line in diff_lines if line.startswith("-") and not line.startswith("---"))

        syntax = Syntax(diff_text, "diff", theme="monokai", line_numbers=False)
        self._console.print(syntax)

        # Change summary
        self._console.print(
            f"  [dim]{path}: "
            f"[green]+{additions}[/green] / [red]-{deletions}[/red][/dim]"
        )

    def show_file_created(self, path: str, size: int = 0) -> None:
        """Display a file creation indicator."""
        size_str = f" ({size:,} bytes)" if size > 0 else ""
        self._console.print(f"  [bold green]+ Created:[/bold green] {path}{size_str}")

    def show_file_deleted(self, path: str) -> None:
        """Display a file deletion indicator."""
        self._console.print(f"  [bold red]- Deleted:[/bold red] {path}")

    # ----- Messages -----

    def show_error(self, message: str) -> None:
        """Display an error message."""
        self._console.print(f"  [error]✗ {message}[/error]")

    def show_info(self, message: str) -> None:
        """Display an info message."""
        self._console.print(f"  [info]{message}[/info]")

    def show_warning(self, message: str) -> None:
        """Display a warning."""
        self._console.print(f"  [bold yellow]⚠ {message}[/bold yellow]")

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
            quota_pct: Remaining free-tier quota (0.0–1.0), or None if not free.
        """
        parts: list[str] = []

        # Token breakdown — human-readable format
        def _fmt(n: int) -> str:
            return f"{n / 1000:.1f}K" if n >= 1000 else str(n)

        parts.append(f"in: {_fmt(tokens_in)}")
        parts.append(f"out: {_fmt(tokens_out)}")
        if thinking_tokens > 0:
            parts.append(f"think: {_fmt(thinking_tokens)}")

        # Cache info
        if cache_read_tokens > 0 or cache_creation_tokens > 0:
            cache_parts = []
            if cache_read_tokens > 0:
                cache_parts.append(f"hit:{_fmt(cache_read_tokens)}")
            if cache_creation_tokens > 0:
                cache_parts.append(f"write:{_fmt(cache_creation_tokens)}")
            parts.append(f"cache: {'/'.join(cache_parts)}")

        # Cost
        if is_free:
            parts.append("cost: [bold green]FREE[/bold green]")
        elif cost is not None and cost > 0:
            parts.append(f"cost: ${cost:.4f}")

        # Free quota remaining
        if quota_pct is not None:
            pct = int(quota_pct * 100)
            if pct > 50:
                color = "green"
            elif pct > 20:
                color = "yellow"
            else:
                color = "red"
            parts.append(f"quota: [{color}]{pct}% left[/{color}]")

        if model:
            parts.append(f"model: {model}")

        stats_str = " | ".join(parts)
        self._console.print(f"  [stats]{stats_str}[/stats]")

    # ----- Banner -----

    def show_banner(self, model: str, provider: str, workspace: str) -> None:
        """Display the startup banner."""
        title = Text()
        title.append("njss", style="banner.name")
        title.append(f" v{__version__}", style="banner.version")

        info = Text()
        info.append(f"  Model:     {model}\n", style="banner.model")
        info.append(f"  Provider:  {provider}\n", style="dim")
        info.append(f"  Workspace: {workspace}\n", style="dim")
        info.append("  Type /help for commands, /exit to quit", style="dim")

        panel = Panel(
            info,
            title=title,
            title_align="left",
            border_style="magenta",
            expand=False,
            padding=(0, 1),
        )
        self._console.print(panel)
        self._console.print()


# Backwards-compatible alias — existing code imports ``Display``.
Display = TerminalDisplay
