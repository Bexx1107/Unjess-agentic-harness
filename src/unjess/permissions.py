"""Permission system — approval gate for tool execution."""

from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

if TYPE_CHECKING:
    from unjess.protocols import InputProtocol


class _FallbackTerminalInput:
    """Minimal terminal input used when no InputProtocol is provided.

    Keeps backwards compatibility — if callers construct a
    ``PermissionManager`` without an ``input_handler`` the old
    ``input()``-based flow still works.
    """

    def __init__(self, console: Console) -> None:
        self._console = console

    def ask_approval(self, tool_name: str, args: dict[str, Any]) -> tuple[bool, bool]:
        """Prompt the user via terminal for approval."""
        # Build preview
        if tool_name == "run_command":
            command = args.get("command", "<unknown>")
            cwd = args.get("cwd", ".")
            preview_text = Text()
            preview_text.append("Command: ", style="bold")
            preview_text.append(command, style="bold yellow")
            preview_text.append(f"\n     cwd: {cwd}", style="dim")
        else:
            preview_text = Text(f"{tool_name}({args})")

        panel = Panel(
            preview_text,
            title="⚠️  Permission Required",
            title_align="left",
            border_style="yellow",
        )
        self._console.print(panel)

        while True:
            try:
                self._console.print(
                    "[bold]Allow?[/bold] [dim](y)es / (n)o / (a)lways[/dim]: ",
                    end="",
                )
                response = input().strip().lower()
            except (EOFError, KeyboardInterrupt):
                self._console.print()
                return (False, False)

            if response in ("y", "yes"):
                return (True, False)
            if response in ("n", "no"):
                return (False, False)
            if response in ("a", "always"):
                return (True, True)

            self._console.print("  [dim]Please enter y, n, or a.[/dim]")


class PermissionManager:
    """Manages tool execution permissions.

    Rules:
        - read_file, list_dir, grep_search → always auto-approve
        - write_file, edit_file → auto-approve within workspace
        - run_command → always ask (unless user chose "Always" for that command)

    Args:
        console: Rich console (used only as fallback if no *input_handler*).
        input_handler: An :class:`~unjess.protocols.InputProtocol` instance
            for prompting the user.  When ``None`` a built-in terminal
            fallback is used.
    """

    # Tools that never need approval
    _ALWAYS_APPROVE = {"read_file", "list_dir", "grep_search"}

    # Tools that are auto-approved within the workspace
    _WORKSPACE_APPROVE = {"write_file", "edit_file"}

    def __init__(
        self,
        console: Console | None = None,
        input_handler: Optional[InputProtocol] = None,
    ) -> None:
        self._console = console or Console()
        self._input_handler: _FallbackTerminalInput | InputProtocol = (
            input_handler if input_handler is not None
            else _FallbackTerminalInput(self._console)
        )
        self._auto_approved_commands: set[str] = set()
        self._always_approve_all_commands: bool = False

    def check(self, tool_name: str, args: dict[str, Any]) -> bool:
        """Check whether a tool call is pre-approved.

        Returns True if the action is auto-approved, False if it needs
        explicit user approval.
        """
        if tool_name in self._ALWAYS_APPROVE:
            return True

        if tool_name in self._WORKSPACE_APPROVE:
            return True  # workspace sandboxing is handled by the tools themselves

        if tool_name == "run_command":
            if self._always_approve_all_commands:
                return True
            command = args.get("command", "")
            # Check if the command prefix was previously "Always"-approved
            for approved in self._auto_approved_commands:
                if command.startswith(approved):
                    return True
            return False

        # Unknown tools default to requiring approval
        return False

    def request_approval(self, tool_name: str, args: dict[str, Any]) -> bool:
        """Prompt the user to approve a tool call.

        Delegates to the configured ``InputProtocol``.

        Returns True if approved, False if denied.
        """
        approved, always = self._input_handler.ask_approval(tool_name, args)

        if always and tool_name == "run_command":
            command = args.get("command", "")
            prefix = command.split()[0] if command.split() else ""
            if prefix:
                self._auto_approved_commands.add(prefix)
                if hasattr(self._console, "print"):
                    self._console.print(
                        f"  [dim]Auto-approving future '{prefix}' commands.[/dim]"
                    )

        return approved

    def approve_all_commands(self) -> None:
        """Disable all command approval prompts (dangerous — for automation)."""
        self._always_approve_all_commands = True

    @property
    def auto_approved(self) -> set[str]:
        """Set of auto-approved command prefixes."""
        return self._auto_approved_commands.copy()

