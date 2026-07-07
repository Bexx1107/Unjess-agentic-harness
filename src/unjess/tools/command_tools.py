"""Command tools — run shell commands with approval gate."""

import logging

import subprocess
import threading
import time
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from unjess.sandbox import Sandbox, SandboxResult
from unjess.tools import ToolRegistry

if TYPE_CHECKING:
    from unjess.permissions import PermissionManager


logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 120
_MAX_OUTPUT_CHARS = 20_000


def _run_command(
    workspace: Path,
    permission_manager: "PermissionManager",
    sandbox: Optional[Sandbox],
    command: str,
    cwd: Optional[str] = None,
    timeout: int = _DEFAULT_TIMEOUT,
) -> str:
    """Run a shell command and return its output.

    Uses a two-phase approach:
    1. Wait up to ``timeout`` seconds for the process to finish.
    2. If still running (e.g. a server), capture whatever output exists,
       leave the process running, and return partial output.

    Args:
        command: The command string to execute.
        cwd: Working directory (relative to workspace). Default: workspace root.
        timeout: Timeout in seconds. Default: 120.

    Returns:
        Combined stdout + stderr output, or an error message.
    """
    # Resolve working directory
    if cwd:
        work_dir = (workspace / cwd).resolve()
        if not str(work_dir).startswith(str(workspace.resolve())):
            return f"Error: Working directory '{cwd}' is outside the workspace."
        if not work_dir.exists():
            return f"Error: Working directory not found: {cwd}"
    else:
        work_dir = workspace

    # Sandbox safety check (runs before permission prompt)
    if sandbox is not None:
        sb_result: SandboxResult = sandbox.check_command(command)
        if not sb_result.allowed:
            return sb_result.reason
        if sb_result.warning:
            logger.warning(sb_result.warning)

    # Check permissions
    if not permission_manager.check("run_command", {"command": command}):
        approved = permission_manager.request_approval("run_command", {"command": command})
        if not approved:
            return "Command execution denied by user."

    # --- Run with Popen for non-blocking control ---
    import os as _os
    env = _os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"  # Force unbuffered Python output
    env["PYTHONIOENCODING"] = "utf-8"  # Prevent charmap codec errors on Windows

    try:
        proc = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(work_dir),
            env=env,
        )
    except Exception as exc:
        return f"Error starting command: {exc}"

    # Collect output in background threads to avoid deadlocks
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []

    def _read_stream(stream: object, target: list[str]) -> None:
        """Read a stream line-by-line into a list."""
        try:
            for line in stream:  # type: ignore[union-attr]
                target.append(line)
        except Exception as exc:
            logger.debug("Stream read error: %s", exc)

    stdout_thread = threading.Thread(
        target=_read_stream, args=(proc.stdout, stdout_chunks), daemon=True
    )
    stderr_thread = threading.Thread(
        target=_read_stream, args=(proc.stderr, stderr_chunks), daemon=True
    )
    stdout_thread.start()
    stderr_thread.start()

    # Poll for completion — return early if process looks long-running
    _EARLY_RETURN = 10  # seconds before we assume it's a server/daemon
    elapsed = 0.0
    while elapsed < timeout:
        if proc.poll() is not None:
            break  # Process finished
        time.sleep(1.0)
        elapsed += 1.0

        # After EARLY_RETURN seconds, if still running, bail out
        # so the agent isn't stuck waiting
        if elapsed >= _EARLY_RETURN:
            break

    # Give reader threads a moment to flush remaining output
    stdout_thread.join(timeout=1)
    stderr_thread.join(timeout=1)

    still_running = proc.poll() is None

    # Build output
    output_parts: list[str] = []

    stdout_text = "".join(stdout_chunks)
    stderr_text = "".join(stderr_chunks)

    if stdout_text:
        output_parts.append(stdout_text)
    if stderr_text:
        if output_parts:
            output_parts.append("\n--- stderr ---\n")
        output_parts.append(stderr_text)

    output = "".join(output_parts).strip()

    if still_running:
        # Process is still going (likely a server or long-running task)
        hint = (
            f"\n\n[Process still running after {int(elapsed)}s — "
            f"showing output so far. PID: {proc.pid}]"
        )
        if output:
            output = output + hint
        else:
            output = f"(process started but produced no output yet){hint}"
    else:
        if not output:
            output = "(no output)"
        if proc.returncode != 0:
            output = f"[Exit code: {proc.returncode}]\n{output}"

    # Truncate if too long
    if len(output) > _MAX_OUTPUT_CHARS:
        half = _MAX_OUTPUT_CHARS // 2
        output = (
            output[:half]
            + f"\n\n... ({len(output) - _MAX_OUTPUT_CHARS} chars truncated) ...\n\n"
            + output[-half:]
        )

    # Scrub secrets from output before returning to LLM context
    if sandbox is not None:
        output = sandbox.scrub_secrets(output)

    return output


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_command_tools(
    registry: ToolRegistry,
    workspace: Path,
    permission_manager: "PermissionManager",
    sandbox: Optional[Sandbox] = None,
    timeout: int = _DEFAULT_TIMEOUT,
) -> None:
    """Register command execution tools on the given registry.

    Uses ``get_current_workspace()`` so commands always run in the
    active workspace, even after conversation switches.

    Args:
        registry: Tool registry to register command tools on.
        workspace: Initial workspace path (stored via file_tools ref).
        permission_manager: Gate for user approval of commands.
        sandbox: Optional Sandbox for blocklist checks and secret scrubbing.
        timeout: Default command timeout in seconds (from Settings).
    """
    from unjess.tools.file_tools import get_current_workspace

    registry.register(
        name="run_command",
        description=(
            "Run a shell command and return its output. "
            "The command runs in the workspace directory by default. "
            "Requires user approval unless auto-approved."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command to execute."},
                "cwd": {
                    "type": "string",
                    "description": "Working directory relative to the workspace. Default: workspace root.",
                },
            },
            "required": ["command"],
        },
        handler=lambda **kw: _run_command(get_current_workspace(), permission_manager, sandbox, timeout=timeout, **kw),
    )
