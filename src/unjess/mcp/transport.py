"""MCP transport — stdio-based communication with MCP servers."""

import logging
import subprocess
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class StdioTransport:
    """Stdio transport for MCP — communicates via stdin/stdout of a subprocess.

    The MCP server runs as a child process. Messages are sent via stdin
    and responses read from stdout. Each message is newline-delimited JSON.

    Args:
        command: The command to run the MCP server.
        args: Additional arguments for the server command.
        cwd: Working directory for the server process.
        env: Optional environment variables.
    """

    def __init__(
        self,
        command: str,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self._command = command
        self._args = args or []
        self._cwd = cwd
        self._env = env
        self._process: Optional[subprocess.Popen[str]] = None
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        """Whether the transport process is alive."""
        return self._process is not None and self._process.poll() is None

    def start(self) -> None:
        """Start the MCP server subprocess."""
        cmd = [self._command] + self._args
        logger.info("Starting MCP server: %s", " ".join(cmd))

        try:
            # Merge custom env with system env — Popen(env=) replaces
            # the entire environment, which would lose PATH etc.
            merged_env = None
            if self._env:
                import os
                merged_env = {**os.environ, **self._env}

            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self._cwd,
                env=merged_env,
                bufsize=1,  # line-buffered
            )
        except FileNotFoundError:
            raise RuntimeError(f"MCP server command not found: {self._command}")
        except Exception as exc:
            raise RuntimeError(f"Failed to start MCP server: {exc}")

    def send(self, message: str) -> str:
        """Send a message and wait for a response.

        Args:
            message: JSON-RPC message string.

        Returns:
            The response JSON string.
        """
        if not self.is_running:
            raise RuntimeError("MCP transport is not running")

        assert self._process is not None
        assert self._process.stdin is not None
        assert self._process.stdout is not None

        with self._lock:
            try:
                # Write message + newline
                self._process.stdin.write(message + "\n")
                self._process.stdin.flush()

                # Read response line
                response = self._process.stdout.readline()
                if not response:
                    raise RuntimeError("MCP server closed the connection")

                return response.strip()
            except BrokenPipeError:
                raise RuntimeError("MCP server process died unexpectedly")

    def send_notification(self, message: str) -> None:
        """Send a notification (no response expected).

        Args:
            message: JSON-RPC notification string.
        """
        if not self.is_running:
            return

        assert self._process is not None
        assert self._process.stdin is not None

        with self._lock:
            try:
                self._process.stdin.write(message + "\n")
                self._process.stdin.flush()
            except BrokenPipeError:
                logger.warning("Failed to send notification — server pipe broken")

    def close(self) -> None:
        """Shut down the MCP server process."""
        if self._process is not None:
            try:
                if self._process.stdin:
                    self._process.stdin.close()
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception:
                self._process.kill()
            finally:
                self._process = None
                logger.info("MCP server stopped")

    def get_stderr(self) -> str:
        """Read any stderr output from the server (for diagnostics).

        Uses non-blocking reads to avoid hanging when the process is
        still running.
        """
        if self._process and self._process.stderr:
            try:
                import os
                if os.name == 'nt':
                    # Windows: poll() checks if process has exited;
                    # only read stderr if process is done (safe, won't block)
                    if self._process.poll() is not None:
                        return self._process.stderr.read()
                    return ''
                else:
                    # Unix: use select to check if data is available
                    import select
                    ready, _, _ = select.select([self._process.stderr], [], [], 0.0)
                    if ready:
                        return self._process.stderr.readline()
                    return ''
            except Exception:
                return ''
        return ""

    def __del__(self) -> None:
        """Ensure process cleanup."""
        self.close()
