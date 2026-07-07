"""Lint feedback loop — run linters after edits and feed errors back to the agent."""

import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LintError:
    """A single lint error."""

    file: str
    line: int
    column: int = 0
    code: str = ""  # e.g. "E501", "W0611"
    message: str = ""
    severity: str = "warning"  # "error", "warning", "info"

    def display(self) -> str:
        """Format for display / LLM feedback."""
        loc = f"{self.file}:{self.line}"
        if self.column:
            loc += f":{self.column}"
        code_str = f" [{self.code}]" if self.code else ""
        return f"{loc}{code_str}: {self.message}"


@dataclass
class LintResult:
    """Results from running a linter."""

    errors: list[LintError] = field(default_factory=list)
    raw_output: str = ""
    exit_code: int = 0
    linter: str = ""

    @property
    def has_errors(self) -> bool:
        """Whether any errors (not just warnings) were found."""
        return any(e.severity == "error" for e in self.errors)

    @property
    def error_count(self) -> int:
        """Number of errors."""
        return sum(1 for e in self.errors if e.severity == "error")

    @property
    def warning_count(self) -> int:
        """Number of warnings."""
        return sum(1 for e in self.errors if e.severity == "warning")

    def summary(self) -> str:
        """One-line summary."""
        if not self.errors:
            return "✅ No lint issues"
        return f"⚠️ {self.error_count} errors, {self.warning_count} warnings"

    def feedback_for_agent(self, max_errors: int = 20) -> str:
        """Build a feedback prompt for the agent to fix lint errors."""
        if not self.errors:
            return ""

        lines = [
            f"## Lint Issues ({self.linter})",
            "",
            f"{self.error_count} errors, {self.warning_count} warnings",
            "",
        ]

        for error in self.errors[:max_errors]:
            lines.append(error.display())

        if len(self.errors) > max_errors:
            lines.append(f"\n... and {len(self.errors) - max_errors} more issues")

        lines.extend(["", "Please fix the lint errors shown above."])
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Linter detection
# ---------------------------------------------------------------------------

_LINTER_CONFIGS: dict[str, dict[str, str]] = {
    "ruff": {
        "files": "ruff.toml,pyproject.toml",
        "check_content": "ruff",
        "command": "python -m ruff check --output-format=text",
        "language": "python",
    },
    "flake8": {
        "files": ".flake8,setup.cfg,tox.ini",
        "check_content": "flake8",
        "command": "python -m flake8",
        "language": "python",
    },
    "pylint": {
        "files": ".pylintrc,pyproject.toml",
        "check_content": "pylint",
        "command": "python -m pylint",
        "language": "python",
    },
    "eslint": {
        "files": ".eslintrc.js,.eslintrc.json,.eslintrc.yml,eslint.config.js",
        "command": "npx eslint . --format=compact",
        "language": "javascript",
    },
    "mypy": {
        "files": "mypy.ini,pyproject.toml",
        "check_content": "mypy",
        "command": "python -m mypy .",
        "language": "python",
    },
}


def detect_linter(workspace: Path) -> Optional[str]:
    """Auto-detect the linter command for the project.

    Args:
        workspace: Project root.

    Returns:
        Linter command, or None.
    """
    for name, config in _LINTER_CONFIGS.items():
        files = config.get("files", "").split(",")
        for filename in files:
            path = workspace / filename.strip()
            if path.exists():
                check = config.get("check_content")
                if check and path.is_file():
                    try:
                        content = path.read_text(encoding="utf-8", errors="replace")
                        if check not in content:
                            continue
                    except OSError:
                        continue
                return config["command"]

    return None


# ---------------------------------------------------------------------------
# Lint execution
# ---------------------------------------------------------------------------

def run_linter(
    workspace: Path,
    command: Optional[str] = None,
    files: list[str] | None = None,
    timeout: int = 60,
) -> LintResult:
    """Run the linter and parse results.

    Args:
        workspace: Project root.
        command: Linter command (auto-detected if None).
        files: Specific files to lint (all if None).
        timeout: Timeout in seconds.

    Returns:
        LintResult with parsed errors.
    """
    if command is None:
        command = detect_linter(workspace)
        if command is None:
            return LintResult(raw_output="No linter detected", exit_code=-1)

    # Append specific files if provided
    full_command = command
    if files:
        import shlex
        safe_files = [shlex.quote(f) for f in files]
        full_command = f"{command} {' '.join(safe_files)}"

    try:
        result = subprocess.run(
            full_command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=str(workspace),
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return LintResult(raw_output=f"Linter timed out after {timeout}s", exit_code=-2)
    except Exception as exc:
        return LintResult(raw_output=f"Failed to run linter: {exc}", exit_code=-3)

    output = result.stdout + "\n" + result.stderr
    linter_name = command.split()[0] if command else "unknown"

    # Parse output
    lint_result = _parse_lint_output(output)
    lint_result.raw_output = output
    lint_result.exit_code = result.returncode
    lint_result.linter = linter_name

    return lint_result


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

# Common lint output format: file:line:col: CODE message
_LINT_PATTERN = re.compile(
    r'^(.+?):(\d+)(?::(\d+))?\s*:\s*(?:(\w\d+)\s+)?(.+)$',
    re.MULTILINE,
)

# Ruff/flake8 specific: file.py:10:1: E501 line too long
_RUFF_PATTERN = re.compile(
    r'^(.+?):(\d+):(\d+):\s+([A-Z]\d+)\s+(.+)$',
    re.MULTILINE,
)


def _parse_lint_output(output: str) -> LintResult:
    """Parse lint output into structured errors."""
    result = LintResult()

    # Try ruff/flake8 format first (more specific)
    for match in _RUFF_PATTERN.finditer(output):
        code = match.group(4)
        severity = "error" if code.startswith("E") else "warning"
        result.errors.append(LintError(
            file=match.group(1),
            line=int(match.group(2)),
            column=int(match.group(3)),
            code=code,
            message=match.group(5).strip(),
            severity=severity,
        ))

    if result.errors:
        return result

    # Fall back to generic pattern
    for match in _LINT_PATTERN.finditer(output):
        code = match.group(4) or ""
        severity = "error" if "error" in match.group(5).lower() else "warning"
        result.errors.append(LintError(
            file=match.group(1),
            line=int(match.group(2)),
            column=int(match.group(3) or 0),
            code=code,
            message=match.group(5).strip(),
            severity=severity,
        ))

    return result


# ---------------------------------------------------------------------------
# Lint Feedback Controller
# ---------------------------------------------------------------------------

class LintFeedback:
    """Manages lint-after-edit feedback loop.

    After the agent edits files, this runs the linter on modified files
    and feeds errors back to the agent for self-correction.

    Args:
        workspace: Project root.
        lint_command: Override lint command.
        max_attempts: Max auto-fix attempts.
    """

    def __init__(
        self,
        workspace: Path,
        lint_command: Optional[str] = None,
        max_attempts: int = 3,
    ) -> None:
        self._workspace = workspace
        self._lint_command = lint_command or detect_linter(workspace)
        self._max_attempts = max_attempts
        self._attempt = 0

    @property
    def has_linter(self) -> bool:
        """Whether a linter was detected."""
        return self._lint_command is not None

    def lint_files(self, files: list[str]) -> LintResult:
        """Lint specific files and return results.

        Args:
            files: List of file paths (relative to workspace).

        Returns:
            LintResult.
        """
        self._attempt += 1
        return run_linter(self._workspace, self._lint_command, files)

    def should_retry(self, result: LintResult) -> bool:
        """Whether to retry fixing lint errors."""
        return (
            result.has_errors
            and self._attempt < self._max_attempts
        )
