"""Testing loop — edit → run tests → fix failures → repeat.

Auto-detects test commands, parses output from common frameworks,
and feeds failures back to the agent for self-correction.
"""

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
class TestFailure:
    """A single test failure."""

    test_name: str
    file: str = ""
    line: int = 0
    message: str = ""
    traceback: str = ""

    def display(self) -> str:
        """Format for display / LLM feedback."""
        location = f"{self.file}:{self.line}" if self.file else ""
        parts = [f"FAIL: {self.test_name}"]
        if location:
            parts.append(f"  at {location}")
        if self.message:
            parts.append(f"  {self.message}")
        if self.traceback:
            # Truncate long tracebacks
            tb_lines = self.traceback.strip().splitlines()
            if len(tb_lines) > 10:
                tb_lines = tb_lines[:5] + ["  ..."] + tb_lines[-5:]
            parts.append("\n".join(f"  {l}" for l in tb_lines))
        return "\n".join(parts)


@dataclass
class TestResult:
    """Results from running a test suite."""

    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    total: int = 0
    failures: list[TestFailure] = field(default_factory=list)
    raw_output: str = ""
    exit_code: int = 0
    duration_seconds: float = 0.0

    @property
    def all_passed(self) -> bool:
        """Whether all tests passed."""
        return self.failed == 0 and self.errors == 0

    def summary(self) -> str:
        """One-line summary."""
        parts = []
        if self.passed:
            parts.append(f"{self.passed} passed")
        if self.failed:
            parts.append(f"{self.failed} failed")
        if self.errors:
            parts.append(f"{self.errors} errors")
        if self.skipped:
            parts.append(f"{self.skipped} skipped")
        status = "✅ PASS" if self.all_passed else "❌ FAIL"
        return f"{status}: {', '.join(parts)} ({self.duration_seconds:.1f}s)"

    def failure_report(self) -> str:
        """Detailed failure report for LLM feedback."""
        if not self.failures:
            return ""

        lines = [
            f"{len(self.failures)} test(s) failed:",
            "",
        ]
        for f in self.failures[:10]:  # cap at 10 failures
            lines.append(f.display())
            lines.append("")

        if len(self.failures) > 10:
            lines.append(f"... and {len(self.failures) - 10} more failures")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Test command detection
# ---------------------------------------------------------------------------

_TEST_COMMANDS: dict[str, dict[str, str]] = {
    "pytest": {
        "files": "pyproject.toml,pytest.ini,setup.cfg,tox.ini",
        "check_content": "pytest",
        "command": "python -m pytest -v --tb=short",
        "framework": "pytest",
    },
    "unittest": {
        "files": "tests/__init__.py,test",
        "command": "python -m unittest discover -v",
        "framework": "unittest",
    },
    "npm_test": {
        "files": "package.json",
        "check_content": '"test"',
        "command": "npm test",
        "framework": "jest",
    },
    "cargo_test": {
        "files": "Cargo.toml",
        "command": "cargo test",
        "framework": "cargo",
    },
    "go_test": {
        "files": "go.mod",
        "command": "go test ./...",
        "framework": "go",
    },
}


def detect_test_command(workspace: Path) -> Optional[str]:
    """Auto-detect the test command for the project.

    Args:
        workspace: Project root directory.

    Returns:
        Test command string, or None if not detected.
    """
    for _, config in _TEST_COMMANDS.items():
        files = config.get("files", "").split(",")
        for filename in files:
            path = workspace / filename.strip()
            if path.exists():
                # Check content if needed
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


def detect_test_framework(workspace: Path) -> str:
    """Detect which test framework is in use.

    Returns:
        Framework name (pytest, jest, cargo, go, unittest, unknown).
    """
    for _, config in _TEST_COMMANDS.items():
        files = config.get("files", "").split(",")
        for filename in files:
            path = workspace / filename.strip()
            if path.exists():
                return config.get("framework", "unknown")
    return "unknown"


# ---------------------------------------------------------------------------
# Test execution
# ---------------------------------------------------------------------------

def run_tests(
    workspace: Path,
    command: Optional[str] = None,
    timeout: int = 120,
) -> TestResult:
    """Run the test suite and return parsed results.

    Args:
        workspace: Project root directory.
        command: Test command to run (auto-detected if None).
        timeout: Timeout in seconds.

    Returns:
        TestResult with parsed output.
    """
    import time

    if command is None:
        command = detect_test_command(workspace)
        if command is None:
            return TestResult(
                raw_output="Could not detect test command. Set it with /settings test_command <cmd>",
                exit_code=-1,
            )

    logger.info("Running tests: %s", command)
    start = time.monotonic()

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=str(workspace),
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return TestResult(
            raw_output=f"Tests timed out after {timeout}s",
            exit_code=-2,
        )
    except Exception as exc:
        return TestResult(
            raw_output=f"Failed to run tests: {exc}",
            exit_code=-3,
        )

    duration = time.monotonic() - start
    output = result.stdout + "\n" + result.stderr

    # Detect framework and parse
    framework = detect_test_framework(workspace)
    test_result = _parse_output(output, framework)
    test_result.raw_output = output
    test_result.exit_code = result.returncode
    test_result.duration_seconds = duration

    return test_result


# ---------------------------------------------------------------------------
# Output parsers
# ---------------------------------------------------------------------------

def _parse_output(output: str, framework: str) -> TestResult:
    """Parse test output based on the framework."""
    if framework == "pytest":
        return _parse_pytest(output)
    elif framework in ("jest", "mocha"):
        return _parse_jest(output)
    elif framework == "cargo":
        return _parse_cargo(output)
    elif framework == "go":
        return _parse_go(output)
    else:
        return _parse_generic(output)


def _parse_pytest(output: str) -> TestResult:
    """Parse pytest output."""
    result = TestResult()

    # Summary line: items can appear in any order — "3 passed, 1 failed" or "1 failed, 3 passed"
    # Search for a line that looks like the summary (contains "passed", "failed", etc. with counts)
    summary_line = re.search(r'=+\s*(.*?)\s*=+\s*$', output, re.MULTILINE)
    summary_text = summary_line.group(1) if summary_line else output

    passed_m = re.search(r'(\d+)\s+passed', summary_text)
    failed_m = re.search(r'(\d+)\s+failed', summary_text)
    error_m = re.search(r'(\d+)\s+error', summary_text)
    skipped_m = re.search(r'(\d+)\s+skipped', summary_text)

    if passed_m or failed_m or error_m or skipped_m:
        result.passed = int(passed_m.group(1)) if passed_m else 0
        result.failed = int(failed_m.group(1)) if failed_m else 0
        result.errors = int(error_m.group(1)) if error_m else 0
        result.skipped = int(skipped_m.group(1)) if skipped_m else 0
        result.total = result.passed + result.failed + result.errors + result.skipped

    # Also match "X failed" only lines
    if result.total == 0:
        failed_only = re.search(r'(\d+)\s+failed', output)
        if failed_only:
            result.failed = int(failed_only.group(1))
            result.total = result.failed

    # Parse individual failures
    failure_blocks = re.findall(
        r'FAILED\s+(\S+)(?:::(\S+))?\s*-\s*(.*?)(?=\n(?:FAILED|={3,}|$))',
        output,
        re.DOTALL,
    )
    for match in failure_blocks:
        result.failures.append(TestFailure(
            test_name=match[1] or match[0],
            file=match[0].split("::")[0] if "::" in match[0] else "",
            message=match[2].strip()[:500],
        ))

    return result


def _parse_jest(output: str) -> TestResult:
    """Parse Jest/npm test output."""
    result = TestResult()

    # Summary: "Tests: 1 failed, 5 passed, 6 total"
    summary_match = re.search(
        r'Tests:\s*'
        r'(?:(\d+)\s+failed,\s*)?'
        r'(?:(\d+)\s+passed,\s*)?'
        r'(\d+)\s+total',
        output,
    )
    if summary_match:
        result.failed = int(summary_match.group(1) or 0)
        result.passed = int(summary_match.group(2) or 0)
        result.total = int(summary_match.group(3) or 0)

    # Individual failures: "● Test Suite > test name"
    for match in re.finditer(r'●\s+(.+?)(?:\n\s+(.+?))?(?=\n\s*●|\Z)', output, re.DOTALL):
        result.failures.append(TestFailure(
            test_name=match.group(1).strip(),
            message=(match.group(2) or "").strip()[:500],
        ))

    return result


def _parse_cargo(output: str) -> TestResult:
    """Parse cargo test output."""
    result = TestResult()

    # Summary: "test result: ok. 5 passed; 0 failed; 0 ignored"
    summary_match = re.search(
        r'test result:\s*\w+\.\s*(\d+)\s+passed;\s*(\d+)\s+failed;\s*(\d+)\s+ignored',
        output,
    )
    if summary_match:
        result.passed = int(summary_match.group(1))
        result.failed = int(summary_match.group(2))
        result.skipped = int(summary_match.group(3))
        result.total = result.passed + result.failed + result.skipped

    # Failures: "---- test_name stdout ----"
    for match in re.finditer(r'----\s+(\S+)\s+stdout\s+----\n(.+?)(?=\n----|\nfailures:|\Z)', output, re.DOTALL):
        result.failures.append(TestFailure(
            test_name=match.group(1),
            traceback=match.group(2).strip()[:800],
        ))

    return result


def _parse_go(output: str) -> TestResult:
    """Parse go test output."""
    result = TestResult()

    # Count PASS/FAIL lines
    result.passed = len(re.findall(r'--- PASS:', output))
    result.failed = len(re.findall(r'--- FAIL:', output))
    result.total = result.passed + result.failed

    # Parse failures
    for match in re.finditer(r'--- FAIL:\s+(\S+)\s+\((.+?)\)\n(.+?)(?=--- |\Z)', output, re.DOTALL):
        result.failures.append(TestFailure(
            test_name=match.group(1),
            message=match.group(3).strip()[:500],
        ))

    return result


def _parse_generic(output: str) -> TestResult:
    """Best-effort parser for unknown frameworks."""
    result = TestResult()
    output_lower = output.lower()

    # Look for common patterns
    pass_count = len(re.findall(r'\bpass(?:ed)?\b', output_lower))
    fail_count = len(re.findall(r'\bfail(?:ed|ure)?\b', output_lower))
    error_count = len(re.findall(r'\berror\b', output_lower))

    result.passed = pass_count
    result.failed = fail_count
    result.errors = error_count
    result.total = pass_count + fail_count + error_count

    return result


# ---------------------------------------------------------------------------
# Testing Loop Controller
# ---------------------------------------------------------------------------

class TestingLoop:
    """Automated edit → test → fix cycle.

    After the agent makes edits, this runs the test suite and
    feeds failures back for automatic correction.

    Args:
        workspace: Project root.
        max_attempts: Maximum fix attempts before giving up.
        test_command: Override test command (auto-detected if None).
    """

    def __init__(
        self,
        workspace: Path,
        max_attempts: int = 5,
        test_command: Optional[str] = None,
    ) -> None:
        self._workspace = workspace
        self._max_attempts = max_attempts
        self._test_command = test_command or detect_test_command(workspace)
        self._results_history: list[TestResult] = []

    @property
    def test_command(self) -> Optional[str]:
        """The detected or configured test command."""
        return self._test_command

    @property
    def results_history(self) -> list[TestResult]:
        """History of test results from this loop."""
        return self._results_history

    def run_tests(self) -> TestResult:
        """Run the test suite once.

        Returns:
            TestResult with parsed output.
        """
        result = run_tests(self._workspace, self._test_command)
        self._results_history.append(result)
        return result

    def build_fix_prompt(self, result: TestResult) -> str:
        """Build a prompt for the agent to fix test failures.

        Args:
            result: The failed test result.

        Returns:
            Prompt string describing the failures for the LLM.
        """
        attempt = len(self._results_history)
        lines = [
            f"## Test Failures (attempt {attempt}/{self._max_attempts})",
            "",
            result.summary(),
            "",
        ]

        if result.failures:
            lines.append(result.failure_report())
        else:
            # Fall back to raw output
            lines.append("Raw output (last 50 lines):")
            lines.append("```")
            raw_lines = result.raw_output.strip().splitlines()
            lines.extend(raw_lines[-50:])
            lines.append("```")

        lines.extend([
            "",
            "Please fix the code to make these tests pass.",
            "Focus on the specific errors shown above.",
        ])

        return "\n".join(lines)

    @property
    def attempts_remaining(self) -> int:
        """Number of fix attempts remaining."""
        return max(0, self._max_attempts - len(self._results_history))

    @property
    def is_exhausted(self) -> bool:
        """Whether we've used all fix attempts."""
        return len(self._results_history) >= self._max_attempts
