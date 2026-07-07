"""Shared fixtures for the unjess test suite."""

import os
import shutil
import subprocess
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Workspace fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_workspace(tmp_path: Path) -> Path:
    """Create a temporary workspace with sample project files.

    Provides a realistic mini-project structure:
        workspace/
        ├── src/
        │   ├── app.py
        │   └── utils.py
        ├── tests/
        │   └── test_app.py
        ├── README.md
        └── pyproject.toml
    """
    # Python source files
    src = tmp_path / "src"
    src.mkdir()
    (src / "app.py").write_text(
        textwrap.dedent("""\
        \"\"\"Main application module.\"\"\"

        def greet(name: str) -> str:
            return f"Hello, {name}!"

        def add(a: int, b: int) -> int:
            return a + b

        if __name__ == "__main__":
            print(greet("World"))
        """),
        encoding="utf-8",
    )
    (src / "utils.py").write_text(
        textwrap.dedent("""\
        \"\"\"Utility functions.\"\"\"

        import os
        from pathlib import Path

        def get_project_root() -> Path:
            return Path(__file__).parent.parent

        SECRET_KEY = "not-a-real-key"
        """),
        encoding="utf-8",
    )

    # Test file
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_app.py").write_text(
        textwrap.dedent("""\
        from src.app import greet, add

        def test_greet():
            assert greet("Test") == "Hello, Test!"

        def test_add():
            assert add(2, 3) == 5
        """),
        encoding="utf-8",
    )

    # Project files
    (tmp_path / "README.md").write_text("# Test Project\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        textwrap.dedent("""\
        [project]
        name = "test-project"
        version = "0.1.0"
        requires-python = ">=3.10"
        """),
        encoding="utf-8",
    )

    return tmp_path


@pytest.fixture
def sample_git_repo(tmp_workspace: Path) -> Path:
    """Turn tmp_workspace into a git repo with an initial commit."""
    git_exe = shutil.which("git")
    if git_exe is None:
        pytest.skip("git not available")
    env = {**os.environ, "GIT_AUTHOR_NAME": "Test", "GIT_AUTHOR_EMAIL": "t@t.com",
           "GIT_COMMITTER_NAME": "Test", "GIT_COMMITTER_EMAIL": "t@t.com"}
    subprocess.run(["git", "init"], cwd=tmp_workspace, capture_output=True, env=env, check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_workspace, capture_output=True, env=env, check=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=tmp_workspace, capture_output=True, env=env, check=True,
    )
    return tmp_workspace


# ---------------------------------------------------------------------------
# Settings fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def settings(tmp_workspace: Path) -> "Settings":
    """Return a default Settings instance with workspace set to tmp_workspace."""
    from unjess.config import Settings
    return Settings(workspace=str(tmp_workspace))


# ---------------------------------------------------------------------------
# Mock display / input
# ---------------------------------------------------------------------------

class MockDisplay:
    """A test double for DisplayProtocol that captures output."""

    def __init__(self) -> None:
        self.messages: list[str] = []
        self.tool_calls: list[tuple[str, dict]] = []
        self.errors: list[str] = []
        self.diffs: list[str] = []
        self.streamed_text: str = ""

    def show_text(self, text: str) -> None:
        self.messages.append(text)

    def show_markdown(self, text: str) -> None:
        self.messages.append(text)

    def show_tool_call(self, name: str, args: dict) -> None:
        self.tool_calls.append((name, args))

    def show_tool_result(self, name: str, result: str, is_error: bool = False) -> None:
        self.messages.append(f"[{name}] {result}")

    def show_error(self, msg: str) -> None:
        self.errors.append(msg)

    def show_diff(self, path: str, old: str, new: str) -> None:
        self.diffs.append(f"{path}: {old} -> {new}")

    def show_info(self, msg: str) -> None:
        self.messages.append(msg)

    def show_stats(self, tokens_in: int = 0, tokens_out: int = 0, cost: float = 0.0) -> None:
        pass

    def stream_start(self) -> None:
        pass

    def stream_token(self, token: str) -> None:
        self.streamed_text += token

    def stream_end(self) -> None:
        pass

    def show_thinking(self, text: str) -> None:
        pass

    def show_spinner(self, msg: str = "") -> Any:
        return MagicMock()

    def show_banner(self, **kwargs: Any) -> None:
        pass


class MockInput:
    """A test double for InputProtocol that auto-approves."""

    def __init__(self, auto_approve: bool = True) -> None:
        self.auto_approve = auto_approve
        self.approval_requests: list[tuple[str, dict]] = []

    def ask_approval(self, tool_name: str, args: dict) -> tuple[bool, bool]:
        """Return (approved, always)."""
        self.approval_requests.append((tool_name, args))
        return (self.auto_approve, False)


@pytest.fixture
def mock_display() -> MockDisplay:
    """Fixture returning a MockDisplay instance."""
    return MockDisplay()


@pytest.fixture
def mock_input() -> MockInput:
    """Fixture returning a MockInput that auto-approves everything."""
    return MockInput(auto_approve=True)


# ---------------------------------------------------------------------------
# Mock LLM provider
# ---------------------------------------------------------------------------

class MockLLMProvider:
    """A fake LLM provider that returns canned responses.

    Configure responses by setting .responses (a list of LLMResponse objects
    that will be returned in order, cycling if needed).
    """

    def __init__(self) -> None:
        from unjess.llm.base import LLMResponse, Usage
        self._responses: list[LLMResponse] = [
            LLMResponse(text="Mock response", usage=Usage(prompt_tokens=10, completion_tokens=5))
        ]
        self._call_index = 0
        self.call_log: list[dict] = []

    @property
    def provider_name(self) -> str:
        return "mock"

    def set_responses(self, responses: list) -> None:
        """Set the sequence of responses to return."""
        self._responses = responses
        self._call_index = 0

    def chat(self, messages: list, tools: list | None = None, model: str = "") -> "LLMResponse":
        self.call_log.append({"messages": messages, "tools": tools, "model": model})
        resp = self._responses[self._call_index % len(self._responses)]
        self._call_index += 1
        return resp

    def chat_stream(self, messages: list, tools: list | None = None, model: str = ""):
        from unjess.llm.base import StreamChunk
        resp = self.chat(messages, tools, model)
        if resp.text:
            yield StreamChunk(text=resp.text)
        yield StreamChunk(done=True, usage=resp.usage)

    def list_models(self) -> list[str]:
        return ["mock-model"]


@pytest.fixture
def mock_llm() -> MockLLMProvider:
    """Fixture returning a MockLLMProvider."""
    return MockLLMProvider()
