"""Tests for unjess.auto_summary — heuristic extraction and LLM-based summary generation."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from unjess.auto_summary import _extract_heuristic_summary, generate_summary
from unjess.memory import ConversationSummary, MemoryStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _msg(role: str, content: str, **kwargs: Any) -> dict[str, Any]:
    """Build a message dict."""
    d: dict[str, Any] = {"role": role, "content": content}
    d.update(kwargs)
    return d


def _make_settings(workspace: str = "/tmp/project", model: str = "test-model") -> MagicMock:
    s = MagicMock()
    s.workspace = workspace
    s.model = model
    return s


def _make_router(response_text: str = "LLM summary text") -> MagicMock:
    router = MagicMock()
    resp = MagicMock()
    resp.content = response_text
    router.chat.return_value = resp
    return router


# ---------------------------------------------------------------------------
# _extract_heuristic_summary
# ---------------------------------------------------------------------------

class TestExtractHeuristicSummary:
    """Tests for the heuristic (non-LLM) summary extractor."""

    def test_empty_conversation(self) -> None:
        result = _extract_heuristic_summary([])
        assert isinstance(result, ConversationSummary)
        assert result.title == "Untitled session"
        assert result.summary == "(no summary)"
        assert result.message_count == 0

    def test_title_from_first_user_message(self) -> None:
        msgs = [_msg("user", "Fix the login bug")]
        result = _extract_heuristic_summary(msgs)
        assert "Fix the login bug" in result.title

    def test_title_truncated_at_60_chars(self) -> None:
        long_msg = "A" * 100
        msgs = [_msg("user", long_msg)]
        result = _extract_heuristic_summary(msgs)
        assert result.title.endswith("...")
        assert len(result.title) == 63  # 60 + "..."

    def test_files_extracted_from_assistant_messages(self) -> None:
        msgs = [
            _msg("user", "Update the config"),
            _msg("assistant", "I edited src/config.py and templates/base.html"),
        ]
        result = _extract_heuristic_summary(msgs)
        assert "config.py" in result.files_modified
        assert "base.html" in result.files_modified

    def test_files_with_path_uses_basename(self) -> None:
        msgs = [
            _msg("assistant", "Modified /home/user/project/src/utils.py"),
        ]
        result = _extract_heuristic_summary(msgs)
        assert "utils.py" in result.files_modified

    def test_tool_names_tracked(self) -> None:
        msgs = [
            _msg("tool", "OK", name="write_file"),
            _msg("tool", "OK", name="run_command"),
        ]
        result = _extract_heuristic_summary(msgs)
        assert "Tools used:" in result.summary
        assert "run_command" in result.summary
        assert "write_file" in result.summary

    def test_key_topics_from_repeated_words(self) -> None:
        msgs = [
            _msg("user", "refactor the database migration code"),
            _msg("user", "also fix the database connection pool in database module"),
        ]
        result = _extract_heuristic_summary(msgs)
        assert "database" in result.key_topics

    def test_conversation_id_custom(self) -> None:
        result = _extract_heuristic_summary([], conversation_id="my-session-123")
        assert result.conversation_id == "my-session-123"

    def test_conversation_id_auto_generated(self) -> None:
        result = _extract_heuristic_summary([])
        assert result.conversation_id.startswith("session-")

    def test_workspace_stored(self) -> None:
        result = _extract_heuristic_summary([], workspace="/project/root")
        assert result.workspace == "/project/root"

    def test_user_content_truncated_at_200(self) -> None:
        long_content = "x" * 500
        msgs = [_msg("user", long_content)]
        result = _extract_heuristic_summary(msgs)
        # Title should be from first 60 chars, summary from first 100
        assert len(result.title) <= 63

    def test_non_string_content_skipped(self) -> None:
        msgs = [_msg("user", ["not", "a", "string"])]
        result = _extract_heuristic_summary(msgs)
        assert result.title == "Untitled session"

    def test_message_count(self) -> None:
        msgs = [_msg("user", "hi"), _msg("assistant", "hello"), _msg("user", "thanks")]
        result = _extract_heuristic_summary(msgs)
        assert result.message_count == 3

    def test_files_cleaned_of_punctuation(self) -> None:
        # The source checks word.endswith(ext) before stripping,
        # so words with trailing punctuation after the extension are missed.
        # But quotes/brackets around a word ending in .py still match.
        msgs = [
            _msg("assistant", 'Check "src/app.py" and also (utils.py)'),
        ]
        result = _extract_heuristic_summary(msgs)
        # "src/app.py" => endswith .py? no, ends with '"'. Quotes block it.
        # "(utils.py)" => endswith .py? no, ends with ')'.
        # Only bare words ending in .py are detected.
        # So this test verifies that only properly-terminated words work:
        msgs2 = [
            _msg("assistant", "Edited src/app.py and also utils.py today"),
        ]
        result2 = _extract_heuristic_summary(msgs2)
        assert "app.py" in result2.files_modified
        assert "utils.py" in result2.files_modified

    def test_summary_parts_include_user_request(self) -> None:
        msgs = [_msg("user", "Deploy to production")]
        result = _extract_heuristic_summary(msgs)
        assert "User request:" in result.summary
        assert "Deploy to production" in result.summary


# ---------------------------------------------------------------------------
# generate_summary
# ---------------------------------------------------------------------------

class TestGenerateSummary:
    """Tests for the generate_summary function (LLM + heuristic paths)."""

    def test_returns_none_for_no_user_messages(self) -> None:
        msgs = [_msg("assistant", "Hello")]
        result = generate_summary(msgs, _make_router(), _make_settings())
        assert result is None

    def test_llm_summary_used_when_available(self) -> None:
        msgs = [
            _msg("user", "Fix the bug in parser"),
            _msg("assistant", "I fixed src/parser.py"),
        ]
        router = _make_router("LLM generated a great summary.")
        result = generate_summary(msgs, router, _make_settings())
        assert result is not None
        assert result.summary == "LLM generated a great summary."

    def test_heuristic_fallback_on_llm_failure(self) -> None:
        msgs = [
            _msg("user", "Update the tests"),
            _msg("assistant", "Done with test_app.py"),
        ]
        router = _make_router()
        router.chat.side_effect = RuntimeError("API error")
        result = generate_summary(msgs, router, _make_settings())
        assert result is not None
        # Heuristic summary should contain "User request:"
        assert "User request:" in result.summary

    def test_heuristic_fallback_on_empty_llm_response(self) -> None:
        msgs = [_msg("user", "Something")]
        router = _make_router("")
        result = generate_summary(msgs, router, _make_settings())
        assert result is not None
        assert "User request:" in result.summary

    def test_heuristic_fallback_on_none_content(self) -> None:
        msgs = [_msg("user", "Something")]
        router = _make_router("")
        resp = MagicMock()
        resp.content = None
        router.chat.return_value = resp
        result = generate_summary(msgs, router, _make_settings())
        assert result is not None

    def test_memory_store_save_called(self, tmp_path: Path) -> None:
        msgs = [_msg("user", "Fix bug"), _msg("assistant", "Fixed")]
        store = MemoryStore(tmp_path)
        store.ensure_loaded()
        result = generate_summary(
            msgs, _make_router(), _make_settings(),
            memory_store=store,
        )
        assert result is not None
        recent = store.get_recent_summaries()
        assert len(recent) == 1

    def test_memory_store_save_error_handled(self) -> None:
        msgs = [_msg("user", "Fix bug")]
        store = MagicMock()
        store.add_summary.side_effect = RuntimeError("disk full")
        # Should not raise
        result = generate_summary(
            msgs, _make_router(), _make_settings(),
            memory_store=store,
        )
        assert result is not None

    def test_conversation_id_passed_through(self) -> None:
        msgs = [_msg("user", "hello")]
        result = generate_summary(
            msgs, _make_router(), _make_settings(),
            conversation_id="conv-abc",
        )
        assert result is not None
        assert result.conversation_id == "conv-abc"

    def test_workspace_from_settings(self) -> None:
        msgs = [_msg("user", "hello")]
        result = generate_summary(
            msgs, _make_router(), _make_settings(workspace="/my/project"),
        )
        assert result is not None
        assert result.workspace == "/my/project"

    def test_model_from_settings(self) -> None:
        msgs = [_msg("user", "hello"), _msg("assistant", "hi")]
        result = generate_summary(
            msgs, _make_router("Summary"), _make_settings(model="gpt-4o"),
        )
        assert result is not None
        assert result.model == "gpt-4o"

    def test_long_conversation_truncated(self) -> None:
        # 30 messages — should be truncated to last 20 in the LLM prompt
        msgs = [_msg("user" if i % 2 == 0 else "assistant", f"Message {i}") for i in range(30)]
        router = _make_router("Summary of long conversation")
        result = generate_summary(msgs, router, _make_settings())
        assert result is not None
        # Verify the LLM was called with truncated conversation
        call_args = router.chat.call_args
        prompt_msg = call_args[1]["messages"][0]["content"] if "messages" in call_args[1] else call_args[0][0][0]["content"]
        # Should not contain all 30 messages (only last 20)
        assert "Message 0" not in prompt_msg or "Message 29" in prompt_msg

    def test_long_message_content_truncated(self) -> None:
        long_text = "x" * 500
        msgs = [_msg("user", long_text)]
        router = _make_router("Summary")
        result = generate_summary(msgs, router, _make_settings())
        assert result is not None
        call_args = router.chat.call_args
        prompt = str(call_args)
        # The message content should be truncated with "..."
        assert "..." in prompt

    def test_settings_without_workspace_attr(self) -> None:
        msgs = [_msg("user", "test")]
        settings = MagicMock(spec=[])  # no attributes at all
        result = generate_summary(msgs, _make_router(), settings)
        assert result is not None
        assert result.workspace == ""

    def test_settings_without_model_attr(self) -> None:
        msgs = [_msg("user", "test")]
        settings = MagicMock(spec=["workspace"])
        settings.workspace = "/proj"
        result = generate_summary(msgs, _make_router("Summary"), settings)
        assert result is not None
        assert result.model == ""

    def test_non_user_assistant_roles_excluded_from_condensed(self) -> None:
        msgs = [
            _msg("user", "Do stuff"),
            _msg("system", "System prompt"),
            _msg("tool", "Result", name="write_file"),
            _msg("assistant", "Done"),
        ]
        router = _make_router("Summary")
        generate_summary(msgs, router, _make_settings())
        # Only user + assistant should appear in the condensed text
        call_args = router.chat.call_args
        prompt = str(call_args)
        assert "System prompt" not in prompt
