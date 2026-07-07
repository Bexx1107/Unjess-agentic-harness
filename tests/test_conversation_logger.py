"""Tests for unjess.conversation_logger — cost tracking and JSONL logging."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from unjess.conversation_logger import (
    ConversationLogger,
    CostTracker,
    LogType,
    _COST_RATES,
    _FREE_PREFIXES,
    _FREE_PROVIDERS,
    is_free_model,
)


# ── helpers ───────────────────────────────────────────────────────────────

def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read all JSON lines from a file."""
    entries: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entries.append(json.loads(line))
    return entries


# =====================================================================
# is_free_model
# =====================================================================

class TestIsFreeModel:
    """Tests for the top-level is_free_model helper."""

    def test_ollama_provider_always_free(self) -> None:
        assert is_free_model("anything-at-all", provider="ollama") is True

    def test_non_free_provider_not_enough(self) -> None:
        assert is_free_model("gpt-4o", provider="openai") is False

    def test_prefix_match_llama(self) -> None:
        assert is_free_model("llama-3.1-8b-instruct") is True

    def test_prefix_match_mixtral(self) -> None:
        assert is_free_model("mixtral-8x7b-v0.1") is True

    def test_prefix_match_codestral(self) -> None:
        assert is_free_model("codestral-latest") is True

    def test_prefix_match_openrouter(self) -> None:
        assert is_free_model("openrouter/something") is True

    def test_prefix_match_case_insensitive(self) -> None:
        # model.lower() is compared against _FREE_PREFIXES
        assert is_free_model("Llama-3.1-8b") is True
        assert is_free_model("MIXTRAL-8x7b") is True

    def test_prefix_match_deepseek(self) -> None:
        assert is_free_model("deepseek-coder-v2") is True

    def test_prefix_match_qwen(self) -> None:
        assert is_free_model("qwen2.5-coder") is True

    def test_explicit_zero_rates_via_exact_key(self) -> None:
        # zai-glm-4.7 has (0,0) — doesn't match _FREE_PREFIXES via
        # "glm-" because the name starts with "zai-".  But the rate
        # lookup should still find it.
        assert is_free_model("zai-glm-4.7") is True

    def test_paid_model_not_free(self) -> None:
        assert is_free_model("gpt-4o") is False

    def test_paid_model_claude_not_free(self) -> None:
        assert is_free_model("claude-sonnet-4") is False

    def test_unknown_model_not_free(self) -> None:
        assert is_free_model("totally-unknown-model-xyz") is False

    def test_empty_string_not_free(self) -> None:
        assert is_free_model("") is False

    def test_free_prefix_gemma(self) -> None:
        assert is_free_model("gemma-7b") is True

    def test_open_mistral_nemo_free(self) -> None:
        assert is_free_model("open-mistral-nemo") is True

    def test_mistral_small_free(self) -> None:
        assert is_free_model("mistral-small-latest") is True

    def test_mistral_large_not_free(self) -> None:
        assert is_free_model("mistral-large") is False

    def test_free_providers_set_contents(self) -> None:
        assert "ollama" in _FREE_PROVIDERS


# =====================================================================
# CostTracker — rate lookup and estimation
# =====================================================================

class TestCostTrackerEstimation:
    """Tests for CostTracker.estimate_cost / _estimate_cost rate lookup."""

    def test_gpt4o_cost_1m_tokens(self) -> None:
        ct = CostTracker()
        # 1M in, 1M out  →  $2.50 + $10.00
        cost = ct.estimate_cost("gpt-4o", 1_000_000, 1_000_000)
        assert cost == pytest.approx(12.50)

    def test_claude_sonnet_4_cost(self) -> None:
        ct = CostTracker()
        cost = ct.estimate_cost("claude-sonnet-4", 1_000_000, 1_000_000)
        assert cost == pytest.approx(18.00)

    def test_claude_opus_4_cost(self) -> None:
        ct = CostTracker()
        cost = ct.estimate_cost("claude-opus-4", 1_000_000, 1_000_000)
        assert cost == pytest.approx(90.00)

    def test_gemini_flash_cost(self) -> None:
        ct = CostTracker()
        cost = ct.estimate_cost("gemini-2.5-flash", 1_000_000, 1_000_000)
        assert cost == pytest.approx(0.75)

    def test_free_model_returns_zero(self) -> None:
        ct = CostTracker()
        cost = ct.estimate_cost("llama-3.3-70b", 1_000_000, 1_000_000)
        assert cost == 0.0

    def test_unknown_model_returns_zero(self) -> None:
        ct = CostTracker()
        cost = ct.estimate_cost("some-totally-unknown-model", 500, 300)
        assert cost == 0.0

    def test_thinking_tokens_billed_at_output_rate(self) -> None:
        ct = CostTracker()
        # gpt-4o: output rate = $10 / 1M
        # 0 in, 0 out, 1M thinking → $10
        cost = ct.estimate_cost("gpt-4o", 0, 0, thinking_tokens=1_000_000)
        assert cost == pytest.approx(10.00)

    def test_thinking_plus_output_combined(self) -> None:
        ct = CostTracker()
        # gpt-4o: 500k in ($1.25), 500k out ($5.00), 500k thinking ($5.00)
        cost = ct.estimate_cost("gpt-4o", 500_000, 500_000, thinking_tokens=500_000)
        assert cost == pytest.approx(11.25)

    def test_zero_tokens_returns_zero(self) -> None:
        ct = CostTracker()
        assert ct.estimate_cost("gpt-4o", 0, 0) == 0.0

    def test_prefix_match_for_model_variant(self) -> None:
        ct = CostTracker()
        # "gpt-4o-2024-05-13" should prefix-match "gpt-4o" → (2.50, 10.00)
        cost = ct.estimate_cost("gpt-4o-2024-05-13", 1_000_000, 0)
        assert cost == pytest.approx(2.50)

    def test_grok_3_cost(self) -> None:
        ct = CostTracker()
        cost = ct.estimate_cost("grok-3", 1_000_000, 1_000_000)
        assert cost == pytest.approx(18.00)


# =====================================================================
# CostTracker — record_llm_call accumulation
# =====================================================================

class TestCostTrackerRecordLLMCall:
    """Tests for CostTracker.record_llm_call and field accumulation."""

    def test_single_call_updates_totals(self) -> None:
        ct = CostTracker()
        ct.record_llm_call("gpt-4o", tokens_in=100, tokens_out=50)
        assert ct.total_tokens_in == 100
        assert ct.total_tokens_out == 50
        assert ct.llm_calls == 1
        assert ct.total_cost > 0

    def test_multiple_calls_accumulate(self) -> None:
        ct = CostTracker()
        ct.record_llm_call("gpt-4o", tokens_in=100, tokens_out=50)
        ct.record_llm_call("gpt-4o", tokens_in=200, tokens_out=100)
        assert ct.total_tokens_in == 300
        assert ct.total_tokens_out == 150
        assert ct.llm_calls == 2

    def test_record_returns_cost(self) -> None:
        ct = CostTracker()
        cost = ct.record_llm_call("gpt-4o", tokens_in=1_000_000, tokens_out=0)
        assert cost == pytest.approx(2.50)

    def test_free_model_records_zero_cost(self) -> None:
        ct = CostTracker()
        cost = ct.record_llm_call("llama-3.3-70b", tokens_in=5000, tokens_out=3000)
        assert cost == 0.0
        assert ct.total_cost == 0.0

    def test_thinking_tokens_accumulated(self) -> None:
        ct = CostTracker()
        ct.record_llm_call("gpt-4o", tokens_in=100, tokens_out=50, thinking_tokens=200)
        assert ct.total_thinking_tokens == 200

    def test_cache_tokens_accumulated(self) -> None:
        ct = CostTracker()
        ct.record_llm_call(
            "claude-sonnet-4",
            tokens_in=100,
            tokens_out=50,
            cache_read_tokens=500,
            cache_creation_tokens=300,
        )
        assert ct.total_cache_read_tokens == 500
        assert ct.total_cache_creation_tokens == 300

    def test_per_model_tracking(self) -> None:
        ct = CostTracker()
        ct.record_llm_call("gpt-4o", tokens_in=100, tokens_out=50)
        ct.record_llm_call("claude-sonnet-4", tokens_in=200, tokens_out=100)
        ct.record_llm_call("gpt-4o", tokens_in=300, tokens_out=150)

        summary = ct.summary()
        per_model = summary["per_model"]
        assert "gpt-4o" in per_model
        assert "claude-sonnet-4" in per_model
        assert per_model["gpt-4o"]["calls"] == 2
        assert per_model["gpt-4o"]["tokens_in"] == 400
        assert per_model["gpt-4o"]["tokens_out"] == 200
        assert per_model["claude-sonnet-4"]["calls"] == 1

    def test_per_model_is_free_flag(self) -> None:
        ct = CostTracker()
        ct.record_llm_call("gpt-4o", tokens_in=10, tokens_out=5)
        ct.record_llm_call("llama-3.3-70b", tokens_in=10, tokens_out=5)
        per_model = ct.summary()["per_model"]
        assert per_model["gpt-4o"]["is_free"] is False
        assert per_model["llama-3.3-70b"]["is_free"] is True


# =====================================================================
# CostTracker — tool calls
# =====================================================================

class TestCostTrackerToolCalls:
    """Tests for tool call counting."""

    def test_record_tool_call_increments(self) -> None:
        ct = CostTracker()
        assert ct.tool_calls == 0
        ct.record_tool_call()
        assert ct.tool_calls == 1
        ct.record_tool_call()
        ct.record_tool_call()
        assert ct.tool_calls == 3

    def test_tool_calls_appear_in_summary(self) -> None:
        ct = CostTracker()
        ct.record_tool_call()
        ct.record_tool_call()
        assert ct.summary()["tool_calls"] == 2


# =====================================================================
# CostTracker — is_session_free
# =====================================================================

class TestCostTrackerIsSessionFree:
    """Tests for CostTracker.is_session_free."""

    def test_empty_session_is_free(self) -> None:
        ct = CostTracker()
        # No models recorded → all(...) over empty is True
        assert ct.is_session_free() is True

    def test_only_free_models_is_free(self) -> None:
        ct = CostTracker()
        ct.record_llm_call("llama-3.3-70b", tokens_in=100, tokens_out=50)
        ct.record_llm_call("codestral", tokens_in=100, tokens_out=50)
        assert ct.is_session_free() is True

    def test_mixed_models_not_free(self) -> None:
        ct = CostTracker()
        ct.record_llm_call("llama-3.3-70b", tokens_in=100, tokens_out=50)
        ct.record_llm_call("gpt-4o", tokens_in=100, tokens_out=50)
        assert ct.is_session_free() is False

    def test_only_paid_model_not_free(self) -> None:
        ct = CostTracker()
        ct.record_llm_call("claude-opus-4", tokens_in=100, tokens_out=50)
        assert ct.is_session_free() is False


# =====================================================================
# CostTracker — summary
# =====================================================================

class TestCostTrackerSummary:
    """Tests for the summary dict returned by CostTracker.summary."""

    def test_summary_keys(self) -> None:
        ct = CostTracker()
        s = ct.summary()
        expected = {
            "tokens_in", "tokens_out", "thinking_tokens",
            "cache_read_tokens", "cache_creation_tokens",
            "total_tokens", "estimated_cost", "is_free",
            "llm_calls", "tool_calls", "per_model",
        }
        assert set(s.keys()) == expected

    def test_total_tokens_sum(self) -> None:
        ct = CostTracker()
        ct.record_llm_call("gpt-4o", tokens_in=100, tokens_out=50, thinking_tokens=30)
        s = ct.summary()
        assert s["total_tokens"] == 100 + 50 + 30

    def test_estimated_cost_rounded(self) -> None:
        ct = CostTracker()
        ct.record_llm_call("gpt-4o", tokens_in=1, tokens_out=1)
        s = ct.summary()
        # Very small numbers; just check it's a float with <= 6 decimal places
        cost_str = f"{s['estimated_cost']:.6f}"
        assert float(cost_str) == s["estimated_cost"]

    def test_summary_fresh_tracker(self) -> None:
        ct = CostTracker()
        s = ct.summary()
        assert s["tokens_in"] == 0
        assert s["tokens_out"] == 0
        assert s["estimated_cost"] == 0.0
        assert s["llm_calls"] == 0
        assert s["tool_calls"] == 0
        assert s["per_model"] == {}


# =====================================================================
# CostTracker — persist_cost / daily / monthly
# =====================================================================

class TestCostTrackerPersistence:
    """Tests for persist_cost, get_daily_cost, get_monthly_cost."""

    def test_persist_cost_writes_jsonl(self, tmp_path: Path) -> None:
        ct = CostTracker()
        ct.record_llm_call("gpt-4o", tokens_in=1000, tokens_out=500)

        fake_history = tmp_path / "cost_history.jsonl"
        with patch("pathlib.Path.home", return_value=tmp_path / ".home"):
            home_unjess = tmp_path / ".home" / ".unjess"
            home_unjess.mkdir(parents=True, exist_ok=True)
            expected_file = home_unjess / "cost_history.jsonl"

            ct.persist_cost()

            assert expected_file.exists()
            entries = _read_jsonl(expected_file)
            assert len(entries) == 1
            assert "timestamp" in entries[0]
            assert entries[0]["llm_calls"] == 1

    def test_get_daily_cost_no_file(self, tmp_path: Path) -> None:
        with patch("pathlib.Path.home", return_value=tmp_path):
            assert CostTracker.get_daily_cost() == 0.0

    def test_get_monthly_cost_no_file(self, tmp_path: Path) -> None:
        with patch("pathlib.Path.home", return_value=tmp_path):
            assert CostTracker.get_monthly_cost() == 0.0

    def test_get_daily_cost_reads_today(self, tmp_path: Path) -> None:
        from datetime import datetime, timezone

        unjess_dir = tmp_path / ".unjess"
        unjess_dir.mkdir()
        history_file = unjess_dir / "cost_history.jsonl"
        today_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        entries = [
            {"timestamp": f"{today_iso}T10:00:00+00:00", "cost": 1.5},
            {"timestamp": f"{today_iso}T11:00:00+00:00", "cost": 2.5},
            {"timestamp": "2020-01-01T00:00:00+00:00", "cost": 99.0},  # old
        ]
        history_file.write_text(
            "\n".join(json.dumps(e) for e in entries) + "\n",
            encoding="utf-8",
        )

        with patch("pathlib.Path.home", return_value=tmp_path):
            assert CostTracker.get_daily_cost() == pytest.approx(4.0)

    def test_get_monthly_cost_reads_this_month(self, tmp_path: Path) -> None:
        from datetime import datetime, timezone

        unjess_dir = tmp_path / ".unjess"
        unjess_dir.mkdir()
        history_file = unjess_dir / "cost_history.jsonl"
        this_month = datetime.now(timezone.utc).strftime("%Y-%m")

        entries = [
            {"timestamp": f"{this_month}-01T10:00:00+00:00", "cost": 3.0},
            {"timestamp": f"{this_month}-15T10:00:00+00:00", "cost": 7.0},
            {"timestamp": "2019-06-01T00:00:00+00:00", "cost": 50.0},  # old
        ]
        history_file.write_text(
            "\n".join(json.dumps(e) for e in entries) + "\n",
            encoding="utf-8",
        )

        with patch("pathlib.Path.home", return_value=tmp_path):
            assert CostTracker.get_monthly_cost() == pytest.approx(10.0)


# =====================================================================
# LogType constants
# =====================================================================

class TestLogType:
    """Tests for LogType constants."""

    def test_constants_defined(self) -> None:
        assert LogType.USER_INPUT == "USER_INPUT"
        assert LogType.MODEL_RESPONSE == "MODEL_RESPONSE"
        assert LogType.TOOL_CALL == "TOOL_CALL"
        assert LogType.TOOL_RESULT == "TOOL_RESULT"
        assert LogType.ERROR == "ERROR"
        assert LogType.SYSTEM == "SYSTEM"


# =====================================================================
# ConversationLogger — construction & paths
# =====================================================================

class TestConversationLoggerInit:
    """Tests for ConversationLogger initialization."""

    def test_custom_conversation_id(self, tmp_path: Path) -> None:
        with patch("unjess.conversation_logger._CONVERSATIONS_DIR", tmp_path):
            cl = ConversationLogger(conversation_id="test-id-123")
            assert cl.conversation_id == "test-id-123"

    def test_auto_generated_conversation_id(self, tmp_path: Path) -> None:
        with patch("unjess.conversation_logger._CONVERSATIONS_DIR", tmp_path):
            cl = ConversationLogger()
            assert len(cl.conversation_id) == 12  # uuid hex[:12]

    def test_log_dir_created(self, tmp_path: Path) -> None:
        with patch("unjess.conversation_logger._CONVERSATIONS_DIR", tmp_path):
            cl = ConversationLogger(conversation_id="myid")
            assert (tmp_path / "myid").is_dir()

    def test_log_path_property(self, tmp_path: Path) -> None:
        with patch("unjess.conversation_logger._CONVERSATIONS_DIR", tmp_path):
            cl = ConversationLogger(conversation_id="sess1")
            assert cl.log_path == tmp_path / "sess1" / "transcript.jsonl"

    def test_has_cost_tracker(self, tmp_path: Path) -> None:
        with patch("unjess.conversation_logger._CONVERSATIONS_DIR", tmp_path):
            cl = ConversationLogger(conversation_id="sess2")
            assert isinstance(cl.cost_tracker, CostTracker)


# =====================================================================
# ConversationLogger — JSONL logging methods
# =====================================================================

class TestConversationLoggerWriting:
    """Tests for JSONL transcript writing."""

    @pytest.fixture
    def logger_in_tmp(self, tmp_path: Path) -> ConversationLogger:
        with patch("unjess.conversation_logger._CONVERSATIONS_DIR", tmp_path):
            return ConversationLogger(conversation_id="test")

    def test_log_user_input(self, logger_in_tmp: ConversationLogger) -> None:
        logger_in_tmp.log_user_input("Hello!")
        entries = _read_jsonl(logger_in_tmp.log_path)
        assert len(entries) == 1
        assert entries[0]["type"] == "USER_INPUT"
        assert entries[0]["content"] == "Hello!"
        assert entries[0]["step_index"] == 1
        assert "timestamp" in entries[0]

    def test_log_model_response(self, logger_in_tmp: ConversationLogger) -> None:
        logger_in_tmp.log_model_response(
            content="Sure thing!",
            tokens_in=100,
            tokens_out=50,
            model="gpt-4o",
        )
        entries = _read_jsonl(logger_in_tmp.log_path)
        assert len(entries) == 1
        assert entries[0]["type"] == "MODEL_RESPONSE"
        assert entries[0]["content"] == "Sure thing!"
        assert entries[0]["model"] == "gpt-4o"
        assert entries[0]["usage"]["tokens_in"] == 100
        assert entries[0]["usage"]["tokens_out"] == 50

    def test_log_model_response_records_cost(self, logger_in_tmp: ConversationLogger) -> None:
        logger_in_tmp.log_model_response(
            content="Hi",
            tokens_in=1_000_000,
            tokens_out=0,
            model="gpt-4o",
        )
        assert logger_in_tmp.cost_tracker.llm_calls == 1
        assert logger_in_tmp.cost_tracker.total_cost == pytest.approx(2.50)

    def test_log_model_response_with_tool_calls(self, logger_in_tmp: ConversationLogger) -> None:
        tc = [{"name": "read_file", "arguments": {"path": "foo.py"}}]
        logger_in_tmp.log_model_response(
            content="",
            tool_calls=tc,
            tokens_in=50,
            tokens_out=30,
            model="gpt-4o",
        )
        entries = _read_jsonl(logger_in_tmp.log_path)
        assert entries[0]["tool_calls"] == tc

    def test_log_model_response_with_thinking_and_cache(self, logger_in_tmp: ConversationLogger) -> None:
        logger_in_tmp.log_model_response(
            content="deep thought",
            tokens_in=100,
            tokens_out=50,
            thinking_tokens=200,
            cache_read_tokens=80,
            cache_creation_tokens=40,
            model="claude-sonnet-4",
        )
        entries = _read_jsonl(logger_in_tmp.log_path)
        usage = entries[0]["usage"]
        assert usage["thinking_tokens"] == 200
        assert usage["cache_read_tokens"] == 80
        assert usage["cache_creation_tokens"] == 40
        assert usage["cost"] > 0

    def test_log_tool_call(self, logger_in_tmp: ConversationLogger) -> None:
        logger_in_tmp.log_tool_call("write_file", {"path": "x.py", "content": "pass"})
        entries = _read_jsonl(logger_in_tmp.log_path)
        assert entries[0]["type"] == "TOOL_CALL"
        assert entries[0]["tool_name"] == "write_file"
        assert entries[0]["arguments"]["path"] == "x.py"
        assert entries[0]["content"] == "Calling write_file"

    def test_log_tool_call_increments_tool_calls(self, logger_in_tmp: ConversationLogger) -> None:
        logger_in_tmp.log_tool_call("foo", {})
        logger_in_tmp.log_tool_call("bar", {})
        assert logger_in_tmp.cost_tracker.tool_calls == 2

    def test_log_tool_result(self, logger_in_tmp: ConversationLogger) -> None:
        logger_in_tmp.log_tool_result("read_file", "file contents here", duration_ms=42)
        entries = _read_jsonl(logger_in_tmp.log_path)
        assert entries[0]["type"] == "TOOL_RESULT"
        assert entries[0]["tool_name"] == "read_file"
        assert entries[0]["duration_ms"] == 42

    def test_log_tool_result_truncates_content(self, logger_in_tmp: ConversationLogger) -> None:
        long_result = "x" * 1000
        logger_in_tmp.log_tool_result("read_file", long_result)
        entries = _read_jsonl(logger_in_tmp.log_path)
        assert len(entries[0]["content"]) == 500

    def test_log_tool_result_no_duration(self, logger_in_tmp: ConversationLogger) -> None:
        logger_in_tmp.log_tool_result("ls", "a\nb\nc")
        entries = _read_jsonl(logger_in_tmp.log_path)
        assert entries[0]["duration_ms"] is None

    def test_log_error(self, logger_in_tmp: ConversationLogger) -> None:
        logger_in_tmp.log_error("tool_runner", "Something broke")
        entries = _read_jsonl(logger_in_tmp.log_path)
        assert entries[0]["type"] == "ERROR"
        assert entries[0]["source"] == "tool_runner"
        assert entries[0]["content"] == "Something broke"

    def test_log_system(self, logger_in_tmp: ConversationLogger) -> None:
        logger_in_tmp.log_system("Session started")
        entries = _read_jsonl(logger_in_tmp.log_path)
        assert entries[0]["type"] == "SYSTEM"
        assert entries[0]["content"] == "Session started"

    def test_step_index_increments(self, logger_in_tmp: ConversationLogger) -> None:
        logger_in_tmp.log_user_input("msg1")
        logger_in_tmp.log_system("event")
        logger_in_tmp.log_error("src", "err")
        entries = _read_jsonl(logger_in_tmp.log_path)
        assert [e["step_index"] for e in entries] == [1, 2, 3]

    def test_multiple_entries_valid_jsonl(self, logger_in_tmp: ConversationLogger) -> None:
        logger_in_tmp.log_user_input("Hello")
        logger_in_tmp.log_model_response("Hi!", tokens_in=10, tokens_out=5, model="gpt-4o")
        logger_in_tmp.log_tool_call("search", {"q": "test"})
        logger_in_tmp.log_tool_result("search", "results")
        logger_in_tmp.log_error("parser", "bad input")
        logger_in_tmp.log_system("done")

        entries = _read_jsonl(logger_in_tmp.log_path)
        assert len(entries) == 6
        types = [e["type"] for e in entries]
        assert types == [
            "USER_INPUT", "MODEL_RESPONSE", "TOOL_CALL",
            "TOOL_RESULT", "ERROR", "SYSTEM",
        ]

    def test_unicode_content(self, logger_in_tmp: ConversationLogger) -> None:
        logger_in_tmp.log_user_input("日本語テスト 🎉")
        entries = _read_jsonl(logger_in_tmp.log_path)
        assert entries[0]["content"] == "日本語テスト 🎉"

    def test_write_failure_does_not_raise(self, tmp_path: Path) -> None:
        with patch("unjess.conversation_logger._CONVERSATIONS_DIR", tmp_path):
            cl = ConversationLogger(conversation_id="failtest")
            # Make the log file path unwritable by pointing to a non-existent
            # deep directory and removing parent
            cl._log_file = tmp_path / "nonexistent" / "deep" / "file.jsonl"
            # Should not raise
            cl.log_user_input("this will fail silently")


# =====================================================================
# _COST_RATES data integrity
# =====================================================================

class TestCostRatesData:
    """Sanity checks on the _COST_RATES dictionary."""

    def test_all_rates_are_tuples_of_two_floats(self) -> None:
        for model, rates in _COST_RATES.items():
            assert isinstance(rates, tuple), f"{model}: expected tuple"
            assert len(rates) == 2, f"{model}: expected 2 elements"
            assert isinstance(rates[0], (int, float)), f"{model}: input rate not numeric"
            assert isinstance(rates[1], (int, float)), f"{model}: output rate not numeric"

    def test_all_rates_non_negative(self) -> None:
        for model, (inp, out) in _COST_RATES.items():
            assert inp >= 0, f"{model}: negative input rate"
            assert out >= 0, f"{model}: negative output rate"

    def test_known_models_present(self) -> None:
        must_have = ["gpt-4o", "claude-sonnet-4", "gemini-2.5-flash", "o3", "grok-3"]
        for m in must_have:
            assert m in _COST_RATES, f"Missing model: {m}"

    def test_at_least_25_models(self) -> None:
        assert len(_COST_RATES) >= 25


# =====================================================================
# _FREE_PREFIXES data integrity
# =====================================================================

class TestFreePrefixes:
    """Sanity checks on _FREE_PREFIXES."""

    def test_is_list_of_strings(self) -> None:
        assert isinstance(_FREE_PREFIXES, list)
        for p in _FREE_PREFIXES:
            assert isinstance(p, str)
            assert len(p) > 0
