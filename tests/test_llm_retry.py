"""Tests for unjess.llm.retry — retry logic with exponential backoff."""

from unittest.mock import MagicMock, patch

import pytest

from unjess.llm.retry import (
    DEFAULT_BACKOFF_SECONDS,
    DEFAULT_MAX_RETRIES,
    call_with_retries,
    is_rate_limit_error,
    is_timeout_error,
)


# ---------------------------------------------------------------------------
# is_rate_limit_error
# ---------------------------------------------------------------------------


class TestIsRateLimitError:
    """Tests for rate-limit error classification."""

    def test_class_named_rate_limit_error(self) -> None:
        class RateLimitError(Exception):
            pass

        assert is_rate_limit_error(RateLimitError("hit limit"))

    def test_message_contains_rate(self) -> None:
        assert is_rate_limit_error(Exception("rate limited, try later"))

    def test_message_contains_quota(self) -> None:
        assert is_rate_limit_error(Exception("quota exceeded"))

    def test_message_contains_429(self) -> None:
        assert is_rate_limit_error(Exception("HTTP 429 Too Many Requests"))

    def test_unrelated_error_returns_false(self) -> None:
        assert not is_rate_limit_error(ValueError("bad value"))

    def test_empty_message_returns_false(self) -> None:
        assert not is_rate_limit_error(Exception(""))

    def test_case_insensitive_message(self) -> None:
        assert is_rate_limit_error(Exception("RATE LIMIT EXCEEDED"))

    def test_class_name_exact_match_only(self) -> None:
        class NotRateLimitError(Exception):
            pass

        assert not is_rate_limit_error(NotRateLimitError("nope"))

    def test_message_with_quota_substring(self) -> None:
        assert is_rate_limit_error(Exception("Resource quota depleted"))


# ---------------------------------------------------------------------------
# is_timeout_error
# ---------------------------------------------------------------------------


class TestIsTimeoutError:
    """Tests for timeout error classification."""

    def test_class_named_with_timeout(self) -> None:
        class APITimeoutError(Exception):
            pass

        assert is_timeout_error(APITimeoutError("took too long"))

    def test_class_named_timeout_error(self) -> None:
        class TimeoutError(Exception):
            pass

        assert is_timeout_error(TimeoutError("timed out"))

    def test_message_contains_timeout(self) -> None:
        assert is_timeout_error(Exception("Connection timeout after 30s"))

    def test_unrelated_error_returns_false(self) -> None:
        assert not is_timeout_error(ValueError("bad value"))

    def test_empty_message_returns_false(self) -> None:
        assert not is_timeout_error(Exception(""))

    def test_case_insensitive_class_name(self) -> None:
        class APITIMEOUTERROR(Exception):
            pass

        assert is_timeout_error(APITIMEOUTERROR("oops"))

    def test_case_insensitive_message(self) -> None:
        assert is_timeout_error(Exception("TIMEOUT waiting for response"))


# ---------------------------------------------------------------------------
# call_with_retries
# ---------------------------------------------------------------------------


class TestCallWithRetries:
    """Tests for the call_with_retries retry wrapper."""

    def test_successful_call_returns_result(self) -> None:
        fn = MagicMock(return_value="ok")
        result = call_with_retries(fn)
        assert result == "ok"
        fn.assert_called_once()

    def test_passes_args_and_kwargs(self) -> None:
        fn = MagicMock(return_value=42)
        result = call_with_retries(fn, "a", "b", key="val")
        assert result == 42
        fn.assert_called_once_with("a", "b", key="val")

    @patch("unjess.llm.retry.time.sleep")
    def test_retries_on_rate_limit_then_succeeds(self, mock_sleep: MagicMock) -> None:
        class RateLimitError(Exception):
            pass

        fn = MagicMock(side_effect=[RateLimitError("429"), "success"])
        result = call_with_retries(fn, max_retries=3)
        assert result == "success"
        assert fn.call_count == 2
        mock_sleep.assert_called_once_with(DEFAULT_BACKOFF_SECONDS[0])

    @patch("unjess.llm.retry.time.sleep")
    def test_rate_limit_backoff_delays(self, mock_sleep: MagicMock) -> None:
        class RateLimitError(Exception):
            pass

        fn = MagicMock(side_effect=[
            RateLimitError("429"),
            RateLimitError("429"),
            "done",
        ])
        result = call_with_retries(fn, max_retries=3)
        assert result == "done"
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(DEFAULT_BACKOFF_SECONDS[0])
        mock_sleep.assert_any_call(DEFAULT_BACKOFF_SECONDS[1])

    @patch("unjess.llm.retry.time.sleep")
    def test_exhausts_max_retries_raises_last_error(self, mock_sleep: MagicMock) -> None:
        class RateLimitError(Exception):
            pass

        fn = MagicMock(side_effect=RateLimitError("429"))
        with pytest.raises(RateLimitError, match="429"):
            call_with_retries(fn, max_retries=3)
        assert fn.call_count == 3

    def test_non_retryable_error_raises_immediately(self) -> None:
        fn = MagicMock(side_effect=ValueError("bad input"))
        with pytest.raises(ValueError, match="bad input"):
            call_with_retries(fn, max_retries=3)
        fn.assert_called_once()

    def test_skip_rate_limit_retry_raises_immediately(self) -> None:
        class RateLimitError(Exception):
            pass

        fn = MagicMock(side_effect=RateLimitError("quota"))
        with pytest.raises(RateLimitError, match="quota"):
            call_with_retries(fn, max_retries=3, skip_rate_limit_retry=True)
        fn.assert_called_once()

    @patch("unjess.llm.retry.time.sleep")
    def test_timeout_retries_once_then_raises(self, mock_sleep: MagicMock) -> None:
        class APITimeoutError(Exception):
            pass

        fn = MagicMock(side_effect=APITimeoutError("timed out"))
        with pytest.raises(APITimeoutError, match="timed out"):
            call_with_retries(fn, max_retries=3)
        # Attempt 0: timeout → retry (attempt < 1 passes)
        # Attempt 1: timeout → not attempt < 1, so raises
        assert fn.call_count == 2
        mock_sleep.assert_not_called()

    @patch("unjess.llm.retry.time.sleep")
    def test_timeout_retries_once_then_succeeds(self, mock_sleep: MagicMock) -> None:
        class APITimeoutError(Exception):
            pass

        fn = MagicMock(side_effect=[APITimeoutError("timed out"), "ok"])
        result = call_with_retries(fn, max_retries=3)
        assert result == "ok"
        assert fn.call_count == 2

    @patch("unjess.llm.retry.time.sleep")
    def test_custom_backoff_seconds(self, mock_sleep: MagicMock) -> None:
        class RateLimitError(Exception):
            pass

        fn = MagicMock(side_effect=[RateLimitError("429"), "ok"])
        result = call_with_retries(fn, max_retries=3, backoff_seconds=[10, 20, 30])
        assert result == "ok"
        mock_sleep.assert_called_once_with(10)

    @patch("unjess.llm.retry.time.sleep")
    def test_backoff_clamps_to_last_element(self, mock_sleep: MagicMock) -> None:
        class RateLimitError(Exception):
            pass

        fn = MagicMock(side_effect=[
            RateLimitError("429"),
            RateLimitError("429"),
            RateLimitError("429"),
            RateLimitError("429"),
        ])
        # backoff_seconds has 2 elements, 4 retries
        with pytest.raises(RateLimitError):
            call_with_retries(fn, max_retries=4, backoff_seconds=[5, 10])
        # Attempts 0,1,2,3 — sleeps at 0,1,2,3 but last is clamped
        assert mock_sleep.call_count == 4
        # attempt 0 → index min(0,1)=0 → 5
        # attempt 1 → index min(1,1)=1 → 10
        # attempt 2 → index min(2,1)=1 → 10
        # attempt 3 → index min(3,1)=1 → 10
        mock_sleep.assert_any_call(5)
        mock_sleep.assert_any_call(10)

    def test_max_retries_one(self) -> None:
        fn = MagicMock(return_value="first")
        result = call_with_retries(fn, max_retries=1)
        assert result == "first"
        fn.assert_called_once()

    @patch("unjess.llm.retry.time.sleep")
    def test_rate_limit_by_message_not_class(self, mock_sleep: MagicMock) -> None:
        fn = MagicMock(side_effect=[Exception("429 Too Many Requests"), "ok"])
        result = call_with_retries(fn, max_retries=3)
        assert result == "ok"
        assert fn.call_count == 2

    def test_default_max_retries_constant(self) -> None:
        assert DEFAULT_MAX_RETRIES == 3

    def test_default_backoff_seconds_constant(self) -> None:
        assert DEFAULT_BACKOFF_SECONDS == [1, 2, 4]

    @patch("unjess.llm.retry.time.sleep")
    def test_none_backoff_uses_default(self, mock_sleep: MagicMock) -> None:
        class RateLimitError(Exception):
            pass

        fn = MagicMock(side_effect=[RateLimitError("429"), "ok"])
        call_with_retries(fn, max_retries=2, backoff_seconds=None)
        mock_sleep.assert_called_once_with(1)  # DEFAULT_BACKOFF_SECONDS[0]
