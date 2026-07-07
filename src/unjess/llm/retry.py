"""Shared retry logic for LLM providers — exponential backoff with error classification."""

import logging
import time
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Default retry configuration
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_SECONDS = [1, 2, 4]


def is_rate_limit_error(exc: Exception) -> bool:
    """Check if an exception is a rate-limit / quota error.

    Works across all provider SDKs by checking both exception types
    and error message contents.
    """
    # Check by class name (covers openai.RateLimitError, anthropic.RateLimitError)
    if type(exc).__name__ == "RateLimitError":
        return True

    # Check by error message (covers Google and generic cases)
    error_str = str(exc).lower()
    return "rate" in error_str or "quota" in error_str or "429" in error_str


def is_timeout_error(exc: Exception) -> bool:
    """Check if an exception is a timeout error.

    Works across all provider SDKs by checking both exception types
    and error message contents.
    """
    # Check by class name (covers openai.APITimeoutError, anthropic.APITimeoutError)
    if "timeout" in type(exc).__name__.lower():
        return True

    return "timeout" in str(exc).lower()


def is_server_error(exc: Exception) -> bool:
    """Check if an exception is a transient server error (5xx).

    These are safe to retry — the server had an internal hiccup.
    """
    error_str = str(exc).lower()
    if type(exc).__name__ in ("InternalServerError", "APIStatusError"):
        return True
    return any(code in error_str for code in ("500", "502", "503", "internal"))


def call_with_retries(
    fn: Callable[..., T],
    *args: Any,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_seconds: list[int] | None = None,
    skip_rate_limit_retry: bool = False,
    **kwargs: Any,
) -> T:
    """Call a function with retry logic for transient errors.

    Retry policy:
        - Rate-limit / quota errors → retry with exponential backoff,
          UNLESS ``skip_rate_limit_retry`` is True (then raise immediately
          so the caller can handle rotation).
        - Timeout errors → retry once (on the first attempt only).
        - All other errors → raise immediately.

    Args:
        fn: The callable to invoke.
        *args: Positional arguments forwarded to *fn*.
        max_retries: Maximum number of attempts (default 3).
        backoff_seconds: Backoff delays per attempt (default [1, 2, 4]).
        skip_rate_limit_retry: If True, rate-limit errors are raised
            immediately without retry.  Use when the caller (e.g. the
            router) handles rotation itself.
        **kwargs: Keyword arguments forwarded to *fn*.

    Returns:
        The return value of *fn*.

    Raises:
        The last exception encountered if all retries are exhausted.
    """
    if backoff_seconds is None:
        backoff_seconds = list(DEFAULT_BACKOFF_SECONDS)

    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if is_rate_limit_error(exc):
                if skip_rate_limit_retry:
                    raise  # let the router handle key rotation
                last_error = exc
                wait = backoff_seconds[min(attempt, len(backoff_seconds) - 1)]
                logger.warning(
                    "Rate limited (attempt %d/%d), waiting %ds...",
                    attempt + 1,
                    max_retries,
                    wait,
                )
                time.sleep(wait)
            elif is_timeout_error(exc) and attempt < 1:
                last_error = exc
                logger.warning("API timeout, retrying once...")
                continue
            elif is_server_error(exc):
                last_error = exc
                wait = backoff_seconds[min(attempt, len(backoff_seconds) - 1)]
                logger.warning(
                    "Server error (attempt %d/%d), waiting %ds: %s",
                    attempt + 1,
                    max_retries,
                    wait,
                    exc,
                )
                time.sleep(wait)
            else:
                raise

    raise last_error  # type: ignore[misc]
