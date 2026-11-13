"""Retry with exponential backoff + jitter (revival), shared by the LiteLLM wrapper and the search tool.
Every retry is logged: printed and appended (one json line) to <session>/events.jsonl via llm_logger.log_event."""
import os
import random
import time

MAX_TRIES = int(os.getenv("MALBO_MAX_TRIES", "5"))
BASE_DELAY = float(os.getenv("MALBO_RETRY_BASE_S", "2.0"))
MAX_DELAY = float(os.getenv("MALBO_RETRY_MAX_S", "60.0"))

RETRY_STATUS = {429, 500, 502, 503, 504}
_RETRY_TEXT = ("ratelimit", "rate limit", "rate-limit", "too many requests", "202 ratelimit", "blocked", "forbidden",
               "temporarily unavailable", "service unavailable", "bad gateway", "gateway timeout", "timed out", "timeout",
               "connection", "captcha", "anomaly", "202")


def backoff_delay(attempt: int, retry_after=None) -> float:
    """attempt = 1 for the first retry. base * 2^(attempt-1) with +-50% jitter (>= retry_after if the server sent one)."""
    d = min(MAX_DELAY, BASE_DELAY * (2 ** (attempt - 1)))
    d = d * (0.5 + random.random())
    if retry_after:
        try:
            d = max(d, min(float(retry_after), MAX_DELAY))
        except (TypeError, ValueError):
            pass
    return d


def llm_error_retryable(err) -> bool:
    """HTTP 429 / 5xx (or the equivalent LiteLLM exception types / connection errors)."""
    st = getattr(err, "status_code", None)
    if isinstance(st, int):
        return st in RETRY_STATUS
    name = type(err).__name__
    return name in ("RateLimitError", "ServiceUnavailableError", "InternalServerError", "BadGatewayError", "Timeout",
                    "APIConnectionError", "APITimeoutError", "APIConnectionError")


def search_error_retryable(err) -> bool:
    s = f"{type(err).__name__} {err}".lower()
    if "no results found" in s:  # an empty result is a normal answer for the agent, not a block
        return False
    return any(t in s for t in _RETRY_TEXT)


def retry_after_of(err):
    r = getattr(err, "response", None)
    h = getattr(r, "headers", None)
    if h is not None:
        try:
            return h.get("retry-after") or h.get("Retry-After")
        except Exception:
            return None
    return None


def call_with_retry(fn, kind: str, retryable, max_tries: int = None, **ctx):
    """Run fn(); on a retryable exception sleep (backoff+jitter) and retry, at most max_tries calls in total."""
    from . import llm_logger
    max_tries = max_tries or MAX_TRIES
    for attempt in range(1, max_tries + 1):
        try:
            return fn()
        except Exception as e:
            if attempt >= max_tries or not retryable(e):
                raise
            delay = backoff_delay(attempt, retry_after_of(e))
            llm_logger.log_event(kind, attempt=attempt, max_tries=max_tries, delay_s=round(delay, 2),
                                 status=getattr(e, "status_code", None), error=f"{type(e).__name__}: {e}"[:300], **ctx)
            time.sleep(delay)
