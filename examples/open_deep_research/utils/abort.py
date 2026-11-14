"""Process-wide 'pause' switch. `trigger()` records the first infra problem (quota / auth / provider down / search blocked) and every
later LLM call (`check()`) raises InfraAbort, so running questions stop within one step. Those questions land in `failed_tasks`, are not
counted, and the evaluation is paused (EvaluationPaused) and resumed later from the question cache.
kind: 'quota' (model-specific) | 'account_limit' (every model) | 'infra' (global)."""
import threading
from datetime import datetime, timezone

_lock = threading.Lock()
_state = {"reason": None, "failed_tasks": set()}

_QUOTA_TEXT = ("quota", "usage limit", "limit exceeded", "exceeded your", "insufficient", "credit", "billing", "allowance",
               "monthly limit", "spend limit", "rate limit", "ratelimit", "too many requests")


class InfraAbort(Exception):
    """Raised inside a question when the evaluation has been paused."""


class EvaluationPaused(Exception):
    """Raised by evaluate_configuration: the evaluation stopped early; completed questions are in the question cache."""

    def __init__(self, reason: dict, n_done: int, n_total: int, session_dir=None):
        self.reason = dict(reason or {})
        self.n_done, self.n_total, self.session_dir = n_done, n_total, session_dir
        super().__init__(f"evaluation paused ({self.reason.get('kind')}, model={self.reason.get('model')}): "
                         f"{self.reason.get('error', '')[:200]} - {n_done}/{n_total} questions kept")


def reset():
    with _lock:
        _state["reason"] = None
        _state["failed_tasks"] = set()


def resets_in_s(err):
    """'... usage limit reached (5-hour, resets in 4h 53m) ...' -> seconds (None if absent)."""
    import re
    m = re.search(r"resets in\s*(?:(\d+)\s*d)?\s*(?:(\d+)\s*h)?\s*(?:(\d+)\s*m)?", f"{err}", re.I)
    if not m or not any(m.groups()):
        return None
    d, h, mi = (int(x or 0) for x in m.groups())
    return d * 86400 + h * 3600 + mi * 60


def classify(err) -> str:
    st = getattr(err, "status_code", None)
    s = f"{type(err).__name__} {err}".lower()
    if "go usage limit reached" in s:   # OpenCode Go account-wide window (5-hour / weekly / monthly): every model is blocked
        return "account_limit"
    if st in (402, 429) or type(err).__name__ == "RateLimitError" or any(t in s for t in _QUOTA_TEXT):
        return "quota"
    return "infra"


def is_quota_text(err) -> bool:
    """True when the error text says the quota/cap is exhausted (no point in retrying)."""
    s = f"{err}".lower()
    return any(t in s for t in ("quota", "usage limit", "limit exceeded", "exceeded your", "insufficient", "credit", "billing",
                                "monthly limit", "spend limit"))


def trigger(source: str, model, err, task_id=None, kind=None):
    with _lock:
        if task_id is not None:
            _state["failed_tasks"].add(task_id)
        if _state["reason"] is None:
            _state["reason"] = {"kind": kind or classify(err), "source": source, "model": model, "resets_in_s": resets_in_s(err),
                                "status": getattr(err, "status_code", None), "error": f"{type(err).__name__}: {err}"[:500],
                                "task_id": task_id, "ts": datetime.now(timezone.utc).isoformat()}
            print(f"⏸️  PAUSE triggered by {source} ({_state['reason']['kind']}, model={model}): {_state['reason']['error'][:200]}")


def mark_failed(task_id):
    if task_id is not None:
        with _lock:
            _state["failed_tasks"].add(task_id)


def check(task_id=None):
    if _state["reason"] is not None:
        mark_failed(task_id)
        raise InfraAbort(f"evaluation paused: {_state['reason']['error'][:200]}")


def reason():
    return _state["reason"]


def failed_tasks():
    with _lock:
        return set(_state["failed_tasks"])
