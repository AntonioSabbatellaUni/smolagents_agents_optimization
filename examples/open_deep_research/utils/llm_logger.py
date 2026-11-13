"""Single-point logging of EVERY LiteLLM request/response (revival).

`install()` monkeypatches `litellm.completion` once, so every agent/tool that goes through LiteLLM is covered
(smolagents LiteLLMModel calls `self.client.completion`, and `self.client` is the litellm module).
One JSONL line per call -> <session dir>/llm_calls.jsonl (set with `set_log_path`, or env MALBO_LLM_LOG).

Also injects extra headers from env MALBO_EXTRA_HEADERS (JSON dict; e.g. {"x-opencode-session": "malbo-gaia-20261112"})
and sets a placeholder price (env MALBO_PRICE_IN / MALBO_PRICE_OUT, USD per 1M tokens) for the `cost` field.

Context (agent role / task id) is thread-local: TrackedModel calls `set_agent`, the evaluator calls `set_task`.
Secrets: Authorization/api_key/cookie headers are dropped and any known secret value is scrubbed from the line.
"""
import json
import os
import re
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

_ctx = threading.local()
_lock = threading.Lock()
_state = {"path": None, "installed": False, "counters": defaultdict(int), "task_tokens": {}}
_SECRET_ENVS = ("OPENCODE_API_KEY", "HF_TOKEN", "MALBO_API_KEY", "SERPER_API_KEY", "SERPAPI_API_KEY",
                "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "AA_API_KEY")
_SECRET_RE = re.compile(r"(oc_sk_[A-Za-z0-9_\-]+|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_\-]{20,}|Bearer\s+[A-Za-z0-9._\-]{16,})")
_DROP_HEADERS = {"authorization", "proxy-authorization", "x-api-key", "api-key", "cookie", "set-cookie"}
_RL_PREFIXES = ("x-ratelimit", "ratelimit", "retry-after", "x-opencode", "opencode", "llm_provider-x-ratelimit",
                "llm_provider-ratelimit", "llm_provider-retry-after", "llm_provider-x-opencode", "llm_provider-opencode")


_NO_STOP = set()  # models that reject the `stop` param (learned at run time)


def set_log_path(path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    _state["path"] = p


def set_agent(name):
    _ctx.agent = name


def set_task(task_id):
    _ctx.task_id = task_id


def _secrets():
    return [v for k in _SECRET_ENVS if (v := os.getenv(k)) and len(v) >= 8]


def _scrub(s: str) -> str:
    for v in _secrets():
        s = s.replace(v, "[REDACTED]")
    return _SECRET_RE.sub("[REDACTED]", s)


def _jsonable(o):
    if hasattr(o, "model_dump"):
        try:
            return o.model_dump()
        except Exception:
            pass
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    return str(o)


def _clean_headers(h):
    out = {}
    for k, v in (h or {}).items():
        if str(k).lower() in _DROP_HEADERS:
            continue
        out[str(k)] = v
    return out


def _rate_headers(h):
    return {k: v for k, v in (h or {}).items() if str(k).lower().startswith(_RL_PREFIXES)}


def _price(model):
    pin = float(os.getenv("MALBO_PRICE_IN", "0.10"))
    pout = float(os.getenv("MALBO_PRICE_OUT", "0.50"))
    return pin, pout


def _write(rec):
    line = _scrub(json.dumps(rec, ensure_ascii=False, default=str))
    path = _state["path"] or (Path(os.environ["MALBO_LLM_LOG"]) if os.getenv("MALBO_LLM_LOG") else None)
    if path is None:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with _lock, open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def log_event(kind, **fields):
    """Append one json line (retry notices etc.) to <dir of the llm log>/events.jsonl, thread-safe; also printed."""
    rec = {"ts": datetime.now(timezone.utc).isoformat(), "kind": kind,
           "agent": getattr(_ctx, "agent", None), "task_id": getattr(_ctx, "task_id", None), **fields}
    line = _scrub(json.dumps(rec, ensure_ascii=False, default=str))
    print(f"[retry] {line}")
    path = _state["path"] or (Path(os.environ["MALBO_LLM_LOG"]) if os.getenv("MALBO_LLM_LOG") else None)
    if path is None:
        return
    try:
        with _lock, open(Path(path).parent / "events.jsonl", "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as e:
        print(f"[llm_logger] event write failed: {type(e).__name__}")


def task_tokens(task_id):
    """In-memory per-question token totals {input, output, calls} (thread-safe; independent of the shared TokenTracker)."""
    with _lock:
        return dict(_state["task_tokens"].get(task_id) or {"input": 0, "output": 0, "calls": 0})


def _usage(resp_dict):
    u = (resp_dict or {}).get("usage") or {}
    ptd = u.get("prompt_tokens_details") or {}
    ctd = u.get("completion_tokens_details") or {}
    cached = ptd.get("cached_tokens") or u.get("cache_read_input_tokens") or 0
    return {"prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
            "total_tokens": u.get("total_tokens"), "cached_tokens": cached,
            "reasoning_tokens": ctd.get("reasoning_tokens") or 0, "raw": u}


_STD_KEYS = {"id", "choices", "created", "model", "object", "system_fingerprint", "usage", "service_tier", "provider_specific_fields"}


def _record(kwargs, t0, t1, ttft, resp, err, streamed=False, attempt=1, will_retry=False):
    agent = getattr(_ctx, "agent", None)
    task_id = getattr(_ctx, "task_id", None)
    with _lock:
        _state["counters"][(task_id, agent)] += 1
        step = _state["counters"][(task_id, agent)]
    rd = _jsonable(resp) if resp is not None else None
    choice = ((rd or {}).get("choices") or [{}])[0] if rd else {}
    msg = choice.get("message") or {}
    usage = _usage(rd)
    pin, pout = _price(kwargs.get("model"))
    lat = t1 - t0
    pt, ct = usage["prompt_tokens"] or 0, usage["completion_tokens"] or 0
    hidden = getattr(resp, "_hidden_params", None) or {}
    headers = _clean_headers(hidden.get("additional_headers") if isinstance(hidden, dict) else None)
    status, err_s = (200 if resp is not None else None), None
    if err is not None:
        status = getattr(err, "status_code", None)
        err_s = f"{type(err).__name__}: {err}"[:2000]
        r = getattr(err, "response", None)
        if r is not None and getattr(r, "headers", None) is not None:
            headers = _clean_headers(dict(r.headers))
    rec = {
        "ts_start": datetime.fromtimestamp(t0, timezone.utc).isoformat(),
        "ts_end": datetime.fromtimestamp(t1, timezone.utc).isoformat(),
        "latency_s": round(lat, 4),
        "time_to_first_token_s": None if ttft is None else round(ttft - t0, 4),
        "streaming": streamed,
        "agent": agent, "step_index": step, "task_id": task_id, "attempt": attempt, "will_retry": will_retry,
        "model": kwargs.get("model"), "api_base": kwargs.get("api_base"),
        "request": {"messages": _jsonable(kwargs.get("messages")), "tools": _jsonable(kwargs.get("tools")),
                    "params": {k: _jsonable(v) for k, v in kwargs.items()
                               if k not in ("messages", "tools", "api_key", "extra_headers", "model", "api_base")}},
        "response": {"content": msg.get("content"), "reasoning_content": msg.get("reasoning_content"),
                     "tool_calls": msg.get("tool_calls"), "message": msg,
                     "extra_fields": {k: v for k, v in (rd or {}).items() if k not in _STD_KEYS}},
        "finish_reason": choice.get("finish_reason"),
        "usage": usage,
        "tokens_per_s": round(ct / lat, 3) if lat > 0 and ct else None,
        "cost_placeholder_usd": round((pt * pin + ct * pout) / 1e6, 8),
        "price_per_1m": {"in": pin, "out": pout},
        "http_status": status, "error": err_s,
        "rate_limit_headers": _rate_headers(headers), "response_headers": headers,
    }
    if resp is not None:
        with _lock:
            tt = _state["task_tokens"].setdefault(task_id, {"input": 0, "output": 0, "calls": 0})
            tt["input"] += pt; tt["output"] += ct; tt["calls"] += 1
    try:
        _write(rec)
    except Exception as e:  # never break the run because of logging
        print(f"[llm_logger] write failed: {type(e).__name__}")


def install():
    if _state["installed"]:
        return
    import litellm
    orig = litellm.completion

    def wrapped(*args, **kwargs):
        eh = os.getenv("MALBO_EXTRA_HEADERS")
        if eh and not kwargs.get("extra_headers"):
            try:
                kwargs["extra_headers"] = json.loads(eh)
            except Exception:
                pass
        from . import retry as _retry
        if kwargs.get("model") in _NO_STOP:
            kwargs.pop("stop", None)
        for attempt in range(1, _retry.MAX_TRIES + 1):
            t0 = time.time()
            try:
                resp = orig(*args, **kwargs)
                break
            except Exception as e:
                if type(e).__name__ == "UnsupportedParamsError" and "'stop'" in str(e) and kwargs.get("stop") is not None \
                        and attempt < _retry.MAX_TRIES:
                    # e.g. gpt-6-luna rejects `stop`: remember the model, resend without it (no sleep)
                    _NO_STOP.add(kwargs.get("model"))
                    kwargs.pop("stop", None)
                    _record(kwargs, t0, time.time(), None, None, e, attempt=attempt, will_retry=True)
                    log_event("drop_stop_param", model=kwargs.get("model"))
                    continue
                will = attempt < _retry.MAX_TRIES and _retry.llm_error_retryable(e)
                _record(kwargs, t0, time.time(), None, None, e, attempt=attempt, will_retry=will)
                if not will:
                    raise
                delay = _retry.backoff_delay(attempt, _retry.retry_after_of(e))
                log_event("llm_retry", attempt=attempt, max_tries=_retry.MAX_TRIES, delay_s=round(delay, 2),
                          status=getattr(e, "status_code", None), model=kwargs.get("model"),
                          error=f"{type(e).__name__}: {e}"[:300])
                time.sleep(delay)
        if kwargs.get("stream"):
            return _wrap_stream(resp, kwargs, t0)
        _record(kwargs, t0, time.time(), None, resp, None, attempt=attempt)
        return resp

    litellm.completion = wrapped
    _state["installed"] = True


def _wrap_stream(stream, kwargs, t0):
    import litellm
    chunks, ttft = [], None
    try:
        for ch in stream:
            if ttft is None:
                ttft = time.time()
            chunks.append(ch)
            yield ch
    except Exception as e:
        _record(kwargs, t0, time.time(), ttft, None, e, streamed=True)
        raise
    try:
        resp = litellm.stream_chunk_builder(chunks, messages=kwargs.get("messages"))
    except Exception:
        resp = None
    _record(kwargs, t0, time.time(), ttft, resp, None, streamed=True)
