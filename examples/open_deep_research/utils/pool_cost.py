"""Per-evaluation cost from the per-call log <session>/llm_calls.jsonl, priced with the pool row of each role.

aa_cost     (the MALBO cost objective) = sum_calls (input_non_cached*AA_in + cache_read*AA_cache + output*AA_out)/1e6
go_expected = same tokens at Go prices, x Go_peak_multiplier on weekdays 01-04 / 06-10 UTC (by ts_start)
go_list_cost / go_allowance = sums of usage.raw.opencode.list_cost / allowance reported by Go.
opencode.tokens.input EXCLUDES cache reads and cache writes (cache_write_5m/1h, billed by Go at go_cache_write, or go_in if the model has no
write price; AA has no write price, so writes count as plain input at aa_in).
`prices[role]` = {"name", "aa_in", "aa_cache", "aa_out", "go_in", "go_cache", "go_out", "go_peak"} (USD per 1M tokens).
"""
import json
from datetime import datetime
from pathlib import Path

PRICE_KEYS = ("aa_in", "aa_cache", "aa_out", "go_in", "go_cache", "go_cache_write", "go_out", "go_peak")


def is_peak(ts_start: str) -> bool:
    try:
        t = datetime.fromisoformat(ts_start)
    except Exception:
        return False
    return t.weekday() < 5 and (1 <= t.hour < 4 or 6 <= t.hour < 10)


def _tokens(usage: dict):
    raw = (usage or {}).get("raw") or {}
    oc = raw.get("opencode") or {}
    tk = oc.get("tokens")
    if tk:
        return (int(tk.get("input") or 0), int(tk.get("cache_read") or 0), int(tk.get("output") or 0), oc,
                int(tk.get("cache_write_5m") or 0) + int(tk.get("cache_write_1h") or 0))
    cached = int((usage or {}).get("cached_tokens") or 0)
    return max(int((usage or {}).get("prompt_tokens") or 0) - cached, 0), cached, int((usage or {}).get("completion_tokens") or 0), oc, 0


def evaluation_costs(jsonl_path, prices: dict, tol: float = 0.02) -> dict:
    """Raises FileNotFoundError if the log is missing."""
    p = Path(jsonl_path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    out = {"aa_cost": 0.0, "go_expected": 0.0, "go_list_cost": 0.0, "go_allowance": 0.0,
           "n_calls": 0, "n_unpriced": 0, "per_role": {}}
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("error") or not r.get("usage"):
            continue
        role = r.get("agent")
        pr = prices.get(role)
        if not pr:
            out["n_unpriced"] += 1
            continue
        i, c, o, oc, w = _tokens(r["usage"])
        aa = ((i + w) * pr["aa_in"] + c * pr["aa_cache"] + o * pr["aa_out"]) / 1e6
        go = (i * pr["go_in"] + c * pr["go_cache"] + w * (pr.get("go_cache_write") or pr["go_in"]) + o * pr["go_out"]) / 1e6
        if is_peak(r.get("ts_start", "")):
            go *= float(pr.get("go_peak") or 1)
        lc, al = float(oc.get("list_cost") or 0), float(oc.get("allowance") or 0)
        out["n_calls"] += 1
        out["aa_cost"] += aa; out["go_expected"] += go; out["go_list_cost"] += lc; out["go_allowance"] += al
        d = out["per_role"].setdefault(role, {"model": pr.get("name"), "calls": 0, "input": 0, "cache_read": 0, "cache_write": 0, "output": 0, "aa_cost": 0.0})
        d["calls"] += 1; d["input"] += i; d["cache_read"] += c; d["cache_write"] += w; d["output"] += o; d["aa_cost"] += aa
    ref = out["go_list_cost"]
    out["mismatch"] = bool(abs(out["go_expected"] - ref) > tol * max(abs(ref), 1e-12)) if ref else bool(out["go_expected"] > 1e-9)
    return out
