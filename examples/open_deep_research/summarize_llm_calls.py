#!/usr/bin/env python3
"""Aggregate llm_calls.jsonl per task and per agent: requests, tokens, latency, tok/s, cost, errors.
Usage: python summarize_llm_calls.py <llm_calls.jsonl | session dir> [--json]"""
import json, sys
from collections import defaultdict
from pathlib import Path

def load(p):
    p = Path(p)
    if p.is_dir():
        p = p / "llm_calls.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]

def agg(rows):
    n = len(rows)
    lat = sum(r["latency_s"] for r in rows)
    pt = sum((r["usage"]["prompt_tokens"] or 0) for r in rows)
    ct = sum((r["usage"]["completion_tokens"] or 0) for r in rows)
    ca = sum((r["usage"]["cached_tokens"] or 0) for r in rows)
    rt = sum((r["usage"]["reasoning_tokens"] or 0) for r in rows)
    return {"requests": n, "errors": sum(1 for r in rows if r.get("error")), "http_429": sum(1 for r in rows if r.get("http_status") == 429),
            "prompt_tok": pt, "completion_tok": ct, "cached_tok": ca, "reasoning_tok": rt,
            "latency_total_s": round(lat, 1), "latency_mean_s": round(lat / n, 2) if n else 0,
            "tok_per_s": round(ct / lat, 2) if lat else 0, "cost_usd": round(sum(r["cost_placeholder_usd"] for r in rows), 6)}

def table(title, groups):
    print(f"\n## {title}")
    cols = ["requests", "errors", "http_429", "prompt_tok", "completion_tok", "cached_tok", "reasoning_tok", "latency_total_s", "latency_mean_s", "tok_per_s", "cost_usd"]
    print("| key | " + " | ".join(cols) + " |\n|" + "---|" * (len(cols) + 1))
    for k, rows in sorted(groups.items(), key=lambda kv: str(kv[0])):
        a = agg(rows)
        print(f"| {k} | " + " | ".join(str(a[c]) for c in cols) + " |")

if __name__ == "__main__":
    rows = load(sys.argv[1])
    by_task, by_agent, by_ta = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in rows:
        by_task[r.get("task_id")].append(r); by_agent[r.get("agent")].append(r); by_ta[f"{r.get('task_id')} / {r.get('agent')}"].append(r)
    if "--json" in sys.argv:
        print(json.dumps({"total": agg(rows), "per_task": {k: agg(v) for k, v in by_task.items()}, "per_agent": {k: agg(v) for k, v in by_agent.items()}}, indent=1))
    else:
        table("TOTAL", {"all": rows}); table("Per task", by_task); table("Per agent", by_agent); table("Per task x agent", by_ta)
        rl = {}
        for r in rows:
            rl.update(r.get("rate_limit_headers") or {})
        print("\nRate-limit/allowance headers seen:", json.dumps(rl) if rl else "none")
        ex = [r["response"]["extra_fields"] for r in rows if r["response"].get("extra_fields")]
        print("Extra response fields (e.g. opencode allowance), last:", json.dumps(ex[-1])[:600] if ex else "none")
