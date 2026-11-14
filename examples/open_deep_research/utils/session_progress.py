"""Live progress of ONE evaluation (30 questions) for tools/malbo_status.py (stdlib only, thread-safe).
Writes in the session dir: progress.json (heartbeat: running / done counts) and questions.jsonl (one line per finished question).
If env MALBO_OUTPUT_DIR is set, also <output_dir>/current_session.json -> {session_dir, started_at} so the monitor finds the session."""
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


def _now():
    return datetime.now(timezone.utc).isoformat()


class SessionProgress:
    def __init__(self, session_dir, run_name, n_questions, parallel):
        self.dir = Path(session_dir).resolve()
        self.lock = threading.Lock()
        self.state = {"run_name": run_name, "session_dir": str(self.dir), "started_at": _now(), "n_questions": n_questions,
                      "parallel": parallel, "running": {}, "done": 0, "correct": 0, "errors": 0, "finished": False}
        self._write()
        out = os.getenv("MALBO_OUTPUT_DIR")
        if out:
            try:
                p = Path(out) / "current_session.json"
                tmp = p.with_suffix(".tmp")
                tmp.write_text(json.dumps({"session_dir": str(self.dir), "run_name": run_name, "started_at": self.state["started_at"]}))
                os.replace(tmp, p)
            except Exception as e:
                print(f"[session_progress] could not write current_session.json: {type(e).__name__}")

    def _write(self):
        self.state["updated_at"] = _now()
        try:
            tmp = self.dir / "progress.json.tmp"
            tmp.write_text(json.dumps(self.state, indent=1))
            os.replace(tmp, self.dir / "progress.json")
        except Exception as e:
            print(f"[session_progress] write failed: {type(e).__name__}")

    def start(self, index, task_id):
        with self.lock:
            self.state["running"][task_id] = {"index": index, "started_at": _now()}
            self._write()

    def finish(self, index, task_id, result):
        rec = {"index": index, "task_id": task_id, "level": result.get("task_level"), "start": result.get("start_time"),
               "end": result.get("end_time"), "duration_s": result.get("duration_seconds"), "ts": _now(),
               "is_correct": bool(result.get("is_correct")), "attempts": result.get("attempts_made"),
               "agent_error": result.get("agent_error"), "token_counts": result.get("token_counts"),
               "prediction": (str(result.get("prediction"))[:200] if result.get("prediction") is not None else None),
               "true_answer": str(result.get("true_answer"))[:200]}
        with self.lock:
            st = self.state
            started = (st["running"].pop(task_id, None) or {}).get("started_at")
            if started and not rec["duration_s"]:
                rec["duration_s"] = (datetime.fromisoformat(_now()) - datetime.fromisoformat(started)).total_seconds()
            st["done"] += 1
            st["correct"] += int(rec["is_correct"])
            st["errors"] += int(bool(rec["agent_error"]))
            try:
                with open(self.dir / "questions.jsonl", "a") as f:
                    f.write(json.dumps(rec, default=str) + "\n")
            except Exception as e:
                print(f"[session_progress] questions.jsonl write failed: {type(e).__name__}")
            self._write()

    def close(self):
        with self.lock:
            self.state["finished"] = True
            self.state["running"] = {}
            self._write()
