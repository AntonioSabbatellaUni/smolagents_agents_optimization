"""Revival: run the full Deep Research team on a few fixed GAIA tasks with ONE model for all roles (e.g. OpenCode beam-free).
Env: OPENCODE_API_KEY, MALBO_TASK_IDS_FILE (json), MALBO_API_BASE, MALBO_MODEL (default openai/beam-free),
     MALBO_EXTRA_HEADERS (json), MALBO_MANAGER_MAX_STEPS, MALBO_SEARCH_MAX_STEPS, MALBO_FLATTEN (1 = plain-string messages), HF_TOKEN.
Writes <session>/llm_calls.jsonl (every LLM call) and <session>/task_results.json (answer vs gold)."""
import json, os, time
from pathlib import Path
from loader import load_agent_models
from utils import EnhancedRunManager
from utils.llm_logger import install
from run_gaia_subset_enhanced import load_gaia_dataset, answer_single_question_with_tracking
from optimization_interface import select_by_task_ids
from scripts.gaia_scorer import question_scorer

install()
model = os.getenv("MALBO_MODEL", "openai/beam-free")
cfg = {r: {"model_class": "LiteLLMModel", "model_id": model, "api_base": os.environ["MALBO_API_BASE"],
           "api_key": os.environ["OPENCODE_API_KEY"], "max_tokens": int(os.getenv("MALBO_MAX_TOKENS", 4096)),
           # iceman-contributor-free answers 422 "Endpoint is unavailable" to assistant messages whose content is a list of parts
           **({"flatten_messages_as_text": True} if os.getenv("MALBO_FLATTEN") == "1" else {})}
       for r in ["manager", "search_agent", "text_inspector", "visual_qa", "reformulator"]}
models = load_agent_models(cfg)
rm = EnhancedRunManager()
sess = rm.setup_session("beam", f"gaia_beam_{time.strftime('%Y%m%d_%H%M%S')}")
rm.start_run()
rows = select_by_task_ids(load_gaia_dataset(use_raw_dataset=True, set_to_run="validation"), os.environ["MALBO_TASK_IDS_FILE"])
out = []
for i, ex in enumerate(rows):
    t0 = time.time()
    r = answer_single_question_with_tracking(ex, models, rm, {"max_retries": 0, "base_delay": 1.0, "backoff_factor": 2.0}, i)
    r["wall_s"] = round(time.time() - t0, 1)
    try:
        r["gaia_scorer_correct"] = bool(question_scorer(str(r["prediction"]), str(r["true_answer"])))
    except Exception as e:
        r["gaia_scorer_correct"] = f"err {e}"
    r.pop("intermediate_steps", None)
    out.append(r)
    print("RESULT", json.dumps({k: r.get(k) for k in ("task_id", "prediction", "true_answer", "is_correct", "gaia_scorer_correct", "wall_s", "agent_error")}), flush=True)
    json.dump(out, open(Path(sess) / "task_results.json", "w"), indent=1, default=str)
print("SESSION", sess)
