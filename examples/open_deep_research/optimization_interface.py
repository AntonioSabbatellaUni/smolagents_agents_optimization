"""
Programmatic interface for GAIA benchmark evaluation.
Designed to be used by optimization loops (BoTorch, Optuna, etc.).
"""

import json
import os
import threading
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from datetime import datetime

# Import from existing battle-tested modules
from loader import load_agent_models
from utils import EnhancedRunManager
from utils.cost_estimator import CostEstimator
from utils.pool_cost import evaluation_costs, merge_reports
from utils.subset_evaluation import create_deterministic_subset, calculate_final_metrics
from run_gaia_subset_enhanced import load_gaia_dataset, answer_single_question_with_tracking


def select_by_task_ids(eval_ds, path: str):
    """Return the dataset rows whose task_id is listed in `path`, in file order (raises if any id is missing)."""
    txt = Path(path).read_text()
    try:
        data = json.loads(txt)
        items = data["tasks"] if isinstance(data, dict) else data
        ids = [t["task_id"] if isinstance(t, dict) else t for t in items]
    except json.JSONDecodeError:
        ids = [l.strip() for l in txt.splitlines() if l.strip()]
    rows = {r["task_id"]: r for r in eval_ds}
    missing = [i for i in ids if i not in rows]
    if missing:
        raise ValueError(f"{len(missing)} task_ids from {path} not in dataset: {missing[:3]}...")
    print(f"📋 Using {len(ids)} fixed task_ids from {path}")
    return [rows[i] for i in ids]


_QCACHE_LOCK = threading.Lock()
QUESTION_TIMEOUT_S = 1800   # counted as a wrong answer, like an agent that never answers


class QuestionCache:
    """<MALBO_QCACHE_DIR>/<key>.jsonl: one line per completed question {task_id, result, cost_report, session_dir}, key = hash of the
    role -> (model_id, reasoning_effort, pool option) mapping + the task list. Disabled (no-op) when MALBO_QCACHE_DIR is unset."""

    def __init__(self, agent_model_configs, task_ids):
        import hashlib
        sig = {r: [c.get("model_id"), c.get("reasoning_effort"), (c.get("pool") or {}).get("name")]
               for r, c in sorted(agent_model_configs.items()) if isinstance(c, dict)}
        key = hashlib.sha1(json.dumps([sig, list(task_ids)], sort_keys=True).encode()).hexdigest()[:16]
        d = os.getenv("MALBO_QCACHE_DIR")
        self.path = Path(d) / f"{key}.jsonl" if d else None
        self.sig, self.task_ids = sig, set(task_ids)

    def load(self):
        out = {}
        if self.path is None or not self.path.exists():
            return out
        for line in self.path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec.get("task_id") in self.task_ids:
                    out[rec["task_id"]] = rec
        return out

    def save(self, results, per_q_costs, session_dir):
        if self.path is None or not results:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.with_suffix(".config.json").exists():
            self.path.with_suffix(".config.json").write_text(json.dumps(self.sig, indent=1))
        with _QCACHE_LOCK, open(self.path, "a") as f:   # questions finish in parallel threads
            for r in results:
                f.write(json.dumps({"task_id": r["task_id"], "result": r, "cost_report": per_q_costs.get(r["task_id"]) or {},
                                    "session_dir": session_dir, "ts": datetime.now().isoformat()}, default=str) + "\n")


class GaiaOptimizationInterface:
    """Clean, reusable interface for GAIA benchmark evaluation."""
    
    def __init__(self, base_config_path: Optional[str] = None):
        self.base_config = {}
        if base_config_path:
            import yaml
            with open(base_config_path, 'r') as f:
                self.base_config = yaml.safe_load(f)
    
    def evaluate_configuration(
        self, agent_model_configs: Dict[str, Any], dataset_limits: Optional[Dict[str, int]] = None,
        run_name: Optional[str] = None, concurrency: Optional[int] = None, random_seed: int = 42,
        save_detailed_results: bool = False
    ) -> Tuple[float, float, Path]:
        """
        Evaluate a specific model configuration on GAIA benchmark.
        
        Args:
            agent_model_configs: Model configuration for each agent role
            dataset_limits: Number of questions per task level
            run_name: Unique identifier for this evaluation
            concurrency: Number of questions answered at the same time; None -> env MALBO_PARALLEL_TASKS (default 1)
            random_seed: Seed for reproducible dataset subset
            save_detailed_results: Whether to save detailed logs
            
        Returns:
            Tuple of (accuracy_percentage, total_cost_usd, session_directory_path)
        """
        
        # Generate defaults if not provided
        if run_name is None:
            run_name = f"opt_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if dataset_limits is None:
            dataset_limits = self.base_config.get('dataset_limits', {'task_1': 3, 'task_2': 0, 'task_3': 0})
        
        if concurrency is None:
            concurrency = int(os.getenv("MALBO_PARALLEL_TASKS") or 1)
        concurrency = max(1, int(concurrency))
        print(f"🚀 Starting evaluation: {run_name} (parallel questions: {concurrency})")
        
        try:
            # Load models and setup tracking
            models = load_agent_models(agent_model_configs)  # strips the non-model `pool` key
            run_manager = EnhancedRunManager()
            session_dir = run_manager.setup_session("optimization", f"subset_{run_name}")
            run_manager.start_run()
            
            # Load dataset subset
            eval_ds = load_gaia_dataset(use_raw_dataset=False, set_to_run="validation")
            task_ids_file = os.getenv("MALBO_TASK_IDS_FILE")
            if task_ids_file:
                # fixed task-id list (json with {"tasks":[{"task_id":..}]} or a plain list / txt, one id per line)
                subset_questions = select_by_task_ids(eval_ds, task_ids_file)
            else:
                subset_questions = create_deterministic_subset(eval_ds, dataset_limits, random_seed)
            
            if not subset_questions:
                print("⚠️ No questions selected for evaluation")
                return 0.0, 0.0, session_dir
            
            # Process questions
            results = []
            retry_config = self.base_config.get('retry_config', 
                {"max_retries": 2, "base_delay": 1.0, "backoff_factor": 2.0})

            from utils import abort
            abort.reset()
            import socket
            socket.setdefaulttimeout(120)   # agent code / tools calling requests without timeout hung questions forever
            from utils import llm_logger as _ll
            for k in [k for k in _ll._state["counters"] if k and k[0] == "bad_request"]:
                del _ll._state["counters"][k]   # repeated-400 detector counts per evaluation
            prices = {r: c["pool"] for r, c in agent_model_configs.items() if isinstance(c, dict) and c.get("pool")}
            # Question cache (resume after a pause): completed questions of this exact config + task list are not run again.
            qcache = QuestionCache(agent_model_configs, [q["task_id"] for q in subset_questions])
            cached = qcache.load()
            todo = [(i, q) for i, q in enumerate(subset_questions) if q["task_id"] not in cached]
            if cached:
                print(f"♻️  question cache {qcache.path.name}: {len(cached)} done earlier, {len(todo)} to run")

            from utils.session_progress import SessionProgress
            progress = SessionProgress(session_dir, run_name, len(subset_questions), concurrency)
            parallel = concurrency > 1
            # parallel: every question builds its own model objects (+ own agents/tools/browser) -> nothing mutable is shared
            factory = (lambda: load_agent_models(agent_model_configs, quiet=True)) if parallel else None

            def stub(example, i, err):
                return {"question_index": i, "question": example["question"], "true_answer": str(example["true_answer"]),
                        "prediction": None, "is_correct": False, "task_level": example["task"], "task_id": example["task_id"],
                        "parsing_error": False, "iteration_limit_exceeded": False, "agent_error": err,
                        "duration_seconds": 0, "token_counts": {"input": 0, "output": 0}}

            saved = set()   # task ids already written to the question cache

            def run_one(example, i):
                if abort.reason() is not None:   # paused before this question started: do not even build the team
                    abort.mark_failed(example["task_id"])
                    return stub(example, i, "not run: evaluation paused")
                progress.start(i, example["task_id"])
                box = {}

                def answer():
                    try:
                        box["r"] = answer_single_question_with_tracking(example, models, run_manager, retry_config, i,
                                                                        models_factory=factory, parallel=parallel)
                    except Exception as e:  # answer_single_question already catches; belt and braces
                        box["r"] = stub(example, i, str(e))
                # wall-clock limit: agent code can block on the network forever (seen: urlopen to a tarpitting site)
                t = threading.Thread(target=answer, daemon=True)
                t.start(); t.join(QUESTION_TIMEOUT_S)
                if t.is_alive():
                    abort.cancel(example["task_id"])
                    box["r"] = {**stub(example, i, f"question timeout {QUESTION_TIMEOUT_S}s"), "duration_seconds": QUESTION_TIMEOUT_S}
                r = box["r"]
                progress.finish(i, example["task_id"], r)
                if abort.reason() is None and not r.get("agent_error") and prices:   # saved at once: a kill/crash loses no finished question
                    log = Path(session_dir) / "llm_calls.jsonl"
                    if log.exists():
                        qcache.save([r], {r["task_id"]: evaluation_costs(log, prices, task_ids={r["task_id"]})}, str(session_dir))
                        saved.add(r["task_id"])
                return r

            if not parallel:
                for i, example in tqdm(todo, desc="Processing questions"):
                    results.append(run_one(example, i))
            else:
                with ThreadPoolExecutor(max_workers=concurrency) as executor:
                    futures = {executor.submit(run_one, example, i): i for i, example in todo}
                    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing questions"):
                        results.append(future.result())
            progress.close()

            log_path = Path(session_dir) / "llm_calls.jsonl"
            failed = abort.failed_tasks() | {r["task_id"] for r in results if "paused" in str(r.get("agent_error") or "")}
            if abort.reason() is not None:
                failed |= {r["task_id"] for r in results if r.get("agent_error")}   # any error during a pause: not trusted
            kept = [r for r in results if r["task_id"] not in failed]
            per_q = {}
            if prices and log_path.exists():
                per_q = {r["task_id"]: evaluation_costs(log_path, prices, task_ids={r["task_id"]}) for r in kept}
            if abort.reason() is not None:
                qcache.save([r for r in kept if r["task_id"] not in saved], per_q, str(session_dir))
                rep = {"reason": abort.reason(), "kept": len(kept), "failed_task_ids": sorted(failed),
                       "n_done_total": len(cached) + len(kept), "n_total": len(subset_questions)}
                if prices and log_path.exists():
                    rep["session_cost"] = evaluation_costs(log_path, prices)
                    rep["wasted_cost"] = evaluation_costs(log_path, prices, task_ids=failed)   # spent on questions that will be redone
                (Path(session_dir) / "paused_report.json").write_text(json.dumps(rep, indent=1, default=str))
                print(f"⏸️  Evaluation paused: {len(cached) + len(kept)}/{len(subset_questions)} questions kept, "
                      f"{len(failed)} to redo; wasted go_list ${rep.get('wasted_cost', {}).get('go_list_cost', 0):.4f}")
                raise abort.EvaluationPaused(abort.reason(), len(cached) + len(kept), len(subset_questions), str(Path(session_dir).resolve()))
            if cached:
                qcache.save([r for r in kept if r["task_id"] not in saved], per_q, str(session_dir))   # so a crash before MALBO logs the result does not lose them
            results = [c["result"] for c in cached.values()] + results
            results.sort(key=lambda r: r.get("question_index", 0))
            if os.getenv("MALBO_SCORER", "exact") == "relaxed":   # also rescores cached questions, so every result uses one scorer
                from scripts.gaia_scorer import rescore
                results = [rescore(r) for r in results]

            # Calculate metrics and agent-specific cost breakdown
            final_metrics = calculate_final_metrics(results)
            accuracy = final_metrics.get("overall_accuracy", 0.0)
            
            # Cost objective: tokens x AA prices of the pool row of each role (from <session>/llm_calls.jsonl);
            # falls back to the old CostEstimator only if the log is missing / roles carry no pool prices.
            cost_breakdown = {}
            total_cost = None
            if prices:
                try:
                    rep = evaluation_costs(log_path, prices)
                    if cached:   # resumed: earlier questions' costs (their own sessions) + this session's calls
                        rep = merge_reports([c["cost_report"] for c in cached.values()] + [rep])
                        rep["resumed_from"] = sorted({c.get("session_dir") for c in cached.values()})
                        rep["n_questions_cached"] = len(cached)
                    total_cost = rep["aa_cost"]
                    (Path(session_dir) / "cost_report.json").write_text(json.dumps(rep, indent=1))
                    for r, d in rep["per_role"].items():
                        cost_breakdown[r] = {'model_id': d["model"], 'cost': d["aa_cost"], 'tokens': d["input"] + d["cache_read"] + d["output"]}
                    print(f"💰 aa_cost=${rep['aa_cost']:.6f} go_expected=${rep['go_expected']:.6f} go_list_cost=${rep['go_list_cost']:.6f} "
                          f"go_allowance=${rep['go_allowance']:.6f} calls={rep['n_calls']} unpriced={rep['n_unpriced']} "
                          f"{'MISMATCH(>2%)' if rep['mismatch'] else 'consistent'}")
                except FileNotFoundError:
                    print("⚠️ WARNING: llm_calls.jsonl missing -> falling back to the old CostEstimator")
            if total_cost is None:
                total_cost = self._legacy_cost(run_manager, agent_model_configs, cost_breakdown)

            # Enhanced result output with cost breakdown
            if save_detailed_results:
                execution_time = time.time() - run_manager.start_time
                run_manager.save_session_summary(session_dir, "optimization", execution_time, final_metrics)
            
            # Print detailed cost breakdown for transparency
            print(f"💰 Cost Breakdown:")
            for agent_name, details in cost_breakdown.items():
                print(f"  {agent_name} ({details['model_id']}): ${details['cost']:.6f} ({details['tokens']:,} tokens)")
            print(f"  Total: ${total_cost:.6f}")
            print(f"✅ Evaluation complete: {accuracy:.2f}% accuracy, ${total_cost:.6f} cost")
            
            return accuracy, total_cost, Path(session_dir).resolve()
            
        except Exception as e:
            # never return a fake (0 %, $0) result: the caller must not record it as an evaluation
            print(f"❌ Evaluation failed: {type(e).__name__}: {e}")
            raise

    @staticmethod
    def _legacy_cost(run_manager, agent_model_configs, cost_breakdown) -> float:
        all_calls = run_manager.get_token_tracker().get_all_calls()
        cost_estimator = CostEstimator()
        total_cost = 0.0
        tokens_per_agent = {}
        for call in all_calls:
            t = tokens_per_agent.setdefault(call.agent_name, {'input': 0, 'output': 0})
            t['input'] += call.input_tokens
            t['output'] += call.output_tokens
        for agent_name, tu in tokens_per_agent.items():
            if agent_name in agent_model_configs:
                model_id = agent_model_configs[agent_name].get('model_id', 'gpt-4o-mini')
                c = cost_estimator.estimate_cost(model_name=model_id, input_tokens=tu['input'], output_tokens=tu['output']).get("total_cost", 0.0)
                total_cost += c
                cost_breakdown[agent_name] = {'model_id': model_id, 'cost': c, 'tokens': tu['input'] + tu['output']}
        return total_cost


def evaluate_configuration(agent_model_configs: Dict[str, Any], dataset_limits: Optional[Dict[str, int]] = None,
                          run_name: Optional[str] = None, **kwargs) -> Tuple[float, float, Path]:
    """Simple function interface for quick evaluations."""
    try:
        interface = GaiaOptimizationInterface("gaia_subset_config.yaml")
    except FileNotFoundError:
        print("⚠️ gaia_subset_config.yaml not found, using interface without base config")
        interface = GaiaOptimizationInterface()
    return interface.evaluate_configuration(agent_model_configs, dataset_limits, run_name, **kwargs)


if __name__ == "__main__":
    import yaml
    
    print("🧪 Testing optimization interface by loading from config file...")
    
    # Test configuration
    # test_models = {
    #     'text_inspector': {'model_class': 'LiteLLMModel', 'model_id': 'gpt-4.1-nano'},
    #     'visual_qa': {'model_class': 'LiteLLMModel', 'model_id': 'gpt-4o-mini'},
    #     'reformulator': {'model_class': 'LiteLLMModel', 'model_id': 'gpt-4.1-nano'}
    # }
    # test_limits = {'task_1': 1, 'task_2': 0, 'task_3': 0}
    
        # Load configuration from YAML file to ensure consistency with real usage
    try:
        with open("gaia_subset_config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        # Extract necessary configurations
        test_models = config['agents']
        test_limits = {'task_1': 1, 'task_2': 0, 'task_3': 0}  # Use 1 question for quick test
        
        print(f"✅ Loaded configuration with {len(test_models)} agent models:")
        for agent_name, agent_config in test_models.items():
            print(f"   {agent_name}: {agent_config['model_id']}")
        
    except FileNotFoundError:
        print("❌ Error: gaia_subset_config.yaml not found. Please ensure the config file exists.")
        exit(1)
    except KeyError as e:
        print(f"❌ Error: Your gaia_subset_config.yaml is missing a required key: {e}")
        exit(1)
    
    accuracy, cost, path = evaluate_configuration(
        agent_model_configs=test_models,
        dataset_limits=test_limits,
        run_name="interface_test_from_yaml",
        save_detailed_results=True,
        concurrency=1  # Use sequential processing for easier debugging
    )
    
    print(f"\n📊 Test Results Summary:")
    print(f"   Accuracy: {accuracy:.2f}%")
    print(f"   Total Cost: ${cost:.6f}")
    print(f"   Session: {path}")
    print("✅ Interface test completed successfully!")
