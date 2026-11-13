"""
Programmatic interface for GAIA benchmark evaluation.
Designed to be used by optimization loops (BoTorch, Optuna, etc.).
"""

import json
import os
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
            
            from utils.session_progress import SessionProgress
            progress = SessionProgress(session_dir, run_name, len(subset_questions), concurrency)
            parallel = concurrency > 1
            # parallel: every question builds its own model objects (+ own agents/tools/browser) -> nothing mutable is shared
            factory = (lambda: load_agent_models(agent_model_configs, quiet=True)) if parallel else None

            def run_one(example, i):
                progress.start(i, example["task_id"])
                try:
                    r = answer_single_question_with_tracking(example, models, run_manager, retry_config, i,
                                                             models_factory=factory, parallel=parallel)
                except Exception as e:  # answer_single_question already catches; belt and braces
                    r = {"question_index": i, "question": example["question"], "true_answer": str(example["true_answer"]),
                         "prediction": None, "is_correct": False, "task_level": example["task"], "task_id": example["task_id"],
                         "parsing_error": False, "iteration_limit_exceeded": False, "agent_error": str(e),
                         "duration_seconds": 0, "token_counts": {"input": 0, "output": 0}}
                progress.finish(i, example["task_id"], r)
                return r

            if not parallel:
                for i, example in enumerate(tqdm(subset_questions, desc="Processing questions")):
                    results.append(run_one(example, i))
            else:
                with ThreadPoolExecutor(max_workers=concurrency) as executor:
                    futures = {executor.submit(run_one, example, i): i for i, example in enumerate(subset_questions)}
                    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing questions"):
                        results.append(future.result())
                results.sort(key=lambda r: r.get("question_index", 0))
            progress.close()
            
            # Calculate metrics and agent-specific cost breakdown
            final_metrics = calculate_final_metrics(results)
            accuracy = final_metrics.get("overall_accuracy", 0.0)
            
            # Cost objective: tokens x AA prices of the pool row of each role (from <session>/llm_calls.jsonl);
            # falls back to the old CostEstimator only if the log is missing / roles carry no pool prices.
            cost_breakdown = {}
            total_cost = None
            prices = {r: c["pool"] for r, c in agent_model_configs.items() if isinstance(c, dict) and c.get("pool")}
            if prices:
                try:
                    from utils.pool_cost import evaluation_costs
                    rep = evaluation_costs(Path(session_dir) / "llm_calls.jsonl", prices)
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
            print(f"❌ Evaluation failed: {e}")
            return 0.0, 0.0, Path("./error")

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
