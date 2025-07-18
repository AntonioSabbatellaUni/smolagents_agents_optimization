"""
Programmatic interface for GAIA benchmark evaluation.
Designed to be used by optimization loops (BoTorch, Optuna, etc.).
"""

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
        run_name: Optional[str] = None, concurrency: int = 1, random_seed: int = 42,
        save_detailed_results: bool = False
    ) -> Tuple[float, float, Path]:
        """
        Evaluate a specific model configuration on GAIA benchmark.
        
        Args:
            agent_model_configs: Model configuration for each agent role
            dataset_limits: Number of questions per task level
            run_name: Unique identifier for this evaluation
            concurrency: Number of parallel workers
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
        
        print(f"🚀 Starting evaluation: {run_name}")
        
        try:
            # Load models and setup tracking
            models = load_agent_models(agent_model_configs)
            run_manager = EnhancedRunManager()
            session_dir = run_manager.setup_session("optimization", f"subset_{run_name}")
            run_manager.start_run()
            
            # Load dataset subset
            eval_ds = load_gaia_dataset(use_raw_dataset=False, set_to_run="validation")
            subset_questions = create_deterministic_subset(eval_ds, dataset_limits, random_seed)
            
            if not subset_questions:
                print("⚠️ No questions selected for evaluation")
                return 0.0, 0.0, session_dir
            
            # Process questions
            results = []
            retry_config = self.base_config.get('retry_config', 
                {"max_retries": 2, "base_delay": 1.0, "backoff_factor": 2.0})
            
            if concurrency == 1:
                for i, example in enumerate(tqdm(subset_questions, desc="Processing questions")):
                    result = answer_single_question_with_tracking(example, models, run_manager, retry_config, i)
                    results.append(result)
            else:
                with ThreadPoolExecutor(max_workers=concurrency) as executor:
                    futures = {executor.submit(answer_single_question_with_tracking, example, models, 
                              run_manager, retry_config, i): i for i, example in enumerate(subset_questions)}
                    
                    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing questions"):
                        results.append(future.result())
            
            # Calculate metrics and cost
            final_metrics = calculate_final_metrics(results)
            accuracy = final_metrics.get("overall_accuracy", 0.0)
            
            total_input_tokens = final_metrics['token_usage']['total_input_tokens']
            total_output_tokens = final_metrics['token_usage']['total_output_tokens']
            cost_estimator = CostEstimator()
            cost_estimate = cost_estimator.estimate_cost("gpt-4o-mini", total_input_tokens, total_output_tokens)
            total_cost = cost_estimate.get("total_cost", 0.0)
            
            # Save summary if requested
            if save_detailed_results:
                execution_time = time.time() - run_manager.start_time
                run_manager.save_session_summary(session_dir, "optimization", execution_time, final_metrics)
            
            print(f"✅ Evaluation complete: {accuracy:.2f}% accuracy, ${total_cost:.6f} cost")
            return accuracy, total_cost, session_dir
            
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            return 0.0, 0.0, Path("./error")


def evaluate_configuration(agent_model_configs: Dict[str, Any], dataset_limits: Optional[Dict[str, int]] = None,
                          run_name: Optional[str] = None, **kwargs) -> Tuple[float, float, Path]:
    """Simple function interface for quick evaluations."""
    interface = GaiaOptimizationInterface("gaia_subset_config.yaml")
    return interface.evaluate_configuration(agent_model_configs, dataset_limits, run_name, **kwargs)


if __name__ == "__main__":
    print("🧪 Testing optimization interface...")
    
    # Test configuration
    test_models = {
        'text_inspector': {'model_class': 'LiteLLMModel', 'model_id': 'gpt-4o-mini'},
        'visual_qa': {'model_class': 'LiteLLMModel', 'model_id': 'gpt-4o-mini'},
        'reformulator': {'model_class': 'LiteLLMModel', 'model_id': 'gpt-4o-mini'}
    }
    test_limits = {'task_1': 2, 'task_2': 0, 'task_3': 0}
    
    accuracy, cost, path = evaluate_configuration(
        agent_model_configs=test_models, dataset_limits=test_limits,
        run_name="interface_test", save_detailed_results=True
    )
    
    print(f"\n📊 Test Results:")
    print(f"   Accuracy: {accuracy:.2f}%")
    print(f"   Cost: ${cost:.6f}")
    print(f"   Logs: {path}")
