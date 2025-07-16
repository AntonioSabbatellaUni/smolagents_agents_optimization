# GAIA subset evaluation utilities
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

import pandas as pd
import datasets
import yaml
from huggingface_hub import snapshot_download

from .cost_estimator import CostEstimator


def load_config(config_path: str) -> Dict[str, Any]:
    """Load the subset configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_gaia_dataset(use_raw_dataset: bool, set_to_run: str) -> datasets.Dataset:
    """Load the GAIA dataset."""
    if use_raw_dataset:
        if set_to_run == "test":
            snapshot_download(
                repo_id="gaia-benchmark/GAIA",
                revision="main",
                repo_type="dataset",
                local_dir="./data_gaia",
            )
        else:
            snapshot_download(
                repo_id="gaia-benchmark/GAIA",
                revision="main",
                repo_type="dataset",
                local_dir="./data_gaia",
            )

        dataset = datasets.load_dataset(
            "json",
            data_files=f"./data_gaia/2023/{set_to_run}.jsonl",
            cache_dir="./data_gaia",
        )
        eval_ds = dataset["train"]
    else:
        eval_ds = datasets.load_dataset(
            "gaia-benchmark/GAIA", "2023_all", cache_dir="./data_gaia"
        )[set_to_run]
    
    return eval_ds


class ProgressTracker:
    """Real-time progress tracking for subset evaluation."""
    
    def __init__(self, progress_file: str):
        self.progress_file = Path(progress_file)
        self.start_time = time.time()
        self.questions = {}
        self.completed_count = 0
        self.total_count = 0
        
    def initialize_questions(self, questions: List[Dict]):
        """Initialize tracking for all questions."""
        self.total_count = len(questions)
        for i, question in enumerate(questions):
            question_id = question.get("question_id", f"q_{i}")
            self.questions[question_id] = {
                "status": "pending",
                "start_time": None,
                "end_time": None,
                "correct": None,
                "runtime_seconds": None
            }
        self._save_progress()
        
    def start_question(self, question_id: str):
        """Mark a question as started."""
        if question_id in self.questions:
            self.questions[question_id]["status"] = "processing"
            self.questions[question_id]["start_time"] = time.time()
            self._save_progress()
            
    def complete_question(self, question_id: str, correct: bool, runtime: float):
        """Mark a question as completed."""
        if question_id in self.questions:
            self.questions[question_id]["status"] = "completed"
            self.questions[question_id]["end_time"] = time.time()
            self.questions[question_id]["correct"] = correct
            self.questions[question_id]["runtime_seconds"] = runtime
            self.completed_count += 1
            self._save_progress()
            
    def _save_progress(self):
        """Save current progress to JSON file."""
        progress_data = {
            "timestamp": datetime.now().isoformat(),
            "total_questions": self.total_count,
            "completed_questions": self.completed_count,
            "progress_percentage": (self.completed_count / self.total_count * 100) if self.total_count > 0 else 0,
            "elapsed_time_seconds": time.time() - self.start_time,
            "questions": self.questions
        }
        
        with open(self.progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
            
    def finalize(self):
        """Mark tracking as complete."""
        self._save_progress()
        print(f"📊 Progress tracking saved to: {self.progress_file}")


def create_deterministic_subset(eval_ds: datasets.Dataset, dataset_limits: Dict[str, int], 
                               seed: int = 42) -> List[Dict]:
    """Create a deterministic subset of the dataset based on task limits."""
    random.seed(seed)
    
    # Convert to pandas for easier manipulation
    df = pd.DataFrame(eval_ds)
    
    print(f"📊 Full dataset distribution:")
    task_distribution = df['task'].value_counts().sort_index()
    print(task_distribution)
    
    selected_examples = []
    
    for task_level, limit in dataset_limits.items():
        task_num = int(task_level.split('_')[1])  # Extract number from 'task_1', etc.
        task_str = str(task_num)  # Convert to string to match dataset format
        
        # Get all questions for this task level
        task_questions = df[df['task'] == task_str].to_dict('records')
        
        available_count = len(task_questions)
        actual_limit = min(limit, available_count)
        
        if available_count == 0:
            print(f"⚠️  No questions available for task level {task_num}")
            continue
        elif available_count < limit:
            print(f"⚠️  Only {available_count} questions available for task level {task_num} (requested {limit})")
        
        # Shuffle deterministically and select first N
        random.shuffle(task_questions)
        selected_task_questions = task_questions[:actual_limit]
        
        selected_examples.extend(selected_task_questions)
        print(f"✅ Selected {len(selected_task_questions)} questions from task level {task_num}")
    
    print(f"📋 Total selected questions: {len(selected_examples)}")
    return selected_examples


def calculate_final_metrics(results: List[Dict]) -> Dict:
    """Calculate comprehensive performance metrics."""
    total_questions = len(results)
    correct_answers = sum(1 for r in results if r.get("correct", r.get("is_correct", False)))
    accuracy = (correct_answers / total_questions) * 100 if total_questions > 0 else 0
    
    # Calculate metrics by task level
    task_metrics = {}
    for task_level in [1, 2, 3]:
        task_results = [r for r in results if str(r.get("task", r.get("task_level", ""))) == str(task_level)]
        if task_results:
            task_correct = sum(1 for r in task_results if r.get("correct", r.get("is_correct", False)))
            task_metrics[f"task_{task_level}"] = {
                "total": len(task_results),
                "correct": task_correct,
                "accuracy": (task_correct / len(task_results)) * 100
            }
    
    # Calculate cost metrics - handle different result formats
    total_input_tokens = 0
    total_output_tokens = 0
    
    for r in results:
        token_counts = r.get("token_counts", {})
        if isinstance(token_counts, dict):
            total_input_tokens += token_counts.get("input", 0)
            total_output_tokens += token_counts.get("output", 0)
    
    # Calculate error rates
    parsing_errors = sum(1 for r in results if r.get("parsing_error", False))
    iteration_limit_errors = sum(1 for r in results if r.get("iteration_limit_exceeded", False))
    agent_errors = sum(1 for r in results if r.get("agent_error") is not None)
    
    return {
        "overall_accuracy": accuracy,
        "correct_answers": correct_answers,
        "total_questions": total_questions,
        "task_breakdown": task_metrics,
        "token_usage": {
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens
        },
        "error_analysis": {
            "parsing_errors": parsing_errors,
            "iteration_limit_errors": iteration_limit_errors,
            "agent_errors": agent_errors,
            "total_errors": parsing_errors + iteration_limit_errors + agent_errors
        }
    }


def save_results_to_files(results: List[Dict], final_metrics: Dict, config: Dict, run_name: str) -> List[Path]:
    """Save all results and configurations to organized files."""
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"results_{run_name}_{timestamp}")
    results_dir.mkdir(exist_ok=True)
    
    saved_files = []
    
    # Save detailed results as JSONL
    results_file = results_dir / "detailed_results.jsonl"
    with open(results_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    saved_files.append(results_file)
    
    # Save summary metrics
    summary_file = results_dir / "summary_metrics.json"
    with open(summary_file, 'w') as f:
        json.dump(final_metrics, f, indent=2)
    saved_files.append(summary_file)
    
    # Save configuration copy
    config_file = results_dir / "subset_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    saved_files.append(config_file)
    
    # Save CSV for analysis
    csv_file = results_dir / "results.csv"
    df_results = pd.DataFrame(results)
    df_results.to_csv(csv_file, index=False)
    saved_files.append(csv_file)
    
    return saved_files


def print_final_summary(results: List[Dict], final_metrics: Dict, run_manager=None, cost_config_path: Optional[str] = None):
    """Print a consolidated final summary with configurable cost estimation."""
    print("\n" + "="*60)
    print("🎯 GAIA SUBSET EVALUATION COMPLETED")
    print("="*60)
    
    print(f"📊 Overall Accuracy: {final_metrics['overall_accuracy']:.1f}%")
    print(f"✅ Correct Answers: {final_metrics['correct_answers']}/{final_metrics['total_questions']}")
    
    if final_metrics['task_breakdown']:
        print("\n📈 Task Level Breakdown:")
        for task, metrics in final_metrics['task_breakdown'].items():
            print(f"  {task}: {metrics['correct']}/{metrics['total']} ({metrics['accuracy']:.1f}%)")
    
    print(f"\n💰 Token Usage:")
    print(f"  Input tokens: {final_metrics['token_usage']['total_input_tokens']:,}")
    print(f"  Output tokens: {final_metrics['token_usage']['total_output_tokens']:,}")
    print(f"  Total tokens: {final_metrics['token_usage']['total_tokens']:,}")
    
    # Enhanced cost estimation with configurable models
    cost_estimator = CostEstimator(cost_config_path)
    
    # Try to determine model from results or use default
    model_usage = {}
    for result in results:
        # Extract model info from results if available
        model_name = result.get("model_name", "gpt-4o-mini")  # Default fallback
        if model_name not in model_usage:
            model_usage[model_name] = {"input": 0, "output": 0}
        
        token_counts = result.get("token_counts", {})
        model_usage[model_name]["input"] += token_counts.get("input", 0)
        model_usage[model_name]["output"] += token_counts.get("output", 0)
    
    # If no model-specific data, use aggregate with default model
    if not model_usage:
        model_usage = {
            "gpt-4o-mini": {  # Default model
                "input": final_metrics['token_usage']['total_input_tokens'],
                "output": final_metrics['token_usage']['total_output_tokens']
            }
        }
    
    # Calculate costs
    if len(model_usage) == 1:
        # Single model
        model_name, tokens = next(iter(model_usage.items()))
        cost_estimate = cost_estimator.estimate_cost(
            model_name, 
            tokens["input"], 
            tokens["output"]
        )
        print(f"  Estimated cost ({cost_estimate['provider']} {cost_estimate['model']}): ${cost_estimate['total_cost']:.4f}")
        print(f"    Input cost: ${cost_estimate['input_cost']:.4f}")
        print(f"    Output cost: ${cost_estimate['output_cost']:.4f}")
    else:
        # Multiple models
        multi_cost = cost_estimator.estimate_multi_model_cost(model_usage)
        print(f"  Total estimated cost: ${multi_cost['total_cost']:.4f}")
        print("  Model breakdown:")
        for model_name, cost_data in multi_cost['model_breakdown'].items():
            print(f"    {cost_data['provider']} {model_name}: ${cost_data['total_cost']:.4f}")
    
    if final_metrics['error_analysis']['total_errors'] > 0:
        print(f"\n⚠️  Errors encountered: {final_metrics['error_analysis']['total_errors']}")
        print(f"   Parsing: {final_metrics['error_analysis']['parsing_errors']}")
        print(f"   Iteration limit: {final_metrics['error_analysis']['iteration_limit_errors']}")
        print(f"   Agent errors: {final_metrics['error_analysis']['agent_errors']}")
    
    if run_manager:
        try:
            # Get cost information from run manager if available
            print(f"\n💲 Enhanced Cost Tracking:")
            cost_summary = run_manager.get_cost_summary()
            if cost_summary:
                print(f"  Total cost: ${cost_summary.get('total_cost', 0):.4f}")
                print(f"  Sessions: {cost_summary.get('session_count', 0)}")
        except Exception as e:
            print(f"   Could not retrieve enhanced cost data: {e}")
    
    print("\n🎉 Evaluation complete!")
