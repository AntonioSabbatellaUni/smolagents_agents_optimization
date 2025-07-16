# Subset evaluation utilities
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

import pandas as pd
import datasets
import yaml
from huggingface_hub import snapshot_download


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
    
    def __init__(self, total_questions: int, output_dir: Path):
        self.total_questions = total_questions
        self.completed = 0
        self.failed = 0
        self.start_time = time.time()
        self.progress_file = output_dir / "progress.json"
        
        # Initialize progress file
        self._update_progress_file()
    
    def complete_question(self, success: bool):
        """Mark a question as completed."""
        if success:
            self.completed += 1
        else:
            self.failed += 1
        self._update_progress_file()
    
    def _update_progress_file(self):
        """Update the progress file with current stats."""
        elapsed = time.time() - self.start_time
        remaining = self.total_questions - self.completed - self.failed
        
        if self.completed > 0:
            avg_time = elapsed / self.completed
            eta = avg_time * remaining
        else:
            eta = 0
        
        progress_data = {
            "timestamp": datetime.now().isoformat(),
            "total_questions": self.total_questions,
            "completed": self.completed,
            "failed": self.failed,
            "remaining": remaining,
            "elapsed_seconds": elapsed,
            "eta_seconds": eta,
            "completion_percentage": (self.completed + self.failed) / self.total_questions * 100,
            "success_rate": self.completed / (self.completed + self.failed) * 100 if (self.completed + self.failed) > 0 else 0
        }
        
        with open(self.progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
    
    def get_status_line(self) -> str:
        """Get a concise status line for printing."""
        completion_pct = (self.completed + self.failed) / self.total_questions * 100
        success_rate = self.completed / (self.completed + self.failed) * 100 if (self.completed + self.failed) > 0 else 0
        return f"Progress: {self.completed + self.failed}/{self.total_questions} ({completion_pct:.1f}%) | Success: {success_rate:.1f}%"


def create_deterministic_subset(eval_ds: datasets.Dataset, dataset_limits: Dict[str, int], 
                               seed: int = 42) -> List[Dict]:
    """Create a deterministic subset of the dataset based on task limits."""
    random.seed(seed)
    df = pd.DataFrame(eval_ds)
    
    print(f"📊 Full dataset: {df['task'].value_counts().sort_index().to_dict()}")
    
    selected_examples = []
    
    for task_level, limit in dataset_limits.items():
        task_num = int(task_level.split('_')[1])
        task_str = str(task_num)
        
        task_questions = df[df['task'] == task_str].to_dict('records')
        available_count = len(task_questions)
        actual_limit = min(limit, available_count)
        
        if available_count == 0:
            continue
        
        random.shuffle(task_questions)
        selected_task_questions = task_questions[:actual_limit]
        selected_examples.extend(selected_task_questions)
    
    print(f"📋 Selected {len(selected_examples)} questions: {dict(pd.Series([ex['task'] for ex in selected_examples]).value_counts().sort_index())}")
    return selected_examples


def calculate_final_metrics(results: List[Dict]) -> Dict:
    """Calculate comprehensive performance metrics."""
    total_questions = len(results)
    correct_answers = sum(1 for r in results if r["is_correct"])
    accuracy = (correct_answers / total_questions) * 100 if total_questions > 0 else 0
    
    # Calculate metrics by task level
    task_metrics = {}
    for task_level in [1, 2, 3]:
        task_results = [r for r in results if r["task_level"] == str(task_level)]
        if task_results:
            task_correct = sum(1 for r in task_results if r["is_correct"])
            task_metrics[f"task_{task_level}"] = {
                "total": len(task_results),
                "correct": task_correct,
                "accuracy": (task_correct / len(task_results)) * 100
            }
    
    # Calculate cost metrics
    total_input_tokens = sum(r["token_counts"]["input"] for r in results)
    total_output_tokens = sum(r["token_counts"]["output"] for r in results)
    
    # Calculate error rates
    parsing_errors = sum(1 for r in results if r["parsing_error"])
    iteration_limit_errors = sum(1 for r in results if r["iteration_limit_exceeded"])
    agent_errors = sum(1 for r in results if r["agent_error"] is not None)
    
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


def save_results_to_files(results: List[Dict], config: Dict, session_dir: Path, 
                         final_metrics: Dict) -> List[Path]:
    """Save all results and configurations to organized files."""
    saved_files = []
    
    # Save detailed results as JSONL
    results_file = session_dir / "detailed_results.jsonl"
    with open(results_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    saved_files.append(results_file)
    
    # Save summary metrics
    summary_file = session_dir / "summary_metrics.json"
    with open(summary_file, 'w') as f:
        json.dump(final_metrics, f, indent=2)
    saved_files.append(summary_file)
    
    # Save configuration copy
    config_file = session_dir / "subset_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    saved_files.append(config_file)
    
    # Save CSV for analysis
    csv_file = session_dir / "results.csv"
    df_results = pd.DataFrame(results)
    df_results.to_csv(csv_file, index=False)
    saved_files.append(csv_file)
    
    return saved_files


def print_final_summary(final_metrics: Dict, session_dir: Path, saved_files: List[Path], 
                       execution_time: float):
    """Print a comprehensive final summary."""
    print("\n" + "="*60)
    print(f"🎯 EXPERIMENT COMPLETED")
    print("="*60)
    
    # Core metrics
    print(f"📊 Accuracy: {final_metrics['correct_answers']}/{final_metrics['total_questions']} ({final_metrics['overall_accuracy']:.1f}%)")
    
    # Task breakdown
    task_summary = []
    for task, metrics in final_metrics['task_breakdown'].items():
        task_summary.append(f"{task}: {metrics['correct']}/{metrics['total']} ({metrics['accuracy']:.1f}%)")
    print(f"📈 By level: {' | '.join(task_summary)}")
    
    # Cost info
    tokens = final_metrics['token_usage']
    cost_per_input = 0.00015 / 1000  # gpt-4o-mini rates
    cost_per_output = 0.0006 / 1000
    estimated_cost = (tokens['total_input_tokens'] * cost_per_input + 
                     tokens['total_output_tokens'] * cost_per_output)
    print(f"💰 Cost: {tokens['total_tokens']:,} tokens (${estimated_cost:.4f})")
    
    # Errors
    if final_metrics['error_analysis']['total_errors'] > 0:
        errors = final_metrics['error_analysis']
        print(f"⚠️  Errors: {errors['total_errors']} (parsing: {errors['parsing_errors']}, agent: {errors['agent_errors']})")
    
    print(f"⏱️  Time: {execution_time:.1f}s | 📁 Results: {session_dir.name}")
    print(f"📄 Files: {len(saved_files)} saved")
