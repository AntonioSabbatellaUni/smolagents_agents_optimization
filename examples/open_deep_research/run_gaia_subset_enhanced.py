# EXAMPLE COMMAND: python run_gaia_subset_enhanced.py --config gaia_subset_config.yaml --run-name economic-test-v1
import argparse
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from huggingface_hub import login
from scripts.reformulator import prepare_response
from tqdm import tqdm
from smolagents.utils import make_json_serializable

# Import tracking utilities
from utils import EnhancedRunManager
from loader import load_agent_models

# Import modular utilities  
from utils.subset_evaluation import (
    load_config as load_subset_config,
    create_deterministic_subset,
    calculate_final_metrics,
    save_results_to_files,
    print_final_summary,
    sanitize_agent_memory
)
from utils.agent_factory import (
    create_agent_team_with_tracking,
    process_file_attachments,
    create_gaia_dataset_loader
)
from utils.cost_estimator import CostEstimator


load_dotenv(override=True)
login(os.getenv("HF_TOKEN"))

append_answer_lock = threading.Lock()


def parse_args():
    parser = argparse.ArgumentParser(description="Run GAIA subset evaluation with enhanced tracking")
    parser.add_argument("--config", type=str, required=True, help="Path to subset configuration YAML file")
    parser.add_argument("--run-name", type=str, required=True, help="Name for this experimental run")
    parser.add_argument("--set-to-run", type=str, default="validation", help="Dataset split to use")
    parser.add_argument("--use-raw-dataset", action="store_true", help="Use raw GAIA dataset instead of annotated")
    parser.add_argument("--cost-config", type=str, help="Path to cost configuration YAML file")
    return parser.parse_args()


### IMPORTANT: EVALUATION SWITCHES

print("Make sure you deactivated any VPN like Tailscale, else some URLs will be blocked!")

custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}


def load_gaia_dataset(use_raw_dataset: bool, set_to_run: str):
    """Load GAIA dataset using the factory function."""
    return create_gaia_dataset_loader(use_raw_dataset, set_to_run)


def answer_single_question_with_tracking(
    example: dict, models: Dict, run_manager: EnhancedRunManager, 
    retry_config: Dict, question_index: int
) -> Dict:
    """Answer a single question with enhanced tracking and retry logic."""
    max_retries = retry_config["max_retries"]
    base_delay = retry_config["base_delay"]
    backoff_factor = retry_config["backoff_factor"]
    
    for attempt in range(max_retries + 1):
        try:
            # Reset session tracking for this question
            run_manager.token_tracker.reset_session()
            
            agent = create_agent_team_with_tracking(models, run_manager)

            # Process file attachments using modular function
            augmented_question = process_file_attachments(example, models)

            start_time = datetime.now()
            
            # Run agent with tracking
            final_result = agent.run(augmented_question)
            
            # Process agent memory
            try:
                raw_agent_memory = agent.write_memory_to_messages()
                agent_memory = sanitize_agent_memory(raw_agent_memory)
                final_result = prepare_response(augmented_question, agent_memory, reformulation_model=models["reformulator"])
            except Exception as memory_error:
                print(f"⚠️  Memory processing error during reformulation: {memory_error}")
                agent_memory = []
                # Keep original final_result from agent.run()
                # Convert to string if it's not already
                final_result = str(final_result) if final_result is not None else ""
            
            # Converti la memoria in un formato serializzabile (lista di dizionari)
            serializable_memory = [{'role': str(msg.role), 'content': msg.content} for msg in agent_memory]

            end_time = datetime.now()
            
            # Calculate metrics
            prediction = str(final_result).strip()
            true_answer = str(example["true_answer"]).strip()
            is_correct = prediction.lower() == true_answer.lower()
            
            # Check for errors
            try:
                parsing_error = any("AgentParsingError" in str(step) for step in agent_memory)
            except:
                parsing_error = False
            
            iteration_limit_exceeded = "Agent stopped due to iteration limit or time limit." in prediction
            
            # Get token counts from tracking
            try:
                # Get token counts from our enhanced tracking system
                session_summary = run_manager.token_tracker.get_session_summary()
                total_token_counts = {
                    "input": session_summary.get("total_input_tokens", 0),
                    "output": session_summary.get("total_output_tokens", 0)
                }
            except Exception as e:
                print(f"⚠️  Token tracking error: {e}")
                total_token_counts = {"input": 0, "output": 0}

            result = {
                "question_index": question_index,
                "question": example["question"],
                "true_answer": true_answer,
                "prediction": prediction,
                "is_correct": is_correct,
                "task_level": example["task"],
                "task_id": example["task_id"],
                "file_name": example.get("file_name", ""),
                "augmented_question": augmented_question,
                "parsing_error": parsing_error,
                "iteration_limit_exceeded": iteration_limit_exceeded,
                "attempts_made": attempt + 1,
                "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
                "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration_seconds": (end_time - start_time).total_seconds(),
                "token_counts": total_token_counts,
                "agent_error": None,
                "intermediate_steps": serializable_memory  # Aggiunta della cronologia completa
            }
            
            print(f"✅ Question {question_index + 1} completed successfully (attempt {attempt + 1})")
            return result
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Question {question_index + 1} failed on attempt {attempt + 1}: {error_msg}")
            
            if attempt < max_retries:
                delay = base_delay * (backoff_factor ** attempt)
                print(f"⏳ Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                # Max retries exceeded, return error result
                result = {
                    "question_index": question_index,
                    "question": example["question"],
                    "true_answer": str(example["true_answer"]).strip(),
                    "prediction": None,
                    "is_correct": False,
                    "task_level": example["task"],
                    "task_id": example["task_id"],
                    "file_name": example.get("file_name", ""),
                    "augmented_question": None,
                    "parsing_error": False,
                    "iteration_limit_exceeded": False,
                    "attempts_made": max_retries + 1,
                    "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "duration_seconds": 0,
                    "token_counts": {"input": 0, "output": 0},
                    "agent_error": error_msg
                }
                return result


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


def main():
    args = parse_args()
    print(f"🚀 Starting GAIA subset evaluation with arguments: {args}")

    # Load configuration
    config = load_subset_config(args.config)
    experiment_id = config["experiment_id"]
    dataset_limits = config["dataset_limits"]
    retry_config = config["retry_config"]
    processing_config = config.get("processing", {"concurrency": 4})
    seed = config.get("random_seed", 42)
    
    print(f"📊 Experiment: {experiment_id}")
    print(f"🎯 Dataset limits: {dataset_limits}")
    print(f"🎲 Random seed: {seed}")
    print(f"⚡ Concurrency: {processing_config['concurrency']}")

    # Load models from config
    models = load_agent_models(config["agents"])
    
    # Initialize enhanced tracking system
    run_manager = EnhancedRunManager()
    session_dir = run_manager.setup_session(experiment_id, f"subset_{args.run_name}")
    
    # Load and create subset dataset
    eval_ds = load_gaia_dataset(args.use_raw_dataset, args.set_to_run)
    subset_questions = create_deterministic_subset(eval_ds, dataset_limits, seed)
    
    if len(subset_questions) == 0:
        print("❌ No questions selected! Check your dataset limits.")
        return
    
    # Start tracking
    run_manager.start_run()
    
    # Process questions with enhanced tracking
    results = []
    concurrency = processing_config["concurrency"]
    
    if concurrency == 1:
        # Sequential processing
        print("🔄 Processing questions sequentially...")
        for i, example in enumerate(tqdm(subset_questions, desc="Processing questions")):
            result = answer_single_question_with_tracking(
                example, models, run_manager, retry_config, i
            )
            results.append(result)
    else:
        # Parallel processing
        print(f"🚀 Processing questions with {concurrency} workers...")
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(
                    answer_single_question_with_tracking, 
                    example, models, run_manager, retry_config, i
                )
                for i, example in enumerate(subset_questions)
            ]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing questions"):
                result = future.result()
                results.append(result)
    
    # Sort results by question index with robust error handling
    try:
        results.sort(key=lambda x: x.get("question_index", 0))
        print(f"✅ Sorted {len(results)} results by question index")
    except Exception as e:
        print(f"⚠️ Error sorting results: {e}")
        print("📋 Results will be processed in current order")

    # Calculate final metrics
    final_metrics = calculate_final_metrics(results)
    
    # Save all results and tracking data using modular function
    saved_files = save_results_to_files(results, final_metrics, config, args.run_name)
    
    # Save enhanced tracking data
    run_manager.end_time = time.time()
    execution_time = run_manager.end_time - run_manager.start_time
    
    run_manager.save_session_summary(
        session_dir=session_dir,
        experiment_id=experiment_id,
        execution_time=execution_time,
        custom_metrics=final_metrics
    )
    
    # Print comprehensive results
    print("\n" + "="*60)
    print(f"🎯 EXPERIMENT COMPLETED: {experiment_id}")
    print("="*60)
    print(f"📊 Overall Accuracy: {final_metrics['overall_accuracy']:.1f}%")
    print(f"✅ Correct Answers: {final_metrics['correct_answers']}/{final_metrics['total_questions']}")
    
    print("\n📈 Task Level Breakdown:")
    for task, metrics in final_metrics['task_breakdown'].items():
        print(f"  {task}: {metrics['correct']}/{metrics['total']} ({metrics['accuracy']:.1f}%)")
    
    print(f"\n💰 Token Usage:")
    print(f"  Input tokens: {final_metrics['token_usage']['total_input_tokens']:,}")
    print(f"  Output tokens: {final_metrics['token_usage']['total_output_tokens']:,}")
    print(f"  Total tokens: {final_metrics['token_usage']['total_tokens']:,}")
    
    # Use configurable cost estimation
    cost_estimator = CostEstimator(getattr(args, 'cost_config', None))
    
    # Estimate cost using default model (can be enhanced to track actual models used)
    cost_estimate = cost_estimator.estimate_cost(
        "gpt-4o-mini",  # Default model - this could be made configurable
        final_metrics['token_usage']['total_input_tokens'],
        final_metrics['token_usage']['total_output_tokens']
    )
    print(f"  Estimated cost ({cost_estimate['provider']} {cost_estimate['model']}): ${cost_estimate['total_cost']:.4f}")
    print(f"    Input cost: ${cost_estimate['input_cost']:.4f}")
    print(f"    Output cost: ${cost_estimate['output_cost']:.4f}")
    
    if final_metrics['error_analysis']['total_errors'] > 0:
        print(f"\n⚠️  Errors encountered: {final_metrics['error_analysis']['total_errors']}")
        print(f"   Parsing: {final_metrics['error_analysis']['parsing_errors']}")
        print(f"   Iteration limit: {final_metrics['error_analysis']['iteration_limit_errors']}")
        print(f"   Agent errors: {final_metrics['error_analysis']['agent_errors']}")
    
    print(f"\n📁 Results saved to: {session_dir}")
    print(f"📄 Files: {', '.join([f.name for f in saved_files])}")
    print(f"⏱️  Total execution time: {execution_time:.1f} seconds")


if __name__ == "__main__":
    main()
