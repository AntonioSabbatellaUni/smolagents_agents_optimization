"""Enhanced run manager to keep run.py clean."""

import time
import json
from pathlib import Path
from typing import Any, Dict
from datetime import datetime

from smolagents.utils import make_json_serializable
from .token_tracker import TokenTracker
from .cost_calculator import CostCalculator
from .session_manager import SessionManager
from .metrics_logger import MetricsLogger
from .tracked_model import TrackedModel


class EnhancedRunManager:
    """Manages the enhanced tracking workflow with minimal code changes."""
    
    def __init__(self):
        """Initialize the enhanced tracking system."""
        self.token_tracker = TokenTracker()
        self.cost_calculator = CostCalculator()
        self.session_manager = SessionManager()
        self.metrics_logger = MetricsLogger(
            self.session_manager, 
            self.token_tracker, 
            self.cost_calculator
        )
        self.start_time = None
        self.end_time = None
    
    def setup_session(self, experiment_id: str, question: str) -> Path:
        """Set up a new tracking session.
        
        Args:
            experiment_id: The experiment identifier
            question: The question being asked
            
        Returns:
            Path to the created session directory
        """
        # Create session with question info
        question_preview = question[:50] + "..." if len(question) > 50 else question
        session_dir = self.session_manager.create_session(
            experiment_id=experiment_id,
            question=question,
            custom_suffix=question_preview.replace(" ", "_").replace("?", "").replace(",", "")[:20]
        )
        
        # Save config copy
        self.session_manager.save_config_copy("agent_models.yaml")
        
        print(f"Created session: {session_dir}")
        print(f"Question: {question}")
        
        return session_dir
    
    def wrap_models(self, models: Dict[str, Any]) -> Dict[str, TrackedModel]:
        """Wrap models with tracking.
        
        Args:
            models: Dictionary of model name to model instance
            
        Returns:
            Dictionary of wrapped models
        """
        tracked_models = {}
        for name, model in models.items():
            tracked_models[name] = TrackedModel(model, self.token_tracker, agent_name=name)
        return tracked_models
    
    def start_run(self):
        """Mark the start of the agent run."""
        self.start_time = time.time()
    
    def finish_run_and_save(self, 
                           question: str,
                           run_result: Any,
                           agent,
                           experiment_id: str,
                           models: Dict[str, Any]) -> Dict[str, Path]:
        """Finish the run and save all tracking data.
        
        Args:
            question: The input question
            run_result: The agent run result
            agent: The agent that was run
            experiment_id: The experiment identifier
            models: The original models dictionary
            
        Returns:
            Dictionary mapping file types to their saved paths
        """
        self.end_time = time.time()
        execution_time = self.end_time - self.start_time
        
        # Gather trace info
        full_message_history = agent.write_memory_to_messages()
        full_steps = agent.memory.get_full_steps() if hasattr(agent.memory, 'get_full_steps') else None
        
        # Create original trace format for compatibility
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        original_trace = {
            "experiment_id": experiment_id,
            "timestamp": timestamp,
            "question": question,
            "answer": getattr(run_result, "output", run_result),
            "token_usage": getattr(run_result, "token_usage", None),
            "messages": full_message_history,
            "steps": full_steps,
            "models": {k: str(v) for k, v in models.items()},
            "execution_time": execution_time
        }
        
        # Save comprehensive tracking data
        saved_files = self.metrics_logger.save_comprehensive_trace(
            question=question,
            answer=getattr(run_result, "output", run_result),
            agent_messages=full_message_history,
            run_result=run_result,
            additional_metadata={
                "execution_time": execution_time,
                "original_models": {k: str(v) for k, v in models.items()}
            }
        )
        
        # Save original trace format for compatibility
        trace_path = self.session_manager.get_file_path("trace.json")
        with open(trace_path, "w") as f:
            json.dump(make_json_serializable(original_trace), f, indent=2)
        
        return saved_files
    
    def print_results(self, original_trace: Dict[str, Any], saved_files: Dict[str, Path]):
        """Print the results and cost analysis.
        
        Args:
            original_trace: The original trace data
            saved_files: Dictionary of saved file paths
        """
        # Generate and print cost report
        cost_report = self.metrics_logger.generate_cost_report()
        print("\n" + cost_report)
        
        # Save cost report to file
        cost_report_path = self.session_manager.get_file_path("cost_report.txt")
        with open(cost_report_path, "w") as f:
            f.write(cost_report)
        
        # Print summary
        session_dir = self.session_manager.get_session_dir()
        print(f"\nGot this answer: {original_trace['answer']}")
        print(f"Session saved to: {session_dir}")
        print(f"Files saved:")
        for file_type, path in saved_files.items():
            print(f"  {file_type}: {path.name}")
        print(f"  trace: trace.json (original format)")
        print(f"  cost_report: cost_report.txt")
    
    def get_token_tracker(self) -> TokenTracker:
        """Get the token tracker instance."""
        return self.token_tracker
    
    def save_session_summary(self, session_dir: Path, experiment_id: str,
                           execution_time: float, custom_metrics: dict = None):
        """Save session summary for batch processing without a single agent."""
        # Get usage data formatted for cost calculation
        usage_data = self.token_tracker.get_usage_for_cost_calculation()
        
        # Calculate total costs
        cost_result = self.cost_calculator.calculate_batch_cost(usage_data) if usage_data else {"total_cost": 0.0}
        total_cost = cost_result.get("total_cost", 0.0) if isinstance(cost_result, dict) else cost_result
        
        # Get session summary for other statistics
        session_summary = self.token_tracker.get_session_summary()
        
        # Create summary data
        summary_data = {
            "experiment_id": experiment_id,
            "execution_time_seconds": execution_time,
            "total_cost_usd": total_cost,
            "token_summary": session_summary,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        if custom_metrics:
            summary_data["custom_metrics"] = custom_metrics
        
        # Save tracking summary with proper JSON serialization
        tracking_file = session_dir / "tracking_summary.json"
        with open(tracking_file, 'w') as f:
            json.dump(make_json_serializable(summary_data), f, indent=2)
        
        # Save cost report
        cost_report_file = session_dir / "cost_report.txt"
        with open(cost_report_file, 'w') as f:
            f.write(f"Experiment Cost Report: {experiment_id}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total Cost: ${total_cost:.4f}\n")
            f.write(f"Execution Time: {execution_time:.2f} seconds\n\n")
            
            if custom_metrics:
                f.write("Performance Metrics:\n")
                f.write(f"  Overall Accuracy: {custom_metrics.get('overall_accuracy', 0):.2%}\n")
                f.write(f"  Total Questions: {custom_metrics.get('total_questions', 0)}\n")
                f.write(f"  Correct Answers: {custom_metrics.get('correct_answers', 0)}\n\n")

            # f.write("Token Usage by Model:\n")
            # for model_name, stats in session_summary.items():
            #     # Use the direct calculate_cost method instead of batch processing
            #     cost_info = self.cost_calculator.calculate_cost(
            #         model_id=model_name,
            #         input_tokens=stats.total_input_tokens,
            #         output_tokens=stats.total_output_tokens
            #     )
                
            #     f.write(f"  {model_name}:\n")
            #     f.write(f"    Input tokens: {stats.total_input_tokens:,}\n")
            #     f.write(f"    Output tokens: {stats.total_output_tokens:,}\n")
            #     f.write(f"    Cost: ${cost_info['total_cost']:.4f}\n")
            #     f.write(f"    Provider: {cost_info['provider']}\n\n")
            # Blocco corretto per utils/run_manager.py

            f.write("Token Usage by Model:\n")
            model_stats_data = self.token_tracker.get_model_stats() 
            for model_name, stats in model_stats_data.items():
                cost_info = self.cost_calculator.calculate_cost(
                    model_id=model_name,
                    input_tokens=stats.total_input_tokens,
                    output_tokens=stats.total_output_tokens
                )
                
                f.write(f"  {model_name}:\n")
                f.write(f"    Input tokens: {stats.total_input_tokens:,}\n")
                f.write(f"    Output tokens: {stats.total_output_tokens:,}\n")
                f.write(f"    Cost: ${cost_info['total_cost']:.4f}\n")
                f.write(f"    Provider: {cost_info['provider']}\n\n")
        return tracking_file, cost_report_file
