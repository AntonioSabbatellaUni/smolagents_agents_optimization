"""Metrics logging utilities for saving comprehensive experiment data."""

import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import asdict

from .token_tracker import TokenTracker, ModelCall
from .cost_calculator import CostCalculator
from .session_manager import SessionManager


class MetricsLogger:
    """Comprehensive metrics logging for experiments."""
    
    def __init__(self, 
                 session_manager: SessionManager,
                 token_tracker: TokenTracker,
                 cost_calculator: CostCalculator):
        """Initialize metrics logger.
        
        Args:
            session_manager: Session manager instance
            token_tracker: Token tracker instance  
            cost_calculator: Cost calculator instance
        """
        self.session_manager = session_manager
        self.token_tracker = token_tracker
        self.cost_calculator = cost_calculator
    
    def save_comprehensive_trace(self, 
                                question: str,
                                answer: Any,
                                agent_messages: List[Dict],
                                run_result: Any,
                                additional_metadata: Optional[Dict] = None) -> Dict[str, Path]:
        """Save comprehensive trace information.
        
        Args:
            question: The input question
            answer: The final answer
            agent_messages: Full agent message history
            run_result: The full run result object
            additional_metadata: Additional metadata to include
            
        Returns:
            Dictionary mapping file types to their paths
        """
        if not self.session_manager.current_session_dir:
            raise ValueError("No active session")
        
        session_info = self.session_manager.get_session_info()
        session_summary = self.token_tracker.get_session_summary()
        all_calls = self.token_tracker.get_all_calls()
        usage_data = self.token_tracker.get_usage_for_cost_calculation()
        cost_analysis = self.cost_calculator.calculate_batch_cost(usage_data)
        
        saved_files = {}
        
        # 1. Main trace file (enhanced version of original)
        trace_data = {
            "session_info": session_info,
            "question": question,
            "answer": answer,
            "token_usage_summary": session_summary,
            "cost_analysis": cost_analysis,
            "messages": agent_messages,
            "run_result": getattr(run_result, "dict", lambda: str(run_result))() if hasattr(run_result, "dict") else str(run_result),
            "metadata": additional_metadata or {}
        }
        
        trace_path = self.session_manager.get_file_path("comprehensive_trace.json")
        with open(trace_path, "w") as f:
            json.dump(self._make_serializable(trace_data), f, indent=2)
        saved_files["trace"] = trace_path
        
        # 2. Detailed cost breakdown
        cost_path = self.session_manager.get_file_path("cost_analysis.json")
        with open(cost_path, "w") as f:
            json.dump(self._make_serializable(cost_analysis), f, indent=2)
        saved_files["cost"] = cost_path
        
        # 3. Token usage per model/agent
        usage_path = self.session_manager.get_file_path("token_usage.json")
        model_stats = self.token_tracker.get_model_stats()
        usage_data = {
            "session_summary": session_summary,
            "model_statistics": {k: self._model_stats_to_dict(v) for k, v in model_stats.items()},
            "detailed_calls": [self._model_call_to_dict(call) for call in all_calls]
        }
        with open(usage_path, "w") as f:
            json.dump(self._make_serializable(usage_data), f, indent=2)
        saved_files["usage"] = usage_path
        
        # 4. CSV export for easy analysis
        csv_path = self.session_manager.get_file_path("model_calls.csv")
        self._save_calls_to_csv(all_calls, csv_path)
        saved_files["csv"] = csv_path
        
        # 5. Session summary
        summary_path = self.session_manager.get_file_path("session_summary.json")
        summary_data = {
            "session_info": session_info,
            "performance_summary": session_summary,
            "cost_summary": {
                "total_cost_usd": cost_analysis["total_cost"],
                "total_tokens": cost_analysis["total_tokens"],
                "models_used": len(cost_analysis["cost_breakdown"]),
                "cost_per_model": {
                    model: data["total_cost"] 
                    for model, data in cost_analysis["cost_breakdown"].items()
                }
            },
            "query_info": {
                "question": question,
                "answer_preview": str(answer)[:200] + "..." if len(str(answer)) > 200 else str(answer)
            }
        }
        with open(summary_path, "w") as f:
            json.dump(self._make_serializable(summary_data), f, indent=2)
        saved_files["summary"] = summary_path
        
        return saved_files
    
    def save_intermediate_checkpoint(self, step_name: str, data: Dict[str, Any]):
        """Save intermediate checkpoint data.
        
        Args:
            step_name: Name of the step/checkpoint
            data: Data to save
        """
        if not self.session_manager.current_session_dir:
            return
        
        checkpoint_dir = self.session_manager.current_session_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%H%M%S")
        filename = f"{timestamp}_{step_name}.json"
        
        checkpoint_path = checkpoint_dir / filename
        with open(checkpoint_path, "w") as f:
            json.dump(self._make_serializable(data), f, indent=2)
    
    def _make_serializable(self, obj: Any, _seen: set = None) -> Any:
        """Make object JSON serializable with circular reference protection."""
        if _seen is None:
            _seen = set()
        
        # Check for circular references
        obj_id = id(obj)
        if obj_id in _seen:
            return f"<circular reference to {type(obj).__name__}>"
        
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif isinstance(obj, (list, tuple)):
            _seen.add(obj_id)
            try:
                result = [self._make_serializable(item, _seen) for item in obj]
            finally:
                _seen.discard(obj_id)
            return result
        elif isinstance(obj, dict):
            _seen.add(obj_id)
            try:
                result = {k: self._make_serializable(v, _seen) for k, v in obj.items()}
            finally:
                _seen.discard(obj_id)
            return result
        elif hasattr(obj, "__dict__"):
            _seen.add(obj_id)
            try:
                # Filter out private attributes and known problematic ones
                safe_attrs = {
                    k: v for k, v in obj.__dict__.items() 
                    if not k.startswith('_') and not callable(v)
                }
                result = {k: self._make_serializable(v, _seen) for k, v in safe_attrs.items()}
            finally:
                _seen.discard(obj_id)
            return result
        else:
            return str(obj)
    
    def _model_call_to_dict(self, call: ModelCall) -> Dict[str, Any]:
        """Convert ModelCall to dictionary."""
        return {
            "timestamp": call.timestamp.isoformat(),
            "model_id": call.model_id,
            "agent_name": call.agent_name,
            "input_tokens": call.input_tokens,
            "output_tokens": call.output_tokens,
            "total_tokens": call.input_tokens + call.output_tokens,
            "input_content": call.input_content,
            "output_content": call.output_content,
            "step_number": call.step_number,
            "call_duration": call.call_duration,
            "metadata": call.metadata
        }
    
    def _model_stats_to_dict(self, stats) -> Dict[str, Any]:
        """Convert ModelUsageStats to dictionary."""
        return {
            "model_id": stats.model_id,
            "total_calls": stats.total_calls,
            "total_input_tokens": stats.total_input_tokens,
            "total_output_tokens": stats.total_output_tokens,
            "total_tokens": stats.total_tokens,
            "total_duration": stats.total_duration,
            "avg_duration": stats.avg_duration,
            "first_call": stats.first_call.isoformat() if stats.first_call else None,
            "last_call": stats.last_call.isoformat() if stats.last_call else None
        }
    
    def _save_calls_to_csv(self, calls: List[ModelCall], csv_path: Path):
        """Save model calls to CSV file."""
        if not calls:
            return
        
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                "timestamp", "model_id", "agent_name", "step_number",
                "input_tokens", "output_tokens", "total_tokens", 
                "call_duration", "input_content_preview", "output_content_preview"
            ])
            
            # Data rows
            for call in calls:
                input_preview = call.input_content[:100] + "..." if len(call.input_content) > 100 else call.input_content
                output_preview = call.output_content[:100] + "..." if len(call.output_content) > 100 else call.output_content
                
                writer.writerow([
                    call.timestamp.isoformat(),
                    call.model_id,
                    call.agent_name,
                    call.step_number,
                    call.input_tokens,
                    call.output_tokens,
                    call.input_tokens + call.output_tokens,
                    call.call_duration,
                    input_preview,
                    output_preview
                ])
    
    def generate_cost_report(self) -> str:
        """Generate a formatted cost report.
        
        Returns:
            Formatted cost report as string
        """
        usage_data = self.token_tracker.get_usage_for_cost_calculation()
        cost_analysis = self.cost_calculator.calculate_batch_cost(usage_data)
        session_summary = self.token_tracker.get_session_summary()
        
        report = []
        report.append("=" * 60)
        report.append("COST ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"Total Cost: ${cost_analysis['total_cost']:.4f}")
        report.append(f"Total Tokens: {cost_analysis['total_tokens']:,}")
        report.append(f"Models Used: {cost_analysis['models_used']}")
        report.append(f"Total Calls: {session_summary['total_calls']}")
        report.append("")
        
        report.append("COST BREAKDOWN BY MODEL:")
        report.append("-" * 40)
        for model_id, data in cost_analysis['cost_breakdown'].items():
            report.append(f"{model_id}:")
            report.append(f"  Cost: ${data['total_cost']:.4f}")
            report.append(f"  Calls: {data['call_count']}")
            report.append(f"  Tokens: {data['input_tokens'] + data['output_tokens']:,}")
            report.append(f"  Provider: {data['provider']}")
            report.append("")
        
        return "\n".join(report)
