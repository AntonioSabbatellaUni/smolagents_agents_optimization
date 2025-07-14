"""Token tracking utilities for monitoring model usage."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import threading


@dataclass
class ModelCall:
    """Information about a single model call."""
    timestamp: datetime
    model_id: str
    agent_name: str
    input_tokens: int
    output_tokens: int
    input_content: str
    output_content: str
    step_number: int
    call_duration: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class ModelUsageStats:
    """Aggregated usage statistics for a model."""
    model_id: str
    total_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_duration: float = 0.0
    first_call: Optional[datetime] = None
    last_call: Optional[datetime] = None
    calls: List[ModelCall] = field(default_factory=list)
    
    def add_call(self, call: ModelCall):
        """Add a model call to the statistics."""
        self.calls.append(call)
        self.total_calls += 1
        self.total_input_tokens += call.input_tokens
        self.total_output_tokens += call.output_tokens
        self.total_duration += call.call_duration
        
        if self.first_call is None or call.timestamp < self.first_call:
            self.first_call = call.timestamp
        if self.last_call is None or call.timestamp > self.last_call:
            self.last_call = call.timestamp
    
    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens
    
    @property
    def avg_duration(self) -> float:
        return self.total_duration / self.total_calls if self.total_calls > 0 else 0.0


class TokenTracker:
    """Thread-safe tracker for model token usage across agents."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._model_stats: Dict[str, ModelUsageStats] = {}
        self._session_calls: List[ModelCall] = []
        self._step_counter = 0
    
    def track_model_call(self, 
                        model_id: str,
                        agent_name: str,
                        input_tokens: int,
                        output_tokens: int,
                        input_content: str,
                        output_content: str,
                        call_duration: float = 0.0,
                        metadata: Optional[Dict[str, Any]] = None) -> ModelCall:
        """Track a model call.
        
        Args:
            model_id: The model identifier
            agent_name: Name of the agent making the call
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens  
            input_content: The actual input content/prompt
            output_content: The actual output content/response
            call_duration: Duration of the call in seconds
            metadata: Additional metadata about the call
            
        Returns:
            ModelCall object representing this call
        """
        with self._lock:
            self._step_counter += 1
            
            call = ModelCall(
                timestamp=datetime.now(),
                model_id=model_id,
                agent_name=agent_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                input_content=input_content,
                output_content=output_content,
                step_number=self._step_counter,
                call_duration=call_duration,
                metadata=metadata or {}
            )
            
            # Add to session calls
            self._session_calls.append(call)
            
            # Update model statistics
            if model_id not in self._model_stats:
                self._model_stats[model_id] = ModelUsageStats(model_id=model_id)
            
            self._model_stats[model_id].add_call(call)
            
            return call
    
    def get_model_stats(self, model_id: Optional[str] = None) -> Dict[str, ModelUsageStats]:
        """Get usage statistics for models.
        
        Args:
            model_id: Specific model to get stats for, or None for all models
            
        Returns:
            Dictionary of model statistics
        """
        with self._lock:
            if model_id:
                return {model_id: self._model_stats.get(model_id, ModelUsageStats(model_id=model_id))}
            return self._model_stats.copy()
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of the current session.
        
        Returns:
            Dictionary with session summary statistics
        """
        with self._lock:
            total_calls = len(self._session_calls)
            total_input_tokens = sum(call.input_tokens for call in self._session_calls)
            total_output_tokens = sum(call.output_tokens for call in self._session_calls)
            total_duration = sum(call.call_duration for call in self._session_calls)
            
            models_used = list(self._model_stats.keys())
            agents_used = list(set(call.agent_name for call in self._session_calls))
            
            return {
                "total_calls": total_calls,
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "total_tokens": total_input_tokens + total_output_tokens,
                "total_duration": total_duration,
                "models_used": models_used,
                "agents_used": agents_used,
                "start_time": self._session_calls[0].timestamp if self._session_calls else None,
                "end_time": self._session_calls[-1].timestamp if self._session_calls else None
            }
    
    def get_all_calls(self) -> List[ModelCall]:
        """Get all calls made in this session.
        
        Returns:
            List of all ModelCall objects
        """
        with self._lock:
            return self._session_calls.copy()
    
    def reset_session(self):
        """Reset the session tracking."""
        with self._lock:
            self._session_calls.clear()
            self._model_stats.clear()
            self._step_counter = 0
    
    def get_usage_for_cost_calculation(self) -> List[Dict[str, Any]]:
        """Get usage data formatted for cost calculation.
        
        Returns:
            List of usage dictionaries suitable for CostCalculator
        """
        with self._lock:
            return [
                {
                    "model_id": call.model_id,
                    "input_tokens": call.input_tokens,
                    "output_tokens": call.output_tokens,
                    "agent_name": call.agent_name,
                    "timestamp": call.timestamp.isoformat(),
                    "step_number": call.step_number
                }
                for call in self._session_calls
            ]
