"""Cost calculation utilities with comprehensive pricing table."""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ModelPricing:
    """Pricing information for a model."""
    input_cost_per_1m: float  # USD per 1M input tokens
    output_cost_per_1m: float  # USD per 1M output tokens
    provider: str = "unknown"


# Comprehensive pricing table based on July 2025 data
MODEL_PRICING: Dict[str, ModelPricing] = {
    # GPT Models
    "gpt-4.1": ModelPricing(2.00, 8.00, "openai"),
    "gpt-4.1-mini": ModelPricing(0.40, 1.60, "openai"),
    "gpt-4.1-nano": ModelPricing(0.10, 0.40, "openai"),
        # OpenAI Vision Models
    "gpt-4o": ModelPricing(
        input_cost_per_1m=2.50,
        output_cost_per_1m=10.0,
        provider="openai"
    ),
    
    "gpt-4o-mini": ModelPricing(0.40, 1.60, "openai"),
    "gpt-3.5-turbo": ModelPricing(0.50, 1.50, "openai"),
    
    # Anthropic Models
    "claude-3-5-sonnet": ModelPricing(3.00, 15.00, "anthropic"),
    "claude-3-sonnet": ModelPricing(3.00, 15.00, "anthropic"),
    "claude-3-haiku": ModelPricing(0.25, 1.25, "anthropic"),
    "anthropic/claude-3-5-sonnet-20240620": ModelPricing(3.00, 15.00, "anthropic"),
    "anthropic/claude-3-5-sonnet-latest": ModelPricing(3.00, 15.00, "anthropic"),
    
    # Google Models  
    "gemini-2.5-pro": ModelPricing(1.25, 10.00, "google"),  # Under 200k tokens
    "gemini-2.5-pro-large": ModelPricing(2.50, 15.00, "google"),  # Over 200k tokens
    "gemini-1.5-pro": ModelPricing(1.25, 5.00, "google"),
    "gemini-1.5-flash": ModelPricing(0.075, 0.30, "google"),
    
    # Meta Models (often via HuggingFace)
    "meta-llama/Llama-3.3-70B-Instruct": ModelPricing(0.60, 2.40, "huggingface"),
    "meta-llama/Llama-3.1-70B-Instruct": ModelPricing(0.60, 2.40, "huggingface"),
    "meta-llama/Llama-3.1-8B-Instruct": ModelPricing(0.15, 0.60, "huggingface"),
    
    # Qwen Models
    "Qwen/Qwen2.5-Coder-32B-Instruct": ModelPricing(0.40, 1.60, "huggingface"),
    "Qwen/Qwen2.5-72B-Instruct": ModelPricing(0.60, 2.40, "huggingface"),
    
    # HuggingFace Vision Models
    "HuggingFaceM4/idefics2-8b-chatty": ModelPricing(0.20, 0.80, "huggingface"),
    
    # SmolLM Models
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": ModelPricing(0.05, 0.20, "huggingface"),
    "HuggingFaceTB/SmolLM2-135M-Instruct": ModelPricing(0.01, 0.04, "huggingface"),
    
    # Other reasoning models
    "o3": ModelPricing(2.00, 8.00, "openai"),
    "o4-mini": ModelPricing(0.85, 4.40, "openai"),  # Average of range
    
    # OpenRouter Models - Moonshot AI (using actual pricing for optimization)
    "openrouter/moonshotai/kimi-k2": ModelPricing(0.14, 2.49, "openrouter/moonshotai"),
    "openrouter/moonshotai/kimi-k2:free": ModelPricing(0.14, 2.49, "openrouter/moonshotai"),
    
    # OpenRouter Models - Mistral
    "openrouter/mistralai/devstral-small": ModelPricing(0.07, 0.28, "openrouter/mistralai"),
    "openrouter/mistralai/mistral-small-3.2-24b-instruct": ModelPricing(0.05, 0.10, "openrouter/mistralai"),
    "openrouter/mistralai/mistral-small-3.2-24b-instruct:free": ModelPricing(0.05, 0.10, "openrouter/mistralai"),
    
    # OpenRouter Models - Google
    "openrouter/google/gemma-3n-e2b-it": ModelPricing(0.05, 0.10, "openrouter/google"),
    "openrouter/google/gemma-3n-e2b-it:free": ModelPricing(0.05, 0.10, "openrouter/google"),
    
    # OpenRouter Models - DeepSeek
    "openrouter/deepseek/deepseek-r1-0528": ModelPricing(0.272, 0.272, "openrouter/deepseek"),
    "openrouter/deepseek/deepseek-r1-0528:free": ModelPricing(0.272, 0.272, "openrouter/deepseek"),
    
    # OpenRouter Models - Qwen
    "openrouter/qwen/qwen3-32b": ModelPricing(0.10, 0.30, "openrouter/qwen"),
    "openrouter/qwen/qwen3-32b:free": ModelPricing(0.10, 0.30, "openrouter/qwen"),
    "openrouter/qwen/qwen3-235b-a22b": ModelPricing(0.13, 0.60, "openrouter/qwen"),
    "openrouter/qwen/qwen3-235b-a22b:free": ModelPricing(0.13, 0.60, "openrouter/qwen"),
    
    # Groq Models (typically cheaper due to fast inference)
    "groq/llama-3.3-70b": ModelPricing(0.50, 2.00, "groq"),
    "groq/llama-3.1-70b": ModelPricing(0.50, 2.00, "groq"),
    
    # Cerebras Models 
    "cerebras/llama-3.3-70b": ModelPricing(0.50, 2.00, "cerebras"),
    
    # Mistral Models
    "mistral/mistral-tiny": ModelPricing(0.15, 0.60, "mistral"),
    "mistral/mistral-small": ModelPricing(0.40, 1.60, "mistral"),
    "mistral/mistral-medium": ModelPricing(1.00, 4.00, "mistral"),
    "mistral/mistral-large": ModelPricing(2.00, 8.00, "mistral"),
}


class CostCalculator:
    """Calculate costs for LLM usage based on token consumption."""
    
    def __init__(self, custom_pricing: Optional[Dict[str, ModelPricing]] = None):
        """Initialize with optional custom pricing.
        
        Args:
            custom_pricing: Additional or override pricing information
        """
        self.pricing = MODEL_PRICING.copy()
        if custom_pricing:
            self.pricing.update(custom_pricing)
    
    def get_model_pricing(self, model_id: str) -> ModelPricing:
        """Get pricing for a specific model.
        
        Args:
            model_id: The model identifier
            
        Returns:
            ModelPricing object, defaults to GPT-4.1-nano pricing if not found
        """
        # Try exact match first
        if model_id in self.pricing:
            return self.pricing[model_id]
        
        # Try partial matches for common patterns
        for known_model, pricing in self.pricing.items():
            if known_model.lower() in model_id.lower() or model_id.lower() in known_model.lower():
                return pricing
        
        # Default to GPT-4.1-nano pricing as specified in requirements
        return self.pricing["gpt-4.1-nano"]
    
    def calculate_cost(self, model_id: str, input_tokens: int, output_tokens: int) -> Dict[str, float]:
        """Calculate cost for token usage.
        
        Args:
            model_id: The model identifier
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Dictionary with cost breakdown
        """
        pricing = self.get_model_pricing(model_id)
        
        input_cost = (input_tokens / 1_000_000) * pricing.input_cost_per_1m
        output_cost = (output_tokens / 1_000_000) * pricing.output_cost_per_1m
        total_cost = input_cost + output_cost
        
        return {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "model_id": model_id,
            "provider": pricing.provider,
            "pricing_used": {
                "input_cost_per_1m": pricing.input_cost_per_1m,
                "output_cost_per_1m": pricing.output_cost_per_1m
            }
        }
    
    def calculate_batch_cost(self, usage_data: list) -> Dict:
        """Calculate total cost for multiple model calls.
        
        Args:
            usage_data: List of dictionaries with 'model_id', 'input_tokens', 'output_tokens'
            
        Returns:
            Dictionary with aggregated cost information
        """
        if not usage_data:
            return {
                "total_cost": 0.0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "cost_breakdown": {}
            }
            
        total_cost = 0.0
        total_input_tokens = 0
        total_output_tokens = 0
        cost_breakdown = {}
        
        for usage in usage_data:
            # Handle both dict and string cases for robustness
            if isinstance(usage, str):
                print(f"Warning: usage_data contains string instead of dict: {usage}")
                continue
                
            if not isinstance(usage, dict):
                print(f"Warning: usage_data contains non-dict item: {type(usage)}")
                continue
                
            model_id = usage.get("model_id", "unknown")
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            
            cost_info = self.calculate_cost(model_id, input_tokens, output_tokens)
            total_cost += cost_info["total_cost"]
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            total_output_tokens += output_tokens
            
            if model_id not in cost_breakdown:
                cost_breakdown[model_id] = {
                    "total_cost": 0.0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "call_count": 0,
                    "provider": cost_info["provider"]
                }
            
            cost_breakdown[model_id]["total_cost"] += cost_info["total_cost"]
            cost_breakdown[model_id]["input_tokens"] += input_tokens
            cost_breakdown[model_id]["output_tokens"] += output_tokens
            cost_breakdown[model_id]["call_count"] += 1
        
        return {
            "total_cost": total_cost,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
            "models_used": len(cost_breakdown),
            "cost_breakdown": cost_breakdown
        }
