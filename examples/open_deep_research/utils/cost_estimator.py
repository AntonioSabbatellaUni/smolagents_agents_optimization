# Cost estimation utilities for different LLM providers
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


class CostEstimator:
    """Configurable cost estimation for different LLM models."""
    
    def __init__(self, cost_config_path: Optional[str] = None):
        """Initialize with cost configuration."""
        self.cost_config = self._load_cost_config(cost_config_path)
    
    def _load_cost_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load cost configuration from YAML file."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Default cost configuration
            return self._get_default_cost_config()
    
    def _get_default_cost_config(self) -> Dict[str, Any]:
        """Default cost configuration for common models."""
        return {
            "models": {
                # OpenAI Models
                "gpt-4o": {
                    "provider": "OpenAI",
                    "input_cost_per_1m_tokens": 2.50,
                    "output_cost_per_1m_tokens": 10.00,
                    "context_window": 128000
                },
                "gpt-4o-mini": {
                    "provider": "OpenAI", 
                    "input_cost_per_1m_tokens": 0.15,
                    "output_cost_per_1m_tokens": 0.60,
                    "context_window": 128000
                },
                "gpt-4-turbo": {
                    "provider": "OpenAI",
                    "input_cost_per_1m_tokens": 10.00,
                    "output_cost_per_1m_tokens": 30.00,
                    "context_window": 128000
                },
                "gpt-3.5-turbo": {
                    "provider": "OpenAI",
                    "input_cost_per_1m_tokens": 0.50,
                    "output_cost_per_1m_tokens": 1.50,
                    "context_window": 16385
                },
                
                # Anthropic Models
                "claude-3-5-sonnet-20241022": {
                    "provider": "Anthropic",
                    "input_cost_per_1m_tokens": 3.00,
                    "output_cost_per_1m_tokens": 15.00,
                    "context_window": 200000
                },
                "claude-3-haiku-20240307": {
                    "provider": "Anthropic",
                    "input_cost_per_1m_tokens": 0.25,
                    "output_cost_per_1m_tokens": 1.25,
                    "context_window": 200000
                },
                
                # Google Models
                "gemini-1.5-pro": {
                    "provider": "Google",
                    "input_cost_per_1m_tokens": 1.25,
                    "output_cost_per_1m_tokens": 5.00,
                    "context_window": 1000000
                },
                "gemini-1.5-flash": {
                    "provider": "Google",
                    "input_cost_per_1m_tokens": 0.075,
                    "output_cost_per_1m_tokens": 0.30,
                    "context_window": 1000000
                },
                
                # Meta Models (via Together AI)
                "meta-llama/Llama-3.1-70B-Instruct-Turbo": {
                    "provider": "Meta/Together AI",
                    "input_cost_per_1m_tokens": 0.88,
                    "output_cost_per_1m_tokens": 0.88,
                    "context_window": 128000
                },
                "meta-llama/Llama-3.1-8B-Instruct-Turbo": {
                    "provider": "Meta/Together AI", 
                    "input_cost_per_1m_tokens": 0.18,
                    "output_cost_per_1m_tokens": 0.18,
                    "context_window": 128000
                },
                
                # DeepSeek Models
                "deepseek-chat": {
                    "provider": "DeepSeek",
                    "input_cost_per_1m_tokens": 0.27,
                    "output_cost_per_1m_tokens": 1.10,
                    "context_window": 64000
                }
            },
            
            "default_model": "gpt-4o-mini"  # Fallback model for cost estimation
        }
    
    def estimate_cost(self, model_name: str, input_tokens: int, output_tokens: int) -> Dict[str, Any]:
        """Estimate cost for a specific model and token usage."""
        # Normalize model name (remove provider prefixes, etc.)
        normalized_name = self._normalize_model_name(model_name)
        
        # Get model config
        model_config = self.cost_config["models"].get(
            normalized_name, 
            self.cost_config["models"][self.cost_config["default_model"]]
        )
        
        # Calculate costs
        input_cost = (input_tokens / 1_000_000) * model_config["input_cost_per_1m_tokens"]
        output_cost = (output_tokens / 1_000_000) * model_config["output_cost_per_1m_tokens"]
        total_cost = input_cost + output_cost
        
        return {
            "model": normalized_name,
            "provider": model_config["provider"],
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "cost_per_1m_input": model_config["input_cost_per_1m_tokens"],
            "cost_per_1m_output": model_config["output_cost_per_1m_tokens"],
            "context_window": model_config["context_window"]
        }
    
    def estimate_multi_model_cost(self, model_usage: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
        """Estimate cost for multiple models with different token usage."""
        total_cost = 0.0
        total_input_tokens = 0
        total_output_tokens = 0
        model_breakdown = {}
        
        for model_name, tokens in model_usage.items():
            input_tokens = tokens.get("input", 0)
            output_tokens = tokens.get("output", 0)
            
            cost_estimate = self.estimate_cost(model_name, input_tokens, output_tokens)
            model_breakdown[model_name] = cost_estimate
            
            total_cost += cost_estimate["total_cost"]
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
        
        return {
            "total_cost": total_cost,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
            "model_breakdown": model_breakdown
        }
    
    def _normalize_model_name(self, model_name: str) -> str:
        """Normalize model name to match configuration keys."""
        # Remove common prefixes and normalize
        name = model_name.lower()
        
        # Handle common variations
        if "gpt-4o-mini" in name:
            return "gpt-4o-mini"
        elif "gpt-4o" in name:
            return "gpt-4o"
        elif "gpt-4-turbo" in name or "gpt-4-1106" in name:
            return "gpt-4-turbo"
        elif "gpt-3.5-turbo" in name:
            return "gpt-3.5-turbo"
        elif "claude-3-5-sonnet" in name or "claude-3.5-sonnet" in name:
            return "claude-3-5-sonnet-20241022"
        elif "claude-3-haiku" in name:
            return "claude-3-haiku-20240307"
        elif "gemini-1.5-pro" in name:
            return "gemini-1.5-pro"
        elif "gemini-1.5-flash" in name:
            return "gemini-1.5-flash"
        elif "llama-3.1-70b" in name:
            return "meta-llama/Llama-3.1-70B-Instruct-Turbo"
        elif "llama-3.1-8b" in name:
            return "meta-llama/Llama-3.1-8B-Instruct-Turbo"
        elif "deepseek" in name:
            return "deepseek-chat"
        
        # Return original if no match found
        return model_name
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get detailed information about a model."""
        normalized_name = self._normalize_model_name(model_name)
        return self.cost_config["models"].get(
            normalized_name,
            self.cost_config["models"][self.cost_config["default_model"]]
        )
    
    def list_supported_models(self) -> Dict[str, str]:
        """List all supported models and their providers."""
        return {
            model: config["provider"] 
            for model, config in self.cost_config["models"].items()
        }


def create_cost_config_template(output_path: str = "cost_config.yaml"):
    """Create a template cost configuration file."""
    estimator = CostEstimator()
    
    with open(output_path, 'w') as f:
        yaml.dump(estimator.cost_config, f, default_flow_style=False, indent=2)
    
    print(f"Cost configuration template created at: {output_path}")
    print("You can modify the costs and add new models as needed.")


if __name__ == "__main__":
    # Example usage
    estimator = CostEstimator()
    
    # Single model estimation
    cost = estimator.estimate_cost("gpt-4o-mini", 10000, 2000)
    print(f"Cost for gpt-4o-mini: ${cost['total_cost']:.4f}")
    
    # Multi-model estimation
    usage = {
        "gpt-4o-mini": {"input": 50000, "output": 10000},
        "claude-3-haiku-20240307": {"input": 30000, "output": 5000}
    }
    multi_cost = estimator.estimate_multi_model_cost(usage)
    print(f"Total multi-model cost: ${multi_cost['total_cost']:.4f}")
    
    # Create template
    create_cost_config_template("example_cost_config.yaml")
