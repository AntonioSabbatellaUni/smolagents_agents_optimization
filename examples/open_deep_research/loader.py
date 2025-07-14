import yaml
from smolagents import LiteLLMModel, InferenceClientModel

MODEL_CLASSES = {
    "LiteLLMModel": LiteLLMModel,
    "InferenceClientModel": InferenceClientModel,
}

def load_experiment_config(config_path="agent_models.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    experiment_id = config.get("experiment_id", "test-default")
    agent_configs = config["agents"]
    return experiment_id, agent_configs

def load_agent_models(agent_configs):
    models = {}
    for agent, model_cfg in agent_configs.items():
        cls = MODEL_CLASSES[model_cfg["model_class"]]
        params = {k: v for k, v in model_cfg.items() if k != "model_class"}
        models[agent] = cls(**params)
    return models 