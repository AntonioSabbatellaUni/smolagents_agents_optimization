import os
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

_GO_MODELS = {}


def _supported_efforts(model_id: str):
    """Reasoning efforts a Go model accepts (/models capabilities.reasoning.effort); [] if unknown/unsupported."""
    if not _GO_MODELS:
        try:
            import json, urllib.request
            base = os.getenv("MALBO_API_BASE", "https://opencode.ai/zen/go/v1").rstrip("/")
            req = urllib.request.Request(base + "/models", headers={"User-Agent": "Mozilla/5.0 (malbo)"})
            for m in json.load(urllib.request.urlopen(req, timeout=20))["data"]:
                _GO_MODELS[m["id"]] = ((m.get("capabilities") or {}).get("reasoning") or {}).get("effort") or []
        except Exception as e:
            print(f"⚠️ could not read /models for reasoning-effort support: {e}")
            _GO_MODELS["__failed__"] = []
    return _GO_MODELS.get(model_id.split("/")[-1], [])


def apply_role_overrides(role: str, params: dict) -> dict:
    """Per-role env overrides: MALBO_MAX_TOKENS_<ROLE>, MALBO_REASONING_EFFORT[_<ROLE>] (role-specific wins).
    Effort is only sent when the model's /models metadata lists it (else skipped with a note)."""
    R = role.upper()
    if os.getenv(f"MALBO_MAX_TOKENS_{R}"):
        params["max_tokens"] = int(os.environ[f"MALBO_MAX_TOKENS_{R}"])
    effort = os.getenv(f"MALBO_REASONING_EFFORT_{R}") or os.getenv("MALBO_REASONING_EFFORT")
    if effort and effort != "none":
        if effort in _supported_efforts(params.get("model_id", "")):
            params["reasoning_effort"] = effort
        else:
            print(f"ℹ️ {role}: reasoning_effort={effort} not supported by {params.get('model_id')} -> not sent")
    return params


def load_agent_models(agent_configs):
    models = {}
    for agent, model_cfg in agent_configs.items():
        cls = MODEL_CLASSES[model_cfg["model_class"]]
        params = {k: (os.environ[v[4:]] if isinstance(v, str) and v.startswith("env:") else v)
                  for k, v in model_cfg.items() if k != "model_class"}
        models[agent] = cls(**apply_role_overrides(agent, params))
        print(f"🧩 {agent}: {params.get('model_id')} max_tokens={params.get('max_tokens')} effort={params.get('reasoning_effort')}")
    return models 