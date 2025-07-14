import argparse
import os
import threading
import time
import json
import shutil
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import login
from scripts.text_inspector_tool import TextInspectorTool
from scripts.text_web_browser import (
    ArchiveSearchTool,
    FinderTool,
    FindNextTool,
    PageDownTool,
    PageUpTool,
    SimpleTextBrowser,
    VisitTool,
)
from scripts.visual_qa import visualizer
from loader import load_experiment_config, load_agent_models
from smolagents import (
    CodeAgent,
    GoogleSearchTool,
    ToolCallingAgent,
)
from smolagents.utils import make_json_serializable


load_dotenv(override=True)
login(os.getenv("HF_TOKEN"))

append_answer_lock = threading.Lock()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "question", type=str, help="for example: 'How many studio albums did Mercedes Sosa release before 2007?'"
    )
    return parser.parse_args()


custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}

user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"

BROWSER_CONFIG = {
    "viewport_size": 1024 * 5,
    "downloads_folder": "downloads_folder",
    "request_kwargs": {
        "headers": {"User-Agent": user_agent},
        "timeout": 300,
    },
    "serpapi_key": os.getenv("SERPAPI_API_KEY"),
}

os.makedirs(f"./{BROWSER_CONFIG['downloads_folder']}", exist_ok=True)


def get_experiment_folder(experiment_id):
    base = Path("experiments")
    folder = base / experiment_id
    if folder.exists():
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        folder = base / f"{experiment_id}_{timestamp}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def create_agent(models):
    text_limit = 100000
    browser = SimpleTextBrowser(**BROWSER_CONFIG)
    WEB_TOOLS = [
        GoogleSearchTool(provider="serper"),
        VisitTool(browser),
        PageUpTool(browser),
        PageDownTool(browser),
        FinderTool(browser),
        FindNextTool(browser),
        ArchiveSearchTool(browser),
        TextInspectorTool(models["text_inspector"], text_limit),
    ]
    text_webbrowser_agent = ToolCallingAgent(
        model=models["text_inspector"],
        tools=WEB_TOOLS,
        max_steps=20,
        verbosity_level=2,
        planning_interval=4,
        name="search_agent",
        description="""A team member that will search the internet to answer your question.
    Ask him for all your questions that require browsing the web.
    Provide him as much context as possible, in particular if you need to search on a specific timeframe!
    And don't hesitate to provide him with a complex search task, like finding a difference between two webpages.
    Your request must be a real sentence, not a google search! Like "Find me this information (...)" rather than a few keywords.
    """,
        provide_run_summary=True,
    )
    text_webbrowser_agent.prompt_templates["managed_agent"]["task"] += """You can navigate to .txt online files.
    If a non-html page is in another format, especially .pdf or a Youtube video, use tool 'inspect_file_as_text' to inspect it.
    Additionally, if after some searching you find out that you need more information to answer the question, you can use `final_answer` with your request for clarification as argument to request for more information."""

    manager_agent = CodeAgent(
        model=models["text_inspector"],
        tools=[visualizer, TextInspectorTool(models["text_inspector"], text_limit)],
        max_steps=12,
        verbosity_level=2,
        additional_authorized_imports=["*"],
        planning_interval=4,
        managed_agents=[text_webbrowser_agent],
    )

    return manager_agent


def main():
    args = parse_args()
    experiment_id, agent_configs = load_experiment_config("agent_models.yaml")
    models = load_agent_models(agent_configs)
    exp_folder = get_experiment_folder(experiment_id)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    # Save a copy of the YAML config used
    shutil.copy("agent_models.yaml", exp_folder / "agent_models.yaml")

    agent = create_agent(models)
    run_result = agent.run(args.question)

    # Gather trace info
    full_message_history = agent.write_memory_to_messages()
    full_steps = agent.memory.get_full_steps() if hasattr(agent.memory, 'get_full_steps') else None
    trace = {
        "experiment_id": experiment_id,
        "timestamp": timestamp,
        "question": args.question,
        "answer": getattr(run_result, "output", run_result),
        "token_usage": getattr(run_result, "token_usage", None),
        "messages": full_message_history,
        "steps": full_steps,
        "models": {k: str(v) for k, v in models.items()},
    }

    with open(exp_folder / "trace.json", "w") as f:
        json.dump(make_json_serializable(trace), f, indent=2)

    print(f"Got this answer: {trace['answer']}")
    print(f"Trace saved to {exp_folder}")


if __name__ == "__main__":
    main()
