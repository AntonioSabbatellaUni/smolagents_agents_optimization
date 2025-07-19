# Agent creation and management utilities for GAIA evaluation
import os
from typing import Dict
from pathlib import Path

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
from scripts.visual_qa import VisualQATool
from scripts.run_agents import get_single_file_description, get_zip_description

from smolagents import CodeAgent, GoogleSearchTool, ToolCallingAgent
from utils import EnhancedRunManager


BROWSER_CONFIG = {
    "viewport_size": 1024 * 5,
    "downloads_folder": "downloads_folder",
    "request_kwargs": {
        "headers": {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
        "timeout": 300,
    },
    "serpapi_key": os.getenv("SERPAPI_API_KEY"),
}

os.makedirs(f"./{BROWSER_CONFIG['downloads_folder']}", exist_ok=True)


def create_agent_team_with_tracking(models: Dict, run_manager: EnhancedRunManager):
    """Create agent team with enhanced tracking integration."""
    text_limit = 100000
    
    # Wrap models with tracking
    tracked_models = run_manager.wrap_models(models)
    
    ti_tool = TextInspectorTool(tracked_models["text_inspector"], text_limit)
    visual_qa_tool = VisualQATool(tracked_models["visual_qa"])
    browser = SimpleTextBrowser(**BROWSER_CONFIG)

    WEB_TOOLS = [
        GoogleSearchTool(provider="serper"),
        VisitTool(browser),
        PageUpTool(browser),
        PageDownTool(browser),
        FinderTool(browser),
        FindNextTool(browser),
        ArchiveSearchTool(browser),
        TextInspectorTool(tracked_models["text_inspector"], text_limit),
    ]

    text_webbrowser_agent = ToolCallingAgent(
        model=tracked_models["search_agent"],  # Use dedicated search agent model
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
        model=tracked_models["manager"],  # Use dedicated manager model
        tools=[visual_qa_tool, ti_tool],
        max_steps=12,
        verbosity_level=2,
        additional_authorized_imports=["*"],
        planning_interval=4,
        managed_agents=[text_webbrowser_agent],
    )
    return manager_agent


def process_file_attachments(example: dict, models: Dict, run_manager=None) -> str:
    """Process file attachments and return augmented question."""
    augmented_question = f"""You have one question to answer. It is paramount that you provide a correct answer.
Give it all you can: I know for a fact that you have access to all the relevant tools to solve it and find the correct answer (the answer does exist).
Failure or 'I cannot answer' or 'None found' will not be tolerated, success will be rewarded.
Run verification steps if that's needed, you must make sure you find the correct answer! Here is the task:

{example["question"]}"""

    # Add file processing if file exists
    if example.get("file_name") and len(str(example["file_name"])) > 0:
        file_path = Path(example["file_name"])
        if file_path.exists():
            # Create tools for file processing (using tracked models if available)
            if run_manager:
                tracked_models = run_manager.wrap_models(models)
                visual_tool = VisualQATool(tracked_models["visual_qa"])
                text_tool = TextInspectorTool(tracked_models["text_inspector"], 100000)
            else:
                from scripts.visual_qa import visualizer
                visual_tool = visualizer
                text_tool = TextInspectorTool(models["text_inspector"], 100000)
            
            if ".zip" in example["file_name"]:
                prompt_use_files = "\n\nTo solve the task above, you will have to use these attached files:\n"
                prompt_use_files += get_zip_description(
                    example["file_name"], example["question"], 
                    visual_tool, text_tool
                )
            else:
                prompt_use_files = "\n\nTo solve the task above, you will have to use this attached file:\n"
                prompt_use_files += get_single_file_description(
                    example["file_name"], example["question"], 
                    visual_tool, text_tool
                )
            augmented_question += prompt_use_files
        else:
            print(f"⚠️  File {example['file_name']} not found, proceeding without file context")
    
    return augmented_question


def create_gaia_dataset_loader(use_raw_dataset: bool, set_to_run: str):
    """Create a GAIA dataset loader function configured for the specific setup."""
    if not os.path.exists("data/gaia"):
        from huggingface_hub import snapshot_download
        
        if use_raw_dataset:
            snapshot_download(
                repo_id="gaia-benchmark/GAIA",
                repo_type="dataset",
                local_dir="data/gaia",
                ignore_patterns=[".gitattributes", "README.md"],
            )
        else:
            # WARNING: this dataset is gated: make sure you visit the repo to require access.
            snapshot_download(
                repo_id="smolagents/GAIA-annotated",
                repo_type="dataset",
                local_dir="data/gaia",
                ignore_patterns=[".gitattributes", "README.md"],
            )

    def preprocess_file_paths(row):
        if len(row["file_name"]) > 0:
            row["file_name"] = f"data/gaia/{set_to_run}/" + row["file_name"]
        return row

    import datasets
    eval_ds = datasets.load_dataset(
        "data/gaia/GAIA.py",
        name="2023_all",
        split=set_to_run,
        # data_files={"validation": "validation/metadata.jsonl", "test": "test/metadata.jsonl"},
    )

    eval_ds = eval_ds.rename_columns({"Question": "question", "Final answer": "true_answer", "Level": "task"})
    eval_ds = eval_ds.map(preprocess_file_paths)
    return eval_ds
