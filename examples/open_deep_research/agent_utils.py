# Agent creation utilities for subset evaluation
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
from scripts.visual_qa import visualizer
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
        model=tracked_models["text_inspector"],
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
        model=tracked_models["text_inspector"],
        tools=[visualizer, ti_tool],
        max_steps=12,
        verbosity_level=2,
        additional_authorized_imports=["*"],
        planning_interval=4,
        managed_agents=[text_webbrowser_agent],
    )
    return manager_agent


def process_file_attachments(example: dict, models: Dict) -> str:
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
            if ".zip" in example["file_name"]:
                prompt_use_files = "\n\nTo solve the task above, you will have to use these attached files:\n"
                prompt_use_files += get_zip_description(
                    example["file_name"], example["question"], 
                    visualizer, TextInspectorTool(models["text_inspector"], 100000)
                )
            else:
                prompt_use_files = "\n\nTo solve the task above, you will have to use this attached file:\n"
                prompt_use_files += get_single_file_description(
                    example["file_name"], example["question"], 
                    visualizer, TextInspectorTool(models["text_inspector"], 100000)
                )
            augmented_question += prompt_use_files
        else:
            print(f"⚠️  File {example['file_name']} not found, proceeding without file context")
    
    return augmented_question
