# Agent creation and management utilities for GAIA evaluation
import os
import threading
import time
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

from smolagents import CodeAgent, GoogleSearchTool, ToolCallingAgent, DuckDuckGoSearchTool
from utils import EnhancedRunManager
from utils.retry import call_with_retry, search_error_retryable


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


class RetryingDuckDuckGoSearchTool(DuckDuckGoSearchTool):
    """DuckDuckGo search with (a) a process-wide minimum interval between requests (shared by all question threads) and
    (b) retry with exponential backoff + jitter on rate-limit / blocked / transient errors (max MALBO_MAX_TRIES, each retry logged)."""
    _glock = threading.Lock()
    _glast = 0.0

    def _enforce_rate_limit(self) -> None:
        if not self._min_interval:
            return
        with RetryingDuckDuckGoSearchTool._glock:
            wait = RetryingDuckDuckGoSearchTool._glast + self._min_interval - time.time()
            if wait > 0:
                time.sleep(wait)
            RetryingDuckDuckGoSearchTool._glast = time.time()

    def forward(self, query: str) -> str:
        from utils import abort, llm_logger
        abort.check(getattr(llm_logger._ctx, "task_id", None))
        try:
            return call_with_retry(lambda: DuckDuckGoSearchTool.forward(self, query), "search_retry", search_error_retryable,
                                   tool="web_search", query=query[:80])
        except Exception as e:
            if search_error_retryable(e):   # still blocked / rate-limited after all retries -> answers would be degraded: pause
                abort.trigger("web_search", None, e, task_id=getattr(llm_logger._ctx, "task_id", None), kind="infra")
                raise abort.InfraAbort(f"paused: web search blocked: {e}"[:300]) from e
            raise


def create_agent_team_with_tracking(models: Dict, run_manager: EnhancedRunManager, task_id: str = None):
    """Create agent team with enhanced tracking integration. Everything mutable (browser, tools, agents) is created per call,
    so one team per question is safe across threads; the browser downloads go to a per-question folder (task_id)."""
    text_limit = 100000
    
    # Wrap models with tracking
    tracked_models = run_manager.wrap_models(models)
    
    ti_tool = TextInspectorTool(tracked_models["text_inspector"], text_limit)
    visual_qa_tool = VisualQATool(tracked_models["visual_qa"])
    browser_cfg = dict(BROWSER_CONFIG, request_kwargs=dict(BROWSER_CONFIG["request_kwargs"]))
    if task_id:
        dl = os.path.join(BROWSER_CONFIG["downloads_folder"], str(task_id))
        os.makedirs(dl, exist_ok=True)
        browser_cfg["downloads_folder"] = dl
    browser = SimpleTextBrowser(**browser_cfg)

    WEB_TOOLS = [
        (GoogleSearchTool(provider="serper") if os.getenv("SERPER_API_KEY") else RetryingDuckDuckGoSearchTool()),  # revival: DDG needs no key
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
        max_steps=int(os.getenv("MALBO_SEARCH_MAX_STEPS", 20)),
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
        max_steps=int(os.getenv("MALBO_MANAGER_MAX_STEPS", 12)),
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
    if os.getenv("GAIA_LOCAL_QUESTIONS") or (not os.path.exists("data/gaia") and not os.getenv("HF_TOKEN")):
        # revival: GAIA is gated; without an HF token use the tiny local multi-hop-ish set
        import datasets as _ds
        path = os.getenv("GAIA_LOCAL_QUESTIONS", "local_questions.jsonl")
        print(f"ℹ️  Using local fallback questions from {path} (GAIA gated / no HF_TOKEN)")
        return _ds.load_dataset("json", data_files=path, split="train")
    if not os.path.exists("data/gaia") and os.getenv("HF_TOKEN"):
        # revival: the real gated GAIA repo only has 2023/<split>/metadata.parquet (no metadata.jsonl)
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id="gaia-benchmark/GAIA", repo_type="dataset", local_dir="data/gaia",
                          allow_patterns=[f"2023/{set_to_run}/*"], token=os.environ["HF_TOKEN"])
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

    # revival: GAIA.py (script dataset) is unsupported by datasets>=4 -> read metadata.jsonl directly
    import datasets
    import json as _json
    meta = next((m for m in (Path("data/gaia/2023") / set_to_run / "metadata.jsonl", Path("data/gaia") / set_to_run / "metadata.jsonl",
                             Path("data/gaia/2023") / set_to_run / "metadata.parquet", Path("data/gaia") / set_to_run / "metadata.parquet")
                 if m.exists()), None)
    if meta is None:
        raise FileNotFoundError("no GAIA metadata.jsonl/parquet under data/gaia")
    if meta.suffix == ".parquet":
        import pandas as _pd
        rows = _pd.read_parquet(meta).to_dict("records")
    else:
        rows = [_json.loads(l) for l in meta.read_text().splitlines() if l.strip()]
    for r in rows:
        r["file_name"] = str(meta.parent / r["file_name"]) if r.get("file_name") else ""
        r["task"] = str(r.pop("Level")); r["question"] = r.pop("Question"); r["true_answer"] = r.pop("Final answer")
        r.pop("Annotator Metadata", None)
    return datasets.Dataset.from_list(rows)
