import base64
import json
import mimetypes
import os
import uuid
from io import BytesIO

import PIL.Image
import requests
from dotenv import load_dotenv

from smolagents import Tool, tool


load_dotenv()   # keep the env given to the container (driver key switch)


# Function to encode the image
def encode_image(image_path):
    if image_path.startswith("http"):
        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"
        request_kwargs = {
            "headers": {"User-Agent": user_agent},
            "stream": True,
        }

        # Send a HTTP request to the URL
        response = requests.get(image_path, **request_kwargs)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")

        extension = mimetypes.guess_extension(content_type)
        if extension is None:
            extension = ".download"

        fname = str(uuid.uuid4()) + extension
        download_path = os.path.abspath(os.path.join("downloads", fname))

        with open(download_path, "wb") as fh:
            for chunk in response.iter_content(chunk_size=512):
                fh.write(chunk)

        image_path = download_path

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def resize_image(image_path, max_side=1568):
    """Downscale (keeping aspect ratio) so the data URL stays small; returns the new path."""
    img = PIL.Image.open(image_path)
    width, height = img.size
    scale = min(1.0, max_side / max(width, height))
    img = img.convert("RGB").resize((max(1, int(width * scale)), max(1, int(height * scale))))
    base, _ = os.path.splitext(image_path)
    new_image_path = f"{base}_resized.jpg"
    img.save(new_image_path, format="JPEG", quality=90)
    return new_image_path


def image_to_data_url(image_path: str, max_bytes: int = 3_000_000) -> str:
    """Local file or http(s) URL -> `data:<mime>;base64,...` (png/jpeg kept as is when small, else re-encoded as JPEG)."""
    if image_path.startswith("http"):
        os.makedirs("downloads", exist_ok=True)
        image_path = _download(image_path)
    mime, _ = mimetypes.guess_type(image_path)
    if mime not in ("image/png", "image/jpeg") or os.path.getsize(image_path) > max_bytes:
        image_path = resize_image(image_path)
        mime = "image/jpeg"
    with open(image_path, "rb") as f:
        return f"data:{mime};base64,{base64.b64encode(f.read()).decode('utf-8')}"


def _download(url: str) -> str:
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
    response = requests.get(url, headers={"User-Agent": user_agent}, stream=True)
    response.raise_for_status()
    ext = mimetypes.guess_extension(response.headers.get("content-type", "").split(";")[0]) or ".download"
    path = os.path.abspath(os.path.join("downloads", str(uuid.uuid4()) + ext))
    with open(path, "wb") as fh:
        for chunk in response.iter_content(chunk_size=512):
            fh.write(chunk)
    return path


def vision_messages(image_path: str, question: str) -> list[dict]:
    """OpenAI-style content parts: text + image_url (base64 data URL)."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
            ],
        }
    ]


def vision_call_kwargs(model_id: str) -> dict:
    """Extra completion kwargs for the vision call. DeepSeek v4.x on Go is a reasoning model that burns the whole token
    budget on hidden reasoning for images (empty content, finish_reason=length) -> disable thinking for it.
    Override with env MALBO_VISUAL_QA_EXTRA_BODY (JSON dict, {} = nothing); optional hard cap with MALBO_VISUAL_QA_MAX_TOKENS."""
    kw = {}  # max_tokens comes from the role model config (>= 32768, see loader.apply_role_overrides)
    if os.getenv("MALBO_VISUAL_QA_MAX_TOKENS"):
        kw["max_tokens"] = int(os.environ["MALBO_VISUAL_QA_MAX_TOKENS"])
    raw = os.getenv("MALBO_VISUAL_QA_EXTRA_BODY")
    body = json.loads(raw) if raw else ({"thinking": {"type": "disabled"}} if "deepseek" in (model_id or "") else {})
    if body:
        kw["extra_body"] = body
    return kw


DEFAULT_VISUAL_QA_MODEL = os.getenv("MALBO_VISUAL_QA_MODEL_ID", "openai/deepseek-v4.1-flash")


class VisualQATool(Tool):
    """Answers questions about an image through a LiteLLM vision call (model = the `visual_qa` role model)."""

    name = "visualizer"
    description = "A tool that can answer questions about attached images."
    inputs = {
        "image_path": {
            "description": "The path to the image on which to answer the question",
            "type": "string",
        },
        "question": {"description": "the question to answer", "type": "string", "nullable": True},
    }
    output_type = "string"

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image_path: str, question: str | None = None) -> str:
        add_note = False
        if not question:
            add_note = True
            question = "Please write a detailed caption for this image."
        if not isinstance(image_path, str):
            raise Exception("You should provide at least `image_path` string argument to this tool!")
        if getattr(self.model, "flatten_messages_as_text", False):
            raise Exception("visual_qa model is configured with flatten_messages_as_text=True: images are not supported")
        # "image_url" content parts are passed through untouched by get_clean_message_list.
        from smolagents.models import ChatMessage

        resp = self.model([ChatMessage.from_dict(m) for m in vision_messages(image_path, question)],
                          **vision_call_kwargs(getattr(self.model, "model_id", "")))
        output = resp.content
        if isinstance(output, list):
            output = " ".join(p.get("text", "") for p in output if isinstance(p, dict))
        if not output:
            raise Exception("The vision model returned an empty answer (it may have run out of tokens while reasoning).")
        if add_note:
            output = (
                f"You did not provide a particular question, so here is a detailed caption for the image: {output}"
            )
        return output


# Helper to create VisualQATool from config-driven model
def create_visual_qa_tool(model):
    return VisualQATool(model)


@tool
def visualizer(image_path: str, question: str | None = None) -> str:
    """A tool that can answer questions about attached images.

    Args:
        image_path: The path to the image on which to answer the question. This should be a local path to downloaded image.
        question: The question to answer.
    """
    import litellm

    if not isinstance(image_path, str):
        raise Exception("You should provide at least `image_path` string argument to this tool!")
    add_note = not question
    question = question or "Please write a detailed caption for this image."
    kw = {}
    if os.getenv("MALBO_API_BASE"):
        kw = dict(api_base=os.environ["MALBO_API_BASE"], api_key=os.getenv("OPENCODE_API_KEY") or os.getenv("MALBO_API_KEY"))
    response = litellm.completion(model=DEFAULT_VISUAL_QA_MODEL, messages=vision_messages(image_path, question),
                                  **vision_call_kwargs(DEFAULT_VISUAL_QA_MODEL), **kw)
    output = response.choices[0].message.content or ""
    if add_note:
        output = f"You did not provide a particular question, so here is a detailed caption for the image: {output}"
    return output
