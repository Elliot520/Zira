"""look_at_image: answers a question about a photo the user attached (Phase 3, 2026-09-27) - "what is in this
picture", "read the text", "describe her outfit" - with a local vision model through Ollama (minicpm-v by default,
already installed; VISION_MODEL changes it).

Memory: the vision model is ~5.5GB and the chat model up to ~5GB, too much side by side on this 16GB Mac next to
macOS. So, like a video (LLMMemoryReleaser), the chat model is unloaded first, the vision model answers with
keep_alive 0 (it leaves memory as soon as it has answered), and the chat model is loaded again to write the
reply in the user's own words and language. It shares the image/video generation lock, so it never runs on the GPU
at the same time as a generation.

When the chat model can see images itself (VISION_CHAT_MODELS, e.g. NEWLIGHT's Qwen3.5 4B), it answers the question
directly: no second model, no unloading, and it stays loaded for the reply.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import re
from pathlib import Path
from typing import Any, Callable

import httpx

from app.tools.base import Tool, ToolResult, current_request
from app.tools.image_safety import check_minor_safety

logger = logging.getLogger("jarvis.tools.look")

_UPLOAD_TAG = re.compile(r"\[Uploaded image: ([A-Za-z0-9._-]+)\]")
MAX_SIDE = 1344  # minicpm-v reads up to ~1.8 megapixels; larger photos only cost time


class LookAtImageTool(Tool):
    name = "look_at_image"
    description = (
        "Look at an image the user attached and answer a question about what it shows - describe it, read text in "
        "it, count or identify things, say what someone is wearing or doing. The user's message contains "
        "'[Uploaded image: <id>]'; copy that <id> exactly as image_id. Use this for questions about the picture, "
        "not to change it (edit_image) or to animate it (create_video)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "image_id": {"type": "string", "description": "The image's id, copied exactly from '[Uploaded image: <id>]'"},
            "question": {"type": "string", "description": "What to find out about the image, in English"},
        },
        "required": ["image_id", "question"],
    }

    def __init__(self, uploads_dir: Path, ollama_host: str, model: str = "minicpm-v", timeout: float = 240.0,
                 generation_lock: asyncio.Lock | None = None, llm_memory: Any = None,
                 chat_model: Callable[[], str] | None = None, seeing_chat_models: frozenset[str] = frozenset()) -> None:
        self._uploads = uploads_dir
        self._host = ollama_host.rstrip("/")
        self.model = model
        self._chat_model = chat_model  # the chat model in use right now (a mode switch changes it)
        self._seeing = seeing_chat_models  # chat models that can look at images themselves
        self._timeout = timeout
        self._lock = generation_lock or asyncio.Lock()
        self._llm_memory = llm_memory  # the chat model steps aside while the vision model looks (LLMMemoryReleaser)

    def relevant(self, text: str) -> bool:
        return "[Uploaded image:" in text

    def describe(self, arguments: dict[str, Any]) -> str:
        return "Looking at the image"

    def _photo(self, image_id: str) -> str:
        """The attached photo, oriented and no larger than MAX_SIDE, as base64 JPEG."""
        from PIL import Image, ImageOps

        uploads = self._uploads.resolve()
        path = (uploads / image_id).resolve()
        if path.parent != uploads or not path.is_file():
            raise FileNotFoundError(f"No uploaded image found with id '{image_id}'.")
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((MAX_SIDE, MAX_SIDE))
            buffer = io.BytesIO()
            image.save(buffer, "JPEG", quality=90)
        return base64.b64encode(buffer.getvalue()).decode()

    async def execute(self, **arguments: Any) -> ToolResult:
        question = arguments.get("question")
        question = question.strip() if isinstance(question, str) and question.strip() else "Describe this image in detail."
        image_id = arguments.get("image_id")
        image_id = image_id.strip() if isinstance(image_id, str) and image_id.strip() else None
        if image_id is None:  # the model forgot the id: the tag in the user's own message
            match = _UPLOAD_TAG.search(current_request.get())
            image_id = match.group(1) if match else None
        if image_id is None:
            return ToolResult.failure("look_at_image needs the image_id from '[Uploaded image: <id>]'.")
        refusal = check_minor_safety(question)  # the existing always-on minor check, as for the other image tools
        if refusal:
            return ToolResult.failure(refusal)
        try:
            photo = await asyncio.to_thread(self._photo, image_id)
        except FileNotFoundError as exc:
            return ToolResult.failure(str(exc))
        except Exception as exc:  # noqa: BLE001
            return ToolResult.failure(f"Could not open the attached image: {exc}")

        chat_model = self._chat_model() if self._chat_model else None
        if chat_model and chat_model in self._seeing:  # the chat model can see: ask it, nothing to swap
            async with self._lock:
                try:
                    answer = await self._ask(photo, question, model=chat_model, keep_alive=None)
                except Exception as exc:  # noqa: BLE001
                    return ToolResult.failure(f"Could not look at the image: {type(exc).__name__}: {exc}")
            if not answer:
                return ToolResult.failure("The model gave no answer about the image.")
            return ToolResult.success(f"What the image shows ({chat_model}): {answer}")

        async with self._lock:
            if self._llm_memory is not None:
                await self._llm_memory.release()
            try:
                answer = await self._ask(photo, question)
            except httpx.HTTPStatusError as exc:
                missing = exc.response.status_code == 404
                return ToolResult.failure(
                    f"The vision model {self.model} is not installed (ollama pull {self.model})." if missing
                    else f"The vision model failed: {exc.response.text[:200]}"
                )
            except Exception as exc:  # noqa: BLE001 - timeout, Ollama down, ...
                return ToolResult.failure(f"Could not look at the image: {type(exc).__name__}: {exc}")
            finally:
                if self._llm_memory is not None:
                    await self._llm_memory.restore()
        if not answer:
            return ToolResult.failure("The vision model gave no answer about the image.")
        return ToolResult.success(f"What the image shows ({self.model}): {answer}")

    async def _ask(self, photo: str, question: str, model: str | None = None, keep_alive: Any = 0) -> str:
        """keep_alive 0: the separate vision model leaves memory as soon as it has answered; None: the chat model's
        own keep-alive (it is needed again right after, for the reply)."""
        body: dict[str, Any] = {
            "model": model or self.model,
            "messages": [{"role": "user", "content": question, "images": [photo]}],
            "stream": False,
            "think": False,
            "options": {"temperature": 0.2},
        }
        if keep_alive is not None:
            body["keep_alive"] = keep_alive
        async with httpx.AsyncClient(timeout=httpx.Timeout(self._timeout, connect=10.0)) as client:
            response = await client.post(f"{self._host}/api/chat", json=body)
            response.raise_for_status()
            answer = (response.json().get("message") or {}).get("content", "").strip()
        logger.info("Vision answer from %s: %d chars", body["model"], len(answer))
        return answer
