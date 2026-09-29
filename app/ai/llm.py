"""LLM abstraction. The rest of the app talks to `LLMBackend`, never to Ollama directly.

To swap models or providers later, implement `LLMBackend` and change the wiring in
`app/main.py`.
"""

from __future__ import annotations

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncIterator, Required, TypedDict

import httpx

from app.config import Settings

logger = logging.getLogger("jarvis.llm")

# Qwen3.5's own recommended sampling for thinking (Qwen/Qwen3.5-4B model card, checked 2026-09-28): general tasks, and
# precise coding. Non-thinking turns keep the Modelfile's instruct preset (0.7 / 0.8 / 20 / 1.5).
THINKING_OPTIONS = {
    "general": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5},
    "code": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 0.0},
}
_THINK_OVER_NOTE = (
    "Your own unfinished reasoning so far (not shown to the user):\n{notes}\n\n"
    "Stop reasoning now. Give the user the final answer directly, in character, without mentioning this note."
)


class _ThoughtTooLong(Exception):
    """The model's thinking went past its budget before it wrote any of the answer."""


class Message(TypedDict, total=False):
    role: Required[str]  # "system" | "user" | "assistant" | "tool"
    content: Required[str]
    tool_calls: list[dict[str, Any]]  # assistant messages that requested tools
    tool_name: str  # tool-result messages


@dataclass(frozen=True)
class ToolCall:
    """The model asked to run a tool."""

    name: str
    arguments: dict[str, Any]


class LLMError(Exception):
    """Base class for LLM failures. `status_code` is the suggested HTTP status."""

    status_code = 502
    code = "llm_error"


class LLMUnavailableError(LLMError):
    status_code = 503
    code = "ollama_unavailable"


class ModelNotFoundError(LLMError):
    status_code = 503
    code = "model_not_found"


class LLMTimeoutError(LLMError):
    status_code = 504
    code = "llm_timeout"


class LLMStreamError(LLMError):
    status_code = 502
    code = "stream_interrupted"


@dataclass(frozen=True)
class OllamaStatus:
    online: bool
    models: tuple[str, ...] = ()
    detail: str | None = None


class LLMBackend(ABC):
    """Interface every model backend must implement."""

    model: str

    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        *,
        format: dict[str, Any] | str | None = None,
        **options: Any,
    ) -> str:
        """Non-streaming reply. `format` is a JSON schema (or "json") to force structured output."""

    @abstractmethod
    def stream(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        **options: Any,
    ) -> AsyncIterator[str | ToolCall]:
        """Stream reply text. If `tools` are given the model may instead yield ToolCall items."""

    @abstractmethod
    async def generate(self, prompt: str, system: str | None = None, **options: Any) -> str: ...

    @abstractmethod
    async def status(self) -> OllamaStatus: ...

    def will_think(self, kind: str | None) -> bool:
        """Whether a streamed reply marked `kind` ("general"/"code", see app/ai/thinking.py) thinks first."""
        return False

    async def aclose(self) -> None:  # optional cleanup
        return None


# Ollama failing to read Qwen3.5's XML tool call, in its different wordings (both seen on 2026-09-28): "XML syntax
# error ... unexpected end element </parameter>" and "expected element type <function> but have <parameter>".
_TOOL_CALL_PARSE_ERROR = re.compile(r"XML syntax error|expected element type <\w+>", re.IGNORECASE)

_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_THINK_UNCLOSED = re.compile(r"<think>.*\Z", re.DOTALL)


def strip_think(text: str) -> str:
    """Remove Qwen3-style <think>...</think> reasoning blocks from a complete reply."""
    text = _THINK_BLOCK.sub("", text)
    text = _THINK_UNCLOSED.sub("", text)
    return text.strip()


class ThinkStripper:
    """Incrementally removes <think>...</think> from a stream, even if tags span chunks."""

    OPEN = "<think>"
    CLOSE = "</think>"

    def __init__(self) -> None:
        self._buf = ""
        self._in_think = False
        self._skip_leading_ws = False

    @staticmethod
    def _partial_suffix_len(text: str, tag: str) -> int:
        for k in range(min(len(tag) - 1, len(text)), 0, -1):
            if text.endswith(tag[:k]):
                return k
        return 0

    def feed(self, chunk: str) -> str:
        self._buf += chunk
        out: list[str] = []
        while True:
            if self._in_think:
                idx = self._buf.find(self.CLOSE)
                if idx == -1:
                    keep = self._partial_suffix_len(self._buf, self.CLOSE)
                    self._buf = self._buf[len(self._buf) - keep :] if keep else ""
                    break
                self._buf = self._buf[idx + len(self.CLOSE) :]
                self._in_think = False
                self._skip_leading_ws = True
            else:
                if self._skip_leading_ws:
                    self._buf = self._buf.lstrip()
                    if not self._buf:
                        break
                    self._skip_leading_ws = False
                idx = self._buf.find(self.OPEN)
                if idx == -1:
                    keep = self._partial_suffix_len(self._buf, self.OPEN)
                    cut = len(self._buf) - keep
                    out.append(self._buf[:cut])
                    self._buf = self._buf[cut:]
                    break
                out.append(self._buf[:idx])
                self._buf = self._buf[idx + len(self.OPEN) :]
                self._in_think = True
        return "".join(out)

    def flush(self) -> str:
        rest = "" if self._in_think else self._buf
        self._buf = ""
        return rest


def _parse_tool_call(raw: Any) -> ToolCall | None:
    function = raw.get("function", {}) if isinstance(raw, dict) else {}
    name = function.get("name")
    arguments = function.get("arguments") or {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            arguments = {}
    if not isinstance(name, str) or not name or not isinstance(arguments, dict):
        return None
    return ToolCall(name=name, arguments=arguments)


def model_matches(model: str, installed: tuple[str, ...]) -> bool:
    """True if `model` (e.g. 'qwen3:8b' or 'llama3') is among Ollama's installed model names."""
    wanted = model if ":" in model else f"{model}:latest"
    return wanted in installed


def _looks_like_missing_model(status: int, body: str) -> bool:
    return status == 404 or "not found" in body.lower() and "model" in body.lower()


class LLM(LLMBackend):
    """Ollama-backed implementation (default model: qwen3:8b)."""

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self.model = settings.active_model
        self._send_think = True  # flipped off if the model rejects the `think` field
        self._think_budget: int | None = None
        self._client = client or httpx.AsyncClient(
            base_url=settings.ollama_host.rstrip("/"),
            timeout=httpx.Timeout(settings.ollama_timeout, connect=5.0),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------ helpers
    def _options(self, overrides: dict[str, Any]) -> dict[str, Any]:
        overrides = dict(overrides)
        # Synthetic flag, not a real Ollama option - set by Agent for voice-originated turns so
        # LIGHT mode (only) can cap reply length for low-latency speech. Never affects DEEP mode or
        # text-typed LIGHT chat.
        voice = overrides.pop("voice", False)
        options: dict[str, Any] = {
            "num_ctx": self.settings.ollama_num_ctx,
            "temperature": self.settings.ollama_temperature,
        }
        if voice and self.model in (self.settings.light_model, self.settings.newlight_model):
            options["num_predict"] = self.settings.light_voice_num_predict
        options.update(overrides)
        return options

    def will_think(self, kind: str | None) -> bool:
        return self._thinks(True, kind)

    def _thinks(self, stream: bool, kind: Any) -> bool:
        """Whether this call thinks first: a streamed NEWLIGHT reply the agent marked as hard (app/ai/thinking.py)."""
        return (bool(kind) and stream and self.settings.newlight_auto_think
                and self.model == self.settings.newlight_model)

    def _payload(self, stream: bool, overrides: dict[str, Any]) -> dict[str, Any]:
        overrides = dict(overrides)
        kind = overrides.pop("think", None)  # "general" / "code" from the agent; not an Ollama option
        thinking = self._thinks(stream, kind)
        options = self._options(overrides)
        if thinking:
            options.update(THINKING_OPTIONS.get(kind, THINKING_OPTIONS["general"]))
        payload: dict[str, Any] = {
            "model": self.model,
            "stream": stream,
            "keep_alive": self.settings.ollama_keep_alive,
            "options": options,
        }
        if self._send_think and thinking:
            payload["think"] = True
        elif self._send_think:
            # LIGHT is always non-thinking, regardless of OLLAMA_THINK - immediate conversational
            # response is the entire point of LIGHT mode. DEEP's streamed chat replies think when
            # DEEP_THINK is on (the thinking text arrives in a separate field and is never shown or
            # spoken); everything else keeps the configurable OLLAMA_THINK behavior.
            if self.model in (self.settings.light_model, self.settings.newlight_model):
                payload["think"] = False
            elif stream and self.model == self.settings.deep_model and self.settings.deep_think:
                payload["think"] = True
            else:
                payload["think"] = self.settings.ollama_think
        return payload

    def _unavailable(self) -> LLMUnavailableError:
        return LLMUnavailableError(
            f"Cannot reach Ollama at {self.settings.ollama_host}. Is it running? Start it with `ollama serve`."
        )

    def _missing_model(self) -> ModelNotFoundError:
        return ModelNotFoundError(
            f"Model '{self.model}' is not available in Ollama. Install it with `ollama pull {self.model}`."
        )

    def _raise_for_body(self, status: int, body: str) -> None:
        """Translate an Ollama HTTP error into a typed LLMError."""
        try:
            message = json.loads(body).get("error", body)
        except (ValueError, AttributeError):
            message = body
        message = str(message).strip()[:300]
        if _looks_like_missing_model(status, message):
            raise self._missing_model()
        raise LLMError(f"Ollama returned HTTP {status}: {message}")

    @staticmethod
    def _is_think_unsupported(status: int, body: str) -> bool:
        return status == 400 and "think" in body.lower()

    async def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        for attempt in (1, 2):
            try:
                resp = await self._client.post(path, json=payload)
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                logger.error("Ollama unreachable: %s", exc)
                raise self._unavailable() from exc
            except httpx.TimeoutException as exc:
                raise LLMTimeoutError(
                    f"Ollama did not respond within {self.settings.ollama_timeout:.0f}s."
                ) from exc
            except httpx.HTTPError as exc:
                raise LLMError(f"Network error talking to Ollama: {exc}") from exc

            if resp.status_code == 200:
                return resp.json()
            if attempt == 1 and "think" in payload and self._is_think_unsupported(resp.status_code, resp.text):
                logger.warning("Model rejected `think`; retrying without it")
                self._send_think = False
                payload = {k: v for k, v in payload.items() if k != "think"}
                continue
            self._raise_for_body(resp.status_code, resp.text)
        raise LLMError("Unexpected LLM failure")  # pragma: no cover

    # --------------------------------------------------------------- public API
    async def chat(
        self,
        messages: list[Message],
        *,
        format: dict[str, Any] | str | None = None,
        **options: Any,
    ) -> str:
        payload = self._payload(False, options) | {"messages": messages}
        if format is not None:
            payload["format"] = format
        data = await self._post_json("/api/chat", payload)
        return strip_think(data.get("message", {}).get("content", ""))

    async def generate(self, prompt: str, system: str | None = None, **options: Any) -> str:
        payload = self._payload(False, options) | {"prompt": prompt}
        if system:
            payload["system"] = system
        data = await self._post_json("/api/generate", payload)
        return strip_think(data.get("response", ""))

    async def stream(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        **options: Any,
    ) -> AsyncIterator[str | ToolCall]:
        options = dict(options)
        self._think_budget = options.pop("think_budget", None)  # the nightly study may think longer
        payload = self._payload(True, options) | {"messages": messages}
        if tools:
            payload["tools"] = tools
        if payload.get("think") is not True:
            async for item in self._resilient(payload):
                yield item
            return
        thoughts: list[str] = []
        started = time.monotonic()
        try:
            async for item in self._resilient(payload, thoughts):
                yield item
            logger.info("Thought for %.1fs (%d chars) before answering", time.monotonic() - started,
                        sum(map(len, thoughts)))
        except _ThoughtTooLong:
            # Nothing of the answer was sent yet: answer again without thinking, from the notes it made so far.
            notes = "".join(thoughts)[-3000:]
            logger.info("Thinking passed its budget (%d chars); answering from its notes", sum(map(len, thoughts)))
            retry_options = {k: v for k, v in options.items() if k != "think"}
            retry = self._payload(True, retry_options) | {
                "messages": list(messages) + [{"role": "system", "content": _THINK_OVER_NOTE.format(notes=notes)}]
            }
            if tools:
                retry["tools"] = tools
            async for item in self._resilient(retry):
                yield item

    async def _resilient(
        self, payload: dict[str, Any], thoughts: list[str] | None = None
    ) -> AsyncIterator[str | ToolCall]:
        """_stream_payload, retried when Ollama cannot parse the model's tool call. Qwen3.5 writes tool calls as XML
        and the 4B model sometimes gets it wrong ("XML syntax error ... unexpected end element </parameter>" - a
        real chat failed this way on 2026-09-28). Nothing has been shown yet then, so it asks again, and the third
        time without tools, so the user gets an answer instead of an error."""
        for attempt in range(3):
            yielded = False
            try:
                async for item in self._stream_payload(payload, thoughts):
                    yielded = True
                    yield item
                return
            except LLMError as exc:
                if yielded or not _TOOL_CALL_PARSE_ERROR.search(str(exc)) or attempt == 2:
                    raise
                logger.warning("Ollama could not read the model's tool call (%s); asking again%s", str(exc)[:120],
                               " without tools" if attempt == 1 else "")
                if attempt == 1:
                    payload = {k: v for k, v in payload.items() if k != "tools"}
                if thoughts is not None:
                    thoughts.clear()

    async def _stream_payload(
        self, payload: dict[str, Any], thoughts: list[str] | None = None
    ) -> AsyncIterator[str | ToolCall]:
        """One streamed /api/chat call. With `thoughts`, the thinking text is collected there, and _ThoughtTooLong
        is raised if it passes the budget before any answer text or tool call has come."""
        budget = self._think_budget or self.settings.newlight_think_budget_chars
        answered = False
        stripper = ThinkStripper()
        got_done = False
        try:
            for attempt in (1, 2):
                async with self._client.stream("POST", "/api/chat", json=payload) as resp:
                    if resp.status_code != 200:
                        body = (await resp.aread()).decode("utf-8", errors="replace")
                        if attempt == 1 and "think" in payload and self._is_think_unsupported(resp.status_code, body):
                            logger.warning("Model rejected `think`; retrying without it")
                            self._send_think = False
                            payload = {k: v for k, v in payload.items() if k != "think"}
                            continue
                        self._raise_for_body(resp.status_code, body)
                    async for line in resp.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line)
                        except ValueError as exc:
                            raise LLMStreamError("Received malformed data from Ollama.") from exc
                        if "error" in data:
                            raise LLMStreamError(f"Ollama error mid-stream: {str(data['error'])[:300]}")
                        message = data.get("message", {})
                        if thoughts is not None and message.get("thinking") and not answered:
                            thoughts.append(message["thinking"])
                            if sum(map(len, thoughts)) > budget:
                                raise _ThoughtTooLong()
                        text = stripper.feed(message.get("content", ""))
                        if text:
                            answered = True
                            yield text
                        for call in message.get("tool_calls") or []:
                            parsed = _parse_tool_call(call)
                            if parsed:
                                answered = True
                                yield parsed
                        if data.get("done"):
                            got_done = True
                            break
                    break
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            logger.error("Ollama unreachable: %s", exc)
            raise self._unavailable() from exc
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(
                f"Ollama stopped responding (no data for {self.settings.ollama_timeout:.0f}s)."
            ) from exc
        except (httpx.RemoteProtocolError, httpx.ReadError) as exc:
            raise LLMStreamError("Connection to Ollama was interrupted mid-response.") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"Network error talking to Ollama: {exc}") from exc

        if not got_done:
            raise LLMStreamError("Ollama closed the stream before the reply finished.")
        tail = stripper.flush()
        if tail:
            yield tail

    async def status(self) -> OllamaStatus:
        try:
            resp = await self._client.get("/api/tags", timeout=5.0)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            return OllamaStatus(online=False, detail=f"Cannot reach Ollama at {self.settings.ollama_host}: {type(exc).__name__}")
        names = tuple(m.get("name", "") for m in resp.json().get("models", []))
        return OllamaStatus(online=True, models=names)

