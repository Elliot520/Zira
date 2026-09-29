"""The agent: user message -> memory -> context -> LLM -> tools (if needed) -> response."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
import re
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import AsyncIterator, Literal

from app.agent.planner import PlanKind, Planner
from app.ai.thinking import thinking_kind
from app.agent.tool_manager import ToolRegistry
from app.ai.context import ContextBuilder
from app.ai.llm import LLMBackend, LLMError, Message, ToolCall
from app.memory.checkpoints import CheckpointManager, estimate_tokens
from app.memory.conversation_store import ConversationStore, new_conversation_id
from app.memory.extractor import MemoryExtractor
from app.memory.memory_manager import MemoryManager, parse_memory_command
from app.models.schemas import Memory
from app.tools.base import ToolResult, current_conversation, current_progress_reporter, current_request

logger = logging.getLogger("jarvis.agent")

# The model may call tools for at most this many rounds; the final round has tools disabled
# so it must answer. Also caps how many tool calls it may make per round.
MAX_TOOL_ROUNDS = 6
MAX_CALLS_PER_ROUND = 2
# Total characters of tool output kept in the model's context; older output is dropped first so a long
# file-reading session cannot overflow the context window (which would silently cut the system prompt).
TOOL_CONTEXT_BUDGET = 9000

OMITTED = "[earlier tool output omitted to save space; call the tool again if you need it]"
# Answers grounded in tool output should follow the results closely, not improvise.
TOOL_ANSWER_TEMPERATURE = 0.3

# A bare greeting ("Hello Zira", "hi", just the wake word with nothing else) - matched by
# stripping every recognized greeting/wake word and checking nothing else is left. The system
# prompt already asks for a warm, varied reply here instead of "How can I help?", but measured for
# real: with no other context, 4/4 fresh conversations all produced the exact same "How's your day
# going?" every time - the model's own sampling doesn't reliably vary on its own. Picking the
# specific angle here and injecting it as an event note (the same mechanism used for "remember
# this") forces real variety instead of hoping for it.
_GREETING_WORD = re.compile(r"\b(hi+|hey+|hello+|namaste|yo|zira|jarvis)\b[\s,!.]*", re.IGNORECASE)
_GREETING_PREFIX = "The user just greeted you with no other request yet (e.g. \"Hello Zira\"). "
_GREETING_SUFFIX = (
    " Say ONLY that, in one short warm sentence, like a person greeted by a friend would - do not "
    "also mention the other two things you could have said; pick this one and stop there."
)
_GREETING_DIRECTIVES = (
    _GREETING_PREFIX + "Ask how their day is going." + _GREETING_SUFFIX,
    _GREETING_PREFIX + "Casually offer to play a song - do not ask about their day." + _GREETING_SUFFIX,
    _GREETING_PREFIX + "Offer to tell a joke - do not ask about their day." + _GREETING_SUFFIX,
)


# Openers for when Zira starts the conversation herself (see Agent.proactive_message). One is picked at
# random each time so it does not always ask the same kind of thing.
_PROACTIVE_PREFIX = (
    "The user has been quiet for a few minutes with the microphone on, and you are starting the "
    "conversation yourself, like an excited friend would. "
)
_PROACTIVE_SUFFIX = (
    " Say it in one or two short sentences, because it is read aloud. Use the language the user usually "
    "talks to you in (look at the recent conversation; Hinglish in Latin letters if they use Hindi). Do "
    "not greet with \"how can I help\" and do not mention that they were quiet."
)
_PROACTIVE_DIRECTIVES = (
    _PROACTIVE_PREFIX + "Ask ONE curious, specific question to learn something new about them that is NOT "
    "already in your long-term memory: their day, work, goals, tastes, or the people they care about."
    + _PROACTIVE_SUFFIX,
    _PROACTIVE_PREFIX + "Genuinely praise them for something specific you know about them from long-term "
    "memory or the recent conversation (their work, a project, a skill, something they shared). Make it "
    "warm and excited; you may end with a light question." + _PROACTIVE_SUFFIX,
    _PROACTIVE_PREFIX + "Bring up something fun connected to one of their interests from long-term memory "
    "(a game, a movie, a technology) and ask for their opinion on it." + _PROACTIVE_SUFFIX,
)


def _bare_greeting_note(text: str) -> str | None:
    if not _GREETING_WORD.search(text):
        return None
    if _GREETING_WORD.sub("", text).strip(" ,.!?"):
        return None  # something besides the greeting/wake word was said - a real request, not this
    return random.choice(_GREETING_DIRECTIVES)

_IMAGE_REQUEST_RE = re.compile(
    r"""
    (?:
        \b(?:create|generate|make|draw|render|produce)\b
        .*?\b(?:image|picture|photo|illustration|artwork|portrait)\b

        |

        \b(?:image|picture|photo|illustration|artwork|portrait)\b
        .*?\b(?:of|showing|depicting)\b

        |

        ^\s*(?:a|an|the)\s+
        .{5,}
        (?:cinematic|photography|photograph|portrait|illustration|artwork|
           realistic|photorealistic|digital art|3d|anime|painting)\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)
# Deliberately separate from image_safety.py's own term lists: that module's job is "is this
# minor-related" (always active, never configurable). This one's job is narrower - "does this look
# like an adult-content image request specifically enough to skip the LLM's own tool-call decision"
# (see is_adult_image_request in _generate below). Compiled once at import time, not per-turn.
# Unused while the direct bypass in _generate is commented out (2026-09-27); kept, together with
# _is_direct_image_request/_is_direct_video_request, so uncommenting that block works as it is.
_ADULT_IMAGE_RE = re.compile(
    r"\b(?:adult|nsfw|nude|nudity|naked|explicit|erotic|sexual|sex|pornographic|porn)\b",
    re.IGNORECASE,
)


def _is_direct_image_request(text: str) -> bool:
    return bool(_IMAGE_REQUEST_RE.search(text))


_VIDEO_REQUEST_RE = re.compile(
    r"""
    (?:
        # Normal video / animation requests
        \b(?:create|generate|make|render|produce)\b
        .*?\b(?:video|clip|animation|animated|cartoon|movie)\b

        |

        # "video of...", "animation showing...", etc.
        \b(?:video|clip|animation|animated|cartoon|movie)\b
        .*?\b(?:of|showing|depicting|about)\b

        |

        # Explicit/adult video requests
        \b(?:create|generate|make|render|produce)\b
        .*?\b(?:adult|nsfw|nude|nudity|naked|explicit|erotic|sexual|sex|pornographic|porn)\b
        .*?\b(?:video|clip|animation|animated|cartoon|movie)\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)



def _is_direct_video_request(text: str) -> bool:
    """Return True when the user explicitly asks to create video content.

    This only controls routing. The create_video tool still performs its
    existing safety checks when it executes.
    """
    return bool(_VIDEO_REQUEST_RE.search(text))


MAX_SOURCES_SHOWN = 4


def sources_block(reply: str, sources: list[dict[str, str]]) -> str:
    """"Sources:" list built from the real search results (never from model-written URLs).

    Shows the results the reply cites as [n]; if it cites none, the top results.
    """
    if not sources:
        return ""
    cited = sorted({int(n) for n in re.findall(r"\[(\d{1,2})\]", reply) if 1 <= int(n) <= len(sources)})
    numbers = cited[:MAX_SOURCES_SHOWN] or list(range(1, min(len(sources), 3) + 1))
    lines = [f"{n}. {sources[n - 1]['title']} - {sources[n - 1]['url']}" for n in numbers]
    return "\n\nSources:\n" + "\n".join(lines)


def files_block(files: list[dict[str, str]]) -> str:
    """"Files:" list of what a tool produced (title + download link), built by the app, not the model."""
    if not files:
        return ""
    return "\n\nFiles:\n" + "\n".join(f"- {f['title']} - {f['url']}" for f in files)


def image_only_reply(calls: list[ToolCall], results: list[ToolResult]) -> str | None:
    """The whole reply for a round that did nothing but successfully make one image: just the image
    link(s), no sentence around them - by explicit user request ("when image generated no error
    message should show... i do not want any reply for image"). The frontend renders a bare
    /api/exports/... link as the image itself. None when this round was anything else (another
    tool, several calls, or a failure), so the model still gets to explain those normally."""
    if (
        len(calls) == 1
        and calls[0].name in ("create_image", "edit_image", "create_video", "create_song", "create_ad")
        and len(results) == 1
        and results[0].ok
        and results[0].files
    ):
        return "\n".join(f["url"] for f in results[0].files)
    return None


def _plain_request(message: str) -> str:
    """The user's request without the attachment tags, short enough for a notification."""
    text = re.sub(r"\[(?:Uploaded image|Video|Document): [^\]]+\]\s*", "", message).strip()
    return text if len(text) <= 120 else text[:117] + "..."


def media_failure_reply(calls: list[ToolCall], results: list[ToolResult]) -> str | None:
    """The whole reply for a round that did nothing but fail to make one image or video: the tool's own message,
    as it is - by user request (Phase 1 of the 2026-09-27 plan). The small model used to reword these and got
    them wrong: it answered "insufficient memory" (copied from earlier replies in the conversation) when the real
    reason was that the video model was set to None. Argument mistakes ("create_video needs a non-empty
    'prompt'") are the model's own and still go back to it to fix."""
    if len(calls) == 1 and calls[0].name in LONG_TOOLS and len(results) == 1 and not results[0].ok:
        error = results[0].error or ""
        if error.startswith(f"{calls[0].name} needs"):
            return None
        return error or "It could not be made."
    return None


_EXPORT_LINK = re.compile(r"/api/exports/[^\s)\]>\"']+")
_EXPORT_PREFIX = "/api/exports/"

# What the model is told when it answered a request with a file link instead of making the file.
_FAKE_LINK_NOTE = (
    "Your last answer was only a file link, but you did not call any tool, so no file exists - never write "
    "/api/exports links yourself. To make a video, call create_video; for an image, call create_image (or "
    "edit_image for an uploaded one), with a clear, descriptive English prompt."
)
# What the user sees when the model does it twice in a row (instead of a blank player).
FAKE_LINK_REPLY = (
    "The video or image was not made: the AI model answered with a link instead of running the tool, so there "
    "was nothing to show. Please send the request again - if it keeps happening, the BALANCED or DEEP model "
    "follows tool instructions more reliably than LIGHT."
)


def could_be_file_link(text: str) -> bool:
    """While a reply is still arriving: could it turn out to be a bare file link? (Held back until it is clear.)"""
    head = text.lstrip()
    return head.startswith(_EXPORT_PREFIX) or _EXPORT_PREFIX.startswith(head)


def fabricated_links(text: str, made: list[dict[str, str]]) -> list[str]:
    """/api/exports links the model wrote that no tool made this turn. Seen for real on 2026-09-27: the 4B model
    answered video requests with a link it made up (to an old video, or to no file at all - a blank player)."""
    real = {f["url"].split("/api/exports/")[-1] for f in made}
    return [link for link in _EXPORT_LINK.findall(text) if link.removeprefix(_EXPORT_PREFIX) not in real]


def compact_tool_messages(messages: list[Message], budget: int = TOOL_CONTEXT_BUDGET) -> None:
    """Replace the oldest tool outputs with a stub until the total is within `budget` (newest is kept)."""
    tool_idx = [i for i, m in enumerate(messages) if m["role"] == "tool"]
    total = sum(len(messages[i]["content"]) for i in tool_idx)
    for i in tool_idx[:-1]:
        if total <= budget:
            break
        total -= len(messages[i]["content"]) - len(OMITTED)
        messages[i]["content"] = OMITTED


@dataclass(frozen=True)
class ChatResult:
    conversation_id: str
    response: str
    estimated_tokens: int = 0


@dataclass(frozen=True)
class AgentEvent:
    type: Literal["start", "token", "tool", "image_progress", "change", "music", "done", "memory", "checkpoint"]
    conversation_id: str
    content: str = ""  # token text, or for "tool" a short description (e.g. the search query)
    tool: str = ""  # tool name, for "tool" events
    memories: tuple[Memory, ...] = ()
    data: dict | None = None  # for "change"/"music"/"checkpoint"/"image_progress": details shown to the user


@dataclass(frozen=True)
class _Turn:
    conversation_id: str
    user_message: str
    messages: list[Message]
    mode: str = "chat"
    estimated_tokens: int = 0
    source: str = "text"  # "text" or "voice" - lets LLM cap reply length for voice in LIGHT mode
    # Set by the chat WebSocket when the user interrupts or stops the reply: ends the model's text stream (so
    # Ollama stops generating). Checked only between streamed tokens, never inside a tool run.
    stop: asyncio.Event | None = None
    studied: bool = False  # a studied answer was handed to the model (app/learning/recall.py): no need to think

    @property
    def stopped(self) -> bool:
        return self.stop is not None and self.stop.is_set()


@dataclass
class _TurnState:
    """Mutable bookkeeping for one user message."""

    tainted: bool = False  # a tool that reads private data has run
    counts: Counter = field(default_factory=Counter)
    sources: list[dict[str, str]] = field(default_factory=list)
    files: list[dict[str, str]] = field(default_factory=list)
    last_results: list["ToolResult"] = field(default_factory=list)  # this round's results, in call order


# Generations long enough that a phone realistically leaves mid-way (switching apps closes its WebSocket).
LONG_TOOLS = ("create_image", "edit_image", "create_video", "create_song", "create_ad")


@dataclass
class _PendingJob:
    """A long generation still running (or finished but not yet in the history) for a conversation. It
    outlives the WebSocket: the page that comes back - reconnected or reloaded - asks for it (GET
    /api/conversations/{id}/pending) to show the user's message and the progress again, and reloads the
    history when it is gone. Ends once the reply is saved."""

    turn: _Turn
    call: ToolCall
    label: str
    started_at: float = field(default_factory=time.time)
    progress: dict | None = None
    progress_at: float | None = None
    result: ToolResult | None = None  # set when the tool is done

    def snapshot(self) -> dict:
        now = time.time()
        return {
            "tool": self.call.name,
            "label": self.label,
            "user_message": self.turn.user_message,
            "elapsed_seconds": round(now - self.started_at, 1),
            "progress": self.progress,
            "progress_age_seconds": round(now - self.progress_at, 1) if self.progress_at else None,
            "finished": self.result is not None,
        }


def long_tool_reply(call: ToolCall, result: ToolResult) -> str:
    """What is saved for a long tool whose reply never reached the client: the link(s) on success (as
    image_only_reply), otherwise what went wrong - so the user's message and the outcome both stay in
    the history instead of vanishing."""
    return image_only_reply([call], [result]) or media_failure_reply([call], [result]) or result.error or result.output


class Agent:
    def __init__(
        self,
        llm: LLMBackend,
        memory: MemoryManager,
        conversations: ConversationStore,
        context: ContextBuilder,
        planner: Planner | None = None,
        tools: ToolRegistry | None = None,
        extractor: MemoryExtractor | None = None,
        checkpoints: CheckpointManager | None = None,
    ) -> None:
        self.llm = llm
        self.memory = memory
        self.conversations = conversations
        self.context = context
        self.planner = planner or Planner(llm)
        self.tools = tools or ToolRegistry()
        self.extractor = extractor
        self.checkpoints = checkpoints
        self.recall = None  # app/learning/recall.py Recall, set by app/main.py when learning is on

    async def _recalled(self, message: str):
        """Studied answer + memories by meaning for this message (nothing if learning is off or fails)."""
        from app.learning.recall import Recalled

        if self.recall is None:
            return Recalled()
        try:
            return await self.recall.recall(message)
        except Exception:  # noqa: BLE001 - recall is a bonus; the reply must never fail because of it
            logger.warning("Recall failed; answering without it", exc_info=True)
            return Recalled()

    def _prepare(self, conversation_id: str | None, message: str, mode: str = "chat", source: str = "text",
                 recalled=None) -> _Turn:
        conversation_id = conversation_id or new_conversation_id()

        event_note = None
        command = parse_memory_command(message)
        if command is not None:
            saved = self.memory.remember(command.text, command.category)
            event_note = (
                f'The user just asked you to remember: "{saved.text}" (category: {saved.category.value}). '
                "It has been saved to long-term memory. Confirm briefly and naturally, in one sentence."
            )
        else:
            event_note = _bare_greeting_note(message)

        built = self.context.build(
            conversation_id, message, event_note=event_note, mode=mode,
            learned_note=recalled.note if recalled else None,
            related_memory_ids=recalled.memory_ids if recalled else (),
        )
        logger.info(
            "Context built conversation=%s history=%d memories=%d%s",
            conversation_id,
            built.history_count,
            len(built.memories),
            " studied=yes" if recalled and recalled.studied else "",
        )
        return _Turn(conversation_id, message, built.messages, mode, built.estimated_tokens, source,
                     studied=bool(recalled and recalled.studied))

    @property
    def _pending(self) -> dict[str, _PendingJob]:
        jobs = self.__dict__.get("_pending_jobs")
        if jobs is None:
            jobs = self.__dict__["_pending_jobs"] = {}
        return jobs

    def pending_job(self, conversation_id: str) -> dict | None:
        """The long generation still running for this conversation, if any (see _PendingJob)."""
        job = self._pending.get(conversation_id)
        return job.snapshot() if job is not None else None

    def _announce(self, call: ToolCall, result: ToolResult, turn: _Turn) -> None:
        """A phone notification (Web Push, app/push.py) when a video is done - made or not; not when the user
        stopped it themselves. Sent in the background: never delays the reply."""
        notifier = getattr(self, "notifier", None)
        if notifier is None or call.name not in ("create_video", "create_ad"):
            return
        what = "ad" if call.name == "create_ad" else "video"
        if result.ok:
            title, body = f"Your {what} is ready", _plain_request(turn.user_message)
        elif (result.error or "").startswith(("Video stopped", "Ad stopped")):
            return
        else:
            title, body = f"The {what} was not made", result.error or ""
        tasks = self.__dict__.setdefault("_background_tasks", set())
        task = asyncio.get_running_loop().create_task(notifier.notify_async(title, body, "/", "zira-video"))
        tasks.add(task)
        task.add_done_callback(tasks.discard)

    def _end_job(self, job: _PendingJob) -> None:
        if self._pending.get(job.turn.conversation_id) is job:
            del self._pending[job.turn.conversation_id]

    def _close_job(self, turn: _Turn, saved: bool) -> None:
        """At the end of a turn: a long tool that finished but whose reply was never saved (the client
        left after it was done, or the model failed afterwards) still gets its outcome into the history.
        A tool still running is left alone - its disconnect callback saves it when it lands."""
        job = self._pending.get(turn.conversation_id)
        if job is None or job.turn is not turn or job.result is None:
            return
        if not saved:
            try:
                self.conversations.add_exchange(turn.conversation_id, turn.user_message, long_tool_reply(job.call, job.result))
                logger.info("Saved a %s result the client never received to conversation=%s", job.call.name, turn.conversation_id)
            except Exception:  # noqa: BLE001
                logger.exception("Could not save a finished %s result", job.call.name)
        self._end_job(job)

    def _save_image_after_disconnect(
        self, turn: _Turn, call: ToolCall, exec_task: "asyncio.Task | None", job: _PendingJob | None = None
    ) -> None:
        """When the client disconnects while create_image/edit_image/create_video is still running, store
        the outcome in the conversation once it completes (see the GeneratorExit handler in _run_calls):
        the link on success, what went wrong on failure. Only these tools: they are the long-running ones
        a phone realistically drops out of."""
        if exec_task is None or call.name not in LONG_TOOLS:
            return

        def store(task: "asyncio.Task") -> None:
            try:
                try:
                    result = task.result()
                except BaseException:  # noqa: BLE001 - cancelled or crashed: nothing to record
                    return
                self._announce(call, result, turn)
                if not result.ok:
                    logger.warning("Tool %s failed after the client disconnected: %s", call.name, (result.error or "")[:300])
                try:
                    self.conversations.add_exchange(turn.conversation_id, turn.user_message, long_tool_reply(call, result))
                    logger.info(
                        "Client had disconnected; saved the %s to conversation=%s",
                        "finished image" if result.ok else "failure", turn.conversation_id,
                    )
                except Exception:  # noqa: BLE001
                    logger.exception("Could not save an image finished after the client disconnected")
            finally:
                if job is not None:
                    self._end_job(job)

        exec_task.add_done_callback(store)

    async def _run_calls(
        self,
        turn: _Turn,
        calls: list[ToolCall],
        messages: list[Message],
        state: "_TurnState",
        round_text: str = "",
    ) -> AsyncIterator[AgentEvent]:
        """Execute tool calls, appending the assistant request and tool results to `messages`."""
        state.last_results = []
        messages.append(
            {
                "role": "assistant",
                "content": round_text,
                "tool_calls": [{"function": {"name": c.name, "arguments": c.arguments}} for c in calls],
            }
        )
        for call in calls:
            tool = self.tools.get(call.name)
            allowed_here = tool is not None and turn.mode in tool.modes
            label = tool.describe(call.arguments) if tool and allowed_here else call.name
            yield AgentEvent("tool", turn.conversation_id, content=label, tool=call.name)
            logger.info("Tool call: %s", call.name)
            if call.name == "create_image":
                logger.info(
                    "Image tool prompt from agent: %s",
                    call.arguments.get("prompt", ""),
                )

            if tool is None or not allowed_here:
                result = ToolResult.failure(f"Unknown tool: {call.name}" if tool is None else f"{call.name} is not available in {turn.mode} mode.")
            elif state.tainted and tool.sends_data_out:
                # Private files were read this turn: nothing may be sent to the outside world.
                result = ToolResult.failure(f"{call.name} is disabled for the rest of this turn because private project files were read.")
            elif tool.max_calls_per_turn is not None and state.counts[call.name] >= tool.max_calls_per_turn:
                result = ToolResult.failure(f"{call.name} was already used {tool.max_calls_per_turn} times this turn; answer with what you have.")
            else:
                state.counts[call.name] += 1
                current_conversation.set(turn.conversation_id)
                current_request.set(turn.user_message)
                # Tool execution runs as a task, not a plain await, so a tool that reports
                # intermediate progress (image generation, via current_progress_reporter) can
                # surface it as image_progress events while still running, instead of the caller
                # blocking silently until the whole call finishes. Tools that never report
                # progress take this same path at negligible cost - the queue just stays empty
                # and this behaves like the plain await it replaces.
                progress_queue: asyncio.Queue = asyncio.Queue()
                loop = asyncio.get_running_loop()
                job = _PendingJob(turn, call, label) if call.name in LONG_TOOLS else None
                if job is not None:
                    self._pending[turn.conversation_id] = job

                def on_progress(info: dict, job: _PendingJob | None = job) -> None:
                    if job is not None:
                        job.progress, job.progress_at = info, time.time()
                    progress_queue.put_nowait(info)

                def report_progress(info: dict) -> None:
                    loop.call_soon_threadsafe(on_progress, info)

                reporter_token = current_progress_reporter.set(report_progress)
                exec_task = get_task = None
                try:
                    exec_task = asyncio.create_task(self.tools.execute(call.name, call.arguments))
                    get_task = asyncio.create_task(progress_queue.get())
                    while True:
                        done, _ = await asyncio.wait({exec_task, get_task}, return_when=asyncio.FIRST_COMPLETED)
                        if get_task in done:
                            yield AgentEvent("image_progress", turn.conversation_id, data=get_task.result())
                            if exec_task not in done:
                                get_task = asyncio.create_task(progress_queue.get())
                        if exec_task in done:
                            if get_task not in done:
                                get_task.cancel()
                                with contextlib.suppress(asyncio.CancelledError):
                                    await get_task
                            break
                    result = exec_task.result()
                    if job is not None:
                        job.result = result
                    self._announce(call, result, turn)
                    # exec_task finishing doesn't guarantee the queue is empty: report_progress()
                    # hands off via call_soon_threadsafe, so a fast final burst of calls (a diffusers
                    # callback firing several times right before the thread returns) can all still be
                    # queued when exec_task's own completion callback runs - real, reproduced in
                    # testing: 3 items queued, asyncio.wait() woke this coroutine only once, with
                    # both tasks already done, so only the first item's yield above ever ran. Drain
                    # whatever is left so no progress event is silently dropped.
                    while not progress_queue.empty():
                        yield AgentEvent("image_progress", turn.conversation_id, data=progress_queue.get_nowait())
                except GeneratorExit:
                    # The client left mid-tool. A phone locking its screen or switching networks
                    # closes its WebSocket, and that closes this generator right here - seen for real:
                    # a phone request dropped mid-image, the image finished and was saved to disk, but
                    # the chat never recorded it, so reloading showed nothing. The tool task keeps
                    # running on its own; record its result when it lands so the image is in the
                    # conversation history the next time the page loads.
                    self._save_image_after_disconnect(turn, call, exec_task, job)
                    raise
                except BaseException:
                    if job is not None:
                        self._end_job(job)  # failed outright: nothing left to show as running
                    raise
                finally:
                    if get_task is not None and not get_task.done():
                        get_task.cancel()  # sync only: awaiting inside a closing generator is not safe
                    try:
                        current_progress_reporter.reset(reporter_token)
                    except ValueError:
                        # When the generator is closed by websocket teardown it runs in a different
                        # asyncio Context than the one that created the token, and reset() refuses -
                        # seen for real in the server log. The context that set the reporter is going
                        # away anyway, so there is nothing left to restore.
                        pass
                if tool.reads_private_data:
                    state.tainted = True
            if not result.ok:
                logger.warning("Tool %s failed: %s", call.name, (result.error or "")[:300])
            state.last_results.append(result)
            if result.ok and result.sources:
                state.sources[:] = result.sources  # numbering restarts per search; keep the latest
            for f in result.files:
                if f not in state.files:
                    state.files.append(f)
            for change in result.changes:
                yield AgentEvent("change", turn.conversation_id, data=change)
            if result.music:
                yield AgentEvent("music", turn.conversation_id, data=result.music)
            messages.append(
                {
                    "role": "tool",
                    "tool_name": call.name,
                    "content": result.output if result.ok else f"Error: {result.error}",
                }
            )
        compact_tool_messages(messages)

    async def _generate(self, turn: _Turn) -> AsyncIterator[AgentEvent]:
        """Run the LLM, executing tool calls (planned or requested by the model).

        Yields `token`, `tool` and `change` events.
        """
        messages = list(turn.messages)
        recent_user_text = " ".join(m["content"] for m in messages if m["role"] == "user")[-1500:]
        state = _TurnState()

        def offered() -> list[dict]:
            return self.tools.schemas(recent_user_text, exclude_outbound=state.tainted, mode=turn.mode)

        first_round = 0

        # DIRECT BYPASS - DISABLED (2026-09-27, by user request: "so it always goes through model").
        # Every image/video request now goes through the LLM, which writes an improved, descriptive
        # prompt for create_image/create_video (see the tools' descriptions) instead of the raw message
        # being sent to the image/video model word for word.
        # Trade-off to know: the model decides whether to call the tool, and it can refuse - seen for
        # real on 2026-09-27 (message 1412, an image request refused by the model). That is why this
        # bypass existed (see the original comment below).
        # TO RE-ENABLE: uncomment the block below (remove one leading "# " from each line) and change
        # the `if self.tools.names():` right after it back to `elif self.tools.names():`.
        #
        # # Explicit/adult image requests: call create_image directly rather than letting the LLM
        # # decide whether to attempt it - some models refuse to even call the tool for this content
        # # (measured for real: Gemma declined outright; Qwen3 needed an explicit system-prompt
        # # permission first - see README/CAPABILITIES "Model mode"). This is a Python-level decision,
        # # independent of the LLM's own judgment either way.
        # #
        # # Deliberately does NOT short-circuit past the LLM afterward (an earlier version of this did,
        # # then returned immediately) - it forces the call and falls through to the normal round loop
        # # below, exactly like the planner-forced branch does. That loop's terminal round already
        # # appends files_block() unconditionally, regardless of what the model says - the same
        # # mechanism that already guarantees create_pdf/create_image links are never dropped - so a
        # # short-circuit bought no real safety here, only cost: a compound request ("...and tell me a
        # # joke") would have silently lost the "tell me a joke" half, the exact bug already found and
        # # fixed once this session for the music fast-path (see FAST_CONFIRM_TOOLS in git history).
        # is_adult_image_request = (
        #     "create_image" in self.tools.names()
        #     and _is_direct_image_request(turn.user_message)
        #     and bool(_ADULT_IMAGE_RE.search(turn.user_message))
        # )
        # if is_adult_image_request:
        #     forced = ToolCall("create_image", {"prompt": turn.user_message})
        #     async for event in self._run_calls(turn, [forced], messages, state):
        #         yield event
        #     # Real bug found via the message history: this forced call used to fall straight into the
        #     # round loop below, so the model got a turn *after* a successful image and wrote things
        #     # like "It seems there was an issue..." / "Could you describe the scene?" above an image
        #     # that had in fact been made. A success needs no model turn at all (see image_only_reply);
        #     # a failure still falls through so the model can explain it.
        #     reply = image_only_reply([forced], state.last_results)
        #     if reply is not None:
        #         yield AgentEvent("token", turn.conversation_id, reply)
        #         return
        #     first_round = 1  # the forced call counts as the first tool round
        #
        # elif (
        #     "create_video" in self.tools.names()
        #     and _is_direct_video_request(turn.user_message)
        # ):
        #     # Explicit video/animation requests are routed directly to create_video
        #     # so the LLM does not have to independently select the tool. This includes
        #     # normal animation, cartoon, children's-rhyme, and nursery-rhyme requests.
        #     # The create_video tool remains responsible for its safety checks.
        #     forced = ToolCall("create_video", {"prompt": turn.user_message})
        #     async for event in self._run_calls(turn, [forced], messages, state):
        #         yield event
        #     reply = image_only_reply([forced], state.last_results)
        #     if reply is not None:
        #         yield AgentEvent("token", turn.conversation_id, reply)
        #         return
        #     first_round = 1

        if self.tools.names():  # was `elif` while the direct bypass above was on
            plan = await self.planner.plan(messages, tuple(self.tools.names()))
            if plan.kind is not PlanKind.RESPOND:
                if plan.kind is PlanKind.SEARCH:
                    forced = ToolCall("web_search", {"query": plan.query})
                else:
                    forced = ToolCall(plan.tool, plan.arguments)
                async for event in self._run_calls(turn, [forced], messages, state):
                    yield event
                if forced.name in LONG_TOOLS:  # a forced song: the player (or what went wrong) is the whole reply
                    reply = image_only_reply([forced], state.last_results) or media_failure_reply([forced], state.last_results)
                    if reply is not None:
                        yield AgentEvent("token", turn.conversation_id, reply)
                        return
                first_round = 1  # the forced call counts as the first tool round

        # Hard questions (maths, logic, code, planning, why/explain) think first on NEWLIGHT (app/ai/thinking.py);
        # the LLM ignores it for other models. Not when the planner already sent the turn to a tool.
        think = None if first_round or turn.studied else thinking_kind(turn.user_message, voice=turn.source == "voice")
        think = think if think and self.llm.will_think(think) else None
        if think:
            logger.info("Thinking turn (%s)", think)
            yield AgentEvent("tool", turn.conversation_id, content="Thinking", tool="thinking")

        streamed: list[str] = []
        fake_link_retried = False
        for round_no in range(first_round, MAX_TOOL_ROUNDS + 1):
            schemas = offered()
            tools_allowed = bool(schemas) and round_no < MAX_TOOL_ROUNDS
            calls: list[ToolCall] = []
            round_text: list[str] = []
            held: list[str] = []  # the start of a reply that may turn out to be a made-up file link

            options = {"temperature": TOOL_ANSWER_TEMPERATURE} if round_no > 0 else {}
            if turn.source == "voice":
                options["voice"] = True
            if think:
                options["think"] = think
            stream = self.llm.stream(messages, tools=schemas if tools_allowed else None, **options)
            async for item in stream:
                if turn.stopped:
                    await stream.aclose()  # closes the request, so Ollama stops generating
                    calls = []
                    break
                if isinstance(item, ToolCall):
                    calls.append(item)
                    continue
                round_text.append(item)
                if held or not streamed:
                    held.append(item)
                    if could_be_file_link("".join(held)):
                        continue  # not shown yet: it may be a link to a file nobody made
                    item, held = "".join(held), []
                streamed.append(item)
                yield AgentEvent("token", turn.conversation_id, item)

            if held and not calls:
                text = "".join(held)
                if fabricated_links(text, state.files):
                    logger.warning("The model answered with a file link but called no tool: %s", text.strip()[:120])
                    if not fake_link_retried and round_no < MAX_TOOL_ROUNDS:
                        fake_link_retried = True  # one more try, told plainly to call the tool
                        messages.append({"role": "system", "content": _FAKE_LINK_NOTE})
                        continue
                    text = FAKE_LINK_REPLY
                held = []
                streamed.append(text)
                yield AgentEvent("token", turn.conversation_id, text)
            elif held:
                item, held = "".join(held), []  # a tool call came with it: show the text as usual
                streamed.append(item)
                yield AgentEvent("token", turn.conversation_id, item)

            if not calls or not tools_allowed:
                reply = "".join(streamed)
                block = sources_block(reply, state.sources) + files_block(state.files)
                if block:
                    yield AgentEvent("token", turn.conversation_id, block)
                return
            round_calls = calls[:MAX_CALLS_PER_ROUND]
            async for event in self._run_calls(turn, round_calls, messages, state, "".join(round_text)):
                yield event
            # Image fast path, by explicit user request: create_image/edit_image results go
            # straight to the user, skipping the model's next round-trip - image generation is
            # already tens of seconds to minutes, so that extra LLM call (composing "here's your
            # image...") is a real, avoidable chunk of the wait after the image itself is ready.
            # Narrower than the old music fast-path below on purpose: exactly one call, exactly an
            # image tool, only on success - a compound request ("make an image AND tell me a joke")
            # still falls through to a normal round below and keeps its second half, unlike the
            # music one that dropped it outright (see the comment on that removal just below). This
            # one can still drop a second half in the same way if the *first* round both creates the
            # image and asks something else in one message - a real, accepted gap, not a fixed one.
            reply = image_only_reply(round_calls, state.last_results)
            if reply is None:
                reply = media_failure_reply(round_calls, state.last_results)  # the real reason, not a model's rewording
            if reply is not None:
                yield AgentEvent("token", turn.conversation_id, reply)
                return
            # No fast-path shortcut here for other tools (there was one, for music; removed - see
            # git history/CAPABILITIES.md "Music player" section): it returned a tool's own canned
            # text directly after a single play_music/control_music call, skipping the model's next
            # round entirely to save latency. Real bug from a live log: "stop the song tell me a
            # joke" fast-pathed on the successful control_music call and permanently dropped the
            # joke, since the model never got a turn to add anything after the tool result - a
            # compound request silently truncated to "Music stopped." By explicit user preference at
            # the time, a slower reply that actually addresses everything asked beat a fast one that
            # might not - the image fast path above knowingly takes the same real risk again,
            # narrowed to a single tool call, on a later, separate explicit user request.

    async def proactive_message(self, conversation_id: str | None) -> ChatResult:
        """Zira starts the conversation (hands-free voice, after a quiet spell): a question to learn
        about the user, a compliment, or a fun point about an interest. Stored as an assistant message
        with no user message before it; the user's spoken answer then goes through the normal chat
        (and memory extraction), which is how Zira learns from it."""
        conversation_id = conversation_id or new_conversation_id()
        built = self.context.build(
            conversation_id, "what do you know about me", event_note=random.choice(_PROACTIVE_DIRECTIVES)
        )
        messages = built.messages[:-1]  # no user message: Zira speaks first, the directive note is last
        text = (await self.llm.chat(messages, temperature=0.9)).strip().strip('"').strip()
        if not text:
            raise LLMError("The model returned an empty response.")
        self.conversations.add_message(conversation_id, "assistant", text)
        logger.info("Proactive message conversation=%s chars=%d", conversation_id, len(text))
        return ChatResult(conversation_id, text, built.estimated_tokens)

    async def extract_memories(self, message: str) -> list[Memory]:
        """Save durable facts from the user's message (no-op when auto-memory is off)."""
        if self.extractor is None:
            return []
        return await self.extractor.extract(message)

    async def maybe_checkpoint(self, conversation_id: str, estimated_tokens: int) -> dict | None:
        """Compact old history into a markdown summary once context is nearly full (no-op if
        checkpointing is off). Returns a small dict for the "checkpoint" event, or None."""
        if self.checkpoints is None:
            return None
        checkpoint = await self.checkpoints.maybe_checkpoint(conversation_id, estimated_tokens)
        if checkpoint is None:
            return None
        return {
            "path": str(self.checkpoints.path_for(conversation_id)),
            "covers_through_id": checkpoint.covers_through_id,
        }

    async def handle_message(
        self, conversation_id: str | None, message: str, mode: str = "chat", source: str = "text"
    ) -> ChatResult:
        turn = self._prepare(conversation_id, message, mode, source, await self._recalled(message))
        saved = False
        try:
            parts = [e.content async for e in self._generate(turn) if e.type == "token"]
            reply = "".join(parts).strip()
            if not reply:
                raise LLMError("The model returned an empty response.")
            self.conversations.add_exchange(turn.conversation_id, turn.user_message, reply)
            saved = True
        finally:
            self._close_job(turn, saved)
        return ChatResult(turn.conversation_id, reply, turn.estimated_tokens + estimate_tokens(reply))

    async def stream_message(
        self, conversation_id: str | None, message: str, mode: str = "chat", source: str = "text",
        stop: asyncio.Event | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """Yield start, (tool | token)..., done, then optionally memory.

        The exchange is saved only if the stream completes. Memory extraction runs after
        `done`, so it never delays the reply.
        """
        turn = self._prepare(conversation_id, message, mode, source, await self._recalled(message))
        if stop is not None:
            turn = replace(turn, stop=stop)
        saved = False
        try:
            yield AgentEvent("start", turn.conversation_id)
            parts: list[str] = []
            async for event in self._generate(turn):
                if event.type == "token":
                    parts.append(event.content)
                yield event
            reply = "".join(parts).strip()
            if not reply and turn.stopped:
                logger.info("Reply stopped before any text: nothing to save")
            elif not reply:
                raise LLMError("The model returned an empty response.")
            else:
                if turn.stopped:
                    logger.info("Reply stopped by the user after %d chars; the part shown is saved", len(reply))
                self.conversations.add_exchange(turn.conversation_id, turn.user_message, reply)
                saved = True
        finally:
            # A long generation's outcome reaches the history even if this stream was cut off
            # after the tool finished (a phone leaving while the reply was being written).
            self._close_job(turn, saved)
        yield AgentEvent("done", turn.conversation_id)

        saved = await self.extract_memories(message)
        if saved:
            yield AgentEvent("memory", turn.conversation_id, memories=tuple(saved))

        checkpoint_info = await self.maybe_checkpoint(turn.conversation_id, turn.estimated_tokens + estimate_tokens(reply))
        if checkpoint_info:
            yield AgentEvent("checkpoint", turn.conversation_id, data=checkpoint_info)
