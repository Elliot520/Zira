# `agent.py` - what it is and how it is used

## What it is

`agent.py` is the part of Zira that handles **one user message from start to finish**: it builds the
context, decides whether a tool is needed, runs the tool(s), streams the reply, and saves the exchange.
Everything else (the LLM, memory, tools, the web API) is plugged into it.

One `Agent` is created at startup in `app/main.py` and stored as `app.state.agent`.
It is given: the LLM, memory, the conversation store, the context builder, the tool registry, and optionally
the memory extractor and the checkpoint manager.

## Who calls it

| Entry point | File | Calls | Used by |
|---|---|---|---|
| `POST /api/chat` | `app/api/chat.py` | `handle_message()` - returns the whole reply at once | scripts, tests |
| `POST /api/chat/stream` | `app/api/chat.py` | `stream_message()` - server-sent events | simple clients |
| `WebSocket /ws/chat` | `app/api/chat.py` | `stream_message()` | **the web UI** (desktop and phone) |

After a reply, the API layer also runs `extract_memories()` (saves durable facts) and `maybe_checkpoint()`
(summarises old history when the context is nearly full).

## What happens to one message

```
message
  |
  v
_prepare ........ conversation id, "remember ..." command, bare-greeting note,
                  ContextBuilder.build -> system prompt + history + memories
  |
  v
_generate ....... 1. adult image request?  -> force create_image        (skips the LLM's decision)
                  2. video request?        -> force create_video        (skips the LLM's decision)
                  3. otherwise the Planner: explicit web lookup / lyrics -> force web_search
                  4. round loop (up to 6 rounds): the LLM streams text and may ask for tools
  |
  v
_run_calls ...... executes each tool as a task, streams progress, applies the safety gates
  |
  v
reply ........... + "Sources:" list (web search) + "Files:" list (tools that made files)
                  or, for a single successful image/edit/video, just the link (image_only_reply)
  |
  v
stream_message .. saves the exchange, sends `done`, then `memory` / `checkpoint` events
```

## Events it produces (`AgentEvent`)

| type | meaning | shown by the UI as |
|---|---|---|
| `start` | conversation id | (internal) |
| `tool` | a tool is about to run (`content` = short label, `tool` = name) | "Generating video: ..." status |
| `image_progress` | step N of M plus ETA. Used by **image and video** generation | "step 5/16 (~300s left)" |
| `token` | a piece of the reply text (or a bare file link) | the chat bubble |
| `change` / `music` | a proposed code edit / a music command | diff card / the player |
| `done` | the reply is complete and saved | - |
| `memory` / `checkpoint` | something was remembered / history was summarised | small notes |

## The routes that skip the LLM's decision

> **Currently disabled (2026-09-27, by user request).** Both routes below are commented out in
> `Agent._generate`, so every image and video request goes through the model. The model writes an
> improved, descriptive prompt for the tool, instead of the raw message being sent word for word.
> The cost is that the model can refuse; that happened on 2026-09-27. To turn the routes back on,
> uncomment the block marked `DIRECT BYPASS - DISABLED` and change the `if self.tools.names():` after it
> back to `elif`. The two tests that cover the routes are skipped until then (`BYPASS_OFF`).

Small local models sometimes refuse a tool call or forget to make it, so two request types are routed in
plain Python, before the model is asked anything:

| Route | Triggers when the message... | Then |
|---|---|---|
| Adult image | is a direct image request **and** contains an adult keyword (`_IMAGE_REQUEST_RE` + `_ADULT_IMAGE_RE`) | calls `create_image` with the whole message as the prompt |
| Video | is a direct video request (`_VIDEO_REQUEST_RE`, e.g. "create a video clip of...") | calls `create_video` with the whole message as the prompt |

- **On success** the reply is only the file link. No sentence from the model is added.
- **On failure** (for example the model is set to None, or the tool errors) the failure goes to the model,
  which explains it in its own words. That is why "video model is switched off" comes back as a model-written
  sentence.
- The tool still runs its own checks. The always-on minor-safety check lives **inside** `create_image` and
  `create_video`, not here, so these routes cannot skip it.
- **Ordinary image requests** (no adult keyword) are not forced: the planner and the model decide.
- **Phrasings the video regex matches:** "create/make/generate/render/produce ... video|clip|animation|animated|cartoon|movie",
  and "video|clip|animation|cartoon ... of|showing|depicting|about". Examples that match: "create a video clip of a
  cat", "I want a video of a dog", "generate animation of a rocket".
- **Phrasings it does not match** (these still work, but the model decides): "tom and jerry video",
  "animated cartoon tom", "video: cat chasing mouse", "please animate a rocket launch".
- **How to tell which path ran:** in `logs/server.log`, a forced route shows `Tool call: create_video` within
  about a second of `Context built`. When the model or planner chose the tool, it takes several seconds.
- **Known gap:** a compound request such as "make a video and tell me a joke" returns only the link, because a
  success ends the turn immediately.

## Tool execution (`_run_calls`)

For each tool call, in order:

1. **Gates** - the tool must exist and be allowed in the current mode (`chat`, `plan` or `edit`); tools that send
   data out are disabled for the rest of the turn once a tool that reads private files has run (`tainted`);
   `max_calls_per_turn` is enforced.
2. **Run** - the tool runs as an asyncio task. Anything it reports through `current_progress_reporter` is turned
   into `image_progress` events while it is still running, so long jobs (images, videos) show live progress.
3. **Result** - files, sources, code-change proposals and music commands are collected; the result text is added
   to the model's context. Old tool output is shortened once it exceeds `TOOL_CONTEXT_BUDGET`.
4. **Client left mid-tool** - if the phone locks or the WebSocket closes, `_save_image_after_disconnect` stores
   the finished link in the conversation when the tool completes, so it is there after reload. It covers
   `create_image`, `edit_image` and `create_video`.

## The helper functions

| Function | Purpose |
|---|---|
| `sources_block(reply, sources)` | Builds the "Sources:" list from the **real** search results, never from URLs the model wrote |
| `files_block(files)` | Builds the "Files:" list of what tools produced |
| `image_only_reply(calls, results)` | The link-only reply for exactly one successful `create_image` / `edit_image` / `create_video` |
| `compact_tool_messages(messages)` | Replaces the oldest tool output with a stub once the total is over the budget |
| `_bare_greeting_note(text)` | Turns "Hello Zira" alone into a varied greeting instruction |
| `_is_direct_image_request` / `_is_direct_video_request` | The routing tests used above (unused while those routes are disabled) |

## Limits and constants

| Constant | Value | Meaning |
|---|---|---|
| `MAX_TOOL_ROUNDS` | 6 | Rounds of tool calls per message. The last round has tools switched off so the model must answer |
| `MAX_CALLS_PER_ROUND` | 2 | Tool calls the model may make in one round |
| `TOOL_CONTEXT_BUDGET` | 9000 chars | Tool output kept in the model's context |
| `TOOL_ANSWER_TEMPERATURE` | 0.3 | Lower temperature for answers grounded in tool results |
| `MAX_SOURCES_SHOWN` | 4 | Most citations listed under a web-search answer |

## How to change or extend it

- **Add a tool:** write it in `app/tools/`, register it in `app/main.py`. The agent needs no change unless you
  also want a link-only reply (add its name to the tuple in `image_only_reply` and `_save_image_after_disconnect`)
  or a forced route.
- **Add a forced route:** add a regex and a helper next to `_VIDEO_REQUEST_RE`, then an `elif` branch in
  `_generate` modelled on the video one. Add a test like the forced-routing tests in `tests/test_video.py`.
- **Change what the planner forces** (web lookups, lyrics): that is `app/agent/planner.py`, not this file.
- **Change which tools the model is offered:** each tool's `relevant()` and `modes` (see `ToolRegistry.schemas`).

## Things to watch for

- **Restart needed.** The server loads `agent.py` once at startup. Saved changes take effect only after a restart.
- **Save the file.** An edit that only exists in the editor is not on disk, so the server never sees it. This
  happened once with the video route.
- **`MAX_SOURCES_SHOWN` must stay defined above `sources_block`.** It was once deleted by accident while pasting
  the video block. Every reply with web-search sources then fails with a `NameError`. The web-search tests catch it.
- **Run the tests after any edit:** `.venv/bin/python -m pytest -q` (relevant files: `tests/test_video.py`,
  `tests/test_image.py`, `tests/test_web_search.py`, `tests/test_chat.py`, `tests/test_planner.py`).
