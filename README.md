# Zira (project: jarvis-ai) — a local AI companion

> **Naming:** the assistant is called **Zira** (set with `ASSISTANT_NAME`) and its wake phrase is
> "Hello Zira". The project, folder, logger names and saved browser settings still say `jarvis`, on purpose
> (renaming them would gain nothing and would reset your saved chat/mode/voice settings). "Hello JARVIS" still
> works as a quiet backup wake word, and much of this document still says "JARVIS" where it means the assistant.

Zira is a personal AI companion that runs **entirely on your Mac**. It talks to a local
LLM through [Ollama](https://ollama.com) (default: **Qwen3 8B**), remembers durable facts about you
across conversations, can search the web when it doesn't know something, and is structured so it can later become the brain of a
Raspberry Pi robot.

**Status: Phase 1 — text chat + local LLM + SQLite memory, plus automatic memory and a web-search fallback. Working.**
Voice, vision and the robot link are architecture-only (see [Roadmap](#roadmap)).

- Your conversations, memories and the model all stay on your machine, and the server binds to `127.0.0.1` only.
  **One exception:** when JARVIS decides it needs the internet, the *search query* it writes is sent to
  DuckDuckGo (see [Web search](#web-search)). Turn that off with `WEB_SEARCH_ENABLED=false`.
- The model is swappable: change `OLLAMA_MODEL` in `.env`, or implement a new backend
  behind `app/ai/llm.py::LLMBackend`.

---

## 1. Requirements

| Requirement | Notes |
|---|---|
| macOS on Apple Silicon | Developed on a MacBook Air M4, 16 GB. Qwen3 8B (4-bit) needs ~5–6 GB of RAM. |
| Homebrew | https://brew.sh |
| Python 3.12+ | 3.14 works (this project was built on 3.14). |
| Ollama | Runs the model locally. |
| ~6 GB free disk | For the `qwen3:8b` download (5.2 GB). |

## 2. macOS setup

### Quick path

```bash
cd jarvis-ai
./setup.sh      # checks everything, creates .venv, installs deps, pulls the model
./run.sh        # starts JARVIS  ->  http://127.0.0.1:8000
```

`setup.sh` is safe to run repeatedly. It never installs system software for you; if something
is missing it prints the command to run.

### Manual path

**Homebrew**

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

**Python**

```bash
brew install python@3.12      # or any 3.12+
```

**Ollama**

```bash
brew install ollama
ollama serve                  # leave running (or: brew services start ollama)
```

**Qwen3 8B**

```bash
ollama pull qwen3:8b          # ~5.2 GB
ollama run qwen3:8b "Say hi"  # optional smoke test
```

**Virtual environment and dependencies**

```bash
cd jarvis-ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## 3. Running the application

```bash
./run.sh
```

`run.sh` checks Python and dependencies, checks that Ollama is running (starting it in the
background if it is not, and stopping it again when JARVIS exits), checks the model is installed,
starts FastAPI, and prints the URL:

```
http://127.0.0.1:8000
```

**Stop JARVIS:** press `Ctrl+C` in the terminal running `./run.sh`.
(If it was started in the background: `kill $(lsof -tiTCP:8000 -sTCP:LISTEN)`.)

The first reply after starting is slower because Ollama loads the model into memory;
`OLLAMA_KEEP_ALIVE` (default `30m`) keeps it warm afterwards.

### Started by macOS (auto-start, on this Mac since 2026-09-27)

On this Mac, macOS runs Zira through a LaunchAgent, `~/Library/LaunchAgents/com.zira.server.plist`:
- It starts `run.sh` at login and starts it again if it ever exits with an error.
- `run.sh` still restarts Zira after a crash or a model-mode switch.
- Output goes to `logs/server.log`.
- FileVault is on, so after a reboot or power cut someone has to log in at the Mac once.

```bash
launchctl bootout gui/$(id -u)/com.zira.server                                    # stop (stays stopped)
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.zira.server.plist     # start
launchctl kickstart -k gui/$(id -u)/com.zira.server                               # restart
```

Don't use `./run.sh` in a terminal while the LaunchAgent is running: port 8000 is taken, so it stops with an error.

**Reached through Tailscale only (since 2026-09-27).** `HOST=127.0.0.1`, so the phone uses
`https://<this-mac>.<tailnet>.ts.net` (Tailscale Serve, which proxies to 127.0.0.1). The LAN-IP and
Tailscale-IP `:8000` rows below apply only if `HOST` is opened up again. The Tailscale Serve name is always
allowed as a Host and Origin, whatever `HOST` is.

## Using JARVIS from your phone (same Wi-Fi, or anywhere via Tailscale)

Everything - the LLM, image generation, speech-to-text - keeps running on the Mac; the phone is
only a browser pointed at it. JARVIS has **no login of its own**, so only expose it to networks you
trust: never port-forward it on your router, and don't put it on a public tunnel.

| From | URL | Needs |
|---|---|---|
| The Mac | `http://127.0.0.1:8000` | nothing |
| Phone on the same Wi-Fi | `http://<mac-lan-ip>:8000` (`ipconfig getifaddr en0`) | `HOST=0.0.0.0` in `.env` |
| Phone anywhere (office, cellular) | `http://<tailscale-ip>:8000` (`tailscale ip -4`) | Tailscale on both devices, `HOST=0.0.0.0` |

`HOST=0.0.0.0` makes the server listen on every interface. Two allowlists still apply on top of that
(`app/config.py`: `allowed_hosts` / `allowed_origins`, enforced by Starlette's `TrustedHostMiddleware`
and the WebSocket origin check): when `HOST` isn't loopback-only, the Mac's LAN IP, its Tailscale IP
and its Tailscale MagicDNS name are detected at startup and added automatically. **Restart JARVIS
after joining/leaving a network or installing Tailscale** so that detection reruns - a request whose
`Host` isn't allowlisted gets a plain `400`, even though the socket accepted it.

**Tailscale setup (macOS, CLI variant):**

```bash
brew install --formula tailscale
sudo brew services start tailscale
sudo tailscale up            # prints a login URL; approve it in a browser
tailscale ip -4              # this Mac's 100.x.y.z address
```

Install the Tailscale app on the iPhone, sign in with the same account, and switch it on.

### Phone layout: one settings menu

On a phone-width screen (640px or narrower) the crowded header and composer are replaced by a compact
top bar - a settings (gear) button, **+ New chat** beside it, and the Zira name with a status dot - with the
mic and the *Continuous* checkbox right under it. Tap the gear (or swipe from the menu, tap outside it, press
Escape) to open a slide-out **Settings** panel grouped into cards: **Mode** (Chat / Plan / Edit, with its
explanation), **Model** (Light / Deep and the loaded model), **Image generation** (style REALISTIC /
LIGHTNING / FLUX with the full model name underneath, and 720 / 1024 resolution) and **Voice** (the speech-model
power switch). The upload (paperclip) button stays in the message row, now inline beside the text box so chat
gets more room. A group with nothing to show (image generation or speech-to-text not enabled) is hidden.

`frontend/mobile-menu.js` does this by *moving* the existing controls into the panel (not copying them), so
all of their behaviour is unchanged, and it puts them back when the window is wider than 640px - desktop looks
exactly as before (checked pixel-for-pixel against the previous page). If the script cannot run, the previous
mobile layout is left as it was. The web server now sends `Cache-Control: no-cache` for the UI files, so a normal
reload picks up UI changes instead of a stale cached page.

### Music and spoken replies play on the phone, not the Mac

The server never plays audio itself: `play_music` returns a URL and the browser plays it, and spoken
replies (`say -o file.wav`) are returned as a WAV for the browser to play. So from a phone they come
out of the phone. `/api/music/local` answers HTTP range requests (`206`), which iPhone Safari needs to
stream a track rather than download it whole. iOS only starts audio from inside a tap, so
`frontend/app.js` keeps one long-lived audio element per role and "unlocks" both with a silent clip
on the first tap anywhere; later plays triggered by a server message reuse the unlocked element.
(Verified in a Node simulation of that logic; not yet confirmed on a physical iPhone.)

### Voice input from the phone needs HTTPS

Browsers expose the microphone (`getUserMedia`) only on a *secure context* - HTTPS or `localhost`.
`http://<tailscale-ip>:8000` is neither, so the mic button works in the Mac's browser but is blocked
on the phone. The recorder already produces `audio/mp4` (what iPhone Safari records) and the server
decodes it with `ffmpeg`, so HTTPS is the only missing piece. Two ways round it:

1. **No setup:** dictate with the iPhone keyboard's own microphone key while the message box is
   focused. That's Apple's speech recognition rather than the Mac's Whisper, but it works over plain
   HTTP.
2. **HTTPS through Tailscale (tailnet-only, not public), in this order:**
   1. **Rename the Mac first** to something generic, e.g. `zira-mac` - in the Tailscale admin console
      (Machines), or from the Mac with `tailscale set --hostname=zira-mac` (needed no password here).
      Enabling HTTPS publishes the machine's name in public certificate-transparency logs (names only;
      access to your devices stays restricted) - and the default name is built from your own name. Free:
      certificates come from Let's Encrypt and Serve is included in the free Personal plan.
   2. Enable Serve for the tailnet once (the link `tailscale serve` prints, of the form
      `https://login.tailscale.com/f/serve?node=...`), and enable HTTPS certificates under DNS if it asks.
   3. Run `tailscale serve --bg --https=443 8000`, then **restart Zira** so it allowlists the (new) tailnet
      name, and open `https://<mac-name>.<tailnet>.ts.net` on the phone (here:
      `https://zira-mac.tail4c16b2.ts.net`). `tailscale serve --https=443 off` undoes it.
      Plain `http://<tailscale-ip>:8000` keeps working alongside.

Once the page is on HTTPS, hands-free ("Continuous") works from the phone. Three iPhone-specific fixes live
in `frontend/app.js`: the silence detector reuses **one** `AudioContext` created inside your first tap (iOS
leaves a context made outside a tap suspended, which used to break end-of-speech detection from the second
turn on); the screen **wake lock** is held while Continuous is on (an auto-locking iPhone suspends the page and
mic); and a mic stream that iOS ended is re-acquired instead of recording silence. iOS still suspends web pages
when Safari is backgrounded or the screen locks, so hands-free is **foreground only**. Not yet confirmed on a
physical iPhone: reply loudness/routing while the mic is open, and battery use.

## Model mode (LIGHT/BALANCED/DEEP)

On a 16GB Mac, keeping multiple LLMs resident in Ollama at once isn't realistic. JARVIS supports
three modes and guarantees **only one model is ever loaded at a time**:

| Mode | Model | For |
|---|---|---|
| **LIGHT** | `qwen3-heretic:4b` (2.5GB, Q4_K_M) | realtime conversation, voice, casual Hinglish chat, low latency: [`DreamFast/qwen3-4b-heretic`](https://huggingface.co/DreamFast/qwen3-4b-heretic), Qwen3 4B with refusals removed (3/100) |
| **BALANCED** | `qwen3-heretic:8b` (4.8GB, Q4_K_M) | everyday chat and tasks: [`jbeslt/Qwen3-8B-heretic`](https://huggingface.co/jbeslt/Qwen3-8B-heretic), Qwen3 8B with its refusals removed |
| **DEEP** (default) | `qwen3:8b` | coding, reasoning, project analysis, tools, complex tasks |

Both share the exact same conversation history, long-term memory, personality, tools, project
context, and voice pipeline — only the underlying model changes. Switching is a button toggle in
the UI header (**LIGHT** / **BALANCED** / **DEEP**, or the Model card in the phone menu), or
`POST /api/model/switch`. BALANCED supports tools and the thinking switch exactly like DEEP (same Qwen3 chat
template); JARVIS sends thinking off by default (`OLLAMA_THINK`), which keeps replies fast.

**Setup:**
```bash
```
LIGHT (`qwen3-heretic:4b`) comes from a ready-made GGUF, so it only needs registering:
```bash
hf download DreamFast/qwen3-4b-heretic gguf/qwen3-4b-heretic-Q4_K_M.gguf --local-dir heretic4b
# Modelfile: "FROM ./heretic4b/gguf/qwen3-4b-heretic-Q4_K_M.gguf" + the TEMPLATE and PARAMETER lines of
# `ollama show qwen3:8b --modelfile`
ollama create qwen3-heretic:4b -f Modelfile
```
BALANCED (`qwen3-heretic:8b`) is not on the Ollama registry: `jbeslt/Qwen3-8B-heretic` only ships full-size
safetensors (16.4GB), so it is built locally once (needs ~30GB free disk while converting; the leftovers are
deleted afterwards):
```bash
brew install llama.cpp
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
hf download jbeslt/Qwen3-8B-heretic --local-dir heretic-src
PYTHONPATH=llama.cpp/gguf-py python llama.cpp/convert_hf_to_gguf.py heretic-src --outtype q8_0 --outfile heretic-q8_0.gguf
llama-quantize --allow-requantize heretic-q8_0.gguf heretic-Q4_K_M.gguf Q4_K_M
# Modelfile: "FROM ./heretic-Q4_K_M.gguf" + the TEMPLATE and PARAMETER lines of `ollama show qwen3:8b --modelfile`
ollama create qwen3-heretic:8b -f Modelfile
```
`.env`:
```bash
MODEL_MODE=deep          # deep (default), balanced or light - which mode starts on launch
LIGHT_MODEL=qwen3-heretic:4b
BALANCED_MODEL=qwen3-heretic:8b
DEEP_MODEL=qwen3:8b       # DEEP_MODEL unset falls back to the older OLLAMA_MODEL if you have one
```

**How switching actually works** — this is a real process restart, not an in-place swap, so there
is never a window where both models might be resident together:
1. Any playing TTS is stopped (the same manual-interrupt path the mic button already uses).
2. The target model is confirmed installed (a clear error if not — no restart attempted).
3. The currently-loaded model is unloaded from Ollama (`keep_alive=0`) and **verified gone** via
   `ollama ps` before continuing — never just assumed.
4. The new mode is persisted to `.env`, and JARVIS restarts (a few seconds of downtime — the page
   shows progress and reloads itself automatically once the new process is back up).
5. Your conversation, memory, and settings are untouched — all of that lives in SQLite, not in
   process memory, so a restart never loses anything.

**Voice/text parity**: play/pause/switch-mode work identically whether triggered by chat or voice —
voice input already flows through the same pipeline as typed text, so there was nothing extra to
build for this.

**LIGHT-mode tuning**: LIGHT always runs with extended thinking off (immediate response is the
whole point) regardless of `OLLAMA_THINK`, and voice-originated replies in LIGHT are capped to
`LIGHT_VOICE_NUM_PREDICT` tokens (default 200) to keep speech responses snappy — typed LIGHT chat
and all of DEEP are unaffected. The system prompt also gets a short "keep it brief" addition while
LIGHT is active, unless you clearly ask for depth or code.

**Known limitations:**
- This is a real restart — a few seconds of downtime, and any turn genuinely in progress at the
  moment of switching is lost (nothing was streaming that's worth preserving across a restart).
- No automatic hands-free barge-in exists (interrupting JARVIS mid-reply just by talking) — only
  manual mic-button-click interruption does, identically in both modes.

## 4. Testing chat

**Browser:** open http://127.0.0.1:8000. The header shows `AI ONLINE` and the model name.

**curl (single reply):**

```bash
curl -s http://127.0.0.1:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message": "Hello JARVIS"}'
# {"conversation_id":"3f1c…","response":"Hello! …"}
```

Pass the returned `conversation_id` back to continue the same conversation.

**curl (streaming, Server-Sent Events):**

```bash
curl -N http://127.0.0.1:8000/api/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{"message": "Tell me a fun fact"}'
# data: {"type": "start", "conversation_id": "…"}
# data: {"type": "token", "conversation_id": "…", "content": "Honey"}
# …
# data: {"type": "done", "conversation_id": "…"}
# data: {"type": "memory", "conversation_id": "…", "memories": [{"id": 3, "text": "My name is Sam.", "category": "personal"}]}
#   (a "tool" event, {"tool": "web_search", "detail": "<query>"}, appears before tokens when JARVIS searches)
```

**WebSocket** (`/ws/chat`, used by the web UI): send `{"conversation_id": null, "message": "hi"}`,
receive `start`, optionally `tool`, many `token`, then `done` (or `error`), and sometimes a `memory` event
after `done`. The connection stays open for more messages.

**Health:** `curl http://127.0.0.1:8000/api/health`

**Automated tests** (no Ollama needed; the LLM is faked):

```bash
.venv/bin/python -m pytest
```

### Errors

Errors always look like `{"error": "<code>", "detail": "<what to do>"}`:

| Situation | HTTP | `error` |
|---|---|---|
| Empty / invalid / too-long message | 422 | `invalid_request` |
| Ollama not running | 503 | `ollama_unavailable` |
| Model not installed | 503 | `model_not_found` |
| Ollama too slow | 504 | `llm_timeout` |
| Stream broke mid-reply | 502 (or an `error` event) | `stream_interrupted` |
| SQLite failure | 500 | `database_error` |

A reply is only saved to history once it has been generated completely.

## 5. Memory examples

JARVIS has two kinds of memory, both in SQLite (`data/jarvis.db`):

1. **Conversation memory** — every message, with timestamp and conversation ID. The most recent
   messages of the current conversation are sent to the model (`CONTEXT_MAX_MESSAGES`, default 20).
2. **Long-term memory** — durable facts about you: text, category, importance, created/updated
   timestamps. They are saved two ways, and both are shared by **every** conversation, so a new
   conversation still knows them:
   - **Explicitly**, when you say "Remember that …".
   - **Automatically**, when you just state something durable ("Hi, my name is Sam and I live in
     Berlin"). See [Automatic memory](#automatic-memory).

```
You:    Remember that I prefer Kotlin.
JARVIS: Got it — I'll remember that you prefer Kotlin.
        (saved as  preference: "I prefer Kotlin.")

You:    What programming language do I prefer?
JARVIS: You prefer Kotlin.
```

Recognised phrasings: "remember that …", "please remember …", "don't forget that …",
"keep in mind that …", "make a note that …". Categories (`preference`, `personal`, `work`,
`project`, `instruction`, `fact`) are picked by simple keyword rules.

### Automatic memory

After JARVIS has replied, the model reads **your message** (never its own reply) and saves durable facts
such as your name, where you live, your job, projects, preferences and standing instructions. The UI
shows a green *"Saved to memory: …"* note under the reply. It runs after the reply is sent, so it never
delays the answer.

```
You:    Hi, my name is Sam and I live in Berlin
JARVIS: Hello, Sam! …
        Saved to memory: My name is Sam. I live in Berlin.
```

Guardrails, because a small local model can misjudge:

- Messages with no first-person content ("write a function that…") skip extraction entirely.
- Questions, one-off requests, moods ("I'm tired") and anything about JARVIS are not saved.
- Output is schema-constrained, then re-checked (3–200 chars, at least 3 words, not a question).
- It is **add-only**: it never edits or deletes an existing memory. If a fact changes (a new name),
  delete the stale one yourself (`DELETE /api/memories/<id>`).
- Extraction failures are logged and ignored; they can never break a chat.

Turn it off with `AUTO_MEMORY=false` (explicit "Remember that …" still works). Memories are
*not* logged; only counts and categories are.

Inspect and manage memories:

```bash
curl -s http://127.0.0.1:8000/api/memories                 # list
curl -s -X DELETE http://127.0.0.1:8000/api/memories/1     # delete (forget)
sqlite3 data/jarvis.db 'select id, category, text from memories;'
```

**How recall works:** before each request the context builder assembles
`system prompt + relevant memories + recent conversation + your message`. With up to 20
memories, all are included and the model picks what is relevant. Beyond that, memories are ranked by
keyword overlap, importance and recency and only the top `MEMORY_TOP_K` (default 8) are sent. The
whole database is never sent. (Embedding-based recall is on the roadmap.)

## Web search

If JARVIS doesn't know something (news, recent events, prices, recent software releases, or a fact it
isn't sure of), it can search the internet and answer from the results. Qwen3's tool calling decides
*when*; JARVIS answers from its own knowledge and your memories otherwise.

```
You:    What is the latest stable version of Python right now?
JARVIS: Searching the web for "latest stable version of Python"…
        The latest stable version is 3.14.7, per the official Python site.
        Sources: https://www.python.org/downloads/latest/
```

- Uses [DuckDuckGo](https://duckduckgo.com) through the `ddgs` package: **no account or API key**.
  It is an unofficial interface, so it can occasionally be slow or rate-limited (one automatic retry;
  on failure JARVIS tells you it couldn't search and answers from what it knows).
- **Explicit lookups are forced.** If you ask for lyrics, or say "search for…", "google…", "look it up" or
  "latest news…", JARVIS runs the search itself; the model only writes the query, using the conversation to
  work out which song or topic you mean. (An 8B model sometimes says "let me search" and then doesn't, and
  invents links; this prevents that.) Requests to *write* lyrics don't trigger a search.
- **Sources are real.** Every web answer ends with a *Sources* list built from the actual search results,
  never from URLs the model wrote, and URLs are clickable in the UI.
- Queries that literally ask for news/headlines use a news search restricted to the past week.
- **Song lyrics:** JARVIS will not print the full lyrics of modern copyrighted songs. It searches, gives a
  one-line description, a short excerpt and links to the lyrics sites. It writes original lyrics on request
  and prints public-domain songs (nursery rhymes, hymns, folk songs) in full, though the model can misremember
  a verse.
- **Privacy:** the query JARVIS writes leaves your machine; nothing else does. The UI shows exactly
  what is being searched. JARVIS is instructed never to put personal details in a query, but it is a
  small model, so disable search if that matters: `WEB_SEARCH_ENABLED=false`.
- **Safety:** search results are treated as untrusted text. The only tool is `web_search`. It cannot
  open arbitrary URLs, run commands or read files. At most 2 tool rounds and 2 searches per round per message.
- **Speed:** expect roughly 40–90 s for an answer that needs a search on a 16 GB MacBook Air (a few seconds of
  searching, then the model reads several thousand tokens of results and writes the answer). Replies that
  don't need the internet take 2–15 s. To speed things up, close memory-hungry apps or try a smaller model
  (`OLLAMA_MODEL=qwen3:4b`, then `ollama pull qwen3:4b`); expect somewhat less reliable answers.
- **Limits:** answers rest on search-result snippets, not full pages, and an 8B model can still
  misread them. Check the sources it lists for anything important.

## Music (your own library, local + self-hosted)

JARVIS can search for and play songs by name/artist through chat or voice ("play believer",
"pause", "stop"). Deliberately **not** a search engine for the open internet: it never scrapes,
downloads, or streams from third-party catalogs. Every track is something you already have —
either a file on your own Mac, or hosted on a server you control. This sidesteps licensing
questions entirely: JARVIS is a player for your own music, not a search engine for the internet's.

**How it decides where to look:** your local folder first; only if nothing matches there does it
check your remote index (if you've configured one).

**Setup — local folder:**
```bash
# .env
MUSIC_LIBRARY_ROOTS=/Users/you/Music/MyLibrary
```
JARVIS searches filenames (not ID3 tags) for a match, case-insensitively, across `.mp3`, `.m4a`,
`.flac`, `.wav`, `.ogg`, `.aac`. Sandboxed exactly like project file access (`app/tools/filesystem.py`):
only files under the configured folder(s) can ever be read, path traversal is blocked, and files
are streamed through JARVIS's own backend (`GET /api/music/local`) rather than exposing raw
filesystem paths to the browser.

**Setup — your own server (optional fallback):** host your audio files anywhere you like, and
maintain a small JSON index file listing them:
```json
[
  {"title": "Song Name", "artist": "Artist Name", "url": "https://your-server.example/music/song.mp3"}
]
```
```bash
# .env
MUSIC_REMOTE_INDEX_URL=https://your-server.example/music/index.json
```
JARVIS fetches and caches this index (re-checked every `MUSIC_REMOTE_INDEX_CACHE_SECONDS`, default
300s) and searches title+artist the same way it searches local filenames. Remote tracks are played
directly from your server's URL — JARVIS's backend never proxies or stores them.

**Mood-based requests ("I'm sad, play something"):** add a `library.json` file directly inside a
local folder (same shape as the remote index above, plus a `mood` list) and JARVIS will search it
too — filename alone usually carries no mood information, so this is what makes "play something
happy/sad/energetic" work instead of just "play <song name>":
```json
[
  {"title": "Song Name", "artist": "Artist Name", "mood": ["happy", "energetic"],
   "url": "/api/music/local?path=Song%20Name.mp3"}
]
```
A file with no entry in `library.json` still stays searchable by filename alone — this is additive,
not required. Suggested mood vocabulary (the model is told about these): `happy`, `sad`,
`energetic`, `chill`, `romantic`, `angry`, `nostalgic`, `party` — but any words work, since matching
is just keyword overlap against whatever you put in `mood`.

**Voice/text parity:** play/pause/resume/stop work identically whether typed or spoken — voice
input already flows through the same chat pipeline as typed text (no separate voice path exists),
so there's nothing extra to configure for this.

**Continuous voice mode safety:** starting a song immediately requires the wake word ("Hello
JARVIS") again for your next voice command, rather than relying on the usual 10-second no-wake-word
grace window. Without this, music playing through your speakers could bleed into the mic and get
mistaken for a follow-up command. (Same mechanism already used when you say "goodbye" to end a
hands-free session immediately — see `requireWakeWordAgain()` in `app.js`.)

**Privacy:** your search query never leaves the machine for local tracks. For a remote-index
search, only the index URL (to fetch/refresh the manifest) and, if a track plays, its own URL are
contacted — both are servers you configured yourself.

**If a song shows "Now playing" but you don't hear anything:** browsers block `audio.play()` calls
that aren't tied to a direct, recent click/tap — and an AI-triggered play command comes from an
async reply (however many seconds the model took to respond), not a click, so it can be silently
blocked by the browser's autoplay policy. Real, verified behavior: JARVIS now catches this and shows
"Playback was blocked by your browser. Press ▶ on the music player below to start it." — pressing
that ▶ button *is* a direct click, which browsers always honor, so it reliably starts the track even
when the automatic trigger was blocked. Previously this failure was silent (only visible in the
browser console) — the AI's own narration ("Now playing X") could be right while nothing was
audible, with no indication anything had gone wrong.

**Known limitations:**
- Search matches on filename (local) or title/artist (remote index) by simple keyword overlap —
  not fuzzy/typo-tolerant, and local search doesn't read ID3 metadata directly (only what you put
  in an optional `library.json`).
- Mood tags are whatever you (or an assistant helping you write `library.json`) assign them —
  there's no automatic audio analysis, so accuracy depends entirely on how the tags were chosen.
- No queue/playlist — playing a new song replaces whatever's playing; there's no "next" beyond
  asking for another song by name.
- No persistence across a page reload (same as spoken replies today) — what's currently
  playing/paused is lost on refresh.
- Music and spoken replies are mutually exclusive: starting one pauses the other, with no
  automatic resume afterward.
- Autoplay can be blocked by the browser on an AI-triggered play (see above) — always recoverable
  with one click on the player's own ▶ button, never a dead end.

## Image generation (local, offline, Apple Silicon)

Off by default (`IMAGE_GENERATION_ENABLED=false`) — a `create_image` tool that turns a text prompt
into a real, downloadable PNG using [`diffusers`](https://github.com/huggingface/diffusers) on
PyTorch's Apple Silicon GPU backend (MPS). Same download-link pattern `create_pdf` and Postman
collections already use.

**Setup:**
```bash
pip install diffusers accelerate   # not in requirements.txt - real, separate dependencies, opt-in on purpose
```
```bash
# .env
IMAGE_GENERATION_ENABLED=true
```
The default model (`stabilityai/sd-turbo`, ~2GB) downloads automatically on first use.

**Real measured numbers** (this M4, `stabilityai/sd-turbo`, 1 inference step):
- First call ever: ~232s (includes the one-time ~2GB download + model load).
- A real end-to-end chat request ("create an image of a cute cat sitting on a laptop") with the
  model already cached: **~48s total** (Ollama deciding to call the tool + loading the pipeline
  from disk onto the GPU + generating + writing the reply) — genuinely good, on-prompt output,
  not a placeholder.
- The pipeline is loaded once and cached for the life of the server process (mirrors how
  `FasterWhisperSTT` caches its model) — only the very first image after a restart pays the load
  cost; generation alone (once warm) measured at ~6s for a single step.

**Why `diffusers` + `stabilityai/sd-turbo`, not `mflux` + FLUX/Z-Image**: `mflux` (MLX-native,
Apple's own framework) was tried first, since it matches the STT engine's successful acceleration
approach. Three models were checked for real before landing here:
- `FLUX.1-schnell` (mflux's flagship) is a **gated** Hugging Face repo — needs your own HF account,
  accepting Black Forest Labs' license, and a token. Not something this project can complete on
  your behalf.
- `z-image-turbo` (`Tongyi-MAI/Z-Image-Turbo`) is ungated and downloads fine, but **crashed on
  load**: a real, reproduced `FileNotFoundError: No safetensors files found in .../text_encoder_2`
  — the actual repo only ships one text encoder; mflux 0.20.0 (confirmed the current latest
  release) expects two for this model. A real bug in mflux's z-image-turbo support, cost a full
  26-minute, 32.8GB download to discover (cleaned up afterward, unusable).
- `Qwen/Qwen-Image` (the other ungated option) was ruled out without attempting it: 57.7GB, nearly
  double what `z-image-turbo` turned out to be, with no guarantee of avoiding its own issues after
  that long a download.

Switched to `diffusers` (Hugging Face's own library, far more mature and widely used than the
newer MLX-native ecosystem) with `stabilityai/sd-turbo` (confirmed ungated, designed for
single-step generation) — worked correctly on the first real attempt, no library bugs hit. If
you've done the FLUX gated-access setup yourself and prefer it, `mflux` remains a valid alternative
architecture, just not the default given the above.

`create_image` calls `diffusers` directly in-process (unlike the `mflux` attempt, which had to
shell out to a CLI since its Python package barely exports anything usable at the top level) —
`diffusers` has a clean, stable, well-documented Python API.

**Known limitations:**
- English prompts only, by design — the tool description tells the model to always write the
  image prompt in English regardless of what language the user asked in.
- CLIP's hard 77-token limit truncates very long/detailed prompts silently (diffusers logs a
  warning server-side, but nothing surfaces this to the user yet) — keep prompts reasonably
  focused; a truncated prompt still generates, just without whatever fell past the cutoff.
- Not coordinated with the LIGHT/DEEP model-mode system — generating an image while a large LLM is
  also loaded is real, simultaneous memory pressure on a 16GB machine.
- No safety checker (neither checkpoint ships with one by default) — diffusers prints a warning
  about this; be mindful of what you generate and who might see it, same as any local image model.

### Chats, memory, photos and documents (Phase 3, 2026-09-27)

- **Chats & memory (💬 in the header).**
  - **Chats tab:** every conversation, most recently active first, titled by your first message and searchable
    (`GET /api/conversations?q=`). Tap one to reopen it; ✕ deletes it (its images and videos stay in the
    gallery).
  - **Memory tab:** everything Zira remembers. Tap a memory to change it, set its importance (★1-5), ✕ to
    forget it, or **+ Add a memory** (`POST`/`PATCH /api/memories`).
  - On the phone, ☾/☀ moved to Settings → Appearance to make room in the header.
- **Ask about a photo.** Attach a photo (📎) and ask: "what is in this picture?", "read the text", "what is she
  wearing?".
  - How it works: `look_at_image` (`app/tools/look.py`) asks an Ollama vision model, `VISION_MODEL`, default
    `minicpm-v`.
  - Memory: the chat model is unloaded first, the vision model answers with `keep_alive: 0`, and the chat model
    is loaded again to write the reply. Measured: about 12 s in all.
- **Ask about a document.** Attach a PDF, Word, RTF, HTML, text or Markdown file (📎) and ask about it, or ask
  for a summary.
  - The text is extracted once (`pypdf`, macOS `textutil`) into ~1,200-character parts with page numbers
    (`app/tools/documents.py`, `POST /api/documents/upload`).
  - `read_document` gives the model the parts that answer the question: a BM25 ranking, or a spread from start
    to end for summaries. At most ~6,000 characters, with `[page N]` labels.
- **NEWLIGHT model mode.** A fourth mode button, added next to LIGHT so the two can be compared before one is
  removed: Qwen3.5 4B heretic (`mradermacher/Qwen3.5-4B-heretic-GGUF`, Q4_K_M).
  - It is registered in Ollama as `qwen3.5-heretic:4b` with Ollama's built-in Qwen3.5 renderer and parser
    (`RENDERER qwen3.5` / `PARSER qwen3.5`, as in Ollama's own `qwen3.5`). It runs like LIGHT: no thinking,
    short voice replies.
  - `NEWLIGHT_MODEL` overrides it, and `run.sh` knows the mode.
- **Model check (`scratchpad llm_bench.py`, 2026-09-27).** 11 real Zira requests through the real prompt and
  tools:

  | Model (mode) | Passed | Median |
  |---|---|---|
  | `qwen3-heretic:4b` (LIGHT) | 10/11 | 2.8 s |
  | `qwen3-heretic:8b` (BALANCED) | 10/11 | 4.1 s |
  | `qwen3:8b` (DEEP) | 11/11 | 10.1 s |

  All three chose the right tools and IDs. The two misses were short, unimproved video prompts.

### Gallery, animate a photo, make a video longer, phone notifications (Phase 2, 2026-09-27)

- **Gallery (▦ in the header).** Every image and video Zira made, newest first, with thumbnails. Tap one to
  see the video or image, what you asked, the prompt the model wrote, the model, size, length and date.
  - Buttons: **Make again**, **Edit request**, **Edit this image**, **Animate** (images), **Make longer**
    (videos), **Download** and **Delete**.
  - It is backed by the `media` table (`app/memory/media_store.py`, `app/api/media.py`). The image and video
    tools record their own files, so a video is recorded even if the phone had disconnected.
  - Files made before the gallery existed were added once, with their requests found in the chat history.
- **Animate a photo.** Attach a photo (📎, or **Animate** in the gallery) and say what should happen. A tall
  photo gives a portrait video. `create_video` gets `image_id`, and the first frame of the video is the photo.
- **Make a video longer.** **Make longer** in the gallery puts `[Video: name.mp4]` in the message. The
  video's frames are copied at the model's frame rate, the new part continues from its last frame, and the
  result is saved as `<name>-longer.mp4`; the original is kept. `create_video` gets `continue_video`.
  - Both this and animating need **FastMetal 5B** or **LTX**; FastMetal 1.3B can only start from text.
  - Both also work when the model forgets to pass the id: the `[Uploaded image: …]` or `[Video: …]` tag in
    your message is used.
- **Phone notifications (Web Push).** Your phone gets a notification when a video is ready, or was not made,
  even with Zira closed. It is not sent when you stopped the video yourself.
  - On an iPhone (iOS 16.4+): in Safari, tap Share, then **Add to Home Screen**. Open Zira from that icon,
    then tap **🔔 Enable notifications** (in Settings on the phone) and allow them. **Send a test** checks it.
  - How it works: `app/push.py`, `app/api/push.py`, `frontend/sw.js`, `frontend/manifest.json` and the
    `push_subscriptions` table. The key pair is made on first use as `data/vapid_private.pem` (0600).
  - Pushes go from the Mac through the browser's push service (Apple's for an iPhone), encrypted and outbound
    only. The sender contact is Zira's own https address (`PUSH_SUBJECT` overrides it), never an email.
  - The library is `pywebpush` (in `requirements.txt`).

### Video generation (local)

**Current state (2026-09-27).** The video models are **LTX 2B**, **FastMetal 1.3B** and **FastMetal 5B**. Wan 2.1
VACE 1.3B was removed at the user's request, so the Wan-specific notes further down are history. Its old `VIDEO_*`
settings in a `.env` are ignored.
- **FastMetal 5B** (FastVideo's FastMetal-5B-QAD, Wan 2.2 TI2V 5B, `third_party/fastmetal/worker5b.py`) makes 5 s
  clips at 24 fps, and each clip continues from the previous clip's last frame, so videos go up to `VIDEO_MAX_SECONDS`.
  FastVideo's Mac version has no image-to-video, so the worker adds it with Wan 2.2's own recipe: the frame is
  encoded by the model's VAE and pinned as latent frame 0 with timestep 0. Measured: 10 s at 320x576 in about
  3.5 min, peak 7.1 GB of GPU memory, no visible jump at the join.
- **FastMetal 1.3B** makes one clip of up to 5 s.
- **Size:** set by the 320P/480P and LANDSCAPE/PORTRAIT toggles.
- **LLM out of the way:** while a video is being made, the chat model is unloaded from Ollama to free memory, and it
  is loaded again afterwards (`LLMMemoryReleaser`, app/ai/model_manager.py). This is best effort: an Ollama problem
  never stops the video.
- **Stop:** the Send button turns into **Stop** while a video is being made, and typing `stop` (or `cancel`) does the
  same (`POST /api/videos/cancel`). A FastMetal worker is ended mid-step, and the next video starts a fresh one.
  LTX stops at its next step. The half-made file is discarded, and the LLM comes back. Images work the same way
  (`POST /api/images/cancel`): create_image/edit_image stop at their next step, nothing is saved, and the image model
  stays loaded for the next one. The stop words also cover `ruk jao`, `ruko`, `band karo`, `bas` and their Devanagari
  spellings. While Zira waits on an image or video, pressing the mic records one clip that can only stop it: saying
  "stop" stops it, and anything else is shown and ignored.
- **Phone disconnects:** a running image or video job, with its prompt and progress, shows again after the page
  reconnects or reloads (`GET /api/conversations/{id}/pending`). A failure is saved to the chat instead of
  vanishing.


`create_video` makes a short, silent clip from a text prompt, fully locally (`VIDEO_GENERATION_ENABLED=true`).
Like the image model it starts on **None** (nothing loaded), loads when you pick **Wan 2.1 1.3B** in the UI
(NO VIDEO / WAN 1.3B buttons next to the image ones, and a "Video generation" card in the phone menu), stays loaded
for every following video, and is unloaded again by picking None. It shares the image system's generation lock
(two MPS generations at once crash the process) and reuses the same "None / load on select / unload" pattern.
The result plays inline in the chat (a `<video>` with controls, iPhone-safe) and is also a download link.

**What it actually uses** (all checked, not assumed):
- Checkpoint: [`Wan-AI/Wan2.1-T2V-1.3B-Diffusers`](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) for the transformer,
  VAE, tokenizer and scheduler config, plus the UMT5-XXL text encoder from
  [`Comfy-Org/Wan_2.1_ComfyUI_repackaged`](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged)
  (`umt5_xxl_fp16.safetensors`, 11.4GB, safetensors, HF-compatible key names). Why not the official text encoder: that
  repo stores it in fp32 (22.7GB; 28.9GB in total), which does not fit the disk or the RAM.
- Library: `diffusers` 0.40.0 `WanPipeline` / `WanTransformer3DModel` / `AutoencoderKLWan`, `UniPCMultistepScheduler`
  (flow shift 3.0), `transformers` 5.17 `UMT5EncoderModel`, PyTorch 2.14 on **MPS**. No CUDA-only features are used.
- Precision: transformer **bf16** (Wan's training precision; fp16 measured the same speed here, 19.8s vs 19.9s per step,
  and also finite), VAE fp32, text encoder computes in bf16. The clip is written as H.264 (yuv420p, faststart) by
  streaming frames into ffmpeg one at a time.
- Memory strategy: the resident pipeline is only the transformer + VAE (~5.4GB footprint). The text encoder cannot stay
  loaded (11.4GB), so it is **streamed block by block** for each prompt (each transformer block's weights are read from
  the file just before it runs and freed right after) and then released: same embeddings as loading it whole (verified,
  max difference 0), but peak 9.3GB instead of 17GB and ~21-34s instead of ~51s. The negative-prompt embedding is
  computed once per process. The **VAE decode runs on the CPU**: measured for real, its 3D convolutions on MPS make the
  Metal driver allocate 17-19GB of its own buffers and hit the process memory cap even for a 17-frame clip (only very
  small tiles squeaked through); on the CPU it needs ~4GB and no GPU memory. Denoising stays on the GPU. Tiling is on.
  After each video: garbage collection plus an MPS synchronize/cache flush. None releases every reference.
- Not used: `enable_model_cpu_offload`/sequential offload (unified memory means "CPU" and "GPU" are the same RAM, so
  moving weights does not free anything; not measured), attention slicing (the pipeline already uses PyTorch's fused
  SDPA), `torch.compile` (unreliable on MPS).

**Two video models** (NO VIDEO / WAN 1.3B / LTX 2B): **LTX-Video 2B distilled** is ~20x faster per second of video
(measured: a real 8-second video in 4 min 14 s; ~15-20 minutes for 30s), with moderate detail; **Wan 2.1 VACE 1.3B**
has better detail but takes ~1 h 45 min for 8s. Both make long videos the same way (continued pieces) and only the
selected one is loaded. LTX details: README section above and CAPABILITIES.md "26".

**Video size** (the 320P / 480P and LANDSCAPE / PORTRAIT toggles, video only): applies to every video model and
to the next video, with no model reload. It is saved to `.env` as `VIDEO_RESOLUTION` (320p = 576x320, 480p = 832x480)
and `VIDEO_ORIENTATION` (portrait swaps width and height, e.g. 480x832), like `IMAGE_RESOLUTION`. Both sizes fit
every model's size rules. 480p is ~2.2x the pixels and correspondingly slower. It is measured for FastMetal only; Wan
and LTX have not been run at 480p on this Mac for a whole video.

**FastMetal 1.3B** (the FASTMETAL button): [FastMetal-QAD 1.3B](https://huggingface.co/FastVideo/FastMetal-1.3B-QAD)
from FastVideo / Hao AI Lab, which is Wan 2.1 1.3B distilled to 3 DMD steps with an INT8 DiT on Apple's MLX. It is the
fastest option. **Measured on this 16GB M4 Air:** a 3s clip (45 frames, 832x480) takes about 5 min 20 s in total
(~92s per denoising step, frame-by-frame decode 5s), with a peak of 2.8GB of GPU memory. FastVideo's own figure,
~110s for 81 frames on an M4 Max, does not carry over: this GPU has a quarter of the cores.
- The decode is TAEHV frame by frame (PyTorch, MPS), not FastVideo's all-frames-at-once MLX decoder. Measured
  here, the parallel decode ran the GPU out of memory (`kIOGPUCommandBufferCallbackErrorOutOfMemory`) even for 45
  frames, after all three steps had finished. It is **text-to-video only**, so a video is one clip of at most `VIDEO_FASTMETAL_FRAMES=81` frames
(~5s at 16 fps). A longer request gets that one clip, and the reply says it was capped.
- It runs in a worker process (`third_party/fastmetal/worker.py`) from its own environment, `.venv-fastmetal`, because
  FastVideo pins torch 2.12, transformers 5.x and MLX, which do not fit Zira's `.venv`. The worker starts when
  FastMetal is selected and ends on None or a switch, which frees all of its memory. Its log is `logs/fastmetal.log`.
- The prompt is encoded by Zira with Wan's streamed UMT5 (same file, ~0.4GB peak) and passed to the worker, so the
  repo's 11GB text encoder is never downloaded or loaded.
- Setup (FastVideo is pinned at commit `e90be598`):
  ```bash
  git clone --depth 1 https://github.com/hao-ai-lab/FastVideo third_party/fastvideo
  /opt/homebrew/bin/python3.11 -m venv .venv-fastmetal
  .venv-fastmetal/bin/pip install -e 'third_party/fastvideo[mlx]'
  ```
  The weights (~2GB, without `text_encoder/*.safetensors`) download on first use. The TAEHV decoder
  (`~/.cache/fastvideo/taehv/taew2_1.pth`) is fetched the first time a clip is decoded.

**Length** (updated): say how long ("30 sec", "1 minute") or get **8 seconds** by default, never more than **30**
(`VIDEO_DEFAULT_SECONDS`, `VIDEO_MAX_SECONDS`). Longer videos are made of pieces of `VIDEO_FRAMES=49` frames (~3s)
with Wan 2.1 VACE 1.3B, each continuing from the previous piece's last `VIDEO_OVERLAP_FRAMES=9` frames, streamed into
one MP4: 8s = 3 pieces (~1 h 45 min), 30s = 12 pieces (~7 hours), measured below. If a later piece fails, the finished
part is kept. **Measured with VACE at 576x320, 49-frame pieces:** ~112s per denoising step (the continuation layers make it
slower than the text-only model's 67s), conditioning encode on the CPU ~110s per piece (the VAE ran the Metal driver out of
memory on the GPU even at 17 frames; an empty video's encoding is computed once and reused), decode on the CPU ~317s -
about 35-37 minutes per piece at 16 steps. 12 steps is ~25% faster; 448x256 is ~40% faster.
Other defaults (configurable: `VIDEO_WIDTH`, `VIDEO_HEIGHT`, `VIDEO_FPS`, `VIDEO_STEPS`, `VIDEO_GUIDANCE_SCALE`):
576x320 at 16 fps, 16 steps, guidance 5.0 (Wan's own reference is 50 steps). The measurements below were taken with
the earlier text-only model and 33-frame clips.

**Measured on this M4 Air 16GB** (through the live app, two videos in a row, `qwen3:8b` resident in Ollama):

| | Video 1 | Video 2 |
|---|---|---|
| Model load (once) | 13-17s | not reloaded |
| Prompt encode | 70s (first, incl. negative prompt) | 34s |
| Denoise, 16 steps | 674s (42s/step) | 664s |
| CPU VAE decode | 219s | 222s |
| Total | **964s (16 min)** | **921s** |
| Peak process memory | 13.0GB | 13.0GB |

Resident with Wan loaded: 5.4GB idle, 8.7-9.2GB after a clip (CPU allocator cache), 1.6GB after selecting None.
Per-step cost by size (2 steps each, includes classifier-free guidance): 320x576: 17 frames 20s, 33 frames 41s, 49 frames
67s; 480x832: 17 frames 52s, 33 frames 117s. Both precisions gave finite output.

**Limits on a 16GB M4 Air, honestly:** the constraint is compute, not memory. The requested 3-5 second, 480p clip is not
practical: 832x480 at 33 frames is ~117s per step (about 30-40 minutes of denoising plus ~10 more of CPU decoding for two
seconds of video), and 81 frames would be several hours. Clips are silent, 16 fps, text-to-video only. At 16 steps
quality is modest (saturated colours, simple motion). A phone that locks its screen mid-render drops the connection,
but the finished clip is still saved to the conversation. Every request still reads the 11.4GB text encoder from disk
(21-70s). The first use downloads about 17.6GB.

### Image style (NONE/REALISTIC/LIGHTNING/REALVIS 5)

**2026-09-28:** FLUX.2 Klein 4B was removed at the user's request and replaced by **REALVIS 5**
([`SG161222/RealVisXL_V5.0`](https://huggingface.co/SG161222/RealVisXL_V5.0), the full non-Lightning SDXL model,
~6.9GB fp16). It loads with DPM++ 2M Karras (its repo ships DDIM; the model card recommends DPM++ Karras), runs 25
steps at CFG 5.0 with the card's quality negative prompt, and, being SDXL, works with `edit_image`. The FLUX notes
below are kept as history. All three styles also take **IP-Adapter FaceID**: attach a photo and ask for a new
picture of that person ("make me an astronaut"), and `create_image` gets `face_image_id`. The face's ArcFace
identity (InsightFace buffalo_l's recognition model via onnxruntime, face found and aligned with OpenCV YuNet)
steers `h94/IP-Adapter-FaceID`'s SDXL adapter, which is loaded only for that image and unloaded right after
(app/tools/face_id.py). Both models are for non-commercial use.

Three selectable checkpoints for `create_image`, switchable at runtime from buttons next to the 📎
upload button, with **only one loaded at a time** (same one-model-resident spirit as LIGHT/DEEP
mode, for the same 16GB-RAM reason). A fourth choice, **NONE**, is the startup state:

**NONE** (`IMAGE_STYLE=none`, the default) means no image model is selected or loaded, so its memory
(6-13GB while a model is resident) stays free for the LLM and macOS. Nothing is loaded at startup. Picking a
model unloads whatever was loaded (never two at once) and loads the chosen one right away, and it then stays
loaded for every following image - a request never loads and unloads it again. Picking NONE unloads it (the
weights are released, then garbage collection and an MPS cache flush) without touching the server, the LLM,
uploads/editing, resolution or the safety check. Asking for an image while NONE is selected returns a clear
"pick a model" message instead of silently choosing one. If a model fails to load, nothing half-loaded is kept,
no other model is substituted, and the selector goes back to NONE with the error shown. The selection is
runtime-only (not saved to `.env`), so a restart always starts on NONE.

**REALISTIC** ([`Perfection Realistic ILXL`](https://huggingface.co/John6666/perfection-realistic-ilxl-illustrious-xl-nsfw-sfw-checkpoint-42-sdxl),
`IMAGE_STYLE=realistic`) is an Illustrious XL fine-tune (SDXL architecture) by 6tZ, loaded from
John6666's diffusers-format conversion of version 4.2. It replaced RealVisXL V3.0 Turbo. Checked
against the real repo: public/ungated, safetensors only, ~6.9GB, fp16 weights stored as the plain
default files (so it loads with `variant=None`, unlike Lightning). Unlike the two few-step
models it is a normal, non-distilled SDXL model, so it needs real steps and classifier-free
guidance: defaults are `IMAGE_STYLE_REALISTIC_STEPS=24` and `IMAGE_STYLE_REALISTIC_GUIDANCE=5.0`
with the repo's Euler Ancestral scheduler. The model card tags it not-for-all-audiences and the
licence is `faipl-1.0-sd`; the always-on minor-safety check in `create_image` applies to it like
any other style. To use a different version, set `IMAGE_STYLE_REALISTIC_MODEL` (any diffusers-format
SDXL repo whose weights are plain, non-variant files).

**LIGHTNING** ([`RealVisXL V4.0 Lightning`](https://huggingface.co/SG161222/RealVisXL_V4.0_Lightning),
`IMAGE_STYLE=lightning`) is an SDXL photorealism model, distilled for few-step generation. Checked
against the real repo: public/ungated, diffusers format with fp16 variant files, ~6.9GB. The model card recommends 4+ steps and CFG
1.0-2.0 with a DPM++ SDE sampler; defaults here are `IMAGE_STYLE_LIGHTNING_STEPS=6` and
`IMAGE_STYLE_LIGHTNING_GUIDANCE=1.0` (guidance above 1 makes SDXL run classifier-free guidance,
roughly doubling every step's cost). **Measured for real on this Mac at 720x720:** 6 steps in 31s
for a hummingbird, and a clean, natural-looking face on a portrait prompt (even eyes, real skin
texture, believable teeth and light) - faces being the reason the image models were replaced in the
first place. The repo's own default scheduler was used, not the card's DPM++ SDE, since the
result was already good. `edit_image` works with LIGHTNING as well as REALISTIC (both are SDXL, so
the same img2img pipeline serves either); only FLUX can't edit.

| Style | Model | For |
|---|---|---|
| **REALISTIC** (default) | [`Perfection Realistic ILXL`](https://huggingface.co/John6666/perfection-realistic-ilxl-illustrious-xl-nsfw-sfw-checkpoint-42-sdxl) (SDXL / Illustrious XL, ~6.9GB fp16) | photorealism from a full-step SDXL model; slower than LIGHTNING (24 steps with CFG), see the measured numbers below |
| **REALVIS 5** | [`RealVisXL V5.0`](https://huggingface.co/SG161222/RealVisXL_V5.0) (SDXL, ~6.9GB fp16) | photorealism from SG161222's newest full model; 25 steps with CFG, so about as slow as REALISTIC |
| ~~FLUX~~ (removed 2026-09-28) | [`FLUX.2 Klein 4B`](https://huggingface.co/Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic) (Black Forest Labs, SDNQ 4-bit quantized, ~5.1GB) | general-purpose, genuinely high quality - real side-by-side test produced more fine detail than REALISTIC on the same prompt, but real-measured **~3-5x slower per image** on this hardware (see below) |

Both replace the old single `stabilityai/sd-turbo` default entirely — sd-turbo's own 1-step
speed-over-fidelity tradeoff is exactly the "can't make faces" problem this feature exists to fix.
`create_image` behaves identically whichever style is active; only the checkpoint underneath changes.
`edit_image` works with the two SDXL styles (REALISTIC, LIGHTNING) but **not** FLUX - see "Editing an
uploaded image" below for why.

**Steps/guidance_scale are real per-model hyperparameters, not interchangeable — a real bug this
caught in its own first live use**: sd-turbo's design (true single-step, CFG-free) is unusual, not
the norm. Reusing its `steps=1`/`guidance_scale=0.0` defaults unconditionally for the new
checkpoints produced pure unconverged noise on a live request (no dog, no recognizable structure at
all) the first time someone actually generated something other than a face with REALISTIC active.
Root-caused via the server log, then fixed properly rather than patched around: `steps`/
`guidance_scale` are now per-style (`IMAGE_STYLE_REALISTIC_STEPS=24`/`IMAGE_STYLE_REALISTIC_GUIDANCE=5.0`
for Perfection Realistic - a full-step model, confirmed by a real 1024x1024 generation, see below;
`IMAGE_STYLE_LIGHTNING_STEPS=6`/`GUIDANCE=1.0`; `IMAGE_STYLE_FLUX_STEPS=4`/`IMAGE_STYLE_FLUX_GUIDANCE=1.0`,
also confirmed via a real direct generation - `guidance<=1` keeps FLUX's own classifier-free guidance
off, verified against the real `Flux2KleinPipeline` source, not assumed), resolved dynamically from
whichever style is *currently* active rather than frozen at server startup — so a runtime style
switch changes generation behavior correctly too, not just which model file is loaded.

**All three styles, measured for real at 1024x1024** (same prompt through the live server, one
image each, style switched from the UI's own endpoint; M4 MacBook Air 16GB with `qwen3:8b` resident in
Ollama and swap already 14-20GB in use, i.e. a busy machine, not a clean benchmark):

| Style | Model | Steps | Generation time | Seconds/step | Peak server memory* |
|---|---|---|---|---|---|
| REALISTIC | Perfection Realistic ILXL | 24 (CFG 5.0) | 396s (about 6.6 min) | 16.5 avg (about 9.3 once fully paged in) | 14.0GB |
| LIGHTNING | RealVisXL 4.0 Lightning | 6 (CFG 1.0) | 73s | 12.2 | 17.0GB |
| FLUX | FLUX.2 Klein 4B (SDNQ 4-bit) | 4 (CFG off) | 150s | 37.4 | 13.0GB |

\* peak `footprint` of the server process, sampled every 2s; on macOS this counts memory the system has
compressed or swapped out, so it can exceed 16GB (LIGHTNING's figure was taken right after switching from
REALISTIC, so it likely still includes some of the previous model's memory). Times are the pipeline's own
denoising time; each first generation also paid a one-off model load (about 25-90s more) and the LLM's own
~25s to decide to call the tool. Perfection Realistic's first steps were far slower (about 30s/step) while
its weights paged in, then settled at about 9s/step. At the default 720x720 (about half the pixels) expect
roughly half these times. Lower `IMAGE_STYLE_REALISTIC_STEPS` (for example 20) to trade a little quality for speed.

**FLUX.2 Klein's real, measured cost on this hardware** — genuinely worth knowing before switching
to it for everything: a real side-by-side test (the same dog-drinking-water prompt used elsewhere in
this doc) measured **~41s/step** for FLUX vs the SDXL styles' **~8-16s/step** - roughly 3-5x slower per
image, confirmed *not* caused by classifier-free guidance (checked directly against
`Flux2KleinPipeline.do_classifier_free_guidance` - it was correctly off). The real cause: SDNQ's
quantized-matmul speedup requires Triton, which the docs confirm is only available on CUDA/ROCm/XPU,
not MPS - so on this hardware it dequantizes every step without the compensating benefit, on top of
being a genuinely bigger model (4B parameters vs SDXL's ~2.6B UNet). A full first-use request (model
download + load + one generation) measured **943.9s** total, which is why
`IMAGE_GENERATION_TIMEOUT` was raised to 1200s.

**A third style, ANIME (Pony Diffusion V6 XL), was built, shipped, then removed** after the user
reported it "not working good" - consistent with this session's own earlier real finding: Pony's
baked-in VAE intermittently produced completely incoherent images on some prompts (a documented SDXL
fp16 VAE overflow issue, seed/latent-dependent - a same-prompt retry would often work fine, which is
exactly what made it easy to miss in a single test but real in production use). FLUX.2 Klein replaces
it, but not as a same-kind-of-thing swap - it's a general-purpose photorealistic model, not an
anime/illustration specialist, so the REALISTIC/ANIME *stylistic* dichotomy this feature originally
shipped with no longer strictly applies; the two styles now differ in underlying model/quality/speed
tradeoffs rather than art style.

**Why FLUX.2 Klein 4B specifically, not the full model** — checked for real, not assumed: full bf16
FLUX.2 Klein needs ~13GB, confirmed too much alongside any LLM Ollama might be holding resident on a
16GB machine. FP8 quantization (the obvious way to shrink it) does not run on Apple's MPS backend at
all (confirmed: no FP8 tensor-core equivalent to NVIDIA Ada+). This build uses
[SDNQ](https://github.com/Disty0/sdnq) 4-bit quantization instead - real, natively supported by
`diffusers>=0.40.0` per its own official docs, confirmed to actually run and generate correctly on
MPS with a real test, not just imported successfully. Needs `pip install sdnq` (a lightweight
quantization backend, not a second inference framework - `mflux`, the MLX-native alternative, was
deliberately not used here; it already caused a real, reproduced bug earlier this session with a
different model, see "Why `diffusers` + `stabilityai/sd-turbo`" below).

### Resolution (720/1024)

**Resolution was never actually configured — everything ran at diffusers' own default.** Neither
`create_image` nor `edit_image` ever passed `height`/`width` to the pipeline, so every image
generated all session (including every real test in this README/CAPABILITIES.md before this) ran at
SDXL's default 1024×1024 (confirmed via the real diffusers source: `height = height or self.
default_sample_size * self.vae_scale_factor`, docstring: *"set to 1024 by default for the best
results"*) — the single largest cost in generation time, since UNet/VAE compute scales with pixel
count.

`IMAGE_RESOLUTION` (default `720`, square) now controls this explicitly — a button next to the
REALISTIC/FLUX toggle, switchable at runtime with **no restart and no model reload** (`POST
/api/images/resolution`): unlike style, resolution isn't tied to a checkpoint at all, so switching it
is just updating a number read at the start of the next generation. Independent of style, unlike
steps/guidance — one shared value, not a per-style default, since the point is a quick "pick before
your next request" control.

Both choices are valid SDXL sizes — confirmed via the real `StableDiffusionXLPipeline.check_inputs`
source that the only hard requirement is divisibility by 8 (`720=8×90`, `1024=8×128`), not the
64-multiple convention some guides suggest.

**Real measured result**: 62.5s (720) vs 125.8s (1024) for the same REALISTIC prompt (the dog photo)
— almost exactly 2× faster (720² is ~49% of 1024²'s pixels), with no meaningful visible quality loss
on direct comparison.

**A third option above 1024 was built, really tested, and dropped — not assumed safe, not left as an
untested caveat either.** 2048 (2× this model family's native training resolution per dimension, 4×
the pixels of 1024) was requested, built, and tested with a real prompt ("a simple red apple on a
white table"):
1. First attempt **crashed outright**: `Invalid buffer size: 16.00 GiB` at the VAE decode step — a
   real MPS allocation limit, not a timeout.
2. Fixed the crash with `pipe.vae.enable_tiling()` (a real, documented `diffusers`/`AutoencoderKL`
   feature for decoding large images in patches instead of one giant buffer) — generation then
   completed, but produced **two small, wrongly-scaled apples floating in a mostly-empty frame**
   instead of one apple properly filling it. A real, reproduced duplication artifact, not a rendering
   glitch — the well-documented result of pushing a base SDXL checkpoint past its native resolution
   without a dedicated hires-fix/upscale pass (generate small, then a second img2img upscale
   pass), which a single `height`/`width` parameter cannot provide.
3. Tried a smaller step in between, 1536 (1.5× native) rather than assuming the problem was 2048-
   specific: **same failure pattern** — a real out-of-memory error first (`MPS backend out of memory
   ... max allowed: 20.13 GiB`, missed by 72MiB), then, with tiling, the same two-small-subjects
   duplication artifact, just as broken as 2048.

Conclusion: there is no usable size above 1024 for this feature as built — the model family's own
composition/scaling behavior breaks down as soon as you push meaningfully past its native resolution
this way, not just at 2048. `IMAGE_RESOLUTION` only accepts `720`/`1024`; a genuine higher-resolution
option would need a real hires-fix/upscale pipeline, a separate, bigger feature not built here.

**Switching is instant, no restart** — deliberately different from LIGHT/DEEP mode. The Ollama LLM
lives in a *separate* server process, so swapping it safely needs a verified unload + full process
restart (see "Model mode" above). `ImagePipelines` lives inside this same JARVIS process, so a
switch just drops the loaded pipeline and clears PyTorch's MPS cache — confirmed for real that this
actually releases the memory, not just assumed: a real 8GB MPS allocation, checked via
`torch.mps.current_allocated_memory()`, measured at exactly 0 after `del` + `gc.collect()` +
`torch.mps.empty_cache()`. The next generation after a switch lazily loads the new checkpoint,
including a real multi-GB download the first time each style is ever used.

**Why these two specifically** (both checked for real before choosing, not assumed from name
recognition):
- The SDXL checkpoints ship proper `diffusers` repos (`model_index.json`, safetensors) - confirmed
  via the real Hugging Face file listings - so they load through the exact same
  `AutoPipelineForText2Image.from_pretrained(...)` call sd-turbo already used (Lightning with
  `variant="fp16"`; Perfection Realistic ILXL, which stores fp16 as its default files, with no variant).
  REALISTIC was RealVisXL V3.0 Turbo until it was replaced by Perfection Realistic ILXL.
- **FLUX.2 Klein 4B was ruled out once, then added anyway once the real blocker was resolved.**
  First pass: full bf16 needs ~13GB, confirmed too much alongside any LLM Ollama might be holding
  resident (real measured sizes: 9.6GB `gemma4:e4b`, 5.2GB `qwen3:8b`). FP8 (the obvious shrink)
  doesn't run on MPS at all. The only 4-bit path known at the time needed `mflux`, a separate
  inference framework this project had already hit a real, reproduced bug with once (see below) -
  so FLUX was set aside. Later, a real, `diffusers`-native 4-bit path was found and confirmed
  working: [SDNQ](https://github.com/Disty0/sdnq) quantization, officially supported by
  `diffusers>=0.40.0`, running correctly on MPS with a real test generation, not just imported
  successfully. `AutoPipelineForText2Image` was also confirmed (checked its real pipeline mapping
  directly) to already resolve `Flux2KleinPipeline` on its own, so no special-cased loader was
  needed - same loading call as RealVisXL, just a different repo/dtype.
- A general-purpose "just use a bigger model" upgrade was set aside in favor of style-specific
  checkpoints because the actual complaint (sd-turbo can't make faces) is a realism problem
  specifically, and RealVisXL was bred for exactly that rather than general-purpose quality.

**Known limitation shared with LIGHT/DEEP mode**: switching styles while a generation is actively
in flight is not guarded against - don't click the toggle mid-generation. In practice this is a
single-user local app with no legitimate reason to do that.

**ANIME (Pony Diffusion V6 XL) was built, shipped, and later removed** - kept here as real history,
not deleted outright, since it explains a real bug this session found and fixed twice on the way to
removing it. A live request produced a completely incoherent image (no recognizable structure, just
colorful static) - re-running the *identical* prompt/steps/guidance immediately after produced a
normal, coherent result, ruling out a deterministic bug in prompt handling or settings (both were
independently confirmed correct first). This matched SDXL's documented fp16 VAE numerical-overflow
issue: certain decoder activations can exceed fp16's ~65504 max depending on the actual latent
values, seed/content-dependent, not something that happens on every call - and exactly why the Pony
V6 XL repo ships its own `sdxl_vae.safetensors` alongside the checkpoint, which this project loaded
as a fix. Despite that real fix working, the user later reported ANIME still "not working good" in
practice and asked for it to be replaced with FLUX.2 Klein instead - the VAE fix narrowed the failure
but evidently didn't fully resolve real-world quality complaints, which is a legitimate, sufficient
reason to drop a whole approach rather than keep patching it further.

### Editing an uploaded image (`edit_image`)

Upload a photo (the 📎 button next to the chat box, shown whenever image generation is enabled) and
ask for a change — no separate setup, it shares REALISTIC's model (not FLUX's - see below). The
frontend uploads on file selection (`POST /api/images/upload`, sandboxed storage under
`UPLOADS_DIR`) and prefixes your next message with `[Uploaded image: <id>]`, which the model is told
to extract and pass to `edit_image`.

Uses `AutoPipelineForImage2Image.from_pipe()` to convert the already-loaded generation pipeline
in-place (confirmed for real: ~0.0s, vs. a full reload) rather than loading a second model.

**Works with REALISTIC and LIGHTNING (both SDXL), but not FLUX - a real, confirmed API
incompatibility, not an arbitrary restriction.** `Flux2KleinPipeline.__call__`'s real signature
(checked directly, not assumed) has no `strength` parameter at all, and takes an `image=` argument
on the *same* call used for text-to-image rather than a separate img2img pipeline the way SDXL does
- a materially different, "unified" API. Rather than guess at a translation or silently fall back
without telling you, `edit_image` checks the active style first and returns a clear error asking you
to switch to REALISTIC or LIGHTNING if FLUX is currently active.

**Real measured tuning**: the first settings tried (`steps=4`, `IMAGE_EDIT_STRENGTH=0.75`) barely
changed the source image at all, even for a completely different prompt. `steps=8` (REALISTIC's
per-style default, see "Image style" above), `IMAGE_EDIT_STRENGTH=0.9` (the actual default) produced
a dramatic, clearly on-prompt result on a realistic photo (apples on a wooden table → blue crystal
gems on the same table, composition kept). A pathologically flat/solid-color source image is a
genuine edge case where even high settings barely move the result — not something to worry about for
real photos.

**Content filtering — two independent layers, by explicit user design:**
1. **Configurable** (`IMAGE_EDIT_CONTENT_FILTER_ENABLED`, **off by default** by explicit request —
   this is a local, single-user tool for editing your own images, and the underlying model already
   has no built-in filter of its own). When on, blocks `edit_image` prompts matching
   `IMAGE_EDIT_BLOCKED_TERMS` (comma-separated, edit the list to taste).
2. **Not configurable, no exception, for anyone, ever**: `app/tools/image_safety.py::check_minor_safety`
   blocks any `create_image` or `edit_image` prompt combining a reference to a minor with
   sexual/explicit content. There is no environment variable, setting, or code path that disables
   this — explicit, repeated user request: *"no exception for minor one."* Both term lists
   (minor-referencing and sexual/explicit) are deliberately broad/over-inclusive: a wrongly-blocked
   edit costs nothing, a missed one would be catastrophic. This is keyword-based, not a trained
   classifier — it catches straightforward phrasing, not every possible evasion; that limitation is
   stated plainly, not hidden.

## Working on your projects: read, plan, edit, Postman

JARVIS can read a code project, explain it, make a plan, generate a Postman collection, and **propose**
code changes. This is **off until you list folders** in `.env`:

```bash
FILE_ACCESS_ROOTS=/Users/you/projects/my-app        # JARVIS may READ these (comma-separated)
FILE_WRITE_ROOTS=/Users/you/projects/my-app         # JARVIS may PROPOSE edits here (optional)
```

Restart JARVIS. A **mode selector** appears in the header:

| Mode | What JARVIS can do |
|---|---|
| **Chat** | Everything normal, plus read your project folders and generate Postman collections. Changes nothing. |
| **Plan** | Read-only. It investigates (reads the files it wants to change) and replies with *Goal / Files to change / Steps / Risks / How to verify*, then shows **Approve plan → Edit mode**. |
| **Edit** | Also lets it **propose** changes. Each proposal appears as a diff card. **Nothing changes until you click Approve.** |

Try (paths must be inside a configured folder):

```
read and understand this project /Users/you/projects/my-app
make a postman collection for all the api in /Users/you/projects/my-app
[Plan]  plan how to add a DELETE endpoint for products
[Edit]  in /Users/you/projects/my-app/app/greeting.py make greet return "Hello, " + name + "!"
```

### How changes are approved

1. In Edit mode the model can only call `propose_edit` (replace exact text that occurs once) or
   `propose_create` (a new file). That stores a **pending diff**; the file on disk is untouched.
2. You see the diff in the chat with **Approve / Reject**. Only your click, a plain request from your browser
   (`POST /api/changes/<id>/approve`), applies it. The model has no tool for that.
3. Approval re-checks that the file is **exactly as it was when proposed**; if you edited it meanwhile, nothing is
   applied and your edit is kept. The write is atomic and keeps file permissions.
4. Any applied change can be **undone** (while the file is still as JARVIS left it).
5. Pending proposals survive a restart, expire after 24 h, and JARVIS is told each proposal's outcome, so it can
   continue a multi-step plan. Config/build/script files (`package.json`, `*.gradle`, `*.sh`, `*.yml`, ...) are
   flagged for extra care.

### What JARVIS understands

- `project_overview`: stack, layout, docs and the **API endpoints it detects**; `code_outline`: classes and
  functions in Kotlin, Java, Python, JS/TS, Go, Rust, Swift, C#, PHP, Ruby; `list_directory`, `read_file`
  (chunked), `search_files`.
- **Postman generation** (`generate_postman_collection`) finds the API in: **Retrofit** (Kotlin/Java, including the
  `@Url` + URL-constant style, wrapper functions and `when`-branch URLs), **Express**, **FastAPI/Flask**,
  **Spring**, **OpenAPI JSON** and **Markdown API docs** tables, and merges them. Request bodies come from the
  data classes / pydantic models; docs supply auth, status and descriptions. It writes a Postman v2.1 file to
  `data/exports/` and shows a download link (`GET /api/exports/<file>`). Endpoints known only from docs get an
  empty body placeholder and are flagged.

### Safety model for project access

- **Opt-in and scoped.** Only files under the folders you list. Paths are resolved with `realpath`, so `..` and
  symlinks that leave the folder are refused.
- **Secrets are never readable**, even inside an allowed folder: `.env*`, keys/keystores (`*.pem`, `*.jks`,
  `*.keystore`), `keystore.properties`, `local.properties`, `gradle.properties`, `google-services.json`, `.mcp.json`,
  `id_rsa*`, `.git`, `.ssh`, `.aws`, and so on. Values that look like credentials in other files are redacted.
- **Read-only unless you approve.** No shell, no delete, no rename, no arbitrary writes. Edits are limited to
  `FILE_WRITE_ROOTS`, never into build/dependency folders.
- **Prompt-injection containment.** File contents are untrusted data. Once JARVIS has read a private file in a
  turn, `web_search` is disabled for the rest of that turn, so a malicious file cannot make it send your code out.
- **Bounded.** At most 6 tool rounds per message; older tool output is dropped to fit the context window.

### Limits (be realistic)

A local 8B model on 8 GB of context is good at *explaining, locating and making small, focused edits*, and the
Postman scanner is deterministic. It is **not** a match for a large hosted coding model on big refactors: it can
misread code, and multi-file changes may need several rounds of review. Expect 30-100 s per tool-using answer.
Raise `OLLAMA_NUM_CTX` (e.g. 12288) if you have the RAM, or try a coding model (`OLLAMA_MODEL=qwen2.5-coder:7b`).
The scanners are regex-based, so unusual code styles can be missed; the summary lists warnings and how many
endpoints came from docs only. Always review diffs before approving.

## Voice (Hinglish V1)

Two ways to talk to JARVIS, both going through the **same** chat pipeline as typed text (same
history, same memory, same tools) — a voice turn is stored and recalled identically to a typed one:

- **Push-to-talk (default):** press and hold the 🎤 button, speak, release.
- **Continuous / hands-free:** tick the "Continuous" checkbox. The mic stays on in a loop; the
  *first* utterance must start with the wake word **"Hello Zira"** (matched loosely - "Hey Zira",
  "Zira, ..." work, and so do the other spellings Whisper produces for the name; see "The wake word:
  Hello Zira" below). Say "Hello Zira" alone and it waits for the actual command next, like a normal
  voice assistant. **You don't have to repeat "Hello Zira" for follow-ups** — once woken, it stays in an
  active-conversation state and treats your next utterances as commands directly, for as long as
  you keep responding within `AWAKE_GRACE_MS` (10s default) of real listening time between them.
  Time JARVIS spends thinking or speaking doesn't count against that 10s — only silence while it's
  actually your turn to talk does. Go quiet for that long and it falls back to requiring "Hello
  JARVIS" again before it'll act on anything. Utterances heard while asleep with no wake word are
  silently discarded.

Off by default; enable it in `.env`:

```bash
STT_PROVIDER=faster-whisper   # local, offline speech-to-text (downloads a model on first use)
TTS_PROVIDER=say              # local, offline text-to-speech (macOS's built-in `say`)
```

Restart JARVIS and hard-refresh the browser tab — the 🎤 button appears next to the composer once
`GET /api/voice/status` reports it's available.

### Voice on/off (a real RAM cost, confirmed and made controllable)

The STT model is a real, sizeable resident load once used — confirmed for real, not assumed:
`mlx-community/whisper-large-v3-turbo` measured at **~1.6GB** active memory
(`mx.get_active_memory()`) after a single transcription. mlx-whisper caches it in a global,
process-wide holder (`mlx_whisper.transcribe.ModelHolder`) with no public unload of its own, so once
you've used voice even once, that 1.6GB stays resident until the whole server restarts — real,
simultaneous pressure alongside whatever LLM/image-style checkpoint is also loaded on a 16GB
machine.

A 🎤 ON/OFF button next to the mic (in the voice bar) controls this directly:
- **Turning it off** calls `POST /api/voice/unload`, which frees the model - confirmed for real: the
  same 1.6GB active-memory reading dropped to ~0 after `ModelHolder.model = None` + `gc.collect()` +
  `mx.clear_cache()` (MLX's own equivalent of `torch.mps.empty_cache()`). The page then reloads, and
  the mic button/voice bar are hidden until you turn it back on.
- **Turning it on** just re-enables the mic — the model lazy-loads again on the next real use, same
  as it already does after a fresh server start. Nothing is eagerly pre-loaded.
- Your choice persists in the browser (`localStorage`), defaulting to on so existing setups see no
  behavior change.

TTS (`say`, macOS's built-in command) is not part of this — it's a subprocess spawned per request,
not a resident model, so it holds no memory to free either way.

### How it works

- **Recording happens in the browser** (`getUserMedia` + `MediaRecorder`), not on the server — no
  microphone library or OS mic permission for the Python process is needed. The recorded clip is
  uploaded to `POST /api/voice/transcribe`, which returns text only.
- That text is sent through the **existing** `/ws/chat` flow exactly as if you had typed it — a voice
  turn shows up in `GET /api/conversations/{id}/messages` and survives a restart identically to a
  typed one. There is no separate voice chatbot or database.
- Once the reply finishes, its text is cleaned for speech (`app/voice/text_cleaning.py` strips
  markdown, code fences, list markers and link/Sources blocks — the stored history is untouched,
  only the copy sent to TTS is cleaned) and sent to `POST /api/voice/speak`, which returns WAV audio
  the browser plays. In continuous mode, JARVIS then automatically starts listening again.
- **Interruption:** pressing/clicking 🎤 again while JARVIS is speaking stops playback immediately
  and starts listening to your new request (client-side; the server request already in flight
  simply finishes and its audio is discarded unplayed).
- **Debug timing:** open the browser console — each turn logs `[voice] mic->transcript`,
  `[voice] chat response`, `[voice] tts`, and `[voice] total turn` timings (`performance.now()`,
  client-side; message content is never included). The server also logs STT/TTS duration per request
  (bytes/char counts only, never the words).
- **Very short/accidental presses are handled gracefully.** A genuine bug was found and fixed here:
  a real ~30ms browser recording (110 bytes) produces an incomplete WebM container that the decoder
  cannot parse, which used to surface as `Transcription failed: [Errno ...] End of file`. It's now
  treated as "no speech" (empty transcript), both server-side (`FasterWhisperSTT` catches the
  specific decode error) and client-side (clips under ~800 bytes are never even uploaded).

### Continuous mode: how the hands-free loop actually works

- **Wake-word matching runs entirely on the existing local Whisper pipeline** — there is no separate
  always-on wake-word engine. Every utterance while listening is fully transcribed, then checked for
  "jarvis" client-side; non-matches are discarded. This means continuous mode is meaningfully more
  CPU/battery-hungry than push-to-talk *while it's turned on*, since the whole faster-whisper model
  runs on every pause in your speech, not just when you press the button.
- **Turn-taking (when to stop listening) uses silence detection** in the browser (Web Audio API
  `AnalyserNode`, RMS amplitude over time — `app.js`, `startSilenceWatch`). The speech-detection
  level is **not** a fixed guessed number — a fixed threshold (0.02) turned out to be too high for
  at least one real microphone and made continuous mode never detect speech at all. It now
  **calibrates to your own mic's ambient noise** for the first ~400ms of every listen
  (`CALIBRATION_MS`), setting the threshold to `ambient × 2.5` (floor `0.004`, so a dead-silent room
  doesn't make it hypersensitive). If nothing crosses that threshold for a while — still possible on
  mics with aggressive automatic gain control, which actively works against amplitude-based
  detection by keeping levels roughly constant whether you're talking or not — the threshold
  **relaxes** (lowers) every 1.5s so it gets progressively more sensitive rather than staying stuck.
  It waits ~1.2s of quiet after it detects you've spoken, or an **8s hard cap** either way, so a bad
  calibration is never more than an 8-second wait, not indefinite silence. A small **live level
  meter** next to the status text shows the current mic level against the threshold (green = above
  it) while listening. **The reliable fallback regardless of any of this:** click the mic button
  while it says LISTENING to force it to stop and process immediately — this always works and does
  not depend on amplitude detection at all. If auto-detection is still wrong for you,
  `console.debug` logs the calibrated/relaxed threshold and live level every second, and
  `SILENCE_MS` / `SPEECH_LEVEL_MULTIPLIER` /
  `MIN_SPEECH_THRESHOLD` / `MAX_LISTEN_MS` are at the top of the voice section in `frontend/app.js`
  to adjust further.
- **A real, local, low-power wake-word engine** (e.g. [openWakeWord](https://github.com/dscripka/openWakeWord),
  which ships a pretrained "hey jarvis" model and needs no torch/GPU — checked: it installs cleanly
  alongside the existing dependencies) would use near-zero CPU while idle instead of continuously
  running Whisper, but requires a small always-on audio-streaming subsystem (the browser would need
  to stream raw PCM to the server continuously, not just upload a clip per utterance) that does not
  exist yet. Worth it if continuous mode's battery/fan-noise cost turns out to be a problem for you;
  not built in this pass since the Whisper-based approach above already delivers the actual hands-free
  behavior using only what was already built and tested.
- JARVIS always replies in Hinglish (Latin script) for any Hindi content, and plain English for pure
  English - never in Devanagari, even if your own message was in Devanagari. This is a
  personality-prompt instruction (`app/ai/prompts.py`), not a voice-layer hack, and it matters
  specifically because of how faster-whisper transcribes spoken Hindi: it renders it in native
  **Devanagari** script by default (there is no romanization option without sacrificing accuracy -
  see `STT_LANGUAGE` below), so without this rule, speaking Hindi to JARVIS would get you a
  Devanagari transcript *and* a Devanagari reply - which sounds far worse read aloud by the default
  voice (see below) than Hinglish does. An earlier, simpler version of this rule ("mirror the user's
  script") was tried and measured against the real model: it reliably produced pure-Devanagari
  replies for Devanagari input, and inconsistently mixed Devanagari and Latin script mid-reply for
  Hinglish input. The current wording - an explicit "never write a single Devanagari character,
  even one word" instruction plus three worked examples (Devanagari input, Hinglish input, and pure
  English input, each with a fully-correct-script example reply) - was needed to get consistent
  behavior; the abstract rule alone was not enough on this model. Verified with 3 repeated real
  runs each of Devanagari, Hinglish and pure-English input against the real model: all 3/3 correct
  with no script mixing.
- **Voice transcript cleanup** (`app/voice/transcript_cleanup.py`, `VOICE_TRANSCRIPT_CLEANUP_ENABLED`,
  on by default): raw STT output is often a disfluent run-on with small words dropped ("i wanted
  play game" instead of "i wanted to play a game") or an occasional mis-heard word. An extra LLM
  call fixes exactly that before the transcript becomes your chat message - restoring dropped small
  words, fixing sentence breaks, correcting clear mis-transcriptions - and nothing else. It never
  translates, never changes script (Hinglish stays Hinglish, Devanagari stays Devanagari), never
  formalizes casual phrasing into "proper" English/Hindi, and never invents content; when it isn't
  confident something is a transcription mistake, the prompt tells it to leave it unchanged. A
  leading wake-word address ("Jarvis"/"Hello Jarvis") is explicitly preserved so continuous mode's
  wake-word detection (`extractCommand()` in `app.js`) keeps working. Any failure, timeout, or
  suspicious-looking output (wildly longer or shorter than the input) falls back to the raw
  transcript unchanged - cleanup must never break voice. Measured added latency on this M4 setup:
  ~2-2.7s per clip, on top of faster-whisper's own ~4-5s (`small` model) - noticeable but accepted
  as a tradeoff for cleaner input. Set `VOICE_TRANSCRIPT_CLEANUP_ENABLED=false` to turn it off.

### Speech-to-text: mlx-whisper on Apple Silicon, faster-whisper elsewhere (local, offline after first use)

Two interchangeable `SpeechToText` providers (`app/voice/speech_to_text.py`), switched by
`STT_PROVIDER`, both Whisper under the hood - same accuracy/language behavior, different inference
backend:

- **`mlx-whisper` (default, recommended on M-series Macs)**: uses Apple's MLX framework, which runs
  on the GPU via Metal. `STT_MODEL` is a Hugging Face repo of MLX-converted weights (default:
  `mlx-community/whisper-large-v3-turbo`).
- **`faster-whisper`**: CTranslate2, CPU-only (no Metal backend exists for CTranslate2 on Mac). Use
  this on Intel Macs, or as a documented fallback if `mlx-whisper` doesn't work for you.
  `STT_MODEL` is `tiny`/`base`/`small`/`medium`/`large-v3`/`large-v3-turbo`.

**Measured for real on a MacBook Air M4, identical model (`large-v3-turbo`), identical clip:**

| Backend | Per-clip inference | Why |
|---|---|---|
| `faster-whisper` | ~23 s | CTranslate2 has no Metal backend on Mac - this runs the full large-v3-turbo model on CPU |
| `mlx-whisper` | **~4-5 s** | MLX runs on the M-series GPU via Metal - same weights, same accuracy, ~5x faster |

This is why `mlx-whisper` is the default: it gets large-v3-turbo's accuracy at roughly `small`
model speed, removing the accuracy-vs-latency tradeoff that used to exist between Whisper sizes on
this hardware. The **first** transcription with either provider downloads the model from Hugging
Face (cached under `~/.cache/huggingface/` afterwards - every call after that is fully offline).

`mlx-whisper` pulls in `torch` as a transitive dependency (used by its audio/tokenizer utilities,
not for the actual inference, which stays on MLX/Metal) - a real, if slightly unfortunate, ~130 MB
part of its footprint, not something this project chose.

Real Hinglish benchmark (`scripts/voice_benchmark.py`, run against the live server - synthesizes
each test sentence with `say`, so no microphone needed):

```bash
./run.sh                                    # in one terminal
.venv/bin/python scripts/voice_benchmark.py  # in another
```

Real results (mlx-whisper, large-v3-turbo, this M4): **~5.0s average per clip** (includes the
transcript-cleanup LLM call, on by default - see below), consistently faithful code-switching
preservation ("Mujhe ek Android app banana hai using Kotlin" → "मुझे एक Android ऐप बनाना है using
Kotlin" - proper nouns/tech terms correctly stay in Latin script, everything else in Devanagari,
never translated to pure English). Two real, honest misses worth knowing about, not hidden:
"parso" (day-after-tomorrow) was consistently misheard as "par sona" (but, to sleep) across repeat
tests - a genuinely hard, less common word; and "calendar" once came back as "kalandar" (phonetic
near-miss). Neither is a bug in this integration - it's Whisper's real accuracy ceiling on those
specific words, measured rather than assumed.

### The wake word: Hello Zira (measured, not guessed)

The wake word is matched against Whisper's *transcript*, so how Whisper **spells** a rare name decides
whether hands-free wakes at all. "Zira" is much rarer than "Jarvis" (and in Hindi ज़ीरा/जीरा means cumin).
`scripts/wake_word_benchmark.py` synthesizes wake phrases with six macOS voices (Rishi and Aman - Indian
English, Lekha - Hindi, Daniel, Karen, Kathy), English and Hinglish, and runs them through the same decode +
mlx-whisper large-v3-turbo path the server uses: 136 transcriptions, raw results in `logs/wake_benchmark.json`.

| What was measured (36 "Hello Zira..." clips) | Woke |
|---|---|
| Whisper wrote exactly "Zira" (no priming) | 14/36 (39%) |
| Real matcher, no priming | **32/36 (89%)** |
| Real matcher, primed with "Hello Zira." | 34/36 (94%) |
| Old wake word, "Jarvis", for comparison (exact spelling / matcher) | 6/12 / 11/12 |

Without priming, Whisper wrote the name as *Zyra* (5x), *Zero* (3x), *Xera*/*Xero*, *Zera*, जीरा, जिरा, زیرا
(Urdu script), *0A*, and a couple of clips lost the greeting entirely. So `frontend/app.js` matches in three
layers: exact spellings, a **phonetic key** (`zyra`/`zeera`/`zera`/`xera`/`sira`/`zirah` all reduce to `zira`;
"Sarah", "Kira" and "Zara" do not), and Devanagari/Urdu spellings. Phonetic-only matches count only at the start
of a clip or right after a greeting (hello/hey/hi/namaste), and the everyday word "zero" only after a greeting.

**Priming** (`STT_INITIAL_PROMPT`, default `Hello <ASSISTANT_NAME>.`) tells Whisper the phrase in advance. It
raised recall (89% -> 94%; Hinglish clips 12/12 with the name kept in Latin script), and silence or noise did
**not** make it invent "Hello Zira" (they produced "Thank you." and "You", primed or not). But it has a cost:
it pulls look-alike names toward the wake word - "Hello Zara" came back as "Hello Zira" 6 times out of 6,
"Sarah" 4/6, and overall the matcher woke on 17 of 24 other-name clips primed versus 1 of 24 unprimed. So the
server only primes clips recorded **while waiting for the wake word** (`POST /api/voice/transcribe?wake=true`);
push-to-talk and follow-ups inside an active conversation are transcribed plainly, so a real command that
names someone is never rewritten. Set `STT_INITIAL_PROMPT=` (empty) to turn priming off entirely.

Limits, stated plainly: these are synthetic voices, not yours. When hands-free is waiting for the wake word and a
clip doesn't match, the transcript line shows `Heard: "..."` for a moment, so a spelling the matcher misses is
visible and can be added. A purpose-trained wake-word engine (e.g. openWakeWord with a custom "Zira" model) would
avoid depending on transcription spelling at all, but is not built.

**A more serious finding, and fixed**: the wake word itself doesn't always survive transcription
faithfully when the rest of the sentence is Hindi. Measured for real: "Haan JARVIS, kya scene hai?"
came back as "हाँ जारवेस, क्या सीन है?" - Whisper transliterated JARVIS into Devanagari (जारवेस)
instead of keeping it as a Latin loanword. `extractCommand()` in `app.js` now also matches a
best-effort list of plausible Devanagari transliterations (`WAKE_WORD_DEVANAGARI`) - not
exhaustive, since transliteration isn't fully predictable, but real coverage for the form actually
observed. A harder, separate failure mode was also measured: in one clip the word "JARVIS" was
dropped from the transcript entirely, with nothing left to pattern-match against - if hands-free
mode ever doesn't wake up, just say "Hello JARVIS" again.

**Also found and fixed while investigating this**: `extractCommand()`'s own text-cleaning regex
(`\p{L}` only, stripping everything else before comparing words) was silently corrupting Devanagari
text - Devanagari vowel signs (matras) are Unicode *Mark* characters, not *Letters*, so `\p{L}`
alone stripped them, turning "जारवेस" into "जरवस" before the wake-word check even ran. Fixed by
keeping `\p{M}` too. This affected any Devanagari word going through wake-word detection, not just
the name JARVIS.

`STT_LANGUAGE` is empty by default (auto-detect) — this is the honest default for Hinglish. Forcing
`STT_LANGUAGE=hi` or `=en` makes Whisper mis-render the *other* language in that clip, so only set it
if you find auto-detect consistently guessing wrong for your voice/accent.

**`STT_LANGUAGE_CANDIDATES=hi,en` (default, `faster-whisper` only — see above)** narrows auto-detect
to just these languages instead of Whisper's full ~100-language space. Unrestricted auto-detect can
drift into acoustically similar languages on short or accented clips — measured for real: a genuine
Hindi clip scored `hi=0.72` but `ur=0.12` (Urdu) as its actual runner-up, occasionally winning
outright on a noisier clip and coming
back as Urdu or, one step further, Arabic. Restricting to a candidate set and picking whichever of
*those* scores highest fixes this without forcing a single language outright (which would still
mangle whichever of Hindi/English wasn't forced). Only applies when `STT_LANGUAGE` is empty; set
`STT_LANGUAGE_CANDIDATES=` (empty) to restore full unrestricted auto-detect.

**Voice-activity detection (VAD) threshold is relaxed from Whisper's default.** Whisper's
`vad_filter=True` trims silence before transcribing (needed for push-to-talk clips, which always
have leading/trailing silence) — but its default speech-probability threshold (0.5) turned out too
strict for at least one real mic/environment: real server logs showed repeated clips with several
seconds of genuine recorded audio come back as `chars=0` (empty transcription) — diagnosed with the
`duration=`/`vad_kept=` figures in the `Transcribed ...` log line (`app/voice/speech_to_text.py`),
which showed VAD discarding the *entire* clip as "no speech" (`vad_kept=0.0s`) despite several
seconds of real `duration`. Lowered to `0.35` so quieter/less clear speech is more likely to be kept
for transcription rather than silently dropped; a near-silent/ambient-noise clip still correctly
comes back empty (verified with a real synthesized pink-noise clip: `duration=3.0s vad_kept=0.0s`,
`chars=0`), so this isn't a blanket "transcribe everything" change. **This does not fix a separate,
harder problem**: audio that's quiet/degraded enough can still be *mis*-transcribed (wrong words,
occasionally even the wrong language) rather than transcribed as empty — that's an audio-quality/ASR-
accuracy limit (mic distance, gain, background noise), not something a VAD threshold can correct. If
recognition still misses you often after this change, check your input device/mic gain in System
Settings and try speaking closer to the mic; `STT_MODEL=medium` (see the table above) is also more
accurate at low volume, at ~3x the latency.

### Text-to-speech: macOS `say` (local, offline, zero install)

Ships with macOS — no extra download. `TTS_VOICE=Rishi` (English-India, male) is the default —
picked because it reads romanized Hinglish (Latin script, the way most people actually text/speak
Hinglish, and the style used throughout this README's examples) clearly. Alternatives, listed with
`say -v ?`:

| Voice | Locale | Gender | Best at |
|---|---|---|---|
| `Rishi` (default) | English-India | male | Romanized Hinglish, English |
| `Tara` | English-India | female | Romanized Hinglish, English |
| `Lekha` | Hindi (`hi_IN`) | female | Devanagari-script Hindi |

**Honest limitation, measured, not assumed:** `say` speaks with **one voice per call**, and no
Indian-accented voice reads *both* scripts well. Concretely, feeding
`"Haan, samajh gaya. Let me check that for you. आज मैं coding कर रहा हूं।"` to `say -v Lekha` and
transcribing the result back produced the Devanagari portion almost perfectly but turned "Let me
check that for you" into gibberish; `say -v Rishi` reading the same sentence handles the romanized
part clearly but can barely read the Devanagari portion at all (recovered only the one English word,
"coding"). This is why the system prompt always asks the model to reply in Hinglish (Latin script)
for any Hindi content, never Devanagari — it sidesteps the problem for most replies rather than
fixing `say` itself.
A cloud TTS engine that natively handles Hinglish (Azure, ElevenLabs, ...) can be plugged in later as
another `TextToSpeech` implementation — see `app/voice/text_to_speech.py` — without touching the
chat/conversation layer at all.

#### Investigated as a local Hinglish-native replacement: SPRINGLab/Indic-Mio — real blocker found, not switched

`Indic-Mio` (22 Indian languages + English, explicitly trained on code-mixed sentences, emotion
tags) looked like it could remove `say`'s single-script-per-call limitation entirely. Real
investigation, not assumption:

- It's a two-stage pipeline: a Qwen3-architecture causal LM (`SPRINGLab/Indic-Mio`, ~1.2 GB)
  generates speech *tokens*, and a separate neural codec (`Aratako/MioCodec-25Hz-24kHz`, via the
  `miocodec` package — installable with `pip install git+https://github.com/Aratako/MioCodec`, not
  on PyPI proper) decodes those tokens into an actual waveform.
- The model card's own reference code hardcodes `device_map="cuda"` and recommends serving via
  `vllm` (both NVIDIA-GPU-oriented) — concerning for a Mac with no CUDA device, but **the generation
  stage worked**: loading with `device_map="mps"` (PyTorch's Apple Silicon GPU backend) instead of
  `"cuda"` worked with no code changes, and generated 133 speech tokens in ~8s on this M4 - genuinely
  promising.
- **The codec decode stage is where it broke**: `MioCodec.from_pretrained("Aratako/MioCodec-25Hz-24kHz")`
  — the exact checkpoint Indic-Mio's own README pairs it with — raises `ValueError: No vocoder
  weights found with prefix 'vocoder.'` in the current pip-installable `miocodec` version. A real,
  reproducible packaging/version mismatch between the library and its own reference checkpoint, not
  a Mac-specific or MPS-specific failure.

**Not switched to Indic-Mio** as a result — `say` stays the active `TTS_PROVIDER`. This wasn't
abandoned on a hunch: the codec bug is 100% reproducible today, and patching around an unofficial
package's internals to work around its own broken pairing with its own model card's recommended
checkpoint is a fragile foundation to build a "primary" TTS engine on. `TextToSpeechProvider`
staying a clean, swappable interface (`app/voice/text_to_speech.py`) means this can be revisited
directly once `miocodec` fixes the mismatch, or by tracing through exactly what its `from_pretrained`
expects vs. what that checkpoint actually contains, either of which is a focused follow-up, not
something to rush.

**Kokoro** (the other realistic candidate raised) does support Hindi as one of 8 languages, but its
architecture assigns one language/voice per synthesis call via a G2P (grapheme-to-phoneme) step
chosen per language — the same fundamental limitation `say` already has for code-switched text, not
a fix for it. Not pursued further given it wouldn't solve the actual problem.

### Kokoro TTS and streaming speech (the default voice since 2026-09-28)

`TTS_PROVIDER=kokoro` replaces IndicF5 in the normal pipeline. [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
(Apache-2.0) runs locally in its own worker (`third_party/kokoro/worker.py`, Python 3.11 in `.venv-kokoro`, because the
`kokoro`/`misaki` packages need Python < 3.13 and Zira's `.venv` is 3.14 - nothing in `.venv` was changed). It is
loaded and warmed up once at startup (`[KOKORO] loading...` / `[KOKORO] ready in 7.0s` in `logs/server.log`), in the
background, so text chat works at once; if it cannot load, Zira logs why, `/api/voice/status` shows `tts_error`, and
text-only Zira keeps working. There is no cloud fallback: no text ever leaves the Mac.

**Install (once):**
```bash
/opt/homebrew/bin/python3.11 -m venv .venv-kokoro
.venv-kokoro/bin/pip install -r third_party/kokoro/requirements.txt
```
The model (~330 MB) and spaCy's small English model download on the first start. Then set `TTS_PROVIDER=kokoro` in
`.env` and restart Zira (`./run.sh`, or the LaunchAgent).

**Streaming.** A voice turn no longer waits for the whole reply. Qwen's tokens go through a speech filter (code
blocks, tables, JSON, log lines and the Sources/Files link list are not read aloud; the text on screen is unchanged)
into a sentence buffer (`app/voice/streaming.py`): a chunk ends at `. ? ! : ; ।` or a line break - never inside a
decimal, URL, file path, abbreviation or markdown link - short sentences are joined, very long ones split at a comma.
Each chunk is synthesized as soon as it is complete and sent on the same chat WebSocket (`{"type": "audio"}` events,
then `speech_end`); the browser plays clip after clip while the next ones are being made. Tapping the mic while Zira
speaks interrupts her: queued audio is dropped and `{"type": "stop"}` ends Qwen's generation (a running image or
video job is never touched). Logged per reply: `[QWEN] first token`, `[QWEN] first sentence`, `[KOKORO] generation`,
`[TTS] first audio ready`, `[AUDIO] first playback` (reported back by the browser), and the total with Kokoro's
real-time factor.

**Hinglish.** Kokoro's English G2P reads Latin-script Hindi as English, and its Hindi G2P (espeak-ng `hi`) needs
Devanagari. So a chunk that is really Hinglish (`app/voice/hinglish.py`) has its Hindi words written in Devanagari -
from a word list of common words, or a simple spelling rule for words not in Kokoro's English dictionary - while
English words stay English; pure English chunks go to the English G2P untouched. Checked by transcribing Kokoro's
audio back with Zira's own Whisper: "Main tumhare Android project ka issue check karta hoon" came back as
"मैं तुम्हारे आन्रिड प्रोजेक्ट का इशू चेक करता हूँ" (it was "Main Tumhair Android project, Kaisu Cech, Karda
Hoon" with the English voice).

**Voice and settings** (`.env`, then restart): `KOKORO_VOICE` (Kokoro's own IDs: `hf_alpha`, `hf_beta` Indian female,
`hm_omega`, `hm_psi` Indian male, `af_heart` American female, `bf_emma` British female ...), `KOKORO_SPEED`,
`KOKORO_LANG` (`a`/`b` for English chunks), `KOKORO_DEVICE` (`cpu` default - keeps the GPU free for Qwen - or `mps`),
`KOKORO_HINGLISH`, `TTS_STREAMING`, `TTS_MIN_CHUNK_LENGTH`, `TTS_MAX_CHUNK_LENGTH`, `TTS_FIRST_CHUNK_LENGTH`. The UI's
reply-voice switch offers KOKORO and RISHI (macOS `say`, a local fallback).

**Measured on this M4 Air (16 GB), `scripts/tts_benchmark.py`, NEWLIGHT (Qwen3.5 4B), Kokoro `hf_alpha` on CPU:**
Kokoro startup 6.8-7.0 s (load + warm-up); synthesis about 0.13x real time (7.7 s of speech in ~1.0 s). "Tell me in
three short sentences why the sky is blue": Qwen first token 1.8 s, first sentence 2.2 s, first audio ready 2.6 s,
text finished 5.6 s - the first sentence was playable ~3 s before the reply was even finished.

**Limits.** Interrupting is by tapping the mic (or starting a new request); Zira does not listen *while* she speaks,
since her own voice would reach the mic. Kokoro's Hindi voices were trained on only minutes of data (grade C), so
English words inside Hinglish get an Indian accent ("weather" can sound like "vedar"), and numbers inside a Hinglish
sentence are read in Hindi. Proactive openers and `/api/voice/speak` still speak a whole text at once.

Emoji and other pictographs (⭐ 😊 ♪ 🙏) are removed before speaking (`strip_symbols`, app/voice/text_cleaning.py):
Kokoro used to read them out by name in the middle of a lullaby. `°C`/`°F` become "degrees Celsius/Fahrenheit".

### Singing (`create_song`, ACE-Step 1.5)

Kokoro can only speak, so songs and lullabies are made by [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5)
(MIT): Zira writes the lyrics ([verse]/[chorus] tags) and a style ("soft lullaby, gentle female vocals, music box"),
and the song - sung vocals plus music - is saved as an mp3 and shown as a player in the chat.

- Setup: `SONG_GENERATION_ENABLED=true`. The repo is cloned in `third_party/ace-step` with the model in its
  `checkpoints/` (~10GB: 2B turbo song model, 1.7B planning model, Qwen3-Embedding-0.6B, audio decoder), installed
  in `.venv-acestep` (Python 3.11). The worker is `third_party/song/worker.py`; its log is `logs/song.log`.
- Memory: one worker process per song, ended afterwards, so all of its memory comes back. While a song is made the
  chat model is unloaded (as for videos) and a loaded image/video model drops its weights - the selections stay and
  they load again on their next use. `SONG_LM=none` skips the planning model; `SONG_QUANTIZATION=int8_weight_only`
  shrinks the song model.
- Length: `seconds` (default 60, max 180: `SONG_DEFAULT_SECONDS`, `SONG_MAX_SECONDS`).
- Stop: the Stop button or typing/saying "stop" (`POST /api/songs/cancel`) ends the worker; nothing is saved.
- The chat stays busy while a song is made (like a video), since the chat model is unloaded.
- "Sing ..." always reaches the tool: the 4B chat model used to answer "sing chanda mama" with the lyrics as a spoken
  reply, so a clear request ("sing ...", "lori sunao", "gaana gao"; not "gaana sunao", which means play music, nor
  "who sang ...") is planned like a forced web search (`Planner._plan_song`, app/agent/planner.py): the model writes
  lyrics/style/title/language as structured output and the agent calls `create_song`; the reply is only the player.
- A saved song is played, not made again (user request): a request that names one ("sing the Nazia song", "play my
  nazia song", "nazia wala gaana sunao") plays it at once (`find_saved_song`: its key words against each gallery
  song's title and the request that made it, 60%+); "create / make / new / naya / banao" or nothing specific
  ("sing a lullaby") makes a new one. The planner checks before asking the model to write lyrics.
- A finished song plays by itself on the music player (the element a tap has unlocked on an iPhone), and stays in
  the chat as an inline player.
- Songs are in the gallery's Audio tab (media kind `audio`; the media table was rebuilt once on 2026-09-28 to allow
  it, every row copied - `Database._allow_audio_media`).
- Measured (2026-09-28, 30s song): ~90s loading + ~80s singing, worker peak 10.2GB. ACE-Step loads float32 on MPS
  and keeps an MLX copy of its song model by default (it ran out of GPU memory at ~22GB); the worker turns the MLX
  copy off and casts to bfloat16 (`ACESTEP_MLX_DIT`, `ACESTEP_DTYPE`).

### Optional voice: IndicF5-Hinglish

A second reply voice, [`Saravananravi/indicf5-hinglish`](https://huggingface.co/Saravananravi/indicf5-hinglish) (a
Hinglish fine-tune of AI4Bharat's IndicF5, an F5-TTS voice-cloning model), switchable at runtime with the **RISHI /
INDICF5** buttons in the voice bar (or the Voice card of the phone menu). Rishi stays the default and a restart always
comes back on Rishi. Picking INDICF5 starts a worker process and waits until the model is loaded (~20s through the
app, measured); picking RISHI stops it, which frees all of its memory (the worker measured 3.3GB).

**How it runs:** F5-TTS needs older libraries (transformers < 4.50, torchaudio with torchcodec) than Zira's own
Python 3.14 environment, so it has its own environment, `.venv-tts` (Python 3.11), and runs in a separate process
(`third_party/indicf5/worker.py`, fp32 on the Apple GPU; fp16 produced NaN audio). Zira never imports it. The worker
log is `logs/indicf5.log`. It speaks in the voice of a reference clip: the default is `MAR_M_WIKI_00001` (male);
others are listed in `third_party/indicf5/voices/voices.json` (`TTS_INDICF5_VOICE`).

**Measured on this Mac** (Whisper was used to check what it actually says):

| Text | Result | Time |
|---|---|---|
| Hindi in Devanagari ("नमस्ते, मैं ज़ीरा हूँ। आज मौसम बहुत अच्छा है।") | clearly understandable (Whisper heard it word for word) | ~23s alone, 40-66s through the live app with the LLM loaded, for 3.4s of speech |
| Hindi with English words ("मैं आज office जा रहा हूँ...") | mostly understandable | ~70s for 5.7s with the LLM loaded |
| Romanized Hinglish, plain English | poor / not understandable | similar |

So it is a slow, Hindi-script voice: expect roughly 10-20 seconds of waiting per second of speech while the LLM is
loaded, and garbled output for English or romanized text. One bug was fixed along the way: F5 sizes speech by UTF-8
byte count, which gave Latin text a third of the time it needs; the worker sets `speed` from the character count.

**Setup (once):** accept the terms of the gated [`ai4bharat/IndicF5`](https://huggingface.co/ai4bharat/IndicF5) on
Hugging Face (its vocabulary file is required), then:
```bash
/opt/homebrew/bin/python3.11 -m venv .venv-tts
.venv-tts/bin/pip install torch torchaudio torchcodec vocos x-transformers torchdiffeq ema-pytorch jieba pypinyin \
  librosa soundfile pydub cached-path einops safetensors huggingface-hub "transformers<4.50" accelerate wandb datasets matplotlib
HF_TOKEN=<read token> .venv/bin/python -c "from huggingface_hub import snapshot_download as d; \
  d('ai4bharat/IndicF5', allow_patterns=['checkpoints/vocab.txt']); d('Saravananravi/indicf5-hinglish'); d('charactr/vocos-mel-24khz')"
```
and set `TTS_INDICF5_ENABLED=true` in `.env`. The worker only reads the local cache (`HF_HUB_OFFLINE=1`).

### Testing voice without a microphone

```bash
# Generate a local test clip (English voice) and transcribe it — no server needed:
say -v Samantha -o /tmp/test.wav --file-format=WAVE --data-format=LEI16@22050 -- "Hello JARVIS"
curl -s -X POST http://127.0.0.1:8000/api/voice/transcribe -F "audio=@/tmp/test.wav;type=audio/wav"

# Speak text directly:
curl -s -X POST http://127.0.0.1:8000/api/voice/speak \
  -H 'Content-Type: application/json' -d '{"text": "Namaste, main JARVIS hoon."}' -o /tmp/reply.wav
```

### Known limitations

- **Code-switched Hinglish TTS** sounds noticeably worse than pure Hindi or pure English in a single
  voice (see above) — the single biggest gap between this V1 and a truly natural bilingual voice.
- **Continuous mode runs Whisper repeatedly while it's on**, not a dedicated low-power wake-word
  chip/model — real CPU and battery cost while active (see above for the openWakeWord alternative).
- **Silence detection auto-calibrates and relaxes over time, but can still misjudge** — a very noisy
  room, a mic with strong automatic gain control (levels stay nearly constant whether you're talking
  or not), or a random sound right at the 400ms calibration instant. The 8s hard cap and the
  click-to-force-stop override both bound how bad this can get, but watch the live level meter if
  continuous mode seems unresponsive.
- **`small` STT model** occasionally mis-spells Hindi words (still fully understandable); `medium` is
  meaningfully more accurate at ~3x the latency.
- **Continuous mode stays silent after an ordinary idle relisten** (nobody was talking — the
  amplitude watch never crossed its threshold) so it doesn't flash "Didn't catch that" every few
  seconds while you're not speaking. If it *did* detect your voice (crossed the amplitude threshold)
  but the clip still failed to transcribe, that's shown as "Didn't catch that — try again." for a
  moment before the next listen starts, so a genuine miss is distinguishable from normal silence.
- **No streaming TTS** — the full reply is synthesized before playback starts, so `SPEAKING…` begins
  after the complete text is ready (chat streaming to the *screen* is unaffected; only speech waits).
- **`say` on macOS only.** There is no fallback local TTS for other OSes in this build (not needed:
  this project targets a Mac).

## Context checkpointing (continue past the context window)

`OLLAMA_NUM_CTX` (default 8192 tokens) bounds how much conversation the model can see at once. Once a
conversation's estimated size crosses `CONTEXT_CHECKPOINT_THRESHOLD` (default 75%) of that window,
JARVIS asks the model itself to compact everything so far into a markdown summary — key facts,
decisions, and (if code work was in progress) exactly which files/steps are done and what's left — and
writes it to `data/checkpoints/<conversation_id>.md`. From then on, the next turn's context is built
from that summary plus only the raw messages *since* the checkpoint, not the full history, which is
what actually keeps the context bounded. If the conversation keeps growing, the checkpoint is
re-compacted the same way (folding the old summary + new messages into one updated summary) rather than
growing forever.

This runs as a background step after a reply is sent — like automatic memory extraction, it never
blocks or delays the visible answer, and if it fails (LLM error, disk error) the chat is unaffected;
compaction is simply retried on a later turn.

Inspect a conversation's checkpoint (if one exists yet — most short conversations never reach the
threshold):

```bash
curl -s http://127.0.0.1:8000/api/conversations/<conversation_id>/checkpoint
# 404 if checkpointing is off, or nothing has been compacted for this conversation yet
```

**Known limitations:**
- Token counts are estimated as `len(text) // 4` (~4 characters/token for English), not Qwen3's real
  tokenizer — good enough to trigger compaction with headroom to spare, but approximate, and Hindi/
  Devanagari text tokenizes differently than this estimate assumes.
- Checkpoint quality depends on how much context window is left for the model to write the summary in.
  At the real default (`OLLAMA_NUM_CTX=8192`), summaries reliably capture task state and personality is
  unaffected. Tested against a pathologically small context window (~1200 tokens) as a stress case: the
  model's replies degraded generally at that size (lost personality framing) — not specific to
  checkpointing, but a reason not to set `OLLAMA_NUM_CTX` unrealistically low.
- Set `CONTEXT_CHECKPOINT_ENABLED=false` to turn this off entirely.

## Self-learning (background research, on by default)

**What this actually is, stated plainly:** a local model served by Ollama cannot retrain itself from
what it reads — that needs real fine-tuning infrastructure this project doesn't have. This is **not**
that. What it does instead: while you're not chatting, JARVIS picks one topic from your long-term
memory (a project, a job, a stated preference — see below for what's excluded) it hasn't looked into
yet, searches the web for it once, asks the model to write a short factual summary, and caches that
summary locally. Future questions about that topic are then answered from the cache — no live search
needed — which is the practical benefit of "study so I don't have to check the internet for
everything," without overclaiming what a local LLM can do.

Unlike `web_search` (only ever called from a message you send), this is the one part of JARVIS that
contacts the internet on its own, unattended, on a timer — a deliberate, opt-out (not opt-in) exception
to this project's usual "internet access is request-driven" rule, because that's what was asked for.
Turn it off with `KNOWLEDGE_LEARNING_ENABLED=false` if you'd rather it never did.

**How it decides what to research:**
- Only from memory categories `project`, `work`, `preference` — never `personal` or `instruction`.
  Researching the user's own identity on the open internet has no benefit and isn't something a
  privacy-conscious local assistant should do.
- On top of that category filter, any memory text mentioning "you"/"your"/"me" is skipped even if it
  landed in an eligible category. This was added after real testing surfaced two concrete failures:
  a memory that fell into the `fact` catch-all category ("I am your creator.") got researched verbatim
  and produced an irrelevant Bible-verse summary; a memory correctly categorized `preference` ("I
  prefer you to say Rehan Ali when referring to me.") triggered a live web search for the user's own
  name and returned an unrelated stranger's social media profiles. Both are statements about the
  user/JARVIS relationship, not external topics — the category alone couldn't tell the difference, so
  a content-level check was added on top of it. `fact` was also dropped from the eligible categories
  entirely, since it's the classifier's catch-all default and is where this kind of statement lands
  most often.
- Never re-researches the same topic within `KNOWLEDGE_TOPIC_COOLDOWN_DAYS` (default 7 days).

**When it runs:** a background check every 60 seconds asks "has it been `KNOWLEDGE_IDLE_MINUTES`
(default 15) since the last chat message, and `KNOWLEDGE_RESEARCH_COOLDOWN_MINUTES` (default 60) since
the last research cycle?" — if both, it researches exactly one topic and goes back to waiting. It never
runs while you're actively chatting, and never runs back-to-back.

**Inspect or trigger it manually:**

```bash
curl -s http://127.0.0.1:8000/api/knowledge                      # everything learned so far
curl -s -X POST http://127.0.0.1:8000/api/knowledge/research-now  # research one topic right now, don't wait
curl -s -X DELETE http://127.0.0.1:8000/api/knowledge/<id>        # forget a specific learned topic
```

Learned entries feed into the same `ContextBuilder` that builds every prompt (`app/ai/context.py`),
the same way long-term memory already does — up to `KNOWLEDGE_CONTEXT_TOP_K` (default 3) entries
relevant to the current message are included, labelled as possibly-outdated cached research so the
model still prefers `web_search` when the user needs something current.

**Known limitations:**
- Cached, not live: a summary can go stale (versions ship, facts change) between when it was
  researched and when it's used. The context note tells the model to prefer `web_search` for anything
  that needs to be current.
- Topic quality is bounded by memory quality: it can only research what's already in your long-term
  memory, using that memory's own text as the search query, which is not always a great query.
- The relational-pronoun filter above is a heuristic (a regex over "you"/"your"/"me"), not a full
  understanding of what's safe to research — it fixed the two real failures found during testing, but
  is not a formal guarantee against every possible edge case.

## 6. Configuration

Everything is configured with environment variables or `.env` (see `.env.example`):

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server |
| `OLLAMA_MODEL` | `qwen3:8b` | Model tag |
| `DATABASE_PATH` | `data/jarvis.db` | SQLite file |
| `HOST` / `PORT` | `127.0.0.1` / `8000` | Bind address. **Keep on 127.0.0.1.** |
| `MODEL_LABEL` | derived (`Qwen3 8B`) | Name shown in the UI |
| `OLLAMA_NUM_CTX` | `8192` | Context window |
| `OLLAMA_TEMPERATURE` | `0.7` | Sampling temperature |
| `OLLAMA_KEEP_ALIVE` | `30m` | How long the model stays loaded |
| `OLLAMA_TIMEOUT` | `120` | Seconds without data before giving up |
| `OLLAMA_THINK` | `false` | Qwen3 "thinking" mode. Off = faster, no `<think>` output |
| `ASSISTANT_NAME`, `PERSONALITY_STYLE`, `PERSONALITY_EXTRA` | JARVIS, … | Personality (see `app/ai/prompts.py`) |
| `CONTEXT_MAX_MESSAGES`, `MEMORY_TOP_K`, `MAX_MESSAGE_CHARS` | 20, 8, 8000 | Context limits |
| `AUTO_MEMORY`, `AUTO_MEMORY_MAX_ITEMS` | `true`, `3` | Automatic memory extraction |
| `WEB_SEARCH_ENABLED`, `WEB_SEARCH_MAX_RESULTS`, `WEB_SEARCH_TIMEOUT` | `true`, `5`, `8` | Web-search fallback |
| `FILE_ACCESS_ROOTS`, `FILE_WRITE_ROOTS` | *(empty)* | Folders JARVIS may read / propose edits in (see above) |
| `FILE_READ_MAX_CHARS`, `EXPORTS_DIR` | `5000`, `data/exports` | Chunk size for `read_file`; where generated files go |
| `STT_PROVIDER`, `STT_MODEL`, `STT_LANGUAGE` | `unavailable`, `small`, *(auto)* | Speech-to-text (see *Voice* above) |
| `STT_LANGUAGE_CANDIDATES` | `hi,en` | Restricts auto-detect to these languages (see *Voice* above) |
| `TTS_PROVIDER`, `TTS_VOICE`, `TTS_RATE` | `unavailable`, `Lekha`, *(default)* | Text-to-speech (see *Voice* above) |
| `VOICE_MAX_AUDIO_BYTES`, `VOICE_MAX_SPEECH_CHARS` | `15000000`, `4000` | Upload/spoken-text size caps |
| `VOICE_TRANSCRIPT_CLEANUP_ENABLED`, `VOICE_TRANSCRIPT_CLEANUP_TIMEOUT` | `true`, `12` | Fix STT disfluencies before sending as a chat message (see *Voice* above) |
| `CONTEXT_CHECKPOINT_ENABLED`, `CONTEXT_CHECKPOINT_THRESHOLD` | `true`, `0.75` | Auto-compact conversations at this fraction of `OLLAMA_NUM_CTX` (see *Context checkpointing* above) |
| `CHECKPOINTS_DIR` | `data/checkpoints` | Where checkpoint `.md` files are written |
| `KNOWLEDGE_LEARNING_ENABLED` | `true` | Background research while idle (see *Self-learning* above) |
| `KNOWLEDGE_IDLE_MINUTES`, `KNOWLEDGE_RESEARCH_COOLDOWN_MINUTES` | `15`, `60` | When a research cycle may run |
| `KNOWLEDGE_TOPIC_COOLDOWN_DAYS`, `KNOWLEDGE_MAX_RESULTS_PER_TOPIC`, `KNOWLEDGE_CONTEXT_TOP_K` | `7`, `4`, `3` | Re-research delay, search results per topic, entries shown in context |
| `MUSIC_LIBRARY_ROOTS`, `MUSIC_REMOTE_INDEX_URL` | *(empty)* | Local folder(s) and/or your own server's index for the music player (see *Music* above) |
| `MUSIC_REMOTE_INDEX_CACHE_SECONDS` | `300` | How often the remote index is re-fetched |
| `MODEL_MODE` | `deep` | `light` or `deep` on launch (see *Model mode* above) |
| `LIGHT_MODEL`, `BALANCED_MODEL`, `DEEP_MODEL` | `qwen3-heretic:4b`, `qwen3-heretic:8b`, `qwen3:8b` | Model tag used by each mode |
| `LIGHT_VOICE_NUM_PREDICT` | `200` | Reply-length cap for voice turns while LIGHT is active |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

Logs go to the terminal. They record startup, Ollama connection, the model, requests (method, path,
status, timing) and memory operations (IDs and categories). **Message content is not logged.**

## 7. Architecture

```
                      ┌──────────────────────────── MacBook (the brain) ────────────────────────────┐
 Browser UI ──HTTP/WS─▶ FastAPI (app/main.py, app/api/*)                                              │
 (frontend/)          │      │                                                                        │
                      │      ▼                                                                        │
                      │   Agent (app/agent)  ── ToolRegistry ── web_search ──▶ DuckDuckGo (internet)     │
                      │      │                                                                        │
                      │      ├─▶ MemoryManager ──┐                                                    │
                      │      │                   ├─ SQLite (data/jarvis.db)                           │
                      │      ├─▶ ConversationStore┘                                                   │
                      │      ▼                                                                        │
                      │      ├─▶ CheckpointManager ── compacts old history ──▶ data/checkpoints/*.md   │
                      │      ▼                                                                        │
                      │   ContextBuilder: system prompt + memories + checkpoint + recent chat + msg   │
                      │      ▼                                                                        │
                      │   LLM (app/ai/llm.py, LLMBackend) ──HTTP──▶ Ollama ──▶ active_model (below)   │
                      └───────────────────────────────────────────────────────────────────────────────┘

 Model mode (implemented - only ever one model resident):
   settings.active_model = light_model (qwen3-heretic:4b), balanced_model or deep_model (qwen3:8b), fixed per process.
   POST /api/model/switch ──▶ OllamaModelManager.unload_and_verify() ──▶ set_env_var(MODEL_MODE) ──▶
     SIGTERM (graceful) ──▶ run.sh loop restarts app.main ──▶ new process reads the rewritten .env

 Voice (implemented, browser mic/speaker):
   Browser mic ──upload──▶ /api/voice/transcribe ──text──▶ same chat pipeline as above ──reply──▶
   /api/voice/speak ──WAV──▶ browser <audio>

 Self-learning (implemented, runs unattended while idle):
   idle_research_loop (app/main.py) ──idle check every 60s──▶ BackgroundResearcher
     picks a topic from MemoryManager ──web_search──▶ DuckDuckGo ──▶ LLM summarizes ──▶
     KnowledgeStore ──▶ data/jarvis.db (`knowledge` table) ──▶ read back into every ContextBuilder.build()

 Music (implemented, local + self-hosted only - no external catalog search):
   Agent ── ToolRegistry ── play_music/control_music ──▶ MusicLibrary ──▶ LocalMusicLibrary
     (sandboxed folder, app/tools/filesystem.py) and/or RemoteMusicIndex (your own hosted JSON
     index) ──▶ Track {title, url} ──"music" AgentEvent──▶ browser <audio> (currentMusicAudio)

 Future (interfaces exist, not implemented):
   Camera ─▶ VisionProvider ─▶ (faces / people / objects) ─▶ Agent
   Raspberry Pi mic/speaker ◀─Wi-Fi WebSocket─▶ RobotClient ─▶ /api/voice/* (same endpoints, new transport)
```

```
app/
  main.py            app factory, security middleware, logging, entry point
  config.py          env-based settings (pydantic-settings)
  api/               chat.py (REST, SSE, WebSocket, memories) · health.py · music.py (local file streaming)
                     model.py (LIGHT/DEEP switch)
  ai/                llm.py (LLMBackend + Ollama LLM) · prompts.py · context.py · model_manager.py
  restart.py         process-restart signaling marker (used by a model-mode switch)
  memory/            database.py · memory_manager.py · conversation_store.py · checkpoints.py
  knowledge/         knowledge_store.py · researcher.py (self-learning: background research)
  agent/             agent.py · planner.py · tool_manager.py
  tools/             base.py · web_search.py · filesystem.py (sandbox + read tools) · code_tools.py
                     api_scanner.py · postman.py · changes.py (approval-gated edits) · music.py
  voice/             speech_to_text.py (faster-whisper) · text_to_speech.py (macOS say)
                     text_cleaning.py · errors.py · transcript_cleanup.py
  vision/            VisionProvider interface                      (TODO)
  robot/             command_schema.py (RobotCommand) · robot_client.py  (TODO: hardware)
  models/schemas.py  Pydantic models
frontend/            index.html · style.css · app.js (no build step)
tests/               memory, context, chat API, LLM layer, robot schema, voice (STT/TTS/cleaning)
```

Swapping the model: set `OLLAMA_MODEL` (and `MODEL_LABEL`) in `.env`, then `ollama pull` it.
Swapping the provider: implement `LLMBackend` (`chat`, `stream`, `generate`, `status`) and pass it to
`create_app(llm=...)`.

### Security model

- Binds to `127.0.0.1` by default; `run.sh` warns if you change it. There is **no authentication**,
  so do not expose it to a network.
- Requests must address a local hostname (blocks DNS rebinding). State-changing requests and
  WebSocket connections that carry a browser `Origin` from another site are rejected.
- The model's output is rendered as text, never as HTML.
- Tools: `web_search`, and (only if you configure folders) read-only project tools plus approval-gated edit
  proposals. There is **no shell, no delete and no arbitrary file access**. See *Safety model for project
  access* above. Search results and file contents are labelled untrusted.
- The change approval endpoints are protected like every other state-changing request (local Host, same-site
  Origin), and the model cannot call them.
- The only data that leaves the machine is the web-search query (to DuckDuckGo). After private files are read,
  even that is disabled for the turn.
- Voice audio never leaves the machine: both STT (faster-whisper) and TTS (`say`) run locally. The only
  network access voice ever triggers is the one-time faster-whisper model download from Hugging Face.
- **Self-learning is the one deliberate exception** to "internet access is request-driven": while idle, it
  sends its own search queries (built from your long-term memory) to DuckDuckGo without you asking each
  time. On by default; see *Self-learning* above for exactly what it can and can't research, and set
  `KNOWLEDGE_LEARNING_ENABLED=false` to turn it off.

## 8. Roadmap

Built in this order; each phase should work fully before the next starts.

- [x] **Phase 1 — AI brain:** chat, streaming, history, explicit long-term memory, personality,
      config, logging, tests.
- [x] **Phase 2 — Voice V1:** `SpeechToText` (faster-whisper, local) and `TextToSpeech` (macOS `say`,
      local) implemented behind the existing interfaces; push-to-talk in the UI; voice turns reuse the
      existing chat/memory pipeline exactly. Still to do: natural code-switched Hinglish TTS (needs a
      bilingual engine or per-segment voice switching), continuous listening/VAD, streaming TTS,
      server-side interruption of in-flight synthesis (client-side stop is implemented).
- [ ] **Phase 3 — Vision:** implement `VisionProvider` (camera capture + a vision model such as
      `minicpm-v`); feed observations into the context.
- [x] **Phase 4 — Tools:** `web_search`, read-only project tools, Postman generation, and approval-gated code
      edits with Plan / Edit modes.
      Still to do: safe page fetching (SSRF protection), running tests/builds *with confirmation*, multi-file
      change sets approved together, more languages in the scanner (Django, NestJS, Ktor, GraphQL).
- [ ] **Phase 5 — Robot:** Pi-side WebSocket server, handshake/auth, acknowledgements, reconnect,
      emergency stop, and having the agent emit `RobotCommand`s. Requires allowing the Pi's host
      in the host/origin allowlist (`app/config.py`) and authenticating the link. The voice endpoints
      (`/api/voice/transcribe`, `/api/voice/speak`) already accept audio bytes over plain HTTP, so the
      Pi can reuse them as-is once it has its own mic/speaker loop.
- [ ] Later: reviewing/merging auto-saved memories (updates instead of add-only), embedding-based recall,
      conversation summarisation, multiple conversations list in the UI, markdown rendering.

Known TODOs are marked `TODO(...)` in the code.

## Troubleshooting

| Symptom | Fix |
|---|---|
| UI shows `AI OFFLINE` | Ollama isn't running: `ollama serve` |
| UI shows `MODEL MISSING` | `ollama pull qwen3:8b` |
| Web search fails or is slow | DuckDuckGo may be rate-limiting; wait a minute and retry, or set `WEB_SEARCH_ENABLED=false` |
| No 🎤 button in the UI | `GET /api/voice/status` says `stt_available: false` — set `STT_PROVIDER=faster-whisper` in `.env` and restart |
| First voice message is very slow | The STT model is downloading/loading (~1 min for `small`, ~2.5 min for `medium`); every message after is fast |
| "Microphone access was denied" | Allow microphone access for this page in your browser's site settings, then reload |
| JARVIS's voice sounds garbled on mixed Hindi/English sentences | Known `say` limitation — see *Voice → Known limitations* above |
| Continuous mode doesn't respond, or cuts you off mid-sentence | Click the mic while it says LISTENING to force it to process immediately — this always works. Also watch the live level meter: if it never turns green while you talk, your mic's automatic gain control may be defeating amplitude detection; adjust `SPEECH_LEVEL_MULTIPLIER`/`MIN_SPEECH_THRESHOLD`/`SILENCE_MS` near the top of the voice section in `frontend/app.js`, or open the console for the `[voice] level=... threshold=...` log line |
| Continuous mode ignores what I said | It only acts after hearing "Zira" (or the backup "jarvis") near the start of the utterance — check the browser console (`[voice] no wake word heard in "..."`) to see what it actually transcribed |
| "Transcription failed: ... End of file" | Fixed — was caused by extremely brief (<~100ms) recordings; update to the latest code if you still see this |
| JARVIS saved something wrong | `curl http://127.0.0.1:8000/api/memories`, then `DELETE /api/memories/<id>` (or `AUTO_MEMORY=false`) |
| `Port 8000 is already in use` | JARVIS may already be running: `lsof -iTCP:8000 -sTCP:LISTEN` |
| First reply takes a long time | The model is loading into memory; later replies are fast |
| Replies contain reasoning text | Keep `OLLAMA_THINK=false` |
| `Python 3.12+ not found` | `brew install python@3.12` then `rm -rf .venv && ./setup.sh` |

## Automatic model switch-on, fallback and Reset (2026-09-28)

- **Automatic switch-on:** an image request while the image model is None switches on `IMAGE_AUTO_STYLE` (default
  `realistic`); a video request while the video model is None switches on `VIDEO_AUTO_MODEL` (default `fastmetal5b`).
  It is the same switch as the buttons (`select_image_style` / `select_video_model`, app/api/images.py and
  videos.py). A model already picked is never changed. The automatic path first frees the *other* model's weights
  (its selection stays; it reloads on its next use), so an image model and FastMetal 5B are never in memory
  together. `off` keeps the old behaviour (refuse until a model is picked).
- **Fallback:** a load that fails or times out leaves None; an image/video that fails or times out switches its model
  back to None, freeing its memory (the chat model is already back after a video). A timeout first stops the
  generation at its next step (FastMetal: the worker is ended) and waits for it, so weights are never unloaded under
  a running generation. The user's own Stop keeps the model.
- **⟲ Reset** (header button, `POST /api/system/reset`): stops whatever is being made (image, video, song), switches
  the image and video models to None and loads the chat model again. Nothing is deleted.
- The selectors follow the server while something is being made, so an automatic switch shows up in the UI.
- **One model at a time:** switching an image model on (button or automatic) parks the video model on None, and the
  other way round. Animating a photo uses only the video model. "Parked" = None with the model remembered
  (`resume_style` / `resume_model`): the next request switches that same model on again, not the default.
- **Idle:** an image/video model unused for `MODEL_IDLE_MINUTES` (default 30) is parked (`model_idle_loop`, a
  once-a-minute timer; never during a generation). Reset parks too.
- While a model is being switched on for a request the chat shows "Setting up the image/video model…"; a model
  that is already loaded is used as it is.

## Scene-based video ads (`create_ad`, 2026-09-28)

"Make a 20 second ad for ZeroGo" (an ad word + a make word or "video"; not questions) is planned like a song: the
chat model writes the script as structured output (`Planner._plan_ad`: 3-5 scenes of what is seen, a caption and
one voice-over line, plus an instrumental music style), then `CreateAdTool` (app/tools/ad.py) makes it one step at a
time: Kokoro records the voice-over lines (each scene is made long enough for its line) -> each scene is one clip of
at most 5s from the video tool (`SCENE_DIR`: its automatic switch-on, Stop and fallback apply; ~70-90s per scene
with FastMetal 5B, 100-170s measured live) -> the video model is parked and ACE-Step makes the jingle -> PIL draws the captions (this ffmpeg
has no drawtext) and one ffmpeg run joins the scenes with 0.4s crossfades, overlays the captions and mixes the
voice over the music. About 12-15 minutes for 20 seconds. The chat model is unloaded once for the whole ad. A missing
voice-over or jingle leaves it out; a failed scene fails the ad. Stop: `POST /api/ads/cancel`. A phone
notification says when it is ready; it is in the gallery as a video (model "ad").
