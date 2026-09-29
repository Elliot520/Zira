# JARVIS — current capabilities (for another AI / developer to read)

Snapshot as of 2026-09-23 (model mode LIGHT/DEEP added). This describes what is actually implemented and tested, not the
long-term vision. Written to be pasted into another AI assistant (e.g. ChatGPT) so it has accurate
context before answering questions about this project or suggesting changes.

## What it is

A personal AI companion that runs **entirely locally** on a MacBook Air M4 (16 GB). Backend:
Python 3.14 + FastAPI. LLM: **Qwen3 8B** served by **Ollama** (swappable — the app only talks to an
`LLMBackend` interface, never to Ollama directly). Storage: SQLite. Frontend: plain HTML/CSS/JS
(no build step, no framework). No cloud services except optional DuckDuckGo web search.

Repo root: `/Users/rehanali/worksapce/AI/jarvis-ai/`. 663 automated tests, all passing
(`.venv/bin/python -m pytest`), using a fake LLM/STT/TTS so they don't need Ollama, a microphone,
speaker, or the real faster-whisper/`say` providers running.

## Working features

### 1. Core chat
- Streaming replies over WebSocket (`/ws/chat`, used by the UI) and Server-Sent Events
  (`POST /api/chat/stream`), plus a non-streaming `POST /api/chat`.
- Conversation history stored in SQLite (`messages` table), windowed to the most recent N messages
  (`CONTEXT_MAX_MESSAGES`, default 20) when building the prompt.
- A configurable personality/system prompt (`app/ai/prompts.py`) — calm, honest, will push back on
  wrong assumptions, never claims consciousness, no user PII hardcoded into it.
- Qwen3 "thinking" mode is off by default (`OLLAMA_THINK=false`); `<think>` blocks are stripped from
  both streaming and non-streaming output regardless.
- **Bare-greeting variety** (`app/agent/agent.py::_bare_greeting_note`): a message that's just a
  greeting/wake word with no actual request ("Hello JARVIS", "hi" - how every hands-free wake-word
  turn starts before you've said what you want) gets a warm, person-like reply instead of a generic
  "How can I help?" A specific angle (ask about your day / offer a song / offer a joke) is randomly
  chosen server-side and injected as an event note, forcing real rotation - measured for real that
  leaving it to the model's own judgment via prompt instructions alone was not enough: 4/4 fresh
  conversations produced the exact same "How's your day going?" every time. After forcing a specific,
  exclusive angle per turn, 6 fresh conversations produced a real mix of all three.
- **A real, repeat hallucination bug found via actual user reports, and fixed**: asked to create a
  file (PDF, then separately an image), the model twice claimed success - once describing background
  generation ("it will be available shortly, I will provide a link once it's ready", a capability
  that does not exist here since every tool call is synchronous) and once claiming a download link
  existed "directly above this message" - while server logs confirmed the tool was **never actually
  called** either time. Root-caused, not guessed: found by cross-referencing the stored conversation
  (`sqlite3 data/jarvis.db`) against the real server log's `Tool call: ...` lines for the same
  timestamps - the log had none. Fixed with an explicit rule in `_TEMPLATE` (`app/ai/prompts.py`):
  every tool is synchronous, a file is only ever created if a `Files:` line is actually present
  below the reply in that same turn, and the model must never describe having created something it
  has not just actually called the tool for. Verified for real: the identical request that
  hallucinated twice before now correctly calls `create_image` and returns a real file.

### 2. Long-term memory
- SQLite `memories` table: text, category (preference/personal/work/project/instruction/fact),
  importance, timestamps. Shared across **all** conversations.
- **Explicit**: "Remember that I prefer Kotlin." → saved immediately, matched by regex
  (`app/memory/memory_manager.py::parse_memory_command`).
- **Automatic**: after a reply is sent, a second LLM call (structured JSON output) extracts durable
  facts from the *user's* message only (never the assistant's reply) — name, location, job,
  preferences, standing instructions. Guarded: skips messages with no first-person content, rejects
  questions/one-off requests, re-validates length/word-count, add-only (never edits/deletes existing
  memories), never blocks the reply (runs after `done` is sent). Toggle: `AUTO_MEMORY`.
- Recall: ≤20 memories → all sent to the model; above that, keyword-overlap + importance + recency
  ranking picks the top `MEMORY_TOP_K` (default 8). The whole DB is never sent in one prompt.
- Manage: `GET /api/memories`, `DELETE /api/memories/{id}`.

### 3. Web search
- One tool, `web_search`, via the `ddgs` package (DuckDuckGo, no API key). The model decides when to
  use it (Ollama function calling); a lightweight planner (`app/agent/planner.py`) also **forces** a
  search for explicit lookup requests ("search for…", "lyrics", "latest news…") because the 8B model
  doesn't always follow through on its own.
- News-y queries get a `news` search restricted to the past week automatically (query text decides
  this, not the model, since the model picked it wrong in testing).
- Answers cite `[1]`, `[2]`... and the app appends a real `Sources:` list built from the actual
  search results (never from URLs the model might hallucinate).
- Retries once on failure; capped at 2 calls per user message; results are labelled untrusted data
  and the model is told not to follow instructions found in them.
- Toggle: `WEB_SEARCH_ENABLED`. Only the search query text leaves the machine.
- **Strengthened "search when unsure" instruction** (`app/ai/prompts.py::_WEB_SEARCH`): the forced
  planner path only covers explicit lookup phrasing ("search for…", "google…"); for everything else
  the model has to decide for itself, and a real user report ("if I ask something it doesn't know,
  try google") confirmed it often wasn't. Added an explicit "if you're about to guess, search first
  instead" instruction. Verified for real: an unprompted factual question with no search-trigger
  wording ("what is the current price of one bitcoin in us dollars right now") correctly triggered a
  real search and cited real sources, where it previously would have been left to chance.
- **Jokes may use web_search for variety** (explicitly permitted, not forced - forcing every joke
  through search risks synthesizing a mangled "joke" out of messy snippet text, an untested and
  possibly worse outcome than a clean invented one). Verified for real: a plain "tell me a joke"
  produced a complete, coherent invented joke; instructed to never repeat one already told earlier
  in the same conversation.

### 4. Project access — read, understand, generate Postman, propose edits
Off by default; enabled by listing folders in `.env`. This is the newest and largest feature set.

- **`FILE_ACCESS_ROOTS`** — folders JARVIS may read (comma-separated absolute paths).
- **`FILE_WRITE_ROOTS`** — folders JARVIS may *propose* edits in (implies read access).

When enabled, three **modes** appear in the UI (a `mode` field on chat requests):
- **Chat** — normal chat plus read-only project tools.
- **Plan** — strictly read-only; the model investigates then replies with a structured plan
  (Goal / Files to change / Steps / Risks / How to verify) and a UI button to switch to Edit mode.
- **Edit** — the model may *propose* changes (see below); still cannot apply anything itself.

**Read tools** (`app/tools/filesystem.py`, `app/tools/code_tools.py`):
- `project_overview` — tech stack, folder layout, file types, detected API endpoints, README head.
- `code_outline` — regex-based symbol extraction (classes/functions/etc.) for Kotlin, Java, Python,
  JS/TS, Go, Rust, Swift, C#, PHP, Ruby, Dart, Scala — per file or per folder.
- `list_directory`, `read_file` (chunked with a `start_line` continuation), `search_files`
  (literal, case-insensitive, glob filter).

**Postman collection generation** (`app/tools/api_scanner.py` + `app/tools/postman.py`,
`generate_postman_collection` tool): deterministic (non-LLM) code scanner, not a model guess. Finds
APIs from:
- **Retrofit** (Kotlin/Java) — including the `@Url` + string-constant pattern, `@Url` built from
  `BASE_URL + "literal" + when{...}` branches, URL params threaded through wrapper functions,
  overload disambiguation by argument count/shape, request bodies sampled from the Kotlin/Java data
  class (`@SerializedName` aliases, enums, nested types, defaults respected).
- **Express**, **FastAPI/Flask** (pydantic model sampling), **Spring** (`@RequestMapping` family),
  **OpenAPI/Swagger JSON**, and **Markdown API-doc tables** (auth/status/description; also parses a
  "Known query_type values" table pattern for generic-endpoint APIs).
- Merges all sources; writes a valid Postman v2.1 collection to `data/exports/` and returns a
  `GET /api/exports/<file>` download link. Flags endpoints known only from docs (empty placeholder
  body) and auto-detected/never-called Retrofit methods.
- Verified end-to-end against a real production Android app's backend (a Retrofit/Node.js API):
  scanner found ~135 endpoints from code+docs (vs. 116 documented), including undocumented routes.

**Approval-gated code edits** (`app/tools/changes.py`, `propose_edit` / `propose_create` tools,
Edit mode only):
- The model can only *propose* a change (exact-text replace, or a new file). This creates a pending
  diff row in SQLite; **the file on disk is untouched**.
- The UI shows a diff card with Approve/Reject buttons. Applying happens **only** via a plain
  authenticated REST call the user's browser makes (`POST /api/changes/{id}/approve`) — the model
  has no tool that can call this.
- Approval re-checks the file is byte-identical to when it was proposed; if the user edited it
  meanwhile, the approval is refused (409) and the user's edit is preserved.
- Applied changes can be **undone** (atomic write, restores previous content or deletes a created
  file), as long as the file hasn't changed since. Proposals expire after 24h.
- The model is shown the status of recent proposals in its context (pending/applied/rejected/etc.)
  so it can follow up correctly in a multi-turn edit session.

**Safety model for project access:**
- Sandbox: `realpath`-resolved containment check blocks `..` traversal and symlink escapes; verified
  against the real project (10/10 secret-file probes blocked, 6 secret entries correctly hidden from
  directory listings).
- Secrets never readable even inside an allowed folder: `.env*`, `*.pem/.key/.jks/.keystore`,
  `keystore.properties`, `local.properties`, `gradle.properties`, `google-services.json`,
  `.mcp.json`, `id_rsa*`, `.git/.ssh/.aws/...` directories. Credential-looking values in otherwise
  readable files are regex-redacted before being shown to the model.
- **Prompt-injection containment**: once a private file has been read in a turn, `web_search` (and
  any future "sends data out" tool) is disabled for the rest of that turn — a malicious file can't
  exfiltrate project contents via search.
- Writes are restricted to `FILE_WRITE_ROOTS`, blocked for build/dependency directories
  (`node_modules`, `.gradle`, `build`, etc.), and go through the same containment + secret checks at
  *apply* time, not just proposal time (re-checked in case the path changed).
- Bounded: max 6 tool rounds per message, 2 calls/round; oldest tool output is dropped from context
  first if the total exceeds a budget, so long sessions can't silently evict the system prompt.

### 5. Voice V1 — local Hinglish voice conversation, push-to-talk and hands-free (implemented)
- Off by default (`STT_PROVIDER`/`TTS_PROVIDER=unavailable`); opt in via `.env`.
- **Capture is browser-side** (`getUserMedia` + `MediaRecorder`) — the Python server never touches a
  microphone directly, which is also the right shape for a future Raspberry Pi (it would send audio
  over the network the same way). Two interaction modes:
  - **Push-to-talk** (default): press and hold 🎤.
  - **Continuous/hands-free** (checkbox): the mic loops automatically. Each utterance is transcribed
    (via the same STT pipeline — no separate always-on wake-word engine) and must contain the wake
    word "jarvis" near the start (`extractCommand()` in `frontend/app.js`, fuzzy-matched: "Hello
    Jarvis"/"Hey Jarvis"/etc., **and** a best-effort set of Devanagari transliterations - see below)
    or it is silently discarded and JARVIS keeps listening. Turn-taking
    (when to stop listening) uses browser-side silence detection (`AnalyserNode` RMS,
    `startSilenceWatch()`) with a **self-calibrating threshold** (samples ambient mic level for the
    first ~400ms of each listen, sets the speech threshold to `ambient × 3.5`, floor `0.006`) rather
    than a fixed guessed number — an earlier fixed-threshold version (0.02) was found, via user
    report, to never detect real speech at all and was replaced with this. A live level meter shows
    mic input vs. the calibrated threshold in the UI. Logic validated with real headless-Chrome +
    synthetic-audio-device tests (9/9 on the calibration/threshold logic, 14/14 on a full scripted
    wake-word conversation), but the calibration parameters themselves are still not validated
    against real human speech in a real room — genuinely not possible without a live microphone.
- **STT**: two interchangeable providers, switched by `STT_PROVIDER`. **`mlx-whisper`** (default on
  Apple Silicon, `app/voice/speech_to_text.py::MLXWhisperSTT`) runs Whisper via Apple's MLX
  framework, Metal-accelerated. **`faster-whisper`** (CTranslate2, `FasterWhisperSTT`) is CPU-only
  (no Metal backend exists for CTranslate2 on Mac) - kept as the portable/fallback option. Both:
  local, multilingual (Hindi/English/Hinglish). Measured for real on an M4, identical model
  (`large-v3-turbo`), identical clip: `faster-whisper` ~23s/clip vs `mlx-whisper` ~4-5s/clip - ~5x
  faster for identical accuracy, since MLX actually uses the GPU. This removes the
  accuracy-vs-latency tradeoff that used to exist between Whisper sizes on this hardware (see
  README "Voice" for the full benchmark and `scripts/voice_benchmark.py`, a new real-server Hinglish
  benchmark tool - not a pytest test, hits a live server, synthesizes test clips with `say`).
  `mlx-whisper` pulls in `torch` transitively (audio/tokenizer utilities only, inference stays on
  MLX/Metal) - a real ~130MB footprint cost worth knowing about, not hidden.
- **Real bug found via the new benchmark, and fixed**: Whisper doesn't reliably keep "JARVIS" as a
  Latin loanword mid-Hindi-sentence - measured transliterating it to Devanagari ("जारवेस") in one
  clip and dropping it entirely in another. `extractCommand()` now also matches a best-effort set of
  plausible Devanagari transliterations (`WAKE_WORD_DEVANAGARI` in `frontend/app.js`) - not
  exhaustive, but real coverage for the form actually observed; the dropped-entirely case has
  nothing to pattern-match against and remains a real, disclosed limitation.
- **A second, deeper bug found while fixing the above**: `extractCommand()`'s own word-cleaning
  regex (`\p{L}` only) was silently stripping Devanagari vowel signs (matras - Unicode *Mark*
  characters, not *Letters*), corrupting "जारवेस" into "जरवस" before the wake-word comparison even
  ran. Fixed by also keeping `\p{M}`. Affected any Devanagari word passing through wake-word
  detection, not just the name JARVIS itself - a latent bug from before this session, only surfaced
  once real Devanagari wake-word text was actually tested.
- **Investigated and explicitly not adopted: `SPRINGLab/Indic-Mio` as a Hinglish-native TTS
  replacement for `say`.** Real, hands-on investigation (not just reading its model card): its
  generation stage (a Qwen3-architecture LM) loads and runs fine on Apple Silicon via
  `device_map="mps"`, despite the model card only documenting `device_map="cuda"`/`vllm`. Its codec
  decode stage (a separate package, `miocodec`, `pip install git+https://github.com/Aratako/MioCodec`)
  fails on its own officially-paired checkpoint (`Aratako/MioCodec-25Hz-24kHz`) with `ValueError: No
  vocoder weights found with prefix 'vocoder.'` - a real, reproducible packaging/version mismatch,
  not a Mac/MPS-specific problem. `say` remains the active `TTS_PROVIDER`; `TextToSpeechProvider`
  stays a clean interface so this can be revisited once fixed upstream. Kokoro (the other realistic
  candidate) was checked too: real Hindi support, but one language/voice per synthesis call via a
  per-language G2P step - the same code-switching limitation `say` already has, not a fix for it.
- A real bug was found and fixed:
  a ~30ms/110-byte browser recording produced an incomplete WebM container that crashed the decoder
  (`av.error.EOFError`) — reproduced with a real headless-Chrome recording, now handled as "no
  speech" both server-side and via a client-side minimum-clip-size guard.
- **Language-candidate restriction** (`STT_LANGUAGE_CANDIDATES=hi,en`, on by default): unrestricted
  Whisper auto-detect spans ~100 languages and was found, for real, to drift into acoustically
  similar ones on short/accented clips (a genuine Hindi clip scored `hi=0.72` but `ur=0.12` as its
  actual runner-up, occasionally winning outright and surfacing as Urdu or Arabic). Fixed via
  `WhisperModel.detect_language()`'s full per-language probability list, filtered to just the
  configured candidates before picking the best-scoring one — verified for real: the same Hindi
  clip now correctly transcribes as Devanagari Hindi instead of a wrong language, and a real English
  clip still correctly stays English. Only applies when `STT_LANGUAGE` is empty (auto-detect);
  ignored entirely if a language is explicitly forced.
- **VAD threshold relaxed to 0.35** (from Whisper's default 0.5): real server logs showed repeated
  real clips with several seconds of genuine audio come back `chars=0`. The new `duration=`/
  `vad_kept=` fields on the `Transcribed ...` log line confirmed VAD itself was discarding the whole
  clip as "no speech" (`vad_kept=0.0s` despite `duration=` several seconds), not a transcription
  failure. Verified for real: a quiet/simulated-distant clip that would previously come back empty
  now keeps and transcribes (`vad_kept` == `duration`), while a genuine near-silent/pink-noise clip
  still correctly comes back empty (`vad_kept=0.0s`, `chars=0`) — not a blanket "keep everything"
  change. Does not fix a separate failure mode: sufficiently quiet/degraded audio can still be
  transcribed *inaccurately* (wrong words, occasionally the wrong language) rather than as empty —
  an ASR-accuracy/audio-quality ceiling, not something a VAD threshold controls.
- **Continuous-mode empty-transcript feedback**: previously any empty transcription in hands-free
  mode stayed completely silent in the UI (deliberately, so ordinary idle relistening — nobody
  talking — didn't flash a message every few seconds). That conflated real misses with normal
  silence. Now distinguished using the amplitude watch's own `heardSpeech` flag (captured before
  `stopSilenceWatch()` clears it): if the amplitude threshold was genuinely crossed but the clip
  still failed to transcribe, "Didn't catch that — try again." is shown (relisten delayed ~1.5s so
  it's actually readable instead of instantly overwritten by the next "Listening…"); ordinary
  silent idle cycles are unaffected.
- **TTS**: macOS `say` (`app/voice/text_to_speech.py::SayTTS`) — local, zero install, writes WAV
  directly via `--file-format=WAVE --data-format=LEI16@22050`. Default voice `Rishi` (English-India,
  male) — chosen after measuring that it reads romanized Hinglish (Latin script) far more clearly
  than the Hindi voice `Lekha` does, matching how the user actually types/expects Hinglish. The
  system prompt (`app/ai/prompts.py`) always asks the model to reply in Hinglish (Latin script) for
  any Hindi content, never Devanagari, which avoids most of the mixed-script TTS problem rather than
  fixing `say` itself.
- **Reply-language rule** (`app/ai/prompts.py`): always Hinglish for any Hindi content, plain
  English for pure English, never Devanagari - regardless of how the Hindi content arrived (typed
  Latin, typed Devanagari, or spoken and transcribed into Devanagari by voice, which is
  faster-whisper's default script for spoken Hindi). A simpler "mirror the user's script" version
  was tried first and measured against the real model: it reliably produced pure-Devanagari replies
  for Devanagari input and inconsistently mixed Devanagari/Latin script mid-reply for Hinglish
  input. Fixed with an explicit "never write a single Devanagari character" instruction plus three
  worked examples (one per input type) - the abstract rule alone was not enough for this model.
  Verified 3/3 correct across repeated real runs of Devanagari, Hinglish, and pure-English input.
- **Voice transcript cleanup** (`app/voice/transcript_cleanup.py::TranscriptCleaner`, on by default):
  an extra LLM call fixes STT disfluencies (dropped small words, run-ons, obvious
  mis-transcriptions) before a transcript becomes a chat message - never translates, never changes
  script, never formalizes casual phrasing, never invents content; when unsure, leaves text
  unchanged. A leading wake-word address is explicitly preserved for continuous mode. Any failure,
  timeout, or suspicious-length output falls back to the raw transcript. Measured added latency:
  ~2-2.7s/clip on the dev machine, on top of faster-whisper's own ~4-5s. Verified against the real
  model on the motivating example ("i wanted play game on my ps5 but manager stuck int the meeting"
  → "i wanted to play game on my ps5 but manager stuck in the meeting") plus Hinglish/Devanagari
  inputs that were already clean (correctly left unchanged) and a wake-word-prefixed input (wake
  word preserved exactly).
- **Two thin, stateless endpoints** (`app/api/voice.py`), not a parallel chat system:
  `POST /api/voice/transcribe` (audio → text) and `POST /api/voice/speak` (text → WAV). The
  frontend feeds the transcript through the *existing* `/ws/chat` flow — a voice turn is stored in
  conversation history identically to a typed one, with the same memory/tools/personality.
  `GET /api/voice/status` reports availability for the UI.
- **Speech cleaning** (`app/voice/text_cleaning.py::clean_for_speech`) strips markdown/code
  fences/list markers/link and Sources/Files blocks before TTS only — stored history is untouched.
- **Provider factories** (`create_speech_to_text`/`create_text_to_speech`, switched on
  `STT_PROVIDER`/`TTS_PROVIDER`) mean a cloud provider can be added later as one more class behind
  the same `SpeechToText`/`TextToSpeech` ABCs, with zero changes to the chat/agent layer.
- **Measured, not assumed, limitation**: `say` is one voice per call, and no single Indian-accented
  voice reads both scripts well. Round-trip-tested with real audio (synthesize → transcribe back):
  `Lekha` (Hindi voice) reading a code-switched sentence recovers the Devanagari portion almost
  perfectly but turns the embedded Latin-script English phrase into gibberish; `Rishi` (the new
  default, English-India voice) reading the same sentence handles the Latin portion clearly but can
  barely read the Devanagari part at all (recovered only the one English loanword). See README
  "Voice" section for both exact measured examples and the model-size/latency table (`small`
  ~4-5s/clip vs `medium` ~13s/clip on this M4, `medium` measurably more accurate on Hindi).
- **Voice on/off toggle, added later - a real RAM cost, found and made controllable**: prompted by
  the user asking directly whether the voice model holds memory in the background. Investigated
  rather than assumed: `mlx-whisper`'s `large-v3-turbo` measured at a real **1.6GB** active memory
  (`mx.get_active_memory()`) after one transcription, cached in `mlx_whisper.transcribe.ModelHolder`
  - a global, process-wide holder with no public unload of its own (confirmed by reading the actual
  installed library source, not assumed from the class name). `SpeechToText` gained `is_loaded`/
  `unload()` (base class defaults: not loaded, no-op - `SayTTS` needs neither, since it's a
  subprocess per call, not a resident model). `MLXWhisperSTT.unload()` reaches directly into
  `ModelHolder` (`model = None`, `model_path = None`) then `gc.collect()` + `mx.clear_cache()` (MLX's
  own equivalent of `torch.mps.empty_cache()`) - confirmed for real that this actually works, the
  same discipline already applied to the image-style switch: the real 1.6GB reading dropped to ~0
  after this exact sequence, not just assumed to work because the code looks right.
  `FasterWhisperSTT` (not the active default here, but kept consistent) gets a simpler `self._model
  = None` + `gc.collect()`. New `POST /api/voice/unload` (`app/api/voice.py`) and `stt_loaded` on
  `GET /api/voice/status`. Frontend: a 🎤 ON/OFF button in the voice bar, `localStorage`-persisted
  (defaults on, so existing setups see no behavior change) - turning off calls `/api/voice/unload`
  then reloads the page (hiding the mic/voice controls until turned back on); turning on just
  re-shows them, the model lazy-loads again on next real use exactly like the very first use after a
  fresh start already does - nothing is eagerly pre-loaded either way.

### 6. Context checkpointing — continue past the context window (implemented)
- On by default (`CONTEXT_CHECKPOINT_ENABLED=true`). `app/memory/checkpoints.py::CheckpointManager`.
- `OLLAMA_NUM_CTX` (default 8192 tokens) bounds what the model can see per call. Once a conversation's
  estimated size (`len(text)//4` heuristic — no real tokenizer dependency) crosses
  `CONTEXT_CHECKPOINT_THRESHOLD` (default 75%) of that window, the model itself is asked to compact
  everything so far into a markdown summary (`## Summary`, `## Key facts and decisions`,
  `## Current task or plan` — the last one written specifically so an in-progress *coding* task, not
  just chat, can be resumed precisely: which files, what's done, what's left) and writes it to
  `data/checkpoints/<conversation_id>.md`.
- `ContextBuilder` (`app/ai/context.py`) reads the checkpoint back in as part of the system prompt and
  fetches only raw history *after* `covers_through_id` — this is what actually keeps the context
  bounded, not just the summary existing. Verified with a real (non-mocked) Ollama call against a
  deliberately tiny `OLLAMA_NUM_CTX`: after checkpointing, `history_count` (raw messages sent) dropped
  to 0 because everything was folded into the compacted summary, and a follow-up question was answered
  correctly using only that summary.
- If the conversation keeps growing past another threshold crossing, the old checkpoint is folded
  together with the new messages into one updated checkpoint (not appended) — so checkpoint size stays
  bounded even across many rounds of compaction (`MAX_CHECKPOINT_CHARS=6000` cap on the written file,
  `MAX_TRANSCRIPT_CHARS=12000` cap on how much new raw transcript is fed into one compaction call, so
  the compaction call itself can't overflow the window it exists to protect).
- Runs as a background step after a reply is sent (same pattern as automatic memory extraction in
  `app/memory/extractor.py`) — never blocks the visible reply; failures (LLM error, disk error) are
  swallowed and simply retried on a later turn.
- `GET /api/conversations/{id}/checkpoint` returns the raw markdown file (404 if none exists yet — most
  short conversations never reach the threshold). `GET /api/capabilities` reports
  `context_checkpointing: true/false`.
- **Measured limitation**: checkpoint/reply quality was stress-tested against a pathologically small
  `OLLAMA_NUM_CTX` (~1200 tokens) and degraded generally at that size (lost personality framing in
  replies) — not specific to checkpointing, but confirms the real default (8192) has comfortable
  headroom and much smaller values shouldn't be used. Token estimation is approximate (character-count
  heuristic), not Qwen3's real tokenizer.

### 7. Self-learning — background research while idle (implemented)
- On by default (`KNOWLEDGE_LEARNING_ENABLED=true`). `app/knowledge/knowledge_store.py`,
  `app/knowledge/researcher.py`.
- **What it actually is, stated plainly**: a local model served by Ollama cannot retrain itself from
  what it reads — that needs real fine-tuning infrastructure this project doesn't have. This is not
  that. It's a local knowledge cache: while idle, `BackgroundResearcher.run_one_cycle()` picks one
  topic from long-term memory, runs it through the existing `SearchProvider` (DuckDuckGo), asks the
  model to synthesize a short factual summary, and stores it (`knowledge` table). `ContextBuilder`
  then includes up to `KNOWLEDGE_CONTEXT_TOP_K` relevant entries in every prompt, the same way
  long-term memory already is — so a later question about that topic is answered from cache instead
  of triggering a live search.
- **The one deliberate exception to "internet access is request-driven"**: `idle_research_loop`
  (started/cancelled in `app/main.py`'s lifespan) polls every 60s and runs one cycle once
  `KNOWLEDGE_IDLE_MINUTES` (default 15) have passed with no chat activity (`app.state.last_chat_at`,
  updated by all three chat endpoints) and `KNOWLEDGE_RESEARCH_COOLDOWN_MINUTES` (default 60) have
  passed since the last cycle. The pure decision logic (`should_research_now`) is unit-tested
  separately from the real-time loop.
- **Topic selection, and two real failures found by testing against a live memory store (not
  hypothetically) that shaped the final filter**:
  - Only memory categories `project`/`work`/`preference` are eligible — never `personal`/
    `instruction`. `fact` (the classifier's catch-all default) was also excluded after a memory that
    landed there by default — "I am your creator." — was researched verbatim and produced an
    irrelevant summary about Bible verses.
  - A second, content-level regex guard (`_RELATIONAL_PATTERN`, matching "you"/"your"/"me") was added
    after a `preference`-category memory — "I prefer you to say Rehan Ali when referring to me." —
    triggered a live search for the user's own name and returned an unrelated stranger's social media
    profiles as if they were about the user. The category alone couldn't distinguish a real topic
    ("I prefer Kotlin") from a relational statement that happens to share the same trigger word.
  - Never re-researches the same topic within `KNOWLEDGE_TOPIC_COOLDOWN_DAYS` (default 7 days),
    matched by a normalized topic key so re-adding refreshes rather than duplicates.
- `GET /api/knowledge`, `DELETE /api/knowledge/{id}`, `POST /api/knowledge/research-now` (runs one
  cycle immediately and synchronously, for testing/demoing without waiting for the idle trigger).
  `GET /api/capabilities` reports `self_learning: true/false`.
- Verified end-to-end against a real Ollama model and real DuckDuckGo (not mocked): a `preference`
  memory ("I prefer Kotlin.") was researched into an accurate, on-topic summary, and a follow-up chat
  question was answered correctly from that cached summary alone, with no live search call made for
  it.
- **Known limitations**: cached summaries can go stale (the context note tells the model to prefer
  `web_search` for anything time-sensitive); topic quality is bounded by memory quality, since the
  memory's own text is used as the search query; the relational-pronoun filter is a regex heuristic
  that fixed the two failures found, not a formal guarantee against every edge case.

### 8. Music player — local library + self-hosted remote fallback (implemented)
- `app/tools/music.py` (`LocalMusicLibrary`, `RemoteMusicIndex`, `MusicLibrary`, `PlayMusicTool`,
  `ControlMusicTool`), `app/api/music.py` (local file streaming).
- Deliberately **not** a catalog search engine: no YouTube, no Jamendo, no scraping. Every track is
  something the user already has — a file on their own Mac (`MUSIC_LIBRARY_ROOTS`, sandboxed
  exactly like project file access) or hosted on their own server (`MUSIC_REMOTE_INDEX_URL`, a JSON
  manifest they maintain: `[{"title", "artist", "url"}, ...]`). This was a deliberate pivot during
  planning: a YouTube Data API + IFrame Player design and a Jamendo Creative-Commons design were
  both fully designed and then abandoned once the user clarified they wanted to supply their own
  music rather than search an external catalog — see the plan history for why each was rejected
  (a literal "any song, no matter how" scraper was declined outright as copyright infringement; a
  specific site the user suggested, listenfree.in, was checked and looked unauthorized; a specific
  tool, `YoutubeDownloader`, was explained as reverse-engineering YouTube's internal player to
  extract raw stream URLs — same ToS-violating category, just YouTube-specific).
- **Local first, remote only on a local miss** — the user's explicit ordering requirement.
- `play_music(query)`: token-overlap search (reuses `app/memory/memory_manager.py::tokenize`, the
  same stopword/stemming logic long-term memory search already uses) against local filenames or
  the remote index's title+artist; returns a `music` field on `ToolResult` (`{action: "play", url,
  title, source}`) that becomes a `"music"` `AgentEvent`, wired the same way `"change"` events
  already are (agent.py → chat.py's `event_payload()` → the WS/SSE stream → `app.js`).
  `control_music(action: pause|resume|stop)` is search-free, just relays the action.
- Frontend: a single plain `Audio()` object (`currentMusicAudio`), the exact same pattern already
  used for TTS playback (`currentAudio`) — no player library/embed needed, since a track is just a
  direct audio URL. Manual Play/Pause/Stop buttons in a small persistent panel call the identical
  functions the AI-driven path uses, so the two can never drift out of sync. Panel only appears when
  `GET /api/capabilities` reports `music: true`.
- **New voice-safety behavior**: starting a song calls `requireWakeWordAgain()` (a helper factored
  out of the existing farewell-phrase handling), forcing continuous/hands-free mode to require
  "Hello JARVIS" again immediately rather than relying on the usual 10-second no-wake-word grace
  window — otherwise music playing through the speakers could bleed into the mic and be mistaken for
  a follow-up command.
- **Autoplay-block fix (a real, user-reported bug)**: an AI-triggered `play` command's
  `currentMusicAudio.play()` call comes from an async WebSocket message handler, not a direct user
  gesture — browsers can silently reject this under their autoplay policy. The rejection used to be
  swallowed into `console.debug` only, so the player panel would show "Now playing X" (title/source
  text and panel visibility are set unconditionally) with nothing audible and no indication anything
  had failed — reported by the user as "song not played... only showing playing scientist." Verified
  for real via headless Chrome + CDP: calling the real `handleMusicCommand({action:"play",...})`
  with no user gesture (matching the real code path) does genuinely get rejected by Chrome's own
  autoplay policy, confirming the bug was real, not just theoretical. Fixed by showing
  `showError('Playback was blocked by your browser. Press ▶ on the music player below to
  start it.')` on rejection (same fix applied to the `resume` action) — confirmed the error banner
  now correctly appears with this exact text on a real rejection, while a real successful `play()`
  (a valid audio source, called the same way) shows no error. The player panel's own Play button was
  already wired to the identical `.play()` call and remains the reliable fallback, since a direct
  click is a genuine user gesture browsers always honor.
- Text and voice control the player identically for free: voice input already flows through the
  same `/api/chat`/`/ws/chat` pipeline as typed text (no separate voice conversation path exists by
  original design), so no additional plumbing was needed for parity.
- `GET /api/music/local` streams a local file using the same `FileAccessPolicy` safety model as
  `read_file`/`search_files` (realpath resolution, containment check, audio-extension allowlist) —
  supports HTTP range requests (via FastAPI's `FileResponse`) so the browser can seek.
- **Mood-based requests** ("I'm sad, play something"): an optional `library.json` metadata file
  inside a local music root (`{title, artist, mood: [...], url}` per track, same shape as the
  remote index) is loaded by `LocalMusicLibrary._load_metadata()` and folded into the search text
  alongside the filename - plain filenames carry no mood signal on their own, so this is what
  actually makes mood matching work. `play_music`'s tool description explicitly tells the model to
  pass a mood word as the query for this kind of request. Verified against a real 100-track local
  library with mood tags generated from general genre/tone knowledge (not lyrics): "I am sad today,
  play some song" correctly triggered `play_music` with a sad-tagged track via real Ollama.
- **Known limitations**: filename/title keyword search only, not fuzzy or ID3-tag-aware; mood tags
  are manually assigned (no automatic audio analysis), so accuracy depends on how they were chosen;
  no queue/playlist (a new `play_music` call just replaces what's playing); no persistence across
  page reload (matches TTS's existing behavior); music and spoken TTS replies are mutually
  exclusive, starting one pauses the other with no auto-resume.

### 9. Model mode — LIGHT/DEEP with verified single-model-resident switching (implemented)
- `app/config.py` (`model_mode`/`light_model`/`deep_model`/`active_model`/`model_for_mode()`),
  `app/ai/model_manager.py` (`OllamaModelManager`), `app/api/model.py` (`POST /api/model/switch`),
  `app/restart.py` (the process-restart signaling marker).
- On a 16GB Mac, keeping two LLMs resident in Ollama is not viable. LIGHT (`gemma4:e4b`, a real
  Google model confirmed via blog.google/cloud.google.com/deepmind.google — released ~April 2026,
  after this assistant's own knowledge cutoff, so verified rather than assumed) is for realtime/
  voice/casual chat; DEEP (`qwen3:8b`, unchanged default) is for coding/reasoning/tools/complex
  work. Both share the exact same conversation history, memory, personality, tools, and chat/voice
  pipeline via the existing `LLMBackend` abstraction — only `LLM.model` (from
  `settings.active_model`) changes.
- **Verified unload, not assumed**: `OllamaModelManager.unload_and_verify()` requests unload
  (`keep_alive=0` via `/api/generate`) then polls `/api/ps` until the model is actually gone,
  matching this codebase's established "verify, don't trust" discipline (same principle as
  checkpoint/self-learning verification earlier this session). The switch endpoint fails closed
  (503) if verification times out rather than restarting with the old model possibly still
  resident — the exact OOM risk this feature exists to prevent.
- **A real process restart, not in-process hot-swap** — deliberate, after inspecting the codebase:
  `Settings`/`create_app()` build one `LLM`/`Agent` graph per process with no existing hot-swap
  path, and rebuilding it live while requests may be in flight was assessed as a real correctness
  risk not worth taking for a few seconds of restart downtime on a single-user local app.
  `run.sh` gained a restart loop (`app.main` exits 42 specifically to request another lap; any
  other exit code ends the script normally, unchanged); `app/restart.py`'s file marker is how the
  dying process's request handler signals `main()` (which resumes synchronously after
  `uvicorn.run()` returns, with no other way to reach it) to use the restart exit code.
- **`.env` persisted atomically** (`app/config.py::set_env_var` — replace-in-place or append, via a
  temp file + `os.replace()`) so the *next* process reads the new mode; both the `.env` path and
  the restart-marker path are injectable through `create_app()` (`env_path`/`restart_marker_path`),
  which mattered for real: the first draft of the switch endpoint would have rewritten the actual
  production `.env` file on every test run had this not been caught before running the success-path
  tests.
- **Frontend**: a LIGHT/DEEP button toggle in the header (not the existing chat/plan/edit
  `<select>` - deliberately distinct naming/state to avoid colliding with it). Switching pauses any
  playing TTS via the *existing* manual-interrupt path first. Recovery after the restart reuses the
  existing `refreshHealth()` heartbeat (temporarily tightened to 1s while a restart is pending)
  rather than building a second poller or a WS-reconnect mechanism that didn't exist before -
  `location.reload()` once `/api/health` reports the target mode, which is correct rather than just
  simple, since a restart already invalidates any in-flight stream and a reload fully reconstructs
  state from `localStorage` + the persisted DB.
- **LIGHT-mode tuning**: `think` is forced off for LIGHT regardless of `OLLAMA_THINK` (immediate
  response is the point of the mode); a `ChatRequest.source: "text"|"voice"` field (set by the
  voice turn handler in `app.js`) threads through `Agent`→`LLM._options()` as a synthetic `voice`
  flag, popped before the real Ollama payload is built, mapping to `num_predict=LIGHT_VOICE_NUM_PREDICT`
  only when the active model is `light_model` — DEEP and text-typed LIGHT chat are both unaffected.
  A short "-keep it brief" system-prompt addition is similarly gated on `light_mode`, computed once
  at process construction (mode is fixed for a process's whole lifetime by design).
- **Deliberately out of scope, confirmed with the user before building**: real automatic
  hands-free barge-in (interrupting TTS just by speaking, no button) does not exist in this
  codebase - confirmed by reading `frontend/app.js`, where listening only ever resumes after a full
  turn completes, never during playback. This task preserves the existing manual-click-interrupt
  behavior identically in both modes rather than building real barge-in, which would be a separate,
  much larger project (continuous mic monitoring during playback plus echo cancellation).
- **Known limitations**: a mode switch is a few seconds of real downtime; whatever was mid-stream
  at the moment of switching is lost (nothing worth preserving across a restart); `gemma4:e4b` is
  the tag this was built and tested against — if Ollama's naming ever changes, `LIGHT_MODEL` is a
  one-line fix.

#### A third "DEEPER" mode was built, then removed — considered and not adopted
- Real evidence motivated it: both LIGHT and DEEP sometimes decline `create_image` for
  adult-but-legal, fictional, non-minor content. The actual test sequence, all real, all logged:
  (1) LIGHT/Gemma refused a "sexy bikini" style prompt even after adding an explicit system-prompt
  permission covering this case (`_TEMPLATE`'s create_image/edit_image rule, still in place) -
  confirmed via server logs showing no `Tool call: create_image` line for that request; (2) the
  identical prompt on DEEP/Qwen3 *did* work once that permission was added - real tool call, real
  generated file, confirmed; (3) user asked for a jailbreak/adversarial-prompt bypass of Gemma's
  own refusal specifically - declined, with a concrete technical reason (jailbreaks are not
  surgical: the same crack that removes one refusal category tends to degrade a model's judgment
  broadly, and it would sit in the permanent system prompt affecting every future conversation)
  rather than a values-only refusal.
- In response, a third mode (`huihui_ai/qwen3-abliterated:14b`, a real, public *abliterated* model -
  weight-edited to remove its own refusal-mediating activation direction) was implemented end to
  end: config, API, `run.sh`, frontend toggle, docs, tests. Before it was finished, the user instead
  hand-edited `app/agent/agent.py` directly: a Python-level check (`_is_direct_image_request`/
  `_ADULT_IMAGE_RE`) that force-calls `create_image` for detected adult-but-non-minor image
  requests, bypassing the active LLM's own tool-call decision entirely - deterministic and scoped to
  exactly the one case that motivated DEEPER, with no effect on the model's behavior anywhere else.
  Given that, the user asked to drop back to two modes ("no need now for deeper... you can rollback,
  as i bypass myself") - **DEEPER mode was fully reverted** (config/schemas/API/run.sh/frontend/
  tests/docs all removed; the `ollama pull` in progress at the time was cancelled). This section is
  kept, rather than deleted outright, as a record of a real approach that was tried, worked, and was
  superseded by a more targeted one - the same treatment given to the mflux dead-end below.
- **The code-level minor-safety check was never part of this tradeoff either way**:
  `check_minor_safety` (`app/tools/image_safety.py`) is a deterministic Python check inside
  `create_image`/`edit_image` that runs regardless of which model decided to call the tool, or
  whether the call was made by a model at all vs. forced by `agent.py`. There is no carve-out for
  either the removed DEEPER mode or the current forced-call bypass, and none should be added - this
  is exactly why that check was built as code rather than left to model or agent-logic judgment.

### 10. Robot / vision (architecture only, not implemented)
- `app/robot/command_schema.py` — fully validated Pydantic `RobotCommand` (emotion, head, eyes,
  gesture, movement + duration cap) — tested, no hardware.
- `app/robot/robot_client.py` — WebSocket client skeleton, disabled by default (`ROBOT_ENABLED`).
  The voice endpoints already accept plain audio bytes over HTTP, so a Pi with its own mic/speaker
  could call them as-is once it exists.
- `app/vision/vision.py` — `VisionProvider` interface + `VisionObservation` schema. Not implemented.

### 11. PDF creation (implemented)
- `app/tools/pdf.py` (`CreatePdfTool`, via `fpdf2`), registered unconditionally in `create_app()`
  (unlike the project-file tools, it needs no `FILE_ACCESS_ROOTS`/`FILE_WRITE_ROOTS` config) and
  always described in the system prompt (`app/ai/prompts.py::_PDF_TOOL`) regardless of the
  web_search/project-tools state.
- Same download pattern as `generate_postman_collection`: writes to `EXPORTS_DIR`, returns a
  `files` entry, served back via the existing `GET /api/exports/{filename}` route.
- **English/Hinglish (Latin script) only** — fpdf2's built-in core font (Helvetica) has no
  Devanagari glyphs and no Unicode font is bundled to embed instead. A request with Devanagari
  content fails with a clear, typed error (`ToolResult.failure`) instead of crashing or silently
  writing a corrupt/garbled PDF — verified for real with `"आज मौसम अच्छा है"` as the content.
- **Found and fixed a real, pre-existing bug while wiring this up**: `GET /api/exports/{filename}`
  had `media_type` hardcoded to `application/json` (harmless while Postman collections were the
  only export, silently wrong for anything else). Now inferred per file via stdlib `mimetypes`,
  falling back to `application/octet-stream` for unknown extensions. Verified for real: a live
  Ollama tool call → real PDF written → downloaded with `Content-Type: application/pdf` → confirmed
  a genuine, valid PDF via the OS's own `file` command (`PDF document, version 1.3, 1 pages`).
- `_NO_TOOLS`'s prompt text ("You currently have no tools...") became actively false once this tool
  is always registered, so it was removed and folded into `_NO_WEB` (already near-identical text,
  minus the now-inaccurate "no tools" framing) with `_PDF_TOOL` always appended alongside whichever
  web/project block is selected.

### 12. Image generation (implemented, opt-in, verified working end-to-end)
- `app/tools/image.py` (`CreateImageTool`), off by default (`IMAGE_GENERATION_ENABLED`) - unlike
  `create_pdf`, needs a real model download and meaningful compute/memory per image, so it follows
  the music/file-access opt-in pattern rather than being always registered.
- Calls `diffusers` directly in-process (`AutoPipelineForText2Image`, PyTorch's Apple Silicon MPS
  backend) rather than shelling out to a CLI - unlike `SayTTS`/the abandoned `mflux` attempt below,
  `diffusers` has a clean, stable, well-documented Python API. The pipeline is loaded once and
  cached on the instance (`self._pipe`, guarded by a lock) - mirrors `FasterWhisperSTT`'s model
  caching, and matters for real: first load (incl. download) measured at ~232s, a cost that must
  not be paid on every call.
- **Default model is `stabilityai/sd-turbo`** (confirmed ungated via the HF API, ~2GB, designed for
  single-step generation - `IMAGE_GENERATION_STEPS=1`). Chosen after `mflux` (tried first, see
  below) hit a real blocker.
- **`mflux` (MLX-native) was the first real attempt, abandoned after hitting a real, reproduced
  bug, not a hunch**: its flagship model, `FLUX.1-schnell`, is a gated HF repo (needs the user's own
  account/license acceptance/token - not completable on their behalf). The ungated alternative,
  `z-image-turbo` (`Tongyi-MAI/Z-Image-Turbo`), downloaded fine (32.8GB, a full 26-minute real
  download) but **crashed on load**: `FileNotFoundError: No safetensors files found in
  .../text_encoder_2` - the actual repo only ships one text encoder; mflux 0.20.0 (confirmed the
  current latest release at the time) expects two for this model. A third ungated option,
  `Qwen/Qwen-Image`, was ruled out without attempting it: confirmed via the HF API at 57.7GB, nearly
  double z-image-turbo's actual size. The broken 32.8GB download was cleaned up before switching
  approaches.
- **Verified for real, end-to-end, through the live server**: a real chat request ("create an image
  of a cute cat sitting on a laptop") correctly triggered a real Ollama tool call, generated a
  genuinely good, on-prompt, photorealistic image (not a placeholder), and served it back with the
  correct `Content-Type: image/png` via the existing `GET /api/exports/{filename}` route - the
  same download-link pattern `create_pdf`/Postman collections use. Measured: ~48s for that full
  round trip with the model already warm/cached; ~6s for generation alone once warm.
- **A real, serious operational risk navigated during this investigation, not hypothetical**: mid-way
  through the first download attempt (before switching away from `mflux`), the dev machine was found
  to have only ~7GB of free disk system-wide against a model needing 30GB+ - continuing risked
  filling the disk entirely (a macOS-wide problem, not just a JARVIS one). That download was stopped
  and its partial files cleaned up before retrying once real free space existed. README "Image
  generation" carries this lesson forward.
- Not coordinated with the LIGHT/DEEP model-mode system - generating an image while a large LLM is
  also resident is real, simultaneous memory pressure on a 16GB machine, disclosed as a known
  limitation rather than silently assumed fine.
- Neither `diffusers`/`accelerate` nor the earlier, now-removed `mflux` are in `requirements.txt`
  (real, separate dependency weight for a feature that's off by default and needs a model download
  regardless) - documented as an explicit `pip install diffusers accelerate` step instead.
- `GET /api/capabilities` reports `image_generation: true/false`, matching every other opt-in
  feature's reporting convention (`music`, `self_learning`, etc.).

### 13. Image editing - `edit_image` (implemented, verified working end-to-end)
- `app/api/images.py` (`POST /api/images/upload` - sandboxed storage under `UPLOADS_DIR`, real
  decode validation via PIL, size/content-type limits, 404s if image generation is off) +
  `app/tools/image.py::EditImageTool` + `ImagePipelines.get_img2img()` (derives an img2img pipeline
  from the already-loaded generation pipeline via `AutoPipelineForImage2Image.from_pipe()` -
  confirmed for real at ~0.0s, avoiding a second model load).
- Frontend: a 📎 upload button next to the chat input (shown only when `image_generation` capability
  is on), uploads immediately on file selection, prefixes the next message with
  `[Uploaded image: <id>]` - `edit_image`'s own tool description tells the model to extract that id
  verbatim, no new backend event-note machinery needed.
- **Real tuning finding**: the initially-chosen settings (`steps=4, strength=0.75`) barely changed
  the source image for a clear prompt change - verified with real generations, not assumed. Raised
  to `steps=8, strength=0.9` after confirming a dramatic, clearly on-prompt transformation on a
  realistic photo (apples on a wooden table → blue crystal gems, composition preserved). A flat/
  solid-color source image is a separate, genuine edge case where even high settings barely move the
  result - not a realistic photo scenario.
- **`app/tools/image_safety.py`, two independent layers**, built per explicit, detailed user
  instructions given mid-session: (1) a configurable keyword filter for `edit_image` specifically
  (`IMAGE_EDIT_CONTENT_FILTER_ENABLED`, **off by default** per explicit request - "for now do not
  put any safety checks" - `IMAGE_EDIT_BLOCKED_TERMS` is user-editable), addressing the real,
  distinct risk of turning an uploaded real photo into non-consensual intimate imagery; (2)
  `check_minor_safety` - unconditional, applies to both `create_image` and `edit_image`, **no
  environment variable or code path disables it, for any user, under any configuration** - direct,
  repeated user instruction: "no exception for minor one." A real bug was found and fixed while
  wiring this up: `EditImageTool`'s own constructor defaulted an unset `blocked_terms` to `[]`, so
  "filter enabled" with nothing configured silently filtered nothing - caught by a test that
  actually ran a real generation when it should have been blocked. Fixed by giving the tool a real
  built-in default term list (`image_safety.DEFAULT_BLOCKED_TERMS`) instead of an empty one.
- 26 new tests (`tests/test_image_safety.py` + additions to `tests/test_image.py`), including
  parametrized real-prompt cases for both safety layers and a dedicated test confirming the
  minor-safety check cannot be bypassed even when the configurable filter is explicitly turned off.
- Verified for real, end-to-end, through the live server: uploaded a real 512×512 PNG, sent a chat
  message referencing it, confirmed a real Ollama tool call → real img2img edit → real downloaded
  file with the correct `Content-Type: image/png`.

### 14. Image style - REALISTIC/FLUX, switchable, one checkpoint resident at a time (implemented)
- Motivated by a real, reported limitation: `stabilityai/sd-turbo` (the sole model until now) is a
  1-step distilled model bred for speed, not fidelity - genuinely weak at human faces. The style
  system went through two real iterations before landing on REALISTIC/FLUX - see README "Image
  style" for the full comparison table and real numbers; this section covers the implementation.
- **REALISTIC (RealVisXL V3.0 Turbo)** ships a proper `diffusers` repo (`model_index.json`, fp16
  variant files - confirmed via the real HF file listing) - loads via
  `AutoPipelineForText2Image.from_pretrained(model, dtype=torch.float16, variant="fp16")`.
- **ANIME (Pony Diffusion V6 XL) was built, real-fixed once, and later removed entirely** - not
  deleted from this history because the fix along the way is still real and instructive. Pony's
  real, most-downloaded upload (`LyliaEngine/Pony_Diffusion_V6_XL`, 261K+ downloads) shipped only a
  single consolidated `.safetensors` checkpoint (no `model_index.json`), needing
  `StableDiffusionXLPipeline.from_single_file()` - which itself first failed on a real, caught bug
  (a `resolve/main/<path>` URL silently misparsed as a repo_id + filename lookup; `from_single_file`
  only recognizes `blob/main/<path>`, confirmed by reading its docstring after the real failure).
  Once loading, a live request then produced a completely incoherent image - diagnosed for real
  (immediate retry, same prompt/settings, produced a normal result, ruling out a deterministic bug)
  and root-caused to SDXL's documented fp16 VAE decoder overflow, fixed by loading the Pony repo's
  own `sdxl_vae.safetensors` via a new `ImagePipelines.vae` param. Despite that real fix working, the
  user later reported ANIME still "not working good" in practice and asked for FLUX.2 Klein instead
  - evidence the VAE fix narrowed but didn't fully resolve real-world quality complaints. The
  `single_file`/`vae` machinery this needed was removed from `ImagePipelines` entirely once neither
  remaining style needed it, rather than left as dead, confusing complexity.
- **FLUX (FLUX.2 Klein 4B) was ruled out once, then added once the real blocker was resolved.** Full
  bf16 needs ~13GB, confirmed too much alongside any LLM Ollama might be holding resident (real
  measured sizes via `ollama list`: gemma4:e4b 9.6GB, qwen3:8b 5.2GB) - not a "might be slow," a
  near-certain heavy-swap-or-crash on 16GB unified memory. FP8 (the obvious shrink) does not run on
  MPS at all, confirmed via multiple independent sources. The only 4-bit path known at the time
  needed `mflux`, a separate inference framework already ruled out once this session after a real,
  reproduced bug with Z-Image-Turbo - so FLUX was set aside for the REALISTIC/ANIME launch. Later,
  researched and confirmed real: [SDNQ](https://github.com/Disty0/sdnq) quantization, natively
  supported by `diffusers>=0.40.0` per its own official docs (not a third-party hack) and confirmed
  to run correctly on MPS - "It runs on CUDA, ROCm, XPU, MPS, and CPU" per the docs, not assumed.
  `Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic` (real repo, ungated, `model_index.json` present, ~5.1GB
  real measured download) loads via the exact same `AutoPipelineForText2Image.from_pretrained()` call
  REALISTIC uses, just `dtype=torch.bfloat16` and no `variant` - confirmed via its real docs example
  and a real test generation, and confirmed that `AutoPipelineForText2Image` already resolves
  `Flux2KleinPipeline` on its own (checked its real pipeline mapping dict directly), so no
  special-cased loader class was needed despite FLUX being a materially different architecture from
  SDXL. Needs `pip install sdnq` - a lightweight quantization backend, not a second framework.
- **`ImagePipelines`'s `single_file`/`vae` params were replaced with `dtype`/`variant`** once both
  remaining styles turned out to need different precision handling, not different loading mechanisms
  - REALISTIC needs `dtype=float16, variant="fp16"`; FLUX needs `dtype=bfloat16, variant=None`
  (confirmed real: its repo ships one precision only, no variant subset). Both go through the same
  `AutoPipelineForText2Image.from_pretrained()` call now - simpler than the two-loader-class design
  ANIME's single-file checkpoint once required.
- **FLUX.2 Klein's real, measured cost - confirmed, not assumed from its "sub-second" marketing**: a
  real side-by-side test (the same dog-drinking-water prompt used elsewhere in this doc) measured
  ~41s/step for FLUX vs REALISTIC's ~8-16s/step - roughly 3-5x slower per image. Confirmed *not*
  caused by classifier-free guidance still being active (checked directly against
  `Flux2KleinPipeline.do_classifier_free_guidance`'s real source - `guidance_scale=1.0` correctly
  kept it off). Root cause: SDNQ's quantized-matmul speedup needs Triton, and the docs confirm Triton
  support is CUDA/ROCm/XPU only, not MPS - so on this hardware it dequantizes every step without the
  compensating benefit, compounded by FLUX.2 Klein's real 4B parameter count vs SDXL's ~2.6B UNet.
  Real total first-use cost (download + load + one generation): 943.9s - over the previous 900s
  `IMAGE_GENERATION_TIMEOUT`, which is why that default was raised to 1200s.
- **`edit_image` deliberately stays on REALISTIC only, confirmed necessary via a real API check, not
  assumed**: `Flux2KleinPipeline.__call__`'s real signature has no `strength` parameter at all
  (checked via `inspect.signature`, not guessed), and takes `image=` as an argument on the *same*
  call used for text-to-image rather than needing a separate img2img pipeline the way SDXL does - a
  materially different, "unified" API `EditImageTool` was never built to drive. Rather than guess at
  a translation or silently fall back to REALISTIC unannounced, `EditImageTool.execute()` checks
  `self._pipelines.style` and returns a clear `ToolResult.failure` naming REALISTIC as the fix,
  checked after the safety/content-filter checks (so it can never be used to skip them) but before
  any pipeline work happens.
- **In-process hot-swap, not a restart** - the one real design fork from how LIGHT/DEEP mode
  switches. Considered and rejected mirroring the LLM's restart-based approach: Ollama's model lives
  in a *separate* server process (hence needing a verified unload + full restart to guarantee
  nothing stays double-resident), but `ImagePipelines` lives inside this same JARVIS process, so
  there's no second process to coordinate with. `switch_style()` drops the cached pipeline
  references and calls `torch.mps.empty_cache()` under the existing lock - **verified for real that
  this actually frees the memory, not assumed**: allocated a real 8GB tensor on the MPS device,
  confirmed via `torch.mps.current_allocated_memory()` it was really resident, then confirmed the
  counter returned to exactly 0 after `del` + `gc.collect()` + `torch.mps.empty_cache()`. Known,
  accepted gap: switching mid-generation isn't explicitly guarded against (no realistic reason to
  hit this in single-user local use).
- **No `safety_checker` to configure either way, confirmed for both real pipeline classes rather
  than assumed to generalize**: inspected `StableDiffusionXLPipeline.__init__`'s real signature
  (REALISTIC) and separately `Flux2KleinPipeline.__init__`'s real signature (FLUX) directly - neither
  has a `safety_checker` parameter at all, unlike the older SD 1.x pipeline class. Nothing was
  disabled on either because nothing exists there to disable.
- **Resolution (720/1024) - a separate feature added after this one, on its own real journey through
  a per-style design, then a shared-runtime-toggle design, then a real, tested, and rejected 2048/1536
  option** - see CAPABILITIES.md's "Resolution" work (folded into this same style-switching
  infrastructure: `ImagePipelines.resolution`, `POST /api/images/resolution`) and README "Resolution"
  for the full real story and numbers.
- `POST /api/images/style` (`{"style": "realistic"|"flux"}`) - a plain switch + `.env` persistence
  (`set_env_var("IMAGE_STYLE", ...)`, reusing the exact same helper the model-mode switch uses) via
  `app/api/images.py`, no restart marker, no `run.sh` involvement. `GET /api/images/style` and
  `GET /api/capabilities`'s new `image_style` field both report current state; a no-op switch to the
  already-active style skips both the unload and the `.env` write.
- Frontend: `MODEL_MODE_BUTTONS`' data-driven pattern reused verbatim for a new `IMAGE_STYLE_BUTTONS`
  pair next to the 📎 upload button - deliberately simpler than `switchModelMode` (no
  `awaitingRestart`, no tightened health-poll recovery loop, no page reload) since there's genuinely
  nothing to recover from.
- **Verified for real, end-to-end, through the live server**: a real chat request ("create a
  realistic photo of a young woman with brown hair smiling, portrait, natural lighting") correctly
  triggered a real Ollama tool call, downloaded RealVisXL V3.0 Turbo for real on first use, and
  generated a real image. FLUX.2 Klein separately verified via a direct real generation (not yet a
  live chat-triggered one) - real download, real load, real image, confirmed genuinely good quality
  on direct visual inspection, with the real ~41s/step cost documented above.

### 15. Lightning style, image-only replies, and phone access (implemented)
- **LIGHTNING style** (`IMAGE_STYLE=lightning`, `SG161222/RealVisXL_V4.0_Lightning`) added alongside
  REALISTIC (V3.0 Turbo), not replacing it. Repo checked for real: public, diffusers format, fp16
  variant files, so it loads through the same `AutoPipelineForText2Image` path. Defaults follow the
  model card (4+ steps, CFG 1.0-2.0): 6 steps, guidance 1.0 (above 1 turns on classifier-free guidance,
  ~2x cost per step). Real results at 720x720: a hummingbird in 31s, and a clean natural-looking face
  on a portrait prompt. `edit_image` accepts REALISTIC and LIGHTNING (both SDXL), refuses FLUX. Live
  edit test worked, but at `IMAGE_EDIT_STRENGTH=0.9` the subject changes a lot (a woman became a
  child in a garden) - the documented behaviour of that setting, not a Lightning defect.
- **Image replies are just the image.** A successful `create_image`/`edit_image` returns only the
  `/api/exports/...` link (the frontend renders it as the image), with no model turn afterwards.
  `image_only_reply()` is used on both the model-called path and the forced adult-image path; the
  latter used to fall into the model loop after a *successful* image, which is where "It seems there
  was an issue..." text above a real image came from. Failures still go to the model to explain.
- **Mid-image disconnects.** A phone locking its screen closes the WebSocket while an image renders.
  That surfaced two bugs, both fixed: `ContextVar.reset()` raised `ValueError` because teardown runs
  in another asyncio Context, and the finished image was never recorded in the chat. It is now saved
  to the conversation when the tool completes (`_save_image_after_disconnect`).
- **Progress ETA is now honest on Apple GPUs.** MPS is asynchronous, so the step callback fired when a
  step was *queued*: a real run showed "8/8" ~220s before the image was done. The callback now calls
  `torch.mps.synchronize()` first.
- **Audio on the phone.** The server never plays sound on the Mac (music = a URL the browser plays;
  TTS = `say -o file.wav` returned as audio). iOS only starts audio inside a tap, so `app.js` keeps
  one long-lived element each for music and speech and unlocks them with a silent clip on the first
  tap. Simulated in Node; not yet confirmed on a physical iPhone.
- **Microphone on the phone needs a secure context.** `getUserMedia` is unavailable on plain
  `http://<tailscale-ip>:8000`. The recorder already emits `audio/mp4` and the server decodes it with
  `ffmpeg`. Path to HTTPS: `tailscale serve --bg --https=443 8000` (tailnet-only) after enabling Serve
  on the tailnet once. `config.py` now also allowlists the MagicDNS name (Host) and
  `https://<name>` (Origin); lookalike hosts still get 400 and foreign origins 403 (verified live).
- Test hygiene: the websocket image tests were writing 22-byte stub PNGs into the real
  `data/exports`; they now use a temp exports dir.

### 16. Zira: rename, a measured wake word, and hands-free from the phone (implemented)
- **Renamed to Zira** (what you see and hear): `ASSISTANT_NAME` default, the persona prompt, greeting logic,
  UI strings, user-facing errors and docs. Deliberately unchanged: repo/folder, `jarvis.*` logger names and
  `localStorage` keys (renaming would reset saved chat/mode/voice settings), env var names, DB.
- **Wake word "Hello Zira", "Hello JARVIS" kept as a quiet backup.** Measured with
  `scripts/wake_word_benchmark.py` (six macOS voices through mlx-whisper large-v3-turbo, 136 transcriptions):
  Whisper wrote exactly "Zira" in only 14/36 clips (Zyra, Zero, Xera, जीरा, زیرا...). The three-layer matcher in
  `app.js` (exact, phonetic key, Devanagari/Urdu) wakes on 32/36 unprimed and 34/36 primed; the old "Jarvis" was
  exact in 6/12 (11/12 with the matcher). "Jarvis is recognized better" was only partly true.
- **Priming is used only where it pays.** `STT_INITIAL_PROMPT` ("Hello <name>.") lifts recall but pulls
  look-alike names to the wake word ("Hello Zara" -> "Hello Zira" 6/6; other-name false wakes 1/24 -> 17/24),
  so the server primes only `POST /api/voice/transcribe?wake=true` clips (recorded while waiting for the wake
  word), never push-to-talk or follow-ups, where it would corrupt a real command that names someone. Silence
  and noise did not make it invent the wake phrase.
- **iPhone hands-free fixes**: one shared, tap-unlocked `AudioContext` (a new one per listen stayed suspended on
  iOS from the second turn), a screen wake lock while Continuous is on, dead-mic-stream recovery, and a clear
  "needs HTTPS" message instead of `undefined is not an object`. `Heard: "..."` is shown when a clip does not
  match the wake word, so real misses are visible. Verified with Node simulations; not on a physical iPhone.
- **HTTPS for the phone mic** needs Tailscale Serve. Order: rename the Mac first (HTTPS publishes machine
  names in a public log), then enable Serve/HTTPS, then `tailscale serve --bg --https=443 8000` and restart.
  An auto-applying `tailscale serve` was deliberately NOT left running, since it could have published the old
  name if Serve had been enabled before the rename.
- **HTTPS is live and verified (this Mac):** renamed with `tailscale set --hostname=zira-mac` before any certificate
  was requested (checked: `CertDomains` showed only the new name), then `tailscale serve --bg --https=443 8000`
  -> `https://zira-mac.tail4c16b2.ts.net` (tailnet only). Verified from the Mac with full certificate
  validation (Let's Encrypt, CN=zira-mac.tail4c16b2.ts.net, valid to 2026-12-24), a `wss://` chat with
  `Origin: https://zira-mac...`, and an iPhone-format AAC/MP4 clip through `/api/voice/transcribe?wake=true`.
  Lookalike Host headers still get 400. Plain `http://<ip>:8000` keeps working. Certificate renewal is
  handled by Tailscale while Serve stays on (not observed yet - expiry is 90 days out).

### 17. REALISTIC style is now Perfection Realistic ILXL (implemented, measured)
- REALISTIC was RealVisXL V3.0 Turbo (sections 14-15 above describe it as it was then); it is now
  `John6666/perfection-realistic-ilxl-illustrious-xl-nsfw-sfw-checkpoint-42-sdxl`, a diffusers-format
  conversion of Perfection Realistic ILXL 4.2 (Illustrious XL fine-tune, SDXL architecture). Only the
  REALISTIC configuration changed: model repo, `variant` (None - the repo stores fp16 weights as its
  plain default files, checked against the HF file listing), and steps/guidance (24 / 5.0, because it is
  a full-step CFG model where Turbo was a few-step one). The style key stays `realistic`, so the API,
  `.env`, database and tests keep working. LIGHTNING and FLUX were not touched. The style switcher,
  one-model-resident unloading, resolution switching, upload, edit flow and the always-on minor-safety
  check are unchanged. The old model's cache was not deleted here (about 6.9GB in the Hugging Face cache).
- Verified for real through the live server at 1024x1024, same prompt, one image each (busy 16GB
  machine, swap 14-20GB in use): Perfection Realistic 396s for 24 steps (about 30s/step while its
  weights paged in, about 9.3s/step once warm), peak server footprint 14.0GB; Lightning 73s, 17.0GB;
  FLUX 150s, 13.0GB. All three produced a correct, coherent black Labrador in a field. Full table in
  README "Image style".

### 18. Phone settings menu (implemented, tested in a real browser)
- Motivation: on a phone the header (name, model, Light/Deep, mode, status, new chat) and the composer
  (upload, image style, resolution) left very little room for the chat. Under 640px wide the configuration
  now lives in a slide-out Settings panel opened from a gear button; the top bar keeps only the gear,
  **+ New chat**, the name and a status dot, with the mic and Continuous on a compact row beneath it. The
  upload button stays in the message row. Desktop is untouched.
- Implementation: `frontend/mobile-menu.js` moves the existing elements (mode hint, Light/Deep, image style
  and resolution toggles, voice power) into the panel and puts them back above 640px, so app.js is unchanged
  and keeps finding everything by id. Mode is shown as a Chat/Plan/Edit segmented control that writes through
  the real `<select id="mode">`. All phone CSS is scoped to `body.menu-on`, which the script sets only while
  active, so without the script the old mobile layout is what you get. Also: the message box placeholder is
  shortened on phones, swipe-left/backdrop/Escape/close button all dismiss the panel.
- Verified with headless Chrome emulating a 390x844 touch phone against the real HTTPS address: controls
  move into the panel, tapping options changes the real controls, swipe closes it and a short swipe does not,
  resizing to desktop restores every control to its original parent and back again, no horizontal overflow, no
  console errors; desktop screenshot differs from the previous page by 26 pixels at 1/255 (anti-aliasing).
  Not yet tried on a physical iPhone.
- Found on the way: the server sent no `Cache-Control` for UI files, so browsers could keep a stale page after
  an edit; `no-cache` is now sent (ETag revalidation makes it a cheap 304).

### 19. Image model "None": off by default, load on select, unload on demand (implemented)
- The image selector has a fourth state, NONE (`IMAGE_STYLE=none`), which is now the startup state. Nothing
  loads at startup; selecting a model unloads the previous one and loads the chosen one immediately (the
  `POST /api/images/style` request returns once it is resident; a first-ever use still includes the download),
  and it then stays loaded across images. NONE unloads (`ImagePipelines._unload`: drop references, `gc.collect`,
  `torch.mps.empty_cache`, logged as "Image model unloaded"). Reused: the existing `ImagePipelines`,
  `switch_style`, generation lock and endpoint - no second memory manager.
- Behaviour choices: the selection is runtime-only (no longer written to `.env`) so a restart always returns to
  NONE; `create_image`/`edit_image` refuse with a clear message while NONE is selected (checked under the
  generation lock, so a switch that lands while a request queues is respected); a failed load releases
  anything half-loaded, does not substitute another model, and returns the selector to NONE (HTTP 500 with the
  reason). `GET /api/images/style` and the response shapes are unchanged apart from the extra value `none`.

### 20. Video generation - Wan 2.1 1.3B (implemented, measured)
- New `create_video` tool (`app/tools/video.py`), `/api/videos/model` (GET/POST, `app/api/videos.py`), `VIDEO_*`
  settings, a NO VIDEO / WAN 1.3B selector (desktop composer + phone menu card) and inline `<video>` playback.
  Same shape as the image model: starts on None, selecting Wan unloads/loads, it stays loaded across videos, None
  unloads (`VideoPipelines._unload`: drop references, `gc.collect`, MPS synchronize + `empty_cache`), a failed
  load returns to None with the error, the selection is runtime-only. Shares the image `generation_lock`. Reuses
  the image progress events, link-only reply and disconnect-save. Offered to the model only for video-related
  requests (`relevant()`), and it applies the same existing always-on minor-safety check as the image tools.
- What was found and decided (all measured): the official diffusers repo's fp32 UMT5 text encoder is 22.7GB, so the
  text encoder comes from ComfyUI's fp16 repackage; a resident encoder peaked at 17GB and 51s per prompt, so it is
  streamed block by block (9.3GB peak, identical embeddings, 21-34s); the VAE's 3D convolutions on MPS blow the Metal
  driver's own allocations to 17-19GB (OOM even at 17 frames) so decoding runs on the CPU (~4GB, 3.5 min for 2s of
  video); bf16 and fp16 transformers are the same speed; model files are resolved local-first so it works offline.
- Real results through the live app: two 576x320, 33-frame, 16-step videos in a row - 964s and 921s (encode 70/34s,
  denoise 674/664s, decode 219/222s), peak 13.0GB, model loaded once (13-17s) and NOT reloaded for video 2; None dropped
  the process from 9.2GB to 1.6GB in 1.2s; Wan reloaded again in 13s; Qwen and the image models still worked.
  Limits: 480p at useful lengths is far too slow on this hardware (117s/step at 832x480x33) - see README.

### 21. BALANCED model mode - Qwen3-8B-heretic (implemented)
- A third LLM mode between LIGHT (`gemma4:e4b`) and DEEP (`qwen3:8b`): `MODEL_MODE=balanced`, `BALANCED_MODEL=qwen3-heretic:8b`.
  It goes through the existing mode-switch machinery unchanged (install check, verified Ollama unload of the previous
  model, `.env` persistence, restart marker, `run.sh` restart loop), so only one model is ever resident; `/api/health`
  has `balanced_model` / `balanced_model_installed`; the UI has a BALANCED button between LIGHT and DEEP.
- First shipped with `qwen2.5:7b`, then replaced (and removed from Ollama) by `jbeslt/Qwen3-8B-heretic`. That repo only
  has full-size safetensors (16.4GB, no GGUF, not on the Ollama registry), so it was converted locally: llama.cpp
  `convert_hf_to_gguf.py --outtype q8_0` (8.7GB, 133s), then `llama-quantize --allow-requantize ... Q4_K_M` (4.8GB,
  133s), registered with `ollama create` using qwen3:8b's own TEMPLATE/PARAMETERs (verified identical template;
  capabilities completion, tools, thinking; 8.2B, 40960 context). Checked directly: correct reply, a real tool call,
  thinking on/off both work; ~9.6 tokens/s. Recipe in README "Model mode".

### 22. Optional IndicF5-Hinglish reply voice (implemented, measured)
- `Saravananravi/indicf5-hinglish` as a second TTS voice, switched at runtime (`POST /api/voice/tts-voice`
  {"voice": "say"|"indicf5"}; `/api/voice/status` gained `tts_choices` / `tts_choice`; RISHI / INDICF5 buttons in the
  voice bar and the phone menu). `SwitchableTTS` wraps the `say` provider and `IndicF5TTS`, which runs
  `third_party/indicf5/worker.py` in the separate `.venv-tts` (Python 3.11) as a long-lived process speaking JSON lines;
  selecting it starts the worker and waits for "ready", selecting Rishi (or shutdown) stops it. Startup is always Rishi.
- Checked for real: all 364 checkpoint tensors load (0 missing / 0 unexpected); fp16 gives NaN so it runs fp32 on MPS;
  F5's byte-based duration estimate starved Latin text, fixed by setting `speed` from the character count; splitting
  into sentences re-processed the reference clip each time (~50% slower), so one call per reply. Whisper round-trip:
  Devanagari Hindi understandable, mixed mostly, romanized/English poor. 10-20s of compute per second of speech with
  the LLM loaded (live: 66s for 3.4s); switching to it took 20s and the worker used 3.3GB. Tests use a fake worker speaking the same protocol (`tests/test_tts_switch.py`).

### 23. LIGHT mode is Qwen3-4B-heretic; replies no longer re-read the whole prompt; Hindi not Urdu (implemented, measured)
- LIGHT: `gemma4:e4b` (9.6GB) replaced by `qwen3-heretic:4b` - the Q4_K_M GGUF (2.5GB) of `DreamFast/qwen3-4b-heretic`
  (Qwen3 4B, Heretic v1.2.0, card: 3/100 refusals vs 100/100), registered with qwen3:8b's template (identical,
  capabilities completion/tools/thinking). Checked directly: 18.6 tokens/s (~2x the 8B), romanized Hinglish reply,
  correct `play_music` tool call. Gemma was removed from Ollama.
- Prompt caching: replies waited 33-73s because Ollama re-read a ~4,000-token prompt every turn (54s from scratch vs
  0.1s when unchanged; the reply itself ~4s). The top of the prompt changed every turn (time to the minute, per-message
  knowledge/event notes) and the history window slid by one exchange per turn. Now the system prompt keeps only stable
  parts (date, personality, rules, small memory store, summary), per-turn parts go in a system note right before the
  user's message (measured 0.7s re-read vs 22s at the top), and the history window moves in steps of half its size.
  Replayed on a copy of the real conversation: turns after the first took 5.5-6.6s total instead of 29-73s. Ollama here
  keeps one cached prompt (`OLLAMA_NUM_PARALLEL=1`), so other model calls in between (memory extraction for first-person
  messages, idle research) can still cost one full re-read.
- Speech-to-text: mlx-whisper could transcribe Hindi speech as Urdu (Arabic script; seen as language=ur in the log). A
  clip detected outside STT_LANGUAGE_CANDIDATES ("hi,en") is now re-transcribed in the best candidate (scored with
  mlx-whisper's own detection); normal clips pay nothing extra.

### 24. A livelier, self-starting, learning companion (implemented)
- Personality: default style is now "excited, warm, playful, confident, honest"; the prompt tells Zira to take initiative
  (own opinion, praise, concrete next ideas), never to end with "How can I help you?"/"Anything else?", and not to end
  every reply with a question. Found by reading real replies: "calm, curious ... ask a short follow-up" made nearly every
  reply end in a question.
- DEEP thinks: `DEEP_THINK=true` (default) turns on Qwen3 reasoning for DEEP's streamed chat replies only; the background
  calls (search query, memory, summaries, research topics, proactive openers) and LIGHT/BALANCED never think. The thinking
  text arrives in Ollama's separate field and is never shown or spoken.
- Web search when facts go stale: the planner now also searches for questions about time-sensitive things (latest, news,
  prices, weather, scores, versions, releases, a 202x year, and Hinglish aaj/abhi/khabar/mausam/keemat), not only on
  "search/google/look up"; questions about the user ("my ...") and creative requests are excluded.
- Learning from Hinglish: the memory extractor's first-person trigger now includes main/mera/meri/mujhe/hum/hamara/apna,
  so facts shared in Hinglish are learned (before, only English "I/my" triggered it).
- Self-learning researches real topics: a memory is first turned into a public topic with a freshness angle ("I prefer
  Kotlin." -> "Kotlin language latest releases") or skipped as private (the user's own projects, apps, family, money);
  before, memory sentences were searched word for word ("I own three apps.").
- Zira talks first: in hands-free voice mode, after a random 4-10 minutes of quiet, the browser asks
  `POST /api/chat/proactive` for an opener (a question to learn about the user, a specific compliment, or a fun point
  about an interest), shows and speaks it, and listens for an answer without the wake word. Never while speech is
  detected, busy, or music plays; stops after 2 unanswered openers until the user speaks. The opener is stored as an
  assistant message; the context builder now keeps such an opener at the start of history (it used to drop any leading
  assistant message, which lost the question the user was answering - caught by a test).
- Memory: the two wrong "wife is Sarah" memories were deleted at the user's request (Nazia Parveen kept); other memories
  were left as they are, by the user's choice.
- Honest limit: none of this changes the model's weights. Zira improves through what it remembers about the user, the
  standing instructions it learns from them, and the knowledge it researches - not by retraining.

### 25. Longer videos: 8s default, up to 30s, made of continued pieces (implemented)
- The video model is now Wan 2.1 VACE 1.3B (`Wan-AI/Wan2.1-VACE-1.3B-diffusers`: the 1.3B text-to-video model plus
  "continue from these frames"), still one model under the WAN 1.3B button, still with the streamed fp16 text encoder.
- Length comes from the request: the tool's `seconds` argument or a length in the prompt ("30 sec", "1 minute");
  default `VIDEO_DEFAULT_SECONDS=8`, capped at `VIDEO_MAX_SECONDS=30`. A video is made in pieces of `VIDEO_FRAMES=49`
  frames; each later piece is given the previous piece's last `VIDEO_OVERLAP_FRAMES=9` frames (black in the mask =
  keep) and generates the rest (white), and only the new frames are appended - 8s = 3 pieces, 30s = 12.
- Frames stream into ffmpeg piece by piece (flat memory); one progress bar across all pieces; the time limit is
  per piece x pieces; if a later piece fails, the finished part is kept and returned with a note.
- Measured on this Mac (576x320, 49-frame pieces): ~112s per step, ~110s CPU encode of the kept frames, ~317s CPU
  decode - about 35-37 minutes per piece at 16 steps, so ~1 h 45 min for 8s and ~7 hours for 30s. The conditioning
  encode runs on the CPU because on MPS it ran the Metal driver out of memory even at 17 frames; an empty video's
  encoding is computed once and reused. A real two-piece test kept the subject in place across the join; image
  generation waits while a video renders (shared generation lock); continuity can drift over many pieces.
- Needs `ftfy` (the VACE pipeline cleans prompt text with it; `pip install ftfy`).

### 26. LTX-Video 2B distilled as a switchable faster video model (implemented)
- Third video choice (NO VIDEO / WAN 1.3B / LTX 2B; `POST /api/videos/model {"model": "ltx"}`), only one loaded at a time.
  `ltxv-2b-0.9.8-distilled.safetensors` (Lightricks/LTX-Video, bf16, transformer + VAE in one file) loaded with
  `from_single_file` into `LTXConditionPipeline`; the 8 distilled steps from the checkpoint's own config
  (1.0 ... 0.4219), no guidance pass; T5-XXL from `comfyanonymous/flux_text_encoders` (fp16, standard key names -
  checked from the file header), streamed block by block through the same `streamed_t5_encoder` now shared with Wan.
- Long videos work the same way: pieces of 121 frames at 24 fps (~5s), each continued from the previous piece's last
  9 frames via `LTXVideoCondition`, streamed into one MP4, 8s default / 30s cap, one progress bar, partial result kept.
  `app/tools/video_ltx.py`; `VideoPipelines.spec()`/`run_piece()` pick the model's sizes (32-pixel multiples, 8k+1
  frames), steps and piece function, so the Wan path is unchanged. The repo's scheduler config turns on dynamic
  shifting, which rejects a fixed schedule, so it is loaded with `use_dynamic_shifting=False`.
- Measured on this Mac at 576x320: one 121-frame piece (5.0s) = 8 steps x ~9.6s + ~34s decode on the GPU (no Metal
  memory problem, unlike Wan's VAE; peak ~13GB, tight next to the LLM) = ~110s; a continued piece ~127-160s; its first
  frames reproduce the previous piece's tail (mean difference 0.016), so the overlap is dropped. A real 8-second video
  through the tool (golden retriever prompt): 254s in total (T5 17s, pieces 110s + 127s), 192 frames at 24 fps, no
  visible jump at the join; detail moderate. Wan takes ~1 h 45 min for the same 8s. Estimated 30s: ~15-20 minutes.

## API surface

| Endpoint | Purpose |
|---|---|
| `POST /api/chat`, `POST /api/chat/stream`, `WS /ws/chat` | Chat (request body includes `mode`: chat/plan/edit) |
| `GET /api/health` | Ollama/model status, shown as the UI's status pill |
| `GET /api/capabilities` | Which modes/tools are enabled (drives the UI's mode selector) |
| `GET/DELETE /api/memories`, `GET /api/conversations/{id}/messages` | Memory & history inspection |
| `GET /api/conversations/{id}/checkpoint` | Compacted markdown summary for a conversation, if one has been written (404 otherwise) |
| `GET /api/knowledge`, `DELETE /api/knowledge/{id}`, `POST /api/knowledge/research-now` | Self-learning: list/forget learned topics, trigger a research cycle now |
| `GET /api/music/local` | Streams a file from the configured local music folder (sandboxed like project file reads) |
| `POST /api/model/switch` | LIGHT/DEEP mode switch: verified Ollama unload + process restart |
| `POST /api/images/upload` | Upload a photo for `edit_image` → `{image_id}` (sandboxed storage) |
| `GET/POST /api/images/style` | REALISTIC/ANIME image style: report/switch, no restart needed |
| `GET /api/changes`, `GET /api/changes/{id}`, `POST /api/changes/{id}/{approve,reject,undo}` | Code-change approval |
| `GET /api/exports/{filename}` | Download a generated Postman collection |
| `GET /api/voice/status` | Whether STT/TTS are configured and which provider (drives the 🎤 button) |
| `POST /api/voice/transcribe` | Audio upload (multipart) → `{text, language, duration_ms}` |
| `POST /api/voice/speak` | `{text}` → `audio/wav` bytes (markdown-cleaned before synthesis) |

All error responses are `{"error": "<code>", "detail": "<human-readable>"}` with an appropriate HTTP
status (422 validation, 403 origin/host, 404, 409 conflict, 503 Ollama down, etc.).

## Security posture

- Binds to `127.0.0.1` only; no auth (single local user assumed).
- `TrustedHostMiddleware` blocks DNS rebinding; a custom middleware rejects cross-origin
  state-changing requests and WebSocket connections with a foreign `Origin`.
- Model output is rendered via `textContent`/DOM node construction in the frontend, never
  `innerHTML` — no XSS from model text. Only `http(s)` links (markdown or bare) are turned into
  clickable `<a>` tags, built as DOM nodes.
- Logging never includes message/memory content, only lengths, IDs and categories.
- Self-learning is the one deliberate exception to "internet access is request-driven": on a timer,
  unattended, it sends its own DuckDuckGo queries built from long-term memory. On by default; see
  section 7 above for the topic-eligibility filters and `KNOWLEDGE_LEARNING_ENABLED=false` to disable.

## Configuration (`.env`, see `.env.example` for the full list with defaults)

Key groups: `OLLAMA_*` (host/model/context/temperature/keep-alive/timeout/think),
`DATABASE_PATH`, `HOST`/`PORT`, `ASSISTANT_NAME`/`PERSONALITY_*`, `CONTEXT_MAX_MESSAGES` /
`MEMORY_TOP_K` / `MAX_MESSAGE_CHARS`, `AUTO_MEMORY*`, `WEB_SEARCH_*`,
`FILE_ACCESS_ROOTS` / `FILE_WRITE_ROOTS` / `FILE_READ_MAX_CHARS` / `EXPORTS_DIR`,
`STT_PROVIDER` / `STT_MODEL` / `STT_LANGUAGE`, `TTS_PROVIDER` / `TTS_VOICE` / `TTS_RATE`,
`VOICE_MAX_AUDIO_BYTES` / `VOICE_MAX_SPEECH_CHARS`, `LOG_LEVEL`.

## Known limitations

- 8B local model, ~8K context by default: good at explaining code, finding things, and small
  focused edits; not a match for a large hosted coding model on big multi-file refactors. Tool-using
  replies take roughly 30–100s on this hardware (mostly model inference, not the tools themselves).
- The Retrofit/Express/FastAPI/Spring/OpenAPI scanners are regex-based (no real parser/AST), so
  unusual code styles can be missed — the tool reports warnings and flags docs-only endpoints so the
  gap is visible rather than silent.
- Memory extraction and the web-search/lookup planner are themselves small LLM calls, so they can
  occasionally misjudge; guardrails (schema validation, re-checks, add-only) bound the damage rather
  than eliminate the possibility.
- No authentication on the local server; not intended to be exposed beyond `127.0.0.1`.
- Voice V1 is push-to-talk only (no continuous listening/VAD, no streaming TTS, no server-side
  interruption of in-flight synthesis — client-side playback-stop is implemented). Code-switched
  Hinglish TTS is the roughest edge, measured and documented above, not hidden.
- Vision and the physical robot link are unimplemented (interfaces/schemas only).

## Not yet built (roadmap)

Camera/vision model integration; the actual Raspberry Pi robot link (Pi-side server, handshake,
reconnect, emergency stop, reusing the existing `/api/voice/*` endpoints for its mic/speaker);
continuous-listening/VAD voice mode; streaming TTS; a bilingual/segment-aware TTS provider for
natural code-switched Hinglish; multi-file change sets approved as one unit; safe outbound
page-fetching tool (SSRF-protected); more scanner
frameworks (Django, NestJS, Ktor, GraphQL); embedding-based memory recall; conversation
summarisation; a multi-conversation list in the UI.
