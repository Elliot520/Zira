# Zira (jarvis-ai) - project overview

A personal AI companion that runs entirely on one Mac (MacBook Air M4, 16GB). It chats, searches the web,
remembers you, speaks and listens, plays your music, reads your code folders, makes PDFs, images and short
videos - and you can use it from the Mac's browser or from your iPhone anywhere over Tailscale.

This file is the map. The details (design decisions, measurements, history) are in
[README.md](README.md) and [CAPABILITIES.md](CAPABILITIES.md).

---

## 1. How it fits together

```
 iPhone / Mac browser  ──  frontend/ (HTML + JS, served by the app)
          │   WebSocket /ws/chat  +  REST /api/...
          ▼
 FastAPI server (app/main.py, started by run.sh, port 8000)
   ├─ Agent (app/agent/)        message -> context -> LLM -> tools -> reply
   ├─ Tools (app/tools/)        web search, music, files, PDF, image, video, code changes
   ├─ Memory (app/memory/)      SQLite: conversations, long-term memories, summaries
   ├─ Knowledge (app/knowledge/) background web research while idle
   └─ Voice (app/voice/)        speech-to-text, text-to-speech
          │
          ├─ Ollama (separate process)         the chat models (Qwen3)
          ├─ PyTorch on the Apple GPU (MPS)    image and video models, in-process
          ├─ mlx-whisper (Apple GPU)           speech-to-text, in-process
          ├─ macOS `say`                       default voice (Rishi)
          └─ .venv-tts worker (separate)       optional IndicF5 Hinglish voice
```

Everything runs locally. The internet is used only for web search, background research, and first-time
model downloads.

---

## 2. Technology stack

| Layer | What | Version |
|---|---|---|
| Language | Python | 3.14 (main), 3.11 (`.venv-tts` for IndicF5 only) |
| Web server | FastAPI + Uvicorn, WebSockets | FastAPI 0.141, Uvicorn 0.53 |
| Data | SQLite (`data/jarvis.db`) | built in |
| Chat models | Ollama | 0.30.7 |
| Image / video | PyTorch (MPS) + diffusers, SDNQ (4-bit FLUX) | torch 2.14, diffusers 0.40, sdnq 0.2.6 |
| Speech-to-text | mlx-whisper (Apple GPU) | 0.4.3 |
| Text-to-speech | macOS `say`; IndicF5 (F5-TTS) in `.venv-tts` | - |
| Web search | DuckDuckGo (`ddgs`) | 9.16 |
| PDFs | fpdf2 | 2.8 |
| Video files | ffmpeg (H.264) | system |
| Frontend | Plain HTML/CSS/JavaScript, no framework | - |
| Remote access | Tailscale (private network) + Tailscale Serve (HTTPS) | - |
| Model conversion (one-off) | llama.cpp (`convert_hf_to_gguf.py`, `llama-quantize`) | brew |

---

## 3. Models and when to use them

Only one chat model and one image/video model are loaded at a time, to fit in 16GB.

### Chat (Ollama) - switch with LIGHT / BALANCED / DEEP (restarts the server)

| Mode | Model | Size | Use it for |
|---|---|---|---|
| LIGHT | `qwen3-heretic:4b` (DreamFast/qwen3-4b-heretic, Q4_K_M) | 2.5GB | image commands, quick voice chat - fastest (~19 tokens/s) |
| BALANCED | `qwen3-heretic:8b` (jbeslt/Qwen3-8B-heretic, converted to Q4_K_M) | 5.0GB | everyday chat |
| DEEP (default) | `qwen3:8b` | 5.2GB | real work: code, planning, analysis. **Thinks before replying** (better, slower) |

Both "heretic" models are Qwen3 with refusals removed; both use qwen3:8b's own chat template.

### Images - pick in the image selector (starts on NONE = nothing loaded)

| Button | Model | Best for |
|---|---|---|
| NONE | - | frees 6-13GB of memory |
| REALISTIC | Perfection Realistic ILXL (SDXL / Illustrious XL) | bold, detailed photoreal images (24 steps, slowest SDXL) |
| LIGHTNING | RealVisXL 4.0 Lightning | fast, lower quality (6 steps) |
| FLUX | FLUX.2 Klein 4B (SDNQ 4-bit) | highest-quality general images (can't edit) |

Resolution 720 or 1024. Editing an uploaded photo (📎) works with REALISTIC and LIGHTNING.

### Video - pick NO VIDEO / WAN 1.3B

| Model | Notes |
|---|---|
| Wan 2.1 1.3B + UMT5-XXL fp16 text encoder | ~2-second 576x320 clips, ~15 minutes each on this Mac |

### Voice

| Job | Model |
|---|---|
| Speech-to-text | `mlx-community/whisper-large-v3-turbo` (Hindi/English, Urdu guesses rewritten as Hindi) |
| Reply voice (default) | macOS `say`, voice Rishi - instant |
| Reply voice (optional) | IndicF5-Hinglish - slow (~10-20s per second of speech), best with Hindi script |

---

## 4. The tools Zira can use

| Tool | What it does | Where |
|---|---|---|
| `web_search` | DuckDuckGo search; real sources listed under the answer | `app/tools/web_search.py` |
| `play_music` / `control_music` | play/pause/stop songs from your `music/` folder (plays on the device you use) | `app/tools/music.py` |
| `create_pdf` | writes a PDF you can download | `app/tools/pdf.py` |
| `create_image` / `edit_image` | makes or edits images with the selected image model | `app/tools/image.py` |
| `create_video` | makes a short clip with Wan | `app/tools/video.py` |
| `project_overview`, `code_outline`, `list_directory`, `read_file`, `search_files` | read your project folders (read-only) | `app/tools/code_tools.py`, `filesystem.py` |
| `propose_edit` / `propose_create` | propose code changes you approve or reject (Edit mode) | `app/tools/changes.py` |
| `generate_postman_collection` | Postman collection from a project's API code | `app/tools/postman.py`, `api_scanner.py` |

The always-on minor-safety check lives inside the image and video tools (`app/tools/image_safety.py`).

---

## 5. What happens to one message

1. **Browser** sends the text (typed, or transcribed from your voice) over the WebSocket.
2. **Agent** (`app/agent/agent.py`) builds the prompt (`app/ai/context.py`):
   stable top (personality + rules + date + memories + summary) → recent history → a short per-turn note
   (time, knowledge, events) → your message. Keeping the top identical lets Ollama reuse its cached reading,
   so replies after the first take seconds instead of a minute.
3. **Routing** before the model decides: adult image requests go straight to `create_image`, video requests
   straight to `create_video`, explicit or time-sensitive questions get a web search (`app/agent/planner.py`).
4. **Model** streams the reply and may call tools (up to 6 rounds). Image/video progress streams live.
5. **Reply** shows in chat; a finished image/video is just the file (shown inline).
6. **After** the reply: facts about you are saved as memories; long conversations get summarised.
7. **Voice**: the reply is spoken, then the mic listens again (hands-free mode).

**Zira talks first:** in hands-free mode, after 4-10 quiet minutes she asks you something, praises you, or
brings up one of your interests; your answer is remembered like anything else.

**Self-learning:** when idle, she researches public topics from your interests (never your private info) and
uses what she learned in later answers.

---

## 6. API (all under `/api` unless noted)

| Area | Endpoints |
|---|---|
| Chat | `WS /ws/chat`, `POST /chat`, `POST /chat/stream`, `POST /chat/proactive`, `GET /conversations/{id}/messages`, `GET /conversations/{id}/checkpoint` |
| Memory | `GET /memories`, `DELETE /memories/{id}` |
| Knowledge | `GET /knowledge`, `DELETE /knowledge/{id}`, `POST /knowledge/research-now` |
| Model | `GET /health`, `POST /model/switch` |
| Images | `GET/POST /images/style`, `GET/POST /images/resolution`, `POST /images/upload` |
| Video | `GET/POST /videos/model` |
| Voice | `GET /voice/status`, `POST /voice/transcribe`, `POST /voice/speak`, `POST /voice/tts-voice`, `POST /voice/unload` |
| Music | `GET /music/local` (streams a file) |
| Files & changes | `GET /exports/{file}`, `GET /changes`, `GET /changes/{id}`, `POST /changes/{id}/approve`, `/reject`, `/undo` |
| App | `GET /capabilities` |

---

## 7. Where things are

| Path | Contents |
|---|---|
| `app/main.py` | builds the app: wires models, tools, memory, voice, routes |
| `app/config.py` | every setting (read from `.env`) |
| `app/agent/` | agent loop, planner, tool registry |
| `app/ai/` | Ollama client, prompt/personality, context builder, model manager |
| `app/api/` | HTTP/WebSocket endpoints |
| `app/tools/` | the tools (section 4) |
| `app/memory/` | database, conversations, memories, extractor, summaries |
| `app/knowledge/` | knowledge store and background researcher |
| `app/voice/` | speech-to-text, text-to-speech, text cleaning |
| `frontend/` | `index.html`, `app.js`, `style.css`, `mobile-menu.js` (phone menu) |
| `third_party/indicf5/` | IndicF5 inference code, sample voices, worker process |
| `data/` | `jarvis.db`, `exports/` (your images, videos, PDFs), `uploads/` |
| `music/` | your songs |
| `logs/` | `server.log`, `ollama.log`, `indicf5.log`, older/crash logs |
| `tests/` | ~890 automated tests |
| `run.sh` / `setup.sh` | start the server / first-time setup |
| `.env` | your settings (not shared) |

---

## 8. Running it

| Task | How |
|---|---|
| Start | `./run.sh` (also starts Ollama if needed; restarts itself after a crash) |
| Stop | `pkill -TERM -f app.main` (run.sh stops too) |
| Restart | stop, then `nohup ./run.sh >> logs/server.log 2>&1 &` |
| Tests | `.venv/bin/python -m pytest -q` |
| Logs | `logs/server.log` (app), `logs/ollama.log` (models) |

Needs a restart: any Python change, `.env` change, or chat-mode switch (done automatically).
No restart needed: frontend changes (served live), image/video/voice model switches, resolution.
At startup the image and video models are always NONE and the reply voice is Rishi.

### Addresses

| From | URL |
|---|---|
| This Mac | http://127.0.0.1:8000 |
| Same Wi-Fi | http://192.168.1.2:8000 |
| Anywhere on your Tailscale network | http://100.105.27.73:8000 |
| iPhone with microphone (HTTPS) | https://zira-mac.tail4c16b2.ts.net |

Private only: Tailscale network, no port forwarding or public tunnel. Microphone on the phone needs the HTTPS
address.

---

## 9. Main settings (`.env`)

| Setting | Current / default | Meaning |
|---|---|---|
| `MODEL_MODE` | balanced (default deep) | chat mode at startup |
| `LIGHT_MODEL`, `BALANCED_MODEL`, `DEEP_MODEL` | qwen3-heretic:4b, qwen3-heretic:8b, qwen3:8b | model per mode |
| `DEEP_THINK` | true | DEEP reasons before replying |
| `HOST`, `PORT` | 0.0.0.0, 8000 | listen on all addresses (needed for phone access) |
| `IMAGE_GENERATION_ENABLED`, `IMAGE_RESOLUTION` | true, 720 | images on; default size |
| `VIDEO_GENERATION_ENABLED` | true | video on (`VIDEO_WIDTH/HEIGHT/FRAMES/FPS/STEPS` for size) |
| `STT_PROVIDER`, `STT_MODEL` | mlx-whisper, whisper-large-v3-turbo | speech-to-text |
| `TTS_PROVIDER`, `TTS_VOICE` | say, Rishi | default voice |
| `TTS_INDICF5_ENABLED` | true | offer the IndicF5 voice switch |
| `PROACTIVE_ENABLED` | true | Zira talks first in hands-free mode |
| `FILE_ACCESS_ROOTS`, `FILE_WRITE_ROOTS` | your folders | what Zira may read / propose changes to |
| `MUSIC_LIBRARY_ROOTS` | music/ | where songs are |

Full list with explanations: [.env.example](.env.example).

---

## 10. Known limits

- 16GB is the constraint: one chat model plus one image/video model at a time. Very large uploaded photos
  (3+ megapixels) for editing once froze the Mac - resize to about 1000px first.
- DEEP with thinking is slow (tens of seconds). The first message of a conversation reads the whole prompt once.
- Video is slow (~15 minutes for 2 seconds) and small (576x320). IndicF5 voice is slow and needs Hindi script.
- Models do not retrain themselves; Zira improves through memory, learned instructions and researched knowledge.
- Not built: generating images of real people from their photos (declined), camera/vision, the robot link.
- `requirements.txt` lists only the core packages; torch, diffusers, transformers, accelerate, sdnq, pillow,
  psutil and the `.venv-tts` packages were installed separately (versions in section 2 and README).

## Updates 2026-09-28 (latest state – see HANDOFF.md and LORA.md)
- **Video:** HunyuanVideo 1.5 480p image-to-video (4-bit, worker `third_party/hunyuan/worker.py`) replaced FastMetal 5B and is the default. Text→video starts from a first frame made by the image model. ~1h+ per 5s clip. LTX and FastMetal 1.3B remain.
- **Automatic models:** image/video models switch on by themselves (REALISTIC / Hunyuan), only one loaded at a time, idle 30 min → None, fallback to None on failure, ⟲ Reset button (`/api/system/reset`).
- **Songs:** ACE-Step 1.5 (`app/tools/song.py`); "sing …" always makes a song, saved songs replay, auto-play, gallery Audio tab.
- **Ads:** `app/tools/ad.py` – scenes + captions + Kokoro voice-over + jingle in one mp4.
- **Maths:** `run_python` in a macOS sandbox (`app/tools/python_runner.py`); maths questions are forced to it.
- **Chat while an image is made**, Stop for images/videos/songs/ads (button, typed or spoken "stop"/"ruk jao").
- **Voice:** Kokoro TTS streamed per sentence; emojis are no longer spoken.
- **LoRA training:** separate LoRA Studio app (see LORA.md); Zira does not load trained LoRAs yet.
- **Pending:** see the "Pending / next" list in HANDOFF.md.
