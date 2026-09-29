# Zira – handoff notes (2026-09-28)

Local AI assistant on an M4 16GB Mac. FastAPI server + Ollama + diffusers. Used from iPhone.
Folder: `/Users/rehanali/worksapce/AI/jarvis-ai`. More detail: README.md, PROJECT_OVERVIEW.md, CAPABILITIES.md.

## Run / restart
- Runs under launchd (`~/Library/LaunchAgents/com.zira.server.plist` → `run.sh` → `python -m app.main`). Never run `./run.sh` by hand, never edit run.sh while running.
- Restart: `touch data/.restart_pending && kill -TERM $(pgrep -f app.main)` → back in ~4s.
- Before restarting, check `logs/server.log` for recent chat/generation (user may be mid-use).
- Check: `curl http://127.0.0.1:8000/api/health` and `https://zira-mac.priv.ziraai.in/api/health` (Headscale private network).
- Tests: `.venv/bin/python -m pytest tests -q` (1257 pass). Python 3.14 in `.venv`.
- Settings: `.env` (read by `app/config.py`).

## Main pieces
| Area | Code | How it works |
|---|---|---|
| Chat agent | `app/agent/agent.py`, `app/agent/planner.py` | Qwen via Ollama. Planner forces tools for clear intents (web search, song, ad, maths) because the 4B model misses tool calls. |
| Models | `app/ai/llm.py`, `app/api/model.py` | Modes LIGHT/NEWLIGHT(qwen3.5-heretic:4b, current)/BALANCED/DEEP. Auto-thinking: `app/ai/thinking.py`. |
| Prompts | `app/ai/prompts.py`, `app/ai/context.py` | Stable system prompt (prompt caching), per-turn notes. |
| Memory / learning | `app/memory/`, `app/learning/`, `app/knowledge/` | SQLite `data/jarvis.db`; nightly self-study; background research. |
| Voice | `app/voice/`, `app/api/voice.py` | STT mlx-whisper (hi/en). TTS Kokoro worker (`third_party/kokoro`, `.venv-kokoro`), streamed per sentence. |
| Images | `app/tools/image.py`, `app/api/images.py` | SDXL: REALISTIC (Perfection), LIGHTNING, REALVIS5. Stop, auto switch-on, fallback to None on failure. Minor-safety check always on (`image_safety.py`) – keep it. |
| Video | `app/tools/video.py`, `video_hunyuan.py`, `video_fastmetal.py`, `video_ltx.py`, `app/api/videos.py` | Default **HunyuanVideo 1.5 480p I2V** (4-bit, worker `third_party/hunyuan/worker.py`, models `models/hunyuan15*`). Also LTX, FastMetal 1.3B. Text→video: image model makes first frame. ~1h+ per 5s clip. |
| Songs | `app/tools/song.py`, `third_party/song/worker.py` | ACE-Step 1.5 in `.venv-acestep` (`third_party/ace-step`). Saved song replays; "create" makes new. Auto-plays. |
| Ads | `app/tools/ad.py` | Script (planner) → Kokoro voice-over → scenes (video tool) → jingle → ffmpeg join + PIL captions. |
| Maths | `app/tools/python_runner.py` | `run_python` in macOS `sandbox-exec` (no network/writes, 10s). |
| Model memory mgmt | `app/main.py` (`model_idle_loop`), `app/api/system.py` | One of image/video loaded at a time; idle 30 min → None; ⟲ Reset button. |
| Chat while image | `app/api/chat.py` | Image turns run in background; `turn_id` on events. |
| Gallery | `app/memory/media_store.py`, `app/api/media.py` | Images, Videos, Audio tabs. |
| Frontend | `frontend/index.html`, `app.js`, `style.css`, `mobile-menu.js` | Served by FastAPI. |
| Other | reminders, calendar, morning brief, backups (`~/ZiraBackups`), push notifications, LoRA Studio | see README. |

Logs: `logs/server.log`, `logs/song.log`, `logs/hunyuan.log`, `logs/fastmetal.log`, `logs/kokoro.log`.

## Rules the user set
- No extra content filters. Only the existing minor check stays.
- Don't restart while the user is chatting/generating; verify both URLs after.
- Don't run GPU tests during a generation.
- App name is **Ziraago** (hyperlocal app).

## Pending / next
1. First full-length HunyuanVideo clip (only a 0.7s test ran: 11.5 min). Run one supervised; watch swap/disk. If too slow or OOM: fewer frames per clip or larger VAE tiles.
2. README still describes FastMetal 5B in its video section – update.
3. Optional: delete `models/hunyuan15/transformer` + `text_encoder` (31GB bf16, only needed to re-run `third_party/hunyuan/convert.py`).
4. Ad planner once returned no scenes – log now prints the reply tail; check `Ad planning returned no scenes`.
5. Proactive questions never fired (need 10+ min quiet with mic on).
6. Kokoro Hindi voice quality is weak.
7. Faster STT: skip Whisper's second pass when it guesses a wrong language (needs a timing test).
8. Feature ideas not started: send messages (WhatsApp/SMS/email), Mac control, tell users apart (voice/device), live translator, transcribe audio/video files, video editing, smart home, live camera.

LoRA training: see LORA.md (Studio app, how training works, and the pending "load LoRA in Zira" step).
