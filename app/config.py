"""Application configuration, loaded from environment variables / .env."""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")


def _local_lan_ip() -> str | None:
    """Best-effort real LAN IP (e.g. for a phone on the same WiFi to reach HOST=0.0.0.0). A UDP
    "connect" never actually sends a packet - it just asks the OS which local interface/address
    would be used to reach that destination, so this works offline and needs no real connectivity.
    Never raises: no network is a real, common case (this machine off WiFi) - it just means no LAN
    IP gets allowlisted below, not a startup crash."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()


def _tailscale_ip() -> str | None:
    """This machine's Tailscale IP (100.x.y.z), if the `tailscale` CLI is installed and this
    machine is actually connected to a tailnet - for remote access (e.g. a phone off-WiFi, over
    Tailscale) to reach HOST=0.0.0.0 the same way _local_lan_ip() covers same-WiFi access.
    Deliberately a separate check, not folded into _local_lan_ip(): that one finds the interface the
    OS would use to reach the public internet, which is the WiFi/ethernet interface, not Tailscale's
    - Tailscale is never the default route in this setup (no exit-node routing here), so it would
    never be found that way. `tailscale ip -4` asks Tailscale directly instead of guessing from
    routing. Never raises: Tailscale not installed, not running, or not logged in are all real, easy
    states to be in - each just means no Tailscale IP gets allowlisted, not a startup crash."""
    if shutil.which("tailscale") is None:
        return None
    try:
        result = subprocess.run(["tailscale", "ip", "-4"], capture_output=True, text=True, timeout=2)
    except (OSError, subprocess.TimeoutExpired):
        return None
    ip = result.stdout.strip().splitlines()[0].strip() if result.stdout.strip() else ""
    return ip or None


def _tailscale_dns_name() -> str | None:
    """This machine's Tailscale MagicDNS name (e.g. rehans-macbook-air.tail4c16b2.ts.net), from
    `tailscale status --json`. Separate from _tailscale_ip() because the two are reached
    differently: `tailscale serve` puts HTTPS in front of JARVIS on this *name*, so requests through
    it carry the name as their Host/Origin, not the 100.x IP. HTTPS matters because a phone's browser
    only exposes the microphone (getUserMedia) on a secure context - plain http://100.x.y.z:8000
    doesn't qualify, so voice input from the phone needs this path. Never raises, same contract as
    _tailscale_ip(): not installed / not logged in / unexpected output all just mean no name."""
    if shutil.which("tailscale") is None:
        return None
    try:
        result = subprocess.run(["tailscale", "status", "--json"], capture_output=True, text=True, timeout=3)
        name = json.loads(result.stdout)["Self"]["DNSName"]
    except (OSError, subprocess.TimeoutExpired, ValueError, KeyError, TypeError):
        return None
    name = str(name).strip().rstrip(".")
    return name or None


def derive_model_label(tag: str) -> str:
    """'qwen3:8b' -> 'Qwen3 8B'; 'llama3.1:70b-instruct' -> 'Llama3.1 70B Instruct'."""
    name, _, variant = tag.partition(":")
    parts = [name.replace("-", " ").replace("_", " ").title()]
    if variant and variant != "latest":
        parts.extend(re.split(r"[-_]", variant))
    return " ".join(p.upper() if re.fullmatch(r"\d+(\.\d+)?[bm]", p, re.I) else p.title() if p.islower() else p for p in parts)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # LLM
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen3:8b"  # deprecated: kept for backward compatibility, see DEEP_MODEL below
    ollama_num_ctx: int = 8192
    ollama_temperature: float = 0.7
    ollama_keep_alive: str = "30m"
    ollama_timeout: float = 120.0
    ollama_think: bool = False
    # Qwen3 "thinks" (reasons privately) before DEEP's actual chat replies - by user choice, for better
    # answers in real work at the cost of time (measured earlier: a simple question went from ~5s to
    # 84s with thinking on). Only the streamed reply thinks: the small background calls (search query,
    # memory extraction, summaries) stay fast, and LIGHT/BALANCED never think.
    deep_think: bool = True
    model_label: str | None = None

    # Model mode: only one model is ever resident in Ollama at a time (16GB-RAM-friendly). LIGHT is
    # for realtime/voice/casual chat (fast, low latency); DEEP is for coding/reasoning/tools/complex
    # work; BALANCED (Qwen3-8B-heretic, see below) sits between them. Switching modes is a
    # full process restart (app/api/model.py, app/restart.py) - never two models loaded together,
    # never an in-process hot-swap.
    model_mode: Literal["light", "newlight", "balanced", "deep"] = "deep"  # unchanged default behavior for existing installs
    # DreamFast/qwen3-4b-heretic (Qwen3 4B, refusals removed with Heretic: 3/100 vs 100/100 for the
    # original), its Q4_K_M GGUF (2.5GB) registered in Ollama under this name with qwen3:8b's chat
    # template. Replaced gemma4:e4b (9.6GB): ~4x smaller and ~2x faster (18.6 vs ~9 tokens/s measured).
    light_model: str = "qwen3-heretic:4b"
    # jbeslt/Qwen3-8B-heretic (Qwen3 8B with its refusal behaviour removed by the Heretic tool). The repo
    # only ships full-size safetensors, so it was converted locally with llama.cpp to Q4_K_M (4.8GB) and
    # registered in Ollama under this name, reusing qwen3:8b's chat template - see README "Model mode".
    balanced_model: str = "qwen3-heretic:8b"
    deep_model: str = "qwen3:8b"
    # NEWLIGHT (added 2026-09-27 by user request, next to LIGHT so the two can be compared before one goes):
    # mradermacher/Qwen3.5-4B-heretic-GGUF Q4_K_M (Qwen3.5 4B, Feb 2026, refusals removed with Heretic), registered
    # in Ollama under this name. Runs like LIGHT: no thinking, short voice replies.
    newlight_model: str = "qwen3.5-heretic:4b"
    light_voice_num_predict: int = Field(default=200, ge=16, le=2000)  # caps LIGHT's voice-turn reply length
    # NEWLIGHT thinks first on hard questions only - maths, logic, code, planning, why/explain (app/ai/thinking.py).
    # A thought longer than the budget (characters, ~4 per token) is cut off and answered from the notes so far,
    # so a reply can never hang on a runaway thought.
    # Reminders and timers (app/reminders.py): phone notification + Mac notification + a line in the chat.
    reminders_enabled: bool = True
    reminders_mac_notification: bool = True
    # Apple Calendar through the helper app bin/ZiraCalendar.app (built by third_party/calendar/build.sh).
    calendar_enabled: bool = True
    # Morning brief (app/brief.py): calendar + reminders + weather for BRIEF_CITY (Open-Meteo; empty = no weather).
    brief_enabled: bool = True
    brief_time: str = "08:00"
    brief_city: str = ""
    brief_name: str = ""  # how the brief greets you, e.g. "Rehan"
    # Backups (app/backup.py): nightly to BACKUP_DIR, then encrypted with restic to your server if configured.
    backup_enabled: bool = True
    backup_time: str = "03:30"
    backup_dir: str = str(Path.home() / "ZiraBackups")
    backup_restic_repository: str = ""  # e.g. sftp:zira-backup:/repo (see scripts/backup_server_setup.sh)
    backup_restic_password_file: str = str(Path.home() / ".config" / "zira" / "restic-password")
    # Learning every day (app/learning/): a nightly study of the day's real questions (thinking on), whose
    # careful answers are used the next time a question like it is asked; memories are also found by meaning.
    # Needs the embedding model: ollama pull qwen3-embedding:0.6b (~0.6 GB). Without it Zira works as before.
    learning_enabled: bool = True
    embedding_model: str = "qwen3-embedding:0.6b"
    study_start_hour: int = Field(default=2, ge=0, le=23)  # local time; the study only runs in this window
    study_end_hour: int = Field(default=6, ge=0, le=23)
    study_min_idle_minutes: float = Field(default=10.0, ge=0)  # and only after this long without a chat
    study_max_questions: int = Field(default=20, ge=1, le=500)  # per night (~2-3 min each)
    study_notify: bool = True  # a phone notification after a study (when the phone signed up for them)
    # A stored question this alike becomes a candidate; it is only reused if the numbers match and the chat model
    # agrees it asks the same thing (app/learning/recall.py - the score alone proved unreliable).
    learned_match_threshold: float = Field(default=0.5, gt=0, le=1)
    memory_semantic_threshold: float = Field(default=0.55, gt=0, le=1)
    newlight_auto_think: bool = True
    newlight_think_budget_chars: int = Field(default=6000, ge=500, le=60000)

    @model_validator(mode="after")
    def _migrate_legacy_ollama_model(self) -> "Settings":
        """A pre-existing OLLAMA_MODEL=<custom tag> in .env keeps working as DEEP_MODEL without any
        edit required - only when DEEP_MODEL itself was not explicitly set, so DEEP_MODEL always wins
        if both are present."""
        if "ollama_model" in self.model_fields_set and "deep_model" not in self.model_fields_set:
            self.deep_model = self.ollama_model
        return self

    def model_for_mode(self, mode: str) -> str:
        if mode == "light":
            return self.light_model
        if mode == "newlight":
            return self.newlight_model
        return self.balanced_model if mode == "balanced" else self.deep_model

    @property
    def active_model(self) -> str:
        return self.model_for_mode(self.model_mode)

    # Storage
    database_path: str = "data/jarvis.db"

    # Server
    host: str = "127.0.0.1"
    port: int = 8000

    # Personality
    assistant_name: str = "Zira"
    personality_style: str = "excited, warm, playful, confident, honest"
    personality_extra: str = ""

    # Automatic memory extraction (LLM decides what facts about the user are worth keeping)
    auto_memory: bool = True
    auto_memory_max_items: int = Field(default=3, ge=1, le=10)

    # Web search fallback (the model decides when it needs the internet)
    web_search_enabled: bool = True
    web_search_max_results: int = Field(default=5, ge=1, le=10)
    web_search_timeout: float = Field(default=8.0, gt=0)

    # Read-only project access (opt-in). Comma-separated absolute folders JARVIS may read.
    # Empty = the file tools are not enabled at all.
    file_access_roots: str = ""
    # Folders where JARVIS may *propose* edits. Nothing is ever applied without your approval.
    file_write_roots: str = ""
    file_read_max_chars: int = Field(default=5000, ge=500, le=20000)
    exports_dir: str = "data/exports"

    # Voice (opt-in). "unavailable" (default) keeps voice off; everything below is only read then.
    stt_provider: str = "unavailable"
    stt_model: str = "small"
    stt_language: str = ""  # empty = auto-detect (best default for code-switched Hinglish)
    # Restricts auto-detect to just these languages (comma-separated codes), picking whichever
    # scores highest among them - unrestricted Whisper auto-detect spans ~100 languages and can
    # drift into acoustically-similar ones (Urdu, Arabic sound close to Hindi) on short/accented
    # clips. Only applies when STT_LANGUAGE is empty (auto-detect); ignored if a language is
    # forced. Empty = no restriction, full auto-detect. See README "Voice" for how this was
    # measured (a real Hindi clip scored hi=0.72 but ur=0.12, its actual runner-up).
    stt_language_candidates: str = "hi,en"
    # Text the speech model is primed with before it decodes ("initial prompt"). The wake word is matched
    # against Whisper's transcript, and Whisper spells a rare name many different ways - measured for
    # real with scripts/wake_word_benchmark.py (macOS voices through mlx-whisper large-v3-turbo); the
    # results are recorded in README "Voice". Unset (default) primes with "Hello <ASSISTANT_NAME>.";
    # set STT_INITIAL_PROMPT to your own text, or to an empty value to turn priming off. Only the
    # mlx-whisper provider uses it.
    stt_initial_prompt: str | None = None
    tts_provider: str = "unavailable"
    tts_voice: str = "Rishi"  # male, Indian-accented English (see README "Voice" for alternatives)
    tts_rate: int = 0  # 0 = provider default words-per-minute
    qwen3_tts_model: str = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit"
    qwen3_tts_adapter: str = "akashicmarga/qwen3-tts-hindi-lora-grpo"
    qwen3_tts_voice: str = "Hindi GRPO"
    qwen3_tts_timeout: float = Field(default=120.0, gt=0)
    # Optional second voice: IndicF5-Hinglish (voice cloning, Hindi-script text sounds best). Runs in its
    # own process from the separate `.venv-tts` environment (see third_party/indicf5/NOTICE.md). When
    # enabled, the UI can switch between the `say` voice and IndicF5 at runtime; startup is always the
    # `say` voice and IndicF5 is only loaded while selected. Measured on a 16GB M4 with the LLM loaded:
    # 10-20s of compute per second of speech (3.3GB while loaded), so it is an opt-in voice, not the default.
    tts_indicf5_enabled: bool = False
    tts_indicf5_python: str = str(PROJECT_ROOT / ".venv-tts" / "bin" / "python")
    tts_indicf5_voice: str = "MAR_M_WIKI_00001"  # a clip in third_party/indicf5/voices/voices.json
    tts_indicf5_steps: int = Field(default=16, ge=4, le=64)  # flow-matching steps: 16 fast, 32 better
    tts_indicf5_timeout: float = Field(default=300.0, gt=0)
    # Kokoro TTS (TTS_PROVIDER=kokoro, 2026-09-28): Kokoro-82M in its own `.venv-kokoro` (Python 3.11) worker,
    # loaded once at startup. Voices are Kokoro's own IDs (hexgrad/Kokoro-82M voices/): hf_alpha / hf_beta are
    # its Indian (Hindi) female voices, hm_omega / hm_psi male; af_heart is its best-rated (American) voice.
    # KOKORO_LANG is the G2P for English chunks ("a" American, "b" British); Hinglish chunks use Kokoro's Hindi
    # G2P (espeak-ng hi) after their Hindi words are written in Devanagari (app/voice/hinglish.py).
    kokoro_python: str = str(PROJECT_ROOT / ".venv-kokoro" / "bin" / "python")
    kokoro_voice: str = "hf_alpha"
    kokoro_speed: float = Field(default=1.0, gt=0.3, le=2.5)
    kokoro_lang: Literal["a", "b"] = "a"
    kokoro_device: Literal["cpu", "mps"] = "cpu"  # the CPU keeps the GPU free for Qwen (see README "Voice")
    kokoro_hinglish: bool = True
    kokoro_timeout: float = Field(default=60.0, gt=0)
    # Streaming speech: a voice reply is spoken sentence by sentence while Qwen is still writing it.
    tts_streaming: bool = True
    tts_min_chunk_length: int = Field(default=20, ge=1, le=500)  # shorter sentences are joined to the next
    tts_max_chunk_length: int = Field(default=250, ge=40, le=2000)  # longer ones split at a comma/space
    tts_first_chunk_length: int = Field(default=60, ge=10, le=500)  # the first chunk may end at a comma
    voice_max_audio_bytes: int = Field(default=15_000_000, gt=0)
    voice_max_speech_chars: int = Field(default=4000, ge=1)

    # Voice transcript cleanup: an extra LLM call that fixes STT disfluencies/dropped words/obvious
    # mis-transcriptions before the text is sent through chat as if typed. Never translates, never
    # changes script, never invents content (see app/voice/transcript_cleanup.py). Was on by default
    # while STT used faster-whisper's "small" model, whose rawer transcripts benefited from it. Now
    # off by default: measured for real on a live server (mlx-whisper/large-v3-turbo, already much
    # cleaner raw output) that this call can take 9-30+ seconds under real system load (running
    # STT+LLM+everything else concurrently on 16GB unified memory), turning "recognition is good but
    # late" into the dominant complaint - a real user report, not a guess. The benefit shrank
    # (better STT needs less cleanup) while the cost didn't, so the tradeoff flipped. Opt back in
    # with VOICE_TRANSCRIPT_CLEANUP_ENABLED=true if you still see disfluencies worth fixing.
    voice_transcript_cleanup_enabled: bool = False
    voice_transcript_cleanup_timeout: float = Field(default=12.0, gt=0)

    # Context-window checkpointing: once a conversation's estimated token count reaches this
    # fraction of OLLAMA_NUM_CTX, older history is compacted into a markdown file (data/checkpoints/)
    # so the chat/coding session can keep going instead of silently losing older messages.
    context_checkpoint_enabled: bool = True
    context_checkpoint_threshold: float = Field(default=0.75, gt=0, le=1)
    checkpoints_dir: str = "data/checkpoints"

    # Self-learning: while idle, research one topic drawn from long-term memory and cache a
    # summary locally (data: the `knowledge` table) so future questions can be answered from that
    # cache instead of searching the internet again. On by default - unlike web_search (only ever
    # called from a live user turn), this contacts the internet on its own; see README
    # "Self-learning" for what that means and how to turn it off.
    knowledge_learning_enabled: bool = True
    knowledge_idle_minutes: float = Field(default=15.0, gt=0)  # no chat activity for this long before a cycle may run
    knowledge_research_cooldown_minutes: float = Field(default=60.0, gt=0)  # min gap between two research cycles
    knowledge_topic_cooldown_days: float = Field(default=7.0, gt=0)  # don't re-research the same topic sooner than this
    knowledge_max_results_per_topic: int = Field(default=4, ge=1, le=10)
    knowledge_context_top_k: int = Field(default=3, ge=0)  # learned entries shown in context per turn

    # Music player: search + play songs from your own local folder and/or your own server. No
    # external catalog search (no YouTube/Jamendo/etc.) - every track is something you already
    # have. At least one of these two must be set for play_music/control_music to be registered.
    music_library_roots: str = ""  # comma-separated local folders JARVIS may search for music
    music_remote_index_url: str = ""  # URL to a JSON manifest of your own hosted tracks (see README)
    music_remote_index_cache_seconds: float = Field(default=300.0, gt=0)

    # Image generation: local, offline text-to-image via diffusers on PyTorch's Apple Silicon GPU
    # backend (MPS). Off by default - unlike create_pdf (a small library), this needs a real model
    # download and meaningful compute/memory per image, so it's opt-in like music/file access rather
    # than unconditional. mflux (MLX-native) was tried first but its z-image-turbo support hit a
    # real, reproduced weight-loading bug (see CAPABILITIES.md "Image generation"); diffusers +
    # stabilityai/sd-turbo worked correctly on the first real attempt. See README "Image generation"
    # for real measured numbers.
    image_generation_enabled: bool = False
    # 300s (sd-turbo's ~2GB first download) proved too tight once bigger checkpoints arrived - a real
    # first-use request once timed out at ~307s. 900s covered RealVisXL/Pony's ~6.9GB fine, but FLUX.2
    # Klein 4B's (since removed) real measured first-use cost (download + load + one generation) was 943.9s - over
    # 900s - so this now has real headroom above the biggest real number measured so far, not just a
    # round number. Bump further in .env if your connection is slower than what was measured here.
    image_generation_timeout: float = Field(default=1200.0, gt=0)

    # Image style: which checkpoint create_image uses - REALISTIC (Perfection Realistic ILXL, an
    # SDXL/Illustrious XL checkpoint for photorealism), LIGHTNING (RealVisXL V4.0 Lightning) and REALVIS5
    # (RealVisXL V5.0, the full non-distilled model; replaced FLUX.2 Klein 4B on 2026-09-28 at the user's request) are
    # switchable at runtime with no restart, unlike LLM model_mode: ImagePipelines lives in this same
    # process (not a separate server like Ollama), so dropping the loaded pipeline +
    # torch.mps.empty_cache() really does free the memory - confirmed for real (an allocated 8GB MPS
    # tensor measured via torch.mps.current_allocated_memory() went to exactly 0 after del +
    # empty_cache). Only one style loaded at a time, same one-model-resident spirit as model_mode,
    # for the same 16GB-RAM reason.
    #
    # A third style, ANIME (Pony Diffusion V6 XL), was built and shipped, then removed - the user
    # reported it "not working good" (matches this session's own earlier real finding: an intermittent
    # fp16 VAE overflow producing incoherent images on some prompts). FLUX.2 Klein replaces it, not
    # as another "style" in the illustrated/anime sense - it's a different general-purpose
    # photorealistic model, so the realistic/anime *stylistic* dichotomy this field's name still
    # implies no longer strictly holds; kept as `image_style` regardless since renaming the field
    # would touch the API/frontend for no functional benefit.
    # "none" is the default and the startup state: no image model is selected or loaded (image
    # generation is off until the user picks one, which frees that memory for the LLM/macOS). The
    # UI selection is runtime-only - it is deliberately not written back to .env, so a restart
    # always comes back up with nothing loaded. Setting IMAGE_STYLE here still selects a style at
    # startup, but it loads lazily on the first generation, never at startup.
    image_style: Literal["none", "realistic", "lightning", "realvis5"] = "none"
    # Asking for an image while the image model is None switches this one on first (user request, 2026-09-28):
    # the same switch as the buttons, so loading/unloading is unchanged. A model already picked is never changed.
    # "off" keeps the old behaviour (the request is refused until a model is picked).
    image_auto_style: Literal["off", "realistic", "lightning", "realvis5"] = "realistic"
    # An image or video model unused this long goes back to None, freeing its memory (user request, 2026-09-28); it is
    # remembered, so the next request switches the same model on again. 0 = never.
    model_idle_minutes: float = Field(default=30.0, ge=0.0, le=1440.0)
    # Perfection Realistic ILXL (by 6tZ; Illustrious XL fine-tune, SDXL architecture). Loaded from
    # John6666's diffusers-format conversion of the CivitAI checkpoint, its newest version (4.2) -
    # checked against the real repo (HF API), not assumed: public/ungated, safetensors only (no
    # pickle), ~6.9GB, and already fp16 (UNet 5.1GB), so it fits in 16GB like the other SDXL style.
    # Licence is faipl-1.0-sd (see the model card); the card tags it not-for-all-audiences.
    image_style_realistic_model: str = "John6666/perfection-realistic-ilxl-illustrious-xl-nsfw-sfw-checkpoint-42-sdxl"
    # RealVisXL V4.0 Lightning - the same author's newer SDXL photorealism model, distilled with
    # SDXL-Lightning for few-step generation. Checked against the real repo (HF API + model card),
    # not assumed: public/ungated, diffusers format with fp16 variant files
    # (loaded with variant="fp16"), and the card recommends 4+ steps and CFG 1.0-2.0
    # with a DPM++ SDE sampler.
    image_style_lightning_model: str = "SG161222/RealVisXL_V4.0_Lightning"
    # RealVisXL V5.0 - SG161222's newest full (non-Lightning) SDXL photorealism model. Checked against the real
    # repo (HF API): public/ungated, openrail++, diffusers format with fp16 variant files (~6.9GB to download).
    # FLUX.2 Klein 4B was this third style until 2026-09-28; the user asked for it to be replaced by this.
    image_style_realvis5_model: str = "SG161222/RealVisXL_V5.0"
    # dtype/variant differ by checkpoint, not just a style preference - confirmed for real, not
    # assumed from convention: RealVisXL Lightning ships fp16-variant files (AutoPipelineForText2Image
    # needs dtype=float16, variant="fp16" to use them); the Perfection Realistic repo ships its fp16
    # weights as the plain default files (no "fp16" variant subset - variant="fp16" would fail to find
    # them), so it takes dtype=float16 with variant=None; RealVisXL V5.0 ships fp16-variant files like V4.0 Lightning.
    image_style_realistic_dtype: str = "float16"
    image_style_realistic_variant: str | None = None
    image_style_lightning_dtype: str = "float16"
    image_style_lightning_variant: str | None = "fp16"
    image_style_realvis5_dtype: str = "float16"
    image_style_realvis5_variant: str | None = "fp16"
    # steps/guidance_scale are real generation hyperparameters, not interchangeable across models -
    # a real, caught bug: the old single image_generation_steps default (1) and a hardcoded
    # guidance_scale=0.0 were both calibrated specifically for sd-turbo's adversarial-diffusion-
    # distillation design (true single-step, CFG-free), and steps=1 produced pure unconverged noise
    # (no dog, no coherent structure at all) on a live request when reused on an SDXL checkpoint.
    # REALISTIC (Perfection Realistic ILXL) is a regular, non-distilled SDXL/Illustrious checkpoint
    # whose repo ships an Euler Ancestral scheduler, so it needs a real step count and classifier-free
    # guidance (CFG only runs above 1.0): 24 steps / CFG 5.0 sits in the usual range for this
    # checkpoint family. That makes it the slowest SDXL style per image (CFG doubles the UNet work per
    # step) - real measured numbers are in README "Image style". FLUX.2 Klein's own real-tested values:
    # steps=4, guidance=1.0 (guidance<=1 keeps CFG off, confirmed via the pipeline's own
    # do_classifier_free_guidance property) produced a genuinely good result in direct testing too -
    # real, but noticeably slower per step than the SDXL styles on this hardware (SDNQ's
    # quantized-matmul speedup needs Triton, unavailable on MPS, so it dequantizes every step without
    # the compensating benefit - see README "Image generation").
    image_style_realistic_steps: int = Field(default=24, ge=1, le=50)
    image_style_realistic_guidance: float = Field(default=5.0, ge=0)
    # Lightning: the model card's own recommended range is 4+ steps and CFG 1.0-2.0. Guidance stays
    # at 1.0 (the bottom of that range) because SDXL only runs classifier-free guidance above 1 -
    # measured earlier this session, CFG roughly doubles the cost of every step, and this machine is
    # already memory-bound. Steps: see the real test result recorded in README "Image style".
    image_style_lightning_steps: int = Field(default=6, ge=1, le=50)
    image_style_lightning_guidance: float = Field(default=1.0, ge=0)
    # RealVisXL V5.0: its model card recommends DPM++ 2M Karras or DPM++ SDE Karras (30+ steps); it is loaded with
    # DPM++ 2M Karras (see _KARRAS_STYLES in app/tools/image.py), which converges well by ~25 steps. CFG 5.0 as
    # for REALISTIC; like REALISTIC it costs about as much per image (a full SDXL model with CFG).
    image_style_realvis5_steps: int = Field(default=25, ge=1, le=50)
    image_style_realvis5_guidance: float = Field(default=5.0, ge=0)
    # IP-Adapter FaceID (app/tools/face_id.py): create_image's face_image_id - a new image of the person in an
    # uploaded photo, with any SDXL style. The scale is how strongly the face steers the image (0-1).
    image_faceid_enabled: bool = True
    image_faceid_adapter_file: str = "ip-adapter-faceid_sdxl.bin"
    image_faceid_scale: float = Field(default=0.8, ge=0, le=1.5)
    # Inpainting (app/tools/inpaint.py): edit_image's `area` repaints only that part of the photo, found from the
    # words with CLIPSeg (CIDAS/clipseg-rd64-refined, ~600MB, CPU).
    image_inpaint_enabled: bool = True

    def image_model_for_style(self, style: str) -> tuple[str, str, str | None]:
        """Returns (model_ref, dtype, variant) for AutoPipelineForText2Image.from_pretrained().
        "none" has no model (nothing is ever loaded for it)."""
        if style == "none":
            return "", "float16", None
        if style == "realvis5":
            return self.image_style_realvis5_model, self.image_style_realvis5_dtype, self.image_style_realvis5_variant
        if style == "lightning":
            return (
                self.image_style_lightning_model,
                self.image_style_lightning_dtype,
                self.image_style_lightning_variant,
            )
        return self.image_style_realistic_model, self.image_style_realistic_dtype, self.image_style_realistic_variant

    def image_generation_params_for_style(self, style: str) -> tuple[int, float]:
        """Returns (steps, guidance_scale) for create_image/edit_image - see the real, caught
        bug documented above image_style_realistic_steps for why these must be per-style, not a
        single shared default."""
        if style == "realvis5":
            return self.image_style_realvis5_steps, self.image_style_realvis5_guidance
        if style == "lightning":
            return self.image_style_lightning_steps, self.image_style_lightning_guidance
        return self.image_style_realistic_steps, self.image_style_realistic_guidance

    # Square height=width, generated at - independent of style (unlike steps/guidance above), and
    # runtime-switchable via POST /api/images/resolution (a button in the UI, no restart, no model
    # reload - see ImagePipelines.resolution) rather than baked per-style, since the user asked for a
    # simple "pick before the next command" control, not different defaults per style. Was previously
    # unset entirely - neither tool ever passed height/width at all, so every generation silently ran
    # at diffusers' own default (1024, confirmed via its docstring: "set to 1024 by default for the
    # best results"), the dominant real cost in every generation this session measured. 720x720 is
    # ~49% of 1024x1024's pixels - UNet/VAE compute scales with pixel count, real measured result:
    # 62.5s vs 125.8s for the same prompt, ~2x faster, no meaningful visible quality loss. Both values
    # confirmed valid via the real StableDiffusionXLPipeline.check_inputs source (only hard
    # requirement: divisibility by 8 - 720=8*90, 1024=8*128).
    #
    # A third option above 1024 (2048, then 1536) was built and really tested, not just assumed to
    # work, and dropped after both failed the same way: real, reproduced duplication artifacts (two
    # small, wrongly-scaled subjects floating in a mostly-empty frame instead of one, properly filling
    # it) - the well-documented result of pushing a base SDXL checkpoint past its native training
    # resolution without a dedicated hires-fix/upscale pass, which this simple height/width parameter
    # cannot provide. 2048 also hard-crashed before that ("Invalid buffer size: 16.00 GiB" at VAE
    # decode - fixed by pipe.vae.enable_tiling(), a real diffusers feature, but that only fixed the
    # crash, not the underlying composition failure). See README "Image generation" for the real
    # tested images. 1024 is this model family's real practical ceiling for this feature as built.
    image_resolution: Literal[720, 1024] = 720

    @field_validator("image_resolution", mode="before")
    @classmethod
    def _coerce_image_resolution(cls, v):
        """Real bug this fixes: pydantic's Literal (unlike a plain int field) does not coerce a
        numeric string to int - it requires the value to already be an int. .env values are always
        strings (confirmed for real: a live POST /api/images/resolution persisted "1024" via
        set_env_var, and the next server restart failed at startup with a literal_error over it), so
        without this the *very switch endpoint this setting exists for* would corrupt the next boot."""
        return int(v) if isinstance(v, str) and v.strip().lstrip("-").isdigit() else v

    # Image editing (img2img): reuses the same loaded pipeline as generation (see
    # app/tools/image.py::ImagePipelines) rather than loading a second model, so no separate
    # enabled/model flag - it's available whenever image generation is, and shares the same
    # per-style steps/guidance_scale as create_image (image_generation_params_for_style above).
    # strength is edit's own distinct knob - how much of the uploaded image is kept vs regenerated
    # (0 = unchanged, 1 = ignores it almost entirely). Measured for real: lower values (steps=4,
    # strength=0.75, tried first) barely changed the image at all for a clear prompt change ("a blue
    # abstract painting" on a solid red square stayed almost entirely red) - this higher default
    # produced a dramatic, clearly on-prompt result on a realistic source image (apples on a table ->
    # blue crystal gems on the same table).
    image_edit_strength: float = Field(default=0.9, gt=0, le=1)
    uploads_dir: str = "data/uploads"
    upload_max_bytes: int = Field(default=15_000_000, gt=0)  # matches voice_max_audio_bytes's default

    # edit_image content filter (app/tools/image_safety.py): editing an uploaded (real) photo into
    # sexual content is a distinct risk from-scratch generation doesn't carry - non-consensual
    # intimate imagery of an identifiable person. Configurable, by explicit user request, since this
    # is their local/private tool - off by default per that same request ("for now do not put any
    # safety checks"). Set to true in .env to turn it back on. IMPORTANT: this flag does NOT affect
    # the separate, unconditional minor-safety check in the same module, which has no flag anywhere
    # in this codebase and cannot be disabled, under any setting, for any user.
    image_edit_content_filter_enabled: bool = False
    image_edit_blocked_terms: str = (
        "nude,naked,nsfw,explicit,sexual,porn,pornographic,topless,undress,strip,lingerie,fetish,erotic"
    )

    @property
    def image_edit_blocked_term_list(self) -> list[str]:
        return [t.strip() for t in self.image_edit_blocked_terms.split(",") if t.strip()]

    # Zira talks first: in hands-free voice mode, after a quiet spell (a random 4-10 minutes) Zira starts
    # the conversation herself - a question to learn about the user, a genuine compliment, or a fun
    # point about one of their interests. The browser decides when (it knows whether the mic is on and
    # the user is idle); it stops trying after `proactive_max_unanswered` openers in a row get no answer.
    proactive_enabled: bool = True
    proactive_min_minutes: float = Field(default=4.0, gt=0)
    proactive_max_minutes: float = Field(default=10.0, gt=0)
    proactive_max_unanswered: int = Field(default=2, ge=1)

    # Video generation: local text-to-video with LTX-Video 2B or FastMetal-QAD (see app/tools/video.py). Opt-in
    # like images. "none" is the default and the startup state: no video model is loaded until the user picks
    # one, and the UI selection is runtime-only (never written back to .env), same as the image model. (Wan 2.1
    # VACE 1.3B was a third model here until 2026-09-27, removed at the user's request; its old VIDEO_* settings
    # in a .env are simply ignored.)
    # Singing (create_song, app/tools/song.py): ACE-Step 1.5 songs with sung vocals and music, added 2026-09-28 at the
    # user's request. Off unless SONG_GENERATION_ENABLED=true (needs .venv-acestep and third_party/ace-step/checkpoints,
    # ~10GB). SONG_LM "none" skips ACE-Step's 1.7B planning model (less memory, plainer songs).
    song_generation_enabled: bool = False
    song_python: str = str(PROJECT_ROOT / ".venv-acestep" / "bin" / "python")
    song_lm: str = "acestep-5Hz-lm-1.7B"
    song_quantization: str = ""
    song_steps: int = Field(default=8, ge=1, le=50)
    song_default_seconds: float = Field(default=60.0, ge=10.0, le=600.0)
    song_max_seconds: float = Field(default=180.0, ge=10.0, le=600.0)

    video_generation_enabled: bool = False
    video_model: Literal["none", "ltx", "fastmetal", "hunyuan"] = "none"
    # The same for videos: a video request while the video model is None switches this one on (user's choice:
    # HunyuanVideo 1.5, which replaced FastMetal 5B on 2026-09-28); a model already picked is never changed; "off" =
    # refuse until one is picked.
    video_auto_model: Literal["off", "ltx", "fastmetal", "hunyuan"] = "hunyuan"
    # Size of every video, whichever model makes it - switched from the UI and persisted here, like
    # IMAGE_RESOLUTION: 320p = 576x320, 480p = 832x480 (both fit LTX's /32 and FastMetal's size rules); portrait
    # swaps width and height. These override each model's own default size. 480p is ~2.2x the pixels: measured
    # for FastMetal 1.3B ~92s/step at 832x480 x 45 frames (LTX has not been tried at 480p on this Mac).
    video_resolution: Literal["320p", "480p"] = "320p"
    video_orientation: Literal["landscape", "portrait"] = "landscape"
    # FastMetal's prompts are encoded with Wan 2.1's UMT5-XXL; its official repo stores it in fp32 (22.7GB), which
    # does not fit a 16GB Mac or its disk budget, so it comes as ComfyUI's repackaged fp16 build of the same
    # weights (safetensors, 11.4GB, HF-compatible key names), streamed block by block.
    video_text_encoder_repo: str = "Comfy-Org/Wan_2.1_ComfyUI_repackaged"
    video_text_encoder_file: str = "split_files/text_encoders/umt5_xxl_fp16.safetensors"
    # Length: taken from the request ("30 sec", "1 minute"); `video_default_seconds` when none is given, never
    # more than `video_max_seconds` (a model that cannot continue from frames makes one piece at most).
    video_default_seconds: float = Field(default=8.0, gt=0)
    video_max_seconds: float = Field(default=30.0, gt=0)
    video_generation_timeout: float = Field(default=3600.0, gt=0)  # per piece; a video gets pieces x this
    # LTX-Video 2B distilled 0.9.8 (the LTX button): 8 steps, moderate detail. Single-file bf16 checkpoint
    # (transformer + VAE); T5-XXL text encoder as fp16 from comfyanonymous/flux_text_encoders (the official one is
    # fp32, 19GB), streamed like FastMetal's UMT5. Pieces of 8k+1 frames at 24 fps; 32-pixel size multiples.
    video_ltx_repo: str = "Lightricks/LTX-Video"
    video_ltx_file: str = "ltxv-2b-0.9.8-distilled.safetensors"
    video_ltx_text_encoder_repo: str = "comfyanonymous/flux_text_encoders"
    video_ltx_text_encoder_file: str = "t5xxl_fp16.safetensors"
    video_ltx_width: int = Field(default=576, ge=64, le=1280)
    video_ltx_height: int = Field(default=320, ge=64, le=1280)
    video_ltx_frames: int = Field(default=121, ge=9, le=257)
    video_ltx_overlap_frames: int = Field(default=9, ge=1, le=64)
    video_ltx_fps: int = Field(default=24, ge=1, le=60)
    # FastMetal-QAD 1.3B (the FastMetal button): Wan 2.1 1.3B distilled to 3 steps with an INT8 MLX DiT, the
    # fastest option (~5s 832x480 clip in a few minutes, ~4GB peak). Runs in a worker from its own
    # `.venv-fastmetal` (FastVideo's pinned libraries do not fit Zira's); the prompt is encoded here with the
    # streamed UMT5 above. Text-to-video only, so one clip per video: at most `video_fastmetal_frames` frames.
    video_fastmetal_repo: str = "FastVideo/FastMetal-1.3B-QAD"
    # (FastMetal-QAD 5B was a second choice here until 2026-09-28, when the user had it replaced by HunyuanVideo 1.5 -
    # app/tools/video_hunyuan.py.)
    video_fastmetal_python: str = str(PROJECT_ROOT / ".venv-fastmetal" / "bin" / "python")
    video_fastmetal_width: int = Field(default=832, ge=64, le=1280)
    video_fastmetal_height: int = Field(default=480, ge=64, le=1280)
    video_fastmetal_frames: int = Field(default=81, ge=5, le=129)
    video_fastmetal_fps: int = Field(default=16, ge=1, le=60)

    # Zira LoRA Studio: a separate app (own folder and .venv) that the LoRA button starts on demand;
    # Zira serves it at /lora-studio/ - see app/api/lora_studio.py.
    lora_studio_dir: str = str(PROJECT_ROOT.parent / "zira_lora_studio_fixed")

    # Context building
    context_max_messages: int = Field(default=20, ge=1)
    memory_top_k: int = Field(default=8, ge=1)
    max_message_chars: int = Field(default=8000, ge=1)

    # Logging
    log_level: str = "INFO"

    # Future robot link (not implemented yet)
    robot_enabled: bool = False
    robot_url: str = "ws://raspberrypi.local:9000/ws"
    # Phone notifications (Web Push, app/push.py). The push standard needs a contact for the sender: Zira's own
    # https address (the Tailscale Serve name) is used when this is empty - deliberately never an email address.
    push_subject: str = ""
    # Questions about an attached photo (look_at_image, app/tools/look.py): an Ollama vision model, loaded only
    # for the question (the chat model steps aside meanwhile). minicpm-v (7.6B, ~5.5GB) is installed on this Mac.
    vision_model: str = "minicpm-v"
    # Chat models that can see images themselves (comma-separated): with one of them in use, a photo question is
    # answered by the chat model directly - no swap to VISION_MODEL. NEWLIGHT's Qwen3.5 4B can (measured: it named a
    # red fox correctly in ~11 s).
    vision_chat_models: str = "qwen3.5-heretic:4b"
    # Documents to ask about (read_document, app/tools/documents.py): the largest file accepted.
    document_max_bytes: int = Field(default=25_000_000, gt=0)

    @property
    def resolved_model_label(self) -> str:
        return self.model_label or derive_model_label(self.active_model)

    @property
    def allowed_hosts(self) -> list[str]:
        hosts = ["localhost", "127.0.0.1", "[::1]"]
        # HOST=0.0.0.0 (or any non-loopback bind) means the user explicitly opened this up beyond
        # this machine (see run.sh's warning) - allowlist this machine's real LAN IP too, or every
        # request from another device on the WiFi would 400 here even though the socket accepted
        # the connection. Real, caught gap: switching HOST alone isn't enough on its own - confirmed
        # via a real request from this machine's own LAN IP failing with 400 before this existed.
        if self.host not in _LOOPBACK_HOSTS:
            lan_ip = _local_lan_ip()
            if lan_ip:
                hosts.append(lan_ip)
            # Same reasoning, separate real gap: Tailscale (remote access off-WiFi) arrives on its
            # own interface, which _local_lan_ip() never finds (see _tailscale_ip()'s docstring) -
            # without this, a request over Tailscale 400s here even though HOST=0.0.0.0 already
            # accepted the connection at the socket level.
            tailscale_ip = _tailscale_ip()
            if tailscale_ip:
                hosts.append(tailscale_ip)
        # `tailscale serve` (HTTPS on this Mac's Tailscale name, tailnet-only) proxies to 127.0.0.1, so it reaches
        # Zira even when it listens on loopback only - the "Tailscale only" setup chosen on 2026-09-27 (HOST=127.0.0.1).
        # Its name is therefore allowed whatever HOST is; without this the phone got 400 "Invalid host header".
        tailscale_name = _tailscale_dns_name()
        if tailscale_name:
            hosts.append(tailscale_name)
        return hosts

    @property
    def allowed_origins(self) -> set[str]:
        origins = {f"http://{host}:{self.port}" for host in ("localhost", "127.0.0.1", "[::1]")}
        if self.host not in _LOOPBACK_HOSTS:
            lan_ip = _local_lan_ip()
            if lan_ip:
                origins.add(f"http://{lan_ip}:{self.port}")
            tailscale_ip = _tailscale_ip()
            if tailscale_ip:
                origins.add(f"http://{tailscale_ip}:{self.port}")
        tailscale_name = _tailscale_dns_name()
        if tailscale_name:
            # https://name (no port: `tailscale serve` listens on 443) is the secure-context URL a phone uses for the
            # microphone - allowed whatever HOST is (see allowed_hosts); http://name:port only when not loopback-only.
            origins.add(f"https://{tailscale_name}")
            if self.host not in _LOOPBACK_HOSTS:
                origins.add(f"http://{tailscale_name}:{self.port}")
        return origins

    @property
    def file_roots(self) -> list[Path]:
        return [Path(r.strip()).expanduser() for r in self.file_access_roots.split(",") if r.strip()]

    @property
    def write_roots(self) -> list[Path]:
        return [Path(r.strip()).expanduser() for r in self.file_write_roots.split(",") if r.strip()]

    @property
    def music_roots(self) -> list[Path]:
        return [Path(r.strip()).expanduser() for r in self.music_library_roots.split(",") if r.strip()]

    @property
    def resolved_stt_initial_prompt(self) -> str | None:
        prompt = f"Hello {self.assistant_name}." if self.stt_initial_prompt is None else self.stt_initial_prompt
        return prompt.strip() or None

    @property
    def stt_language_candidate_list(self) -> list[str] | None:
        codes = [c.strip().lower() for c in self.stt_language_candidates.split(",") if c.strip()]
        return codes or None

    @property
    def exports_path(self) -> Path:
        path = Path(self.exports_dir)
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def uploads_path(self) -> Path:
        path = Path(self.uploads_dir)
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def checkpoints_path(self) -> Path:
        path = Path(self.checkpoints_dir)
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def checkpoint_threshold_tokens(self) -> int:
        return int(self.ollama_num_ctx * self.context_checkpoint_threshold)

    @property
    def database_file(self) -> Path:
        path = Path(self.database_path)
        return path if path.is_absolute() else PROJECT_ROOT / path


@lru_cache
def get_settings() -> Settings:
    return Settings()


def set_env_var(key: str, value: str, path: Path = PROJECT_ROOT / ".env") -> None:
    """Persists a single KEY=value into .env, atomically, without disturbing anything else in the
    file - used by the model-mode switch so the *next* process (after a restart) reads the new mode.
    Never touches the currently-running process's cached Settings (get_settings() is @lru_cache'd
    per-process; a restart is a fresh process, so there is nothing to invalidate here)."""
    line = f"{key}={value}\n"
    pattern = re.compile(rf"^{re.escape(key)}=.*$")
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True) if path.is_file() else []
    for i, existing in enumerate(lines):
        if pattern.match(existing.rstrip("\n")):
            lines[i] = line
            break
    else:
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        lines.append(line)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text("".join(lines), encoding="utf-8")
    os.replace(tmp_path, path)
