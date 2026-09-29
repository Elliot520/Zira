"""FastAPI application factory and entry point."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.agent.agent import Agent
from app.agent.tool_manager import ToolRegistry
from app.ai.context import ContextBuilder
from app.ai.llm import LLM, LLMBackend, LLMError, model_matches
from app.ai.model_manager import LLMMemoryReleaser, OllamaModelManager
from app.ai.prompts import Personality
from app.api import daily as daily_api, learning as learning_api
from app.api import chat, health, images as images_api, knowledge as knowledge_api, lora_studio, documents as documents_api, media as media_api, model as model_api, push as push_api, music as music_api, songs as songs_api, system as system_api, ads as ads_api, videos as videos_api, voice, workspace
from app.api.chat import error_payload
from app.api.model import ModelError
from app.config import PROJECT_ROOT, Settings, get_settings
from app.knowledge.knowledge_store import KnowledgeStore
from app.knowledge.researcher import BackgroundResearcher, idle_research_loop
from app.memory.conversation_store import ConversationStore
from app.memory.database import DatabaseError, init_database
from app.memory.media_store import MediaStore
from app.push import PushNotifier
from app.memory.checkpoints import CheckpointManager
from app.memory.extractor import MemoryExtractor
from app.memory.memory_manager import MemoryManager
from app.tools.changes import ChangeError, ChangeStore, ProposeCreateTool, ProposeEditTool
from app.tools.code_tools import CodeOutlineTool, ProjectOverviewTool
from app.tools.filesystem import FileAccessPolicy, ListDirectoryTool, ReadFileTool, SearchFilesTool
from app.tools.image import CreateImageTool, EditImageTool, ImagePipelines
from app.tools.documents import DocumentStore, ReadDocumentTool
from app.tools.look import LookAtImageTool
from app.tools.video import CreateVideoTool, VideoConfig, VideoPipelines
from app.tools.video_fastmetal import FastMetalConfig
from app.tools.video_ltx import LTXConfig
from app.tools.music import ControlMusicTool, LocalMusicLibrary, MusicLibrary, PlayMusicTool, RemoteMusicIndex
from app.tools.pdf import CreatePdfTool
from app.tools.postman import GeneratePostmanCollectionTool
from app.tools.web_search import DuckDuckGoSearch, SearchProvider, WebSearchTool
from app.voice.errors import VoiceError
from app.voice.speech_to_text import SpeechToText, create_speech_to_text
from app.voice.text_to_speech import TextToSpeech, create_text_to_speech
from app.voice.transcript_cleanup import TranscriptCleaner

logger = logging.getLogger("jarvis")

FRONTEND_DIR = PROJECT_ROOT / "frontend"


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


class _RevalidatingStaticFiles(StaticFiles):
    """No Cache-Control on the UI files let browsers keep a stale copy for a long time after an edit
    (a phone kept showing the old page until a hard refresh). `no-cache` makes them revalidate on every
    load; the ETag Starlette already sends turns that into a cheap 304 when nothing changed."""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


def _setup_daily(settings, db, tools, notifier, conversations):
    """Reminders, calendar, morning brief and backups (2026-09-28). start() launches their loops in the lifespan."""
    import datetime as dt
    from types import SimpleNamespace

    from app.backup import Backup, backup_loop
    from app.brief import MorningBrief, Weather, brief_loop
    from app.config import PROJECT_ROOT
    from app.reminders import (CancelReminderTool, ListRemindersTool, ReminderStore, SetReminderTool,
                               reminder_loop)
    from app.tools.calendar import AddCalendarEventTool, CalendarEventsTool, MacCalendar

    reminders = ReminderStore(db) if settings.reminders_enabled else None
    if reminders is not None:
        for tool in (SetReminderTool(reminders), ListRemindersTool(reminders), CancelReminderTool(reminders)):
            tools.register(tool)
    calendar = MacCalendar(PROJECT_ROOT / "bin" / "ZiraCalendar.app") if settings.calendar_enabled else None
    if calendar is not None and calendar.available:
        tools.register(CalendarEventsTool(calendar))
        tools.register(AddCalendarEventTool(calendar))
    brief = MorningBrief(db, calendar=calendar, reminders=reminders,
                         weather=Weather(settings.brief_city) if settings.brief_city else None,
                         notifier=notifier, conversations=conversations,
                         name=settings.brief_name) if settings.brief_enabled else None
    backup = Backup(Path(settings.database_path).resolve().parent, Path(settings.database_path).resolve(),
                    Path(settings.backup_dir).expanduser(), env_file=PROJECT_ROOT / ".env",
                    restic_repository=settings.backup_restic_repository,
                    restic_password_file=settings.backup_restic_password_file) if settings.backup_enabled else None

    def clock(value: str) -> dt.time:
        hour, minute = (int(x) for x in value.split(":"))
        return dt.time(hour, minute)

    def start() -> list:
        tasks = []
        if reminders is not None:
            tasks.append(asyncio.create_task(reminder_loop(reminders, notifier=notifier, conversations=conversations,
                                                           mac=settings.reminders_mac_notification)))
        if brief is not None:
            tasks.append(asyncio.create_task(brief_loop(brief, clock(settings.brief_time))))
        if backup is not None:
            tasks.append(asyncio.create_task(backup_loop(backup, clock(settings.backup_time))))
        logger.info("Daily: reminders %s, calendar %s, morning brief %s, backups %s",
                    "on" if reminders else "off",
                    "on" if calendar is not None and calendar.available else "off",
                    f"at {settings.brief_time}" + (f" ({settings.brief_city})" if settings.brief_city else "") if brief else "off",
                    f"at {settings.backup_time} to {settings.backup_dir}" + (" + server" if settings.backup_restic_repository else "") if backup else "off")
        return tasks

    return SimpleNamespace(reminders=reminders, calendar=calendar, brief=brief, backup=backup, start=start)


def _setup_learning(settings, db, llm, memory, agent, notifier, image_pipelines, video_pipelines):
    """The nightly study and per-message recall (app/learning/), or None when LEARNING_ENABLED is off."""
    if not settings.learning_enabled:
        return None
    from types import SimpleNamespace

    from app.learning.recall import Recall
    from app.learning.store import LearnedStore
    from app.learning.study import NightlyStudy
    from app.memory.embeddings import Embedder, VectorIndex

    embedder = Embedder(settings.ollama_host, settings.embedding_model)
    index = VectorIndex(db)
    store = LearnedStore(db, index)
    from app.learning.recall import same_question

    agent.recall = Recall(embedder, index, store, memory, learned_threshold=settings.learned_match_threshold,
                          memory_threshold=settings.memory_semantic_threshold,
                          verify=lambda stored, asked: same_question(llm, stored, asked))
    study = NightlyStudy(llm, db, store, index, embedder, max_items=settings.study_max_questions,
                         notifier=notifier if settings.study_notify else None)

    def busy() -> bool:
        """An image or video is being made (the GPU is taken)."""
        locks = [p.generation_lock for p in (image_pipelines, video_pipelines) if p is not None]
        return any(lock.locked() for lock in locks)

    return SimpleNamespace(store=store, study=study, recall=agent.recall, busy=busy, task=None)


async def model_idle_loop(image_pipelines, video_pipelines, idle_minutes: float, check_seconds: float = 60.0) -> None:
    """An image or video model unused for `idle_minutes` goes back to None, freeing its memory (user request,
    2026-09-28). It is remembered: the next request switches the same model on again by itself."""
    from app.api.images import park_image_model
    from app.api.videos import park_video_model

    idle = idle_minutes * 60
    while True:
        await asyncio.sleep(check_seconds)
        now = time.monotonic()
        try:
            for pipelines, loaded, park in (
                (image_pipelines, lambda p: p.style != "none", park_image_model),
                (video_pipelines, lambda p: p.model != "none", park_video_model),
            ):
                if pipelines is not None and loaded(pipelines) and not pipelines.generating \
                        and now - pipelines.last_used >= idle:
                    await park(pipelines, f"unused for {idle_minutes:g} minutes")
        except Exception:  # noqa: BLE001 - never let the loop die
            logger.exception("Idle model check failed")


async def _start_tts(tts: TextToSpeech) -> None:
    try:
        await tts.start()
    except Exception as exc:  # noqa: BLE001 - text-only Zira keeps working
        logger.error("Text-to-speech could not start: %s", exc)


def create_app(
    settings: Settings | None = None,
    llm: LLMBackend | None = None,
    search_provider: SearchProvider | None = None,
    stt: SpeechToText | None = None,
    tts: TextToSpeech | None = None,
    checkpoints: CheckpointManager | None = None,
    knowledge: KnowledgeStore | None = None,
    transcript_cleaner: TranscriptCleaner | None = None,
    music_library: MusicLibrary | None = None,
    model_manager: OllamaModelManager | None = None,
    env_path: Path | None = None,
    restart_marker_path: Path | None = None,
    image_pipelines: ImagePipelines | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(settings.log_level)

    try:
        db = init_database(settings.database_file)
    except DatabaseError as exc:
        logger.error("Startup failed: %s", exc)
        raise

    llm = llm or LLM(settings)
    stt = stt or create_speech_to_text(settings)
    tts = tts or create_text_to_speech(settings)
    memory = MemoryManager(db)
    conversations = ConversationStore(db)
    # The gallery's library of images and videos (thumbnails cached next to the exports folder).
    media_store = MediaStore(db, settings.exports_path, settings.exports_path.parent / "thumbs")
    tools = ToolRegistry()
    # Constructed unconditionally (cheap - no I/O until .search() is called) so self-learning can
    # use it even when the interactive web_search tool below is switched off for chat.
    provider = search_provider or DuckDuckGoSearch(timeout=settings.web_search_timeout)
    if settings.web_search_enabled:
        tools.register(WebSearchTool(provider, max_results=settings.web_search_max_results))
    # Project access is opt-in. Write roots imply read access so JARVIS can read what it proposes to change.
    write_policy = FileAccessPolicy(settings.write_roots)
    file_policy = FileAccessPolicy([*settings.file_roots, *settings.write_roots])
    changes = ChangeStore(db, write_policy) if write_policy.enabled else None
    # Root-relative, not f"http://{settings.host}:{settings.port}" (the old, real bug this fixed):
    # tools are built once at startup, before any request exists, so a single baked-in host can
    # never be right for every client - it showed http://0.0.0.0:8000/... (unreachable by design;
    # 0.0.0.0 means "any interface", not a valid destination) once HOST was opened up for LAN
    # access, and would have been equally wrong the other way (a LAN IP baked in would break the
    # Mac's own browser). A relative URL sidesteps the whole question: the browser resolves
    # /api/exports/<file> against whatever origin actually loaded the page, correct for 127.0.0.1,
    # localhost, and the LAN IP alike. frontend/app.js's LINK_PATTERN was updated to match and
    # linkify these alongside absolute http(s) URLs.
    public_url = ""
    if file_policy.enabled:
        for tool in (
            ProjectOverviewTool(file_policy),
            CodeOutlineTool(file_policy),
            ListDirectoryTool(file_policy),
            ReadFileTool(file_policy, settings.file_read_max_chars),
            SearchFilesTool(file_policy),
            GeneratePostmanCollectionTool(file_policy, settings.exports_path, public_url),
        ):
            tools.register(tool)
    tools.register(CreatePdfTool(settings.exports_path, public_url))
    if settings.image_generation_enabled:
        style_model, style_dtype, style_variant = settings.image_model_for_style(settings.image_style)
        image_pipelines = image_pipelines or ImagePipelines(
            style_model, style=settings.image_style, dtype=style_dtype, variant=style_variant,
            resolution=settings.image_resolution,
        )
        from app.tools.face_id import FaceID, FaceIDConfig
        from app.tools.inpaint import AreaMasker

        face_id = FaceID(
            FaceIDConfig(adapter_file=settings.image_faceid_adapter_file, scale=settings.image_faceid_scale),
            Path(__file__).resolve().parent.parent,
        ) if settings.image_faceid_enabled else None
        create_image_tool = CreateImageTool(
            image_pipelines, settings.exports_path, public_url,
            params_for_style=settings.image_generation_params_for_style,
            timeout=settings.image_generation_timeout, media=media_store,
            face_id=face_id, uploads_dir=settings.uploads_path,
        )
        tools.register(create_image_tool)
        image_tools = [create_image_tool]
        tools.register(
            EditImageTool(
                image_pipelines, settings.uploads_path, settings.exports_path, public_url,
                params_for_style=settings.image_generation_params_for_style,
                strength=settings.image_edit_strength,
                timeout=settings.image_generation_timeout,
                content_filter_enabled=settings.image_edit_content_filter_enabled,
                blocked_terms=settings.image_edit_blocked_term_list, media=media_store,
                masker=AreaMasker() if settings.image_inpaint_enabled else None,
            )
        )
        image_tools.append(tools.get("edit_image"))
        # Automatic switch-on and fallback (user request, 2026-09-28): the same switch as the buttons.
        from app.api.images import select_image_style

        async def auto_image() -> None:
            """The model Zira parked earlier if any, else the default ("off": none - the request is refused)."""
            style = image_pipelines.resume_style or (
                settings.image_auto_style if settings.image_auto_style != "off" else None)
            if style is None:
                return
            from app.api.videos import park_video_model

            await park_video_model(video_pipelines, "an image was asked for")  # never both in memory together
            image_pipelines.resume_style = None
            await select_image_style(image_pipelines, settings, style)

        for tool in image_tools:
            tool.auto_select = auto_image
            tool.release_model = lambda: select_image_style(image_pipelines, settings, "none")
    model_manager = model_manager or OllamaModelManager(settings)
    video_pipelines: VideoPipelines | None = None
    if settings.video_generation_enabled:
        # Nothing is loaded here - the pipeline only loads when the user picks a video model. It shares
        # the image system's generation lock: two MPS generations at once crash the process.
        video_pipelines = VideoPipelines(
            VideoConfig(
                text_encoder_repo=settings.video_text_encoder_repo, text_encoder_file=settings.video_text_encoder_file,
                default_seconds=settings.video_default_seconds, max_seconds=settings.video_max_seconds,
            ),
            model=settings.video_model,
            generation_lock=image_pipelines.generation_lock if image_pipelines is not None else None,
            ltx_config=LTXConfig(
                repo=settings.video_ltx_repo, file=settings.video_ltx_file,
                text_encoder_repo=settings.video_ltx_text_encoder_repo,
                text_encoder_file=settings.video_ltx_text_encoder_file, width=settings.video_ltx_width,
                height=settings.video_ltx_height, frames=settings.video_ltx_frames,
                overlap_frames=settings.video_ltx_overlap_frames, fps=settings.video_ltx_fps,
            ),
            resolution=settings.video_resolution, orientation=settings.video_orientation,
            fastmetal_config=FastMetalConfig(
                repo=settings.video_fastmetal_repo, python=settings.video_fastmetal_python,
                worker=str(PROJECT_ROOT / "third_party" / "fastmetal" / "worker.py"),
                log_path=str(PROJECT_ROOT / "logs" / "fastmetal.log"), width=settings.video_fastmetal_width,
                height=settings.video_fastmetal_height, frames=settings.video_fastmetal_frames,
                fps=settings.video_fastmetal_fps,
            ),
        )
        video_tool = CreateVideoTool(
            video_pipelines, settings.exports_path, public_url, timeout=settings.video_generation_timeout,
            llm_memory=LLMMemoryReleaser(model_manager, lambda: llm.model, settings.ollama_keep_alive),
            media=media_store, uploads_dir=settings.uploads_path,
        )
        from app.api.videos import select_video_model

        async def auto_video() -> None:
            """The model Zira parked earlier if any, else the default ("off": none - the request is refused)."""
            model = video_pipelines.resume_model or (
                settings.video_auto_model if settings.video_auto_model != "off" else None)
            if model is None:
                return
            from app.api.images import park_image_model

            # A video model next to an image model would not fit in 16GB: never both in memory together.
            await park_image_model(image_pipelines, "a video was asked for")
            video_pipelines.resume_model = None
            await select_video_model(video_pipelines, settings, model)

        video_tool.auto_select = auto_video
        if image_pipelines is not None:
            async def first_frame(prompt: str, width: int, height: int):
                """An image-to-video model's opening frame, made by the user's image model (or the default)."""
                import numpy as np

                style = image_pipelines.resume_style or (image_pipelines.style if image_pipelines.style != "none"
                                                         else None) or (settings.image_auto_style
                                                                        if settings.image_auto_style != "off"
                                                                        else "lightning")
                model, dtype, variant = settings.image_model_for_style(style)
                steps, guidance = settings.image_generation_params_for_style(style)
                image = await asyncio.to_thread(image_pipelines.one_off_image, style, model, dtype, variant, prompt,
                                                width, height, steps, guidance)
                return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0

            video_tool.first_frame = first_frame
        video_tool.release_model = lambda: select_video_model(video_pipelines, settings, "none")
        tools.register(video_tool)
    song_tool = None
    if settings.song_generation_enabled:
        from app.tools.song import CreateSongTool, SongConfig, SongWorker

        song_tool = CreateSongTool(
            SongWorker(SongConfig(
                python=settings.song_python, worker=str(PROJECT_ROOT / "third_party" / "song" / "worker.py"),
                root=str(PROJECT_ROOT / "third_party" / "ace-step"), log_path=str(PROJECT_ROOT / "logs" / "song.log"),
                lm=settings.song_lm, quantization=settings.song_quantization or None, steps=settings.song_steps,
                default_seconds=settings.song_default_seconds, max_seconds=settings.song_max_seconds,
            )),
            settings.exports_path, public_url,
            generation_lock=image_pipelines.generation_lock if image_pipelines is not None else asyncio.Lock(),
            llm_memory=LLMMemoryReleaser(model_manager, lambda: llm.model, settings.ollama_keep_alive),
            image_pipelines=image_pipelines, video_pipelines=video_pipelines, media=media_store,
        )
        tools.register(song_tool)
    ad_tool = None
    if video_pipelines is not None:
        # Scene-based ads (app/tools/ad.py): the video tool for the scenes, the reply voice for the voice-over, the
        # song worker (when singing is on) for the jingle.
        from app.tools.ad import CreateAdTool

        ad_tool = CreateAdTool(
            tools.get("create_video"), settings.exports_path, public_url,
            generation_lock=video_pipelines.generation_lock, tts=tts,
            song_worker=song_tool._worker if song_tool is not None else None,
            llm_memory=LLMMemoryReleaser(model_manager, lambda: llm.model, settings.ollama_keep_alive),
            media=media_store, video_pipelines=video_pipelines,
        )
        tools.register(ad_tool)
    if changes is not None:
        tools.register(ProposeEditTool(changes))
        tools.register(ProposeCreateTool(changes))
    extractor = MemoryExtractor(llm, memory, max_items=settings.auto_memory_max_items) if settings.auto_memory else None
    if checkpoints is None and settings.context_checkpoint_enabled:
        checkpoints = CheckpointManager(llm, conversations, settings.checkpoints_path, settings.checkpoint_threshold_tokens)
    if knowledge is None and settings.knowledge_learning_enabled:
        knowledge = KnowledgeStore(db)
    if transcript_cleaner is None and settings.voice_transcript_cleanup_enabled:
        transcript_cleaner = TranscriptCleaner(llm, timeout=settings.voice_transcript_cleanup_timeout)
    if music_library is None:
        local_music = LocalMusicLibrary(settings.music_roots) if settings.music_roots else None
        remote_music = (
            RemoteMusicIndex(settings.music_remote_index_url, cache_seconds=settings.music_remote_index_cache_seconds)
            if settings.music_remote_index_url
            else None
        )
        music_library = MusicLibrary(local_music, remote_music) if (local_music or remote_music) else None
    if music_library is not None and music_library.enabled:
        tools.register(PlayMusicTool(music_library))
        tools.register(ControlMusicTool())
    env_path = env_path or PROJECT_ROOT / ".env"
    restart_marker_path = restart_marker_path or (PROJECT_ROOT / "data" / ".restart_pending")
    researcher = (
        BackgroundResearcher(
            llm,
            provider,
            memory,
            knowledge,
            max_results_per_topic=settings.knowledge_max_results_per_topic,
            topic_cooldown_days=settings.knowledge_topic_cooldown_days,
        )
        if knowledge is not None
        else None
    )
    context = ContextBuilder(
        memory,
        conversations,
        Personality.from_settings(settings),
        max_messages=settings.context_max_messages,
        memory_top_k=settings.memory_top_k,
        web_search=settings.web_search_enabled,
        project_roots=[str(r) for r in file_policy.roots],
        can_write=changes is not None,
        changes=changes,
        checkpoints=checkpoints,
        knowledge=knowledge,
        knowledge_top_k=settings.knowledge_context_top_k,
        light_mode=settings.model_mode in ("light", "newlight"),  # both are small models: the concise block
    )
    if image_pipelines is not None:
        # Questions about an attached photo (the image upload exists when image generation is on).
        tools.register(LookAtImageTool(
            settings.uploads_path, settings.ollama_host, model=settings.vision_model,
            generation_lock=image_pipelines.generation_lock,
            llm_memory=LLMMemoryReleaser(model_manager, lambda: llm.model, settings.ollama_keep_alive),
            chat_model=lambda: llm.model,
            seeing_chat_models=frozenset(m.strip() for m in settings.vision_chat_models.split(",") if m.strip()),
        ))
    # Documents the user attaches to ask about (kept next to the database).
    document_store = DocumentStore(db, settings.exports_path.parent / "documents")
    tools.register(ReadDocumentTool(document_store))
    if settings.web_search_enabled:  # reading a link is internet access too, under the same switch
        from app.tools.web_page import ReadWebPageTool

        tools.register(ReadWebPageTool())
    from app.tools.python_runner import RunPythonTool

    tools.register(RunPythonTool())  # exact maths in a sandbox
    agent = Agent(llm, memory, conversations, context, tools=tools, extractor=extractor, checkpoints=checkpoints)
    if song_tool is not None:
        agent.planner.song_finder = song_tool.find_saved  # a song Zira already made is played, not made again
    # Phone notifications when a video is done (app/push.py); the key pair lives next to the database.
    from app.config import _tailscale_dns_name

    push_subject = settings.push_subject or f"https://{_tailscale_dns_name() or 'localhost'}"
    push_notifier = PushNotifier(db, settings.exports_path.parent / "vapid_private.pem", push_subject)
    agent.notifier = push_notifier

    learning = _setup_learning(settings, db, llm, memory, agent, push_notifier, image_pipelines, video_pipelines)
    daily = _setup_daily(settings, db, tools, push_notifier, conversations)
    research_task: asyncio.Task | None = None

    @asynccontextmanager
    async def lifespan(app_: FastAPI) -> AsyncIterator[None]:
        nonlocal research_task
        logger.info(
            "JARVIS starting: model=%s ollama=%s db=%s",
            llm.model,
            settings.ollama_host,
            db.path,
        )
        # Kokoro (TTS_PROVIDER=kokoro) loads once, in the background: the server answers at once and text chat
        # works while the voice warms up; a voice that fails to load only turns speech off (logged, see /status).
        tts_start_task = asyncio.create_task(_start_tts(tts)) if hasattr(tts, "start") else None
        daily_tasks = daily.start()
        study_task = None
        if learning is not None:
            from app.learning.study import nightly_study_loop

            study_task = asyncio.create_task(nightly_study_loop(
                app_.state, learning.study, start_hour=settings.study_start_hour, end_hour=settings.study_end_hour,
                min_idle_minutes=settings.study_min_idle_minutes, busy=learning.busy,
            ))
            logger.info("Self-study: on (nightly %02d:00-%02d:00 when idle, %d learned so far, embeddings %s)",
                        settings.study_start_hour, settings.study_end_hour, learning.store.count(),
                        settings.embedding_model)
        status = await llm.status()
        if not status.online:
            logger.warning("Ollama is not reachable at %s. Start it with `ollama serve`.", settings.ollama_host)
        elif not model_matches(llm.model, status.models):
            logger.warning("Ollama is online but model '%s' is missing. Run: ollama pull %s", llm.model, llm.model)
        else:
            logger.info("Ollama connected; model '%s' is available", llm.model)
        logger.info("Long-term memories stored: %d (auto-memory %s, tools: %s)",
            memory.count(),
            "on" if extractor else "off",
            ", ".join(tools.names()) or "none",
        )
        logger.info("Voice: STT=%s TTS=%s", stt.name, tts.name)
        logger.info("Voice transcript cleanup: %s", "on" if transcript_cleaner else "off")
        logger.info(
            "Music: %s",
            f"on (local={bool(music_library.local)}, remote={bool(music_library.remote)})"
            if music_library is not None and music_library.enabled
            else "off",
        )
        if checkpoints is not None:
            logger.info(
                "Context checkpointing: on (threshold=%d tokens of %d, dir=%s)",
                settings.checkpoint_threshold_tokens, settings.ollama_num_ctx, settings.checkpoints_path,
            )
        if researcher is not None:
            research_task = asyncio.create_task(
                idle_research_loop(
                    app_.state, researcher, settings.knowledge_idle_minutes, settings.knowledge_research_cooldown_minutes
                )
            )
            logger.info(
                "Self-learning: on (idle >= %.0fmin, cooldown %.0fmin, topic cooldown %.0fd, %d topics learned so far)",
                settings.knowledge_idle_minutes,
                settings.knowledge_research_cooldown_minutes,
                settings.knowledge_topic_cooldown_days,
                knowledge.count() if knowledge else 0,
            )
        idle_task = None
        if settings.model_idle_minutes > 0 and (image_pipelines is not None or video_pipelines is not None):
            idle_task = asyncio.create_task(model_idle_loop(image_pipelines, video_pipelines, settings.model_idle_minutes))
            logger.info("Idle models: an image/video model unused for %g minutes goes back to None",
                        settings.model_idle_minutes)
        try:
            yield
        finally:
            logger.info("JARVIS shutting down")
            if idle_task is not None:
                idle_task.cancel()
            if research_task is not None:
                research_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await research_task
            await llm.aclose()
            if tts_start_task is not None and not tts_start_task.done():
                tts_start_task.cancel()
            if study_task is not None:
                study_task.cancel()
            for task in daily_tasks:
                task.cancel()
            if hasattr(tts, "aclose"):
                await tts.aclose()  # stops the Kokoro / IndicF5 voice worker, if it is running
            db.close()

    app = FastAPI(title="Zira", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.llm = llm
    app.state.db = db
    app.state.media = media_store
    app.state.push = push_notifier
    app.state.documents = document_store
    app.state.memory = memory
    app.state.conversations = conversations
    app.state.agent = agent
    app.state.file_policy = file_policy
    app.state.changes = changes
    app.state.stt = stt
    app.state.tts = tts
    app.state.learning = learning
    app.state.reminders = daily.reminders
    app.state.brief = daily.brief
    app.state.backup = daily.backup
    app.state.transcript_cleaner = transcript_cleaner
    app.state.music_library = music_library
    app.state.model_manager = model_manager
    app.state.env_path = env_path
    app.state.restart_marker_path = restart_marker_path
    app.state.image_pipelines = image_pipelines
    app.state.video_pipelines = video_pipelines
    app.state.song_tool = song_tool
    app.state.ad_tool = ad_tool
    app.state.checkpoints = checkpoints
    app.state.knowledge = knowledge
    app.state.researcher = researcher
    app.state.last_chat_at = time.monotonic()  # updated by chat endpoints; read by idle_research_loop

    # Blocks DNS-rebinding: only answer requests addressed to a local hostname.
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)
    # Worked out once: the Tailscale name comes from a `tailscale status` subprocess, too slow to run per request.
    allowed_origins = app.state.allowed_origins = settings.allowed_origins

    @app.middleware("http")
    async def guard_and_log(request: Request, call_next):
        # Reject cross-site state-changing requests from browsers.
        origin = request.headers.get("origin")
        if request.method not in ("GET", "HEAD", "OPTIONS") and origin and origin not in allowed_origins:
            logger.warning("Rejected %s %s: disallowed origin", request.method, request.url.path)
            return JSONResponse(
                {"error": "forbidden_origin", "detail": "Cross-origin requests are not allowed."},
                status_code=403,
            )
        start = time.perf_counter()
        response = await call_next(request)
        if request.url.path not in ("/api/health", "/lora-studio" + lora_studio.STATUS_PATH):
            logger.info(
                "%s %s -> %d (%.0f ms)",
                request.method,
                request.url.path,
                response.status_code,
                (time.perf_counter() - start) * 1000,
            )
        return response

    @app.exception_handler(LLMError)
    async def llm_error_handler(_: Request, exc: LLMError) -> JSONResponse:
        logger.error("LLM error (%s): %s", exc.code, exc)
        return JSONResponse(error_payload(exc), status_code=exc.status_code)

    @app.exception_handler(ChangeError)
    async def change_error_handler(_: Request, exc: ChangeError) -> JSONResponse:
        return JSONResponse({"error": "change_refused", "detail": str(exc)}, status_code=exc.status_code)

    @app.exception_handler(VoiceError)
    async def voice_error_handler(_: Request, exc: VoiceError) -> JSONResponse:
        logger.error("Voice error (%s): %s", exc.code, exc)
        return JSONResponse({"error": exc.code, "detail": str(exc)}, status_code=exc.status_code)

    @app.exception_handler(ModelError)
    async def model_error_handler(_: Request, exc: ModelError) -> JSONResponse:
        logger.error("Model switch error (%s): %s", exc.code, exc)
        return JSONResponse({"error": exc.code, "detail": str(exc)}, status_code=exc.status_code)

    @app.exception_handler(DatabaseError)
    async def db_error_handler(_: Request, exc: DatabaseError) -> JSONResponse:
        return JSONResponse(error_payload(exc), status_code=500)

    @app.exception_handler(RequestValidationError)
    async def validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        parts = []
        for err in exc.errors():
            if err["type"] == "json_invalid":
                parts.append("request body is not valid JSON")
                continue
            field = ".".join(str(p) for p in err["loc"] if p != "body")
            msg = err["msg"].removeprefix("Value error, ")
            parts.append(f"{field}: {msg}" if field else msg)
        return JSONResponse({"error": "invalid_request", "detail": "; ".join(parts)}, status_code=422)

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = {404: "not_found", 405: "method_not_allowed", 422: "invalid_request"}.get(exc.status_code, "http_error")
        return JSONResponse({"error": code, "detail": str(exc.detail)}, status_code=exc.status_code)

    @app.exception_handler(Exception)
    async def unhandled_handler(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error")
        return JSONResponse(error_payload(exc), status_code=500)

    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(chat.ws_router)
    app.include_router(workspace.router)
    app.include_router(voice.router)
    app.include_router(knowledge_api.router)
    app.include_router(music_api.router)
    app.include_router(model_api.router)
    app.include_router(images_api.router)
    app.include_router(videos_api.router)
    app.include_router(songs_api.router)
    app.include_router(system_api.router)
    app.include_router(ads_api.router)
    app.include_router(media_api.router)
    app.include_router(push_api.router)
    app.include_router(documents_api.router)
    app.include_router(lora_studio.router)
    app.include_router(learning_api.router)
    app.include_router(daily_api.router)

    if Path(FRONTEND_DIR).is_dir():
        app.mount("/", _RevalidatingStaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

    return app


def main() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        access_log=False,
        log_level=settings.log_level.lower(),
    )
    # No code here reliably runs after a model-mode-triggered restart: a real test found that a
    # process sending itself SIGTERM (app/api/model.py's restart trigger) never returns control to
    # this function - uvicorn logs a full graceful shutdown first, but the process still ends up
    # terminated by the signal itself, not by returning here. run.sh's loop instead checks the
    # restart-marker *file* (app/restart.py) directly after this process exits, which is unaffected
    # by exactly how it died - see run.sh for the actual restart-vs-exit decision.


if __name__ == "__main__":
    main()
