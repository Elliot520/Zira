"""Voice endpoints: speech-to-text and text-to-speech, decoupled from the chat pipeline.

Voice does NOT get its own conversation/memory/LLM path. The frontend calls `/api/voice/transcribe`
to turn a recorded clip into text, then sends that text through the *existing* `/ws/chat` (or
`/api/chat`) flow exactly like a typed message — same conversation id, same history, same memory,
same tools. `/api/voice/speak` is called afterwards on the assistant's reply text to get audio back.
Both endpoints are stateless and provider-agnostic (see app/voice/{speech_to_text,text_to_speech}.py).
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException, Request, UploadFile
from fastapi.responses import Response

from app.models.schemas import SpeakRequest, TranscribeResponse, TTSVoiceSwitchRequest, VoiceStatusResponse
from app.voice.errors import AudioTooLargeError, VoiceError
from app.voice.speech_to_text import UnavailableSpeechToText
from app.voice.text_cleaning import clean_for_speech
from app.voice.text_to_speech import SwitchableTTS, UnavailableTextToSpeech

logger = logging.getLogger("jarvis.api.voice")

router = APIRouter(prefix="/api/voice", tags=["voice"])


@router.get("/status", response_model=VoiceStatusResponse)
async def status(request: Request) -> VoiceStatusResponse:
    state = request.app.state
    return VoiceStatusResponse(
        stt_available=not isinstance(state.stt, UnavailableSpeechToText),
        stt_provider=state.stt.name,
        stt_loaded=state.stt.is_loaded,
        tts_available=not isinstance(state.tts, UnavailableTextToSpeech),
        tts_provider=state.tts.name,
        tts_voice=getattr(state.tts, "voice", ""),
        tts_choices=list(state.tts.choices) if isinstance(state.tts, SwitchableTTS) else [],
        tts_choice=state.tts.choice if isinstance(state.tts, SwitchableTTS) else "",
        tts_streaming=streaming_speech(state) is not None,
        tts_error=getattr(getattr(state.tts, "current", state.tts), "error", None) or "",
    )


def streaming_speech(state) -> object | None:
    """The TTS engine to stream voice replies with, or None when streaming is off or no voice is available."""
    if not state.settings.tts_streaming or isinstance(state.tts, UnavailableTextToSpeech):
        return None
    return state.tts


@router.post("/tts-voice")
async def switch_tts_voice(body: TTSVoiceSwitchRequest, request: Request) -> dict:
    """Switch between the configured voices (Kokoro / `say`, or `say` / IndicF5). Choosing a worker voice starts
    it and waits until its model is loaded. Runtime-only: a restart always starts on the default voice."""
    tts = request.app.state.tts
    if not isinstance(tts, SwitchableTTS):
        raise HTTPException(status_code=404, detail="Only one voice is set up (TTS_PROVIDER=kokoro sets up two).")
    if body.voice not in tts.choices:
        raise HTTPException(status_code=404, detail=f"The {body.voice} voice is not set up here.")
    previous = tts.choice
    try:
        changed = await tts.switch(body.voice)
    except VoiceError as exc:
        logger.warning("Voice switch to %s failed: %s", body.voice, exc)
        raise HTTPException(status_code=500, detail=f"{exc} The voice stays {tts.choice}.") from exc
    if changed:
        logger.info("TTS voice switched: %s -> %s", previous, tts.choice)
    return {"voice": tts.choice, "previous_voice": previous, "label": tts.voice}


@router.post("/unload")
async def unload(request: Request) -> dict:
    """Frees the STT model from memory (see SpeechToText.unload) - the backend half of the voice
    on/off toggle. The next real transcribe() call reloads it lazily, same as the very first call
    after startup already does; nothing else about voice is disabled server-side by this alone (the
    frontend is what actually hides the mic button - this endpoint only ever touches memory)."""
    state = request.app.state
    state.stt.unload()
    return {"stt_loaded": state.stt.is_loaded}


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(request: Request, audio: UploadFile, wake: bool = False) -> TranscribeResponse:
    """`wake=true`: the clip was recorded while hands-free mode waits for the wake word, so the speech
    model may be primed with it (see SpeechToText.transcribe's `prime`). Everything else is unprimed."""
    state = request.app.state
    limit = state.settings.voice_max_audio_bytes

    data = await audio.read(limit + 1)
    if len(data) > limit:
        raise AudioTooLargeError(f"Audio clip is too large (max {limit // 1_000_000} MB).")
    if not data:
        raise HTTPException(status_code=422, detail="No audio data was uploaded.")

    start = time.perf_counter()
    text = await (state.stt.transcribe(data, prime=True) if wake else state.stt.transcribe(data))
    if state.transcript_cleaner is not None:
        text = await state.transcript_cleaner.clean(text)
    duration_ms = round((time.perf_counter() - start) * 1000)
    logger.info("STT request audio_bytes=%d duration_ms=%d chars=%d", len(data), duration_ms, len(text))

    return TranscribeResponse(text=text, language="", duration_ms=duration_ms)


@router.post("/speak")
async def speak(body: SpeakRequest, request: Request) -> Response:
    state = request.app.state
    limit = state.settings.voice_max_speech_chars
    if len(body.text) > limit:
        raise HTTPException(status_code=422, detail=f"text is too long to speak (max {limit} characters)")

    spoken = clean_for_speech(body.text)
    if not spoken:
        raise HTTPException(status_code=422, detail="Nothing speakable remained after cleaning the text.")

    start = time.perf_counter()
    audio = await state.tts.synthesize(spoken)
    duration_ms = round((time.perf_counter() - start) * 1000)
    logger.info("TTS request chars=%d duration_ms=%d audio_bytes=%d", len(spoken), duration_ms, len(audio))

    return Response(content=audio, media_type="audio/wav", headers={"X-Synthesis-Ms": str(duration_ms)})


def error_payload(exc: VoiceError) -> dict[str, str]:
    return {"error": exc.code, "detail": str(exc)}
