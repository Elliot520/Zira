"""Typed errors for the voice pipeline, mapped to the app's standard {error, detail} response shape."""

from __future__ import annotations


class VoiceError(Exception):
    """Base class for voice failures. `status_code` is the suggested HTTP status."""

    status_code = 502
    code = "voice_error"


class STTUnavailableError(VoiceError):
    status_code = 503
    code = "stt_unavailable"


class TTSUnavailableError(VoiceError):
    status_code = 503
    code = "tts_unavailable"


class TranscriptionError(VoiceError):
    status_code = 502
    code = "transcription_failed"


class SynthesisError(VoiceError):
    status_code = 502
    code = "synthesis_failed"


class AudioTooLargeError(VoiceError):
    status_code = 413
    code = "audio_too_large"
