"""Kokoro TTS worker (hexgrad/Kokoro-82M, Apache-2.0), run by app/voice/text_to_speech.py::KokoroTTS with the
`.venv-kokoro` Python (3.11): the `kokoro`/`misaki` packages need Python < 3.13, and Zira's own environment is 3.14.

Protocol: JSON lines. On start it loads the model once, warms it up, and prints
    {"ready": true, "device": ..., "voice": ..., "voices": [...], "load_seconds": ...}
Then, per request line {"id": n, "text": ..., "voice"?: ..., "speed"?: ...} it answers
    {"id": n, "ok": true, "wav": <base64 16-bit mono WAV>, "lang": "a"|"h", "audio_seconds": .., "seconds": ..}
or {"id": n, "ok": false, "error": ...}. The worker exits when its input closes.

Settings come from the environment: KOKORO_VOICE, KOKORO_SPEED, KOKORO_LANG (English G2P: "a" American, "b"
British), KOKORO_DEVICE (cpu | mps), KOKORO_HINGLISH (1/0), KOKORO_REPO.
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # for app/voice/hinglish.py (pure Python, no Zira imports)

REPO = os.environ.get("KOKORO_REPO", "hexgrad/Kokoro-82M")
VOICE = os.environ.get("KOKORO_VOICE", "hf_alpha")
SPEED = float(os.environ.get("KOKORO_SPEED", "1.0"))
ENGLISH = os.environ.get("KOKORO_LANG", "a")
DEVICE = os.environ.get("KOKORO_DEVICE", "cpu")
HINGLISH = os.environ.get("KOKORO_HINGLISH", "1") == "1"
RATE = 24000
# Replies go to the real stdout only; anything the libraries print (spaCy downloads, warnings) goes to stderr,
# which Zira sends to logs/kokoro.log, so it can never corrupt the JSON protocol.
_OUT = sys.stdout
sys.stdout = sys.stderr


def reply(payload: dict) -> None:
    _OUT.write(json.dumps(payload, ensure_ascii=False) + "\n")
    _OUT.flush()


def english_words():
    """misaki's own English dictionary (the words its English G2P knows), for telling English from Hindi."""
    import importlib.resources

    words: set[str] = set()
    for name in ("us_gold.json", "us_silver.json", "gb_gold.json"):
        try:
            data = json.loads((importlib.resources.files("misaki") / "data" / name).read_text())
            words.update(k.lower() for k in data)
        except (FileNotFoundError, ValueError):
            pass
    return words


def main() -> None:
    started = time.monotonic()
    try:
        import numpy as np
        import soundfile as sf
        import torch
        from kokoro import KModel, KPipeline

        from app.voice.hinglish import is_hinglish, to_hinglish_speech

        if DEVICE == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("KOKORO_DEVICE=mps but MPS is not available")
        model = KModel(repo_id=REPO).to(DEVICE).eval()
        pipelines = {ENGLISH: KPipeline(lang_code=ENGLISH, repo_id=REPO, model=model)}
        if HINGLISH:
            pipelines["h"] = KPipeline(lang_code="h", repo_id=REPO, model=model)
        english = english_words()
        voices = {}

        def voice_pack(name: str):
            if name not in voices:
                voices[name] = pipelines[ENGLISH].load_voice(name)
            return voices[name]

        def speak(text: str, voice: str, speed: float) -> tuple[bytes, str, float]:
            lang = ENGLISH
            if HINGLISH and is_hinglish(text):
                lang, text = "h", to_hinglish_speech(text, english.__contains__)
            chunks = [r.audio.numpy() for r in pipelines[lang](text, voice=voice_pack(voice), speed=speed)
                      if r.audio is not None]
            if not chunks:
                raise RuntimeError("Kokoro produced no audio")
            audio = np.concatenate(chunks)
            buf = io.BytesIO()
            sf.write(buf, audio, RATE, format="WAV", subtype="PCM_16")
            return buf.getvalue(), lang, len(audio) / RATE

        # Warm-up: the first call of each pipeline is several times slower (lazy loads, kernel setup).
        speak("Hello, I am ready.", VOICE, SPEED)
        if HINGLISH:
            speak("Namaste, main taiyaar hoon.", VOICE, SPEED)
    except Exception as exc:  # noqa: BLE001 - reported to Zira, which keeps working without speech
        reply({"ready": False, "error": f"{type(exc).__name__}: {exc}"})
        return
    reply({"ready": True, "device": DEVICE, "voice": VOICE, "voices": sorted(voices),
           "load_seconds": round(time.monotonic() - started, 2), "hinglish": HINGLISH, "english_lang": ENGLISH})

    for line in sys.stdin:
        if not line.strip():
            continue
        began = time.monotonic()
        request_id = None
        try:
            request = json.loads(line)
            request_id = request.get("id")
            text = str(request.get("text", "")).strip()
            if not text:
                raise ValueError("empty text")
            wav, lang, seconds = speak(text, request.get("voice") or VOICE, float(request.get("speed") or SPEED))
            reply({"id": request_id, "ok": True, "wav": base64.b64encode(wav).decode("ascii"), "lang": lang,
                   "audio_seconds": round(seconds, 2), "seconds": round(time.monotonic() - began, 3)})
        except Exception as exc:  # noqa: BLE001
            reply({"id": request_id, "ok": False, "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()
