"""ZIRA TTS benchmark: real Qwen (Ollama) + real Kokoro worker, through Zira's own chat WebSocket with streaming
speech - the same path a voice turn takes. Run it only while Zira is idle (it shares the GPU and Ollama).

Measures: Kokoro startup, then per prompt the first Qwen token, the first complete sentence (first audio event's
text), Kokoro's first audio (when it reached the client), the text finish, the speech finish, and Kokoro's
real-time factor (synthesis time / audio length, from Kokoro's own per-chunk timing). "First audio heard" needs a
browser; here it is the moment the first clip reached the client, which is what the browser starts playing.

Usage: .venv/bin/python scripts/tts_benchmark.py [MODEL]   (default: Zira's NEWLIGHT model)
"""

from __future__ import annotations

import base64
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402
from app.voice.streaming import _wav_seconds  # noqa: E402

PROMPTS = [
    "Mujhe Jupiter ke baare mein ek interesting astronomy fact batao, teen chhote sentences mein.",
    "Tell me in three short sentences why the sky is blue.",
]


def main() -> None:
    base = Settings()  # the real .env: Kokoro voice, speed, model names
    model = sys.argv[1] if len(sys.argv) > 1 else base.newlight_model
    work = Path(tempfile.mkdtemp(prefix="zira-tts-bench-"))
    settings = base.model_copy(update={
        "database_path": str(work / "bench.db"), "model_mode": "newlight", "newlight_model": model,
        "tts_provider": "kokoro", "tts_streaming": True, "auto_memory": False, "proactive_enabled": False,
        "knowledge_enabled": False, "web_search_enabled": False, "log_level": "WARNING",
    })
    app = create_app(settings=settings, env_path=work / "bench.env")
    print("ZIRA TTS BENCHMARK")
    print(f"Model: {model}   Voice: {settings.kokoro_voice}   Kokoro device: {settings.kokoro_device}")
    with TestClient(app, base_url="http://localhost") as client:
        kokoro = client.app.state.tts.engines["kokoro"]
        started = time.monotonic()
        while not kokoro.running and kokoro.error is None and time.monotonic() - started < 300:
            time.sleep(0.1)
        if not kokoro.running:
            print(f"Kokoro did not start: {kokoro.error}")
            return
        print(f"Kokoro startup (load + warm-up): {kokoro.load_seconds:.2f} s")
        for prompt in PROMPTS:
            print(f"\nPrompt: {prompt}")
            with client.websocket_connect("ws://localhost/ws/chat") as ws:
                sent = time.monotonic()
                ws.send_json({"message": prompt, "source": "voice", "speak": True})
                first_token = first_audio = text_done = None
                first_sentence = ""
                audio_seconds = 0.0
                chunks = 0
                while True:
                    event = ws.receive_json()
                    now = time.monotonic() - sent
                    if event["type"] == "token" and first_token is None:
                        first_token = now
                    elif event["type"] == "audio":
                        chunks += 1
                        audio_seconds += _wav_seconds(base64.b64decode(event["audio"]))
                        if first_audio is None:
                            first_audio, first_sentence = now, event["text"]
                    elif event["type"] == "done":
                        text_done = now
                    elif event["type"] in ("speech_end", "error"):
                        speech_done = now
                        if event["type"] == "error":
                            print(f"  error: {event.get('detail')}")
                        break
            ms = lambda s: f"{s * 1000:.0f} ms" if s is not None and s < 1 else (f"{s:.2f} s" if s is not None else "-")  # noqa: E731
            print(f"  Qwen first token:        {ms(first_token)}")
            print(f"  First sentence (spoken): {first_sentence!r}")
            print(f"  Kokoro first audio:      {ms(first_audio)} after the request "
                  f"({ms(first_audio - first_token) if first_audio and first_token else '-'} after the first token)")
            print(f"  Qwen text finished:      {ms(text_done)}")
            print(f"  Speech finished (sent):  {ms(speech_done)}   {chunks} chunk(s), {audio_seconds:.1f} s of audio")
            if text_done and first_audio:
                print(f"  First audio came {text_done - first_audio:.2f} s BEFORE the text finished"
                      if first_audio < text_done else "  First audio came after the text finished")
        print("\nKokoro real-time factor: see '[KOKORO] generation' lines in the log above, or logs/server.log.")


if __name__ == "__main__":
    main()
