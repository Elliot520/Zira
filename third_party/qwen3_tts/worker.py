"""Qwen3-TTS worker using MLX-Audio on Apple Silicon."""

from __future__ import annotations

import json
import os
import sys
import time

_protocol = os.fdopen(os.dup(1), "w", buffering=1)
os.dup2(2, 1)
sys.stdout = sys.stderr


def send(message: dict) -> None:
    _protocol.write(json.dumps(message, ensure_ascii=False) + "\n")
    _protocol.flush()


def main() -> None:
    model_id = os.environ.get("QWEN3_TTS_MODEL", "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit")
    try:
        from mlx_audio.audio_io import write as write_audio
        from mlx_audio.tts.utils import load_model
        import numpy as np

        model = load_model(model_id)
    except Exception as exc:  # noqa: BLE001 - return startup errors through the worker protocol
        send({"ready": False, "error": f"{type(exc).__name__}: {exc}"})
        return

    send({"ready": True, "model": model_id, "sample_rate": model.sample_rate})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            started = time.monotonic()
            results = list(model.generate_custom_voice(
                text=request["text"],
                speaker=request.get("voice", os.environ.get("QWEN3_TTS_VOICE", "Ryan")),
                language=request.get("language", os.environ.get("QWEN3_TTS_LANGUAGE", "English")),
            ))
            if not results:
                raise RuntimeError("The model returned no audio segments.")
            audio = np.concatenate([np.asarray(result.audio).reshape(-1) for result in results])
            if audio.size == 0 or not np.isfinite(audio).all():
                raise RuntimeError("The model returned empty or invalid audio.")
            write_audio(request["out"], audio, model.sample_rate, format="wav")
            send({"ok": True, "seconds": round(time.monotonic() - started, 2),
                  "audio_seconds": round(audio.size / model.sample_rate, 2)})
        except Exception as exc:  # noqa: BLE001 - a failed synthesis must not kill the text chat
            send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()