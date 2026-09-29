"""How does the real Whisper model transcribe a wake phrase? (measures, does not guess)

Why this exists: the wake word is matched against Whisper's *transcript*, so its spelling of a rare word
decides whether hands-free mode wakes up at all. "JARVIS" is a common word; "Zira" is not (and in Hindi
ज़ीरा/जीरा means cumin). This synthesizes wake phrases with several macOS voices, runs them through the
same decode + mlx-whisper path the server uses (`app/voice/speech_to_text.py::MLXWhisperSTT._run`), with
and without an `initial_prompt` that biases the decoder toward the wake phrase, and writes every raw
transcript to JSON so the matcher in `frontend/app.js` can be tested against real output.

Runs in-process (loads its own copy of the model, ~1.6GB), so it does not need the server running.
Synthetic voices are not your voice: use the results to build the alias list, then confirm with real speech.

Usage:
    .venv/bin/python scripts/wake_word_benchmark.py --out logs/wake_benchmark.json
    .venv/bin/python scripts/wake_word_benchmark.py --wake "Zira" --prompt "Hello Zira." --out /tmp/x.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

MODEL = "mlx-community/whisper-large-v3-turbo"
# Indian English (the accent this project's TTS voice uses), Hindi, and other English accents.
VOICES = ["Rishi", "Aman", "Lekha", "Daniel", "Karen", "Kathy"]
RATE_WPM = 185


def phrases(wake: str) -> dict[str, list[str]]:
    return {
        "positive_english": [
            f"Hello {wake}, what's the time?",
            f"Hello {wake}, play a song.",
            f"Hey {wake}, tell me a joke.",
            f"Hello {wake}.",
        ],
        "positive_hinglish": [
            f"Hello {wake}, aaj ka mausam kaisa hai?",
            f"Hello {wake}, ek gaana chalao.",
        ],
        # Other names that sound close: a matcher that wakes on these is too loose.
        "negative_names": [
            "Hello Sarah, what's the time?",
            "Hello Kira, how are you?",
            "Hello Sira, play a song.",
            "Hello Zara, tell me a joke.",
        ],
        # The old wake word: the baseline the new one has to be compared with.
        "baseline_jarvis": [
            "Hello Jarvis, what's the time?",
            "Hello Jarvis, aaj ka mausam kaisa hai?",
        ],
    }


def synthesize(text: str, voice: str, out: Path) -> bool:
    proc = subprocess.run(
        ["say", "-v", voice, "-r", str(RATE_WPM), "-o", str(out), "--file-format=WAVE",
         "--data-format=LEI16@16000", "--", text],
        capture_output=True,
    )
    return proc.returncode == 0 and out.exists() and out.stat().st_size > 1000


def transcribe(audio: np.ndarray, prompt: str | None) -> tuple[str, str]:
    from mlx_whisper import transcribe as mlx_transcribe

    kwargs = {"path_or_hf_repo": MODEL, "language": None}
    if prompt:
        kwargs["initial_prompt"] = prompt
        # Without this the prompt is re-fed as context for every later window of the clip.
        kwargs["condition_on_previous_text"] = False
    result = mlx_transcribe(audio, **kwargs)
    return result["text"].strip(), result.get("language") or ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wake", default="Zira", help="the wake word being evaluated")
    parser.add_argument("--prompt", default="Hello Zira.", help="initial_prompt for the biased run")
    parser.add_argument("--out", default="logs/wake_word_benchmark.json")
    args = parser.parse_args()

    from faster_whisper.audio import decode_audio

    conditions = {"no_prompt": None, "with_prompt": args.prompt}
    rows: list[dict] = []
    started = time.time()

    def record(group: str, text: str, voice: str, audio: np.ndarray) -> None:
        for cond, prompt in conditions.items():
            if group == "baseline_jarvis" and cond == "with_prompt":
                continue  # only needed to see how the *current* setup hears the old word
            t = time.time()
            heard, lang = transcribe(audio, prompt)
            rows.append({"group": group, "said": text, "voice": voice, "condition": cond,
                         "heard": heard, "language": lang, "seconds": round(time.time() - t, 1)})
            print(f"[{len(rows):3d}] {group:18s} {voice:8s} {cond:11s} said={text!r:48} heard={heard!r}", flush=True)

    with tempfile.TemporaryDirectory() as tmp:
        for group, texts in phrases(args.wake).items():
            for text in texts:
                for voice in VOICES:
                    wav = Path(tmp) / "clip.wav"
                    if not synthesize(text, voice, wav):
                        print(f"(skipping voice {voice}: not installed)", flush=True)
                        continue
                    record(group, text, voice, decode_audio(str(wav), sampling_rate=16000))

    # Controls: what does the model invent from nothing? (A prompt that leaks into silence would
    # make hands-free mode wake itself up.)
    rng = np.random.default_rng(0)
    record("control_silence", "(3s of silence)", "-", np.zeros(48000, dtype=np.float32))
    record("control_noise", "(3s of low noise)", "-", (rng.standard_normal(48000) * 0.01).astype(np.float32))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"wake": args.wake, "prompt": args.prompt, "rows": rows}, ensure_ascii=False, indent=1))
    print(f"\nDONE {len(rows)} transcriptions in {time.time() - started:.0f}s -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
